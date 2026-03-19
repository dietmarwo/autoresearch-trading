import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="ema_sma_adx_roc_obv_atr_stop_v1",
        variables=["ema_period", "sma_period", "adx_period", "adx_threshold",
                   "roc_period", "roc_threshold", "atr_period", "atr_mult",
                   "obv_period", "wait_buy", "wait_sell"],
        bounds=([30, 35, 10, 15, 12, 0.2, 14, 1.5, 10, 20, 80],
                [70, 65, 25, 35, 30, 2.0, 42, 4.0, 25, 80, 200]),
        simulate=simulate,
    )

def simulate(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             volume: np.ndarray, x: np.ndarray) -> tuple:
    """
    Strategy: EMA-SMA crossover + ADX trend filter + ROC momentum + OBV volume + ATR trailing stop
    """
    ema_period = int(x[0])
    sma_period = int(x[1])
    adx_period = max(int(x[2]), 1)
    adx_threshold = float(x[3])
    roc_period = max(int(x[4]), 1)
    roc_threshold = float(x[5])
    atr_period = max(int(x[6]), 1)
    atr_mult = float(x[7])
    obv_period = max(int(x[8]), 1)
    wait_buy = int(x[9])
    wait_sell = int(x[10])
    
    # Pre-compute all indicators
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    adx_data = adx_np(high, low, close, adx_period)
    adx = adx_data[0]
    roc = roc_np(close, roc_period)
    atr = atr_np(high, low, close, atr_period)
    obv = obv_np(close, volume)
    
    return _execute(close, 1_000_000.0, ema, sma, adx, roc, atr, obv,
                    ema_period, sma_period, adx_period, roc_period,
                    atr_period, adx_threshold, roc_threshold, atr_mult,
                    obv_period, wait_buy, wait_sell)

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, adx, roc, atr, obv,
             ema_period, sma_period, adx_period, roc_period, atr_period,
             adx_threshold, roc_threshold, atr_mult,
             obv_period, wait_buy, wait_sell):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    peak_price = 0.0
    
    for i in range(len(close)):
        price = close[i]
        c_ema = ema[i]
        c_sma = sma[i]
        c_adx = adx[i]
        c_roc = roc[i]
        c_atr = atr[i]
        c_obv = obv[i]
        
        # Skip NaN values
        if np.isnan(c_ema) or np.isnan(c_sma) or np.isnan(c_adx):
            continue
        if np.isnan(c_roc) or np.isnan(c_atr) or np.isnan(c_obv):
            continue
        
        # Update peak for trailing stop
        if num_coins > 0 and price > peak_price:
            peak_price = price
        
        # TRAILING STOP: ATR-based exit (adaptive to volatility)
        if num_coins > 0 and peak_price > 0 and c_atr > 0:
            stop_price = peak_price * (1.0 - (c_atr * atr_mult / peak_price))
            if price <= stop_price:
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = 0.0
                continue
        
        # BUY SIGNAL: Multiple confirmations (asymmetric - more strict than sell)
        if num_coins == 0:
            # 1. EMA > SMA (trend direction)
            # 2. ADX > threshold (strong trend exists, filters mean-reverting markets)
            # 3. ROC > threshold (momentum positive and significant)
            # 4. OBV rising (volume confirms the move)
            # 5. Cooldown period respected
            if (c_ema > c_sma and 
                c_adx > adx_threshold and 
                c_roc > roc_threshold and
                c_obv > 0 and
                i > last_trade + wait_buy):
                cash, num_coins = buy_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = price
        # SELL SIGNAL: Simpler exit (EMA crossover + cooldown)
        elif num_coins > 0:
            # Exit on EMA crossover down with cooldown (less strict than buy)
            if c_ema < c_sma and i > last_trade + wait_sell:
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1

    # Force-sell at end of period
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades