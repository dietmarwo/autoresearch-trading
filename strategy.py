import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="ema_sma_adx_roc_obv_atr_v2",
        variables=["ema_period", "sma_period", "adx_period", "adx_threshold",
                   "roc_period", "roc_threshold", "atr_period", 
                   "trailing_stop_pct", "volatility_threshold"],
        bounds=([35, 40, 12, 14, 18, 0.0, 25, 1.5, 0.05],
                [55, 55, 20, 19, 28, 1.5, 40, 2.5, 0.25]),
        simulate=simulate,
    )

def simulate(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             volume: np.ndarray, x: np.ndarray) -> tuple:
    """
    Streamlined hybrid trend-following strategy:
    - EMA-SMA crossover as core entry/exit signal
    - ADX trend filter to avoid whipsaws (especially effective on AAPL)
    - ROC momentum confirmation for entry quality
    - OBV volume confirmation for signal strength
    - ATR trailing stop for risk management
    - Volatility filter to avoid choppy regimes
    - Asymmetric cooldowns (shorter buy, longer sell)
    """
    ema_period = int(x[0])
    sma_period = int(x[1])
    adx_period = max(int(x[2]), 1)
    adx_threshold = float(x[3])
    roc_period = max(int(x[4]), 1)
    roc_threshold = float(x[5])
    atr_period = max(int(x[6]), 1)
    trailing_stop_pct = float(x[7])
    volatility_threshold = float(x[8])
    
    # Pre-compute indicators
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    adx_data = adx_np(high, low, close, adx_period)
    adx = adx_data[0]
    roc = roc_np(close, roc_period)
    obv = obv_np(close, volume)
    atr = atr_np(high, low, close, atr_period)
    hist_vol = rolling_std_np(close, 20)
    
    return _execute(close, 1_000_000.0, ema, sma, adx, roc, obv, atr,
                    hist_vol, adx_threshold, roc_threshold,
                    trailing_stop_pct, volatility_threshold)

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, adx, roc, obv, atr,
             hist_vol, adx_threshold, roc_threshold,
             trailing_stop_pct, volatility_threshold):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    peak_price = 0.0
    obv_prev = 0.0
    buy_cooldown = 7
    sell_cooldown = 45
    
    for i in range(len(close)):
        price = close[i]
        c_ema = ema[i]
        c_sma = sma[i]
        c_adx = adx[i]
        c_roc = roc[i]
        c_obv = obv[i]
        c_atr = atr[i]
        c_vol = hist_vol[i]
        
        # Skip NaN values
        if np.isnan(c_ema) or np.isnan(c_sma) or np.isnan(c_adx):
            continue
        if np.isnan(c_roc) or np.isnan(c_obv) or np.isnan(c_atr) or np.isnan(c_vol):
            continue
        
        # Update peak for trailing stop (track highest price since entry)
        if num_coins > 0 and price > peak_price:
            peak_price = price
        
        # TRAILING STOP: Percentage-based exit to lock in gains
        if num_coins > 0 and peak_price > 0:
            stop_price = peak_price * (1.0 - trailing_stop_pct)
            if price <= stop_price:
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = 0.0
                continue
        
        # BUY SIGNAL: Multiple confirmations required
        if num_coins == 0:
            # 1. EMA > SMA (uptrend direction)
            # 2. ADX > threshold (trend strength, filters whipsaws)
            # 3. ROC > threshold (positive momentum)
            # 4. OBV rising (volume confirmation)
            # 5. Volatility in reasonable range (avoid choppy regimes)
            # 6. Buy cooldown period respected
            obv_rising = c_obv > obv_prev if i > 0 else True
            vol_ok = c_vol > volatility_threshold if c_vol > 0 else False
            
            if (c_ema > c_sma and 
                c_adx > adx_threshold and 
                c_roc > roc_threshold and
                obv_rising and
                vol_ok and
                i > last_trade + buy_cooldown):
                cash, num_coins = buy_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = price
        
        # SELL SIGNAL: EMA crosses below SMA (trend reversal)
        # Longer cooldown to avoid being shaken out of strong trends
        elif num_coins > 0 and c_ema < c_sma and i > last_trade + sell_cooldown:
            cash, num_coins = sell_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1
        
        obv_prev = c_obv

    # Force-sell at end of period
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades