import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="ema_sma_adx_rsi_roc_atr_v11",
        variables=["ema_period", "sma_period", "adx_period", "adx_threshold",
                   "roc_period", "rsi_buy", "rsi_sell", "atr_period", "atr_stop_pct"],
        bounds=([30, 35, 10, 10, 10, 52, 68, 20, 0.5],
                [50, 50, 18, 16, 20, 62, 74, 35, 1.3]),
        simulate=simulate,
    )

def simulate(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             volume: np.ndarray, x: np.ndarray) -> tuple:
    """
    Enhanced regime-switching with 9 parameters:
    - EMA-SMA crossover as core trend signal
    - ADX determines regime (trend vs mean-reversion)
    - Asymmetric RSI for buy/sell thresholds
    - ROC momentum filter for entry quality
    - Tunable ATR period for trailing stop
    - Dual exit: ATR stop + RSI overbought backup
    - Fixed asymmetric cooldowns (7 buy / 55 sell)
    """
    ema_period = int(x[0])
    sma_period = int(x[1])
    adx_period = max(int(x[2]), 1)
    adx_threshold = float(x[3])
    roc_period = max(int(x[4]), 1)
    rsi_buy = float(x[5])
    rsi_sell = float(x[6])
    atr_period = max(int(x[7]), 1)
    atr_stop_pct = float(x[8])
    
    # Pre-compute indicators
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    adx_data = adx_np(high, low, close, adx_period)
    adx = adx_data[0]
    roc = roc_np(close, roc_period)
    rsi = rsi_np(close, 14)
    atr = atr_np(high, low, close, atr_period)
    
    # Fixed asymmetric cooldowns (proven optimal)
    buy_cooldown = 7
    sell_cooldown = 55
    
    return _execute(close, 1_000_000.0, ema, sma, adx, roc, rsi, atr,
                    adx_threshold, rsi_buy, rsi_sell, atr_stop_pct, 
                    buy_cooldown, sell_cooldown)

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, adx, roc, rsi, atr,
             adx_threshold, rsi_buy, rsi_sell, atr_stop_pct, 
             buy_cooldown, sell_cooldown):
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
        c_rsi = rsi[i]
        c_atr = atr[i]
        
        # Skip NaN values from indicator warmup
        if np.isnan(c_ema) or np.isnan(c_sma) or np.isnan(c_adx):
            continue
        if np.isnan(c_roc) or np.isnan(c_rsi) or np.isnan(c_atr):
            continue
        
        # Update peak for trailing stop (track highest price since entry)
        if num_coins > 0 and price > peak_price:
            peak_price = price
        
        # DUAL EXIT: ATR trailing stop + RSI overbought backup
        if num_coins > 0:
            # Exit 1: ATR trailing stop
            if peak_price > 0:
                stop_price = peak_price * (1.0 - atr_stop_pct)
                if price <= stop_price:
                    cash, num_coins = sell_all(cash, num_coins, price)
                    last_trade = i
                    num_trades += 1
                    peak_price = 0.0
                    continue
            
            # Exit 2: RSI overbought (only in mean-reversion regime)
            in_trend_regime = c_adx > adx_threshold
            if not in_trend_regime and c_rsi > rsi_sell and i > last_trade + sell_cooldown:
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = 0.0
                continue
        
        # BUY SIGNAL: Regime-dependent with momentum confirmation
        if num_coins == 0:
            in_trend_regime = c_adx > adx_threshold
            roc_positive = c_roc > 0
            
            if in_trend_regime:
                # Trend regime: EMA crossover + ROC momentum + RSI entry
                if (c_ema > c_sma and 
                    c_rsi < rsi_buy and
                    roc_positive and
                    i > last_trade + buy_cooldown):
                    cash, num_coins = buy_all(cash, num_coins, price)
                    last_trade = i
                    num_trades += 1
                    peak_price = price
            else:
                # Mean-reversion regime: RSI entry + ROC momentum
                if (c_rsi < rsi_buy and
                    roc_positive and
                    i > last_trade + buy_cooldown):
                    cash, num_coins = buy_all(cash, num_coins, price)
                    last_trade = i
                    num_trades += 1
                    peak_price = price
        
        # SELL SIGNAL: Trend reversal with cooldown
        elif num_coins > 0:
            in_trend_regime = c_adx > adx_threshold
            
            if in_trend_regime:
                # Trend regime: EMA cross below + cooldown
                if c_ema < c_sma and i > last_trade + sell_cooldown:
                    cash, num_coins = sell_all(cash, num_coins, price)
                    last_trade = i
                    num_trades += 1
            else:
                # Mean-reversion regime: RSI overbought + cooldown (already handled above)
                pass
    
    # Force-sell at end of period
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades