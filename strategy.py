import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="ema_sma_adx_regime_obv_atr_v6",
        variables=["ema_period", "sma_period", "adx_period", "adx_threshold",
                   "rsi_buy", "rsi_sell", "obv_period", "atr_period", 
                   "atr_stop_pct", "sell_cooldown"],
        bounds=([30, 35, 10, 10, 50, 60, 10, 20, 0.8, 45],
                [50, 50, 18, 16, 65, 80, 20, 35, 2.0, 65]),
        simulate=simulate,
    )

def simulate(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             volume: np.ndarray, x: np.ndarray) -> tuple:
    """
    Regime-switching trend-following with volume confirmation:
    - EMA-SMA crossover as core signal
    - ADX determines regime (trend vs mean-reversion)
    - Asymmetric RSI for buy/sell quality
    - OBV volume confirmation
    - ATR trailing stop for risk management
    - 10 parameters optimized for stability
    """
    ema_period = int(x[0])
    sma_period = int(x[1])
    adx_period = max(int(x[2]), 1)
    adx_threshold = float(x[3])
    rsi_buy = float(x[4])
    rsi_sell = float(x[5])
    obv_period = max(int(x[6]), 1)
    atr_period = max(int(x[7]), 1)
    atr_stop_pct = float(x[8])
    sell_cooldown = int(x[9])
    
    # Fixed buy cooldown (shorter for faster entries)
    buy_cooldown = 8
    
    # Pre-compute indicators
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    adx_data = adx_np(high, low, close, adx_period)
    adx = adx_data[0]
    rsi = rsi_np(close, 14)
    obv = obv_np(close, volume)
    obv_sma = sma_np(obv, obv_period)
    atr = atr_np(high, low, close, atr_period)
    
    return _execute(close, 1_000_000.0, ema, sma, adx, rsi, obv, obv_sma, atr,
                    adx_threshold, rsi_buy, rsi_sell, atr_stop_pct,
                    buy_cooldown, sell_cooldown)

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, adx, rsi, obv, obv_sma, atr,
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
        c_rsi = rsi[i]
        c_obv = obv[i]
        c_obv_sma = obv_sma[i]
        c_atr = atr[i]
        
        # Skip NaN values
        if np.isnan(c_ema) or np.isnan(c_sma) or np.isnan(c_adx):
            continue
        if np.isnan(c_rsi) or np.isnan(c_obv) or np.isnan(c_obv_sma):
            continue
        if np.isnan(c_atr):
            continue
        
        # Update peak for trailing stop
        if num_coins > 0 and price > peak_price:
            peak_price = price
        
        # TRAILING STOP: ATR-based percentage exit
        if num_coins > 0 and peak_price > 0:
            stop_price = peak_price * (1.0 - atr_stop_pct)
            if price <= stop_price:
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak_price = 0.0
                continue
        
        # BUY SIGNAL: Regime-dependent logic
        if num_coins == 0:
            # Trend regime (ADX high): EMA crossover + volume confirmation
            # Mean-reversion regime (ADX low): RSI oversold + volume confirmation
            in_trend_regime = c_adx > adx_threshold
            obv_rising = c_obv > c_obv_sma
            
            if in_trend_regime:
                # Trend-following entry
                if (c_ema > c_sma and 
                    c_rsi < rsi_buy and
                    obv_rising and
                    i > last_trade + buy_cooldown):
                    cash, num_coins = buy_all(cash, num_coins, price)
                    last_trade = i
                    num_trades += 1
                    peak_price = price
            else:
                # Mean-reversion entry (low trend strength)
                if (c_rsi < rsi_buy and
                    obv_rising and
                    i > last_trade + buy_cooldown):
                    cash, num_coins = buy_all(cash, num_coins, price)
                    last_trade = i
                    num_trades += 1
                    peak_price = price
        
        # SELL SIGNAL: Trend reversal or RSI overbought
        elif num_coins > 0:
            in_trend_regime = c_adx > adx_threshold
            
            if in_trend_regime:
                # Trend regime: EMA cross below + cooldown
                if (c_ema < c_sma and i > last_trade + sell_cooldown):
                    cash, num_coins = sell_all(cash, num_coins, price)
                    last_trade = i
                    num_trades += 1
            else:
                # Mean-reversion regime: RSI overbought + cooldown
                if (c_rsi > rsi_sell and i > last_trade + sell_cooldown):
                    cash, num_coins = sell_all(cash, num_coins, price)
                    last_trade = i
                    num_trades += 1
    
    # Force-sell at end of period
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades