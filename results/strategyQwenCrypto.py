import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy():
    return dict(
        name="btc_patience_v11",
        variables=["ema_period", "adx_period", "adx_threshold", "atr_period", 
                   "atr_mult_min", "atr_mult_max", "min_hold", "cooldown_sell", 
                   "profit_buffer_pct"],
        bounds=([10, 10, 12, 10, 4, 8, 5, 30, 0.5], 
                [60, 50, 40, 50, 14, 22, 60, 120, 8.0]),
        simulate=simulate,
    )

def simulate(close, high, low, volume, x):
    ema = ema_np(close, max(int(x[0]), 1))
    adx_data = adx_np(high, low, close, max(int(x[1]), 1))
    adx = adx_data[0]
    atr = atr_np(high, low, close, max(int(x[3]), 1))
    return _execute(close, 1_000_000.0, ema, adx, atr,
                    int(x[2]), int(x[4]), int(x[5]), 
                    int(x[6]), int(x[7]), float(x[8]))

@njit(fastmath=True)
def _execute(close, start_cash, ema, adx, atr, adx_threshold, 
             atr_mult_min, atr_mult_max, min_hold, cooldown_sell, profit_buffer_pct):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    entry_price = 0.0
    peak = 0.0
    stop_level = 0.0
    
    for i in range(len(close)):
        if np.isnan(ema[i]) or np.isnan(adx[i]) or np.isnan(atr[i]):
            continue
        
        price = close[i]
        is_holding = num_coins > 0
        
        # Track peak for trailing stop
        if is_holding and price > peak:
            peak = price
        
        # Calculate dynamic ATR stop based on ADX trend strength
        # Higher ADX = wider stop (stronger trend can absorb more volatility)
        adx_val = adx[i]
        atr_mult = atr_mult_min + (adx_val / 100.0) * (atr_mult_max - atr_mult_min)
        atr_dist = atr_mult * atr[i]
        stop_level = peak - atr_dist
        
        # Entry: ADX confirms trend strength + price above EMA (uptrend)
        if num_coins == 0 and adx[i] > adx_threshold and ema[i] > close[i - 1] if i > 0 else True:
            # Minimum hold protection: skip if too close to recent low
            cash, num_coins = buy_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1
            entry_price = price
            peak = price
            continue
        
        # Exit logic: only consider exit after minimum hold period
        if is_holding:
            days_since_trade = i - last_trade
            
            # Allow minimum hold period before considering exit
            if days_since_trade >= min_hold:
                # Calculate profit buffer threshold
                profit_threshold = entry_price * (1.0 + profit_buffer_pct / 100.0)
                
                # Only stop if we're in profit AND stop level hit
                # This prevents stop-outs during early volatility
                if price >= profit_threshold and price <= stop_level:
                    cash, num_coins = sell_all(cash, num_coins, price)
                    last_trade = i
                    num_trades += 1
                    peak = 0.0
                    entry_price = 0.0
                    continue
        
        # Extended cooldown: don't re-enter too quickly after selling
        if num_coins == 0 and (i - last_trade) < cooldown_sell:
            continue
    
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades