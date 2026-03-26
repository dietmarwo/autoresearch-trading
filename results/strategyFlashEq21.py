import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="dpo_st_adx_v26_optimized",
        variables=[
            "st_p", "st_mult", "dpo_p", "dpo_threshold",
            "adx_p", "adx_min", "atr_p", "atr_mult",
            "rsi_p", "rsi_exit", "wait_buy"
        ],
        bounds=(
            [8, 0.0, 3, -1.5, 7, 5, 10, 0.5, 7, 80, 0],
            [18, 1.5, 12, 1.5, 16, 20, 25, 2.5, 15, 95, 5]
        ),
        simulate=simulate,
    )

def simulate(close, high, low, volume, x):
    # Parameter mapping
    st_p = max(int(x[0]), 1)
    st_mult = x[1]
    dpo_p = max(int(x[2]), 1)
    dpo_threshold = x[3]
    adx_p = max(int(x[4]), 1)
    adx_min = x[5]
    atr_p = max(int(x[6]), 1)
    atr_mult = x[7]
    rsi_p = max(int(x[8]), 1)
    rsi_exit = x[9]
    wait_buy = int(x[10])

    # Indicators
    st, direction = supertrend_np(high, low, close, st_p, st_mult)
    dpo = dpo_np(close, dpo_p)
    adx, pdi, mdi = adx_np(high, low, close, adx_p)
    atr = atr_np(high, low, close, atr_p)
    rsi = rsi_np(close, rsi_p)
    
    return _execute(
        close, 1_000_000.0, direction, dpo, adx, atr, rsi, 
        dpo_threshold, adx_min, atr_mult, rsi_exit, wait_buy
    )

@njit
def _execute(close, start_cash, direction, dpo, adx, atr, rsi, dpo_threshold, adx_min, atr_mult, rsi_exit, wait_buy):
    cash = start_cash
    num_coins = 0
    peak = 0.0
    last_trade = -1000
    num_trades = 0
    
    for i in range(len(close)):
        # Warmup skip: ensure indicators have valid values
        if np.isnan(direction[i]) or np.isnan(dpo[i]) or np.isnan(adx[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]):
            continue
            
        # Entry Logic: Bullish SuperTrend + DPO Pullback + Sufficient ADX Trend Strength
        if num_coins == 0:
            if i > last_trade + wait_buy:
                # 1. SuperTrend confirms uptrend
                # 2. DPO below threshold confirms pullback
                # 3. ADX confirms sufficient trend strength
                if direction[i] == 1 and dpo[i] < dpo_threshold and adx[i] > adx_min:
                    cash, num_coins = buy_all(cash, num_coins, close[i])
                    peak = close[i]
                    last_trade = i
                    num_trades += 1
        
        # Exit Logic: Dual-exit (ATR trailing stop OR RSI profit-taking OR Trend flip)
        elif num_coins > 0:
            # Update peak for trailing stop
            if close[i] > peak:
                peak = close[i]
            
            # Exit triggers
            is_st_flip = (direction[i] == -1)
            is_atr_stop = (close[i] < (peak - atr[i] * atr_mult))
            is_rsi_profit = (rsi[i] > rsi_exit)
            
            if is_st_flip or is_atr_stop or is_rsi_profit:
                cash, num_coins = sell_all(cash, num_coins, close[i])
                last_trade = i
                num_trades += 1
                
    # Final cleanup
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades