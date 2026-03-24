import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy() -> dict:
    return dict(
        name="dpo_supertrend_adx_refined_v7",
        variables=[
            "st_p", "st_mult", "dpo_p", "dpo_threshold", 
            "adx_p", "adx_min", "atr_p", "atr_mult", 
            "rsi_p", "rsi_exit", "wait_buy"
        ],
        bounds=(
            [5, 0.0, 5, -4.0, 5, 10, 10, 1.0, 5, 70, 1], 
            [30, 2.5, 25, 4.0, 25, 40, 30, 5.0, 25, 95, 10]
        ),
        simulate=simulate,
    )

def simulate(close, high, low, volume, x):
    # Trend regime via SuperTrend
    # Allow st_mult to be 0 for sensitivity, up to 2.5 for broader trend regimes
    st, direction = supertrend_np(high, low, close, max(int(x[0]), 1), x[1])
    
    # Dip entry via DPO (Detrended Price Oscillator)
    dpo = dpo_np(close, max(int(x[2]), 1))
    
    # Trend strength via ADX
    adx, pdi, mdi = adx_np(high, low, close, max(int(x[4]), 1))
    
    # Exit via ATR-based trailing stop
    atr = atr_np(high, low, close, max(int(x[6]), 1))
    
    # Momentum for profit-taking
    rsi = rsi_np(close, max(int(x[8]), 1))
    
    # Pass all indicators and parameters to execution
    return _execute(
        close, 1_000_000.0, direction, dpo, adx, atr, rsi, 
        x[3], x[5], x[7], x[9], int(x[10])
    )

@njit
def _execute(close, start_cash, direction, dpo, adx, atr, rsi, dpo_threshold, adx_min, atr_mult, rsi_exit, wait_buy):
    cash = start_cash
    num_coins = 0
    peak = 0.0
    last_trade = -1000
    num_trades = 0
    
    for i in range(len(close)):
        # Skip warmup periods
        if np.isnan(direction[i]) or np.isnan(dpo[i]) or np.isnan(adx[i]) or np.isnan(atr[i]) or np.isnan(rsi[i]):
            continue
            
        # Buy logic: Trend-aligned dip entry
        # SuperTrend(direction=1) -> Uptrend
        # DPO < dpo_threshold -> Dip/Pullback
        # ADX > adx_min -> Trend is strong enough
        if num_coins == 0:
            if i > last_trade + wait_buy:
                if direction[i] == 1 and dpo[i] < dpo_threshold and adx[i] > adx_min:
                    cash, num_coins = buy_all(cash, num_coins, close[i])
                    peak = close[i]
                    last_trade = i
                    num_trades += 1
        
        # Exit logic: Dual-exit (Trend reversal, Volatility stop, or Profit take)
        elif num_coins > 0:
            if close[i] > peak:
                peak = close[i]
            
            # Exit if:
            # 1. SuperTrend flips to bearish (direction == -1)
            # 2. Price drops below ATR-based trail (volatility protection)
            # 3. RSI indicates extreme overbought (profit-taking)
            if direction[i] == -1 or close[i] < (peak - atr[i] * atr_mult) or rsi[i] > rsi_exit:
                cash, num_coins = sell_all(cash, num_coins, close[i])
                last_trade = i
                num_trades += 1
                
    # Force sell at end
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades