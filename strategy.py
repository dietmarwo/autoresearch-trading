import numpy as np
from numba import njit
from strategy_helpers import *

# ---------------------------------------------------------------------------
#  Strategy definition
# ---------------------------------------------------------------------------

def get_strategy() -> dict:
    return dict(
        name="ema_sma_adx_rsi_v2",
        variables=[
            "ema_period", "sma_period", "adx_period", "rsi_period",
            "adx_threshold", "rsi_oversold", "wait_buy", "wait_sell", "stop_pct"
        ],
        bounds=([
            15, 25, 10, 10, 50, 20, 10, 10, 1
        ], [
            50, 100, 40, 50, 80, 45, 150, 200, 15
        ]),
        simulate=simulate,
    )

# ---------------------------------------------------------------------------
#  Simulation entry point (regular Python — bridges numpy prep and numba)
# ---------------------------------------------------------------------------

def simulate(close, high, low, volume, x):
    """
    Compute indicators, then delegate to the numba-compiled trading loop.
    """
    ema_period = max(int(x[0]), 1)
    sma_period = max(int(x[1]), 1)
    adx_period = max(int(x[2]), 1)
    rsi_period = max(int(x[3]), 1)
    adx_threshold = float(x[4])
    rsi_oversold = float(x[5])
    wait_buy = int(x[6])
    wait_sell = int(x[7])
    stop_pct = float(x[8]) / 100.0
    
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    adx_adx, adx_pdi, adx_mdi = adx_np(high, low, close, adx_period)
    rsi = rsi_np(close, rsi_period)
    
    return _execute(
        close, 1_000_000.0, ema, sma, adx_adx, rsi,
        float(adx_threshold), float(rsi_oversold),
        wait_buy, wait_sell, stop_pct
    )

# ---------------------------------------------------------------------------
#  Numba-compiled trading loop (the hot path)
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, adx, rsi, 
             adx_threshold, rsi_oversold, wait_buy, wait_sell, stop_pct):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    peak = 0.0  # track highest price since entry for trailing stop
    
    for i in range(len(close)):
        price = close[i]
        
        # Skip if any indicator is NaN
        if (np.isnan(ema[i]) or np.isnan(sma[i]) or 
            np.isnan(adx[i]) or np.isnan(rsi[i])):
            continue
        
        # Update peak for trailing stop when in position
        if num_coins > 0:
            if price > peak:
                peak = price
            # Trailing stop exit
            if price <= peak * (1.0 - stop_pct):
                cash, num_coins = sell_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak = price  # reset peak
                continue
        
        # Entry conditions (only when not in position)
        if num_coins == 0:
            # EMA crossover
            ema_cross = ema[i] > sma[i]
            # ADX trend filter (only trade when trend is strong enough)
            trend_filter = adx[i] >= adx_threshold
            # RSI confirmation (avoid overbought)
            rsi_filter = rsi[i] < rsi_oversold
            # Cooldown filter
            cooldown_ok = (i > last_trade + wait_buy)
            
            if ema_cross and trend_filter and rsi_filter and cooldown_ok:
                cash, num_coins = buy_all(cash, num_coins, price)
                last_trade = i
                num_trades += 1
                peak = price  # set initial peak for trailing stop
                continue
        
        # Exit on EMA crossover reversal (when no trailing stop hit)
        if num_coins > 0 and ema[i] < sma[i]:
            cash, num_coins = sell_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1
    
    # Force-sell at end
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades