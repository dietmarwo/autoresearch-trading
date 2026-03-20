# strategy.py — THE FILE THE AI AGENT EDITS
#
# This is the equivalent of Karpathy's train.py for autoresearch.
# The framework imports get_strategy() and plugs it into walk-forward
# evaluation.  The AI agent modifies this file to propose new strategies.
#
# Contract:
#   get_strategy() returns a dict with:
#     name       : str               — human-readable strategy name
#     variables  : list[str]         — parameter names (for logging)
#     bounds     : (list, list)      — (lower_bounds, upper_bounds)
#     simulate   : callable          — see signature below
#
#   simulate(close, high, low, volume, x) -> (growth_factor, num_trades)
#     close  : np.ndarray float64   — daily close prices
#     high   : np.ndarray float64   — daily high prices
#     low    : np.ndarray float64   — daily low prices
#     volume : np.ndarray float64   — daily volume
#     x      : np.ndarray float64   — decision variables from optimizer
#     returns: (float, int)         — (cash / start_cash, number_of_trades)
#
# Rules for the AI agent:
#   1. get_strategy() must be importable and return the dict above.
#   2. simulate() must be fast — use @njit for the inner trading loop.
#   3. simulate() receives NUMPY arrays only (no pandas).
#   4. The optimizer minimizes; framework negates factor, so higher = better.
#   5. Use: from strategy_helpers import *
#   6. You may define as many internal helper functions as needed.
#   7. All decision variables are continuous floats — cast to int inside
#      simulate() where needed (e.g., window sizes).
#   8. Keep variable count reasonable (4–15). More variables = harder to
#      optimise, more prone to overfit.

import numpy as np
from numba import njit
from strategy_helpers import *

# ---------------------------------------------------------------------------
#  Strategy definition
# ---------------------------------------------------------------------------

def get_strategy() -> dict:
    return dict(
        name="ema_sma_crossover_v1",
        variables=["ema_period", "sma_period", "wait_buy", "wait_sell"],
        bounds=([20, 50, 10, 10],
                [50, 100, 200, 200]),
        simulate=simulate,
    )

# ---------------------------------------------------------------------------
#  Simulation entry point (regular Python — bridges numpy prep and numba)
# ---------------------------------------------------------------------------

def simulate(close: np.ndarray, high: np.ndarray, low: np.ndarray,
             volume: np.ndarray, x: np.ndarray) -> tuple:
    """
    Compute indicators, then delegate to the numba-compiled trading loop.
    This function is called ~10k+ times per optimisation window, so keep
    indicator computation efficient (numpy only, no pandas).
    """
    ema_period = int(x[0])
    sma_period = int(x[1])
    ema = ema_np(close, ema_period)
    sma = sma_np(close, sma_period)
    return _execute(close, 1_000_000.0, ema, sma, int(x[2]), int(x[3]))

# ---------------------------------------------------------------------------
#  Numba-compiled trading loop (the hot path)
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _execute(close, start_cash, ema, sma, wait_buy, wait_sell):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0

    for i in range(len(close)):
        c_ema = ema[i]; c_sma = sma[i]; price = close[i]
        if np.isnan(c_ema) or np.isnan(c_sma):
            continue
        # buy when EMA crosses above SMA, respecting cooldown
        if num_coins == 0 and c_ema > c_sma and i > last_trade + wait_buy:
            cash, num_coins = buy_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1
        # sell when EMA crosses below SMA, respecting cooldown
        elif num_coins > 0 and c_ema < c_sma and i > last_trade + wait_sell:
            cash, num_coins = sell_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1

    # force-sell at end
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades
