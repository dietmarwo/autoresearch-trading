import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy():
    return dict(
        name="dpo_williams_adx_regime_v2",
        variables=["dpo_period", "ema_period", "adx_period", "adx_min",
                   "williams_period", "williams_buy", "atr_period", "trail_mult",
                   "wait_buy", "pct_stop"],
        bounds=([6, 3, 8, 15, 7, 80, 12, 1.5, 4, 4],
                [14, 12, 16, 28, 14, 94, 22, 3.5, 10, 12]),
        simulate=simulate,
    )

def simulate(close, high, low, volume, x):
    dpo_period = max(int(x[0]), 1)
    ema_period = max(int(x[1]), 1)
    adx_period = max(int(x[2]), 1)
    adx_min = x[3]
    williams_period = max(int(x[4]), 1)
    williams_buy = x[5]
    atr_period = max(int(x[6]), 1)
    trail_mult = x[7]
    wait_buy = int(x[8])
    pct_stop = x[9]

    dpo = dpo_np(close, dpo_period)
    ema = ema_np(close, ema_period)
    adx, pdi, mdi = adx_np(high, low, close, adx_period)
    williams = williams_r_np(high, low, close, williams_period)
    atr = atr_np(high, low, close, atr_period)

    return _execute(close, 1_000_000.0, dpo, ema, adx, williams, atr,
                    dpo_period, adx_period, adx_min, williams_period, williams_buy,
                    atr_period, trail_mult, wait_buy, pct_stop)

@njit(fastmath=True)
def _execute(close, start_cash, dpo, ema, adx, williams, atr,
             dpo_period, adx_period, adx_min, williams_period, williams_buy,
             atr_period, trail_mult, wait_buy, pct_stop):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    peak_price = 0.0

    for i in range(len(close)):
        if i < adx_period or i < williams_period or i < atr_period:
            continue
        if np.isnan(dpo[i]) or np.isnan(ema[i]) or np.isnan(adx[i]) or np.isnan(williams[i]) or np.isnan(atr[i]):
            continue

        if num_coins == 0:
            # Buy: ADX confirms trend, Williams oversold, DPO negative (cycle trough)
            # EMA confirms price above trend
            if adx[i] > adx_min and williams[i] < williams_buy and dpo[i] < 0 and close[i] > ema[i]:
                if i - last_trade >= wait_buy:
                    cash, num_coins = buy_all(cash, num_coins, close[i])
                    last_trade = i
                    num_trades += 1
                    peak_price = close[i]
        else:
            # Trail peak
            if close[i] > peak_price:
                peak_price = close[i]

            # Exit: dual stop - ATR trail AND pct floor
            atr_stop = peak_price - trail_mult * atr[i]
            pct_floor = peak_price * (1.0 - pct_stop / 100.0)
            exit_price = max(atr_stop, pct_floor)

            if close[i] <= exit_price:
                cash, num_coins = sell_all(cash, num_coins, close[i])
                last_trade = i
                num_trades += 1

    # Force sell at end
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades