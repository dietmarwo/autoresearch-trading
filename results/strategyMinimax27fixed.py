import numpy as np
from numba import njit
from strategy_helpers import *


def get_strategy() -> dict:
    return dict(
        name="mfi_rsi_adx_chandelier_v6_fixed",
        variables=[
            "mfi_period",
            "mfi_entry_lo",
            "mfi_entry_hi",
            "mfi_exit",
            "rsi_period",
            "rsi_entry",
            "adx_period",
            "adx_thresh",
            "atr_period",
            "atr_mult",
            "hh_bars",
            "wait_buy",
            "wait_sell",
            "profit_atr_mult",
            "partial_pct",
            "atr_exit_mult",
        ],
        bounds=(
            [10, 18, 48, 55, 8, 28, 12, 14, 14, 2.0, 15, 5, 85, 3.0, 25, 5.0],
            [22, 35, 62, 72, 22, 45, 20, 26, 24, 4.0, 35, 15, 115, 5.5, 50, 8.0],
        ),
        simulate=simulate,
    )


def simulate(close, high, low, volume, x):
    mfi_period = max(int(x[0]), 1)
    mfi_entry_lo = int(x[1])
    mfi_entry_hi = int(x[2])
    mfi_exit = int(x[3])
    rsi_period = max(int(x[4]), 1)
    rsi_entry = int(x[5])
    adx_period = max(int(x[6]), 1)
    adx_thresh = float(x[7])
    atr_period = max(int(x[8]), 1)
    atr_mult = float(x[9])
    hh_bars = max(int(x[10]), 5)
    wait_buy = int(x[11])
    wait_sell = int(x[12])
    profit_atr_mult = float(x[13])
    partial_pct = float(x[14]) / 100.0
    atr_exit_mult = float(x[15])

    mfi = mfi_np(high, low, close, volume, mfi_period)
    rsi = rsi_np(close, rsi_period)
    adx, pdi, mdi = adx_np(high, low, close, adx_period)
    atr = atr_np(high, low, close, atr_period)

    return _execute(
        close,
        high,
        low,
        1_000_000.0,
        mfi,
        rsi,
        adx,
        atr,
        mfi_entry_lo,
        mfi_entry_hi,
        mfi_exit,
        rsi_entry,
        adx_thresh,
        atr_mult,
        hh_bars,
        wait_buy,
        wait_sell,
        profit_atr_mult,
        partial_pct,
        atr_exit_mult,
    )


@njit
def _execute(
    close,
    high,
    low,
    start_cash,
    mfi,
    rsi,
    adx,
    atr,
    mfi_entry_lo,
    mfi_entry_hi,
    mfi_exit,
    rsi_entry,
    adx_thresh,
    atr_mult,
    hh_bars,
    wait_buy,
    wait_sell,
    profit_atr_mult,
    partial_pct,
    atr_exit_mult,
):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    stop_price = 0.0
    peak_price = 0.0
    entry_price = 0.0
    partial_done = False
    cum_atr_since_entry = 0.0

    for i in range(len(close)):
        price = close[i]
        c_mfi = mfi[i]
        c_rsi = rsi[i]
        c_adx = adx[i]
        c_atr = atr[i]

        if np.isnan(c_mfi) or np.isnan(c_rsi) or np.isnan(c_adx) or np.isnan(c_atr) or c_atr <= 0:
            continue

        trending = c_adx >= adx_thresh

        if num_coins == 0:
            if trending and mfi_entry_lo <= c_mfi <= mfi_entry_hi and c_rsi > rsi_entry and i > last_trade + wait_buy:
                cash, num_coins = buy_all(cash, num_coins, price)
                last_trade = i
                peak_price = price
                entry_price = price
                stop_price = price - atr_mult * c_atr
                partial_done = False
                cum_atr_since_entry = 0.0
                num_trades += 1
            continue

        cum_atr_since_entry += c_atr
        active_stop = stop_price

        # The stop carried into this bar is the only stop that can trigger on it.
        if low[i] <= active_stop:
            fill_price = price if price < active_stop else active_stop
            cash, num_coins = sell_all(cash, num_coins, fill_price)
            last_trade = i
            num_trades += 1
            stop_price = 0.0
            peak_price = 0.0
            entry_price = 0.0
            partial_done = False
            cum_atr_since_entry = 0.0
            continue

        if c_mfi > mfi_exit and i > last_trade + wait_sell:
            cash, num_coins = sell_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1
            stop_price = 0.0
            peak_price = 0.0
            entry_price = 0.0
            partial_done = False
            cum_atr_since_entry = 0.0
            continue

        if cum_atr_since_entry > atr_exit_mult * c_atr and c_adx < adx_thresh * 0.6 and i > last_trade + wait_sell:
            cash, num_coins = sell_all(cash, num_coins, price)
            last_trade = i
            num_trades += 1
            stop_price = 0.0
            peak_price = 0.0
            entry_price = 0.0
            partial_done = False
            cum_atr_since_entry = 0.0
            continue

        partial_stop = active_stop
        if not partial_done and price >= entry_price + profit_atr_mult * c_atr:
            coins_to_sell = int(num_coins * partial_pct)
            if coins_to_sell > 0:
                cash += price * coins_to_sell
                num_coins -= coins_to_sell
                partial_done = True
                partial_stop = max(partial_stop, price - 1.5 * c_atr)

        if high[i] > peak_price:
            peak_price = high[i]

        if i >= hh_bars:
            hh = high[i - hh_bars]
            for j in range(i - hh_bars + 1, i):
                if high[j] > hh:
                    hh = high[j]
            chandelier_stop = hh - atr_mult * c_atr
        else:
            chandelier_stop = peak_price - atr_mult * c_atr

        stop_price = max(active_stop, partial_stop, chandelier_stop)

    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades
