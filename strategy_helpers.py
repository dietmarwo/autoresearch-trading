# strategy_helpers.py — Reusable indicators and trading primitives
#
# The AI agent imports from here when building strategies.
# All functions operate on numpy arrays.  Functions marked @njit can be
# called from inside other @njit functions (the numba trading loop).
#
# This file is NOT modified by the AI agent.  It is part of the framework.
#
# Organisation:
#   1. Trading primitives        — buy, sell, position sizing
#   2. Moving averages           — SMA, EMA, WMA, DEMA, TEMA, HMA, KAMA, VWMA, ZLEMA, FRAMA
#   3. Momentum / oscillators    — RSI, MACD, Stochastic, Williams %R, CCI, ROC, MFI,
#                                  TSI, Awesome Osc, Stoch RSI, CMO, DPO
#   4. Trend strength            — ADX, Aroon, Supertrend, Parabolic SAR,
#                                  TRIX, Vortex, Mass Index, linear regression
#   5. Volatility                — Bollinger, ATR, Keltner, NATR,
#                                  historical vol, Chaikin vol, ulcer index
#   6. Volume                    — OBV, CMF, Force Index, A/D line, VWAP
#   7. Price channels            — Donchian, pivot points, Ichimoku
#   8. Statistical / utility     — rolling stats, z-score, percentile,
#                                  drawdown, crossover, slope, etc.
#   9. Warmup

import numpy as np
from numba import njit

# ═══════════════════════════════════════════════════════════════════════════
#  1. TRADING PRIMITIVES
# ═══════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def buy_all(cash, num_coins, price):
    """Buy as many whole units as cash allows."""
    num = int(cash / price)
    return cash - num * price, num_coins + num

@njit(fastmath=True)
def sell_all(cash, num_coins, price):
    """Sell all held units."""
    return cash + num_coins * price, 0

@njit(fastmath=True)
def buy_fraction(cash, num_coins, price, fraction):
    """Buy using *fraction* of available cash (0..1)."""
    spend = cash * min(max(fraction, 0.0), 1.0)
    num = int(spend / price)
    return cash - num * price, num_coins + num

@njit(fastmath=True)
def sell_fraction(cash, num_coins, price, fraction):
    """Sell *fraction* of held coins (0..1)."""
    to_sell = int(num_coins * min(max(fraction, 0.0), 1.0))
    return cash + to_sell * price, num_coins - to_sell

@njit(fastmath=True)
def hodl(close, start_cash):
    """HODL factor: buy day 1, sell last day."""
    cash, nc = buy_all(start_cash, 0, close[0])
    cash, nc = sell_all(cash, nc, close[-1])
    return cash / start_cash

@njit(fastmath=True)
def position_size_kelly(win_rate, win_loss_ratio):
    """Kelly criterion fraction: f* = p - q/b."""
    if win_loss_ratio <= 0:
        return 0.0
    f = win_rate - (1.0 - win_rate) / win_loss_ratio
    return max(0.0, min(f, 1.0))

@njit(fastmath=True)
def trailing_stop_hit(price, peak, stop_pct):
    """True if price has dropped *stop_pct* (0..1) from *peak*."""
    return price <= peak * (1.0 - stop_pct)

@njit(fastmath=True)
def portfolio_value(cash, num_coins, price):
    """Current total portfolio value."""
    return cash + num_coins * price

# ═══════════════════════════════════════════════════════════════════════════
#  2. MOVING AVERAGES
# ═══════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def ema_np(close, period):
    """Exponential moving average."""
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    for i in range(period - 1):
        out[i] = np.nan
    s = 0.0
    for i in range(period):
        s += close[i]
    out[period - 1] = s / period
    alpha = 2.0 / (period + 1)
    for i in range(period, n):
        out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]
    return out

@njit(fastmath=True)
def sma_np(close, period):
    """Simple moving average."""
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    for i in range(period - 1):
        out[i] = np.nan
    s = 0.0
    for i in range(period):
        s += close[i]
    out[period - 1] = s / period
    for i in range(period, n):
        s += close[i] - close[i - period]
        out[i] = s / period
    return out

@njit(fastmath=True)
def wma_np(close, period):
    """Weighted moving average — recent prices get linearly higher weight."""
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    denom = period * (period + 1) / 2.0
    for i in range(period - 1):
        out[i] = np.nan
    for i in range(period - 1, n):
        s = 0.0
        for j in range(period):
            s += close[i - period + 1 + j] * (j + 1)
        out[i] = s / denom
    return out

@njit(fastmath=True)
def dema_np(close, period):
    """Double EMA — 2*EMA - EMA(EMA).  Less lag than single EMA."""
    e1 = ema_np(close, period)
    e2 = ema_np(e1, period)
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(e1[i]) or np.isnan(e2[i]):
            out[i] = np.nan
        else:
            out[i] = 2.0 * e1[i] - e2[i]
    return out

@njit(fastmath=True)
def tema_np(close, period):
    """Triple EMA — 3*EMA - 3*EMA(EMA) + EMA(EMA(EMA)).  Even less lag."""
    e1 = ema_np(close, period)
    e2 = ema_np(e1, period)
    e3 = ema_np(e2, period)
    n = len(close)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(e1[i]) or np.isnan(e2[i]) or np.isnan(e3[i]):
            out[i] = np.nan
        else:
            out[i] = 3.0 * e1[i] - 3.0 * e2[i] + e3[i]
    return out

@njit(fastmath=True)
def hma_np(close, period):
    """Hull Moving Average — very responsive, minimal lag."""
    half = max(period // 2, 1)
    sqrt_p = max(int(np.sqrt(period)), 1)
    w_half = wma_np(close, half)
    w_full = wma_np(close, period)
    n = len(close)
    diff = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(w_half[i]) or np.isnan(w_full[i]):
            diff[i] = np.nan
        else:
            diff[i] = 2.0 * w_half[i] - w_full[i]
    start = 0
    for i in range(n):
        if not np.isnan(diff[i]):
            start = i
            break
    if start + sqrt_p > n:
        return diff
    valid = diff[start:]
    hull_valid = wma_np(valid, sqrt_p)
    out = np.empty(n, dtype=np.float64)
    for i in range(start):
        out[i] = np.nan
    for i in range(len(hull_valid)):
        out[start + i] = hull_valid[i]
    return out

@njit(fastmath=True)
def kama_np(close, period, fast_sc, slow_sc):
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    for i in range(period):
        out[i] = np.nan
    fast_alpha = 2.0 / (fast_sc + 1)
    slow_alpha = 2.0 / (slow_sc + 1)
    out[period] = close[period]
    for i in range(period + 1, n):
        direction = abs(close[i] - close[i - period])
        volatility = 0.0
        for j in range(period):
            volatility += abs(close[i - j] - close[i - j - 1])
        if volatility == 0:
            er = 0.0
        else:
            er = direction / volatility
        sc = (er * (fast_alpha - slow_alpha) + slow_alpha) ** 2
        out[i] = out[i - 1] + sc * (close[i] - out[i - 1])
    return out

@njit(fastmath=True)
def vwma_np(close, volume, period):
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    for i in range(period - 1):
        out[i] = np.nan
    for i in range(period - 1, n):
        cv_sum = 0.0
        v_sum = 0.0
        for j in range(period):
            cv_sum += close[i - j] * volume[i - j]
            v_sum += volume[i - j]
        out[i] = cv_sum / v_sum if v_sum != 0 else close[i]
    return out

@njit(fastmath=True)
def zlema_np(close, period):
    """Zero-Lag EMA — subtracts the lag from input before applying EMA."""
    lag = (period - 1) // 2
    n = len(close)
    adjusted = np.empty(n, dtype=np.float64)
    for i in range(n):
        adjusted[i] = 2.0 * close[i] - close[i - lag] if i >= lag else close[i]
    return ema_np(adjusted, period)

@njit(fastmath=True)
def frama_np(close, period):
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    half = max(period // 2, 1)
    for i in range(period - 1):
        out[i] = np.nan
    out[period - 1] = close[period - 1]
    for i in range(period, n):
        hh1 = close[i - period + 1]; ll1 = close[i - period + 1]
        for j in range(i - period + 1, i - half + 1):
            if close[j] > hh1: hh1 = close[j]
            if close[j] < ll1: ll1 = close[j]
        hh2 = close[i - half + 1]; ll2 = close[i - half + 1]
        for j in range(i - half + 1, i + 1):
            if close[j] > hh2: hh2 = close[j]
            if close[j] < ll2: ll2 = close[j]
        hh = max(hh1, hh2); ll = min(ll1, ll2)
        n1 = (hh1 - ll1) / half if half > 0 else 0.0
        n2 = (hh2 - ll2) / half if half > 0 else 0.0
        n3 = (hh - ll) / period if period > 0 else 0.0
        if n1 + n2 > 0 and n3 > 0:
            d = (np.log(n1 + n2) - np.log(n3)) / np.log(2.0)
        else:
            d = 1.0
        alpha = np.exp(-4.6 * (d - 1.0))
        alpha = max(0.01, min(alpha, 1.0))
        out[i] = alpha * close[i] + (1.0 - alpha) * out[i - 1]
    return out

# ═══════════════════════════════════════════════════════════════════════════
#  3. MOMENTUM / OSCILLATORS
# ═══════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def rsi_np(close, period):
    """Relative Strength Index (Wilder's smoothing).  Range 0..100."""
    n = len(close)
    if period <= 0 or period + 1 >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    out[0] = np.nan
    gains = np.empty(n, dtype=np.float64); losses = np.empty(n, dtype=np.float64)
    for i in range(1, n):
        diff = close[i] - close[i - 1]
        gains[i] = diff if diff > 0 else 0.0
        losses[i] = -diff if diff < 0 else 0.0
    for i in range(1, period):
        out[i] = np.nan
    ag = 0.0; al = 0.0
    for i in range(1, period + 1):
        ag += gains[i]; al += losses[i]
    ag /= period; al /= period
    out[period] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    for i in range(period + 1, n):
        ag = (ag * (period - 1) + gains[i]) / period
        al = (al * (period - 1) + losses[i]) / period
        out[i] = 100.0 if al == 0 else 100.0 - 100.0 / (1.0 + ag / al)
    return out

@njit(fastmath=True)
def stoch_rsi_np(close, rsi_period, stoch_period, k_smooth, d_smooth):
    """Stochastic RSI.  Returns (%K, %D) — both 0..100."""
    n = len(close)
    if rsi_period <= 0 or stoch_period <= 0 or rsi_period + stoch_period >= n:
        return (np.full(n, np.nan), np.full(n, np.nan))
    r = rsi_np(close, rsi_period)
    warmup = rsi_period + stoch_period
    raw = np.empty(n, dtype=np.float64)
    for i in range(warmup - 1):
        raw[i] = np.nan
    for i in range(warmup - 1, n):
        rmin = r[i]; rmax = r[i]
        for j in range(1, stoch_period):
            v = r[i - j]
            if not np.isnan(v):
                if v < rmin: rmin = v
                if v > rmax: rmax = v
        rng = rmax - rmin
        raw[i] = 50.0 if rng == 0 else 100.0 * (r[i] - rmin) / rng
    k = sma_np(raw, k_smooth)
    d = sma_np(k, d_smooth)
    return k, d

@njit(fastmath=True)
def macd_np(close, fast, slow, signal):
    """MACD.  Returns (macd_line, signal_line, histogram)."""
    ef = ema_np(close, fast); es = ema_np(close, slow)
    n = len(close)
    ml = np.empty(n, dtype=np.float64)
    for i in range(n):
        ml[i] = ef[i] - es[i]
    start = max(fast, slow) - 1
    sl = np.empty(n, dtype=np.float64)
    for i in range(start + signal - 1):
        sl[i] = np.nan
    s = 0.0
    for i in range(start, start + signal):
        s += ml[i]
    sl[start + signal - 1] = s / signal
    alpha = 2.0 / (signal + 1)
    for i in range(start + signal, n):
        sl[i] = alpha * ml[i] + (1.0 - alpha) * sl[i - 1]
    h = np.empty(n, dtype=np.float64)
    for i in range(n):
        h[i] = ml[i] - sl[i]
    return ml, sl, h

@njit(fastmath=True)
def stochastic_np(high, low, close, k_period, d_period):
    """Stochastic Oscillator.  Returns (%K, %D) — both 0..100."""
    n = len(close)
    if k_period <= 0 or d_period <= 0 or k_period >= n:
        return (np.full(n, np.nan), np.full(n, np.nan))
    k = np.empty(n, dtype=np.float64)
    for i in range(k_period - 1):
        k[i] = np.nan
    for i in range(k_period - 1, n):
        hh = high[i]; ll = low[i]
        for j in range(1, k_period):
            if high[i-j] > hh: hh = high[i-j]
            if low[i-j] < ll: ll = low[i-j]
        rng = hh - ll
        k[i] = 50.0 if rng == 0 else 100.0 * (close[i] - ll) / rng
    d = sma_np(k, d_period)
    return k, d

@njit(fastmath=True)
def williams_r_np(high, low, close, period):
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    for i in range(period - 1):
        out[i] = np.nan
    for i in range(period - 1, n):
        hh = high[i]; ll = low[i]
        for j in range(1, period):
            if high[i-j] > hh: hh = high[i-j]
            if low[i-j] < ll: ll = low[i-j]
        rng = hh - ll
        out[i] = -50.0 if rng == 0 else -100.0 * (hh - close[i]) / rng
    return out

@njit(fastmath=True)
def cci_np(high, low, close, period):
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    tp = np.empty(n, dtype=np.float64)
    for i in range(n):
        tp[i] = (high[i] + low[i] + close[i]) / 3.0
    tp_sma = sma_np(tp, period)
    out = np.empty(n, dtype=np.float64)
    for i in range(period - 1):
        out[i] = np.nan
    for i in range(period - 1, n):
        mad = 0.0
        for j in range(period):
            mad += abs(tp[i - j] - tp_sma[i])
        mad /= period
        out[i] = 0.0 if mad == 0 else (tp[i] - tp_sma[i]) / (0.015 * mad)
    return out

@njit(fastmath=True)
def roc_np(close, period):
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    for i in range(period):
        out[i] = np.nan
    for i in range(period, n):
        out[i] = 0.0 if close[i-period] == 0 else 100.0 * (close[i] - close[i-period]) / close[i-period]
    return out

@njit(fastmath=True)
def momentum_np(close, period):
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    for i in range(period):
        out[i] = np.nan
    for i in range(period, n):
        out[i] = close[i] - close[i - period]
    return out

@njit(fastmath=True)
def mfi_np(high, low, close, volume, period):
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    tp = np.empty(n, dtype=np.float64)
    for i in range(n):
        tp[i] = (high[i] + low[i] + close[i]) / 3.0
    out = np.empty(n, dtype=np.float64)
    for i in range(period):
        out[i] = np.nan
    for i in range(period, n):
        pos = 0.0; neg = 0.0
        for j in range(period):
            idx = i - j
            mf = tp[idx] * volume[idx]
            if idx > 0 and tp[idx] > tp[idx-1]:
                pos += mf
            elif idx > 0:
                neg += mf
        out[i] = 100.0 if neg == 0 else 100.0 - 100.0 / (1.0 + pos / neg)
    return out

@njit(fastmath=True)
def tsi_np(close, long_period, short_period):
    """True Strength Index.  Range -100..100."""
    n = len(close)
    if long_period <= 0 or short_period <= 0 or long_period + short_period >= n:
        return np.full(n, np.nan)
    mom = np.empty(n, dtype=np.float64); am = np.empty(n, dtype=np.float64)
    mom[0] = 0.0; am[0] = 0.0
    for i in range(1, n):
        mom[i] = close[i] - close[i-1]
        am[i] = abs(mom[i])
    ds_m = ema_np(ema_np(mom, long_period), short_period)
    ds_a = ema_np(ema_np(am, long_period), short_period)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(ds_m[i]) or np.isnan(ds_a[i]) or ds_a[i] == 0:
            out[i] = np.nan
        else:
            out[i] = 100.0 * ds_m[i] / ds_a[i]
    return out

@njit(fastmath=True)
def awesome_oscillator_np(high, low, fast, slow):
    """Awesome Oscillator — SMA(median_price, fast) - SMA(median_price, slow)."""
    n = len(high)
    mp = np.empty(n, dtype=np.float64)
    for i in range(n):
        mp[i] = (high[i] + low[i]) / 2.0
    sf = sma_np(mp, fast); ss = sma_np(mp, slow)
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan if (np.isnan(sf[i]) or np.isnan(ss[i])) else sf[i] - ss[i]
    return out

@njit(fastmath=True)
def cmo_np(close, period):
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    for i in range(period):
        out[i] = np.nan
    for i in range(period, n):
        up = 0.0; dn = 0.0
        for j in range(period):
            diff = close[i-j] - close[i-j-1]
            if diff > 0: up += diff
            else: dn -= diff
        total = up + dn
        out[i] = 0.0 if total == 0 else 100.0 * (up - dn) / total
    return out

@njit(fastmath=True)
def dpo_np(close, period):
    """Detrended Price Oscillator."""
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    s = sma_np(close, period)
    shift = period // 2 + 1
    out = np.empty(n, dtype=np.float64)
    for i in range(n):
        si = i + shift
        if si < n and not np.isnan(s[si]):
            out[i] = close[i] - s[si]
        elif not np.isnan(s[i]):
            out[i] = close[i] - s[i]
        else:
            out[i] = np.nan
    return out

# ═══════════════════════════════════════════════════════════════════════════
#  4. TREND STRENGTH
# ═══════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def true_range_np(high, low, close):
    """True Range — standalone."""
    n = len(close)
    tr = np.empty(n, dtype=np.float64)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i-1])
        lc = abs(low[i] - close[i-1])
        tr[i] = max(hl, max(hc, lc))
    return tr

@njit(fastmath=True)
def adx_np(high, low, close, period):
    n = len(close)
    if period <= 0 or period >= n:
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    tr = true_range_np(high, low, close)
    pdm = np.empty(n, dtype=np.float64); mdm = np.empty(n, dtype=np.float64)
    pdm[0] = 0.0; mdm[0] = 0.0
    for i in range(1, n):
        up = high[i] - high[i-1]; down = low[i-1] - low[i]
        pdm[i] = up if (up > down and up > 0) else 0.0
        mdm[i] = down if (down > up and down > 0) else 0.0
    atr_s = np.empty(n, dtype=np.float64)
    ps = np.empty(n, dtype=np.float64); ms = np.empty(n, dtype=np.float64)
    for i in range(period - 1):
        atr_s[i] = np.nan; ps[i] = np.nan; ms[i] = np.nan
    a = 0.0; p = 0.0; m = 0.0
    for i in range(period):
        a += tr[i]; p += pdm[i]; m += mdm[i]
    atr_s[period-1] = a/period; ps[period-1] = p/period; ms[period-1] = m/period
    for i in range(period, n):
        atr_s[i] = (atr_s[i-1]*(period-1)+tr[i])/period
        ps[i] = (ps[i-1]*(period-1)+pdm[i])/period
        ms[i] = (ms[i-1]*(period-1)+mdm[i])/period
    pdi = np.empty(n, dtype=np.float64); mdi = np.empty(n, dtype=np.float64)
    dx = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(atr_s[i]) or atr_s[i] == 0:
            pdi[i] = np.nan; mdi[i] = np.nan; dx[i] = np.nan
        else:
            pdi[i] = 100.0*ps[i]/atr_s[i]; mdi[i] = 100.0*ms[i]/atr_s[i]
            ds = pdi[i]+mdi[i]
            dx[i] = 0.0 if ds == 0 else 100.0*abs(pdi[i]-mdi[i])/ds
    adx = np.empty(n, dtype=np.float64)
    w = 2*period-1
    for i in range(w):
        adx[i] = np.nan
    if w < n:
        s = 0.0; c = 0
        for i in range(period-1, w):
            if not np.isnan(dx[i]): s += dx[i]; c += 1
        adx[w-1] = s/c if c > 0 else 0.0
        for i in range(w, n):
            adx[i] = (adx[i-1]*(period-1)+dx[i])/period if not np.isnan(dx[i]) else adx[i-1]
    return adx, pdi, mdi

@njit(fastmath=True)
def aroon_np(high, low, period):
    n = len(high)
    if period <= 0 or period >= n:
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    up = np.empty(n, dtype=np.float64); dn = np.empty(n, dtype=np.float64)
    osc = np.empty(n, dtype=np.float64)
    for i in range(period):
        up[i] = np.nan; dn[i] = np.nan; osc[i] = np.nan
    for i in range(period, n):
        hi = 0; li = 0; hv = high[i-period]; lv = low[i-period]
        for j in range(1, period+1):
            if high[i-period+j] >= hv: hv = high[i-period+j]; hi = j
            if low[i-period+j] <= lv: lv = low[i-period+j]; li = j
        up[i] = 100.0*hi/period; dn[i] = 100.0*li/period
        osc[i] = up[i] - dn[i]
    return up, dn, osc

@njit(fastmath=True)
def atr_np(high, low, close, period):
    """Average True Range (Wilder smoothing)."""
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    tr = true_range_np(high, low, close)
    out = np.empty(n, dtype=np.float64)
    for i in range(period - 1):
        out[i] = np.nan
    s = 0.0
    for i in range(period):
        s += tr[i]
    out[period - 1] = s / period
    for i in range(period, n):
        out[i] = (out[i-1]*(period-1)+tr[i])/period
    return out

@njit(fastmath=True)
def supertrend_np(high, low, close, period, multiplier):
    n = len(close)
    if period <= 0 or period >= n:
        return (np.full(n, np.nan), np.full(n, np.nan))
    a = atr_np(high, low, close, period)
    st = np.empty(n, dtype=np.float64); dr = np.empty(n, dtype=np.float64)
    upper = np.empty(n, dtype=np.float64); lower = np.empty(n, dtype=np.float64)
    for i in range(n):
        mid = (high[i]+low[i])/2.0
        if np.isnan(a[i]):
            upper[i] = np.nan; lower[i] = np.nan
        else:
            upper[i] = mid+multiplier*a[i]; lower[i] = mid-multiplier*a[i]
    for i in range(period-1):
        st[i] = np.nan; dr[i] = 0.0
    if period-1 < n:
        st[period-1] = upper[period-1]; dr[period-1] = -1.0
    for i in range(period, n):
        if np.isnan(upper[i]):
            st[i] = st[i-1]; dr[i] = dr[i-1]; continue
        if dr[i-1] == 1.0 and not np.isnan(lower[i-1]):
            if lower[i] < lower[i-1]: lower[i] = lower[i-1]
        if dr[i-1] == -1.0 and not np.isnan(upper[i-1]):
            if upper[i] > upper[i-1]: upper[i] = upper[i-1]
        if dr[i-1] == -1.0:
            if close[i] > upper[i]: dr[i] = 1.0; st[i] = lower[i]
            else: dr[i] = -1.0; st[i] = upper[i]
        else:
            if close[i] < lower[i]: dr[i] = -1.0; st[i] = upper[i]
            else: dr[i] = 1.0; st[i] = lower[i]
    return st, dr

@njit(fastmath=True)
def psar_np(high, low, af_start, af_step, af_max):
    """Parabolic SAR.  Returns (sar, direction).  1=up, -1=down."""
    n = len(high)
    sar = np.empty(n, dtype=np.float64); dr = np.empty(n, dtype=np.float64)
    is_long = True; af = af_start; ep = high[0]
    sar[0] = low[0]; dr[0] = 1.0
    for i in range(1, n):
        ps = sar[i-1]
        if is_long:
            sar[i] = ps + af*(ep-ps)
            if i >= 2: sar[i] = min(sar[i], min(low[i-1], low[i-2]))
            elif i >= 1: sar[i] = min(sar[i], low[i-1])
            if low[i] < sar[i]:
                is_long = False; sar[i] = ep; ep = low[i]; af = af_start; dr[i] = -1.0
            else:
                dr[i] = 1.0
                if high[i] > ep: ep = high[i]; af = min(af+af_step, af_max)
        else:
            sar[i] = ps + af*(ep-ps)
            if i >= 2: sar[i] = max(sar[i], max(high[i-1], high[i-2]))
            elif i >= 1: sar[i] = max(sar[i], high[i-1])
            if high[i] > sar[i]:
                is_long = True; sar[i] = ep; ep = high[i]; af = af_start; dr[i] = 1.0
            else:
                dr[i] = -1.0
                if low[i] < ep: ep = low[i]; af = min(af+af_step, af_max)
    return sar, dr

@njit(fastmath=True)
def trix_np(close, period):
    """TRIX — 1-period pct ROC of triple-smoothed EMA."""
    e3 = ema_np(ema_np(ema_np(close, period), period), period)
    n = len(close); out = np.empty(n, dtype=np.float64); out[0] = np.nan
    for i in range(1, n):
        out[i] = np.nan if (np.isnan(e3[i]) or np.isnan(e3[i-1]) or e3[i-1]==0) else 100.0*(e3[i]-e3[i-1])/e3[i-1]
    return out

@njit(fastmath=True)
def vortex_np(high, low, close, period):
    n = len(close); tr = true_range_np(high, low, close)
    if period <= 0 or period >= n:
        return (np.full(n, np.nan), np.full(n, np.nan))
    vip = np.empty(n, dtype=np.float64); vim = np.empty(n, dtype=np.float64)
    for i in range(period):
        vip[i] = np.nan; vim[i] = np.nan
    for i in range(period, n):
        vp = 0.0; vm = 0.0; ts = 0.0
        for j in range(period):
            idx = i-j
            vp += abs(high[idx]-low[idx-1]); vm += abs(low[idx]-high[idx-1]); ts += tr[idx]
        if ts == 0: vip[i] = 1.0; vim[i] = 1.0
        else: vip[i] = vp/ts; vim[i] = vm/ts
    return vip, vim

@njit(fastmath=True)
def mass_index_np(high, low, ema_period, sum_period):
    """Mass Index — identifies reversals via range expansion/contraction."""
    n = len(high)
    if ema_period <= 0 or sum_period <= 0 or 2 * ema_period + sum_period >= n:
        return np.full(n, np.nan)
    rng = np.empty(n, dtype=np.float64)
    for i in range(n): rng[i] = high[i]-low[i]
    e1 = ema_np(rng, ema_period); e2 = ema_np(e1, ema_period)
    ratio = np.empty(n, dtype=np.float64)
    for i in range(n):
        ratio[i] = np.nan if (np.isnan(e1[i]) or np.isnan(e2[i]) or e2[i]==0) else e1[i]/e2[i]
    out = np.empty(n, dtype=np.float64)
    for i in range(sum_period-1):
        out[i] = np.nan
    for i in range(sum_period-1, n):
        s = 0.0; ok = True
        for j in range(sum_period):
            if np.isnan(ratio[i-j]): ok = False; break
            s += ratio[i-j]
        out[i] = s if ok else np.nan
    return out

@njit(fastmath=True)
def linreg_slope_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period-1): out[i] = np.nan
    xbar = (period-1)/2.0
    sxx = 0.0
    for j in range(period): sxx += (j-xbar)**2
    for i in range(period-1, n):
        ybar = 0.0
        for j in range(period): ybar += data[i-period+1+j]
        ybar /= period
        sxy = 0.0
        for j in range(period): sxy += (j-xbar)*(data[i-period+1+j]-ybar)
        out[i] = sxy/sxx if sxx != 0 else 0.0
    return out

@njit(fastmath=True)
def linreg_np(data, period):
    n = len(data); slope = linreg_slope_np(data, period)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    out = np.empty(n, dtype=np.float64)
    for i in range(period-1): out[i] = np.nan
    xbar = (period-1)/2.0
    for i in range(period-1, n):
        if np.isnan(slope[i]): out[i] = np.nan
        else:
            ybar = 0.0
            for j in range(period): ybar += data[i-period+1+j]
            ybar /= period
            out[i] = ybar + slope[i]*((period-1)-xbar)
    return out

@njit(fastmath=True)
def linreg_r2_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period-1): out[i] = np.nan
    xbar = (period-1)/2.0
    sxx = 0.0
    for j in range(period): sxx += (j-xbar)**2
    for i in range(period-1, n):
        ybar = 0.0
        for j in range(period): ybar += data[i-period+1+j]
        ybar /= period
        sxy = 0.0; syy = 0.0
        for j in range(period):
            y = data[i-period+1+j]
            sxy += (j-xbar)*(y-ybar); syy += (y-ybar)**2
        if syy == 0 or sxx == 0: out[i] = 0.0
        else: r = sxy/np.sqrt(sxx*syy); out[i] = r*r
    return out

# ═══════════════════════════════════════════════════════════════════════════
#  5. VOLATILITY
# ═══════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def bollinger_np(close, period, num_std):
    n = len(close); mid = sma_np(close, period)
    if period <= 0 or period >= n:
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    upper = np.empty(n, dtype=np.float64); lower = np.empty(n, dtype=np.float64)
    for i in range(period-1): upper[i] = np.nan; lower[i] = np.nan
    for i in range(period-1, n):
        s = 0.0
        for j in range(period):
            d = close[i-j]-mid[i]; s += d*d
        std = np.sqrt(s/period)
        upper[i] = mid[i]+num_std*std; lower[i] = mid[i]-num_std*std
    return mid, upper, lower

@njit(fastmath=True)
def bollinger_bandwidth_np(close, period, num_std):
    """Bollinger Bandwidth — (upper-lower)/middle.  Squeeze detection."""
    mid, upper, lower = bollinger_np(close, period, num_std)
    n = len(close); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan if (np.isnan(mid[i]) or mid[i]==0) else (upper[i]-lower[i])/mid[i]
    return out

@njit(fastmath=True)
def bollinger_pctb_np(close, period, num_std):
    """Bollinger %B — where price is within the bands.  0=lower, 1=upper."""
    mid, upper, lower = bollinger_np(close, period, num_std)
    n = len(close); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        bw = upper[i]-lower[i]
        out[i] = np.nan if (np.isnan(bw) or bw==0) else (close[i]-lower[i])/bw
    return out

@njit(fastmath=True)
def natr_np(high, low, close, period):
    """Normalized ATR — ATR/close*100.  Comparable across price levels."""
    a = atr_np(high, low, close, period)
    n = len(close); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan if (np.isnan(a[i]) or close[i]==0) else 100.0*a[i]/close[i]
    return out

@njit(fastmath=True)
def keltner_np(high, low, close, ema_period, atr_period, multiplier):
    """Keltner Channel.  Returns (middle, upper, lower)."""
    n = len(close)
    if ema_period <= 0 or atr_period <= 0 or max(ema_period, atr_period) >= n:
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    mid = ema_np(close, ema_period); a = atr_np(high, low, close, atr_period)
    upper = np.empty(n, dtype=np.float64); lower = np.empty(n, dtype=np.float64)
    for i in range(n):
        if np.isnan(mid[i]) or np.isnan(a[i]):
            upper[i] = np.nan; lower[i] = np.nan
        else:
            upper[i] = mid[i]+multiplier*a[i]; lower[i] = mid[i]-multiplier*a[i]
    return mid, upper, lower

@njit(fastmath=True)
def historical_vol_np(close, period):
    """Annualised historical volatility (std of log returns * sqrt(252))."""
    lr = log_return_np(close); std = rolling_std_np(lr, period)
    n = len(close); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan if np.isnan(std[i]) else std[i]*np.sqrt(252.0)
    return out

@njit(fastmath=True)
def chaikin_vol_np(high, low, ema_period, roc_period):
    """Chaikin Volatility — ROC of EMA(high-low)."""
    n = len(high)
    if ema_period <= 0 or roc_period <= 0 or ema_period + roc_period >= n:
        return np.full(n, np.nan)
    hl = np.empty(n, dtype=np.float64)
    for i in range(n): hl[i] = high[i]-low[i]
    ehl = ema_np(hl, ema_period)
    out = np.empty(n, dtype=np.float64)
    for i in range(roc_period): out[i] = np.nan
    for i in range(roc_period, n):
        if np.isnan(ehl[i]) or np.isnan(ehl[i-roc_period]) or ehl[i-roc_period]==0:
            out[i] = np.nan
        else:
            out[i] = 100.0*(ehl[i]-ehl[i-roc_period])/ehl[i-roc_period]
    return out

@njit(fastmath=True)
def ulcer_index_np(close, period):
    n = len(close); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period-1): out[i] = np.nan
    for i in range(period-1, n):
        peak = close[i-period+1]; ss = 0.0
        for j in range(period):
            idx = i-period+1+j
            if close[idx] > peak: peak = close[idx]
            pdd = 100.0*(close[idx]-peak)/peak if peak > 0 else 0.0
            ss += pdd*pdd
        out[i] = np.sqrt(ss/period)
    return out

# ═══════════════════════════════════════════════════════════════════════════
#  6. VOLUME
# ═══════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def obv_np(close, volume):
    """On-Balance Volume."""
    n = len(close); out = np.empty(n, dtype=np.float64); out[0] = volume[0]
    for i in range(1, n):
        if close[i] > close[i-1]: out[i] = out[i-1]+volume[i]
        elif close[i] < close[i-1]: out[i] = out[i-1]-volume[i]
        else: out[i] = out[i-1]
    return out

@njit(fastmath=True)
def cmf_np(high, low, close, volume, period):
    n = len(close)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    mfm = np.empty(n, dtype=np.float64)
    for i in range(n):
        hl = high[i]-low[i]
        mfm[i] = 0.0 if hl==0 else ((close[i]-low[i])-(high[i]-close[i]))/hl
    out = np.empty(n, dtype=np.float64)
    for i in range(period-1): out[i] = np.nan
    for i in range(period-1, n):
        mv = 0.0; vs = 0.0
        for j in range(period):
            mv += mfm[i-j]*volume[i-j]; vs += volume[i-j]
        out[i] = 0.0 if vs==0 else mv/vs
    return out

@njit(fastmath=True)
def force_index_np(close, volume, period):
    """Force Index — EMA of (close_change * volume)."""
    n = len(close); raw = np.empty(n, dtype=np.float64); raw[0] = 0.0
    for i in range(1, n): raw[i] = (close[i]-close[i-1])*volume[i]
    return ema_np(raw, period)

@njit(fastmath=True)
def ad_line_np(high, low, close, volume):
    """Accumulation/Distribution Line."""
    n = len(close); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        hl = high[i]-low[i]
        clv = 0.0 if hl==0 else ((close[i]-low[i])-(high[i]-close[i]))/hl
        out[i] = clv*volume[i] + (out[i-1] if i > 0 else 0.0)
    return out

@njit(fastmath=True)
def vwap_np(high, low, close, volume):
    """Cumulative VWAP."""
    n = len(close); out = np.empty(n, dtype=np.float64)
    ctv = 0.0; cv = 0.0
    for i in range(n):
        tp = (high[i]+low[i]+close[i])/3.0
        ctv += tp*volume[i]; cv += volume[i]
        out[i] = close[i] if cv==0 else ctv/cv
    return out

@njit(fastmath=True)
def volume_oscillator_np(volume, fast, slow):
    """Volume Oscillator — (EMA_fast-EMA_slow)/EMA_slow*100."""
    ef = ema_np(volume, fast); es = ema_np(volume, slow)
    n = len(volume); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan if (np.isnan(ef[i]) or np.isnan(es[i]) or es[i]==0) else 100.0*(ef[i]-es[i])/es[i]
    return out

@njit(fastmath=True)
def volume_ratio_np(volume, period):
    """Current volume / SMA(volume).  >1 = above average."""
    avg = sma_np(volume, period)
    n = len(volume); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan if (np.isnan(avg[i]) or avg[i]==0) else volume[i]/avg[i]
    return out

# ═══════════════════════════════════════════════════════════════════════════
#  7. PRICE CHANNELS
# ═══════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def donchian_np(high, low, period):
    n = len(high)
    if period <= 0 or period >= n:
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    upper = np.empty(n, dtype=np.float64); lower = np.empty(n, dtype=np.float64)
    middle = np.empty(n, dtype=np.float64)
    for i in range(period-1):
        upper[i] = np.nan; lower[i] = np.nan; middle[i] = np.nan
    for i in range(period-1, n):
        hh = high[i]; ll = low[i]
        for j in range(1, period):
            if high[i-j] > hh: hh = high[i-j]
            if low[i-j] < ll: ll = low[i-j]
        upper[i] = hh; lower[i] = ll; middle[i] = (hh+ll)/2.0
    return upper, lower, middle

@njit(fastmath=True)
def pivot_points_np(high, low, close):
    """Classic pivot points.  Returns (pivot, r1, s1, r2, s2, r3, s3)."""
    n = len(close)
    pv = np.empty(n, dtype=np.float64)
    r1 = np.empty(n, dtype=np.float64); s1 = np.empty(n, dtype=np.float64)
    r2 = np.empty(n, dtype=np.float64); s2 = np.empty(n, dtype=np.float64)
    r3 = np.empty(n, dtype=np.float64); s3 = np.empty(n, dtype=np.float64)
    pv[0]=np.nan; r1[0]=np.nan; s1[0]=np.nan
    r2[0]=np.nan; s2[0]=np.nan; r3[0]=np.nan; s3[0]=np.nan
    for i in range(1, n):
        p = (high[i-1]+low[i-1]+close[i-1])/3.0
        pv[i] = p
        r1[i] = 2.0*p-low[i-1]; s1[i] = 2.0*p-high[i-1]
        r2[i] = p+(high[i-1]-low[i-1]); s2[i] = p-(high[i-1]-low[i-1])
        r3[i] = high[i-1]+2.0*(p-low[i-1]); s3[i] = low[i-1]-2.0*(high[i-1]-p)
    return pv, r1, s1, r2, s2, r3, s3

@njit(fastmath=True)
def ichimoku_np(high, low, close, tenkan, kijun, senkou_b_period):
    """Ichimoku Cloud.  Returns (tenkan_sen, kijun_sen, senkou_a, senkou_b, chikou).
    Senkou A/B are NOT shifted forward."""
    n = len(close)
    max_p = max(tenkan, max(kijun, senkou_b_period))
    if tenkan <= 0 or kijun <= 0 or senkou_b_period <= 0 or max_p >= n:
        return (np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan))
    ts = np.empty(n, dtype=np.float64); ks = np.empty(n, dtype=np.float64)
    sa = np.empty(n, dtype=np.float64); sb = np.empty(n, dtype=np.float64)
    ch = np.empty(n, dtype=np.float64)
    for i in range(n):
        if i < tenkan-1: ts[i] = np.nan
        else:
            hh = high[i]; ll = low[i]
            for j in range(1, tenkan):
                if high[i-j] > hh: hh = high[i-j]
                if low[i-j] < ll: ll = low[i-j]
            ts[i] = (hh+ll)/2.0
        if i < kijun-1: ks[i] = np.nan
        else:
            hh = high[i]; ll = low[i]
            for j in range(1, kijun):
                if high[i-j] > hh: hh = high[i-j]
                if low[i-j] < ll: ll = low[i-j]
            ks[i] = (hh+ll)/2.0
        sa[i] = np.nan if (np.isnan(ts[i]) or np.isnan(ks[i])) else (ts[i]+ks[i])/2.0
        if i < senkou_b_period-1: sb[i] = np.nan
        else:
            hh = high[i]; ll = low[i]
            for j in range(1, senkou_b_period):
                if high[i-j] > hh: hh = high[i-j]
                if low[i-j] < ll: ll = low[i-j]
            sb[i] = (hh+ll)/2.0
        ch[i] = close[i]
    return ts, ks, sa, sb, ch

# ═══════════════════════════════════════════════════════════════════════════
#  8. STATISTICAL / UTILITY
# ═══════════════════════════════════════════════════════════════════════════

@njit(fastmath=True)
def log_return_np(close):
    """Daily log-returns.  First element is 0.0."""
    n = len(close); out = np.empty(n, dtype=np.float64); out[0] = 0.0
    for i in range(1, n): out[i] = np.log(close[i]/close[i-1])
    return out

@njit(fastmath=True)
def pct_change_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period): out[i] = np.nan
    for i in range(period, n):
        out[i] = 0.0 if data[i-period]==0 else (data[i]-data[i-period])/data[i-period]
    return out

@njit(fastmath=True)
def rolling_std_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period-1): out[i] = np.nan
    for i in range(period-1, n):
        s = 0.0; s2 = 0.0
        for j in range(period):
            v = data[i-j]; s += v; s2 += v*v
        mean = s/period; var = s2/period - mean*mean
        out[i] = np.sqrt(max(var, 0.0))
    return out

@njit(fastmath=True)
def rolling_mean_np(data, period):
    """Rolling mean (alias for SMA)."""
    return sma_np(data, period)

@njit(fastmath=True)
def rolling_sum_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period-1): out[i] = np.nan
    s = 0.0
    for i in range(period): s += data[i]
    out[period-1] = s
    for i in range(period, n):
        s += data[i]-data[i-period]; out[i] = s
    return out

@njit(fastmath=True)
def rolling_max_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period-1): out[i] = np.nan
    for i in range(period-1, n):
        mx = data[i]
        for j in range(1, period):
            if data[i-j] > mx: mx = data[i-j]
        out[i] = mx
    return out

@njit(fastmath=True)
def rolling_min_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period-1): out[i] = np.nan
    for i in range(period-1, n):
        mn = data[i]
        for j in range(1, period):
            if data[i-j] < mn: mn = data[i-j]
        out[i] = mn
    return out

@njit(fastmath=True)
def rolling_median_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    buf = np.empty(period, dtype=np.float64)
    for i in range(period-1): out[i] = np.nan
    for i in range(period-1, n):
        for j in range(period): buf[j] = data[i-period+1+j]
        for a in range(period):
            for b in range(a+1, period):
                if buf[b] < buf[a]: tmp = buf[a]; buf[a] = buf[b]; buf[b] = tmp
        out[i] = buf[period//2] if period%2==1 else (buf[period//2-1]+buf[period//2])/2.0
    return out

@njit(fastmath=True)
def zscore_np(data, period):
    """Rolling z-score: (value - mean) / std."""
    mean = sma_np(data, period); std = rolling_std_np(data, period)
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan if (np.isnan(mean[i]) or np.isnan(std[i]) or std[i]==0) else (data[i]-mean[i])/std[i]
    return out

@njit(fastmath=True)
def percentile_rank_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period-1): out[i] = np.nan
    for i in range(period-1, n):
        c = 0
        for j in range(period):
            if data[i-j] <= data[i]: c += 1
        out[i] = c/period
    return out

@njit(fastmath=True)
def drawdown_np(close):
    """Drawdown from running peak.  Always <= 0."""
    n = len(close); out = np.empty(n, dtype=np.float64)
    peak = close[0]
    for i in range(n):
        if close[i] > peak: peak = close[i]
        out[i] = (close[i]-peak)/peak if peak > 0 else 0.0
    return out

@njit(fastmath=True)
def drawdown_duration_np(close):
    """Number of bars since last peak (0 = at peak)."""
    n = len(close); out = np.empty(n, dtype=np.float64)
    peak = close[0]; bars = 0
    for i in range(n):
        if close[i] >= peak: peak = close[i]; bars = 0
        else: bars += 1
        out[i] = float(bars)
    return out

@njit(fastmath=True)
def normalize_np(data, period):
    """Rolling min-max normalisation to 0..1."""
    mn = rolling_min_np(data, period); mx = rolling_max_np(data, period)
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan if (np.isnan(mn[i]) or np.isnan(mx[i]) or mx[i]==mn[i]) else (data[i]-mn[i])/(mx[i]-mn[i])
    return out

@njit(fastmath=True)
def crossover_np(a, b):
    """1.0 where a crosses above b, else 0.0."""
    n = len(a); out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(a[i]) and not np.isnan(b[i]) and not np.isnan(a[i-1]) and not np.isnan(b[i-1]):
            if a[i] > b[i] and a[i-1] <= b[i-1]: out[i] = 1.0
    return out

@njit(fastmath=True)
def crossunder_np(a, b):
    """1.0 where a crosses below b, else 0.0."""
    n = len(a); out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if not np.isnan(a[i]) and not np.isnan(b[i]) and not np.isnan(a[i-1]) and not np.isnan(b[i-1]):
            if a[i] < b[i] and a[i-1] >= b[i-1]: out[i] = 1.0
    return out

@njit(fastmath=True)
def slope_np(data):
    """Bar-to-bar slope (first difference).  First element is 0."""
    n = len(data); out = np.empty(n, dtype=np.float64); out[0] = 0.0
    for i in range(1, n): out[i] = data[i]-data[i-1]
    return out

@njit(fastmath=True)
def diff_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period): out[i] = np.nan
    for i in range(period, n): out[i] = data[i]-data[i-period]
    return out

@njit(fastmath=True)
def clamp_np(data, lo, hi):
    """Clamp values to [lo, hi]."""
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        v = data[i]
        if v < lo: v = lo
        if v > hi: v = hi
        out[i] = v
    return out

@njit(fastmath=True)
def lag_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period): out[i] = np.nan
    for i in range(period, n): out[i] = data[i-period]
    return out

@njit(fastmath=True)
def sign_np(data):
    """Element-wise sign: -1, 0, or +1."""
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        if data[i] > 0: out[i] = 1.0
        elif data[i] < 0: out[i] = -1.0
        else: out[i] = 0.0
    return out

@njit(fastmath=True)
def abs_np(data):
    """Element-wise absolute value."""
    n = len(data); out = np.empty(n, dtype=np.float64)
    for i in range(n): out[i] = abs(data[i])
    return out

@njit(fastmath=True)
def highest_bars_ago_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period-1): out[i] = np.nan
    for i in range(period-1, n):
        mi = 0; mv = data[i]
        for j in range(1, period):
            if data[i-j] > mv: mv = data[i-j]; mi = j
        out[i] = float(mi)
    return out

@njit(fastmath=True)
def lowest_bars_ago_np(data, period):
    n = len(data); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period-1): out[i] = np.nan
    for i in range(period-1, n):
        mi = 0; mv = data[i]
        for j in range(1, period):
            if data[i-j] < mv: mv = data[i-j]; mi = j
        out[i] = float(mi)
    return out

@njit(fastmath=True)
def bars_since_np(condition):
    """Bars since condition was last nonzero.  0 on the bar it fires."""
    n = len(condition); out = np.empty(n, dtype=np.float64)
    count = float(n)
    for i in range(n):
        if condition[i] != 0: count = 0.0
        out[i] = count
        count += 1.0
    return out

@njit(fastmath=True)
def above_np(a, threshold):
    """1.0 where a > threshold, else 0.0."""
    n = len(a); out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(a[i]) and a[i] > threshold: out[i] = 1.0
    return out

@njit(fastmath=True)
def below_np(a, threshold):
    """1.0 where a < threshold, else 0.0."""
    n = len(a); out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(a[i]) and a[i] < threshold: out[i] = 1.0
    return out

@njit(fastmath=True)
def between_np(a, lo, hi):
    """1.0 where lo <= a <= hi, else 0.0."""
    n = len(a); out = np.zeros(n, dtype=np.float64)
    for i in range(n):
        if not np.isnan(a[i]) and lo <= a[i] <= hi: out[i] = 1.0
    return out

@njit(fastmath=True)
def ema_cross_signal_np(close, fast, slow):
    """+1 on EMA golden cross, -1 on death cross, 0 otherwise."""
    ef = ema_np(close, fast); es = ema_np(close, slow)
    n = len(close); out = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        if np.isnan(ef[i]) or np.isnan(es[i]) or np.isnan(ef[i-1]) or np.isnan(es[i-1]):
            continue
        if ef[i] > es[i] and ef[i-1] <= es[i-1]: out[i] = 1.0
        elif ef[i] < es[i] and ef[i-1] >= es[i-1]: out[i] = -1.0
    return out

@njit(fastmath=True)
def decay_linear_np(data, period):
    """Linearly decaying weighted sum (= WMA)."""
    return wma_np(data, period)

@njit(fastmath=True)
def decay_exp_np(data, halflife):
    """Exponential decay filter with given halflife in bars."""
    alpha = 1.0 - np.exp(-np.log(2.0)/halflife)
    n = len(data); out = np.empty(n, dtype=np.float64); out[0] = data[0]
    for i in range(1, n): out[i] = alpha*data[i]+(1.0-alpha)*out[i-1]
    return out

@njit(fastmath=True)
def mean_reversion_score_np(close, period):
    """(close - SMA) / rolling_std.  Positive = above mean."""
    mean = sma_np(close, period); std = rolling_std_np(close, period)
    n = len(close); out = np.empty(n, dtype=np.float64)
    for i in range(n):
        out[i] = np.nan if (np.isnan(mean[i]) or np.isnan(std[i]) or std[i]==0) else (close[i]-mean[i])/std[i]
    return out

@njit(fastmath=True)
def trend_strength_np(close, period):
    n = len(close); out = np.empty(n, dtype=np.float64)
    if period <= 0 or period >= n:
        return np.full(n, np.nan)
    for i in range(period): out[i] = np.nan
    for i in range(period, n):
        direction = abs(close[i]-close[i-period])
        vol = 0.0
        for j in range(period): vol += abs(close[i-j]-close[i-j-1])
        out[i] = 0.0 if vol==0 else direction/vol
    return out

# ═══════════════════════════════════════════════════════════════════════════
#  9. WARMUP
# ═══════════════════════════════════════════════════════════════════════════

def warmup():
    """Force-compile all @njit functions before any fork()."""
    c = np.array([100.0, 101.0, 99.0, 102.0, 98.0, 103.0, 97.0, 104.0,
                  96.0, 105.0, 95.0, 106.0, 94.0, 107.0, 93.0, 108.0,
                  92.0, 109.0, 91.0, 110.0, 100.0, 102.0, 98.0, 103.0,
                  97.0, 104.0, 96.0, 105.0, 95.0, 106.0], dtype=np.float64)
    h = c + 2.0; l = c - 2.0; v = np.ones(30, dtype=np.float64)*1000.0
    p = 5
    # 1. trading
    buy_all(1000.0, 0, 100.0); sell_all(0.0, 10, 100.0)
    buy_fraction(1000.0, 0, 100.0, 0.5); sell_fraction(0.0, 10, 100.0, 0.5)
    hodl(c, 1000.0); position_size_kelly(0.6, 1.5)
    trailing_stop_hit(95.0, 100.0, 0.05); portfolio_value(500.0, 10, 100.0)
    # 2. moving averages
    ema_np(c, p); sma_np(c, p); wma_np(c, p); dema_np(c, p); tema_np(c, p)
    hma_np(c, p); kama_np(c, p, 2, 30); vwma_np(c, v, p)
    zlema_np(c, p); frama_np(c, p)
    # 3. momentum
    rsi_np(c, p); stoch_rsi_np(c, p, p, 3, 3); macd_np(c, 3, p, 3)
    stochastic_np(h, l, c, p, 3); williams_r_np(h, l, c, p)
    cci_np(h, l, c, p); roc_np(c, p); momentum_np(c, p)
    mfi_np(h, l, c, v, p); tsi_np(c, p, 3)
    awesome_oscillator_np(h, l, 3, p); cmo_np(c, p); dpo_np(c, p)
    # 4. trend
    true_range_np(h, l, c); adx_np(h, l, c, p); aroon_np(h, l, p)
    supertrend_np(h, l, c, p, 2.0); psar_np(h, l, 0.02, 0.02, 0.2)
    trix_np(c, 3); vortex_np(h, l, c, p); mass_index_np(h, l, 3, p)
    linreg_slope_np(c, p); linreg_np(c, p); linreg_r2_np(c, p)
    # 5. volatility
    bollinger_np(c, p, 2.0); bollinger_bandwidth_np(c, p, 2.0)
    bollinger_pctb_np(c, p, 2.0); atr_np(h, l, c, p); natr_np(h, l, c, p)
    keltner_np(h, l, c, p, p, 1.5); historical_vol_np(c, p)
    chaikin_vol_np(h, l, 3, p); ulcer_index_np(c, p)
    # 6. volume
    obv_np(c, v); cmf_np(h, l, c, v, p); force_index_np(c, v, p)
    ad_line_np(h, l, c, v); vwap_np(h, l, c, v)
    volume_oscillator_np(v, 3, p); volume_ratio_np(v, p)
    # 7. channels
    donchian_np(h, l, p); pivot_points_np(h, l, c); ichimoku_np(h, l, c, 3, p, 2*p)
    # 8. utility
    log_return_np(c); pct_change_np(c, p)
    rolling_std_np(c, p); rolling_mean_np(c, p); rolling_sum_np(c, p)
    rolling_max_np(c, p); rolling_min_np(c, p); rolling_median_np(c, p)
    zscore_np(c, p); percentile_rank_np(c, p)
    drawdown_np(c); drawdown_duration_np(c); normalize_np(c, p)
    e = ema_np(c, 3); s = sma_np(c, p)
    crossover_np(e, s); crossunder_np(e, s)
    slope_np(c); diff_np(c, p); clamp_np(c, 95.0, 105.0); lag_np(c, 3)
    sign_np(c-100.0); abs_np(c-100.0)
    highest_bars_ago_np(c, p); lowest_bars_ago_np(c, p)
    bars_since_np(crossover_np(e, s))
    above_np(c, 100.0); below_np(c, 100.0); between_np(c, 95.0, 105.0)
    ema_cross_signal_np(c, 3, p); decay_linear_np(c, p); decay_exp_np(c, 5.0)
    mean_reversion_score_np(c, p); trend_strength_np(c, p)
