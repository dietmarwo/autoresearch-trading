# Copyright (c) Dietmar Wolz.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory.

# Walk-forward validated trading strategy optimizer with stationary bootstrap.
# Adapted from crypto.py — see
# https://github.com/dietmarwo/fast-cma-es/blob/master/tutorials/CryptoTrading.adoc
#
# This is the FRAMEWORK.  It does not contain any trading strategy.
# The strategy lives in strategy.py, which the AI agent edits.

# Install runtime dependencies:
#   pip install fcmaes loguru numba numpy pandas scipy yfinance

import yfinance as yf
import pandas as pd
import numpy as np
import importlib

np.set_printoptions(legacy='1.25')
import time, sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List as TList

from fcmaes import retry
from fcmaes.optimizer import Bite_cpp, dtime
from scipy.optimize import Bounds
import ctypes as ct
import multiprocessing as mp
from numba import njit

from loguru import logger

logger.remove()
logger.add(sys.stdout, format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")
logger.add("log_{time}.txt", format="{time:HH:mm:ss.SS} | {process} | {level} | {message}", level="INFO")

# ---------------------------------------------------------------------------
#  Strategy loading
# ---------------------------------------------------------------------------

def load_strategy(module_name: str = "strategy") -> dict:
    """
    Import the strategy module and call its get_strategy().

    Returns a dict with keys:
        name      : str
        variables : list[str]
        bounds    : (list, list)    — (lower, upper)
        simulate  : callable(close, high, low, volume, x) -> (factor, ntrades)
    """
    mod = importlib.import_module(module_name)
    spec = mod.get_strategy()
    # validate
    required = {"name", "variables", "bounds", "simulate"}
    missing = required - set(spec.keys())
    if missing:
        raise ValueError(f"Strategy module {module_name} missing keys: {missing}")
    lo, hi = spec["bounds"]
    if len(lo) != len(hi) or len(lo) != len(spec["variables"]):
        raise ValueError("bounds dimensions must match number of variables")

    # --- JIT warmup: compile everything BEFORE any fork() ---
    # This prevents "corrupted size vs. prev_size" heap corruption
    # caused by numba JIT-compiling in forked child processes.
    from strategy_helpers import warmup as _warmup_helpers
    _warmup_helpers()
    _warmup_strategy(spec)

    logger.info(f"Loaded strategy: {spec['name']}  "
                f"({len(spec['variables'])} vars: {spec['variables']})")
    return spec


def _warmup_strategy(spec: dict):
    """
    Call the strategy's simulate() once with dummy data so all @njit
    functions inside it are compiled before multiprocessing forks.
    """
    # also warm up framework-internal @njit functions
    _stationary_bootstrap_indices(10, 5.0, 0)

    # Dummy data must be longer than any period the strategy might use.
    # Upper bounds can contain indicator periods AND cooldown days.
    # Use max(upper_bounds) + generous margin.
    lo, hi = spec["bounds"]
    max_param = max(hi) if hi else 200
    n = int(max_param) + 100  # e.g., bounds up to 200 → n=300
    n = max(n, 300)           # absolute minimum 300 bars

    dummy_close  = np.linspace(100.0, 100.0 + n * 0.1, n)
    dummy_high   = dummy_close + 2.0
    dummy_low    = dummy_close - 2.0
    dummy_volume = np.ones(n, dtype=np.float64) * 1000.0

    # use LOWER bounds for x — smallest valid periods, safest for warmup
    dummy_x = np.array([float(l) for l in lo])
    try:
        spec["simulate"](dummy_close, dummy_high, dummy_low, dummy_volume, dummy_x)
    except Exception:
        pass  # some strategies may fail on dummy data — that's fine,
              # the important thing is that numba compiled the functions

    # Also call with upper bounds to compile any parameter-dependent code paths
    dummy_x_hi = np.array([float(h) for h in hi])
    try:
        spec["simulate"](dummy_close, dummy_high, dummy_low, dummy_volume, dummy_x_hi)
    except Exception:
        pass

# ---------------------------------------------------------------------------
#  Data loading and caching
# ---------------------------------------------------------------------------

def get_history(ticker, start, end):
    """Download price history for *ticker* and cache as compressed CSV."""
    p = Path('ticker_cache')
    p.mkdir(exist_ok=True)
    fname = f'history_{ticker}_{start}_{end}.xz'
    file_path = p / fname
    if file_path.exists():
        df = pd.read_csv(file_path, compression='xz', index_col=0)
        df.index = pd.to_datetime(df.index)
        return df
    df = yf.download(ticker, start=start, end=end, auto_adjust=True)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if not df.empty:
        df.index = pd.to_datetime(df.index)
        df.index.name = 'Date'
        df.to_csv(file_path, compression='xz')
    return df


def load_tickers(tickers: TList[str], start: str, end: str):
    """Return dict ticker -> DataFrame for the full date range."""
    histories = {}
    for t in tickers:
        histories[t] = get_history(t, start=start, end=end)
    return histories

# ---------------------------------------------------------------------------
#  OHLCV extraction — the bridge between DataFrames and the strategy
# ---------------------------------------------------------------------------

def extract_ohlcv(df: pd.DataFrame) -> dict:
    """Extract OHLCV numpy arrays from a DataFrame (or a date-sliced view)."""
    return dict(
        close=df['Close'].to_numpy(dtype=np.float64),
        high=df['High'].to_numpy(dtype=np.float64),
        low=df['Low'].to_numpy(dtype=np.float64),
        volume=df['Volume'].to_numpy(dtype=np.float64),
    )

def synthetic_ohlcv(close: np.ndarray) -> dict:
    """
    Build an OHLCV dict from synthetic close prices (bootstrap path).
    High = Low = Close, Volume = 0.  Strategies that rely on H/L/V will
    get degraded signals — this is intentional: bootstrap tests the
    close-price behaviour of the strategy.
    """
    return dict(
        close=close,
        high=close.copy(),
        low=close.copy(),
        volume=np.zeros_like(close),
    )

# ---------------------------------------------------------------------------
#  Fitness for a single window (calls the strategy's simulate function)
# ---------------------------------------------------------------------------

class WindowFitness:
    """Evaluate a parameter vector on a fixed set of OHLCV windows."""

    def __init__(self, ohlcv_per_ticker: dict, simulate_fn):
        """
        ohlcv_per_ticker : {ticker: {close, high, low, volume}}
        simulate_fn      : the strategy's simulate(close, high, low, volume, x)
        """
        self.ohlcv = ohlcv_per_ticker
        self.tickers = list(ohlcv_per_ticker.keys())
        self.simulate_fn = simulate_fn
        # shared counters for parallel optimisation
        self.evals = mp.RawValue(ct.c_int, 0)
        self.best_y = mp.RawValue(ct.c_double, np.inf)
        self.t0 = time.perf_counter()

    def _simulate(self, ticker, x):
        d = self.ohlcv[ticker]
        return self.simulate_fn(d['close'], d['high'], d['low'], d['volume'], x)

    def __call__(self, x):
        factors = []
        for t in self.tickers:
            f, _ = self._simulate(t, x)
            factors.append(max(f, 1e-12))
        geo_mean = np.prod(factors) ** (1.0 / len(factors))
        y = -geo_mean
        self.evals.value += 1
        if y < self.best_y.value:
            self.best_y.value = y
            logger.info("nsim={0}: t={1:.1f}s fac={2:.3f} {3} x={4}".format(
                self.evals.value, dtime(self.t0), -y,
                [round(f, 3) for f in factors],
                [int(xi) for xi in x]))
        return y

    def evaluate(self, x):
        """Return per-ticker factors and trades (no logging / counters)."""
        factors, trades = [], []
        for t in self.tickers:
            f, nt = self._simulate(t, x)
            factors.append(f)
            trades.append(nt)
        return factors, trades

# ---------------------------------------------------------------------------
#  Single-window optimiser  (convenience wrapper)
# ---------------------------------------------------------------------------

def optimize_window(ohlcv_per_ticker, bounds, simulate_fn,
                    num_retries=24, max_evals=500):
    fit = WindowFitness(ohlcv_per_ticker, simulate_fn)
    ret = retry.minimize(fit, bounds, num_retries=num_retries,
                         optimizer=Bite_cpp(max_evals))
    return ret.x, -ret.fun

# ---------------------------------------------------------------------------
#  Scoring: condense walk-forward results to a single objective
# ---------------------------------------------------------------------------
#
#  The AI agent needs one number to decide "did my strategy change help?"
#
#  We use a log-wealth Sharpe score:
#
#      fold_log_returns = log(fold_geo_mean) for each fold
#      score = mean(fold_log_returns) - lambda * std(fold_log_returns)
#
#  Interpretation:
#    score = 0.0   -> capital preservation (broke even)
#    score > 0     -> profitable with good risk-adjusted growth
#    score < 0     -> losing money and/or too volatile
#
#  lambda = 0.5 is inspired by the Kelly criterion (optimal growth ~ mean - 0.5*var).
#  Using std instead of var gives slightly stronger risk-penalisation,
#  appropriate for out-of-sample evaluation where overfit strategies
#  tend to have fat-tailed losses.

def compute_score(fold_factors: TList[float], risk_lambda: float = 0.5) -> dict:
    """
    Compute the log-wealth Sharpe score from per-fold geometric-mean factors.

    Returns dict with keys: score, growth_rate, volatility, geo_mean,
    n_folds, frac_beat, worst_fold, best_fold.
    """
    f = np.array(fold_factors, dtype=np.float64)
    f = np.clip(f, 1e-12, None)
    log_f = np.log(f)
    mean_r = float(np.mean(log_f))
    std_r  = float(np.std(log_f))
    score  = mean_r - risk_lambda * std_r
    return dict(
        score=score,
        growth_rate=mean_r,
        volatility=std_r,
        geo_mean=float(np.exp(mean_r)),
        n_folds=len(f),
        frac_beat=float(np.mean(f > 1.0)),
        worst_fold=float(np.min(f)),
        best_fold=float(np.max(f)),
    )

# ---------------------------------------------------------------------------
#  Walk-forward validation
# ---------------------------------------------------------------------------

@dataclass
class WalkForwardFold:
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    best_x: np.ndarray = None
    train_factor: float = 0.0
    test_factors: TList[float] = field(default_factory=list)
    test_trades: TList[int] = field(default_factory=list)
    test_geo_mean: float = 0.0

@dataclass
class WalkForwardResult:
    folds: TList[WalkForwardFold] = field(default_factory=list)
    oos_geo_mean: float = 0.0
    oos_factors_per_ticker: dict = field(default_factory=dict)

    def score(self, risk_lambda: float = 0.5) -> dict:
        """Compute the single-objective score from all fold results."""
        fold_factors = [f.test_geo_mean for f in self.folds]
        return compute_score(fold_factors, risk_lambda)

    def summary(self, risk_lambda: float = 0.5) -> str:
        s = self.score(risk_lambda)
        lines = []
        lines.append(f"Walk-forward: {len(self.folds)} folds, "
                      f"OOS geo_mean = {self.oos_geo_mean:.4f}")
        lines.append(f"  >>> SCORE = {s['score']:.4f}  "
                      f"(growth={s['growth_rate']:.4f}, "
                      f"vol={s['volatility']:.4f}, "
                      f"lambda={risk_lambda})")
        lines.append(f"  profitable in {s['frac_beat']*100:.0f}% of folds, "
                      f"worst={s['worst_fold']:.3f}, "
                      f"best={s['best_fold']:.3f}")
        for i, fold in enumerate(self.folds):
            lines.append(
                f"  fold {i}: train [{fold.train_start}..{fold.train_end}] "
                f"test [{fold.test_start}..{fold.test_end}] "
                f"train_fac={fold.train_factor:.3f} "
                f"test_fac={fold.test_geo_mean:.3f} "
                f"x={[int(xi) for xi in fold.best_x]} "
                f"per_ticker={[round(f,3) for f in fold.test_factors]}")
        for t, fs in self.oos_factors_per_ticker.items():
            lines.append(f"  {t}: OOS factors across folds = "
                         f"{[round(f,3) for f in fs]}, "
                         f"geo_mean = {np.prod(fs)**(1/len(fs)):.3f}")
        return '\n'.join(lines)


def walk_forward(tickers: TList[str], start: str, end: str,
                 strategy_spec: dict,
                 train_days: int = 365, test_days: int = 90,
                 step_days: int = 90,
                 num_retries: int = 24, max_evals: int = 500) -> WalkForwardResult:
    """
    Rolling walk-forward optimisation.

    Slides a (train_days + test_days) window through the data advancing by
    step_days each fold.  Optimises on the training portion, evaluates the
    winning parameters on the test portion.
    """
    lo, hi = strategy_spec["bounds"]
    bounds = Bounds(lo, hi)
    simulate_fn = strategy_spec["simulate"]

    histories = load_tickers(tickers, start, end)
    ref = histories[tickers[0]]
    dates = ref.index
    n = len(dates)

    folds: TList[WalkForwardFold] = []
    i = 0
    while i + train_days + test_days <= n:
        train_idx = dates[i : i + train_days]
        test_idx  = dates[i + train_days : i + train_days + test_days]

        fold = WalkForwardFold(
            train_start=str(train_idx[0].date()),
            train_end=str(train_idx[-1].date()),
            test_start=str(test_idx[0].date()),
            test_end=str(test_idx[-1].date()),
        )

        # build training and test OHLCV dicts
        train_ohlcv, test_ohlcv = {}, {}
        for t in tickers:
            h = histories[t]
            train_ohlcv[t] = extract_ohlcv(h.loc[train_idx])
            test_ohlcv[t]  = extract_ohlcv(h.loc[test_idx])

        logger.info(f"=== Fold {len(folds)}: train [{fold.train_start}..{fold.train_end}], "
                     f"test [{fold.test_start}..{fold.test_end}] ===")

        # optimise on training window
        best_x, best_fac = optimize_window(
            train_ohlcv, bounds, simulate_fn,
            num_retries=num_retries, max_evals=max_evals)
        fold.best_x = best_x
        fold.train_factor = best_fac

        # evaluate on unseen test window
        test_fit = WindowFitness(test_ohlcv, simulate_fn)
        factors, trades = test_fit.evaluate(best_x)
        fold.test_factors = factors
        fold.test_trades = [int(nt) for nt in trades]
        fold.test_geo_mean = float(np.prod(factors) ** (1.0 / len(factors)))

        logger.info(f"  train_fac={fold.train_factor:.3f}  "
                     f"test_fac={fold.test_geo_mean:.3f}  "
                     f"per_ticker={[round(f,3) for f in factors]}  "
                     f"x={[int(xi) for xi in best_x]}")
        folds.append(fold)
        i += step_days

    # aggregate out-of-sample results
    result = WalkForwardResult(folds=folds)
    all_test_factors = [f.test_geo_mean for f in folds]
    if all_test_factors:
        result.oos_geo_mean = float(
            np.prod(all_test_factors) ** (1.0 / len(all_test_factors)))
    for t in tickers:
        result.oos_factors_per_ticker[t] = [
            f.test_factors[tickers.index(t)] for f in folds]
    return result

# ---------------------------------------------------------------------------
#  Stationary bootstrap  (Politis & Romano 1994)
# ---------------------------------------------------------------------------

@njit(fastmath=True)
def _stationary_bootstrap_indices(n: int, avg_block_len: float, seed: int) -> np.ndarray:
    np.random.seed(seed)
    p = 1.0 / avg_block_len
    idx = np.empty(n, dtype=np.int64)
    idx[0] = np.random.randint(0, n)
    for i in range(1, n):
        if np.random.random() < p:
            idx[i] = np.random.randint(0, n)
        else:
            idx[i] = (idx[i - 1] + 1) % n
    return idx


def stationary_bootstrap_prices(close: np.ndarray, avg_block_len: float,
                                n_samples: int, base_seed: int = 42) -> TList[np.ndarray]:
    """
    Generate *n_samples* synthetic price paths by stationary-bootstrapping
    the log-return series and reconstructing prices.
    """
    log_ret = np.diff(np.log(close))
    m = len(log_ret)
    paths = []
    for k in range(n_samples):
        idx = _stationary_bootstrap_indices(m, avg_block_len, base_seed + k)
        resampled_ret = log_ret[idx]
        prices = close[0] * np.exp(np.concatenate(([0.0], np.cumsum(resampled_ret))))
        paths.append(prices)
    return paths


def bootstrap_evaluate(tickers: TList[str], start: str, end: str,
                       strategy_spec: dict,
                       avg_block_len: float = 20.0,
                       n_bootstrap: int = 50,
                       num_retries: int = 24, max_evals: int = 500,
                       base_seed: int = 42) -> dict:
    """
    1. Optimise on the *real* data to get best_x.
    2. Generate n_bootstrap synthetic price paths per ticker.
    3. Re-evaluate best_x on each synthetic path.
    4. Report distribution of factors.
    """
    lo, hi = strategy_spec["bounds"]
    bounds = Bounds(lo, hi)
    simulate_fn = strategy_spec["simulate"]

    histories = load_tickers(tickers, start, end)
    real_ohlcv = {t: extract_ohlcv(histories[t]) for t in tickers}

    logger.info("=== Bootstrap: optimising on real data ===")
    best_x, real_factor = optimize_window(
        real_ohlcv, bounds, simulate_fn,
        num_retries=num_retries, max_evals=max_evals)
    real_fit = WindowFitness(real_ohlcv, simulate_fn)
    real_factors, _ = real_fit.evaluate(best_x)
    logger.info(f"  real factor = {real_factor:.4f}, per_ticker = "
                f"{[round(f,3) for f in real_factors]}, x = {[int(xi) for xi in best_x]}")

    logger.info(f"=== Bootstrap: generating {n_bootstrap} synthetic paths "
                f"(avg_block_len={avg_block_len}) ===")
    boot_paths = {}
    for t in tickers:
        boot_paths[t] = stationary_bootstrap_prices(
            real_ohlcv[t]['close'], avg_block_len, n_bootstrap, base_seed)

    boot_geo = np.empty(n_bootstrap)
    boot_per_ticker = {t: np.empty(n_bootstrap) for t in tickers}

    for k in range(n_bootstrap):
        factors = []
        for t in tickers:
            ohlcv = synthetic_ohlcv(boot_paths[t][k])
            f, _ = simulate_fn(ohlcv['close'], ohlcv['high'],
                               ohlcv['low'], ohlcv['volume'], best_x)
            factors.append(max(f, 1e-12))
            boot_per_ticker[t][k] = f
        boot_geo[k] = np.prod(factors) ** (1.0 / len(factors))

    ci5, ci95 = np.percentile(boot_geo, [5, 95])
    logger.info(f"=== Bootstrap results ({n_bootstrap} samples) ===")
    logger.info(f"  real factor        = {real_factor:.4f}")
    logger.info(f"  bootstrap median   = {np.median(boot_geo):.4f}")
    logger.info(f"  bootstrap mean     = {np.mean(boot_geo):.4f}")
    logger.info(f"  90% CI             = [{ci5:.4f}, {ci95:.4f}]")

    return dict(
        best_x=best_x,
        real_factor=real_factor,
        real_per_ticker=real_factors,
        bootstrap_factors=boot_geo,
        bootstrap_per_ticker=boot_per_ticker,
        ci_5=ci5, ci_95=ci95,
    )

# ---------------------------------------------------------------------------
#  Combined walk-forward + bootstrap
# ---------------------------------------------------------------------------

def walk_forward_bootstrap(tickers: TList[str], start: str, end: str,
                           strategy_spec: dict,
                           train_days: int = 365, test_days: int = 90,
                           step_days: int = 90,
                           avg_block_len: float = 20.0,
                           n_bootstrap: int = 50,
                           num_retries: int = 24, max_evals: int = 500,
                           base_seed: int = 42) -> dict:
    """
    Run walk-forward first, then for each fold additionally bootstrap the
    *training* window and re-optimise to measure parameter stability.
    """
    lo, hi = strategy_spec["bounds"]
    bounds = Bounds(lo, hi)
    simulate_fn = strategy_spec["simulate"]

    wf = walk_forward(tickers, start, end, strategy_spec,
                      train_days, test_days, step_days,
                      num_retries, max_evals)

    histories = load_tickers(tickers, start, end)
    ref_dates = histories[tickers[0]].index

    fold_boot = []
    for fi, fold in enumerate(wf.folds):
        logger.info(f"=== Bootstrap stability: fold {fi} training window ===")

        train_mask = (ref_dates >= fold.train_start) & (ref_dates <= fold.train_end)
        train_idx = ref_dates[train_mask]

        boot_close = {}
        for t in tickers:
            real = histories[t].loc[train_idx, 'Close'].to_numpy()
            boot_close[t] = stationary_bootstrap_prices(
                real, avg_block_len, n_bootstrap, base_seed + fi * 1000)

        boot_factors = np.empty(n_bootstrap)
        boot_xs = []
        for k in range(n_bootstrap):
            sample = {t: synthetic_ohlcv(boot_close[t][k]) for t in tickers}
            bx, bf = optimize_window(sample, bounds, simulate_fn,
                                     num_retries=max(4, num_retries // 4),
                                     max_evals=max_evals)
            boot_factors[k] = bf
            boot_xs.append(bx)

        ci5, ci95 = np.percentile(boot_factors, [5, 95])
        logger.info(f"  fold {fi}: real train_fac = {fold.train_factor:.3f}, "
                     f"bootstrap 90% CI = [{ci5:.3f}, {ci95:.3f}]")
        fold_boot.append(dict(
            fold=fi,
            real_train_factor=fold.train_factor,
            boot_train_factors=boot_factors,
            ci_5=ci5, ci_95=ci95,
            boot_xs=boot_xs,
        ))

    return dict(walk_forward=wf, fold_bootstrap=fold_boot)

# ---------------------------------------------------------------------------
#  CLI entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description="Walk-forward trading strategy optimizer with bootstrap validation")
    parser.add_argument('--tickers', nargs='+',
                        default=['AAPL', 'AMD', 'GOOGL', 'NVDA'],
                        help='Ticker symbols (yahoo finance)')
    parser.add_argument('--start', default='2019-01-01')
    parser.add_argument('--end', default='2030-04-30')
    parser.add_argument('--strategy', default='strategy',
                        help='Python module name containing get_strategy()')
    parser.add_argument('--mode', choices=['walkforward', 'bootstrap', 'combined', 'simple'],
                        default='walkforward',
                        help='Evaluation mode')
    parser.add_argument('--train-days', type=int, default=365,
                        help='Training window in trading days')
    parser.add_argument('--test-days', type=int, default=90,
                        help='Test window in trading days')
    parser.add_argument('--step-days', type=int, default=90,
                        help='Step size between folds in trading days')
    parser.add_argument('--n-bootstrap', type=int, default=50,
                        help='Number of bootstrap samples')
    parser.add_argument('--avg-block-len', type=float, default=20.0,
                        help='Mean block length for stationary bootstrap')
    parser.add_argument('--num-retries', type=int, default=24,
                        help='Parallel retries for optimizer')
    parser.add_argument('--max-evals', type=int, default=500,
                        help='Max evaluations per retry')
    parser.add_argument('--risk-lambda', type=float, default=0.5,
                        help='Risk penalty for score (0=pure growth, 0.5=Kelly, 1.0=strong)')
    args = parser.parse_args()

    # --- load strategy from the specified module ---
    spec = load_strategy(args.strategy)
    lo, hi = spec["bounds"]
    bounds = Bounds(lo, hi)
    simulate_fn = spec["simulate"]

    if args.mode == 'simple':
        histories = load_tickers(args.tickers, args.start, args.end)
        ohlcv = {t: extract_ohlcv(histories[t]) for t in args.tickers}
        best_x, best_fac = optimize_window(
            ohlcv, bounds, simulate_fn,
            num_retries=args.num_retries, max_evals=args.max_evals)
        logger.info(f"Best factor = {best_fac:.4f}, "
                    f"vars = {dict(zip(spec['variables'], [round(xi,1) for xi in best_x]))}")

    elif args.mode == 'walkforward':
        result = walk_forward(
            args.tickers, args.start, args.end, spec,
            train_days=args.train_days, test_days=args.test_days,
            step_days=args.step_days,
            num_retries=args.num_retries, max_evals=args.max_evals)
        logger.info('\n' + result.summary(risk_lambda=args.risk_lambda))

    elif args.mode == 'bootstrap':
        result = bootstrap_evaluate(
            args.tickers, args.start, args.end, spec,
            avg_block_len=args.avg_block_len, n_bootstrap=args.n_bootstrap,
            num_retries=args.num_retries, max_evals=args.max_evals)
        logger.info(f"Bootstrap 90% CI: [{result['ci_5']:.4f}, {result['ci_95']:.4f}]")

    elif args.mode == 'combined':
        result = walk_forward_bootstrap(
            args.tickers, args.start, args.end, spec,
            train_days=args.train_days, test_days=args.test_days,
            step_days=args.step_days,
            avg_block_len=args.avg_block_len, n_bootstrap=args.n_bootstrap,
            num_retries=args.num_retries, max_evals=args.max_evals)
        wf = result['walk_forward']
        logger.info('\n' + wf.summary(risk_lambda=args.risk_lambda))
        for fb in result['fold_bootstrap']:
            logger.info(f"  fold {fb['fold']}: bootstrap 90% CI on train factor = "
                         f"[{fb['ci_5']:.3f}, {fb['ci_95']:.3f}]")
