# autoresearch — trading strategy optimisation

This is an experiment to have the LLM autonomously discover trading strategies
that beat buy-and-hold across multiple assets and time periods.

## How it works

A conventional optimiser (BiteOpt via fcmaes) handles parameter tuning.
Your job is the *creative* part: designing the strategy structure — which
indicators to use, how to combine them, what the buy/sell logic should be,
how to size positions.  The optimiser finds the best parameters for whatever
structure you propose.  Walk-forward validation then tests whether those
parameters generalise to unseen data.

The single number you are optimising is the **SCORE**, a log-wealth Sharpe:

```
score = mean(log(fold_factors)) - 0.5 * std(log(fold_factors))
```

- `score = 0.0` means you broke even (capital preservation).
- `score > 0` means you are profitable with good risk-adjusted growth.
- `score < 0` means you are losing money and/or too volatile.

The score decomposes into two parts reported after each run:

- **growth** = mean of log-factors (are you growing capital on average?)
- **volatility** = std of log-factors (are you consistent across folds?)

Both matter.  A strategy with great average growth but wild swings across
folds is probably overfit to specific market regimes.  The λ=0.5 penalty
(inspired by the Kelly criterion) balances these.

## Repository structure

```
strategy.py          — the file you modify (your "train.py")
strategy_helpers.py  — 93 @njit indicator functions (read-only)
trading.py           — walk-forward framework (read-only)
program_trade.md     — this file (read-only)
```

## Setup

To set up a new experiment, work with the user to:

1. **Agree on a run tag**: propose a tag based on today's date (e.g. `mar18`).
   The branch `autoresearch/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b autoresearch/<tag>` from master.
3. **Read the in-scope files** for full context:
   - `strategy.py` — the file you modify.
   - `strategy_helpers.py` — the indicator library. Read this carefully.
     It contains 93 `@njit` functions across 8 categories: trading primitives,
     moving averages (EMA/SMA/WMA/DEMA/TEMA/HMA/KAMA/VWMA/ZLEMA/FRAMA),
     momentum (RSI/MACD/Stochastic/Williams %R/CCI/ROC/MFI/TSI/CMO),
     trend (ADX/Aroon/Supertrend/Parabolic SAR/TRIX/Vortex/linear regression),
     volatility (Bollinger/ATR/Keltner/NATR/historical vol/ulcer index),
     volume (OBV/CMF/Force Index/A-D line/VWAP),
     channels (Donchian/pivot points/Ichimoku),
     and utility (crossover/zscore/drawdown/normalize/bars_since/etc).
   - `trading.py` — the framework. You do not modify this, but understanding
     how `WindowFitness` calls your `simulate()` function is useful.
4. **Verify ticker cache**: Check that `ticker_cache/` contains data files.
   If not, run `python trading.py --mode simple` once to download.
5. **Initialise results.tsv**: Create with just the header row.
6. **Confirm and go**: Confirm setup looks good.

Once you get confirmation, kick off the experimentation.

## The strategy interface

Your `strategy.py` must define `get_strategy()` returning a dict:

```python
def get_strategy() -> dict:
    return dict(
        name="my_strategy_v1",           # human-readable name
        variables=["param1", "param2"],   # parameter names for logging
        bounds=([lo1, lo2], [hi1, hi2]),  # optimiser bounds
        simulate=simulate,               # the function below
    )
```

And a `simulate` function with this exact signature:

```python
def simulate(close, high, low, volume, x) -> (growth_factor, num_trades):
```

- `close`, `high`, `low`, `volume` — numpy float64 arrays (daily OHLCV)
- `x` — numpy float64 array of decision variables from the optimiser
- returns `(float, int)` — cash / start_cash (growth factor), and trade count

**Architecture pattern** — your simulate function has two layers:

1. **Indicator computation** (regular Python/numpy): call helpers like
   `ema_np(close, period)`, `rsi_np(close, period)` etc. to produce arrays.
2. **Trading loop** (`@njit`): iterate through bars, use the precomputed
   arrays and scalar parameters to decide buy/sell actions.

The indicator layer runs once per call.  The trading loop is compiled by
numba for speed.  This separation is important because the optimiser calls
`simulate()` 10,000+ times per fold.

## What you CAN do

- **Modify `strategy.py`** — this is the only file you edit. Everything is
  fair game: indicators, entry/exit logic, position sizing, number of
  parameters, bound ranges.
- **Import any function from `strategy_helpers`**.  All 93 functions are
  `@njit`-compiled and can be called both from indicator prep and from
  inside `@njit` trading loops.
- **Add new `@njit` helper functions** inside `strategy.py`.
- **Change the number of decision variables**.  4–12 is a good range.
  More than ~15 variables makes the optimiser's job much harder and
  increases overfitting risk.

## What you CANNOT do

- Modify `trading.py` or `strategy_helpers.py`.  They are read-only.
- Install new packages or add dependencies.
- Use pandas or any non-numba-compatible code inside `@njit` functions.
- Use `print()` or logging inside `simulate()` — it is called thousands
  of times per second.

## Running an experiment

```bash
python trading.py --mode walkforward --strategy strategy > run.log 2>&1
```

Default settings: 4 US large-cap stocks (AAPL, MSFT, GOOGL, AMZN), 365/90/90 day
train/test/step, 24 parallel retries, 500 evaluations per retry.

For faster iteration during exploration, you can reduce the load:

```bash
# Quick test (~15s) — fewer retries, fewer evals
python trading.py --mode walkforward --num-retries 8 --max-evals 250 > run.log 2>&1

# Default (~1-2min) — good balance for iteration
python trading.py --mode walkforward > run.log 2>&1

# Full validation (~5-10min) — for confirming promising changes
python trading.py --mode walkforward --num-retries 48 --max-evals 2000 > run.log 2>&1
```

You can also change the tickers and time range:

```bash
# Crypto (more volatile, harder to beat breakeven in bull markets)
python trading.py --mode walkforward --tickers BTC-USD ETH-USD XRP-USD ADA-USD > run.log 2>&1

# Different time range
python trading.py --mode walkforward --start 2015-01-01 --end 2025-01-01 > run.log 2>&1
```

## Reading results

The output ends with a summary block.  Extract the key metric:

```bash
grep "SCORE\|growth=\|vol=" run.log | tail -5
```

The summary looks like:

```
Walk-forward: 25 folds, OOS geo_mean = 0.9155
  >>> SCORE = -0.2790  (growth=-0.0883, vol=0.3815, lambda=0.5)
  profitable in 40% of folds, worst=0.361, best=2.412
```

Read the **SCORE** line.  Higher is better.  Zero means capital preservation.

If grep returns nothing, the run crashed.  Use `tail -n 50 run.log` to
see the traceback.

## Output format for results.tsv

Tab-separated, 5 columns:

```
commit	score	status	growth_vol	description
```

1. git commit hash (short, 7 chars)
2. score achieved (e.g. -0.2790) — use 0.0000 for crashes
3. status: `keep`, `discard`, or `crash`
4. growth and vol (e.g. `g=-0.088/v=0.381`) — use `n/a` for crashes
5. short text description of what this experiment tried

Example:

```
commit	score	status	growth_vol	description
a1b2c3d	-0.2790	keep	g=-0.088/v=0.381	baseline EMA/SMA crossover
b2c3d4e	-0.1500	keep	g=-0.030/v=0.240	add RSI filter oversold<30
c3d4e5f	-0.3100	discard	g=-0.050/v=0.520	add Bollinger squeeze (too volatile)
d4e5f6g	0.0000	crash	n/a	numba type error in _execute
```

## The experiment loop

LOOP FOREVER:

1. Look at the current state: the branch, last SCORE, what has been tried.
2. Think about what to try next.  Consider:
   - Which component of the score to attack (growth vs. volatility)?
   - What information is unused?  (volume? high/low? volatility regime?)
   - What successful trading strategies look like in the literature?
   - Can you simplify the current strategy without losing performance?
3. Edit `strategy.py` with your experimental idea.
4. `git add strategy.py && git commit -m "description of change"`
5. Run: `python trading.py --mode walkforward --strategy strategy > run.log 2>&1`
6. Read results: `grep "SCORE\|growth=\|vol=" run.log | tail -5`
7. If grep is empty → crash.  `tail -n 50 run.log` to diagnose.
8. Log results to results.tsv (do NOT commit this file).
9. If SCORE improved (higher): keep the commit, advance the branch.
10. If SCORE is equal or worse: `git reset --soft HEAD~1` to revert without
    touching other tracked files.

## Strategy design guidance

### What tends to work

- **Regime detection**: Use ADX or trend_strength to distinguish trending
  vs. mean-reverting markets.  Apply different logic in each regime.
- **Confirmation signals**: Don't trade on a single indicator.  Use a fast
  signal (eMA cross) confirmed by a slower filter (RSI not overbought,
  volume above average, ADX > 20).
- **Adaptive parameters**: Use KAMA or FRAMA instead of fixed-period MAs.
  These automatically adjust to market conditions.
- **Volatility filters**: Don't enter trades during extreme volatility
  (high ATR or Bollinger bandwidth).  Or conversely, enter after a
  volatility squeeze (low bandwidth → expansion).
- **Trailing stops**: Use trailing_stop_hit or ATR-based exits instead
  of pure indicator crossovers.  This lets winners run.
- **Asymmetric timing**: Different cooldown periods for buys vs. sells.
  Markets crash faster than they rally.
- **Partial positions**: Use buy_fraction/sell_fraction instead of
  all-in/all-out.  Scale into positions.

### What tends to NOT work (and why)

- **Too many indicators**: Adding 8+ indicators often hurts because the
  optimiser overfits their interaction to training data.
- **Very short periods** (< 5 days): These capture noise, not signal.
  The optimizer loves them because they fit training data perfectly.
- **Complex entry conditions with many ANDs**: The more conditions
  required, the fewer trades happen, and the few that do are more
  likely to be coincidental fits.
- **Curve-fitting exotic combinations**: If a strategy only works with
  RSI(17) + EMA(43) + wait(137), it is overfit.  Robust strategies
  work across a range of similar parameter values.

### Thinking about the score components

If **growth is negative** (you are losing money on average):
→ Your entry/exit signals are poorly timed.  Try different indicators
  or different logic.  Check if you're buying tops / selling bottoms.

If **growth is positive but volatility is high**:
→ Your strategy works in some market regimes but fails in others.
  Add regime detection.  Or add filters that prevent trading in
  unfavourable conditions.

If **growth is near zero and volatility is low**:
→ Your strategy is conservative but not adding value.  Try being
  more aggressive in favourable conditions while keeping the filters.

### Numba constraints

- All functions called inside `@njit` must also be `@njit`.
- No Python objects, strings, lists, dicts inside `@njit`.
- Use `np.nan` checks: `if np.isnan(x): continue`.
- Cast float parameters to int where needed: `period = int(x[0])`.
- Ensure period parameters are ≥ 1: `period = max(int(x[0]), 1)`.
- All arrays must be float64 numpy arrays.

### Common mistakes

- **Forgetting `max(int(x[i]), 1)`** for period parameters.
  Period = 0 causes division by zero inside indicator functions.
- **Not handling NaN** at the start of indicator arrays.  The first
  `period-1` values of any MA/RSI/etc are NaN.  Skip them.
- **Indicator period > window length**.  If your training window is
  365 days and you use SMA(200), only 165 days have valid signals.
  With SMA(300), almost nothing is valid.
- **Returning negative factors**.  If your strategy can lose more than
  100% (it cannot with buy_all/sell_all), clamp the return.
- **Not force-selling at the end**.  Always `sell_all` at the last bar
  to realise the final position value.

## Timeout and crash handling

- Each walk-forward run should complete in 1–2 minutes with default
  settings.  If it exceeds 10 minutes, kill it and treat as crash.
- Common crash causes: numba type errors (calling non-njit from @njit),
  division by zero (period=0), index out of bounds.
- If a crash is a simple fix (typo, missing import), fix and re-run.
- If the idea is fundamentally broken, log as crash and revert.

## NEVER STOP

Once the experiment loop has begun, do NOT pause to ask the human if you
should continue.  The human may be away and expects you to work
*indefinitely* until manually stopped.  If you run out of ideas:

- Re-read `strategy_helpers.py` for indicators you have not tried.
- Try combining two near-miss strategies.
- Try the opposite of what failed (if momentum failed, try mean reversion).
- Try different assets (`--tickers BTC-USD ETH-USD XRP-USD ADA-USD` for crypto).
- Try removing complexity from a working strategy.
- Try adding one carefully chosen filter to a working strategy.
- Look at the per-ticker breakdown — if one ticker drags the score down,
  think about what makes it different.

The loop runs until the human interrupts you.
