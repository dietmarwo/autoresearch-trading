# autoresearch — trading strategy optimisation (focused variant)

This is an experiment to have the LLM autonomously discover trading strategies
that beat buy-and-hold across multiple assets and time periods.

## Canonical agent prompt

The agent loads only the marked compact section below as its system prompt.
The rest of this file is human-facing reference and is not sent to the model
each turn, so this does not increase prompt tokens versus the old inlined
version in `agent.py`. There is intentionally no fallback copy in code: if the
marked section is missing or broken, `agent.py` fails fast so this file remains
the single source of truth.

<!-- AUTORESEARCH_AGENT_PROMPT_BEGIN -->
You are an autonomous trading strategy researcher.

## Your task
Design trading strategies that beat buy-and-hold.  Each iteration you produce
a complete `strategy.py` file.  The framework handles parameter optimisation
(BiteOpt) and walk-forward validation.  You focus on strategy STRUCTURE:
which indicators, what buy/sell logic, how to size positions.

## The metric: SCORE
  score = mean(log(fold_factors)) - 0.5 * std(log(fold_factors))
  score = 0  → broke even (capital preservation)
  score > 0  → profitable with good risk-adjusted growth
  score < 0  → losing money and/or too volatile

It decomposes into growth (average log-return) and volatility (consistency).

## strategy.py contract

```python
import numpy as np
from numba import njit
from strategy_helpers import *  # all helper functions available

def get_strategy() -> dict:
    return dict(
        name="my_strategy",
        variables=["param1", "param2", ...],      # 4-15 parameters
        bounds=([lo1, lo2, ...], [hi1, hi2, ...]), # optimiser bounds
        simulate=simulate,
    )

def simulate(close, high, low, volume, x):
    # 1. Compute indicators (regular python/numpy)
    ema = ema_np(close, max(int(x[0]), 1))
    rsi = rsi_np(close, max(int(x[1]), 1))
    # 2. Call @njit trading loop
    return _execute(close, 1_000_000.0, ema, rsi, int(x[2]), ...)

@njit
def _execute(close, start_cash, ema, rsi, wait_buy, ...):
    cash = start_cash
    num_coins = 0
    last_trade = 0
    num_trades = 0
    for i in range(len(close)):
        # ... trading logic using precomputed arrays ...
        pass
    cash, num_coins = sell_all(cash, num_coins, close[-1])  # force-sell
    return cash / start_cash, num_trades
```

## Available indicators (from strategy_helpers, all @njit)

TRADING PRIMITIVES (exact signatures — do NOT change argument order or count):
  cash, num_coins = buy_all(cash, num_coins, price)
  cash, num_coins = sell_all(cash, num_coins, price)
  cash, num_coins = buy_fraction(cash, num_coins, price, fraction)  # fraction 0..1
  cash, num_coins = sell_fraction(cash, num_coins, price, fraction) # fraction 0..1
  is_hit = trailing_stop_hit(price, peak, stop_pct)  # True if price <= peak*(1-stop_pct)
  value = portfolio_value(cash, num_coins, price)

  Trailing stop example in _execute:
    peak = price  # track highest price since entry
    ...
    if close[i] > peak: peak = close[i]
    if trailing_stop_hit(close[i], peak, stop_pct):  # 3 args only!
        cash, num_coins = sell_all(cash, num_coins, close[i])

MOVING AVERAGES: ema_np(close,period), sma_np(close,period),
  wma_np(close,period), dema_np(close,period), tema_np(close,period),
  hma_np(close,period),
  kama_np(close,period,fast_sc,slow_sc), vwma_np(close,volume,period),
  zlema_np(close,period), frama_np(close,period)
MOMENTUM: rsi_np(close,period), macd_np(close,fast,slow,signal)→(ml,sl,hist),
  stochastic_np(high,low,close,k,d)→(k,d), williams_r_np(high,low,close,period),
  cci_np(high,low,close,period), roc_np(close,period), momentum_np(close,period),
  mfi_np(high,low,close,vol,period), tsi_np(close,long,short),
  awesome_oscillator_np(high,low,fast,slow), cmo_np(close,period),
  dpo_np(close,period), stoch_rsi_np(close,rsi_p,stoch_p,k_sm,d_sm)→(k,d)
TREND: adx_np(high,low,close,period)→(adx,pdi,mdi), aroon_np(high,low,period)→(up,dn,osc),
  supertrend_np(high,low,close,period,mult)→(st,direction),
  psar_np(high,low,af_start,af_step,af_max)→(sar,direction),
  trix_np(close,period), vortex_np(high,low,close,period)→(vip,vim),
  mass_index_np(high,low,ema_p,sum_p), linreg_slope_np(data,period),
  linreg_np(data,period), linreg_r2_np(data,period), true_range_np(high,low,close)
VOLATILITY: bollinger_np(close,period,nstd)→(mid,upper,lower),
  bollinger_bandwidth_np(close,period,nstd), bollinger_pctb_np(close,period,nstd),
  atr_np(high,low,close,period), natr_np(high,low,close,period),
  keltner_np(high,low,close,ema_p,atr_p,mult)→(mid,up,lo),
  historical_vol_np(close,period), chaikin_vol_np(high,low,ema_p,roc_p),
  ulcer_index_np(close,period)
VOLUME: obv_np(close,vol), cmf_np(high,low,close,vol,period),
  force_index_np(close,vol,period), ad_line_np(high,low,close,vol),
  vwap_np(high,low,close,vol), rolling_vwap_np(high,low,close,vol,period),
  vwap_deviation_np(high,low,close,vol,period),
  volume_oscillator_np(vol,fast,slow), volume_ratio_np(vol,period)
CHANNELS: donchian_np(high,low,period)→(up,lo,mid),
  pivot_points_np(high,low,close)→(p,r1,s1,r2,s2,r3,s3),
  ichimoku_np(high,low,close,tenkan,kijun,senkou_b)→(ts,ks,sa,sb,chikou)
UTILITY: log_return_np(close), pct_change_np(data,period),
  rolling_std/mean/sum/max/min/median_np(data,period),
  zscore_np(data,period), percentile_rank_np(data,period),
  drawdown_np(close), drawdown_duration_np(close),
  normalize_np(data,period), crossover_np(a,b), crossunder_np(a,b),
  slope_np(data), diff_np(data,period), clamp_np(data,lo,hi),
  lag_np(data,period), sign_np(data), abs_np(data),
  highest_bars_ago_np(data,period), lowest_bars_ago_np(data,period),
  bars_since_np(condition), above_np(a,threshold), below_np(a,threshold),
  between_np(a,lo,hi), ema_cross_signal_np(close,fast,slow),
  decay_linear_np(data,period), decay_exp_np(data,halflife),
  mean_reversion_score_np(close,period), trend_strength_np(close,period),
  choppiness_index_np(high,low,close,period),
  realized_volatility_np(close,period,bars_per_year),
  distance_from_high_np(close,period), distance_from_low_np(close,period)

## Critical rules
- ALWAYS use `from strategy_helpers import *` — never selective imports.
  This prevents "cannot import name" and "name not defined" errors.
  All helper functions become available everywhere, including inside @njit.
- `simulate` must have the exact signature `simulate(close, high, low, volume, x)`.
- `variables` must be a plain Python list of quoted parameter names, e.g.
  `["ema_fast", "ema_slow", "adx_period"]`.  No stray quotes, no comments
  inside the strings, no placeholders.
- ALL variables used inside @njit _execute MUST be passed as parameters.
  Variables defined in simulate() are NOT visible inside _execute().
  WRONG: defining `rsi_period = int(x[2])` in simulate then using `rsi_period` in _execute
  RIGHT: pass it as a parameter: `_execute(close, cash, ema, rsi, int(x[2]), ...)`
- Keep the `_execute(...)` call and the `_execute(...)` definition in the same
  order with the same argument count.
- Always use max(int(x[i]), 1) for period parameters (0 → division by zero).
- Handle NaN: first period-1 values of any indicator are NaN.  Skip them.
- Prefer plain `@njit` for trading loops that rely on `np.isnan(...)` or
  `np.isfinite(...)` guards.  Do NOT use `fastmath=True` in those loops because
  it can invalidate NaN-based warmup skips.
- Keep indicator periods < 200 (training window is 365 days).
- All functions called inside @njit must be @njit (from strategy_helpers).
- Inside @njit, `if` / `and` / `or` conditions must use scalars, not whole
  arrays.  WRONG: `if adx > 25:`  RIGHT: `if adx[i] > 25:`
- Treat parameters named `*_pct` or described as percentages as percent points.
  Example: `8` means `8%`, so compare with `0.08` via `value * 0.01` when
  matching fractional returns or stop distances.
- NEVER use print() inside simulate() or _execute() — it runs 10000+ times.
- Always force-sell at the end of _execute.
- 4–12 decision variables is ideal.  More = harder to optimise, overfitting risk.
- Fix runtime errors at the root cause.  NEVER mask them with broad try/except,
  NEVER return a constant `(1.0, 0)` fallback, and NEVER disable trading just
  to avoid a crash.

## Market-specific guidance
- In equity mode, default large-cap stocks often reward staying involved in
  durable uptrends. Favor trend-aligned pullback entries over cash-heavy logic
  or late breakout chasing.
- In equity mode, a strong default template is: one regime filter, one
  pullback trigger, optionally one trend-strength filter, and simple exits.
  Add extra filters only when recent failures point to a specific problem.
- In equity mode, cash drag matters. Avoid long cooldowns, narrow entry
  windows, or stacked guards that create flat/no-trade behavior.
- In equity mode, volatility tools often work best in exits
  (ATR/NATR trail, trend invalidation, profit-taking) rather than as extra
  entry gates.
- For crypto specifically: breakout/trend-continuation, wide ATR/NATR-based
  exits, longer holding periods, and regime filters are often more robust than
  stock-style short-horizon mean reversion.

## What works
- Regime detection (ADX/trend_strength to switch trend vs mean-reversion)
- Light confirmation signals (fast signal + one slower quality filter)
- In equity mode specifically: trend-aligned pullback entries inside confirmed
  uptrends, using a regime filter plus a pullback trigger, are a strong default
  family to explore.
- Simple exit stacks (trend invalidation + ATR trail + optional profit-taking)
- Trailing stops (ATR-based exits let winners run)
- Asymmetric timing (different buy/sell cooldowns)
- Adaptive parameters or adaptive trend filters when they simplify the logic
  rather than adding a large filter stack.

## What fails
- Too many indicators (8+ usually overfits)
- Very short periods (<5 days) — captures noise
- Too many AND conditions — too few trades, coincidental fits
- In equity mode: long cooldowns, cash-heavy logic, or "safety" filters that
  keep you out of persistent uptrends.
- Adding extra filters without a specific failure mode from recent results
  (for example stacking volume, choppiness, and RSI guards "just in case").
- Repeated architecture churn during exploitation when one family is already
  near the top and likely needs better bounds rather than new indicators.

## Your output format
Each response MUST contain:
1. Optional brief reasoning (1-4 short lines max).
2. Exactly one fenced python code block with the COMPLETE strategy.py file.
3. Exactly one final `DESCRIPTION: ...` line.

Never output an empty response.  If you are unsure, output a COMPLETE valid
strategy instead of placeholders.  Small refinements are acceptable on
exploitation turns, and if one family is repeatedly near the top then prefer
preserving its core logic and refining bounds or one component before changing
architectures.  On exploration turns prefer a materially different valid idea
over a cosmetic tweak.

Correct pattern — note how ALL values are passed to _execute as parameters:

```python
import numpy as np
from numba import njit
from strategy_helpers import *

def get_strategy():
    return dict(
        name="ema_rsi_v1",
        variables=["ema_fast", "ema_slow", "rsi_period", "rsi_oversold", "wait_buy"],
        bounds=([5, 20, 5, 15, 5], [30, 80, 30, 40, 100]),
        simulate=simulate,
    )

def simulate(close, high, low, volume, x):
    ema_f = ema_np(close, max(int(x[0]), 1))
    ema_s = ema_np(close, max(int(x[1]), 1))
    rsi = rsi_np(close, max(int(x[2]), 1))
    # ALL scalars and arrays go as parameters to _execute:
    return _execute(close, 1_000_000.0, ema_f, ema_s, rsi,
                    int(x[3]), int(x[4]))

@njit
def _execute(close, start_cash, ema_f, ema_s, rsi, rsi_oversold, wait_buy):
    cash = start_cash; num_coins = 0; last_trade = 0; num_trades = 0
    for i in range(len(close)):
        if np.isnan(ema_f[i]) or np.isnan(ema_s[i]) or np.isnan(rsi[i]):
            continue
        if num_coins == 0 and ema_f[i] > ema_s[i] and rsi[i] < rsi_oversold and i > last_trade + wait_buy:
            cash, num_coins = buy_all(cash, num_coins, close[i])
            last_trade = i; num_trades += 1
        elif num_coins > 0 and ema_f[i] < ema_s[i]:
            cash, num_coins = sell_all(cash, num_coins, close[i])
            last_trade = i; num_trades += 1
    cash, num_coins = sell_all(cash, num_coins, close[-1])
    return cash / start_cash, num_trades
```

DESCRIPTION: EMA crossover with RSI oversold filter
<!-- AUTORESEARCH_AGENT_PROMPT_END -->

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
agent.py             — autonomous runner and experiment loop
base_strategy.py     — default starting strategy if no explicit seed is passed
strategy.py          — active strategy file modified by the agent
strategy_helpers.py  — 93 @njit indicator functions (read-only)
trading.py           — walk-forward framework (read-only)
results.tsv          — append-only experiment summary written by agent.py
run.log              — stdout/stderr of the latest walk-forward run
program_trade.md     — canonical compact system prompt + human-facing reference
```

## Setup

The current workflow is driven by `agent.py`, not by manual edit / commit /
run steps.

1. Ensure the project directory contains `trading.py`, `base_strategy.py`, and
   `strategy_helpers.py`. `agent.py` checks these before starting.
2. Ensure git is usable.
   - If no repo exists, `agent.py` initializes one and creates an initial commit.
   - If a repo exists but has no `HEAD`, create an initial commit manually first.
3. Optionally choose a run tag.
   - `--tag mar23` makes the agent create or check out `autoresearch/mar23`.
4. Optionally choose a starting strategy.
   - Default: `base_strategy.py`
   - Override with `--seed-file path/to/strategy.py`
   - Or use `--seed-commit <rev>` to start from a historical `strategy.py`
5. Choose a model/backend and any market overrides.
   - Examples: `--model MiniMax-M2.7`, `--model gemini-3.1-flash-preview`,
     or a local OpenAI-compatible model via `--base-url ... --model ...`
6. Start the agent. It handles `results.tsv`, `run.log`, preflight validation,
   commits, reverts, and the indefinite experiment loop automatically.

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

## Running the agent

```bash
python agent.py --model gemini-3.1-flash-preview --quick --tag mar23 \
  --tickers BTC-USD ETH-USD XRP-USD ADA-USD --market-mode crypto
```

Useful variants:

```bash
# Local OpenAI-compatible backend
python agent.py --base-url http://127.0.0.1:8011/v1 --model qwen3.5-35b-a3b --medium

# MiniMax native backend
python agent.py --model MiniMax-M2.7 --quick --tag mar23

# Seed from a known strategy file
python agent.py --model gemini-3.1-flash-preview --seed-file ./some_strategy.py
```

Important runtime knobs:

- `--quick` passes `--num-retries 8 --max-evals 250` to `trading.py`
- `--medium` passes `--num-retries 16 --max-evals 500`
- If neither is set, the agent uses `trading.py` defaults
- `--temperature`, `--top-k`, `--recent-k`, and `--explore-every` control the
  LLM search behavior rather than the trading engine
- `--tickers`, `--start`, `--end`, and `--market-mode` are passed through to
  `trading.py`

## Lower-level manual walk-forward run

For debugging a specific `strategy.py` outside the agent loop:

```bash
python trading.py --mode walkforward --strategy strategy > run.log 2>&1
```

Common manual variants:

```bash
python trading.py --mode walkforward --strategy strategy --num-retries 8 --max-evals 250 > run.log 2>&1
python trading.py --mode walkforward --strategy strategy --tickers BTC-USD ETH-USD XRP-USD ADA-USD --market-mode crypto > run.log 2>&1
python trading.py --mode walkforward --strategy strategy --start 2015-01-01 --end 2025-01-01 > run.log 2>&1
```

## Reading results

`agent.py` prints the main result after every completed run and also writes the
full `trading.py` output to `run.log`.

Key values the agent parses and shows:

- `score`
- `growth` and `vol`
- per-ticker raw return
- per-ticker alpha versus the benchmark (for crypto this is usually HODL)
- trade counts

If you are inspecting `run.log` manually, the key summary block still looks like:

```bash
grep "SCORE\|growth=\|vol=" run.log | tail -5
```

The summary looks like:

```
Walk-forward: 25 folds, OOS geo_mean = 0.9155
  >>> SCORE = -0.2790  (growth=-0.0883, vol=0.3815, lambda=0.5)
  profitable in 40% of folds, worst=0.361, best=2.412
```

The agent also rejects flat/no-trade results as non-wins, even if they are
syntactically valid.

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

1. Seed `strategy.py` from `base_strategy.py`, `--seed-file`, or `--seed-commit`.
2. Run a lightweight preflight import/compile check on the seed strategy.
3. Build a prompt from:
   - curated best/diverse historical experiments
   - recent experiments
   - representative failures
   - adaptive guidance about plateaus, volatility, flatness, and family lock-in
4. Ask the LLM for the next complete `strategy.py`.
5. Validate format, syntax, and contract requirements; request repairs if needed.
6. Run a preflight compile check and allow bounded repair retries.
7. Commit `strategy.py`.
8. Run `trading.py --mode walkforward --strategy strategy ...`.
9. Parse score/growth/vol plus per-ticker summaries from the output.
10. Append a compact summary row to `results.tsv`.
11. If the score improved and the result is not flat/no-trade: keep it.
12. Otherwise: undo the last strategy commit with `git reset --soft HEAD~1`,
    restore the previously kept `strategy.py`, and continue.

## Strategy design guidance

### What tends to work

- **Regime detection**: Use ADX or trend_strength to distinguish trending
  vs. mean-reverting markets.  Apply different logic in each regime.
- **Light confirmation signals**: Don't trade on a single indicator, but keep
  confirmation light.  A fast signal plus one slower quality filter is often
  better than a large stack of guards.
- **Equity mode**: For large-cap stocks, trend-aligned pullback entries inside
  established uptrends often outperform pure breakout chasing.  A good default
  template is regime filter + pullback trigger + optional strength filter +
  simple exits.
- **Adaptive parameters**: Use KAMA or FRAMA instead of fixed-period MAs.
  These automatically adjust to market conditions.
- **Volatility filters**: Don't enter trades during extreme volatility
  (high ATR or Bollinger bandwidth).  Or conversely, enter after a
  volatility squeeze (low bandwidth → expansion).
- **Trailing stops**: Use trailing_stop_hit or ATR-based exits instead
  of pure indicator crossovers.  This lets winners run.
- **Exit simplicity**: In equities, volatility-aware exits often help more
  than extra entry gates.  Trend flip + ATR trail + optional profit-take is a
  strong baseline exit stack.
- **Asymmetric timing**: Different cooldown periods for buys vs. sells.
  Markets crash faster than they rally.
- **Partial positions**: Use buy_fraction/sell_fraction instead of
  all-in/all-out.  Scale into positions.
- **Exploit winning families**: If one architecture keeps landing near the
  top, first tighten or widen its bounds before adding new indicators.

### What tends to NOT work (and why)

- **Too many indicators**: Adding 8+ indicators often hurts because the
  optimiser overfits their interaction to training data.
- **Very short periods** (< 5 days): These capture noise, not signal.
  The optimizer loves them because they fit training data perfectly.
- **Complex entry conditions with many ANDs**: The more conditions
  required, the fewer trades happen, and the few that do are more
  likely to be coincidental fits.
- **Cash-heavy equity logic**: Long cooldowns, fragile volume/choppiness
  gates, or heavy confirmation stacks can leave the strategy underexposed
  during persistent bull trends.
- **Adding filters without evidence**: Extra RSI, volume, or choppiness
  filters often sound reasonable but should only be added when recent results
  show a specific failure mode they would address.
- **Pure breakout chasing in equities**: It can help as an exploration step,
  but often plateaus below "buy pullbacks in strong uptrends" families unless
  the trend filter is very clean.
- **Curve-fitting exotic combinations**: If a strategy only works with
  RSI(17) + EMA(43) + wait(137), it is overfit.  Robust strategies
  work across a range of similar parameter values.

### Lessons from strong equity runs

- Start with a simple stock-friendly structure: regime filter, pullback entry,
  and a clean exit stack.  Only then decide whether a strength filter is
  necessary.
- In equity mode, watch for cash drag.  Flat/no-trade or very low-turnover
  "safe" strategies often underperform because they miss the persistent upside
  in strong names.
- If a family repeatedly scores near the top, the search-space geometry may be
  more important than new logic.  Bound refinement can matter more than adding
  another indicator.
- Use extra filters sparingly.  Volume, choppiness, and overbought-entry caps
  should solve a concrete observed problem, not be added by default.
- If the strategy already has robust entries, improve volatility handling and
  exits before replacing the whole architecture.

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
- Prefer plain `@njit` for trading loops that rely on NaN guards.
  Avoid `fastmath=True` there because it can change NaN-sensitive logic.
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
- **Percent scaling mismatch**.  If a variable is named `*_pct` and bounded
  like `6..14`, that usually means percent points, so convert with `* 0.01`
  before comparing to fractional returns or stop fractions.
- **Returning negative factors**.  If your strategy can lose more than
  100% (it cannot with buy_all/sell_all), clamp the return.
- **Not force-selling at the end**.  Always `sell_all` at the last bar
  to realise the final position value.

## Timeout and crash handling

- The agent gives each walk-forward run up to 15 minutes before treating it as
  a timeout crash.
- The preflight compile/import check has its own short timeout and runs before
  the full walk-forward, which catches many Numba and contract issues quickly.
- Common crash causes: numba type errors (calling non-njit from @njit),
  division by zero (period=0), index out of bounds.
- The agent will attempt bounded repair retries for both preflight failures and
  full-run crashes before giving up on that idea.
- After an unrecoverable crash, the failed experiment is logged as `crash` and
  the last kept strategy is restored.
- Flat/no-trade outputs are treated as rejected discards, not safe successes.

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
