#!/usr/bin/env python3
"""
agent.py — Autonomous trading strategy researcher.

Connects to a local LLM (llama-server / OpenAI-compatible API) or native
frontier APIs (Claude/Gemini) and runs the experiment loop from
program_trade.md:

  1. LLM proposes a new strategy.py
  2. Runner validates syntax, commits, runs walk-forward
  3. Runner parses SCORE, keeps or reverts
  4. Results feed back to LLM for the next iteration

Usage:
    # Start llama-server first, then:
    python agent.py                          # defaults
    python agent.py --tag mar18              # custom branch tag
    python agent.py --quick                  # fast iteration (fewer retries)
    python agent.py --tickers BTC-USD ETH-USD XRP-USD ADA-USD  # crypto
    python agent.py --base-url http://host:8011/v1
    python agent.py --base-url https://api.openai.com/v1 --model gpt-5.4-mini
    python agent.py --model claude-sonnet-4-6
    python agent.py --model gemini-3.1-pro-preview
"""

import os
import sys
import ast
import re
import json
import time
import math
import subprocess
import argparse
import textwrap
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from openai import OpenAI

# ═══════════════════════════════════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════════════════════════════════

STRATEGY_FILE = "strategy.py"
BASE_STRATEGY_FILE = "base_strategy.py"
RESULTS_FILE = "results.tsv"
RUN_LOG = "run.log"

MAX_CRASH_RETRIES = 3       # retries before giving up on one idea
RUN_TIMEOUT = 900            # 15 min max per walk-forward run
MAX_CONTEXT_EXCHANGES = 2    # keep last N exchanges (lightweight, no code duplication)
TOP_K = 10                   # curated best/diverse experiments shown to LLM
RECENT_K = 10                # recent experiments shown to LLM
TEMPERATURE = 0.7
REFERENCE_CODE_CHARS = 3500  # keep discarded-code references compact for 50K context
MAX_LLM_OUTPUT_TOKENS = 8192

# Resolved at startup — the directory containing agent.py, strategy.py, trading.py
PROJECT_DIR: Path = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════
#  System prompt — compact version of program_trade.md
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = r"""You are an autonomous trading strategy researcher.

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
from strategy_helpers import *  # all 93 njit functions available

def get_strategy() -> dict:
    return dict(
        name="my_strategy",
        variables=["param1", "param2", ...],      # 4-15 parameters
        bounds=([lo1, lo2, ...], [hi1, hi2, ...]), # optimiser bounds
        simulate=simulate,
    )

def simulate(close, high, low, volume, x):
    # 1. Compute indicators (regular python/numpy)
    ema = ema_np(close, int(x[0]))
    rsi = rsi_np(close, int(x[1]))
    # 2. Call @njit trading loop
    return _execute(close, 1_000_000.0, ema, rsi, int(x[2]), ...)

@njit(fastmath=True)
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
  vwap_np(high,low,close,vol), volume_oscillator_np(vol,fast,slow),
  volume_ratio_np(vol,period)
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
  mean_reversion_score_np(close,period), trend_strength_np(close,period)

## Critical rules
- ALWAYS use `from strategy_helpers import *` — never selective imports.
  This prevents "cannot import name" and "name not defined" errors.
  All 93 functions become available everywhere, including inside @njit.
- ALL variables used inside @njit _execute MUST be passed as parameters.
  Variables defined in simulate() are NOT visible inside _execute().
  WRONG: defining `rsi_period = int(x[2])` in simulate then using `rsi_period` in _execute
  RIGHT: pass it as a parameter: `_execute(close, cash, ema, rsi, int(x[2]), ...)`
- Always use max(int(x[i]), 1) for period parameters (0 → division by zero).
- Handle NaN: first period-1 values of any indicator are NaN.  Skip them.
- Keep indicator periods < 200 (training window is 365 days).
- All functions called inside @njit must be @njit (from strategy_helpers).
- NEVER use print() inside simulate() or _execute() — it runs 10000+ times.
- Always force-sell at the end of _execute.
- 4–12 decision variables is ideal.  More = harder to optimise, overfitting risk.

## What works
- Regime detection (ADX/trend_strength to switch trend vs mean-reversion)
- Confirmation signals (fast signal + slower filter)
- Volatility filters (don't trade during extremes or trade after squeezes)
- Trailing stops (ATR-based exits let winners run)
- Asymmetric timing (different buy/sell cooldowns)

## What fails
- Too many indicators (8+ usually overfits)
- Very short periods (<5 days) — captures noise
- Too many AND conditions — too few trades, coincidental fits

## Your output format
Each response MUST contain exactly one fenced python code block with the
COMPLETE strategy.py file.  Before it, explain your reasoning.  After it,
give a one-line DESCRIPTION: tag.

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

@njit(fastmath=True)
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
"""

# ═══════════════════════════════════════════════════════════════════════════
#  Data classes
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentResult:
    experiment_id: int
    commit: str
    score: float
    growth: float
    volatility: float
    status: str            # keep, discard, crash
    description: str
    strategy_name: str = ""
    beat_pct: float = 0.0
    worst_fold: float = 0.0
    best_fold: float = 0.0
    median_params: str = ""        # "ema_fast=18, ema_slow=52, ..."
    per_ticker: str = ""           # "BTC:1.23, ETH:0.87, ..."
    strategy_code: str = ""        # full strategy.py retained for prompt references
    family: str = ""               # coarse strategy family for diversity-aware prompts

@dataclass
class AgentState:
    best_score: float = -999.0
    best_commit: str = ""
    experiment_count: int = 0
    history: list = field(default_factory=list)   # list of ExperimentResult
    current_strategy: str = ""                     # content of strategy.py (best)
    last_discarded_code: str = ""                  # kept for backward compatibility
    best_per_ticker: str = ""                      # per-ticker breakdown of best

    def _format_experiment(self, r, label: str = "") -> str:
        """Format one experiment result for display."""
        folds_per_year = 252.0 / 90
        ann = (math.exp(r.growth * folds_per_year) - 1) * 100
        sign = "+" if ann >= 0 else ""
        prefix = f"{label}: " if label else ""
        family = r.family or infer_strategy_family(r.description, r.strategy_code)
        fam_str = f" fam={family}" if family and family != "misc" else ""
        line = (f"  {prefix}#{r.experiment_id} [{r.status}]{fam_str} score={r.score:.4f} "
                f"g={r.growth:.3f}/v={r.volatility:.3f} "
                f"ann={sign}{ann:.1f}% — {r.description}")
        if r.median_params:
            line += f"\n    params: {r.median_params}"
        if r.per_ticker:
            line += f"\n    per-ticker: {r.per_ticker}"
        return line

    def _non_crash_sorted(self) -> list:
        """Return all non-crash results sorted by score descending."""
        scored = [r for r in self.history if r.status != "crash"]
        return sorted(scored, key=lambda r: r.score, reverse=True)

    def _best_overall(self, limit: int, exclude_ids: Optional[set] = None) -> list:
        """Top scoring experiments regardless of family."""
        exclude_ids = exclude_ids or set()
        chosen = []
        for r in self._non_crash_sorted():
            if r.experiment_id in exclude_ids:
                continue
            chosen.append(r)
            if len(chosen) >= limit:
                break
        return chosen

    def _diverse_best(self, limit: int, exclude_ids: Optional[set] = None) -> list:
        """Best experiment per family, sorted by score."""
        exclude_ids = exclude_ids or set()
        chosen = []
        seen_families = set()
        for r in self._non_crash_sorted():
            if r.experiment_id in exclude_ids:
                continue
            family = r.family or infer_strategy_family(r.description, r.strategy_code)
            if family in seen_families:
                continue
            seen_families.add(family)
            chosen.append(r)
            if len(chosen) >= limit:
                break
        return chosen

    def _strong_discards(self, limit: int, exclude_ids: Optional[set] = None) -> list:
        """High-scoring discarded experiments that nearly made the cut."""
        exclude_ids = exclude_ids or set()
        discards = [r for r in self.history
                    if r.status == "discard" and r.experiment_id not in exclude_ids]
        discards.sort(key=lambda r: r.score, reverse=True)
        return discards[:limit]

    def _representative_failures(self, exclude_ids: Optional[set] = None) -> list:
        """Representative failures to show distinct ways an idea can fail."""
        exclude_ids = exclude_ids or set()
        chosen = []
        used_ids = set(exclude_ids)

        crashes = [r for r in self.history
                   if r.status == "crash" and r.experiment_id not in used_ids]
        if crashes:
            chosen.append(("recent crash", crashes[-1]))
            used_ids.add(crashes[-1].experiment_id)

        discards = [r for r in self.history
                    if r.status == "discard" and r.experiment_id not in used_ids]
        if discards:
            volatile = max(discards, key=lambda r: (r.growth, r.volatility))
            chosen.append(("high growth / high vol", volatile))
            used_ids.add(volatile.experiment_id)

        discards = [r for r in self.history
                    if r.status == "discard" and r.experiment_id not in used_ids]
        if discards:
            flat = min(discards, key=lambda r: (abs(r.growth), r.volatility))
            chosen.append(("flat / low edge", flat))
            used_ids.add(flat.experiment_id)

        discards = [r for r in self.history
                    if r.status == "discard" and r.experiment_id not in used_ids]
        if discards:
            worst = min(discards, key=lambda r: r.score)
            chosen.append(("worst score", worst))

        return chosen

    def _recent_results(self, limit: int, exclude_ids: Optional[set] = None) -> list:
        """Recent results not already shown elsewhere."""
        exclude_ids = exclude_ids or set()
        recent_all = self.history[-limit * 3:]  # grab extra to backfill after dedup
        recent = [r for r in recent_all if r.experiment_id not in exclude_ids]
        return recent[-limit:]

    def _reference_discard(self) -> Optional[ExperimentResult]:
        """
        Pick the most informative discarded strategy code for the prompt.

        Prefer a high-scoring discarded idea from a different family than the
        current best, so the LLM sees one credible alternative branch.
        """
        discards = [r for r in self.history if r.status == "discard" and r.strategy_code]
        if not discards:
            return None

        discards.sort(key=lambda r: r.score, reverse=True)
        best_family = infer_strategy_family("current best", self.current_strategy)
        for r in discards:
            family = r.family or infer_strategy_family(r.description, r.strategy_code)
            if family != best_family:
                return r
        return discards[0]

    def summary(self, top_k: int = 10, recent_k: int = 10) -> str:
        """
        Build a curated prompt summary.

        Instead of dumping only the raw top-K plus recent-K runs, this mixes:
        - best overall experiments (exploitation)
        - best per family (diversity)
        - strong discarded near-misses
        - representative failure modes
        - recent experiments

        This keeps the prompt useful on 50K-context machines while reducing the
        chance that the model overfits to one strategy family.
        """
        lines = [f"Experiments run: {self.experiment_count}",
                 f"Best SCORE so far: {self.best_score:.4f}"]
        if self.best_per_ticker:
            lines.append(f"Best per-ticker: {self.best_per_ticker}")

        if not self.history:
            return "\n".join(lines)

        shown_ids = set()

        best_overall_n = min(3, top_k)
        diverse_n = max(0, top_k - best_overall_n)
        near_miss_n = min(3, max(1, top_k // 3))

        best_overall = self._best_overall(best_overall_n)
        if best_overall:
            lines.append(f"\nBest overall experiments:")
            for r in best_overall:
                lines.append(self._format_experiment(r))
                shown_ids.add(r.experiment_id)

        diverse = self._diverse_best(diverse_n, exclude_ids=shown_ids)
        if diverse:
            lines.append(f"\nBest per strategy family:")
            for r in diverse:
                lines.append(self._format_experiment(r))
                shown_ids.add(r.experiment_id)

        near_misses = self._strong_discards(near_miss_n, exclude_ids=shown_ids)
        if near_misses:
            lines.append(f"\nStrong discarded near-misses:")
            for r in near_misses:
                lines.append(self._format_experiment(r))
                shown_ids.add(r.experiment_id)

        failures = self._representative_failures(exclude_ids=shown_ids)
        if failures:
            lines.append(f"\nRepresentative failure modes:")
            for label, r in failures:
                lines.append(self._format_experiment(r, label=label))
                shown_ids.add(r.experiment_id)

        recent = self._recent_results(recent_k, exclude_ids=shown_ids)
        if recent:
            lines.append(f"\nRecent experiments:")
            for r in recent:
                lines.append(self._format_experiment(r))

        return "\n".join(lines)

# ═══════════════════════════════════════════════════════════════════════════
#  LLM interaction
# ═══════════════════════════════════════════════════════════════════════════

def is_local_base_url(base_url: Optional[str]) -> bool:
    """Return True for local/default OpenAI-compatible endpoints."""
    if not base_url:
        return True
    base = base_url.lower()
    return "127.0.0.1" in base or "localhost" in base


def pick_llm_backend(model_name: Optional[str], base_url: Optional[str]) -> str:
    """Choose between OpenAI-compatible and native provider SDKs."""
    model = (model_name or "").lower()
    if "qwen" in model and is_local_base_url(base_url):
        return "openai"
    if "claude" in model and is_local_base_url(base_url):
        return "claude"
    if "gemini" in model and is_local_base_url(base_url):
        return "gemini"
    if "minimax" in model and is_local_base_url(base_url):
        return "minimax"
    return "openai"


def resolve_api_key(base_url: str) -> str:
    """Resolve API key for the selected endpoint."""
    if os.environ.get("OPENAI_API_KEY"):
        return os.environ["OPENAI_API_KEY"]
    if "anthropic.com" in base_url and os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    if "minimax.io" in base_url and os.environ.get("MINIMAX_API_KEY"):
        return os.environ["MINIMAX_API_KEY"]
    return "dummy"


def pick_model_id(client: OpenAI, requested_model: Optional[str] = None) -> str:
    """Resolve model id, preferring an explicit CLI override."""
    if requested_model:
        return requested_model
    try:
        models = client.models.list()
        return models.data[0].id if models.data else "default"
    except Exception as e:
        print(f"  [WARN] Could not list models ({e}); using provider default")
        return "default"


def flatten_messages_for_native(messages: list) -> tuple[str, str]:
    """Convert OpenAI-style messages into a system prompt and a transcript."""
    system_prompt = ""
    transcript = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        if role == "system" and not system_prompt:
            system_prompt = str(content)
            continue
        label = "Assistant" if role == "assistant" else "User"
        transcript.append(f"{label}:\n{content}")
    return system_prompt, "\n\n".join(transcript).strip()


def extract_anthropic_text(response) -> str:
    """Extract plain text from an Anthropic response."""
    blocks = getattr(response, "content", None) or []
    text_parts = []
    for block in blocks:
        block_type = getattr(block, "type", "")
        block_text = getattr(block, "text", "")
        if block_type == "text" and block_text:
            text_parts.append(block_text)
    if text_parts:
        return "\n".join(text_parts).strip()
    for block in reversed(blocks):
        block_text = getattr(block, "text", "")
        if block_text:
            return block_text.strip()
    return ""

def call_minimax_native(system_prompt: str, user_prompt: str, model_id: str,
                        temperature: float = TEMPERATURE) -> str:
    """Call MiniMax's API using the Anthropic SDK compatibility layer."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        print("  [LLM ERROR] MiniMax support via Anthropic requires the 'anthropic' package.")
        print("  Install with: pip install anthropic")
        return ""

    # Look for the MiniMax key specifically
    api_key = os.environ.get("MINIMAX_API_KEY")
    if not api_key:
        print("  [LLM ERROR] MINIMAX_API_KEY is not set.")
        return ""

    try:
        # The crucial step: Redirect the Claude SDK to MiniMax's servers
        client = anthropic.Anthropic(
            api_key=api_key,
            base_url="https://api.minimax.io/anthropic" 
        )
        
        request = dict(
            model=model_id, # e.g., "MiniMax-M2.7"
            max_tokens=MAX_LLM_OUTPUT_TOKENS, 
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        
        try:
            # Try passing Claude's thinking block. MiniMax's compatibility layer 
            # might gracefully accept it, or it might reject it.
            response = client.messages.create(
                **request,
                output_config={"effort": "high"},
            )
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                total = u.input_tokens + u.output_tokens
                print(f"  [USAGE] Total: {total} | Prompt: {u.input_tokens} | Output: {u.output_tokens} (Thinking included in Output)")
        except Exception as e:
            # Safe Fallback: If MiniMax's endpoint rejects the explicit 'thinking' kwargs,
            # strip them out. M2.7 is naturally agentic and will "think" regardless!
            error_str = str(e).lower()
            if any(token in error_str for token in ("thinking", "adaptive", "effort", "unrecognized")):
                response = client.messages.create(**request)
            else:
                raise
                
        # Because MiniMax mimics Claude, your existing extractor will work perfectly
        return extract_anthropic_text(response)
        
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return ""
    

def call_claude_native(system_prompt: str, user_prompt: str, model_id: str,
                       temperature: float = TEMPERATURE) -> str:
    """Call Anthropic's native SDK with lazy optional import."""
    try:
        import anthropic  # type: ignore
    except ImportError:
        print("  [LLM ERROR] Claude support requires the optional 'anthropic' package.")
        print("  Install with: pip install anthropic")
        return ""

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("  [LLM ERROR] ANTHROPIC_API_KEY is not set.")
        return ""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        request = dict(
            model=model_id,
            max_tokens=MAX_LLM_OUTPUT_TOKENS,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
            temperature=temperature,
        )
        try:
            response = client.messages.create(
                **request,
                output_config={"effort": "high"},
            )
            if hasattr(response, "usage") and response.usage:
                u = response.usage
                total = u.input_tokens + u.output_tokens
                print(f"  [USAGE] Total: {total} | Prompt: {u.input_tokens} | Output: {u.output_tokens} (Thinking included in Output)")
        except Exception as e:
            # Some Claude models/endpoints may not accept thinking controls.
            if any(token in str(e).lower() for token in ("thinking", "adaptive", "effort")):
                response = client.messages.create(**request)
            else:
                raise
        return extract_anthropic_text(response)
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return ""


def call_gemini_native(system_prompt: str, user_prompt: str, model_id: str,
                       temperature: float = TEMPERATURE) -> str:
    """Call Google's native Gemini SDK with lazy optional import."""
    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore
    except ImportError:
        print("  [LLM ERROR] Gemini support requires the optional 'google-genai' package.")
        print("  Install with: pip install google-genai")
        return ""

    api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        print("  [LLM ERROR] GEMINI_API_KEY or GOOGLE_API_KEY is not set.")
        return ""

    try:
        client = genai.Client(api_key=api_key)
        config_kwargs = dict(
            system_instruction=system_prompt,
            temperature=temperature,
        )
        try:
            config = types.GenerateContentConfig(
                **config_kwargs,
                thinking_config=types.ThinkingConfig(thinking_level="HIGH"),
            )
            response = client.models.generate_content(
                model=model_id,
                contents=user_prompt,
                config=config,
            )                 
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                u = response.usage_metadata
                thinking = getattr(u, 'thoughts_token_count', 0)
                print(f"  [USAGE] Total: {u.total_token_count} | Prompt: {u.prompt_token_count} | Output: {u.candidates_token_count} (Thinking: {thinking})")
                
        except Exception as e:
            # Fallback if the installed SDK/model does not expose thinking controls.
            if any(token in str(e).lower() for token in ("thinking", "thinking_level")):
                config = types.GenerateContentConfig(**config_kwargs)
                response = client.models.generate_content(
                    model=model_id,
                    contents=user_prompt,
                    config=config,
                )
            else:
                raise
        return (getattr(response, "text", "") or "").strip()
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return ""


def build_user_message(state: AgentState, top_k: int = TOP_K,
                       recent_k: int = RECENT_K, extra: str = "") -> str:
    """Build the user message for the next iteration."""
    parts = []

    if state.experiment_count == 0:
        parts.append("This is the FIRST experiment.  Run the baseline EMA/SMA "
                      "crossover as-is to establish the starting score.  "
                      "Output the current strategy.py unchanged.")
    else:
        parts.append(state.summary(top_k=top_k, recent_k=recent_k))
        parts.append(f"\nCurrent best strategy.py:\n```python\n{state.current_strategy}\n```")

        # Show the most informative discarded strategy, not just the latest one.
        ref_discard = state._reference_discard()
        if ref_discard and ref_discard.strategy_code:
            discard_display = ref_discard.strategy_code
            if len(discard_display) > REFERENCE_CODE_CHARS:
                discard_display = discard_display[:REFERENCE_CODE_CHARS] + "\n# ... (truncated)"
            parts.append(
                f"\nReference discarded strategy.py "
                f"(strong result but not the current best; family={ref_discard.family or infer_strategy_family(ref_discard.description, ref_discard.strategy_code)}):"
                f"\n```python\n{discard_display}\n```"
            )

        parts.append("\nPropose the next strategy.  Think about:\n"
                      "- Which strategy families keep winning, and which families are under-explored\n"
                      "- What the median params tell you (hitting bounds? converging?)\n"
                      "- What the strong near-misses got right, and what likely kept them below the best\n"
                      "- What the representative failures reveal about overfitting, flatness, or volatility\n"
                      "- Which tickers are weak in per-ticker breakdown\n"
                      "- How to improve the current best without making it more fragile\n"
                      "Output the COMPLETE strategy.py in a python code block.")

    if extra:
        parts.append(f"\n{extra}")

    return "\n\n".join(parts)


def build_crash_message(error_text: str, attempt: int) -> str:
    """Build a message asking the LLM to fix a crash."""
    return (f"CRASH (attempt {attempt})!  The strategy failed with this error:\n\n"
            f"```\n{error_text[-2000:]}\n```\n\n"
            f"Fix the bug and output the corrected COMPLETE strategy.py.")


def extract_strategy_code(response_text: str) -> Optional[str]:
    """Extract the python code block from the LLM response."""
    # Look for ```python ... ``` blocks
    pattern = r'```python\s*\n(.*?)```'
    matches = re.findall(pattern, response_text, re.DOTALL)
    if not matches:
        # try without language tag
        pattern = r'```\s*\n(.*?)```'
        matches = re.findall(pattern, response_text, re.DOTALL)
    if not matches:
        return None
    # Take the longest match (most likely the full file)
    code = max(matches, key=len).strip()
    # Basic validation: must contain get_strategy
    if "get_strategy" not in code:
        return None
    return code


def extract_description(response_text: str) -> str:
    """Extract the DESCRIPTION line from the LLM response."""
    for line in response_text.split("\n"):
        line = line.strip()
        if line.upper().startswith("DESCRIPTION:"):
            return line[len("DESCRIPTION:"):].strip()
    # fallback: first non-empty line that isn't code
    for line in response_text.split("\n"):
        line = line.strip()
        if line and not line.startswith("```") and not line.startswith("import"):
            return line[:100]
    return "no description"


def call_llm(messages: list, args, client: Optional[OpenAI] = None,
             model_id: Optional[str] = None) -> str:
    """Call the configured LLM backend and return the response text."""
    backend = pick_llm_backend(model_id or args.model, args.base_url)
    temperature = getattr(args, "temperature", TEMPERATURE)

    if backend == "claude":
        system_prompt, user_prompt = flatten_messages_for_native(messages)
        return call_claude_native(system_prompt, user_prompt,
                                  model_id or args.model or "claude-sonnet-4-6",
                                  temperature)

    if backend == "minimax":
        system_prompt, user_prompt = flatten_messages_for_native(messages)
        return call_minimax_native(system_prompt, user_prompt,
                                  model_id or args.model or "MiniMax-M2.7",
                                  temperature)

    if backend == "gemini":
        system_prompt, user_prompt = flatten_messages_for_native(messages)
        return call_gemini_native(system_prompt, user_prompt,
                                  model_id or args.model or "gemini-3.1-flash-preview",
                                  temperature)

    if client is None:
        print("  [LLM ERROR] OpenAI-compatible backend selected without a client.")
        return ""

    try:
        response = client.chat.completions.create(
            model=model_id or "default",
            messages=messages,
            temperature=temperature,
            max_tokens=MAX_LLM_OUTPUT_TOKENS,
        )
        if hasattr(response, "usage") and response.usage:
            u = response.usage
            details = getattr(u, "completion_tokens_details", None)
            thinking = getattr(details, "reasoning_tokens", 0) if details else 0
            print(f"  [USAGE] Total: {u.total_tokens} | Prompt: {u.prompt_tokens} | Output: {u.completion_tokens} (Thinking: {thinking})")

        return response.choices[0].message.content or ""
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return ""

# ═══════════════════════════════════════════════════════════════════════════
#  Strategy validation and execution
# ═══════════════════════════════════════════════════════════════════════════

def validate_syntax(code: str) -> Optional[str]:
    """Check Python syntax.  Returns error string or None if ok."""
    try:
        ast.parse(code)
        return None
    except SyntaxError as e:
        return f"SyntaxError: {e}"


def validate_contract(code: str) -> Optional[str]:
    """Check that the strategy defines get_strategy with correct structure."""
    if "def get_strategy" not in code:
        return "Missing get_strategy() function"
    if "def simulate" not in code:
        return "Missing simulate() function"
    if "bounds" not in code:
        return "Missing bounds in get_strategy"
    if "variables" not in code:
        return "Missing variables in get_strategy"
    return None


def extract_strategy_meta(code: str) -> dict:
    """
    Parse variable names and bounds from strategy code.
    Returns {'variables': [...], 'lo': [...], 'hi': [...]} or empty dict.
    """
    meta = {}
    # Extract variables list
    var_match = re.search(r'variables\s*=\s*\[([^\]]+)\]', code)
    if var_match:
        raw = var_match.group(1)
        meta['variables'] = [s.strip().strip('"').strip("'")
                             for s in raw.split(',')]

    # Extract bounds — look for bounds=([...], [...])
    bounds_match = re.search(
        r'bounds\s*=\s*\(\s*\[([^\]]+)\]\s*,\s*\[([^\]]+)\]\s*\)', code)
    if bounds_match:
        try:
            meta['lo'] = [float(x.strip()) for x in bounds_match.group(1).split(',')]
            meta['hi'] = [float(x.strip()) for x in bounds_match.group(2).split(',')]
        except ValueError:
            pass
    return meta


def format_strategy_meta(meta: dict) -> str:
    """Format variables with bounds as a compact display string."""
    if not meta or 'variables' not in meta:
        return ""
    parts = []
    variables = meta['variables']
    lo = meta.get('lo', [])
    hi = meta.get('hi', [])
    for i, v in enumerate(variables):
        if i < len(lo) and i < len(hi):
            parts.append(f"{v}[{lo[i]:g}..{hi[i]:g}]")
        else:
            parts.append(v)
    return ", ".join(parts)


def fix_strategy_code(code: str) -> str:
    """Auto-fix common LLM mistakes in strategy code before writing to disk."""
    lines = code.split('\n')
    new_lines = []
    has_wildcard_import = False
    has_any_helpers_import = False

    for line in lines:
        stripped = line.strip()
        # Replace any selective strategy_helpers import with wildcard
        if stripped.startswith('from strategy_helpers import'):
            if stripped == 'from strategy_helpers import *':
                has_wildcard_import = True
                new_lines.append(line)
            else:
                # Replace selective import with wildcard
                if not has_wildcard_import:
                    new_lines.append('from strategy_helpers import *')
                    has_wildcard_import = True
                # else: skip duplicate
            has_any_helpers_import = True
        # Remove print() calls (LLMs love adding debug prints)
        elif stripped.startswith('print(') and '@njit' not in stripped:
            continue  # drop the line
        else:
            new_lines.append(line)

    # If no strategy_helpers import at all, add one after numpy import
    if not has_any_helpers_import:
        final_lines = []
        inserted = False
        for line in new_lines:
            final_lines.append(line)
            if not inserted and line.strip().startswith('import numpy'):
                final_lines.append('from strategy_helpers import *')
                inserted = True
        if not inserted:
            # Prepend at the top after any comments
            final_lines.insert(0, 'from strategy_helpers import *')
        new_lines = final_lines

    return '\n'.join(new_lines)


def infer_strategy_family(description: str, code: str = "") -> str:
    """Infer a coarse strategy family label from free text and code."""
    text = f"{description}\n{code[:4000]}".lower()

    tags = []

    def has_any(patterns):
        return any(p in text for p in patterns)

    if (has_any(["ema_np("]) and has_any(["sma_np("])) or has_any(["ema/sma", "ema_sma"]):
        tags.append("ema-sma")
    elif has_any(["hma_np("]) and has_any(["sma_np("]):
        tags.append("hma-sma")
    elif has_any(["supertrend_np(", "supertrend"]):
        tags.append("supertrend")
    elif has_any(["bollinger_np(", "bollinger_bandwidth_np(", "bollinger band"]):
        tags.append("bollinger")
    elif has_any(["donchian_np(", "donchian"]):
        tags.append("donchian")

    if has_any(["rsi_np(", "rsi "]):
        tags.append("rsi")
    if has_any(["adx_np(", " adx", "regime"]):
        tags.append("adx")
    if has_any(["atr_np(", "trail_mult", "trailing_stop_hit(", "trailing stop"]):
        tags.append("atr-stop")
    if has_any(["tier1_", "tier2_", "tier3_", "tiered"]):
        tags.append("tiered-stop")
    if has_any(["stop_loss", "stoploss", "hard stop", "hard-stop"]):
        tags.append("hard-stop")
    if has_any(["lockin", "lock-in", "profit lock", "lock-in floor"]):
        tags.append("lockin")
    if has_any(["pullback"]):
        tags.append("pullback")
    if has_any(["volume", "obv_np(", "cmf_np(", "vwap_np("]):
        tags.append("volume")
    if has_any(["roc_np(", "roc "]):
        tags.append("roc")
    if has_any(["cci_np(", "cci "]):
        tags.append("cci")
    if has_any(["trend filter", "filter_sma", "filter_period", "price-above-sma", "above long sma"]):
        tags.append("trend-filter")

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)

    if not unique:
        return "misc"
    return "+".join(unique[:4])


def write_strategy(code: str):
    """Write strategy.py in the project directory."""
    (PROJECT_DIR / STRATEGY_FILE).write_text(code)


def read_strategy() -> str:
    """Read current strategy.py from the project directory."""
    p = PROJECT_DIR / STRATEGY_FILE
    return p.read_text() if p.exists() else ""


class GitError(RuntimeError):
    """Raised when git bookkeeping cannot be completed safely."""


def _run_git(*args) -> subprocess.CompletedProcess:
    """Run a git command in the project directory."""
    return subprocess.run(["git"] + list(args),
                          capture_output=True, text=True, cwd=PROJECT_DIR)


def _git_cmd_str(*args) -> str:
    """Format a git command for diagnostics."""
    return "git " + " ".join(args)


def _format_git_error(action: str, args: tuple, result: subprocess.CompletedProcess) -> str:
    """Build a helpful git error message with common remediation hints."""
    stderr = (result.stderr or "").strip()
    stdout = (result.stdout or "").strip()
    detail = stderr or stdout or "unknown git error"
    msg = f"{action} failed: {_git_cmd_str(*args)}\n{detail}"

    lower = detail.lower()
    if ("please tell me who you are" in lower or
            "unable to auto-detect email address" in lower or
            "author identity unknown" in lower):
        msg += ("\nConfigure git identity in this repo, e.g.:\n"
                '  git config user.name "Autoresearch Bot"\n'
                '  git config user.email "autoresearch@local"')
    return msg


def _run_git_checked(*args, action: Optional[str] = None) -> subprocess.CompletedProcess:
    """Run a git command and raise a clear error if it fails."""
    result = _run_git(*args)
    if result.returncode != 0:
        raise GitError(_format_git_error(action or "Git command", args, result))
    return result


def git_has_head() -> bool:
    """Return True if the current repo has an initial commit."""
    result = _run_git("rev-parse", "--verify", "HEAD")
    return result.returncode == 0


def git_head_commit() -> str:
    """Return short HEAD commit hash or raise if HEAD is missing."""
    result = _run_git_checked("rev-parse", "--short", "HEAD",
                              action="Resolve current HEAD")
    commit = result.stdout.strip()
    if not commit:
        raise GitError("Resolve current HEAD failed: git returned an empty commit hash")
    return commit


def git_ensure_repo():
    """Initialise a git repo in the project directory if one doesn't exist."""
    git_dir = PROJECT_DIR / ".git"
    if git_dir.is_dir():
        if not git_has_head():
            raise GitError(
                f"Git repo exists in {PROJECT_DIR} but has no initial commit.\n"
                "Create one before running the agent, e.g.:\n"
                "  git add .\n"
                '  git commit -m "initial commit"'
            )
        return
    print(f"  Initialising git repo in {PROJECT_DIR} ...")
    _run_git_checked("init", action="Initialize git repo")
    _run_git_checked("add", ".", action="Stage initial project files")
    _run_git_checked("commit", "-m", "initial commit",
                     action="Create initial git commit")
    if not git_has_head():
        raise GitError("Git initialization completed but HEAD is still missing")


def git_commit(message: str) -> str:
    """Stage strategy.py, commit, return short hash."""
    _run_git_checked("add", STRATEGY_FILE, action="Stage strategy.py")

    diff_result = _run_git("diff", "--cached", "--quiet", "--", STRATEGY_FILE)
    if diff_result.returncode == 0:
        return git_head_commit()
    if diff_result.returncode != 1:
        raise GitError(_format_git_error(
            "Check staged strategy diff",
            ("diff", "--cached", "--quiet", "--", STRATEGY_FILE),
            diff_result,
        ))

    _run_git_checked("commit", "-m", message, action="Create strategy commit")
    return git_head_commit()


def git_revert():
    """Undo the last strategy commit.
    
    Uses --soft to undo the commit without touching any working tree files.
    The caller is responsible for restoring strategy.py via write_strategy().
    This ensures trading.py and other files are never overwritten by git.
    """
    _run_git_checked("reset", "--soft", "HEAD~1",
                     action="Revert last strategy commit")


def git_setup_branch(tag: str):
    """Create the experiment branch if it doesn't exist."""
    result = _run_git("branch", "--list", f"autoresearch/{tag}")
    if result.stdout.strip():
        print(f"  Branch autoresearch/{tag} exists, checking out...")
        _run_git_checked("checkout", f"autoresearch/{tag}",
                         action=f"Checkout branch autoresearch/{tag}")
    else:
        print(f"  Creating branch autoresearch/{tag}...")
        _run_git_checked("checkout", "-b", f"autoresearch/{tag}",
                         action=f"Create branch autoresearch/{tag}")


def preflight_check() -> Optional[str]:
    """
    Quick sanity check: import the strategy, call simulate() once with
    realistic dummy data.  Catches numba TypingErrors, import errors,
    undefined variables — in seconds instead of waiting for walk-forward.

    Returns None if OK, or error string if failed.
    """
    check_script = f'''
import sys, os, importlib, traceback
import numpy as np
os.chdir({str(PROJECT_DIR)!r})
sys.path.insert(0, {str(PROJECT_DIR)!r})

# Force reimport (agent may have written a new strategy.py)
for mod_name in list(sys.modules):
    if mod_name in ("strategy",) or mod_name.startswith("strategy."):
        del sys.modules[mod_name]

try:
    import strategy
    spec = strategy.get_strategy()
    lo, hi = spec["bounds"]
    n = max(max(hi), 200) + 100
    n = max(n, 400)
    np.random.seed(42)
    close = np.cumsum(np.random.randn(n) * 0.02) + 100
    close = np.abs(close) + 10
    high = close + np.abs(np.random.randn(n)) * 2
    low = close - np.abs(np.random.randn(n)) * 2
    volume = np.random.rand(n) * 1e6 + 1
    x_lo = np.array([float(l) for l in lo])
    x_hi = np.array([float(h) for h in hi])
    x_mid = (x_lo + x_hi) / 2.0
    # Run with mid, lo, and hi to compile all code paths
    spec["simulate"](close, high, low, volume, x_mid)
    spec["simulate"](close, high, low, volume, x_lo)
    spec["simulate"](close, high, low, volume, x_hi)
    print("PREFLIGHT_OK")
except Exception:
    traceback.print_exc()
    print("PREFLIGHT_FAIL")
'''
    result = subprocess.run(
        [sys.executable, "-c", check_script],
        capture_output=True, text=True, timeout=120, cwd=PROJECT_DIR)
    output = result.stdout + "\n" + result.stderr

    if "PREFLIGHT_OK" in output:
        return None

    # Extract the traceback
    tb_match = re.search(r'(Traceback \(most recent call last\):.*?)PREFLIGHT_FAIL',
                         output, re.DOTALL)
    if tb_match:
        return tb_match.group(1).strip()
    return output.strip()[-2000:]


def run_experiment(extra_args: str = "") -> dict:
    """
    Run the walk-forward experiment in the project directory.
    Returns dict with: success, score, growth, vol, beat_pct, worst, best, error
    """
    trading_py = PROJECT_DIR / "trading.py"
    cmd = (f"{sys.executable} {trading_py} "
           f"--mode walkforward --strategy strategy {extra_args}")
    print(f"  Running: {cmd}")
    t0 = time.time()

    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True,
            timeout=RUN_TIMEOUT, cwd=PROJECT_DIR)
        elapsed = time.time() - t0
        print(f"  Completed in {elapsed:.0f}s")

        # Write combined output for debugging
        output = result.stdout + "\n" + result.stderr
        (PROJECT_DIR / RUN_LOG).write_text(output)

    except subprocess.TimeoutExpired:
        (PROJECT_DIR / RUN_LOG).write_text(f"TIMEOUT: run exceeded {RUN_TIMEOUT}s")
        return dict(success=False, error="Timeout exceeded")

    # Parse results from output
    return parse_results(output)


def parse_results(output: str) -> dict:
    """Parse SCORE, growth, vol, and per-fold parameters from walk-forward output."""
    result = dict(success=False, score=0.0, growth=0.0, vol=0.0,
                  beat_pct=0.0, worst=0.0, best=0.0, error="",
                  fold_xs=[])

    # Look for the SCORE line
    score_match = re.search(
        r'SCORE\s*=\s*([-\d.]+)\s*\(growth=([-\d.]+),\s*vol=([-\d.]+)', output)
    if not score_match:
        # Try to find actual Python traceback
        tb_match = re.search(r'(Traceback \(most recent call last\):.*)', output, re.DOTALL)
        if tb_match:
            result["error"] = tb_match.group(1).strip()[-3000:]
        else:
            lines = output.strip().split("\n")
            result["error"] = "\n".join(lines[-80:])
        return result

    result["success"] = True
    result["score"] = float(score_match.group(1))
    result["growth"] = float(score_match.group(2))
    result["vol"] = float(score_match.group(3))

    # Parse additional stats
    beat_match = re.search(r'profitable in (\d+)% of folds', output)
    if beat_match:
        result["beat_pct"] = float(beat_match.group(1))

    worst_match = re.search(r'worst=([\d.]+)', output)
    if worst_match:
        result["worst"] = float(worst_match.group(1))

    best_match = re.search(r'best=([\d.]+)', output)
    if best_match:
        result["best"] = float(best_match.group(1))

    # Extract per-fold optimal x values: "x=[26, 57, 24, 98]"
    fold_xs = []
    for m in re.finditer(r'x=\[([\d,\s.]+)\]', output):
        try:
            vals = [float(v.strip()) for v in m.group(1).split(',')]
            fold_xs.append(vals)
        except ValueError:
            pass
    result["fold_xs"] = fold_xs

    # Extract per-ticker OOS geo_means: "BTC-USD: OOS factors ... geo_mean = 0.925"
    per_ticker = {}
    for m in re.finditer(r'(\S+):\s*OOS factors.*?geo_mean\s*=\s*([\d.]+)', output):
        per_ticker[m.group(1)] = float(m.group(2))
    result["per_ticker"] = per_ticker

    return result


def format_optimal_params(meta: dict, fold_xs: list) -> str:
    """Format median optimal parameters across folds."""
    if not fold_xs or not meta or 'variables' not in meta:
        return ""
    import numpy as np
    variables = meta['variables']
    n_vars = len(variables)
    # Filter to matching-length x vectors
    valid = [x for x in fold_xs if len(x) == n_vars]
    if not valid:
        return ""
    arr = np.array(valid)
    medians = np.median(arr, axis=0)
    parts = [f"{variables[i]}={int(medians[i])}" for i in range(n_vars)]
    return ", ".join(parts)


def format_per_ticker(per_ticker: dict, test_days: int = 90) -> str:
    """Format per-ticker geo_means with annualized returns.
    
    Example: 'AAPL:1.002(+0.6%/yr), MSFT:1.016(+4.5%/yr)'
    """
    if not per_ticker:
        return ""
    folds_per_year = 252.0 / test_days  # ~2.8 for 90-day folds
    parts = []
    for t, v in per_ticker.items():
        ticker = t.replace('-USD', '').replace('-', '')
        ann_return = (v ** folds_per_year - 1) * 100
        sign = "+" if ann_return >= 0 else ""
        parts.append(f"{ticker}:{v:.3f}({sign}{ann_return:.1f}%/yr)")
    return ", ".join(parts)


def init_results_tsv():
    """Create results.tsv with header if it doesn't exist."""
    p = PROJECT_DIR / RESULTS_FILE
    if not p.exists():
        p.write_text("commit\tscore\tstatus\tgrowth_vol\tdescription\n")


def log_result(r: ExperimentResult):
    """Append a result to results.tsv."""
    gv = f"g={r.growth:.3f}/v={r.volatility:.3f}" if r.status != "crash" else "n/a"
    line = f"{r.commit}\t{r.score:.4f}\t{r.status}\t{gv}\t{r.description}\n"
    with open(PROJECT_DIR / RESULTS_FILE, "a") as f:
        f.write(line)

# ═══════════════════════════════════════════════════════════════════════════
#  Conversation management (sliding window for context budget)
# ═══════════════════════════════════════════════════════════════════════════

class Conversation:
    """Manages the message list for the LLM, keeping it within context limits.
    
    Past exchanges store only lightweight summaries (not the full user message
    which contains history+code).  The current turn's user message has the
    full context — past exchanges just provide continuity.
    """

    def __init__(self, system_prompt: str):
        self.system = {"role": "system", "content": system_prompt}
        self.exchanges: list = []  # list of (user_summary, assistant_msg) pairs

    def messages(self, user_msg: str) -> list:
        """Build the messages list for the next API call."""
        msgs = [self.system]
        # Keep only recent exchanges (lightweight)
        recent = self.exchanges[-MAX_CONTEXT_EXCHANGES:]
        for u, a in recent:
            msgs.append({"role": "user", "content": u})
            msgs.append({"role": "assistant", "content": a})
        msgs.append({"role": "user", "content": user_msg})
        return msgs

    def add_exchange(self, user_summary: str, assistant_msg: str):
        """Record a completed exchange with lightweight summaries.
        
        user_summary should be a brief description (not the full user message).
        The full context is regenerated fresh each turn by build_user_message().
        """
        # Cap assistant message to prevent bloat
        if len(assistant_msg) > 2000:
            assistant_msg = assistant_msg[:2000] + "\n...[truncated]..."
        self.exchanges.append((user_summary, assistant_msg))

# ═══════════════════════════════════════════════════════════════════════════
#  Main agent loop
# ═══════════════════════════════════════════════════════════════════════════

def run_agent(args):
    """The main autonomous experiment loop."""

    # --- Validate project directory ---
    print(f"Project directory: {PROJECT_DIR}")
    required_files = ["trading.py", "base_strategy.py", "strategy_helpers.py"]
    missing = [f for f in required_files if not (PROJECT_DIR / f).exists()]
    if missing:
        print(f"  [ERROR] Missing files in {PROJECT_DIR}: {missing}")
        print(f"  Place all project files in the same directory as agent.py.")
        sys.exit(1)

    # --- Connect to LLM ---
    base_url = args.base_url
    backend = pick_llm_backend(args.model, base_url)
    temperature = getattr(args, 'temperature', TEMPERATURE)
    client: Optional[OpenAI] = None
    model_id = args.model

    if backend == "openai":
        api_key = resolve_api_key(base_url)
        client = OpenAI(base_url=base_url, api_key=api_key)
        model_id = pick_model_id(client, args.model)
        print(f"Connected to LLM: {model_id} at {base_url} (temp={temperature})")
    else:
        model_id = args.model or ("claude-sonnet-4-6" if backend == "claude"
                                  else "gemini-3.1-pro")
        print(f"Connected to native {backend} client: {model_id} (temp={temperature})")

    # --- Always start from a clean state ---
    # Copy base_strategy.py → strategy.py so we have a known-good baseline,
    # regardless of what a previous (possibly killed) agent run left behind.
    import shutil
    base_path = PROJECT_DIR / BASE_STRATEGY_FILE
    strat_path = PROJECT_DIR / STRATEGY_FILE
    shutil.copy2(base_path, strat_path)
    print(f"  Copied {BASE_STRATEGY_FILE} → {STRATEGY_FILE}")

    # Preflight check the baseline
    print(f"  Preflight check on baseline...")
    pf_err = preflight_check()
    if pf_err is not None:
        print(f"  [ERROR] base_strategy.py fails preflight:")
        for line in pf_err.strip().split('\n')[-15:]:
            print(f"    {line}")
        print(f"\n  Fix {BASE_STRATEGY_FILE} and re-run.")
        sys.exit(1)
    print(f"  Preflight OK")

    # --- Setup git ---
    git_ensure_repo()
    if args.tag:
        git_setup_branch(args.tag)
    init_results_tsv()

    state = AgentState()
    state.current_strategy = read_strategy()
    conv = Conversation(SYSTEM_PROMPT)

    # Build extra args for trading.py
    extra_args_parts = []
    if args.quick:
        extra_args_parts.append("--num-retries 8 --max-evals 250")
    elif args.medium:
        extra_args_parts.append("--num-retries 16 --max-evals 500")
    if args.tickers:
        extra_args_parts.append(f"--tickers {' '.join(args.tickers)}")
    if args.start:
        extra_args_parts.append(f"--start {args.start}")
    if args.end:
        extra_args_parts.append(f"--end {args.end}")
    extra_args = " ".join(extra_args_parts)

    print(f"Extra args: {extra_args or '(default)'}")
    print("=" * 60)
    print("Starting autonomous experiment loop.  Ctrl+C to stop.")
    print("=" * 60)

    # --- Experiment loop ---
    is_baseline = True

    while True:
        state.experiment_count += 1
        exp_id = state.experiment_count
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT #{exp_id}  |  best={state.best_score:.4f}  |  "
              f"{datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        if is_baseline:
            # --- Baseline: run the known-good strategy (already preflighted) ---
            is_baseline = False
            this_is_baseline = True
            code = read_strategy()
            meta = extract_strategy_meta(code)
            description = "baseline"
            user_msg = "(baseline run)"
            meta_str = format_strategy_meta(meta)
            print(f"  Running baseline strategy...")
            if meta_str:
                print(f"  Variables: {meta_str}")
            commit = git_commit("baseline")

        else:
            this_is_baseline = False
            # --- Step 1: Ask LLM for next strategy ---
            user_msg = build_user_message(state, top_k=args.top_k, recent_k=args.recent_k)
            messages = conv.messages(user_msg)

            print("  Asking LLM for next strategy...")
            response_text = call_llm(messages, args, client=client, model_id=model_id)

            if not response_text:
                print("  [WARN] Empty LLM response, retrying in 5s...")
                time.sleep(5)
                state.experiment_count -= 1
                continue

            # --- Step 2: Extract and validate strategy code ---
            code = extract_strategy_code(response_text)
            description = extract_description(response_text)

            if code is None:
                print("  [WARN] No code block in response, nudging LLM...")
                conv.add_exchange(
                    "Propose the next strategy.",
                    "(no code block found — output the COMPLETE strategy.py "
                    "inside a ```python ... ``` block)")
                state.experiment_count -= 1
                continue

            # Auto-fix common mistakes (selective imports → wildcard, strip prints)
            code = fix_strategy_code(code)

            # Syntax check
            syntax_err = validate_syntax(code)
            if syntax_err:
                print(f"  [SYNTAX ERROR] {syntax_err}")
                fix_msg = (f"Syntax error in your strategy.py:\n{syntax_err}\n\n"
                           f"Fix it and output the complete corrected file.")
                fix_messages = conv.messages(fix_msg)
                fix_response = call_llm(fix_messages, args, client=client, model_id=model_id)
                code = extract_strategy_code(fix_response)
                code = fix_strategy_code(code) if code else code
                if code is None or validate_syntax(code):
                    print("  [SKIP] Could not fix syntax, skipping...")
                    state.experiment_count -= 1
                    continue
                description += " (syntax fix)"

            # Contract check
            contract_err = validate_contract(code)
            if contract_err:
                print(f"  [CONTRACT ERROR] {contract_err}, skipping...")
                state.experiment_count -= 1
                continue

            print(f"  Strategy: {description}")

            # Show strategy variables and bounds
            meta = extract_strategy_meta(code)
            meta_str = format_strategy_meta(meta)
            if meta_str:
                print(f"  Variables: {meta_str}")

            # Write strategy and preflight-check (catches numba errors in seconds)
            write_strategy(code)
            preflight_attempts = 0
            while preflight_attempts < MAX_CRASH_RETRIES:
                print(f"  Preflight check...")
                pf_err = preflight_check()
                if pf_err is None:
                    print(f"  Preflight OK")
                    break
                preflight_attempts += 1
                # Show the error
                pf_lines = pf_err.strip().split('\n')
                print(f"  [PREFLIGHT FAIL {preflight_attempts}/{MAX_CRASH_RETRIES}]")
                for line in pf_lines[-15:]:
                    print(f"    {line}")

                if preflight_attempts >= MAX_CRASH_RETRIES:
                    break
                # Ask LLM to fix
                fix_msg = build_crash_message(pf_err, preflight_attempts)
                fix_messages = conv.messages(fix_msg)
                fix_response = call_llm(fix_messages, args, client=client, model_id=model_id)
                fix_code = extract_strategy_code(fix_response)
                if fix_code:
                    fix_code = fix_strategy_code(fix_code)
                if fix_code and not validate_syntax(fix_code) and not validate_contract(fix_code):
                    code = fix_code
                    write_strategy(code)
                    description += f" (preflight fix {preflight_attempts})"
                else:
                    break

            if pf_err is not None:
                # Preflight failed after all retries — skip this experiment
                print(f"  Preflight failed — skipping experiment")
                # Restore previous strategy.py
                if state.current_strategy:
                    write_strategy(state.current_strategy)
                result = ExperimentResult(
                    experiment_id=exp_id, commit=git_head_commit(),
                    score=0.0, growth=0.0, volatility=0.0,
                    status="crash", description=f"{description} (preflight fail)",
                    strategy_code=code,
                    family=infer_strategy_family(description, code))
                log_result(result)
                state.history.append(result)
                conv.add_exchange(
                    f"Tried: {description}",
                    f"(experiment #{exp_id} preflight crash: {pf_err[-500:]})")
                continue

            commit = git_commit(description)
        print(f"  Committed: {commit}")

        # Run with crash retry loop
        run_result = None
        crash_count = 0
        while crash_count < MAX_CRASH_RETRIES:
            run_result = run_experiment(extra_args)

            if run_result["success"]:
                break

            crash_count += 1
            # Show useful error info
            err = run_result['error']
            err_lines = err.strip().split('\n')
            # Show last 20 lines of error (usually the traceback tail)
            err_display = '\n'.join(err_lines[-20:])
            print(f"  [CRASH {crash_count}/{MAX_CRASH_RETRIES}]")
            print(f"  {err_display}")
            print(f"  Full output: {PROJECT_DIR / RUN_LOG}")

            if crash_count >= MAX_CRASH_RETRIES:
                break

            # For baseline runs, don't ask LLM to fix — just retry once
            # (crash may be transient, e.g. numba cache issue)
            if this_is_baseline:
                continue

            # Ask LLM to fix the crash
            crash_msg = build_crash_message(run_result["error"], crash_count)
            crash_messages = conv.messages(crash_msg)
            fix_response = call_llm(crash_messages, args, client=client, model_id=model_id)
            fix_code = extract_strategy_code(fix_response)
            if fix_code:
                fix_code = fix_strategy_code(fix_code)

            if fix_code and not validate_syntax(fix_code) and not validate_contract(fix_code):
                write_strategy(fix_code)
                # Amend the commit
                _run_git_checked("add", STRATEGY_FILE,
                                 action="Stage fixed strategy.py")
                _run_git_checked("commit", "--amend", "--no-edit",
                                 action="Amend strategy commit after crash fix")
                description += f" (fix {crash_count})"
            else:
                break  # can't fix, give up

        # --- Step 4: Record results ---
        if not run_result["success"]:
            result = ExperimentResult(
                experiment_id=exp_id, commit=commit,
                score=0.0, growth=0.0, volatility=0.0,
                status="crash", description=description,
                strategy_code=code,
                family=infer_strategy_family(description, code))
            log_result(result)
            state.history.append(result)

            if this_is_baseline:
                print(f"\n  BASELINE CRASHED — cannot continue.")
                print(f"  Check the error above and inspect: {PROJECT_DIR / RUN_LOG}")
                print(f"  Fix {BASE_STRATEGY_FILE} and re-run agent.py.")
                sys.exit(1)
            else:
                git_revert()
                # Ensure strategy.py is the last known-good version
                write_strategy(state.current_strategy)
                print(f"  CRASH — reverted to last good strategy")
                conv.add_exchange(
                    f"Tried: {description}",
                    f"(experiment #{exp_id} crashed: "
                    f"{run_result['error'][-500:]})")
                continue

        score = run_result["score"]
        growth = run_result["growth"]
        vol = run_result["vol"]

        # Annualized return from per-fold growth
        folds_per_year = 252.0 / 90  # default test_days=90
        ann_return = (math.exp(growth * folds_per_year) - 1) * 100

        # Compute display strings
        fold_xs = run_result.get("fold_xs", [])
        params_str = format_optimal_params(meta, fold_xs)
        per_ticker_dict = run_result.get("per_ticker", {})
        per_ticker_str = format_per_ticker(per_ticker_dict)

        if params_str:
            print(f"  Params (median): {params_str}")
        if per_ticker_str:
            print(f"  Per-ticker: {per_ticker_str}")

        improved = score > state.best_score
        prev_best = state.best_score

        sign = "+" if ann_return >= 0 else ""
        if improved:
            status = "keep"
            state.best_score = score
            state.best_commit = commit
            state.current_strategy = code
            state.best_per_ticker = per_ticker_str
            delta = score - prev_best
            print(f"  ✓ KEEP  score={score:.4f}  g={growth:.3f}/v={vol:.3f}  "
                  f"ann={sign}{ann_return:.1f}%  (+{delta:.4f} vs prev best)")
        else:
            status = "discard"
            state.last_discarded_code = code
            git_revert()
            write_strategy(state.current_strategy)
            print(f"  ✗ DISCARD  score={score:.4f}  g={growth:.3f}/v={vol:.3f}  "
                  f"ann={sign}{ann_return:.1f}%  (best={state.best_score:.4f})")

        result = ExperimentResult(
            experiment_id=exp_id, commit=commit,
            score=score, growth=growth, volatility=vol,
            status=status, description=description,
            beat_pct=run_result.get("beat_pct", 0),
            worst_fold=run_result.get("worst", 0),
            best_fold=run_result.get("best", 0),
            median_params=params_str,
            per_ticker=per_ticker_str,
            strategy_code=code,
            family=infer_strategy_family(description, code))
        log_result(result)
        state.history.append(result)

        # Record exchange for context
        result_parts = [
            f"Experiment #{exp_id} [{status}]: score={score:.4f} "
            f"(growth={growth:.3f}, vol={vol:.3f}) — {description}"]
        if params_str:
            result_parts.append(f"  params: {params_str}")
        if per_ticker_str:
            result_parts.append(f"  per-ticker: {per_ticker_str}")
        conv.add_exchange(
            f"Tried: {description}",
            "\n".join(result_parts))

        # Brief pause to avoid hammering the LLM
        time.sleep(1)

# ═══════════════════════════════════════════════════════════════════════════
#  CLI
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Autonomous trading strategy researcher")
    parser.add_argument("--project-dir", default=None,
                        help="Project directory (default: directory containing agent.py)")
    parser.add_argument("--base-url", default="http://127.0.0.1:8011/v1",
                        help="OpenAI-compatible API base URL (ignored by native Claude/Gemini unless you point to a remote compatible endpoint)")
    parser.add_argument("--model", default=None,
                        help="Model id. Names containing 'claude' or 'gemini' or 'MiniMax' use native SDKs on local/default base URLs; otherwise use the OpenAI-compatible API")
    parser.add_argument("--tag", default=None,
                        help="Branch tag (e.g. mar18). Creates autoresearch/<tag>")
    parser.add_argument("--quick", action="store_true",
                        help="Fast iteration: 8 retries, 250 evals (~15s per run)")
    parser.add_argument("--medium", action="store_true",
                        help="Medium iteration: 16 retries, 500 evals (~1min per run)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override ticker symbols")
    parser.add_argument("--start", default=None, help="Data start date")
    parser.add_argument("--end", default=None, help="Data end date")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="LLM temperature")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Max curated best/diverse experiments shown to LLM (default: 10)")
    parser.add_argument("--recent-k", type=int, default=RECENT_K,
                        help="Recent experiments shown to LLM (default: 10)")
    args = parser.parse_args()

    # Override PROJECT_DIR if specified
    global PROJECT_DIR
    if args.project_dir:
        PROJECT_DIR = Path(args.project_dir).resolve()

    try:
        run_agent(args)
    except KeyboardInterrupt:
        print("\n\nStopped by user.  Results saved in results.tsv.")
        sys.exit(0)
    except GitError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
