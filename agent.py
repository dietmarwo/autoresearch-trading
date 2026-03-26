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
    python agent.py --seed-file alt.py       # start from a strategy file
    python agent.py --seed-commit abc1234    # start from strategy.py at a git revision
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
TEMPERATURE = 1.0
REFERENCE_CODE_CHARS = 2200  # keep multiple reference strategies compact for 50K context
EXPLORE_EVERY = 3            # exploration prompt cadence (0 disables)
EXPLOIT_REFERENCE_K = 2      # alternative strategy code references on exploit turns
EXPLORE_REFERENCE_K = 3      # alternative strategy code references on explore turns
MAX_LLM_OUTPUT_TOKENS = 8192
DEFAULT_TEST_DAYS = 90
TRADING_DAYS_PER_YEAR = 252.0
CRYPTO_DAYS_PER_YEAR = 365.0

# Resolved at startup — the directory containing agent.py, strategy.py, trading.py
PROJECT_DIR: Path = Path(__file__).resolve().parent

# ═══════════════════════════════════════════════════════════════════════════
#  System prompt
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT_FILE = "program_trade.md"
SYSTEM_PROMPT_BEGIN = "<!-- AUTORESEARCH_AGENT_PROMPT_BEGIN -->"
SYSTEM_PROMPT_END = "<!-- AUTORESEARCH_AGENT_PROMPT_END -->"

class PromptError(RuntimeError):
    """Raised when the canonical system prompt cannot be loaded."""


def load_system_prompt(project_dir: Optional[Path] = None) -> str:
    """Load the compact agent prompt from the marked section in program_trade.md."""
    base_dir = project_dir or PROJECT_DIR
    prompt_path = base_dir / SYSTEM_PROMPT_FILE
    try:
        text = prompt_path.read_text(encoding="utf-8")
    except OSError as e:
        raise PromptError(f"Could not read {prompt_path}: {e}") from e

    start = text.find(SYSTEM_PROMPT_BEGIN)
    end = text.find(SYSTEM_PROMPT_END)
    if start == -1 or end == -1 or end <= start:
        raise PromptError(
            f"Could not find valid prompt markers in {prompt_path}. "
            f"Expected {SYSTEM_PROMPT_BEGIN} ... {SYSTEM_PROMPT_END}"
        )

    prompt = text[start + len(SYSTEM_PROMPT_BEGIN):end].strip()
    if not prompt:
        raise PromptError(f"Prompt section in {prompt_path} is empty")

    return prompt

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
    per_ticker: str = ""           # raw geo_mean display
    per_ticker_alpha: str = ""     # alpha geo_mean display vs benchmark
    trade_counts: str = ""         # "BTC:23, ETH:18, ..."
    benchmark_name: str = ""       # e.g. HODL in crypto mode
    strategy_code: str = ""        # full strategy.py retained for prompt references
    family: str = ""               # coarse strategy family for diversity-aware prompts

@dataclass(frozen=True)
class InitialStrategy:
    code: str
    source_label: str
    run_label: str
    commit_message: str
    fix_target: str

@dataclass
class AgentState:
    best_score: float = -999.0
    best_commit: str = ""
    experiment_count: int = 0
    bars_per_year: float = TRADING_DAYS_PER_YEAR
    history: list = field(default_factory=list)   # list of ExperimentResult
    current_strategy: str = ""                     # content of strategy.py (best)
    last_discarded_code: str = ""                  # kept for backward compatibility
    best_per_ticker: str = ""                      # per-ticker breakdown of best
    best_per_ticker_alpha: str = ""                # benchmark-relative per-ticker view
    benchmark_name: str = ""                       # label for benchmark-relative stats

    def recent_history(self, limit: int = 10,
                       non_crash_only: bool = False) -> list[ExperimentResult]:
        """Return the most recent experiments, optionally excluding crashes."""
        history = self.history
        if non_crash_only:
            history = [r for r in history if r.status != "crash"]
        return history[-limit:]

    def experiments_since_keep(self) -> int:
        """How many experiments have run since the last KEEP result."""
        count = 0
        for r in reversed(self.history):
            if r.status == "keep":
                return count
            count += 1
        return count

    def recent_family_counts(self, limit: int = 8,
                             non_crash_only: bool = True) -> list[tuple[str, int, int]]:
        """
        Count recent strategy families.

        Returns tuples of (family, count, total_recent_count), sorted by count.
        """
        recent = self.recent_history(limit=limit, non_crash_only=non_crash_only)
        counts = {}
        total = 0
        for r in recent:
            family = r.family or infer_strategy_family(r.description, r.strategy_code)
            if not family:
                continue
            counts[family] = counts.get(family, 0) + 1
            total += 1
        ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        return [(family, count, total) for family, count in ranked]

    def dominant_recent_family(self, limit: int = 8,
                               min_count: int = 4,
                               min_share: float = 0.6) -> Optional[tuple[str, int, int]]:
        """Return the dominant recent family when one clearly takes over."""
        ranked = self.recent_family_counts(limit=limit, non_crash_only=True)
        if not ranked:
            return None
        family, count, total = ranked[0]
        if family == "misc":
            return None
        if total < min_count:
            return None
        if count < min_count:
            return None
        if (count / max(total, 1)) < min_share:
            return None
        return family, count, total

    def recent_failure_signals(self, limit: int = 8) -> dict:
        """Summarize common recent failure modes for adaptive prompting."""
        recent = self.recent_history(limit=limit, non_crash_only=False)
        return {
            "flat": sum("flat/no-trade" in r.description.lower() for r in recent),
            "crashes": sum(r.status == "crash" for r in recent),
            "pos_growth_neg_score": sum(
                r.status != "crash" and r.growth > 0 and r.score < 0
                for r in recent
            ),
        }

    def _format_experiment(self, r, label: str = "") -> str:
        """Format one experiment result for display."""
        folds_per_year = self.bars_per_year / DEFAULT_TEST_DAYS
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
            line += f"\n    per-ticker raw: {r.per_ticker}"
        if r.per_ticker_alpha:
            alpha_label = f" vs {r.benchmark_name}" if r.benchmark_name else ""
            line += f"\n    per-ticker alpha{alpha_label}: {r.per_ticker_alpha}"
        if r.trade_counts:
            line += f"\n    trades: {r.trade_counts}"
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

    def _reference_strategies(self, limit: int = 3) -> list:
        """
        Pick informative alternative strategy code examples for the prompt.

        Prefer strong results from families different from the current best.
        Use at most one strategy per family on the first pass so the prompt gets
        genuinely different code paths instead of minor variants.
        """
        if limit <= 0:
            return []

        best_family = infer_strategy_family("current best", self.current_strategy)
        candidates = [r for r in self._non_crash_sorted()
                      if r.strategy_code and r.commit != self.best_commit]
        if not candidates:
            return []

        def family_of(r: ExperimentResult) -> str:
            return r.family or infer_strategy_family(r.description, r.strategy_code)

        # Prefer different families first, and slightly prefer keeps over discards
        # when scores are close.
        candidates.sort(
            key=lambda r: (
                family_of(r) != best_family,
                r.status == "keep",
                r.score,
            ),
            reverse=True,
        )

        chosen = []
        chosen_ids = set()
        seen_families = set()

        for r in candidates:
            family = family_of(r)
            if family in seen_families:
                continue
            chosen.append(r)
            chosen_ids.add(r.experiment_id)
            seen_families.add(family)
            if len(chosen) >= limit:
                return chosen

        for r in candidates:
            if r.experiment_id in chosen_ids:
                continue
            chosen.append(r)
            if len(chosen) >= limit:
                break

        return chosen

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
            lines.append(f"Best per-ticker raw: {self.best_per_ticker}")
        if self.best_per_ticker_alpha:
            alpha_label = f" vs {self.benchmark_name}" if self.benchmark_name else ""
            lines.append(f"Best per-ticker alpha{alpha_label}: {self.best_per_ticker_alpha}")

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


def is_exploration_turn(state: AgentState, explore_every: int = EXPLORE_EVERY) -> bool:
    """Return True when the next prompt should favor exploration over hill-climbing."""
    return explore_every > 0 and len(state.history) > 0 and len(state.history) % explore_every == 0


def should_force_exploration(state: AgentState,
                             explore_every: int = EXPLORE_EVERY) -> bool:
    """
    Force exploration when the search is plateauing inside one dominant family.

    This helps models that over-refine one near-miss family instead of making
    a bigger structural jump.
    """
    if not state.history:
        return False
    plateau = state.experiments_since_keep()
    dominant = state.dominant_recent_family(limit=max(8, explore_every + 2))
    if plateau >= max(4, explore_every) and dominant:
        return True

    signals = state.recent_failure_signals(limit=max(8, explore_every + 2))
    if plateau >= max(6, explore_every + 2) and signals["flat"] >= 2:
        return True

    return False


def build_adaptive_guidance(state: AgentState, model_name: str = "",
                            exploration_mode: bool = False) -> str:
    """Inject steering based on recent search behavior and model tendencies."""
    notes = []
    plateau = state.experiments_since_keep()
    dominant = state.dominant_recent_family(limit=8)
    family_counts = state.recent_family_counts(limit=8, non_crash_only=True)
    signals = state.recent_failure_signals(limit=8)
    model = (model_name or "").lower()

    if plateau >= 4:
        notes.append(f"- Plateau: no KEEP for {plateau} experiments.")

    repeated_families = [f"`{family}` x{count}"
                         for family, count, total in family_counts[:2]
                         if family != "misc" and count >= 2]
    if repeated_families:
        notes.append("- Recent family mix: " + ", ".join(repeated_families) + ".")

    if dominant:
        family, count, total = dominant
        notes.append(
            f"- Family lock-in: `{family}` appeared in {count}/{total} recent non-crash runs."
        )
        if exploration_mode or plateau >= 4:
            notes.append(
                f"- On this turn, avoid another minor `{family}` variant. "
                "Change the primary entry family or the exit architecture."
            )

    if signals["pos_growth_neg_score"] >= 2:
        notes.append(
            "- Several recent runs had positive raw growth but still negative SCORE. "
            "The bottleneck is volatility and/or lagging HODL, so favor fewer trades, "
            "wider exits, slower churn, and steadier alpha."
        )

    if signals["flat"] >= 2:
        notes.append(
            "- Several recent runs were flat/no-trade rejects. Reduce stacked filters "
            "and hard gates; prefer one clear trigger plus one slower regime filter."
        )

    if signals["crashes"] >= 2:
        notes.append(
            "- Recent crashes suggest the strategy wiring is getting fragile. Prefer "
            "simpler indicator plumbing and fewer dependent arrays."
        )

    if "gemini" in model:
        notes.append(
            "- Gemini steer: favor structural jumps over tiny threshold edits when the "
            "score plateaus."
        )
        if dominant:
            notes.append(
                "- Gemini steer: stop micro-tuning the dominant family. Replace the "
                "core signal family instead of nudging thresholds."
            )

    if "minimax" in model:
        notes.append(
            "- MiniMax steer: avoid over-constraining entries with stacked confirmation "
            "filters. A simpler edge is better than a 'safer' flat strategy."
        )
        if dominant:
            notes.append(
                "- MiniMax steer: do not add yet another small filter onto the dominant "
                "family on this turn; use a different entry engine or a materially "
                "different exit."
            )

    if not notes:
        return ""

    return "Adaptive guidance:\n" + "\n".join(notes)


def build_user_message(state: AgentState, top_k: int = TOP_K,
                       recent_k: int = RECENT_K, extra: str = "",
                       exploration_mode: bool = False,
                       model_name: str = "") -> str:
    """Build the user message for the next iteration."""
    parts = []

    if state.experiment_count == 0:
        parts.append("This is the FIRST experiment.  Run the baseline EMA/SMA "
                      "crossover as-is to establish the starting score.  "
                      "Output the current strategy.py unchanged.")
    else:
        best_family = infer_strategy_family("current best", state.current_strategy)
        parts.append(state.summary(top_k=top_k, recent_k=recent_k))
        adaptive_guidance = build_adaptive_guidance(
            state, model_name=model_name, exploration_mode=exploration_mode)
        if adaptive_guidance:
            parts.append("\n" + adaptive_guidance)
        reference_k = EXPLORE_REFERENCE_K if exploration_mode else EXPLOIT_REFERENCE_K

        if not exploration_mode:
            parts.append(f"\nCurrent best strategy.py:\n```python\n{state.current_strategy}\n```")

        reference_strategies = state._reference_strategies(limit=reference_k)
        if reference_strategies:
            intro = ("\nAlternative reference strategy.py files "
                     "(diverse strong results; use as inspiration, not templates):")
            parts.append(intro)
            for idx, ref in enumerate(reference_strategies, start=1):
                family = ref.family or infer_strategy_family(ref.description, ref.strategy_code)
                code_display = ref.strategy_code
                if len(code_display) > REFERENCE_CODE_CHARS:
                    code_display = code_display[:REFERENCE_CODE_CHARS] + "\n# ... (truncated)"
                parts.append(
                    f"\nReference #{idx} [{ref.status}] score={ref.score:.4f} family={family}: "
                    f"{ref.description}\n```python\n{code_display}\n```"
                )

        if exploration_mode:
            parts.append(
                "\nPrompt mode: EXPLORATION\n"
                f"- The current best family is `{best_family}`.\n"
                "- Do NOT ask to see or reconstruct the current best code on this turn.\n"
                "- Do NOT make a tiny tweak to the current leader just to stay safe.\n"
                "- Prefer a materially different valid strategy family or a clearly different entry/exit structure.\n"
                "- Use the alternative references to branch into other promising directions, not to clone them.\n"
                "- If you feel uncertain, choose a complete valid strategy that is diverse, not a placeholder.\n\n"
                "Propose the next strategy.  Think about:\n"
                "- Which strategy families are under-explored relative to the current leader\n"
                "- Which weak tickers suggest the current family is over-specialized\n"
                "- Which alternative references point to different entry logic, regime filters, or exits\n"
                "- How to increase diversity without exploding parameter count\n"
                "Output the COMPLETE strategy.py in a python code block."
            )
        else:
            parts.append(
                "\nPrompt mode: EXPLOITATION\n"
                f"- The current best family is `{best_family}`.\n"
                "- Focus on improving the current best without making it more fragile.\n"
                "- Meaningful refinements beat cosmetic rewrites.\n\n"
                "Propose the next strategy.  Think about:\n"
                "- Which strategy families keep winning, and which families are under-explored\n"
                "- What the median params tell you (hitting bounds? converging?)\n"
                "- What the strong near-misses got right, and what likely kept them below the best\n"
                "- What the representative failures reveal about overfitting, flatness, or volatility\n"
                "- Which tickers are weak in per-ticker breakdown\n"
                "- How to improve the current best without making it more fragile\n"
                "Output the COMPLETE strategy.py in a python code block."
            )

    if extra:
        parts.append(f"\n{extra}")

    return "\n\n".join(parts)


def is_crypto_ticker(ticker: str) -> bool:
    """Heuristic: common Yahoo crypto pairs end with quote-currency suffixes."""
    t = (ticker or "").upper()
    return (t.endswith("-USD") or t.endswith("-USDT") or t.endswith("-USDC")
            or t.endswith("-BTC") or t.endswith("-ETH"))


def resolve_market_mode(tickers: Optional[list[str]],
                        requested_mode: str = "auto") -> str:
    """Mirror trading.py market-mode behavior inside the agent."""
    if requested_mode != "auto":
        return requested_mode
    if tickers and all(is_crypto_ticker(t) for t in tickers):
        return "crypto"
    return "equity"


def infer_bars_per_year(market_mode: str) -> float:
    """Use 365 for crypto-like 24/7 markets, otherwise 252 trading days."""
    if market_mode == "crypto":
        return CRYPTO_DAYS_PER_YEAR
    return TRADING_DAYS_PER_YEAR


def build_market_context(market_mode: str) -> str:
    """Optional market-specific prompt guidance."""
    if market_mode == "equity":
        return (
            "Market context:\n"
            "- Equity mode scores fold-by-fold return versus cash, not alpha versus HODL.\n"
            "- In persistent bull trends, long time spent in cash has a high opportunity cost.\n"
            "- Favor simple, durable exposure with timely re-entry over heavily gated 'safe' logic.\n"
            "- Avoid stacking many confirmation filters that create flat/no-trade behavior.\n"
            "- Exits should reduce major drawdowns without causing the strategy to miss every recovery leg.\n"
            "- A flat/no-trade strategy is not a win. Fix crashes at the root cause instead of suppressing trades."
        )
    if market_mode != "crypto":
        return ""
    return (
        "Market context:\n"
        "- The ticker set looks like crypto spot pairs trading 24/7.\n"
        "- Crypto mode scores fold-by-fold excess return versus HODL, not raw return versus cash.\n"
        "- Favor persistent trend/breakout/regime-aware logic over stock-style short-horizon mean reversion.\n"
        "- Wider ATR/NATR stops, longer hold times, and de-risking after volatility spikes are often better fits.\n"
        "- Yahoo crypto volume can be noisy; use volume as a secondary confirmation, not the entire edge.\n"
        "- A flat/no-trade strategy is not a win. Fix crashes at the root cause instead of suppressing trades."
    )


def build_format_repair_message(issue: str) -> str:
    """Ask the LLM to re-emit a valid strategy file after format drift."""
    return (
        f"FORMAT ERROR: your previous reply had {issue}.\n\n"
        "Reply again with:\n"
        "1. Exactly one fenced ```python``` code block containing the COMPLETE strategy.py file.\n"
        "2. Exactly one final `DESCRIPTION: ...` line.\n\n"
        "Do not omit `get_strategy`, `simulate`, or the njit trading loop. "
        "Do not output placeholders or prose-only responses."
    )


def build_contract_message(error_text: str) -> str:
    """Ask the LLM to fix a strategy interface/contract violation."""
    return (
        "CONTRACT ERROR!  The strategy file violated the required interface:\n\n"
        f"```\n{error_text}\n```\n\n"
        "Fix the contract and output the corrected COMPLETE strategy.py.\n"
        "- `simulate` must be exactly `simulate(close, high, low, volume, x)`.\n"
        "- `variables` must be a plain Python list of quoted names.\n"
        "- `bounds` must contain two lists with the same length as `variables`.\n"
        "- Keep `_execute(...)` call/definition argument count and order aligned."
    )


def build_crash_message(error_text: str, attempt: int) -> str:
    """Build a message asking the LLM to fix a crash."""
    extra = ("Fix the root cause.  Do NOT mask the error with broad try/except, "
             "do NOT return a constant `(1.0, 0)`, and do NOT disable trading "
             "just to avoid the crash.")
    lower = error_text.lower()
    if "ZeroDivisionError" in error_text:
        extra += (" For divide-by-zero issues, clamp denominators with a small "
                  "epsilon or skip invalid bars instead.")
    if any(tok in lower for tok in (
        "typingerror", "cannot determine numba type", "no implementation of function",
        "array(float", "array(int", "reflected list"
    )):
        extra += (" Inside @njit, branch conditions must use scalars, not whole arrays. "
                  "Index indicator arrays with `[i]` before comparing or combining them.")
    if any(tok in lower for tok in (
        "positional argument", "takes ", "were given", "missing 1 required positional argument"
    )):
        extra += (" Match function signatures exactly: keep helper argument counts unchanged, "
                  "keep `simulate(close, high, low, volume, x)` unchanged, and ensure the "
                  "`_execute(...)` call and definition have the same order and count.")
    return (f"CRASH (attempt {attempt})!  The strategy failed with this error:\n\n"
            f"```\n{error_text[-2000:]}\n```\n\n"
            f"{extra}\n\n"
            f"Output the corrected COMPLETE strategy.py.")


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
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    fn_defs = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    if "get_strategy" not in fn_defs:
        return "Missing get_strategy() function"
    if "simulate" not in fn_defs:
        return "Missing simulate() function"

    simulate_fn = fn_defs["simulate"]
    sim_args = [arg.arg for arg in simulate_fn.args.args]
    if (simulate_fn.args.posonlyargs or simulate_fn.args.kwonlyargs or
            simulate_fn.args.vararg or simulate_fn.args.kwarg or
            sim_args != ["close", "high", "low", "volume", "x"]):
        return "simulate() must have exact signature: simulate(close, high, low, volume, x)"

    meta = extract_strategy_meta(code)
    variables = meta.get("variables")
    if not variables:
        return "Missing or malformed variables in get_strategy"
    if any(not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', v) for v in variables):
        return "variables must be quoted identifier-like names only"
    if len(set(variables)) != len(variables):
        return "variables entries must be unique"

    lo = meta.get("lo")
    hi = meta.get("hi")
    if lo is None or hi is None:
        return "Missing or malformed bounds in get_strategy"
    if len(lo) != len(hi) or len(lo) != len(variables):
        return "bounds dimensions must match number of variables"

    if re.search(r'^\s*return\s*\(?\s*1(?:\.0+)?\s*,\s*0\s*\)?\s*$', code, re.MULTILINE):
        return "Constant flat fallback `return 1.0, 0` is not allowed"
    if re.search(r'^\s*except\s+Exception\s*:', code, re.MULTILINE):
        return "Broad `except Exception:` is not allowed in strategy.py"
    if re.search(r'^\s*except\s+ZeroDivisionError\s*:', code, re.MULTILINE):
        return "Do not mask ZeroDivisionError in strategy.py; fix the denominator"
    return None


def extract_strategy_meta(code: str) -> dict:
    """
    Parse variable names and bounds from strategy code.
    Returns {'variables': [...], 'lo': [...], 'hi': [...]} or empty dict.
    """
    meta = {}

    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None

    if tree is not None:
        get_strategy_fn = next(
            (node for node in tree.body
             if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name == "get_strategy"),
            None,
        )
        if get_strategy_fn is not None:
            for node in ast.walk(get_strategy_fn):
                if not isinstance(node, ast.Return) or node.value is None:
                    continue

                value_map = {}
                if isinstance(node.value, ast.Dict):
                    for key_node, value_node in zip(node.value.keys, node.value.values):
                        if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                            value_map[key_node.value] = value_node
                elif (isinstance(node.value, ast.Call) and isinstance(node.value.func, ast.Name)
                      and node.value.func.id == "dict"):
                    for kw in node.value.keywords:
                        if kw.arg:
                            value_map[kw.arg] = kw.value

                variables_node = value_map.get("variables")
                if variables_node is not None:
                    try:
                        raw_variables = ast.literal_eval(variables_node)
                        if (isinstance(raw_variables, (list, tuple))
                                and all(isinstance(v, str) for v in raw_variables)):
                            meta["variables"] = list(raw_variables)
                    except Exception:
                        pass

                bounds_node = value_map.get("bounds")
                if bounds_node is not None:
                    try:
                        raw_bounds = ast.literal_eval(bounds_node)
                        if isinstance(raw_bounds, (list, tuple)) and len(raw_bounds) == 2:
                            lo, hi = raw_bounds
                            if isinstance(lo, (list, tuple)) and isinstance(hi, (list, tuple)):
                                meta["lo"] = [float(x) for x in lo]
                                meta["hi"] = [float(x) for x in hi]
                    except Exception:
                        pass

                if meta:
                    return meta

    # Regex fallback for partially malformed code.
    var_match = re.search(r'variables\s*=\s*\[([^\]]+)\]', code)
    if var_match:
        raw = var_match.group(1)
        variables = [s.strip().strip('"').strip("'") for s in raw.split(',')]
        if variables and all(variables):
            meta['variables'] = variables

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
    """Infer a medium-grained structural family label from free text and code."""
    text = f"{description}\n{code[:4000]}".lower()

    def has_any(patterns):
        return any(p in text for p in patterns)

    def detect_first(options):
        for tag, patterns in options:
            if has_any(patterns):
                return tag
        return ""

    def detect_many(options, limit: int) -> list[str]:
        found = []
        for tag, patterns in options:
            if has_any(patterns):
                found.append(tag)
            if len(found) >= limit:
                break
        return found

    if has_any(["regime switch", "regime-switch", "regime adaptive", "regime-adaptive"]):
        archetype = "regime"
    elif has_any(["breakout", "channel breakout", "new high", "range break"]):
        archetype = "breakout"
    elif has_any(["mean reversion", "mean-reversion", "reversion_score", "fade move"]):
        archetype = "mean-rev"
    elif has_any(["pullback", "dip-buy", "dip buying"]):
        archetype = "pullback"
    elif has_any(["trend-following", "trend following", "trend-continuation", "trend continuation"]):
        archetype = "trend"
    elif has_any(["momentum", "acceleration"]):
        archetype = "momentum"
    else:
        archetype = detect_first([
            ("regime", [" choppiness", "choppiness_index_np(", "trend_strength_np("]),
            ("breakout", ["donchian_np(", "distance_from_high_np("]),
            ("mean-rev", ["mean_reversion_score_np(", "bollinger_pctb_np(", "bollinger band mean"]),
            ("trend", ["supertrend_np(", "ema crossover", "trend filter"]),
            ("momentum", ["macd histogram", "macd_np(", "roc_np(", "trix_np(", "tsi_np("]),
        ])

    signal_tags = detect_many([
        ("ema-sma", ["ema/sma", "ema_sma"]),
        ("ema", ["ema_np(", " ema crossover", " fast ema", " slow ema"]),
        ("sma", ["sma_np(", " simple moving average", " slow sma"]),
        ("macd", ["macd_np(", "macd histogram", "macd crossover", "macd flip"]),
        ("supertrend", ["supertrend_np(", "supertrend"]),
        ("rsi", ["rsi_np(", "rsi oversold", "rsi overbought", "rsi "]),
        ("bollinger", ["bollinger_np(", "bollinger_bandwidth_np(", "bollinger band"]),
        ("donchian", ["donchian_np(", "donchian"]),
        ("stoch", ["stochastic_np(", "stoch_rsi_np(", "stochastic", "stoch rsi"]),
        ("roc", ["roc_np(", " roc "]),
        ("ichimoku", ["ichimoku_np(", "ichimoku"]),
        ("psar", ["psar_np(", "parabolic sar", " psar"]),
        ("vwap", ["vwap_np(", "rolling_vwap_np(", "vwap"]),
        ("cci", ["cci_np(", " cci "]),
    ], limit=2)

    filter_tag = detect_first([
        ("adx", ["adx_np(", " adx", "adx filter", "trend strength"]),
        ("vol-filter", ["volatility filter", "realized_volatility_np(", "historical_vol_np(", "ulcer_index_np("]),
        ("chop-filter", ["choppiness_index_np(", "choppiness"]),
        ("volume", ["volume confirmation", "volume filter", "obv_np(", "cmf_np(", "mfi_np(", "volume-based"]),
    ])

    exit_tag = detect_first([
        ("dual-exit", ["dual exit", "dual-exit", "backup exit", "secondary exit"]),
        ("natr-stop", ["natr trailing", "natr stop", "natr_np("]),
        ("atr-stop", ["atr trailing", "atr stop", "atr_np(", "trailing_stop_hit("]),
        ("pct-stop", ["trail_pct", "stop_pct", "percentage trailing", "fixed percentage stop"]),
        ("profit-target", ["profit target", "take profit", "take-profit", "fixed % target"]),
        ("trend-exit", ["trend destruction", "trend reversal", "bearish crossover", "supertrend flip exit", "macd flip exit"]),
        ("partial-size", ["buy_fraction(", "sell_fraction(", "half position", "partial position", "position sizing"]),
    ])

    tags = []
    if archetype:
        tags.append(archetype)
    tags.extend(signal_tags)
    if filter_tag:
        tags.append(filter_tag)
    if exit_tag:
        tags.append(exit_tag)

    # Deduplicate while preserving order.
    seen = set()
    unique = []
    for tag in tags:
        if tag not in seen:
            seen.add(tag)
            unique.append(tag)

    if not unique:
        return "misc"
    return "+".join(unique[:5])


def write_strategy(code: str):
    """Write strategy.py in the project directory."""
    (PROJECT_DIR / STRATEGY_FILE).write_text(code)


def read_strategy() -> str:
    """Read current strategy.py from the project directory."""
    p = PROJECT_DIR / STRATEGY_FILE
    return p.read_text() if p.exists() else ""


class GitError(RuntimeError):
    """Raised when git bookkeeping cannot be completed safely."""


class SeedError(RuntimeError):
    """Raised when the chosen starting strategy cannot be resolved safely."""


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


def git_read_file(revision: str, path: str) -> str:
    """Read a tracked file from a git revision."""
    result = _run_git_checked("show", f"{revision}:{path}",
                              action=f"Read {path} from git revision {revision}")
    content = result.stdout
    if not content.strip():
        raise GitError(f"Read {path} from git revision {revision} returned empty content")
    return content


def load_initial_strategy(args) -> InitialStrategy:
    """Resolve the starting strategy from baseline, a file, or a git revision."""
    if args.seed_file:
        seed_path = Path(args.seed_file).expanduser()
        if not seed_path.is_absolute():
            seed_path = (Path.cwd() / seed_path).resolve()
        else:
            seed_path = seed_path.resolve()
        if not seed_path.is_file():
            raise SeedError(f"Seed file not found: {seed_path}")
        code = seed_path.read_text()
        if not code.strip():
            raise SeedError(f"Seed file is empty: {seed_path}")
        return InitialStrategy(
            code=code,
            source_label=str(seed_path),
            run_label=f"seed ({seed_path.name})",
            commit_message=f"seed from file {seed_path.name}",
            fix_target=str(seed_path),
        )

    if args.seed_commit:
        code = git_read_file(args.seed_commit, STRATEGY_FILE)
        return InitialStrategy(
            code=code,
            source_label=f"{args.seed_commit}:{STRATEGY_FILE}",
            run_label=f"seed ({args.seed_commit})",
            commit_message=f"seed from commit {args.seed_commit}",
            fix_target=f"{args.seed_commit}:{STRATEGY_FILE}",
        )

    base_path = PROJECT_DIR / BASE_STRATEGY_FILE
    code = base_path.read_text()
    if not code.strip():
        raise SeedError(f"{BASE_STRATEGY_FILE} is empty")
    return InitialStrategy(
        code=code,
        source_label=BASE_STRATEGY_FILE,
        run_label="baseline",
        commit_message="baseline",
        fix_target=BASE_STRATEGY_FILE,
    )


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
                  fold_xs=[], per_ticker_trades={}, per_ticker_benchmark={},
                  per_ticker_alpha={}, benchmark_name="")

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
    beat_match = re.search(r'(?:profitable|beat [^\n]+?) in (\d+)% of folds', output)
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

    per_ticker = {}
    per_ticker_trades = {}
    per_ticker_benchmark = {}
    per_ticker_alpha = {}

    headline_match = re.search(
        r'Walk-forward:.*?,\s*([A-Za-z0-9_]+)\s+geo_mean\s*=\s*([-\d.]+),\s*alpha_geo_mean',
        output,
    )
    if headline_match:
        result["benchmark_name"] = headline_match.group(1).upper()

    for line in output.splitlines():
        if ": OOS factors across folds" not in line:
            continue
        ticker_match = re.match(r'\s*(\S+):\s*OOS factors across folds', line)
        if not ticker_match:
            continue
        ticker = ticker_match.group(1)

        geo_match = re.search(r'geo_mean\s*=\s*([-\d.]+)', line)
        if geo_match:
            per_ticker[ticker] = float(geo_match.group(1))

        trades_match = re.search(r'total_trades\s*=\s*(\d+)', line)
        if trades_match:
            per_ticker_trades[ticker] = int(trades_match.group(1))

        alpha_match = re.search(r'alpha_geo_mean\s*=\s*([-\d.]+)', line)
        if alpha_match:
            per_ticker_alpha[ticker] = float(alpha_match.group(1))

        for label, value in re.findall(r'([A-Za-z0-9_]+_geo_mean)\s*=\s*([-\d.]+)', line):
            if label == "alpha_geo_mean":
                continue
            base = label[:-len("_geo_mean")]
            if base:
                result["benchmark_name"] = base.upper()
                per_ticker_benchmark[ticker] = float(value)
                break

    result["per_ticker"] = per_ticker
    result["per_ticker_trades"] = per_ticker_trades
    result["per_ticker_benchmark"] = per_ticker_benchmark
    result["per_ticker_alpha"] = per_ticker_alpha

    return result


def is_flat_result(run_result: dict) -> bool:
    """
    Detect degenerate cash-preservation results.

    We only see rounded values from the runner output, so the check is exact on
    those rounded numbers: score/growth/vol must all be zero and every
    per-ticker factor must print as 1.000.
    """
    if not run_result.get("success"):
        return False
    per_ticker_trades = run_result.get("per_ticker_trades") or {}
    if per_ticker_trades and all(v == 0 for v in per_ticker_trades.values()):
        return True
    if run_result.get("score") != 0.0 or run_result.get("growth") != 0.0 or run_result.get("vol") != 0.0:
        return False
    per_ticker = run_result.get("per_ticker") or {}
    return bool(per_ticker) and all(v == 1.0 for v in per_ticker.values())


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


def format_per_ticker(per_ticker: dict, test_days: int = DEFAULT_TEST_DAYS,
                      bars_per_year: float = TRADING_DAYS_PER_YEAR) -> str:
    """Format per-ticker geo_means with annualized returns.
    
    Example: 'AAPL:1.002(+0.6%/yr), MSFT:1.016(+4.5%/yr)'
    """
    if not per_ticker:
        return ""
    folds_per_year = bars_per_year / test_days
    parts = []
    for t, v in per_ticker.items():
        ticker = t.replace('-USD', '').replace('-', '')
        ann_return = (v ** folds_per_year - 1) * 100
        sign = "+" if ann_return >= 0 else ""
        parts.append(f"{ticker}:{v:.3f}({sign}{ann_return:.1f}%/yr)")
    return ", ".join(parts)


def format_trade_counts(per_ticker_trades: dict) -> str:
    """Format per-ticker total trade counts."""
    if not per_ticker_trades:
        return ""
    parts = []
    for t, n in per_ticker_trades.items():
        ticker = t.replace('-USD', '').replace('-', '')
        parts.append(f"{ticker}:{int(n)}")
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

    # --- Setup git ---
    git_ensure_repo()
    if args.tag:
        git_setup_branch(args.tag)

    # --- Seed the starting strategy deterministically on the chosen branch ---
    initial_strategy = load_initial_strategy(args)
    write_strategy(initial_strategy.code)
    print(f"  Seeded {STRATEGY_FILE} from {initial_strategy.source_label}")

    print(f"  Preflight check on initial strategy...")
    pf_err = preflight_check()
    if pf_err is not None:
        print(f"  [ERROR] Initial strategy fails preflight:")
        for line in pf_err.strip().split('\n')[-15:]:
            print(f"    {line}")
        print(f"\n  Fix {initial_strategy.fix_target} and re-run.")
        sys.exit(1)
    print(f"  Preflight OK")

    init_results_tsv()

    market_mode = resolve_market_mode(args.tickers, args.market_mode)
    bars_per_year = infer_bars_per_year(market_mode)
    state = AgentState(bars_per_year=bars_per_year)
    state.current_strategy = read_strategy()
    conv = Conversation(load_system_prompt())
    market_context = build_market_context(market_mode)

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
    extra_args_parts.append(f"--market-mode {market_mode}")
    extra_args = " ".join(extra_args_parts)

    print(f"Extra args: {extra_args or '(default)'}")
    print("=" * 60)
    print("Starting autonomous experiment loop.  Ctrl+C to stop.")
    print("=" * 60)

    # --- Experiment loop ---
    is_initial_strategy = True

    while True:
        state.experiment_count += 1
        exp_id = state.experiment_count
        print(f"\n{'='*60}")
        print(f"  EXPERIMENT #{exp_id}  |  best={state.best_score:.4f}  |  "
              f"{datetime.now().strftime('%H:%M:%S')}")
        print(f"{'='*60}")

        if is_initial_strategy:
            # --- Initial seed: run the chosen starting strategy unchanged ---
            is_initial_strategy = False
            this_is_initial_strategy = True
            code = read_strategy()
            meta = extract_strategy_meta(code)
            description = initial_strategy.run_label
            user_msg = "(initial seed run)"
            meta_str = format_strategy_meta(meta)
            if initial_strategy.commit_message == "baseline":
                print(f"  Running baseline strategy...")
            else:
                print(f"  Running seed strategy from {initial_strategy.source_label}...")
            if meta_str:
                print(f"  Variables: {meta_str}")
            commit = git_commit(initial_strategy.commit_message)

        else:
            this_is_initial_strategy = False
            # --- Step 1: Ask LLM for next strategy ---
            scheduled_exploration = is_exploration_turn(state, args.explore_every)
            forced_exploration = should_force_exploration(state, args.explore_every)
            exploration_mode = scheduled_exploration or forced_exploration
            if forced_exploration and not scheduled_exploration:
                dominant = state.dominant_recent_family(limit=max(8, args.recent_k))
                if dominant:
                    family, count, total = dominant
                    mode_label = (f"FORCED EXPLORATION "
                                  f"(plateau={state.experiments_since_keep()}, "
                                  f"dominant={family} {count}/{total})")
                else:
                    mode_label = f"FORCED EXPLORATION (plateau={state.experiments_since_keep()})"
            else:
                mode_label = "EXPLORATION" if exploration_mode else "EXPLOITATION"
            print(f"  Prompt mode: {mode_label}")
            user_msg = build_user_message(
                state, top_k=args.top_k, recent_k=args.recent_k,
                extra=market_context, exploration_mode=exploration_mode,
                model_name=model_id or args.model or "")
            messages = conv.messages(user_msg)

            print("  Asking LLM for next strategy...")
            response_text = call_llm(messages, args, client=client, model_id=model_id)

            # --- Step 2: Extract and validate strategy code ---
            code = extract_strategy_code(response_text)
            description = extract_description(response_text)

            if code is None:
                format_issue = "an empty response" if not response_text.strip() else "no python code block"
                print(f"  [FORMAT ERROR] {format_issue}")
                fix_msg = build_format_repair_message(format_issue)
                fix_messages = conv.messages(fix_msg)
                fix_response = call_llm(fix_messages, args, client=client, model_id=model_id)
                if description == "no description":
                    fix_desc = extract_description(fix_response)
                    if fix_desc != "no description":
                        description = fix_desc
                code = extract_strategy_code(fix_response)
                if code:
                    description += " (format fix)"
                else:
                    print("  [SKIP] Could not recover a valid python code block.")
                    conv.add_exchange(
                        "Propose the next strategy.",
                        "(format failure — output the COMPLETE strategy.py "
                        "inside a single ```python ... ``` block)")
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
                print(f"  [CONTRACT ERROR] {contract_err}")
                fix_msg = build_contract_message(contract_err)
                fix_messages = conv.messages(fix_msg)
                fix_response = call_llm(fix_messages, args, client=client, model_id=model_id)
                fix_code = extract_strategy_code(fix_response)
                if fix_code:
                    fix_code = fix_strategy_code(fix_code)
                fix_err = validate_contract(fix_code) if fix_code else "Missing corrected code block"
                if fix_code is None or validate_syntax(fix_code) or fix_err:
                    print("  [SKIP] Could not fix contract, skipping...")
                    state.experiment_count -= 1
                    continue
                if description == "no description":
                    fix_desc = extract_description(fix_response)
                    if fix_desc != "no description":
                        description = fix_desc
                code = fix_code
                description += " (contract fix)"

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

            # For the initial seed, don't ask LLM to fix — just retry once
            # (crash may be transient, e.g. numba cache issue)
            if this_is_initial_strategy:
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

            if this_is_initial_strategy:
                print(f"\n  INITIAL STRATEGY CRASHED — cannot continue.")
                print(f"  Check the error above and inspect: {PROJECT_DIR / RUN_LOG}")
                print(f"  Fix {initial_strategy.fix_target} and re-run agent.py.")
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
        folds_per_year = bars_per_year / DEFAULT_TEST_DAYS
        ann_return = (math.exp(growth * folds_per_year) - 1) * 100

        # Compute display strings
        fold_xs = run_result.get("fold_xs", [])
        params_str = format_optimal_params(meta, fold_xs)
        per_ticker_dict = run_result.get("per_ticker", {})
        per_ticker_alpha_dict = run_result.get("per_ticker_alpha", {})
        per_ticker_trades = run_result.get("per_ticker_trades", {})
        benchmark_name = run_result.get("benchmark_name", "")
        if not benchmark_name and market_mode == "crypto":
            benchmark_name = "HODL"
        per_ticker_str = format_per_ticker(
            per_ticker_dict, test_days=DEFAULT_TEST_DAYS, bars_per_year=bars_per_year)
        per_ticker_alpha_str = format_per_ticker(
            per_ticker_alpha_dict, test_days=DEFAULT_TEST_DAYS, bars_per_year=bars_per_year)
        trade_counts_str = format_trade_counts(per_ticker_trades)
        flat_result = is_flat_result(run_result)
        reject_flat = flat_result and not this_is_initial_strategy
        result_description = (description + " (flat/no-trade result rejected)"
                              if reject_flat else description)

        if params_str:
            print(f"  Params (median): {params_str}")
        if per_ticker_str:
            print(f"  Per-ticker raw: {per_ticker_str}")
        if per_ticker_alpha_str:
            print(f"  Per-ticker alpha vs {benchmark_name}: {per_ticker_alpha_str}")
        if trade_counts_str:
            print(f"  Trades: {trade_counts_str}")
        if reject_flat:
            print("  [GUARDRAIL] Rejecting flat/no-trade result "
                  "(score/growth/vol == 0 and every ticker == 1.000)")

        improved = (score > state.best_score) and not reject_flat
        prev_best = state.best_score

        sign = "+" if ann_return >= 0 else ""
        if improved:
            status = "keep"
            state.best_score = score
            state.best_commit = commit
            state.current_strategy = code
            state.best_per_ticker = per_ticker_str
            state.best_per_ticker_alpha = per_ticker_alpha_str
            state.benchmark_name = benchmark_name
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
            status=status, description=result_description,
            beat_pct=run_result.get("beat_pct", 0),
            worst_fold=run_result.get("worst", 0),
            best_fold=run_result.get("best", 0),
            median_params=params_str,
            per_ticker=per_ticker_str,
            per_ticker_alpha=per_ticker_alpha_str,
            trade_counts=trade_counts_str,
            benchmark_name=benchmark_name,
            strategy_code=code,
            family=infer_strategy_family(description, code))
        log_result(result)
        state.history.append(result)

        # Record exchange for context
        result_parts = [
            f"Experiment #{exp_id} [{status}]: score={score:.4f} "
            f"(growth={growth:.3f}, vol={vol:.3f}) — {result_description}"]
        if params_str:
            result_parts.append(f"  params: {params_str}")
        if per_ticker_str:
            result_parts.append(f"  per-ticker raw: {per_ticker_str}")
        if per_ticker_alpha_str:
            result_parts.append(
                f"  per-ticker alpha vs {benchmark_name}: {per_ticker_alpha_str}")
        if trade_counts_str:
            result_parts.append(f"  trades: {trade_counts_str}")
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
    seed_group = parser.add_mutually_exclusive_group()
    seed_group.add_argument("--seed-file", default=None,
                            help="Path to a strategy.py file to score first instead of base_strategy.py")
    seed_group.add_argument("--seed-commit", default=None,
                            help="Git revision whose strategy.py should be scored first instead of base_strategy.py")
    parser.add_argument("--quick", action="store_true",
                        help="Fast iteration: 8 retries, 250 evals (~15s per run)")
    parser.add_argument("--medium", action="store_true",
                        help="Medium iteration: 16 retries, 500 evals (~1min per run)")
    parser.add_argument("--tickers", nargs="+", default=None,
                        help="Override ticker symbols")
    parser.add_argument("--start", default=None, help="Data start date")
    parser.add_argument("--end", default=None, help="Data end date")
    parser.add_argument("--market-mode", choices=["auto", "equity", "crypto"],
                        default="auto",
                        help="Market-specific behavior passed through to trading.py")
    parser.add_argument("--temperature", type=float, default=TEMPERATURE,
                        help="LLM temperature")
    parser.add_argument("--top-k", type=int, default=TOP_K,
                        help="Max curated best/diverse experiments shown to LLM (default: 10)")
    parser.add_argument("--recent-k", type=int, default=RECENT_K,
                        help="Recent experiments shown to LLM (default: 10)")
    parser.add_argument("--explore-every", type=int, default=EXPLORE_EVERY,
                        help="Run an exploration prompt every N completed experiments (0 disables; default: 6)")
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
    except SeedError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except GitError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)
    except PromptError as e:
        print(f"\n[ERROR] {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
