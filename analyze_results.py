#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("XDG_CACHE_HOME", "/tmp")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

import trading
from strategy_helpers import warmup as warmup_helpers

START_CAPITAL = 1_000_000.0
DEFAULT_START = "2019-01-01"
DEFAULT_END = "2030-04-30"
DEFAULT_EQUITY_TICKERS = ["AAPL", "AMD", "GOOGL", "NVDA"]
DEFAULT_CRYPTO_TICKERS = ["BTC-USD", "ETH-USD", "XRP-USD", "ADA-USD"]


@dataclass
class StrategyAnalysis:
    file_name: str
    strategy_name: str
    market: str
    tickers: list[str]
    summary: dict[str, Any]
    dates: list[str]
    cumulative_values: list[float]
    cumulative_scores: list[float]
    fold_factors: list[float]
    benchmark_factors: list[float]
    hodl_values: list[float]
    hodl_scores: list[float]
    hodl_factors: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze all strategy files in results/ with walk-forward optimization."
    )
    parser.add_argument("--results-dir", default="results", help="Directory containing strategy .py files")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for logs, summaries, and plots. Defaults to the results directory.",
    )
    parser.add_argument("--start", default=DEFAULT_START, help="Backtest start date")
    parser.add_argument("--end", default=DEFAULT_END, help="Backtest end date")
    parser.add_argument("--equity-tickers", nargs="+", default=DEFAULT_EQUITY_TICKERS)
    parser.add_argument("--crypto-tickers", nargs="+", default=DEFAULT_CRYPTO_TICKERS)
    parser.add_argument("--train-days", type=int, default=365)
    parser.add_argument("--test-days", type=int, default=90)
    parser.add_argument("--step-days", type=int, default=90)
    parser.add_argument("--num-retries", type=int, default=24)
    parser.add_argument("--max-evals", type=int, default=500)
    parser.add_argument("--risk-lambda", type=float, default=0.5)
    parser.add_argument("--start-capital", type=float, default=START_CAPITAL)
    return parser.parse_args()


def configure_logging(output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "analyze_results.log"
    logger.remove()
    logger.add(
        sys.stdout,
        format="{time:HH:mm:ss.SS} | {level} | {message}",
        level="INFO",
    )
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss.SS} | {level} | {message}",
        level="INFO",
    )
    return log_path


def discover_strategy_files(results_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in results_dir.glob("*.py")
        if path.is_file() and not path.name.startswith("_")
    )


def infer_market(path: Path) -> str:
    return "crypto" if "crypto" in path.stem.lower() else "equity"


def load_strategy_from_path(path: Path) -> dict[str, Any]:
    module_key = hashlib.md5(str(path.resolve()).encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    module_name = f"analyze_results_{path.stem}_{module_key}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create import spec for {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    if not hasattr(module, "get_strategy"):
        raise ValueError(f"{path} does not define get_strategy()")

    strategy_spec = module.get_strategy()
    required = {"name", "variables", "bounds", "simulate"}
    missing = required - set(strategy_spec.keys())
    if missing:
        raise ValueError(f"{path} missing strategy keys: {sorted(missing)}")

    lo, hi = strategy_spec["bounds"]
    if len(lo) != len(hi) or len(lo) != len(strategy_spec["variables"]):
        raise ValueError(f"{path} has inconsistent bounds/variables lengths")

    warmup_helpers()
    trading._warmup_strategy(strategy_spec)
    return strategy_spec


def build_curves(
    result: trading.WalkForwardResult,
    start_capital: float,
    risk_lambda: float,
) -> tuple[list[str], list[float], list[float], list[float], list[float]]:
    dates: list[str] = []
    cumulative_values: list[float] = []
    cumulative_scores: list[float] = []
    fold_factors: list[float] = []
    benchmark_factors: list[float] = []

    value = float(start_capital)
    for fold in result.folds:
        factor = float(fold.test_geo_mean)
        benchmark = float(fold.benchmark_geo_mean) if fold.benchmark_geo_mean > 0 else 1.0
        value *= factor

        dates.append(fold.test_end)
        cumulative_values.append(value)
        fold_factors.append(factor)
        benchmark_factors.append(benchmark)

        score = trading.compute_score(
            fold_factors,
            risk_lambda=risk_lambda,
        )["score"]
        cumulative_scores.append(float(score))

    return dates, cumulative_values, cumulative_scores, fold_factors, benchmark_factors


def build_hodl_curves(
    tickers: list[str],
    result: trading.WalkForwardResult,
    start: str,
    end: str,
    start_capital: float,
    risk_lambda: float,
) -> tuple[list[float], list[float], list[float]]:
    histories = trading.load_tickers(tickers, start, end)
    hodl_factors: list[float] = []
    hodl_values: list[float] = []
    hodl_scores: list[float] = []
    hodl_value = float(start_capital)

    for fold in result.folds:
        per_ticker_factors = []
        for ticker in tickers:
            window = histories[ticker].loc[fold.test_start:fold.test_end]
            ohlcv = trading.extract_ohlcv(window)
            factor = max(float(trading.hodl(ohlcv["close"], start_capital)), 1e-12)
            per_ticker_factors.append(factor)

        fold_hodl = float(np.prod(per_ticker_factors) ** (1.0 / len(per_ticker_factors)))
        hodl_factors.append(fold_hodl)
        hodl_value *= fold_hodl
        hodl_values.append(hodl_value)
        hodl_scores.append(
            float(
                trading.compute_score(
                    hodl_factors,
                    risk_lambda=risk_lambda,
                )["score"]
            )
        )

    return hodl_values, hodl_scores, hodl_factors


def summarize_result(
    strategy_path: Path,
    strategy_spec: dict[str, Any],
    market: str,
    tickers: list[str],
    result: trading.WalkForwardResult,
    dates: list[str],
    cumulative_values: list[float],
    cumulative_scores: list[float],
    fold_factors: list[float],
    benchmark_factors: list[float],
    hodl_values: list[float],
    hodl_scores: list[float],
    hodl_factors: list[float],
    risk_lambda: float,
) -> StrategyAnalysis:
    score = result.score(risk_lambda=risk_lambda)
    total_trades = int(sum(sum(fold.test_trades) for fold in result.folds))
    final_value = cumulative_values[-1] if cumulative_values else float("nan")
    final_score = cumulative_scores[-1] if cumulative_scores else float("nan")

    summary = dict(
        strategy_file=strategy_path.name,
        strategy_name=strategy_spec["name"],
        market=market,
        tickers=",".join(tickers),
        folds=len(result.folds),
        oos_geo_mean=float(result.oos_geo_mean),
        final_value=float(final_value),
        final_log_sharpe=float(final_score),
        score=float(score["score"]),
        growth_rate=float(score["growth_rate"]),
        volatility=float(score["volatility"]),
        basis=str(score["basis"]),
        frac_beat=float(score["frac_beat"]),
        total_trades=total_trades,
        benchmark_geo_mean=float(result.oos_benchmark_geo_mean),
    )
    return StrategyAnalysis(
        file_name=strategy_path.name,
        strategy_name=strategy_spec["name"],
        market=market,
        tickers=tickers,
        summary=summary,
        dates=dates,
        cumulative_values=cumulative_values,
        cumulative_scores=cumulative_scores,
        fold_factors=fold_factors,
        benchmark_factors=benchmark_factors if market == "crypto" else [],
        hodl_values=hodl_values,
        hodl_scores=hodl_scores,
        hodl_factors=hodl_factors,
    )


def analyze_strategy(path: Path, args: argparse.Namespace) -> StrategyAnalysis:
    market = infer_market(path)
    tickers = args.crypto_tickers if market == "crypto" else args.equity_tickers
    strategy_spec = load_strategy_from_path(path)

    logger.info(
        f"=== Analyzing {path.name} | strategy={strategy_spec['name']} | "
        f"market={market} | tickers={tickers} ==="
    )
    result = trading.walk_forward(
        tickers,
        args.start,
        args.end,
        strategy_spec,
        train_days=args.train_days,
        test_days=args.test_days,
        step_days=args.step_days,
        num_retries=args.num_retries,
        max_evals=args.max_evals,
        market_mode=market,
    )
    dates, cumulative_values, cumulative_scores, fold_factors, benchmark_factors = build_curves(
        result,
        start_capital=args.start_capital,
        risk_lambda=args.risk_lambda,
    )
    hodl_values, hodl_scores, hodl_factors = build_hodl_curves(
        tickers,
        result,
        start=args.start,
        end=args.end,
        start_capital=args.start_capital,
        risk_lambda=args.risk_lambda,
    )
    analysis = summarize_result(
        path,
        strategy_spec,
        market,
        tickers,
        result,
        dates,
        cumulative_values,
        cumulative_scores,
        fold_factors,
        benchmark_factors,
        hodl_values,
        hodl_scores,
        hodl_factors,
        args.risk_lambda,
    )

    logger.info(result.summary(risk_lambda=args.risk_lambda))
    logger.info(
        f"Final current value={analysis.summary['final_value']:.2f} | "
        f"current log sharpe={analysis.summary['final_log_sharpe']:.4f} | "
        f"hodl value={analysis.hodl_values[-1]:.2f} | "
        f"hodl log sharpe={analysis.hodl_scores[-1]:.4f}"
    )
    return analysis


def write_summary(analyses: list[StrategyAnalysis], output_dir: Path) -> None:
    summary_path = output_dir / "summary.tsv"
    if not analyses:
        summary_path.write_text("", encoding="utf-8")
        return

    ordered_rows = sorted(
        (analysis.summary for analysis in analyses),
        key=lambda row: (row["market"], -row["score"], row["strategy_file"]),
    )
    columns = [
        "strategy_file",
        "strategy_name",
        "market",
        "tickers",
        "folds",
        "oos_geo_mean",
        "final_value",
        "final_log_sharpe",
        "score",
        "growth_rate",
        "volatility",
        "basis",
        "frac_beat",
        "total_trades",
        "benchmark_geo_mean",
    ]
    lines = ["\t".join(columns)]
    for row in ordered_rows:
        lines.append("\t".join(str(row[column]) for column in columns))
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    details_path = output_dir / "details.json"
    payload = [asdict(analysis) for analysis in analyses]
    details_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    logger.info(f"Wrote summary to {summary_path}")
    logger.info(f"Wrote detailed curves to {details_path}")


def cumulative_product(factors: list[float]) -> list[float]:
    value = 1.0
    out: list[float] = []
    for factor in factors:
        value *= float(factor)
        out.append(value)
    return out


def plot_market(
    analyses: list[StrategyAnalysis],
    market: str,
    value_path: Path,
    sharpe_path: Path,
) -> None:
    market_analyses = [analysis for analysis in analyses if analysis.market == market]
    if not market_analyses:
        logger.info(f"No {market} strategies found; skipping plots.")
        return

    fig, ax = plt.subplots(figsize=(12, 7))
    for analysis in market_analyses:
        ax.plot(
            analysis.dates,
            cumulative_product(analysis.fold_factors),
            marker="o",
            linewidth=2,
            label=analysis.file_name,
        )
    reference = market_analyses[0]
    ax.plot(
        reference.dates,
        cumulative_product(reference.hodl_factors),
        linestyle="--",
        linewidth=2.5,
        color="black",
        label="HODL",
    )
    ax.set_title(f"{market.title()} Strategies: Relative Value Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Relative value")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(value_path, dpi=160)
    plt.close(fig)
    logger.info(f"Wrote value plot to {value_path}")

    fig, ax = plt.subplots(figsize=(12, 7))
    for analysis in market_analyses:
        ax.plot(analysis.dates, analysis.cumulative_scores, marker="o", linewidth=2, label=analysis.file_name)
    ax.plot(
        reference.dates,
        reference.hodl_scores,
        linestyle="--",
        linewidth=2.5,
        color="black",
        label="HODL",
    )
    ax.set_title(f"{market.title()} Strategies: Current Log-Sharpe Over Time")
    ax.set_xlabel("Time")
    ax.set_ylabel("Current log sharpe")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(sharpe_path, dpi=160)
    plt.close(fig)
    logger.info(f"Wrote log-sharpe plot to {sharpe_path}")


def main() -> int:
    args = parse_args()
    results_dir = Path(args.results_dir)

    if not results_dir.exists():
        logger.error(f"Results directory does not exist: {results_dir}")
        return 1

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    configure_logging(output_dir)

    strategy_files = discover_strategy_files(results_dir)
    if not strategy_files:
        logger.error(f"No strategy .py files found in {results_dir}")
        return 1

    logger.info(f"Discovered {len(strategy_files)} strategy files in {results_dir}")
    analyses: list[StrategyAnalysis] = []

    for path in strategy_files:
        try:
            analyses.append(analyze_strategy(path, args))
        except Exception as exc:
            logger.exception(f"Failed to analyze {path.name}: {exc}")

    write_summary(analyses, output_dir)
    plot_market(
        analyses,
        market="equity",
        value_path=output_dir / "equity_value.png",
        sharpe_path=output_dir / "equity_log_sharpe.png",
    )
    plot_market(
        analyses,
        market="crypto",
        value_path=output_dir / "crypto_value.png",
        sharpe_path=output_dir / "crypto_log_sharpe.png",
    )

    if not analyses:
        logger.error("All strategy analyses failed.")
        return 1

    logger.info("Finished analyzing strategies.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
