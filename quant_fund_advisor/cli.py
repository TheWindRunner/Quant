"""Command line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .backtest import BacktestConfig, run_backtest
from .correlation import (
    best_lead_relationships,
    lagged_sector_correlations,
    transmission_signal,
)
from .data import (
    ASSET_TYPES,
    fetch_history,
    fetch_manifest_prices,
    load_market_manifest,
    load_price_csv,
)
from .demo import make_demo_prices
from .daily_report import run_daily_report
from .goal_research import (
    run_etf_guard_robustness,
    run_expanded_goal_research,
    run_corrected_execution_audit,
    run_goal_research,
    run_goal_live_signal,
)
from .model import latest_recommendations, score_assets
from .meta_strategy import run_meta_strategy
from .news import aggregate_news, fetch_cls_news
from .report import write_html_report
from .run_research import run_research


def run_demo(output_dir: Path, live_news: bool = False) -> None:
    us_prices, cn_prices = make_demo_prices()
    correlations = lagged_sector_correlations(us_prices, cn_prices)
    relationships = best_lead_relationships(correlations)
    transmission = transmission_signal(us_prices, relationships).reindex(cn_prices.index).ffill()

    news_scores = None
    if live_news:
        items = fetch_cls_news()
        news_scores = aggregate_news(items)["news_score"]

    scores = score_assets(cn_prices, transmission=transmission, news_scores=news_scores)
    recommendations = latest_recommendations(scores)
    result = run_backtest(cn_prices, scores, BacktestConfig())

    output_dir.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(output_dir / "recommendations.csv", encoding="utf-8-sig")
    relationships.to_csv(output_dir / "sector_correlations.csv", index=False, encoding="utf-8-sig")
    result["equity"].to_csv(output_dir / "backtest_equity.csv", encoding="utf-8-sig")
    result["weights"].to_csv(output_dir / "backtest_weights.csv", encoding="utf-8-sig")
    write_html_report(
        output_dir / "report.html",
        recommendations,
        relationships,
        result["equity"],
        result["metrics"],
        as_of=str(cn_prices.index[-1].date()),
    )
    print(f"Report: {(output_dir / 'report.html').resolve()}")
    print(pd.Series(result["metrics"]).to_string())


def run_prices(
    us_prices: pd.DataFrame, cn_prices: pd.DataFrame, output_dir: Path
) -> None:
    correlations = lagged_sector_correlations(us_prices, cn_prices)
    relationships = best_lead_relationships(correlations)
    transmission = transmission_signal(us_prices, relationships).reindex(cn_prices.index).ffill()
    scores = score_assets(cn_prices, transmission=transmission)
    recommendations = latest_recommendations(scores)
    result = run_backtest(cn_prices, scores)
    output_dir.mkdir(parents=True, exist_ok=True)
    recommendations.to_csv(
        output_dir / "recommendations.csv", encoding="utf-8-sig"
    )
    relationships.to_csv(
        output_dir / "sector_correlations.csv", index=False, encoding="utf-8-sig"
    )
    result["equity"].to_csv(
        output_dir / "backtest_equity.csv", encoding="utf-8-sig"
    )
    result["weights"].to_csv(
        output_dir / "backtest_weights.csv", encoding="utf-8-sig"
    )
    write_html_report(
        output_dir / "report.html",
        recommendations,
        relationships,
        result["equity"],
        result["metrics"],
        as_of=str(cn_prices.index[-1].date()),
    )
    print(recommendations.to_string())


def run_csv(us_path: Path, cn_path: Path, output_dir: Path) -> None:
    run_prices(load_price_csv(us_path), load_price_csv(cn_path), output_dir)


def run_live(
    manifest_path: Path,
    output_dir: Path,
    start: str | None,
    end: str | None,
) -> None:
    manifest = load_market_manifest(manifest_path)
    us_prices = fetch_manifest_prices(manifest, "us", start=start, end=end)
    cn_prices = fetch_manifest_prices(manifest, "cn", start=start, end=end)
    output_dir.mkdir(parents=True, exist_ok=True)
    us_prices.to_csv(output_dir / "us_prices.csv", encoding="utf-8-sig")
    cn_prices.to_csv(output_dir / "cn_prices.csv", encoding="utf-8-sig")
    run_prices(us_prices, cn_prices, output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fund sector rotation research toolkit")
    sub = parser.add_subparsers(dest="command", required=True)
    demo = sub.add_parser("demo", help="run a deterministic synthetic-data demo")
    demo.add_argument("--output", type=Path, default=Path("output"))
    demo.add_argument("--live-news", action="store_true")
    csv = sub.add_parser("csv", help="run with US and China sector price CSV files")
    csv.add_argument("--us", type=Path, required=True)
    csv.add_argument("--cn", type=Path, required=True)
    csv.add_argument("--output", type=Path, default=Path("output"))
    history = sub.add_parser("history", help="download one stock, ETF, or fund")
    history.add_argument("--type", choices=sorted(ASSET_TYPES), required=True)
    history.add_argument("--symbol", required=True)
    history.add_argument("--start")
    history.add_argument("--end")
    history.add_argument("--adjust", choices=["", "qfq", "hfq"], default="")
    history.add_argument("--output", type=Path)
    live = sub.add_parser("live", help="fetch a market manifest and run the workflow")
    live.add_argument(
        "--manifest", type=Path, default=Path("data/market_manifest.csv")
    )
    live.add_argument("--start")
    live.add_argument("--end")
    live.add_argument("--output", type=Path, default=Path("output"))
    daily = sub.add_parser(
        "daily", help="daily CPO, memory and PCB signals with six-month backtests"
    )
    daily.add_argument("--output", type=Path, default=Path("output/daily"))
    research = sub.add_parser(
        "research", help="run locked six-month multi-model fund research"
    )
    research.add_argument("--output", type=Path, default=Path("output/research"))
    meta = sub.add_parser(
        "meta-strategy", help="run six-fund meta-strategy robustness research"
    )
    meta.add_argument("--output", type=Path, default=Path("output/meta_strategy"))
    goal = sub.add_parser(
        "goal-research",
        help="validate candidates against trailing two-year and three-month goals",
    )
    goal.add_argument("--output", type=Path, default=Path("output/goal_research"))
    goal_expanded = sub.add_parser(
        "goal-research-expanded",
        help="validate the expanded active-tech fund goal strategy",
    )
    goal_expanded.add_argument(
        "--output", type=Path, default=Path("output/goal_research_expanded")
    )
    robustness = sub.add_parser(
        "goal-robustness",
        help="stress test the same-day ETF guard strategy",
    )
    robustness.add_argument(
        "--output", type=Path, default=Path("output/goal_research_robustness")
    )
    live_goal = sub.add_parser(
        "goal-live-signal",
        help="generate the executable 14:30 signal for the current goal strategy",
    )
    live_goal.add_argument("--output", type=Path, default=Path("output/goal_live_signal"))
    live_goal.add_argument("--cutoff", default="14:30")
    live_goal.add_argument("--threshold", type=float, default=-3.0)
    corrected = sub.add_parser(
        "goal-corrected-execution",
        help="audit ETF guard with correct open-fund same-day NAV execution",
    )
    corrected.add_argument(
        "--output", type=Path, default=Path("output/goal_research_corrected_execution")
    )
    corrected.add_argument("--threshold", type=float, default=-3.0)
    corrected.add_argument("--fee", type=float, default=0.003)
    args = parser.parse_args()
    if args.command == "demo":
        run_demo(args.output, args.live_news)
    elif args.command == "csv":
        run_csv(args.us, args.cn, args.output)
    elif args.command == "history":
        frame = fetch_history(
            args.type,
            args.symbol,
            start=args.start,
            end=args.end,
            adjust=args.adjust,
        )
        output = args.output or Path("data") / f"{args.type}_{args.symbol}.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(output, encoding="utf-8-sig")
        print(f"Saved {len(frame)} rows: {output.resolve()}")
    elif args.command == "live":
        run_live(args.manifest, args.output, args.start, args.end)
    elif args.command == "daily":
        print(run_daily_report(args.output).to_string(index=False))
    elif args.command == "meta-strategy":
        result = run_meta_strategy(args.output)
        print(result["param_aggregate"].head(10).to_string(index=False))
        print(f"Summary: {result['summary_path']}")
    elif args.command == "goal-research":
        result = run_goal_research(args.output)
        print(result["summary"].head(12).to_string(index=False))
        print(f"Report: {result['report_path']}")
    elif args.command == "goal-research-expanded":
        result = run_expanded_goal_research(args.output)
        print(result["summary"].head(12).to_string(index=False))
        print(f"Summary: {result['summary_path']}")
    elif args.command == "goal-robustness":
        result = run_etf_guard_robustness(args.output)
        print(result["detail"].head(20).to_string(index=False))
        print(f"Detail: {result['detail_path']}")
    elif args.command == "goal-live-signal":
        result = run_goal_live_signal(args.output, args.cutoff, args.threshold)
        print(result["detail"].to_string(index=False))
        print(f"Report: {result['report_path']}")
    elif args.command == "goal-corrected-execution":
        result = run_corrected_execution_audit(args.output, args.threshold, args.fee)
        print(result["summary"].to_string(index=False))
        print(f"Report: {result['report_path']}")
    else:
        result = run_research(args.output)
        print(result["locked_results"].to_string(index=False))
        print(f"Report: {result['report_path']}")


if __name__ == "__main__":
    main()
