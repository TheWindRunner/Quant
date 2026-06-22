"""Unified evaluation for the model zoo on a locked test period."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .daily_strategy import backtest_daily
from .fund_backtest import FundFeeSchedule, backtest_open_fund
from .execution_policy import apply_open_fund_execution_policy
from .model_zoo import ModelSignal, build_model_zoo
from .model_zoo import robust_ensemble
from .validation import purged_training_folds


def evaluate_signal(
    nav: pd.Series,
    signal: ModelSignal,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, float | str]:
    test_nav = nav.loc[(nav.index >= start) & (nav.index <= end)]
    practical_position = apply_open_fund_execution_policy(signal.position)
    decisions = pd.DataFrame(
        {
            "target_position": practical_position.reindex(test_nav.index)
            .ffill()
            .fillna(0.0)
        }
    )
    prior_signal = practical_position.loc[practical_position.index < start]
    initial_target = float(prior_signal.iloc[-1]) if len(prior_signal) else 0.0
    result = backtest_open_fund(
        test_nav,
        decisions["target_position"],
        schedule=FundFeeSchedule(),
        initial_target=initial_target,
        liquidate_at_end=True,
    )
    return {"model": signal.name, **result["metrics"]}


def evaluate_model_zoo(
    datasets: dict[str, dict[str, pd.Series | pd.DataFrame | None]],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    rows = []
    for asset, dataset in datasets.items():
        nav = dataset["nav"]
        if not isinstance(nav, pd.Series):
            raise TypeError(f"{asset}.nav must be a Series")
        models = build_model_zoo(
            nav,
            market_nav=dataset.get("market_nav"),
            peers=dataset.get("peers"),
            external_factors=dataset.get("external_factors"),
        )
        for model in models:
            rows.append(
                {
                    "asset": asset,
                    **evaluate_signal(nav, model, test_start, test_end),
                }
            )
    return pd.DataFrame(rows)


def robustness_summary(results: pd.DataFrame) -> pd.DataFrame:
    benchmark = (
        results[results["model"] == "buy_hold"]
        .set_index("asset")[["total_return", "max_drawdown"]]
        .rename(
            columns={
                "total_return": "benchmark_return",
                "max_drawdown": "benchmark_drawdown",
            }
        )
    )
    merged = results.join(benchmark, on="asset")
    merged["excess_return"] = merged["total_return"] - merged["benchmark_return"]
    merged["drawdown_improvement"] = (
        merged["max_drawdown"] - merged["benchmark_drawdown"]
    )
    return (
        merged.groupby("model")
        .agg(
            median_return=("total_return", "median"),
            median_excess_return=("excess_return", "median"),
            assets_beating_benchmark=("excess_return", lambda x: int((x > 0).sum())),
            median_drawdown_improvement=("drawdown_improvement", "median"),
            median_sharpe=("sharpe", "median"),
            median_trade_count=("trade_count", "median"),
        )
        .sort_values(
            ["assets_beating_benchmark", "median_excess_return"],
            ascending=False,
        )
    )


def select_ensemble_on_training(
    datasets: dict[str, dict[str, pd.Series | pd.DataFrame | None]],
    test_start: pd.Timestamp,
    top_n: int = 3,
) -> tuple[list[str], pd.DataFrame]:
    """Rank fixed model families using only purged pre-test folds."""
    rows = []
    for asset, dataset in datasets.items():
        nav = dataset["nav"]
        if not isinstance(nav, pd.Series):
            raise TypeError(f"{asset}.nav must be a Series")
        models = build_model_zoo(
            nav,
            market_nav=dataset.get("market_nav"),
            peers=dataset.get("peers"),
        )
        models = [
            model for model in models
            if model.name not in {"buy_hold", "robust_ensemble"}
        ]
        for fold_start, fold_end in purged_training_folds(nav.index, test_start):
            benchmark_model = next(
                model for model in build_model_zoo(
                    nav,
                    market_nav=dataset.get("market_nav"),
                    peers=dataset.get("peers"),
                    external_factors=dataset.get("external_factors"),
                )
                if model.name == "buy_hold"
            )
            benchmark = evaluate_signal(
                nav, benchmark_model, fold_start, fold_end
            )
            for model in models:
                metrics = evaluate_signal(nav, model, fold_start, fold_end)
                excess_return = (
                    float(metrics["total_return"])
                    - float(benchmark["total_return"])
                )
                drawdown_improvement = (
                    float(metrics["max_drawdown"])
                    - float(benchmark["max_drawdown"])
                )
                objective = (
                    excess_return
                    + 0.60 * drawdown_improvement
                    + 0.03 * np.clip(float(metrics["sharpe"]), -2, 2)
                    - 0.001 * float(metrics["trade_count"])
                    - 0.02 * float(metrics.get("under_30_day_redemption_ratio", 0.0))
                )
                rows.append(
                    {
                        "asset": asset,
                        "fold_start": fold_start,
                        "fold_end": fold_end,
                        "model": model.name,
                        "objective": objective,
                        "benchmark_return": benchmark["total_return"],
                        "excess_return": excess_return,
                        "drawdown_improvement": drawdown_improvement,
                        **metrics,
                    }
                )
    diagnostics = pd.DataFrame(rows)
    ranking = (
        diagnostics.groupby("model")
        .agg(
            median_objective=("objective", "median"),
            positive_fold_ratio=("objective", lambda x: float((x > 0).mean())),
            worst_fold=("objective", "min"),
            median_holding_days=("average_holding_days", "median"),
            median_under_30_ratio=("under_30_day_redemption_ratio", "median"),
            median_excess_return=("excess_return", "median"),
            median_drawdown_improvement=("drawdown_improvement", "median"),
        )
        .sort_values(
            ["positive_fold_ratio", "median_objective", "worst_fold"],
            ascending=False,
        )
    )
    eligible = ranking[
        (ranking["positive_fold_ratio"] >= 0.50)
        & (ranking["median_holding_days"] >= 14)
        & (ranking["median_under_30_ratio"] <= 0.65)
        & (
            (ranking["median_excess_return"] >= 0)
            | (
                (ranking["median_excess_return"] >= -0.03)
                & (ranking["median_drawdown_improvement"] >= 0.03)
            )
        )
    ]
    source = eligible
    ordered = source.index.tolist()

    # Avoid an ensemble made of near-duplicate signals with different names.
    selected: list[str] = []
    for candidate in ordered:
        if not selected:
            selected.append(candidate)
        else:
            correlations = []
            for asset, dataset in datasets.items():
                nav = dataset["nav"]
                if not isinstance(nav, pd.Series):
                    continue
                zoo = {
                    model.name: model.position
                    for model in build_model_zoo(
                        nav,
                        market_nav=dataset.get("market_nav"),
                        peers=dataset.get("peers"),
                        external_factors=dataset.get("external_factors"),
                    )
                }
                for chosen in selected:
                    if candidate in zoo and chosen in zoo:
                        correlations.append(
                            abs(zoo[candidate].corr(zoo[chosen]))
                        )
            if correlations and max(correlations) < 0.95:
                selected.append(candidate)
        if len(selected) >= top_n:
            break
    return selected, diagnostics


def evaluate_locked_selected_ensemble(
    datasets: dict[str, dict[str, pd.Series | pd.DataFrame | None]],
    selected_models: list[str],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    """Evaluate buy-hold, MA and the frozen training-selected ensemble."""
    rows = []
    for asset, dataset in datasets.items():
        nav = dataset["nav"]
        if not isinstance(nav, pd.Series):
            raise TypeError(f"{asset}.nav must be a Series")
        zoo = build_model_zoo(
            nav,
            market_nav=dataset.get("market_nav"),
            peers=dataset.get("peers"),
            external_factors=dataset.get("external_factors"),
        )
        by_name = {model.name: model for model in zoo}
        chosen = [by_name[name] for name in selected_models if name in by_name]
        if not chosen:
            ensemble = ModelSignal(
                "no_qualified_model",
                by_name["buy_hold"].position,
                "No training-qualified tactical model; retain benchmark",
            )
        else:
            ensemble = robust_ensemble(chosen)
            ensemble = ModelSignal(
                "trained_ensemble",
                ensemble.position,
                "Training-selected equal-weight median ensemble",
            )
        for model_name in ("buy_hold", "dual_ma"):
            rows.append(
                {
                    "asset": asset,
                    **evaluate_signal(nav, by_name[model_name], test_start, test_end),
                }
            )
        rows.append(
            {
                "asset": asset,
                **evaluate_signal(nav, ensemble, test_start, test_end),
            }
        )
    return pd.DataFrame(rows)


def select_theme_models_on_training(
    datasets: dict[str, dict[str, pd.Series | pd.DataFrame | None]],
    test_start: pd.Timestamp,
) -> tuple[dict[str, str], pd.DataFrame]:
    """Select at most one suitable model per theme using pre-test folds only."""
    _, diagnostics = select_ensemble_on_training(
        datasets,
        test_start,
        top_n=1,
    )
    aggregate = (
        diagnostics.groupby("model")
        .agg(
            cross_theme_median_excess=("excess_return", "median"),
            cross_theme_worst_excess=("excess_return", lambda x: float(x.quantile(0.10))),
        )
    )
    selected: dict[str, str] = {}
    selection_rows = []
    for asset, asset_rows in diagnostics.groupby("asset"):
        summary = (
            asset_rows.groupby("model")
            .agg(
                median_objective=("objective", "median"),
                positive_fold_ratio=("objective", lambda x: float((x > 0).mean())),
                median_excess_return=("excess_return", "median"),
                median_drawdown_improvement=("drawdown_improvement", "median"),
                median_holding_days=("average_holding_days", "median"),
                median_under_30_ratio=("under_30_day_redemption_ratio", "median"),
                worst_fold_excess=("excess_return", "min"),
            )
            .join(aggregate)
        )
        suitable = summary[
            (summary["positive_fold_ratio"] >= 0.50)
            & (summary["median_holding_days"] >= 14)
            & (summary["median_under_30_ratio"] <= 0.65)
            & (summary["cross_theme_median_excess"] >= -0.05)
            & (summary["cross_theme_worst_excess"] >= -0.30)
            & (
                (summary["median_excess_return"] >= 0)
                | (
                    (summary["median_excess_return"] >= -0.03)
                    & (summary["median_drawdown_improvement"] >= 0.03)
                )
            )
        ].sort_values(
            [
                "positive_fold_ratio",
                "median_objective",
                "median_excess_return",
            ],
            ascending=False,
        )
        chosen = suitable.index[0] if not suitable.empty else "buy_hold"
        selected[asset] = chosen
        selected_summary = (
            summary.loc[chosen].to_dict()
            if chosen in summary.index
            else {}
        )
        selection_rows.append(
            {
                "asset": asset,
                "selected_model": chosen,
                "eligible_model_count": int(len(suitable)),
                **selected_summary,
            }
        )
    return selected, pd.DataFrame(selection_rows)


def evaluate_locked_theme_models(
    datasets: dict[str, dict[str, pd.Series | pd.DataFrame | None]],
    selected_models: dict[str, str],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    """Evaluate frozen per-theme selections beside buy-hold and dual MA."""
    rows = []
    for asset, dataset in datasets.items():
        nav = dataset["nav"]
        if not isinstance(nav, pd.Series):
            raise TypeError(f"{asset}.nav must be a Series")
        zoo = {
            model.name: model
            for model in build_model_zoo(
                nav,
                market_nav=dataset.get("market_nav"),
                peers=dataset.get("peers"),
                external_factors=dataset.get("external_factors"),
            )
        }
        selected_name = selected_models.get(asset, "buy_hold")
        selected = zoo.get(selected_name, zoo["buy_hold"])
        frozen = ModelSignal(
            "theme_selected",
            selected.position,
            f"Training-selected theme model: {selected.name}",
        )
        for signal in (zoo["buy_hold"], zoo["dual_ma"], frozen):
            row = {
                "asset": asset,
                "selected_model": (
                    selected.name if signal.name == "theme_selected" else signal.name
                ),
                **evaluate_signal(nav, signal, test_start, test_end),
            }
            rows.append(row)
    return pd.DataFrame(rows)
