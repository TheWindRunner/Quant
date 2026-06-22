"""Walk-forward model selection and locked six-month out-of-sample tests."""

from __future__ import annotations

from dataclasses import replace
from itertools import product

import numpy as np
import pandas as pd

from .daily_strategy import backtest_daily, decisions_from_multifactor
from .multifactor import MultiFactorConfig, score_factor_panel


def candidate_configs() -> list[MultiFactorConfig]:
    """Small, pre-declared search space to limit data-mining."""
    configs = []
    trend_momentum = ((0.16, 0.14), (0.20, 0.16), (0.22, 0.18))
    risk_breadth = ((0.10, 0.10), (0.12, 0.12), (0.14, 0.10))
    thresholds = ((0.20, 0.45), (0.25, 0.50), (0.30, 0.55))
    for (trend, momentum), (quality, breadth), (buy, strong) in product(
        trend_momentum, risk_breadth, thresholds
    ):
        residual = 1.0 - trend - momentum - quality - breadth
        weights = {
            "trend": trend,
            "momentum": momentum,
            "quality": quality,
            "breadth": breadth,
            "cross_market": residual * 0.25,
            "risk_appetite": residual * 0.20,
            "news": residual * 0.10,
            "flow": residual * 0.20,
            "structure": residual * 0.25,
        }
        configs.append(
            MultiFactorConfig(
                weights=weights,
                buy_score=buy,
                strong_buy_score=strong,
            )
        )
    return configs


def _evaluate(
    nav: pd.Series,
    factor_panel: pd.DataFrame,
    config: MultiFactorConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, float]:
    scored = score_factor_panel(factor_panel, config)
    decisions = decisions_from_multifactor(scored, config.max_theme_weight)
    sliced_nav = nav.loc[(nav.index >= start) & (nav.index <= end)]
    sliced_decisions = decisions.reindex(sliced_nav.index).ffill()
    result = backtest_daily(sliced_nav, sliced_decisions, start=start)
    metrics = result["metrics"]
    return {
        "total_return": float(metrics.get("total_return", 0.0)),
        "max_drawdown": float(metrics.get("max_drawdown", 0.0)),
        "sharpe": float(metrics.get("sharpe", 0.0)),
        "trade_count": float(metrics.get("trade_count", 0.0)),
    }


def purged_training_folds(
    index: pd.DatetimeIndex,
    test_start: pd.Timestamp,
    fold_months: int = 6,
    purge_days: int = 20,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Non-overlapping pre-test folds with a purge gap before the test."""
    eligible = index[index < test_start - pd.Timedelta(days=purge_days)]
    if len(eligible) < 252:
        raise ValueError("At least one year of pre-test history is required")
    fold_end = eligible.max()
    folds = []
    while True:
        fold_start = fold_end - pd.DateOffset(months=fold_months)
        if fold_start < eligible.min():
            break
        folds.append((pd.Timestamp(fold_start), pd.Timestamp(fold_end)))
        fold_end = fold_start - pd.Timedelta(days=1)
    if len(folds) < 2:
        raise ValueError("At least two training folds are required")
    return list(reversed(folds))


def select_config_on_training(
    datasets: dict[str, tuple[pd.Series, pd.DataFrame]],
    test_start: pd.Timestamp,
    configs: list[MultiFactorConfig] | None = None,
) -> tuple[MultiFactorConfig, pd.DataFrame]:
    """Choose one config across funds and pre-test folds only."""
    configs = configs or candidate_configs()
    rows = []
    for config_id, config in enumerate(configs):
        fold_scores = []
        for asset, (nav, panel) in datasets.items():
            for fold_start, fold_end in purged_training_folds(
                nav.index, test_start
            ):
                metrics = _evaluate(nav, panel, config, fold_start, fold_end)
                # Prefer robust risk-adjusted returns; cap Sharpe contribution.
                objective = (
                    metrics["total_return"]
                    + 0.50 * max(metrics["max_drawdown"], -0.30)
                    + 0.03 * np.clip(metrics["sharpe"], -2, 2)
                    - 0.001 * metrics["trade_count"]
                )
                fold_scores.append(objective)
                rows.append(
                    {
                        "config_id": config_id,
                        "asset": asset,
                        "fold_start": fold_start,
                        "fold_end": fold_end,
                        "objective": objective,
                        **metrics,
                    }
                )
        median_score = float(np.median(fold_scores))
        rows.append(
            {
                "config_id": config_id,
                "asset": "__aggregate__",
                "objective": median_score,
            }
        )
    diagnostics = pd.DataFrame(rows)
    aggregates = diagnostics[diagnostics["asset"] == "__aggregate__"]
    best_id = int(aggregates.loc[aggregates["objective"].idxmax(), "config_id"])
    return configs[best_id], diagnostics


def locked_test_comparison(
    datasets: dict[str, tuple[pd.Series, pd.DataFrame]],
    config: MultiFactorConfig,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    """Compare buy-hold, MA and multifactor without changing the chosen config."""
    rows = []
    for asset, (nav, panel) in datasets.items():
        test_nav = nav.loc[(nav.index >= test_start) & (nav.index <= test_end)]
        returns = test_nav.pct_change(fill_method=None).fillna(0.0)
        buy_hold_equity = (1 + returns).cumprod()
        buy_hold_drawdown = buy_hold_equity / buy_hold_equity.cummax() - 1
        rows.append(
            {
                "asset": asset,
                "strategy": "buy_hold",
                "total_return": float(buy_hold_equity.iloc[-1] - 1),
                "max_drawdown": float(buy_hold_drawdown.min()),
                "trade_count": 1.0,
            }
        )

        ma20 = nav.rolling(20).mean()
        ma60 = nav.rolling(60).mean()
        ma_position = ((nav > ma20) & (ma20 > ma60)).astype(float)
        ma_decisions = pd.DataFrame({"target_position": ma_position}, index=nav.index)
        ma_nav = nav.loc[(nav.index >= test_start) & (nav.index <= test_end)]
        ma_test_decisions = ma_decisions.reindex(ma_nav.index)
        ma_result = backtest_daily(ma_nav, ma_test_decisions, start=test_start)
        rows.append(
            {"asset": asset, "strategy": "ma20_60", **ma_result["metrics"]}
        )

        multi_metrics = _evaluate(nav, panel, config, test_start, test_end)
        rows.append(
            {"asset": asset, "strategy": "multifactor", **multi_metrics}
        )
    return pd.DataFrame(rows)


def model_acceptance(comparison: pd.DataFrame) -> dict[str, object]:
    """Accept changes only when they generalize beyond the CPO test asset."""
    pivot_return = comparison.pivot(
        index="asset", columns="strategy", values="total_return"
    )
    pivot_drawdown = comparison.pivot(
        index="asset", columns="strategy", values="max_drawdown"
    )
    excess = pivot_return["multifactor"] - pivot_return["buy_hold"]
    drawdown_improvement = (
        pivot_drawdown["multifactor"] - pivot_drawdown["buy_hold"]
    )
    majority = int((excess > 0).sum()) >= max(2, int(np.ceil(len(excess) * 0.6)))
    risk_adjusted_majority = int(
        ((excess > -0.02) & (drawdown_improvement > 0.03)).sum()
    ) >= max(2, int(np.ceil(len(excess) * 0.6)))
    return {
        "accepted": bool(majority or risk_adjusted_majority),
        "assets_beating_buy_hold": int((excess > 0).sum()),
        "asset_count": int(len(excess)),
        "median_excess_return": float(excess.median()),
        "median_drawdown_improvement": float(drawdown_improvement.median()),
    }
