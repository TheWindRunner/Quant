"""Low-turnover relative-strength rotation across technology fund themes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .portfolio_backtest import backtest_open_fund_portfolio
from .validation import purged_training_folds


@dataclass(frozen=True)
class RotationConfig:
    rebalance_days: int
    top_n: int
    strategic_core: float
    volatility_penalty: float = 0.10
    tail_risk_overlay: bool = False

    @property
    def name(self) -> str:
        return (
            f"rotation_{self.rebalance_days}d_top{self.top_n}_"
            f"core{int(self.strategic_core * 100)}"
            f"{'_tail' if self.tail_risk_overlay else ''}"
        )


def _cap_and_redistribute(
    weights: pd.Series,
    cap: float,
) -> pd.Series:
    """Cap concentration and redistribute excess among uncapped assets."""
    result = weights.clip(lower=0.0)
    if result.sum() <= 0:
        return pd.Series(1.0 / len(result), index=result.index)
    result = result / result.sum()
    for _ in range(len(result)):
        excess = float((result - cap).clip(lower=0.0).sum())
        result = result.clip(upper=cap)
        if excess <= 1e-12:
            break
        room = (cap - result).clip(lower=0.0)
        if room.sum() <= 1e-12:
            break
        result += excess * room / room.sum()
    return result / result.sum()


def risk_budget_momentum_weights(
    navs: pd.DataFrame,
    rebalance_days: int = 20,
    volatility_window: int = 60,
    momentum_window: int = 60,
    momentum_tilt: float = 0.35,
    maximum_weight: float = 0.60,
) -> pd.DataFrame:
    """Monthly inverse-volatility allocation with a modest momentum tilt."""
    returns = navs.pct_change(fill_method=None)
    volatility = returns.rolling(volatility_window).std() * np.sqrt(252)
    momentum = navs.pct_change(momentum_window, fill_method=None)
    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    current = pd.Series(1.0 / len(navs.columns), index=navs.columns)
    for location, date in enumerate(navs.index):
        if location % rebalance_days == 0 and location >= volatility_window:
            inverse_vol = 1.0 / volatility.loc[date].replace(0.0, np.nan)
            rank = momentum.loc[date].rank(pct=True).fillna(0.5)
            tilted = inverse_vol * (1.0 + momentum_tilt * (rank - 0.5))
            if tilted.notna().all() and tilted.sum() > 0:
                current = _cap_and_redistribute(tilted, maximum_weight)
        weights.loc[date] = current
    return weights.mul(portfolio_tail_risk_exposure(navs), axis=0)


def rotation_configs() -> list[RotationConfig]:
    """Monthly/biweekly designs consistent with 1-3 month fund holding."""
    base = [
        RotationConfig(10, 1, 0.60),
        RotationConfig(10, 2, 0.60),
        RotationConfig(20, 1, 0.60),
        RotationConfig(20, 2, 0.60),
        RotationConfig(20, 2, 0.75),
    ]
    return base + [
        RotationConfig(
            config.rebalance_days,
            config.top_n,
            config.strategic_core,
            config.volatility_penalty,
            True,
        )
        for config in base
    ]


def portfolio_tail_risk_exposure(navs: pd.DataFrame) -> pd.Series:
    """Daily defensive overlay requiring trend, drawdown and volatility stress."""
    composite = navs.div(navs.iloc[0]).mean(axis=1)
    returns = composite.pct_change(fill_method=None)
    ma60 = composite.rolling(60).mean()
    ma120 = composite.rolling(120).mean()
    drawdown = composite / composite.rolling(120, min_periods=20).max() - 1
    volatility = returns.rolling(20).std() * np.sqrt(252)
    high_volatility = volatility > volatility.rolling(
        252, min_periods=120
    ).quantile(0.75)
    defensive = (
        (composite < ma120)
        & (drawdown < -0.10)
        & high_volatility
    )
    recovered = (composite > ma60) & (drawdown > -0.06)
    state = 1.0
    exposure = []
    for date in navs.index:
        if defensive.loc[date]:
            state = 0.50
        elif recovered.loc[date]:
            state = 1.0
        exposure.append(state)
    return pd.Series(exposure, index=navs.index, name="portfolio_exposure")


def relative_strength_weights(
    navs: pd.DataFrame,
    config: RotationConfig,
) -> pd.DataFrame:
    """Combine 1/3/6-month momentum, trend and volatility without lookahead."""
    returns = navs.pct_change(fill_method=None)
    score = (
        0.45 * navs.pct_change(20, fill_method=None)
        + 0.35 * navs.pct_change(60, fill_method=None)
        + 0.20 * navs.pct_change(120, fill_method=None)
        - config.volatility_penalty
        * returns.rolling(20).std()
        * np.sqrt(252)
    )
    eligible = (
        (navs.pct_change(60, fill_method=None) > 0)
        & (navs > navs.rolling(120).mean())
    )
    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    current = pd.Series(0.0, index=navs.columns)
    equal_core = config.strategic_core / len(navs.columns)
    for location, date in enumerate(navs.index):
        if location % config.rebalance_days == 0 and location >= 120:
            available = score.loc[date].where(eligible.loc[date]).dropna()
            current = pd.Series(equal_core, index=navs.columns)
            if not available.empty:
                selected = available.nlargest(config.top_n).index
                tactical = (1.0 - config.strategic_core) / len(selected)
                current.loc[selected] += tactical
        weights.loc[date] = current
    if config.tail_risk_overlay:
        weights = weights.mul(portfolio_tail_risk_exposure(navs), axis=0)
    return weights


def equal_weight_targets(navs: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        1.0 / len(navs.columns),
        index=navs.index,
        columns=navs.columns,
    )


def evaluate_rotation(
    navs: pd.DataFrame,
    targets: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, float]:
    sliced_navs = navs.loc[(navs.index >= start) & (navs.index <= end)]
    prior = targets.loc[targets.index < start]
    sliced_targets = targets.reindex(sliced_navs.index).ffill().fillna(0.0)
    return backtest_open_fund_portfolio(
        sliced_navs,
        sliced_targets,
        initial_weights=prior.iloc[-1] if len(prior) else None,
        liquidate_at_end=True,
    )["metrics"]


def select_rotation_on_training(
    navs: pd.DataFrame,
    test_start: pd.Timestamp,
) -> tuple[RotationConfig | None, pd.DataFrame]:
    """Select one rotation configuration across purged pre-test folds."""
    rows = []
    benchmark_targets = equal_weight_targets(navs)
    for fold_start, fold_end in purged_training_folds(navs.index, test_start):
        benchmark = evaluate_rotation(
            navs, benchmark_targets, fold_start, fold_end
        )
        for config in rotation_configs():
            metrics = evaluate_rotation(
                navs,
                relative_strength_weights(navs, config),
                fold_start,
                fold_end,
            )
            excess = metrics["total_return"] - benchmark["total_return"]
            drawdown_improvement = (
                metrics["max_drawdown"] - benchmark["max_drawdown"]
            )
            objective = (
                excess
                + 0.60 * drawdown_improvement
                + 0.03 * np.clip(metrics["sharpe"], -2, 2)
                - 0.001 * metrics["trade_count"]
                - 0.02 * metrics["under_30_day_redemption_ratio"]
            )
            rows.append(
                {
                    "fold_start": fold_start,
                    "fold_end": fold_end,
                    "model": config.name,
                    "objective": objective,
                    "excess_return": excess,
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
            median_excess_return=("excess_return", "median"),
            median_drawdown_improvement=("drawdown_improvement", "median"),
            median_holding_days=("average_holding_days", "median"),
            median_under_30_ratio=("under_30_day_redemption_ratio", "median"),
            worst_fold_excess=("excess_return", "min"),
        )
        .sort_values(
            ["positive_fold_ratio", "median_objective"],
            ascending=False,
        )
    )
    eligible = ranking[
        (ranking["positive_fold_ratio"] >= 0.50)
        & (ranking["median_holding_days"] >= 60)
        & (ranking["median_under_30_ratio"] <= 0.10)
        & (ranking["worst_fold_excess"] >= -0.20)
        & (
            (ranking["median_excess_return"] >= 0)
            | (
                (ranking["median_excess_return"] >= -0.02)
                & (ranking["median_drawdown_improvement"] >= 0.03)
            )
        )
    ]
    if eligible.empty:
        return None, diagnostics
    chosen_name = eligible.index[0]
    chosen = next(
        config for config in rotation_configs() if config.name == chosen_name
    )
    return chosen, diagnostics


def locked_rotation_comparison(
    navs: pd.DataFrame,
    config: RotationConfig | None,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> pd.DataFrame:
    benchmark = evaluate_rotation(
        navs,
        equal_weight_targets(navs),
        test_start,
        test_end,
    )
    if config is None:
        candidate = benchmark.copy()
        model = "no_qualified_rotation"
    else:
        candidate = evaluate_rotation(
            navs,
            relative_strength_weights(navs, config),
            test_start,
            test_end,
        )
        model = config.name
    return pd.DataFrame(
        [
            {"model": "equal_weight_buy_hold", **benchmark},
            {"model": model, **candidate},
        ]
    )


def walk_forward_rotation_validation(
    navs: pd.DataFrame,
    final_test_start: pd.Timestamp,
) -> pd.DataFrame:
    """Re-select using older data, then freeze the model for each next fold."""
    rows = []
    evaluation_folds = purged_training_folds(navs.index, final_test_start)
    for fold_start, fold_end in evaluation_folds:
        try:
            selected, _ = select_rotation_on_training(navs, fold_start)
        except ValueError:
            continue
        benchmark = evaluate_rotation(
            navs,
            equal_weight_targets(navs),
            fold_start,
            fold_end,
        )
        if selected is None:
            candidate = benchmark.copy()
            model = "no_qualified_rotation"
        else:
            candidate = evaluate_rotation(
                navs,
                relative_strength_weights(navs, selected),
                fold_start,
                fold_end,
            )
            model = selected.name
        rows.append(
            {
                "fold_start": fold_start,
                "fold_end": fold_end,
                "selected_model": model,
                "benchmark_return": benchmark["total_return"],
                "candidate_return": candidate["total_return"],
                "excess_return": (
                    candidate["total_return"] - benchmark["total_return"]
                ),
                "benchmark_drawdown": benchmark["max_drawdown"],
                "candidate_drawdown": candidate["max_drawdown"],
                "drawdown_improvement": (
                    candidate["max_drawdown"] - benchmark["max_drawdown"]
                ),
                "candidate_sharpe": candidate["sharpe"],
                "trade_count": candidate["trade_count"],
                "average_holding_days": candidate["average_holding_days"],
                "under_30_day_redemption_ratio": candidate[
                    "under_30_day_redemption_ratio"
                ],
            }
        )
    return pd.DataFrame(rows)


def fixed_strategy_fold_validation(
    navs: pd.DataFrame,
    targets: pd.DataFrame,
    final_test_start: pd.Timestamp,
    model_name: str,
) -> pd.DataFrame:
    """Evaluate one predeclared strategy on each historical six-month fold."""
    rows = []
    benchmark_targets = equal_weight_targets(navs)
    for fold_start, fold_end in purged_training_folds(
        navs.index, final_test_start
    ):
        benchmark = evaluate_rotation(
            navs, benchmark_targets, fold_start, fold_end
        )
        candidate = evaluate_rotation(navs, targets, fold_start, fold_end)
        rows.append(
            {
                "fold_start": fold_start,
                "fold_end": fold_end,
                "selected_model": model_name,
                "benchmark_return": benchmark["total_return"],
                "candidate_return": candidate["total_return"],
                "excess_return": (
                    candidate["total_return"] - benchmark["total_return"]
                ),
                "benchmark_drawdown": benchmark["max_drawdown"],
                "candidate_drawdown": candidate["max_drawdown"],
                "drawdown_improvement": (
                    candidate["max_drawdown"] - benchmark["max_drawdown"]
                ),
                "candidate_sharpe": candidate["sharpe"],
                "trade_count": candidate["trade_count"],
                "average_holding_days": candidate["average_holding_days"],
                "under_30_day_redemption_ratio": candidate[
                    "under_30_day_redemption_ratio"
                ],
            }
        )
    return pd.DataFrame(rows)
