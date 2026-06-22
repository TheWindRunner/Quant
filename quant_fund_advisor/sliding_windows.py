"""Sliding-window strategy comparison for entry-time robustness."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .fund_backtest import backtest_open_fund
from .style_timing import (
    classify_style,
    entry_percentile,
    initial_position_from_entry_risk,
    read_cached_nav,
    read_external_factors,
    routed_config,
    style_timing_target,
)


def all_in_target(index: pd.Index) -> pd.Series:
    return pd.Series(1.0, index=index, name="all_in")


def dca_target(
    index: pd.Index,
    days: int = 20,
    start: pd.Timestamp | None = None,
) -> pd.Series:
    values = np.zeros(len(index), dtype=float)
    start_loc = 0
    if start is not None:
        matches = np.flatnonzero(index >= start)
        start_loc = int(matches[0]) if len(matches) else len(index)
    if start_loc < len(index):
        ramp = np.minimum(1.0, (np.arange(len(index) - start_loc) + 1) / max(days, 1))
        values[start_loc:] = ramp
    return pd.Series(values, index=index, name=f"dca_{days}")


def daily_dca_metrics(nav: pd.Series, purchase_fee_rate: float = 0.0015) -> dict[str, float]:
    """Equal cash contribution every NAV date, normalized to committed capital."""
    nav = nav.dropna().sort_index()
    if nav.empty:
        raise ValueError("NAV is empty")
    contribution = 1.0 / len(nav)
    units = 0.0
    values = []
    invested = 0.0
    for price in nav:
        units += contribution * (1 - purchase_fee_rate) / float(price)
        invested += contribution
        values.append(units * float(price) / invested)
    equity = pd.Series(values, index=nav.index)
    returns = equity.pct_change(fill_method=None).fillna(0.0)
    from .backtest import performance_metrics

    metrics = performance_metrics(returns)
    metrics.update(
        {
            "trade_count": float(len(nav)),
            "purchase_fees": float(purchase_fee_rate),
            "redemption_fees": 0.0,
            "total_fees": 0.0,
            "invested_day_ratio": 1.0,
            "average_holding_days": 0.0,
            "under_7_day_redemption_ratio": 0.0,
            "under_30_day_redemption_ratio": 0.0,
        }
    )
    return metrics


def entry_control_target(
    nav: pd.Series,
    factors: pd.DataFrame,
    start: pd.Timestamp,
) -> pd.Series:
    initial = initial_position_from_entry_risk(nav, factors, start)
    target = pd.Series(1.0, index=nav.index, name="entry_control")
    window = target.loc[target.index >= start].index[:20]
    if initial < 1.0 and len(window):
        ramp = np.linspace(initial, 1.0, len(window))
        target.loc[window] = ramp
    return target


def high_entry_dca_target(
    nav: pd.Series,
    start: pd.Timestamp,
    threshold: float = 0.975,
    days: int = 20,
) -> pd.Series:
    if entry_percentile(nav, start, 120) >= threshold:
        return dca_target(nav.index, days, start).rename(f"high_entry_dca_{threshold:.3f}")
    return all_in_target(nav.index).rename(f"high_entry_dca_{threshold:.3f}")


def simple_timing_target(
    nav: pd.Series,
    factors: pd.DataFrame,
    core: float = 0.9,
) -> pd.Series:
    factors = factors.reindex(nav.index).ffill()
    etf_change = factors.get("etf_change_pct", pd.Series(0.0, index=nav.index)).fillna(0.0)
    amount = factors.get("etf_amount", pd.Series(np.nan, index=nav.index))
    amount_ratio = (amount / amount.rolling(20).mean()).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    ma5 = nav.rolling(5).mean()
    drawdown10 = nav / nav.rolling(10).max() - 1
    current = 1.0
    cooldown = 0
    targets = []
    for date in nav.index:
        if cooldown > 0:
            cooldown -= 1
        overheat = etf_change.loc[date] >= 5 and amount_ratio.loc[date] >= 1.3 and nav.loc[date] >= ma5.loc[date]
        buyback = etf_change.loc[date] <= -3 or drawdown10.loc[date] <= -0.03
        if current >= 0.999 and cooldown == 0 and overheat:
            current = core
            cooldown = 1
        elif current < 0.999 and buyback:
            current = 1.0
            cooldown = 1
        targets.append(current)
    return pd.Series(targets, index=nav.index, name="simple_t")


def trend_quality_guard_target(nav: pd.Series) -> pd.Series:
    log_nav = np.log(nav)

    def score(values: np.ndarray) -> float:
        if len(values) < 20 or not np.isfinite(values).all():
            return np.nan
        x = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(x, values, 1)
        fitted = slope * x + intercept
        total = ((values - values.mean()) ** 2).sum()
        residual = ((values - fitted) ** 2).sum()
        r_squared = 1 - residual / total if total > 0 else 0.0
        return float(np.expm1(slope * 252) * max(0.0, r_squared))

    quality = log_nav.rolling(60).apply(score, raw=True)
    median = quality.expanding(min_periods=120).median()
    position = pd.Series(1.0, index=nav.index)
    position[(quality < median) & (quality < 0)] = 0.8
    return position.rename("trend_quality_guard")


def dual_ma_target(nav: pd.Series, fast: int = 5, slow: int = 20) -> pd.Series:
    fast_ma = nav.rolling(fast).mean()
    slow_ma = nav.rolling(slow).mean()
    position = ((nav > fast_ma) & (fast_ma > slow_ma)).astype(float)
    return position.rename(f"dual_ma_{fast}_{slow}")


def _rolling_alpha(
    nav: pd.Series,
    benchmark: pd.Series | None,
    window: int = 60,
) -> pd.Series:
    if benchmark is None or benchmark.empty:
        return pd.Series(0.0, index=nav.index)
    fund_returns = nav.pct_change(fill_method=None).shift(1)
    market_returns = benchmark.reindex(nav.index).ffill().pct_change(fill_method=None).shift(1)
    alpha_values = []
    for end_loc in range(len(nav)):
        start_loc = max(0, end_loc - window + 1)
        y = fund_returns.iloc[start_loc:end_loc + 1].dropna()
        x = market_returns.iloc[start_loc:end_loc + 1].dropna()
        aligned = pd.concat([y.rename("fund"), x.rename("market")], axis=1).dropna()
        if len(aligned) < max(30, window // 2) or float(aligned["market"].std()) == 0:
            alpha_values.append(np.nan)
            continue
        beta, intercept = np.polyfit(aligned["market"], aligned["fund"], 1)
        alpha_values.append(float(intercept * 252))
    return pd.Series(alpha_values, index=nav.index)


def alpha_trend_core_target(
    nav: pd.Series,
    benchmark: pd.Series | None,
    core: float = 0.80,
    low: float = 0.60,
) -> pd.Series:
    """Keep a core position, cutting only when relative alpha and trend both decay."""
    prev_nav = nav.shift(1)
    ma20 = prev_nav.rolling(20).mean()
    ma60 = prev_nav.rolling(60).mean()
    alpha = _rolling_alpha(nav, benchmark, 60)
    alpha_ma = alpha.rolling(10).mean()
    alpha_bad = alpha_ma < 0
    bad_streak = alpha_bad.astype(int).groupby((alpha_bad != alpha_bad.shift()).cumsum()).cumsum()
    position = pd.Series(1.0, index=nav.index)
    position[(bad_streak >= 10) & (prev_nav < ma20)] = core
    position[(bad_streak >= 20) & (prev_nav < ma60)] = low
    return position.ffill().fillna(1.0).rename("alpha_trend_core")


def crowding_trim_core_target(
    nav: pd.Series,
    factors: pd.DataFrame,
    core: float = 0.85,
    cooldown_days: int = 3,
) -> pd.Series:
    """Trim only a small sleeve when ETF proxy shows blow-off volume and price extension."""
    factors = factors.reindex(nav.index).ffill()
    etf_change = factors.get("etf_change_pct", pd.Series(0.0, index=nav.index)).fillna(0.0)
    amount = factors.get("etf_amount", pd.Series(np.nan, index=nav.index))
    amount_ratio = (amount / amount.rolling(20).mean()).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    amplitude = factors.get("etf_amplitude", pd.Series(0.0, index=nav.index)).fillna(0.0)
    prev_nav = nav.shift(1)
    prev_ma5 = prev_nav.rolling(5).mean()
    prev_ma20 = prev_nav.rolling(20).mean()
    prev_percentile = prev_nav.rolling(120, min_periods=30).apply(
        lambda values: float((values <= values[-1]).mean()),
        raw=True,
    )

    current = 1.0
    cooldown = 0
    targets = []
    for date in nav.index:
        if cooldown > 0:
            cooldown -= 1
        extended = (
            prev_percentile.loc[date] >= 0.90
            and prev_nav.loc[date] > prev_ma5.loc[date]
            and prev_ma5.loc[date] > prev_ma20.loc[date]
        )
        crowded = etf_change.loc[date] >= 4.5 and (
            amount_ratio.loc[date] >= 1.25 or amplitude.loc[date] >= 5.5
        )
        buyback = etf_change.loc[date] <= -2.0 or cooldown == 0
        if current >= 0.999 and extended and crowded:
            current = core
            cooldown = cooldown_days
        elif current < 0.999 and buyback:
            current = 1.0
            cooldown = cooldown_days
        targets.append(current)
    return pd.Series(targets, index=nav.index, name="crowding_trim_core")


def adaptive_alpha_crowding_target(
    nav: pd.Series,
    factors: pd.DataFrame,
    benchmark: pd.Series | None,
    start: pd.Timestamp,
) -> pd.Series:
    """A non-leaky router: strong pre-trend keeps core; weak trend uses alpha defense."""
    style = classify_style(nav, factors, start)
    pre_return = float(style.get("pre120_return", 0.0) or 0.0)
    pre_quality = float(style.get("pre120_trend_quality", 0.0) or 0.0)
    pre_drawdown = float(style.get("pre120_max_drawdown", 0.0) or 0.0)
    if pre_return > 0.15 and pre_quality > 0.10:
        target = crowding_trim_core_target(nav, factors, core=0.90, cooldown_days=2)
        if entry_percentile(nav, start, 120) >= 0.985:
            staged = dca_target(nav.index, 10, start)
            target = pd.concat([target, staged], axis=1).min(axis=1)
    elif pre_drawdown < -0.18 or pre_return < -0.05:
        target = alpha_trend_core_target(nav, benchmark, core=0.75, low=0.50)
    else:
        target = alpha_trend_core_target(nav, benchmark, core=0.85, low=0.65)
    return target.rename("adaptive_alpha_crowding")


def factor_forecast_core_target(
    nav: pd.Series,
    factors: pd.DataFrame,
    core: float = 0.80,
    low: float = 0.65,
) -> pd.Series:
    """Use yesterday's ETF breadth/volatility proxy to decide today's core exposure."""
    factors = factors.reindex(nav.index).ffill()
    amplitude = factors.get("etf_amplitude", pd.Series(0.0, index=nav.index)).shift(1)
    change = factors.get("etf_change_pct", pd.Series(0.0, index=nav.index)).shift(1)
    amount = factors.get("etf_amount", pd.Series(np.nan, index=nav.index))
    amount_ratio = (amount / amount.rolling(20).mean()).shift(1)
    prev_nav = nav.shift(1)
    trend = prev_nav / prev_nav.rolling(20).mean() - 1

    amp_rank = amplitude.rolling(120, min_periods=30).rank(pct=True)
    change_rank = change.rolling(120, min_periods=30).rank(pct=True)
    amount_rank = amount_ratio.rolling(120, min_periods=30).rank(pct=True)
    score = 0.55 * amp_rank + 0.25 * change_rank + 0.20 * amount_rank

    position = pd.Series(1.0, index=nav.index)
    position[(score < 0.25) & (trend < 0.03)] = core
    position[(score < 0.15) & (trend < 0.00)] = low
    position[(score > 0.65) | (trend > 0.08)] = 1.0
    return position.ffill().fillna(1.0).rename("factor_forecast_core")


def alpha_mdd_repair_target(
    nav: pd.Series,
    benchmark: pd.Series | None = None,
    mdd_lookback: int = 756,
) -> pd.Series:
    """Left-side drawdown grid, trailing-profit sell, and alpha failure exit."""
    cache_key = (
        "alpha_mdd_repair",
        len(nav),
        str(nav.index.min()),
        str(nav.index.max()),
        float(nav.iloc[0]),
        float(nav.iloc[-1]),
        mdd_lookback,
        None if benchmark is None or benchmark.empty else float(benchmark.reindex(nav.index).ffill().iloc[-1]),
    )
    cached = _TARGET_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()
    benchmark = nav if benchmark is None or benchmark.empty else benchmark.reindex(nav.index).ffill()
    returns = nav.pct_change(fill_method=None).fillna(0.0)
    market_returns = benchmark.pct_change(fill_method=None).fillna(0.0)
    rolling_high = nav.cummax()
    drawdown = nav / rolling_high - 1

    historical_mdd = (nav / nav.cummax() - 1).rolling(
        mdd_lookback,
        min_periods=120,
    ).min()
    typical_mdd = historical_mdd.expanding(min_periods=120).quantile(0.25).abs()
    typical_mdd = typical_mdd.clip(lower=0.15, upper=0.45).fillna(0.30)

    combined = pd.concat([returns, market_returns], axis=1).dropna()
    alpha = pd.Series(np.nan, index=nav.index)
    if len(combined) >= 90:
        alpha_values = []
        for end_loc in range(len(nav)):
            start_loc = max(0, end_loc - 89)
            y = returns.iloc[start_loc:end_loc + 1].to_numpy()
            x = market_returns.iloc[start_loc:end_loc + 1].to_numpy()
            if len(y) < 60 or np.nanstd(x) == 0:
                alpha_values.append(np.nan)
            else:
                beta, intercept = np.polyfit(x, y, 1)
                alpha_values.append(float(intercept * 252))
        alpha = pd.Series(alpha_values, index=nav.index)
    alpha_bad = (alpha < 0).astype(int)
    alpha_bad_streak = alpha_bad.groupby((alpha_bad != alpha_bad.shift()).cumsum()).cumsum()

    current = 0.0
    entry_stage = 0
    cost_basis = np.nan
    target_hit_high = np.nan
    targets = []
    for date in nav.index:
        price = float(nav.loc[date])
        dd = float(drawdown.loc[date])
        threshold = float(typical_mdd.loc[date]) * 0.50
        if alpha_bad_streak.loc[date] >= 20:
            current = 0.0
            entry_stage = 0
            cost_basis = np.nan
            target_hit_high = np.nan
            targets.append(current)
            continue

        if current <= 0.001:
            if dd <= -threshold:
                current = 1 / 6
                entry_stage = 1
                cost_basis = price
        else:
            next_stage_dd = -(threshold + 0.05 * entry_stage)
            if entry_stage < 3 and dd <= next_stage_dd:
                increments = {1: 2 / 6, 2: 3 / 6}
                add = increments.get(entry_stage, 0.0)
                new_position = min(1.0, current + add)
                if new_position > current:
                    cost_basis = (
                        cost_basis * current + price * (new_position - current)
                    ) / new_position
                    current = new_position
                    entry_stage += 1

            if current > 0 and np.isfinite(cost_basis):
                holding_return = price / cost_basis - 1
                if holding_return >= 0.15:
                    target_hit_high = price if not np.isfinite(target_hit_high) else max(target_hit_high, price)
                if np.isfinite(target_hit_high) and price / target_hit_high - 1 <= -0.05:
                    current *= 0.50
                    target_hit_high = np.nan
                    if current < 0.05:
                        current = 0.0
                        entry_stage = 0
                        cost_basis = np.nan
        targets.append(current)
    result = pd.Series(targets, index=nav.index, name="alpha_mdd_repair")
    _TARGET_CACHE[cache_key] = result
    return result.copy()


_TARGET_CACHE: dict[tuple[object, ...], pd.Series] = {}


def evaluate_strategy_window(
    nav: pd.Series,
    target: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    initial_target: float,
) -> dict[str, float]:
    sliced = nav.loc[(nav.index >= start) & (nav.index <= end)]
    result = backtest_open_fund(
        sliced,
        target.reindex(sliced.index),
        initial_target=initial_target,
        execution_lag_days=0,
        minimum_rebalance_fraction=0.01,
        liquidate_final_trade_count=False,
    )
    return result["metrics"]


def quick_metrics(nav: pd.Series, target: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> dict[str, float]:
    sliced = nav.loc[(nav.index >= start) & (nav.index <= end)]
    position = target.reindex(sliced.index).ffill().fillna(0.0)
    returns = sliced.pct_change(fill_method=None).fillna(0.0) * position
    equity = (1 + returns).cumprod()
    drawdown = equity / equity.cummax() - 1
    volatility = float(returns.std() * np.sqrt(252))
    sharpe = float((returns.mean() * 252) / volatility) if volatility > 0 else 0.0
    trade_count = float(position.ne(position.shift()).sum())
    return {
        "total_return": float(equity.iloc[-1] - 1) if len(equity) else 0.0,
        "max_drawdown": float(drawdown.min()) if len(drawdown) else 0.0,
        "sharpe": sharpe,
        "trade_count": trade_count,
    }


def _candidate_targets(
    nav: pd.Series,
    factors: pd.DataFrame,
    sector_group: str,
    start: pd.Timestamp,
) -> dict[str, pd.Series]:
    style = classify_style(nav, factors, start)
    routed = routed_config(style, sector_group)
    routed_target, _ = style_timing_target(nav, factors, routed)
    benchmark = factors.get("etf_close", pd.Series(dtype=float))
    return {
        "all_in": all_in_target(nav.index),
        "dca_20": dca_target(nav.index, 20, start),
        "high_entry_dca_975": high_entry_dca_target(nav, start, 0.975, 15),
        "entry_control": entry_control_target(nav, factors, start),
        "simple_t": simple_timing_target(nav, factors),
        "routed_style": routed_target.rename("routed_style"),
        "trend_quality_guard": trend_quality_guard_target(nav),
        "dual_ma_5_20": dual_ma_target(nav, 5, 20),
        "alpha_mdd_repair": alpha_mdd_repair_target(nav, benchmark),
        "alpha_trend_core": alpha_trend_core_target(nav, benchmark),
        "crowding_trim_core": crowding_trim_core_target(nav, factors),
        "adaptive_alpha_crowding": adaptive_alpha_crowding_target(nav, factors, benchmark, start),
        "factor_forecast_core": factor_forecast_core_target(nav, factors),
    }


def _select_on_prior_training(
    nav: pd.Series,
    factors: pd.DataFrame,
    sector_group: str,
    test_start: pd.Timestamp,
    train_days: int,
) -> tuple[str, pd.Series, pd.DataFrame]:
    start_loc = nav.index.get_loc(test_start)
    train_start_loc = max(0, start_loc - train_days)
    train_index = nav.index[train_start_loc:start_loc]
    if len(train_index) < 30:
        target = all_in_target(nav.index)
        diagnostics = pd.DataFrame(
            [{"strategy": "all_in", "objective": 0.0, "reason": "insufficient_training"}]
        )
        return "all_in", target, diagnostics
    train_start = train_index[0]
    train_end = train_index[-1]
    rows = []
    candidates = _candidate_targets(nav, factors, sector_group, train_start)
    benchmark = quick_metrics(nav, candidates["all_in"], train_start, train_end)
    for name, target in candidates.items():
        metrics = quick_metrics(nav, target, train_start, train_end)
        excess = float(metrics["total_return"]) - float(benchmark["total_return"])
        drawdown_improvement = float(metrics["max_drawdown"]) - float(benchmark["max_drawdown"])
        objective = (
            excess
            + 0.40 * drawdown_improvement
            + 0.02 * np.clip(float(metrics["sharpe"]), -2, 2)
            - 0.001 * float(metrics["trade_count"])
        )
        rows.append(
            {
                "strategy": name,
                "train_start": train_start,
                "train_end": train_end,
                "objective": objective,
                "train_excess": excess,
                "train_drawdown_improvement": drawdown_improvement,
                **metrics,
            }
        )
    diagnostics = pd.DataFrame(rows).sort_values(
        ["objective", "train_excess", "train_drawdown_improvement"],
        ascending=False,
    )
    chosen = str(diagnostics.iloc[0]["strategy"])
    # Rebuild targets using only the real test start for entry-control state.
    test_candidates = _candidate_targets(nav, factors, sector_group, test_start)
    return chosen, test_candidates[chosen], diagnostics


def compare_sliding_windows(
    asset: str,
    fund_code: str,
    sector_group: str,
    end: pd.Timestamp | None = None,
    window_days: int = 90,
    window_count: int = 30,
    train_days: int = 120,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    nav = read_cached_nav(fund_code)
    factors = read_external_factors(asset).reindex(nav.index).ffill()
    end = pd.Timestamp(end) if end is not None else nav.index.max()
    end_loc = nav.index.get_indexer([nav.index[nav.index <= end][-1]])[0]
    start_last = end_loc - window_days + 1
    start_first = start_last - window_count + 1
    if start_first < 0:
        raise ValueError("Not enough history for requested sliding windows")
    rows = []
    histories = []
    for offset, start_loc in enumerate(range(start_first, start_last + 1)):
        start = nav.index[start_loc]
        window_end = nav.index[start_loc + window_days - 1]
        style = classify_style(nav, factors, start)
        routed = routed_config(style, sector_group)
        targets = _candidate_targets(nav, factors, sector_group, start)
        selected_name, selected_target, train_diagnostics = _select_on_prior_training(
            nav,
            factors,
            sector_group,
            start,
            train_days,
        )
        targets["trained_selector"] = selected_target.rename("trained_selector")
        sliced_for_daily_dca = nav.loc[(nav.index >= start) & (nav.index <= window_end)]
        for name, target in targets.items():
            initial = float(target.loc[start]) if start in target.index else 1.0
            metrics = evaluate_strategy_window(nav, target, start, window_end, initial)
            rows.append(
                {
                    "asset": asset,
                    "fund_code": fund_code,
                    "sector_group": sector_group,
                    "window_id": offset + 1,
                    "start": start,
                    "end": window_end,
                    "strategy": name,
                    "entry_percentile_120": float((nav.loc[:start].tail(120) <= nav.loc[start]).mean()),
                    "style": style.get("style", "unknown"),
                    "routed_model": routed.name,
                    "train_days": train_days,
                    "trained_selected_strategy": selected_name,
                    "trained_selected_objective": float(train_diagnostics.iloc[0]["objective"]),
                    **metrics,
                }
            )
        dca_metrics = daily_dca_metrics(sliced_for_daily_dca)
        rows.append(
            {
                "asset": asset,
                "fund_code": fund_code,
                "sector_group": sector_group,
                "window_id": offset + 1,
                "start": start,
                "end": window_end,
                "strategy": "daily_dca",
                "entry_percentile_120": float((nav.loc[:start].tail(120) <= nav.loc[start]).mean()),
                "style": style.get("style", "unknown"),
                "routed_model": routed.name,
                "train_days": train_days,
                "trained_selected_strategy": selected_name,
                "trained_selected_objective": float(train_diagnostics.iloc[0]["objective"]),
                **dca_metrics,
            }
        )
    detail = pd.DataFrame(rows)
    winners = (
        detail.sort_values(["window_id", "total_return"], ascending=[True, False])
        .groupby("window_id")
        .head(1)[["window_id", "strategy"]]
        .rename(columns={"strategy": "winner"})
    )
    detail = detail.merge(winners, on="window_id", how="left")
    all_in = detail.loc[detail["strategy"] == "all_in"].set_index("window_id")
    detail["excess_vs_all_in"] = detail.apply(
        lambda row: row["total_return"] - all_in.loc[row["window_id"], "total_return"],
        axis=1,
    )
    detail["drawdown_improvement_vs_all_in"] = detail.apply(
        lambda row: row["max_drawdown"] - all_in.loc[row["window_id"], "max_drawdown"],
        axis=1,
    )
    summary = (
        detail.groupby("strategy")
        .agg(
            windows=("window_id", "count"),
            mean_return=("total_return", "mean"),
            median_return=("total_return", "median"),
            return_variance=("total_return", "var"),
            mean_max_drawdown=("max_drawdown", "mean"),
            median_max_drawdown=("max_drawdown", "median"),
            mean_excess_vs_all_in=("excess_vs_all_in", "mean"),
            median_excess_vs_all_in=("excess_vs_all_in", "median"),
            beat_all_in_ratio=("excess_vs_all_in", lambda x: float((x > 0).mean())),
            mean_drawdown_improvement=("drawdown_improvement_vs_all_in", "mean"),
        )
        .reset_index()
    )
    win_counts = detail.drop_duplicates("window_id")["winner"].value_counts()
    summary["wins"] = summary["strategy"].map(win_counts).fillna(0).astype(int)
    summary["win_ratio"] = summary["wins"] / window_count
    return detail, summary.sort_values(["wins", "mean_return"], ascending=False)
