"""Style-aware same-day open-fund timing experiments.

The strategy uses a 120-trading-day pre-test window to classify the fund's
style, then applies fixed timing rules in the locked test window.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .fund_backtest import backtest_open_fund


@dataclass(frozen=True)
class StyleTimingConfig:
    mode: str
    core: float
    low: float
    trim_up_pct: float
    add_down_pct: float
    amount_ratio: float
    cooldown_days: int
    stop_drawdown: float

    @property
    def name(self) -> str:
        return (
            f"{self.mode}_core{self.core:.2f}_low{self.low:.2f}"
            f"_up{self.trim_up_pct:.1f}_down{abs(self.add_down_pct):.1f}"
            f"_amt{self.amount_ratio:.1f}_cd{self.cooldown_days}"
            f"_dd{abs(self.stop_drawdown):.2f}"
        )


def read_cached_nav(fund_code: str, cache_dir: str | Path = "data/nav_cache") -> pd.Series:
    frame = pd.read_csv(Path(cache_dir) / f"{fund_code}.csv", parse_dates=["date"])
    return (
        pd.to_numeric(frame.set_index("date")["close"], errors="coerce")
        .dropna()
        .sort_index()
    )


def read_external_factors(asset: str, cache_dir: str | Path = "data/external_factors") -> pd.DataFrame:
    path = Path(cache_dir) / f"{asset}.csv"
    if not path.exists():
        return pd.DataFrame()
    factors = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    if "etf_close" not in factors and "etf_change_pct" in factors:
        change = pd.to_numeric(factors["etf_change_pct"], errors="coerce").fillna(0.0) / 100
        factors["etf_close"] = (1 + change).cumprod()
    return factors


def classify_style(
    nav: pd.Series,
    factors: pd.DataFrame,
    test_start: pd.Timestamp,
    lookback: int = 120,
) -> dict[str, float | str]:
    pre = nav.loc[nav.index < test_start].tail(lookback)
    returns = pre.pct_change(fill_method=None).dropna()
    if len(pre) < 30:
        return {
            "style": "insufficient_history",
            "pre120_observations": float(len(pre)),
        }
    total_return = float(pre.iloc[-1] / pre.iloc[0] - 1)
    volatility = float(returns.std() * np.sqrt(252))
    max_drawdown = float((pre / pre.cummax() - 1).min())
    trend_quality = _trend_quality_score(pre)
    etf = factors.reindex(pre.index).ffill()
    etf_corr = 0.0
    if "etf_change_pct" in etf:
        aligned = pd.concat(
            [returns.rename("fund"), (etf["etf_change_pct"] / 100).rename("etf")],
            axis=1,
        ).dropna()
        if len(aligned) > 20:
            etf_corr = float(aligned["fund"].corr(aligned["etf"]))
    if total_return > 0.20 and trend_quality > 0.25 and volatility > 0.30:
        style = "high_beta_trend"
    elif total_return > 0.08 and trend_quality > 0.15:
        style = "smooth_trend"
    elif max_drawdown < -0.18 and volatility > 0.28:
        style = "cyclical_reversal"
    else:
        style = "range_or_defensive"
    return {
        "style": style,
        "pre120_observations": float(len(pre)),
        "pre120_return": total_return,
        "pre120_volatility": volatility,
        "pre120_max_drawdown": max_drawdown,
        "pre120_trend_quality": trend_quality,
        "pre120_etf_corr": etf_corr,
    }


def _trend_quality_score(nav: pd.Series) -> float:
    values = np.log(nav.dropna().to_numpy(dtype=float))
    if len(values) < 20 or not np.isfinite(values).all():
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    total = ((values - values.mean()) ** 2).sum()
    residual = ((values - fitted) ** 2).sum()
    r_squared = 1 - residual / total if total > 0 else 0.0
    return float(np.expm1(slope * 252) * max(0.0, r_squared))


def configs_for_style(style: str) -> list[StyleTimingConfig]:
    if style == "high_beta_trend":
        return [
            StyleTimingConfig("exhaustion", 0.95, 0.85, 5.0, -3.0, 1.3, 1, -0.06),
            StyleTimingConfig("exhaustion", 0.90, 0.80, 6.0, -4.0, 1.5, 2, -0.08),
            StyleTimingConfig("hybrid", 0.95, 0.85, 6.0, -3.0, 1.3, 1, -0.06),
        ]
    if style == "smooth_trend":
        return [
            StyleTimingConfig("trend_quality_guard", 0.80, 0.50, 5.0, -3.0, 1.2, 3, -0.08),
            StyleTimingConfig("exhaustion", 0.90, 0.75, 4.0, -3.0, 1.2, 2, -0.06),
        ]
    if style == "cyclical_reversal":
        return [
            StyleTimingConfig("reversal", 0.85, 0.65, 4.0, -3.0, 1.0, 2, -0.08),
            StyleTimingConfig("hybrid", 0.85, 0.65, 5.0, -4.0, 1.2, 2, -0.10),
        ]
    return [
        StyleTimingConfig("trend_guard", 0.80, 0.60, 5.0, -3.0, 1.2, 3, -0.06),
        StyleTimingConfig("exhaustion", 0.90, 0.75, 4.0, -3.0, 1.2, 2, -0.06),
    ]


def routed_config(
    style_metrics: dict[str, float | str],
    sector_group: str = "unknown",
) -> StyleTimingConfig:
    """Choose one fixed strategy from pre-test style only."""
    style = str(style_metrics.get("style", "range_or_defensive"))
    pre_return = float(style_metrics.get("pre120_return", 0.0) or 0.0)
    pre_volatility = float(style_metrics.get("pre120_volatility", 0.0) or 0.0)
    if style == "high_beta_trend":
        if sector_group in {"materials", "cyclical"}:
            return StyleTimingConfig("hybrid", 0.95, 0.85, 6.0, -3.0, 1.3, 1, -0.06)
        if pre_return > 0.80:
            return StyleTimingConfig("exhaustion", 0.95, 0.85, 5.0, -3.0, 1.3, 1, -0.06)
        if pre_volatility > 0.35:
            return StyleTimingConfig("exhaustion", 0.90, 0.80, 6.0, -4.0, 1.5, 2, -0.08)
        return StyleTimingConfig("exhaustion", 0.95, 0.85, 5.0, -3.0, 1.3, 1, -0.06)
    if style == "smooth_trend":
        return StyleTimingConfig("trend_quality_guard", 0.80, 0.50, 5.0, -3.0, 1.2, 3, -0.08)
    if style == "cyclical_reversal":
        return StyleTimingConfig("reversal", 0.85, 0.65, 4.0, -3.0, 1.0, 2, -0.08)
    return StyleTimingConfig("exhaustion", 0.90, 0.75, 4.0, -3.0, 1.2, 2, -0.06)


def style_timing_target(
    nav: pd.Series,
    factors: pd.DataFrame,
    config: StyleTimingConfig,
) -> tuple[pd.Series, pd.DataFrame]:
    factors = factors.reindex(nav.index).ffill()
    etf_change = factors.get("etf_change_pct", pd.Series(0.0, index=nav.index)).fillna(0.0)
    amount = factors.get("etf_amount", pd.Series(np.nan, index=nav.index))
    amount_ratio = (amount / amount.rolling(20).mean()).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    amplitude = factors.get("etf_amplitude", pd.Series(0.0, index=nav.index)).fillna(0.0)
    ma5 = nav.rolling(5).mean()
    ma20 = nav.rolling(20).mean()
    ma60 = nav.rolling(60).mean()
    drawdown10 = nav / nav.rolling(10).max() - 1
    drawdown20 = nav / nav.rolling(20).max() - 1
    momentum3 = nav.pct_change(3, fill_method=None)
    momentum5 = nav.pct_change(5, fill_method=None)
    trend_quality = nav.rolling(60).apply(
        lambda values: _trend_quality_score(pd.Series(values)),
        raw=False,
    )

    current = 1.0
    cooldown = 0
    tactical_age = 0
    targets = []
    rows = []
    for date in nav.index:
        if cooldown > 0:
            cooldown -= 1
        tactical_age = tactical_age + 1 if current < 0.999 else 0
        overheat = (
            etf_change.loc[date] >= config.trim_up_pct
            and (amount_ratio.loc[date] >= config.amount_ratio or amplitude.loc[date] >= config.trim_up_pct)
            and nav.loc[date] >= ma5.loc[date]
        )
        breakdown = (
            drawdown10.loc[date] <= config.stop_drawdown
            and momentum3.loc[date] < 0
            and nav.loc[date] < ma5.loc[date]
        )
        buy_dip = (
            (etf_change.loc[date] <= config.add_down_pct or drawdown10.loc[date] <= config.add_down_pct / 100)
            and nav.loc[date] >= ma20.loc[date] * 0.90
        )
        repair = (
            (etf_change.loc[date] > 0.5 and nav.loc[date] > ma5.loc[date])
            or momentum3.loc[date] > 0.025
        )
        quality_exit = (
            trend_quality.loc[date] < 0
            and nav.loc[date] < ma20.loc[date]
            and momentum5.loc[date] < 0
        )
        quality_reentry = (
            trend_quality.loc[date] > 0
            and nav.loc[date] > ma20.loc[date]
            and ma20.loc[date] >= ma60.loc[date] * 0.98
        )
        action = "HOLD"
        reason = "no_signal"
        if config.mode == "exhaustion":
            if current >= 0.999 and cooldown == 0 and overheat:
                current = config.core
                cooldown = config.cooldown_days
                tactical_age = 0
                action = "TRIM"
                reason = "etf_overheat"
            elif current < 0.999 and (buy_dip or tactical_age >= config.cooldown_days + 2):
                current = 1.0
                cooldown = config.cooldown_days
                action = "BUYBACK"
                reason = "dip_or_time_buyback"
        elif config.mode == "trend_guard":
            if current >= 0.999 and cooldown == 0 and breakdown:
                current = config.core
                cooldown = config.cooldown_days
                tactical_age = 0
                action = "TRIM"
                reason = "short_breakdown"
            elif current < 0.999 and repair:
                current = 1.0
                cooldown = config.cooldown_days
                action = "BUYBACK"
                reason = "repair"
        elif config.mode == "trend_quality_guard":
            if current >= 0.999 and cooldown == 0 and quality_exit:
                current = config.core
                cooldown = config.cooldown_days
                tactical_age = 0
                action = "TRIM"
                reason = "quality_exit"
            elif current < 0.999 and quality_reentry:
                current = 1.0
                cooldown = config.cooldown_days
                action = "BUYBACK"
                reason = "quality_reentry"
        elif config.mode == "reversal":
            if current >= 0.999 and cooldown == 0 and overheat:
                current = config.core
                cooldown = config.cooldown_days
                tactical_age = 0
                action = "TRIM"
                reason = "cyclical_overheat"
            elif current < 0.999 and (buy_dip or repair):
                current = 1.0
                cooldown = config.cooldown_days
                action = "BUYBACK"
                reason = "reversal_buyback"
        else:
            if current >= 0.999 and cooldown == 0 and overheat:
                current = config.core
                cooldown = config.cooldown_days
                tactical_age = 0
                action = "TRIM"
                reason = "hybrid_overheat"
            elif current >= config.core + 0.01 and cooldown == 0 and breakdown:
                current = config.low
                cooldown = config.cooldown_days
                tactical_age = 0
                action = "TRIM"
                reason = "hybrid_breakdown"
            elif current < 0.999 and (buy_dip or repair or tactical_age >= config.cooldown_days + 3):
                current = 1.0
                cooldown = config.cooldown_days
                action = "BUYBACK"
                reason = "hybrid_buyback"
        targets.append(current)
        rows.append(
            {
                "date": date,
                "target_position": current,
                "action": action,
                "reason": reason,
                "etf_change_pct": float(etf_change.loc[date]),
                "amount_ratio": float(amount_ratio.loc[date]),
                "drawdown10": float(drawdown10.loc[date]) if pd.notna(drawdown10.loc[date]) else np.nan,
                "drawdown20": float(drawdown20.loc[date]) if pd.notna(drawdown20.loc[date]) else np.nan,
            }
        )
    return pd.Series(targets, index=nav.index, name=config.name), pd.DataFrame(rows)


def evaluate_style_timing(
    asset: str,
    fund_code: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, float | str]]:
    nav = read_cached_nav(fund_code)
    factors = read_external_factors(asset).reindex(nav.index).ffill()
    style = classify_style(nav, factors, test_start)
    sliced_nav = nav.loc[(nav.index >= test_start) & (nav.index <= test_end)]
    benchmark = backtest_open_fund(
        sliced_nav,
        pd.Series(1.0, index=sliced_nav.index),
        initial_target=1.0,
        execution_lag_days=0,
        minimum_rebalance_fraction=0.01,
    )["metrics"]
    rows = [{"asset": asset, "model": "buy_hold", **style, **benchmark}]
    histories = []
    for config in configs_for_style(str(style["style"])):
        target, history = style_timing_target(nav, factors, config)
        metrics = backtest_open_fund(
            sliced_nav,
            target.loc[sliced_nav.index],
            initial_target=1.0,
            execution_lag_days=0,
            minimum_rebalance_fraction=0.01,
        )["metrics"]
        rows.append({"asset": asset, "model": config.name, **style, **metrics})
        history = history.loc[
            (history["date"] >= test_start) & (history["date"] <= test_end)
        ].copy()
        history["asset"] = asset
        history["model"] = config.name
        histories.append(history)
    result = pd.DataFrame(rows)
    benchmark_return = float(benchmark["total_return"])
    benchmark_drawdown = float(benchmark["max_drawdown"])
    result["excess_return"] = result["total_return"] - benchmark_return
    result["drawdown_improvement"] = result["max_drawdown"] - benchmark_drawdown
    return result, pd.concat(histories, ignore_index=True), style


def evaluate_routed_style_timing(
    asset: str,
    fund_code: str,
    sector_group: str,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[dict[str, object], pd.DataFrame]:
    nav = read_cached_nav(fund_code)
    factors = read_external_factors(asset).reindex(nav.index).ffill()
    style = classify_style(nav, factors, test_start)
    config = routed_config(style, sector_group)
    target, history = style_timing_target(nav, factors, config)
    sliced_nav = nav.loc[(nav.index >= test_start) & (nav.index <= test_end)]
    benchmark = backtest_open_fund(
        sliced_nav,
        pd.Series(1.0, index=sliced_nav.index),
        initial_target=1.0,
        execution_lag_days=0,
        minimum_rebalance_fraction=0.01,
    )["metrics"]
    candidate = backtest_open_fund(
        sliced_nav,
        target.loc[sliced_nav.index],
        initial_target=1.0,
        execution_lag_days=0,
        minimum_rebalance_fraction=0.01,
    )["metrics"]
    row: dict[str, object] = {
        "asset": asset,
        "fund_code": fund_code,
        "sector_group": sector_group,
        "model": config.name,
        **style,
        **candidate,
        "buy_hold_return": benchmark["total_return"],
        "buy_hold_max_drawdown": benchmark["max_drawdown"],
        "buy_hold_sharpe": benchmark["sharpe"],
    }
    row["excess_return"] = float(candidate["total_return"]) - float(benchmark["total_return"])
    row["drawdown_improvement"] = float(candidate["max_drawdown"]) - float(benchmark["max_drawdown"])
    history = history.loc[
        (history["date"] >= test_start) & (history["date"] <= test_end)
    ].copy()
    history["asset"] = asset
    history["model"] = config.name
    return row, history


def entry_percentile(nav: pd.Series, date: pd.Timestamp, window: int = 120) -> float:
    history = nav.loc[nav.index <= date].tail(window)
    if len(history) < 20:
        return 0.5
    current = float(history.iloc[-1])
    return float((history <= current).mean())


def initial_position_from_entry_risk(
    nav: pd.Series,
    factors: pd.DataFrame,
    date: pd.Timestamp,
) -> float:
    """Avoid assuming new money is invested all at once near local extremes."""
    percentile = entry_percentile(nav, date, 120)
    recent_return = float(nav.loc[nav.index <= date].pct_change(20, fill_method=None).iloc[-1])
    factor_row = factors.reindex(nav.index).ffill().loc[date] if date in factors.reindex(nav.index).ffill().index else pd.Series(dtype=float)
    etf_change = float(factor_row.get("etf_change_pct", 0.0) or 0.0)
    if percentile >= 0.95 and recent_return > 0.20:
        return 0.60
    if percentile >= 0.90 and recent_return > 0.12:
        return 0.70
    if percentile >= 0.85 and etf_change > 4:
        return 0.75
    return 1.0


def evaluate_window_with_entry_control(
    asset: str,
    fund_code: str,
    sector_group: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, object]:
    nav = read_cached_nav(fund_code)
    factors = read_external_factors(asset).reindex(nav.index).ffill()
    style = classify_style(nav, factors, start)
    config = routed_config(style, sector_group)
    target, _ = style_timing_target(nav, factors, config)
    initial_target = initial_position_from_entry_risk(nav, factors, start)
    if len(target):
        target = target.copy()
        target.loc[target.index >= start] = target.loc[target.index >= start].clip(upper=1.0)
        first_dates = target.loc[target.index >= start].index[:5]
        target.loc[first_dates] = np.minimum(target.loc[first_dates], initial_target)
    sliced_nav = nav.loc[(nav.index >= start) & (nav.index <= end)]
    if len(sliced_nav) < 30:
        raise ValueError("Window too short")
    buy_hold = backtest_open_fund(
        sliced_nav,
        pd.Series(1.0, index=sliced_nav.index),
        initial_target=1.0,
        execution_lag_days=0,
        minimum_rebalance_fraction=0.01,
    )["metrics"]
    staged_hold = backtest_open_fund(
        sliced_nav,
        pd.Series(initial_target, index=sliced_nav.index),
        initial_target=initial_target,
        execution_lag_days=0,
        minimum_rebalance_fraction=0.01,
    )["metrics"]
    candidate = backtest_open_fund(
        sliced_nav,
        target.loc[sliced_nav.index],
        initial_target=initial_target,
        execution_lag_days=0,
        minimum_rebalance_fraction=0.01,
    )["metrics"]
    row: dict[str, object] = {
        "asset": asset,
        "fund_code": fund_code,
        "sector_group": sector_group,
        "start": start,
        "end": end,
        "model": config.name,
        "entry_percentile_120": entry_percentile(nav, start, 120),
        "initial_target": initial_target,
        **style,
        **candidate,
        "buy_hold_return": buy_hold["total_return"],
        "buy_hold_max_drawdown": buy_hold["max_drawdown"],
        "staged_hold_return": staged_hold["total_return"],
        "staged_hold_max_drawdown": staged_hold["max_drawdown"],
    }
    row["excess_vs_buy_hold"] = float(candidate["total_return"]) - float(buy_hold["total_return"])
    row["excess_vs_staged_hold"] = float(candidate["total_return"]) - float(staged_hold["total_return"])
    row["drawdown_improvement_vs_buy_hold"] = float(candidate["max_drawdown"]) - float(buy_hold["max_drawdown"])
    return row


def rolling_entry_stress_test(
    assets: dict[str, tuple[str, str]],
    window_days: int = 90,
    step_days: int = 10,
    min_pre_days: int = 30,
) -> pd.DataFrame:
    rows = []
    for asset, (fund_code, sector_group) in assets.items():
        nav = read_cached_nav(fund_code)
        dates = nav.index
        for start_loc in range(min_pre_days, max(min_pre_days, len(dates) - window_days), step_days):
            start = dates[start_loc]
            end = dates[min(len(dates) - 1, start_loc + window_days - 1)]
            try:
                rows.append(
                    evaluate_window_with_entry_control(
                        asset,
                        fund_code,
                        sector_group,
                        start,
                        end,
                    )
                )
            except Exception:
                continue
    return pd.DataFrame(rows)
