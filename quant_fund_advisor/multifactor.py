"""Explainable multi-factor model for medium-term mutual-fund decisions."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FundStructure:
    """Latest disclosed fund structure; values may be missing."""

    scale_billion_cny: float | None = None
    holder_count: int | None = None
    institution_ratio: float | None = None
    top10_concentration: float | None = None
    quarterly_share_growth: float | None = None
    tracking_error: float | None = None


@dataclass(frozen=True)
class MultiFactorConfig:
    weights: dict[str, float] = field(
        default_factory=lambda: {
            "trend": 0.18,
            "momentum": 0.16,
            "quality": 0.10,
            "breadth": 0.12,
            "cross_market": 0.10,
            "risk_appetite": 0.09,
            "news": 0.07,
            "flow": 0.08,
            "structure": 0.10,
        }
    )
    buy_score: float = 0.25
    strong_buy_score: float = 0.50
    reduce_score: float = -0.10
    sell_score: float = -0.30
    max_theme_weight: float = 0.20
    rolling_window: int = 252
    minimum_history: int = 60


def rolling_percentile(
    series: pd.Series,
    window: int = 252,
    minimum_history: int = 60,
) -> pd.Series:
    """Historical-only percentile mapped to [-1, 1]."""

    def percentile(values: np.ndarray) -> float:
        current = values[-1]
        valid = values[np.isfinite(values)]
        if len(valid) == 0 or not np.isfinite(current):
            return 0.0
        rank = (valid <= current).mean()
        return float(rank * 2 - 1)

    return series.rolling(window, min_periods=minimum_history).apply(
        percentile, raw=True
    ).fillna(0.0)


def _safe_series(
    value: pd.Series | None,
    index: pd.Index,
    default: float = 0.0,
) -> pd.Series:
    if value is None:
        return pd.Series(default, index=index, dtype=float)
    return value.reindex(index).ffill().fillna(default).astype(float)


def structure_score(structure: FundStructure | None) -> tuple[float, dict[str, float]]:
    """Score liquidity, crowding and tracking quality without assuming scale dilutes NAV."""
    if structure is None:
        return 0.0, {}
    parts: dict[str, float] = {}
    scale = structure.scale_billion_cny
    if scale is not None:
        # Very small funds have liquidation/tracking risk; very large thematic funds
        # can suffer capacity and crowded-redemption pressure.
        if scale < 0.05:
            parts["scale"] = -1.0
        elif scale < 0.20:
            parts["scale"] = -0.3
        elif scale <= 5:
            parts["scale"] = 0.6
        elif scale <= 15:
            parts["scale"] = 0.2
        else:
            parts["scale"] = -0.2
    holders = structure.holder_count
    if holders is not None:
        if holders < 1_000:
            parts["holders"] = -0.6
        elif holders <= 300_000:
            parts["holders"] = 0.4
        else:
            parts["holders"] = -0.15
    if structure.institution_ratio is not None:
        ratio = structure.institution_ratio
        parts["institution"] = -0.5 if ratio > 0.8 else (0.3 if 0.1 <= ratio <= 0.6 else 0.0)
    if structure.top10_concentration is not None:
        concentration = structure.top10_concentration
        parts["concentration"] = (
            -0.7 if concentration > 0.75 else (0.2 if concentration < 0.55 else 0.0)
        )
    if structure.quarterly_share_growth is not None:
        growth = structure.quarterly_share_growth
        parts["share_growth"] = -0.6 if growth > 1.0 else (-0.2 if growth > 0.5 else 0.2)
    if structure.tracking_error is not None:
        parts["tracking"] = (
            -0.7 if structure.tracking_error > 0.04 else
            (0.4 if structure.tracking_error < 0.015 else 0.0)
        )
    return (float(np.mean(list(parts.values()))) if parts else 0.0), parts


def point_in_time_structure_score(
    history: pd.DataFrame,
    target_index: pd.DatetimeIndex,
    publication_lag_days: int = 90,
) -> pd.Series:
    """Map disclosed structure to dates only after a conservative publication lag."""
    if history.empty:
        return pd.Series(0.0, index=target_index)
    required = {"report_date"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Structure history missing columns: {sorted(missing)}")
    rows = []
    for row in history.itertuples(index=False):
        structure = FundStructure(
            scale_billion_cny=getattr(row, "scale_billion_cny", None),
            holder_count=getattr(row, "holder_count", None),
            institution_ratio=getattr(row, "institution_ratio", None),
            top10_concentration=getattr(row, "top10_concentration", None),
            quarterly_share_growth=getattr(row, "quarterly_share_growth", None),
            tracking_error=getattr(row, "tracking_error", None),
        )
        score, _ = structure_score(structure)
        available_date = pd.Timestamp(row.report_date) + pd.Timedelta(
            days=publication_lag_days
        )
        rows.append((available_date, score))
    disclosed = pd.Series(
        [score for _, score in rows],
        index=pd.DatetimeIndex([date for date, _ in rows]),
    ).sort_index()
    return disclosed.reindex(target_index).ffill().fillna(0.0)


def compute_factor_panel(
    fund_nav: pd.Series,
    cn_constituents: pd.DataFrame,
    us_basket_nav: pd.Series | None = None,
    market_nav: pd.Series | None = None,
    volume: pd.Series | None = None,
    advance_ratio: pd.Series | None = None,
    news_sentiment: pd.Series | None = None,
    news_attention: pd.Series | None = None,
    risk_appetite: pd.Series | None = None,
    structure: FundStructure | None = None,
    structure_history: pd.DataFrame | None = None,
    config: MultiFactorConfig | None = None,
) -> pd.DataFrame:
    config = config or MultiFactorConfig()
    nav = fund_nav.sort_index().astype(float)
    index = nav.index
    returns = nav.pct_change(fill_method=None)

    ma20 = nav.rolling(20).mean()
    ma60 = nav.rolling(60).mean()
    trend_raw = 0.55 * (nav / ma60 - 1) + 0.45 * (ma20 / ma60 - 1)

    momentum_raw = (
        0.35 * nav.pct_change(20, fill_method=None)
        + 0.40 * nav.pct_change(60, fill_method=None)
        + 0.25 * nav.pct_change(120, fill_method=None)
    )
    reversal_penalty = nav.pct_change(5, fill_method=None).clip(lower=0) ** 2
    momentum_raw = momentum_raw - reversal_penalty

    downside = returns.clip(upper=0).rolling(20).std() * np.sqrt(252)
    drawdown = nav / nav.rolling(120, min_periods=20).max() - 1
    quality_raw = -0.6 * downside + 0.4 * drawdown

    constituent_returns = cn_constituents.reindex(index).ffill().pct_change(fill_method=None)
    breadth_default = (constituent_returns.rolling(20).mean() > 0).mean(axis=1)
    breadth_raw = _safe_series(advance_ratio, index, np.nan).fillna(breadth_default)
    dispersion = constituent_returns.rolling(20).std().mean(axis=1)
    breadth_raw = breadth_raw - 0.5 * dispersion

    if us_basket_nav is not None:
        us_aligned = us_basket_nav.reindex(index).ffill()
        cross_raw = (
            0.6 * us_aligned.pct_change(20, fill_method=None).shift(1)
            + 0.4
            * us_aligned.pct_change(fill_method=None).shift(1).rolling(20).mean()
        )
    else:
        cross_raw = pd.Series(0.0, index=index)

    market = _safe_series(market_nav, index, np.nan)
    if market.isna().all():
        market_raw = pd.Series(0.0, index=index)
    else:
        market_raw = 0.6 * market.pct_change(20, fill_method=None) + 0.4 * (
            market / market.rolling(60).mean() - 1
        )
    risk_raw = market_raw + _safe_series(risk_appetite, index)

    news_raw = _safe_series(news_sentiment, index)
    attention = _safe_series(news_attention, index)
    volume_ratio = _safe_series(volume, index, np.nan)
    if volume_ratio.isna().all():
        volume_ratio = pd.Series(1.0, index=index)
    else:
        volume_ratio = volume_ratio / volume_ratio.rolling(20).mean()

    # Excessive attention, volume and short-term return together are crowding.
    crowding_raw = (
        rolling_percentile(attention, config.rolling_window, config.minimum_history).clip(lower=0)
        * rolling_percentile(volume_ratio, config.rolling_window, config.minimum_history).clip(lower=0)
        * rolling_percentile(nav.pct_change(20, fill_method=None), config.rolling_window, config.minimum_history).clip(lower=0)
    )
    flow_raw = volume_ratio - 1.0 - 0.8 * crowding_raw

    struct_value, struct_parts = structure_score(structure)
    raw_factors = {
        "trend": trend_raw,
        "momentum": momentum_raw,
        "quality": quality_raw,
        "breadth": breadth_raw,
        "cross_market": cross_raw,
        "risk_appetite": risk_raw,
        "news": news_raw - 0.6 * crowding_raw,
        "flow": flow_raw,
    }
    panel = pd.DataFrame(index=index)
    for name, raw in raw_factors.items():
        panel[name] = rolling_percentile(
            raw, config.rolling_window, config.minimum_history
        )
    if structure_history is not None:
        panel["structure"] = point_in_time_structure_score(
            structure_history, pd.DatetimeIndex(index)
        )
    else:
        # Latest snapshots are valid for today's decision only, not historical tests.
        panel["structure"] = 0.0
        if len(panel):
            panel.iloc[-1, panel.columns.get_loc("structure")] = struct_value
    panel["crowding_penalty"] = crowding_raw.fillna(0.0).clip(0, 1)
    panel["volatility20"] = returns.rolling(20).std() * np.sqrt(252)
    panel.attrs["structure_details"] = struct_parts
    return panel


def score_factor_panel(
    panel: pd.DataFrame,
    config: MultiFactorConfig | None = None,
) -> pd.DataFrame:
    config = config or MultiFactorConfig()
    result = panel.copy()
    weighted = pd.Series(0.0, index=panel.index)
    available_weight = pd.Series(0.0, index=panel.index)
    for factor, weight in config.weights.items():
        values = panel[factor].where(panel[factor].notna())
        weighted = weighted.add(values.fillna(0.0) * weight)
        available_weight = available_weight.add(values.notna().astype(float) * weight)
    result["base_score"] = weighted / available_weight.replace(0.0, np.nan)

    volatility_penalty = ((result["volatility20"] - 0.45) / 0.35).clip(0, 1)
    result["score"] = (
        result["base_score"]
        - 0.25 * result["crowding_penalty"]
        - 0.20 * volatility_penalty
    ).clip(-1, 1).fillna(0.0)

    result["target_position"] = np.select(
        [
            result["score"] >= config.strong_buy_score,
            result["score"] >= config.buy_score,
            result["score"] >= config.reduce_score,
            result["score"] >= config.sell_score,
        ],
        [
            config.max_theme_weight,
            config.max_theme_weight * 0.5,
            config.max_theme_weight * 0.25,
            0.0,
        ],
        default=0.0,
    )
    result["action"] = np.select(
        [
            result["score"] >= config.strong_buy_score,
            result["score"] >= config.buy_score,
            result["score"] >= config.reduce_score,
            result["score"] >= config.sell_score,
        ],
        ["BUY_OR_ADD", "BUY_SMALL", "HOLD_OR_REDUCE", "SELL"],
        default="SELL",
    )
    return result


def factor_contributions(
    scored_panel: pd.DataFrame,
    config: MultiFactorConfig | None = None,
) -> pd.DataFrame:
    config = config or MultiFactorConfig()
    return pd.DataFrame(
        {
            name: scored_panel[name] * weight
            for name, weight in config.weights.items()
        },
        index=scored_panel.index,
    )
