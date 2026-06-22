"""Composite asset scoring and trade-state generation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .indicators import (
    annualized_volatility,
    cross_sectional_zscore,
    rolling_max_drawdown,
    trend_strength,
)


@dataclass(frozen=True)
class AdvisorConfig:
    momentum_windows: tuple[int, int, int] = (20, 60, 120)
    momentum_weight: float = 0.35
    trend_weight: float = 0.25
    risk_weight: float = 0.20
    transmission_weight: float = 0.15
    news_weight: float = 0.05
    buy_threshold: float = 0.35
    sell_threshold: float = -0.15


def score_assets(
    prices: pd.DataFrame,
    transmission: pd.DataFrame | None = None,
    news_scores: pd.Series | None = None,
    config: AdvisorConfig | None = None,
) -> pd.DataFrame:
    config = config or AdvisorConfig()
    w1, w2, w3 = config.momentum_windows
    momentum = (
        cross_sectional_zscore(prices.pct_change(w1, fill_method=None)) * 0.5
        + cross_sectional_zscore(prices.pct_change(w2, fill_method=None)) * 0.3
        + cross_sectional_zscore(prices.pct_change(w3, fill_method=None)) * 0.2
    )
    trend = cross_sectional_zscore(trend_strength(prices))
    volatility = cross_sectional_zscore(annualized_volatility(prices))
    drawdown = cross_sectional_zscore(rolling_max_drawdown(prices))
    risk = (-0.6 * volatility + 0.4 * drawdown).clip(-3, 3)

    if transmission is None:
        transmission_component = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    else:
        transmission_component = cross_sectional_zscore(
            transmission.reindex_like(prices).ffill().fillna(0.0)
        )

    if news_scores is None:
        news_component = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    else:
        aligned = news_scores.reindex(prices.columns).fillna(0.0)
        news_component = pd.DataFrame(
            np.tile(aligned.to_numpy(), (len(prices), 1)),
            index=prices.index,
            columns=prices.columns,
        )

    score = (
        momentum * config.momentum_weight
        + trend * config.trend_weight
        + risk * config.risk_weight
        + transmission_component * config.transmission_weight
        + news_component * config.news_weight
    )
    return score.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def latest_recommendations(
    scores: pd.DataFrame,
    config: AdvisorConfig | None = None,
) -> pd.DataFrame:
    config = config or AdvisorConfig()
    latest = scores.iloc[-1].sort_values(ascending=False).rename("score").to_frame()
    latest["action"] = np.select(
        [latest["score"] >= config.buy_threshold, latest["score"] <= config.sell_threshold],
        ["BUY", "SELL"],
        default="HOLD",
    )
    latest["confidence"] = (latest["score"].abs() / 1.5).clip(0, 1)
    return latest

