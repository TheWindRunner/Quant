"""Vectorized indicators used by the scoring and backtest engines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def returns(prices: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    return prices.pct_change(periods, fill_method=None)


def annualized_volatility(prices: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    return returns(prices).rolling(window).std() * np.sqrt(252)


def rolling_max_drawdown(prices: pd.DataFrame, window: int = 120) -> pd.DataFrame:
    rolling_peak = prices.rolling(window, min_periods=max(20, window // 4)).max()
    return prices / rolling_peak - 1.0


def trend_strength(prices: pd.DataFrame, fast: int = 20, slow: int = 60) -> pd.DataFrame:
    fast_ma = prices.rolling(fast).mean()
    slow_ma = prices.rolling(slow).mean()
    return fast_ma / slow_ma - 1.0


def cross_sectional_zscore(values: pd.DataFrame) -> pd.DataFrame:
    mean = values.mean(axis=1)
    std = values.std(axis=1).replace(0.0, np.nan)
    return values.sub(mean, axis=0).div(std, axis=0).fillna(0.0).clip(-3, 3)


def expanding_zscore(series: pd.Series, min_periods: int = 20) -> pd.Series:
    mean = series.expanding(min_periods=min_periods).mean()
    std = series.expanding(min_periods=min_periods).std().replace(0.0, np.nan)
    return ((series - mean) / std).fillna(0.0).clip(-3, 3)

