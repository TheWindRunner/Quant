"""Deterministic synthetic data for smoke tests and first-run demos."""

from __future__ import annotations

import numpy as np
import pandas as pd


def make_demo_prices(
    periods: int = 756,
    sectors: tuple[str, ...] = (
        "technology",
        "semiconductor",
        "healthcare",
        "financials",
        "materials",
        "energy",
    ),
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=periods)
    common = rng.normal(0.0002, 0.007, size=periods)
    us_data: dict[str, np.ndarray] = {}
    cn_data: dict[str, np.ndarray] = {}
    for index, sector in enumerate(sectors):
        cycle = np.sin(np.arange(periods) / (35 + index * 3)) * 0.0008
        us_ret = common * 0.55 + cycle + rng.normal(0, 0.009, periods)
        cn_ret = (
            np.roll(us_ret, 1) * (0.18 + index * 0.025)
            + common * 0.45
            + rng.normal(0, 0.011, periods)
        )
        cn_ret[0] = 0.0
        us_data[sector] = 100 * np.cumprod(1 + us_ret)
        cn_data[sector] = 100 * np.cumprod(1 + cn_ret)
    return pd.DataFrame(us_data, index=dates), pd.DataFrame(cn_data, index=dates)

