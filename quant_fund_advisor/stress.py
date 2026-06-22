"""Robustness tests for model families and open-fund execution costs."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .experiment import evaluate_signal
from .model_zoo import (
    dual_moving_average,
    time_series_momentum,
    volatility_managed_momentum,
)


def moving_average_sensitivity(
    nav: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
    pairs: tuple[tuple[int, int], ...] = (
        (10, 40),
        (15, 50),
        (20, 60),
        (25, 75),
        (30, 90),
    ),
) -> pd.DataFrame:
    rows = []
    for fast, slow in pairs:
        signal = dual_moving_average(nav, fast, slow)
        metrics = evaluate_signal(nav, signal, start, end)
        rows.append({"fast": fast, "slow": slow, **metrics})
    return pd.DataFrame(rows)


def block_bootstrap_nav(
    nav: pd.Series,
    simulations: int = 200,
    block_size: int = 10,
    seed: int = 42,
) -> list[pd.Series]:
    """Resample return blocks to retain short-horizon serial dependence."""
    returns = nav.pct_change(fill_method=None).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    paths = []
    required_blocks = int(np.ceil(len(returns) / block_size))
    max_start = max(1, len(returns) - block_size + 1)
    for simulation in range(simulations):
        starts = rng.integers(0, max_start, size=required_blocks)
        sampled = np.concatenate(
            [returns[start : start + block_size] for start in starts]
        )[: len(returns)]
        values = nav.iloc[0] * np.cumprod(np.r_[1.0, 1.0 + sampled])
        paths.append(
            pd.Series(
                values,
                index=nav.index,
                name=f"bootstrap_{simulation}",
            )
        )
    return paths


def bootstrap_model_comparison(
    nav: pd.Series,
    simulations: int = 200,
    block_size: int = 10,
) -> pd.DataFrame:
    rows = []
    for simulation, path in enumerate(
        block_bootstrap_nav(nav, simulations, block_size)
    ):
        models = [
            dual_moving_average(path),
            time_series_momentum(path),
            volatility_managed_momentum(path),
        ]
        buy_hold_return = float(path.iloc[-1] / path.iloc[0] - 1)
        for model in models:
            metrics = evaluate_signal(path, model, path.index[0], path.index[-1])
            rows.append(
                {
                    "simulation": simulation,
                    "model": model.name,
                    "excess_return": float(metrics["total_return"]) - buy_hold_return,
                    "max_drawdown": float(metrics["max_drawdown"]),
                }
            )
    return pd.DataFrame(rows)


def summarize_bootstrap(results: pd.DataFrame) -> pd.DataFrame:
    return results.groupby("model").agg(
        median_excess_return=("excess_return", "median"),
        probability_beating_buy_hold=("excess_return", lambda x: float((x > 0).mean())),
        fifth_percentile_excess=("excess_return", lambda x: float(x.quantile(0.05))),
        median_max_drawdown=("max_drawdown", "median"),
    )

