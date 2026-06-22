"""Translate daily model conviction into practical open-fund target positions."""

from __future__ import annotations

import pandas as pd


def discretize_position(
    position: pd.Series,
    levels: tuple[float, ...] = (
        0.0,
        0.25,
        0.50,
        0.60,
        0.70,
        0.80,
        0.90,
        1.0,
    ),
) -> pd.Series:
    """Map noisy forecasts to a small set of practical allocation levels."""
    clipped = position.fillna(0.0).clip(0, 1)
    level_array = pd.Series(levels, dtype=float).to_numpy()
    values = [
        float(level_array[abs(level_array - value).argmin()])
        for value in clipped.to_numpy()
    ]
    return pd.Series(values, index=clipped.index)


def apply_open_fund_execution_policy(
    position: pd.Series,
    minimum_days_between_increases: int = 5,
    confirmation_days: int = 2,
) -> pd.Series:
    """Require persistent target changes and throttle subscriptions/additions."""
    desired = discretize_position(position)
    actual = []
    current = 0.0
    last_increase = -minimum_days_between_increases
    candidate = current
    candidate_days = 0
    for location, target in enumerate(desired):
        if target == current:
            candidate = current
            candidate_days = 0
        else:
            if target == candidate:
                candidate_days += 1
            else:
                candidate = float(target)
                candidate_days = 1
            if candidate_days >= confirmation_days:
                if candidate < current:
                    current = candidate
                    candidate_days = 0
                elif location - last_increase >= minimum_days_between_increases:
                    current = candidate
                    last_increase = location
                    candidate_days = 0
        actual.append(current)
    return pd.Series(actual, index=desired.index, name="target_position")
