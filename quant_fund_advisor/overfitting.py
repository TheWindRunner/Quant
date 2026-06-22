"""Diagnostics for backtest selection bias and probability of overfitting."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def probability_of_backtest_overfitting(
    diagnostics: pd.DataFrame,
    fold_column: str = "fold_end",
    model_column: str = "model",
    score_column: str = "excess_return",
    max_splits: int = 252,
) -> dict[str, object]:
    """Approximate CSCV PBO from a fold-by-model performance matrix."""
    matrix = diagnostics.pivot_table(
        index=fold_column,
        columns=model_column,
        values=score_column,
        aggfunc="mean",
    ).dropna(axis=0, how="any")
    if len(matrix) < 4 or matrix.shape[1] < 2:
        return {
            "pbo": np.nan,
            "split_count": 0,
            "model_count": int(matrix.shape[1]),
            "selection_frequency": {},
        }
    fold_count = len(matrix)
    in_sample_size = fold_count // 2
    all_splits = list(combinations(range(fold_count), in_sample_size))
    # A split and its complement contain the same information in reverse.
    all_splits = all_splits[: max(1, min(max_splits, len(all_splits) // 2))]

    failures = 0
    selection_counts = {str(model): 0 for model in matrix.columns}
    out_of_sample_percentiles = []
    for in_sample_locations in all_splits:
        in_sample_set = set(in_sample_locations)
        out_sample_locations = [
            location
            for location in range(fold_count)
            if location not in in_sample_set
        ]
        in_sample = matrix.iloc[list(in_sample_locations)].mean(axis=0)
        selected = str(in_sample.idxmax())
        selection_counts[selected] += 1
        out_sample = matrix.iloc[out_sample_locations].mean(axis=0)
        percentile = float(out_sample.rank(pct=True)[selected])
        out_of_sample_percentiles.append(percentile)
        failures += int(percentile <= 0.50)

    split_count = len(all_splits)
    return {
        "pbo": float(failures / split_count),
        "split_count": split_count,
        "model_count": int(matrix.shape[1]),
        "median_oos_percentile": float(np.median(out_of_sample_percentiles)),
        "selection_frequency": {
            model: count / split_count
            for model, count in selection_counts.items()
            if count
        },
    }
