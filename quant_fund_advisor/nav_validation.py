"""Integrity checks for cached mutual-fund NAV histories."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .data import _normalize_open_fund_index


def validate_nav_series(
    nav: pd.Series,
    minimum_rows: int = 500,
    maximum_gap_days: int = 12,
) -> dict[str, object]:
    series = nav.dropna().sort_index()
    duplicates = int(series.index.duplicated().sum())
    daily_change = series.pct_change(fill_method=None)
    gaps = series.index.to_series().diff().dt.days
    issues = []
    if len(series) < minimum_rows:
        issues.append(f"too_few_rows:{len(series)}")
    if duplicates:
        issues.append(f"duplicate_dates:{duplicates}")
    if (series <= 0).any():
        issues.append("non_positive_nav")
    if daily_change.abs().max() > 0.25:
        issues.append(f"extreme_daily_change:{daily_change.abs().max():.4f}")
    if gaps.max() > maximum_gap_days:
        issues.append(f"large_calendar_gap:{int(gaps.max())}")
    return {
        "rows": len(series),
        "start": series.index.min(),
        "end": series.index.max(),
        "duplicates": duplicates,
        "max_abs_daily_change": float(daily_change.abs().max()),
        "max_calendar_gap_days": int(gaps.max()) if not gaps.dropna().empty else 0,
        "valid": not issues,
        "issues": "|".join(issues),
    }


def validate_nav_cache(
    cache_dir: str | Path,
    codes: tuple[str, ...] = ("007817", "008887", "008585"),
) -> pd.DataFrame:
    rows = []
    directory = Path(cache_dir)
    for code in codes:
        path = directory / f"{code}.csv"
        if not path.exists():
            rows.append({"fund_code": code, "valid": False, "issues": "missing_file"})
            continue
        frame = pd.read_csv(path, parse_dates=["date"])
        if "close" not in frame:
            rows.append({"fund_code": code, "valid": False, "issues": "missing_close"})
            continue
        nav = pd.Series(
            pd.to_numeric(frame["close"], errors="coerce").to_numpy(),
            index=_normalize_open_fund_index(frame["date"]),
            name=code,
        )
        rows.append({"fund_code": code, **validate_nav_series(nav)})
    return pd.DataFrame(rows)
