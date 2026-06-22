"""Lagged US-to-China sector correlation analysis."""

from __future__ import annotations

import pandas as pd


def lagged_sector_correlations(
    us_prices: pd.DataFrame,
    cn_prices: pd.DataFrame,
    window: int = 120,
    max_lag: int = 5,
) -> pd.DataFrame:
    """Return latest rolling correlations; positive lag means US leads China."""
    us_ret = us_prices.pct_change(fill_method=None)
    cn_ret = cn_prices.pct_change(fill_method=None)
    rows: list[dict] = []
    for sector in sorted(set(us_ret.columns) & set(cn_ret.columns)):
        pair = pd.concat(
            [us_ret[sector].rename("us"), cn_ret[sector].rename("cn")], axis=1
        ).dropna()
        for lag in range(max_lag + 1):
            aligned = pd.concat(
                [pair["us"].shift(lag).rename("us"), pair["cn"]], axis=1
            ).dropna().tail(window)
            corr = aligned["us"].corr(aligned["cn"]) if len(aligned) >= 30 else float("nan")
            rows.append(
                {
                    "sector": sector,
                    "us_lead_days": lag,
                    "correlation": corr,
                    "observations": len(aligned),
                }
            )
    return pd.DataFrame(rows)


def best_lead_relationships(correlations: pd.DataFrame) -> pd.DataFrame:
    valid = correlations.dropna(subset=["correlation"]).copy()
    if valid.empty:
        return valid
    valid["abs_correlation"] = valid["correlation"].abs()
    idx = valid.groupby("sector")["abs_correlation"].idxmax()
    return valid.loc[idx].sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def transmission_signal(
    us_prices: pd.DataFrame,
    relationships: pd.DataFrame,
    lookback: int = 5,
) -> pd.DataFrame:
    """Build a daily sector signal using only previously known US returns."""
    signal = pd.DataFrame(0.0, index=us_prices.index, columns=us_prices.columns)
    us_momentum = us_prices.pct_change(lookback, fill_method=None)
    for row in relationships.itertuples():
        if row.sector not in signal.columns:
            continue
        lag = max(1, int(row.us_lead_days))
        signal[row.sector] = us_momentum[row.sector].shift(lag) * float(row.correlation)
    return signal

