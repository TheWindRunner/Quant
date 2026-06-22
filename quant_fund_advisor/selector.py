"""Map sector signals to an Alipay-confirmed fund universe."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .indicators import annualized_volatility, trend_strength


def rank_funds(
    universe: pd.DataFrame,
    fund_nav: pd.DataFrame,
    sector_recommendations: pd.DataFrame,
    top_per_sector: int = 2,
) -> pd.DataFrame:
    """Rank only funds explicitly marked as available in the user's Alipay list."""
    required = {"fund_code", "fund_name", "sector", "available_on_alipay"}
    missing = required - set(universe.columns)
    if missing:
        raise ValueError(f"Fund universe missing columns: {sorted(missing)}")

    eligible_flag = universe["available_on_alipay"].astype(str).str.lower()
    eligible = universe[eligible_flag.isin({"1", "true", "yes", "y"})].copy()
    eligible["fund_code"] = eligible["fund_code"].astype(str).str.zfill(6)
    nav = fund_nav.copy()
    nav.columns = nav.columns.astype(str).str.zfill(6)

    latest_trend = trend_strength(nav).iloc[-1]
    latest_vol = annualized_volatility(nav).iloc[-1]
    rows = []
    for row in eligible.itertuples():
        code = row.fund_code
        if code not in nav.columns or row.sector not in sector_recommendations.index:
            continue
        sector_row = sector_recommendations.loc[row.sector]
        purchase_fee = float(getattr(row, "purchase_fee", 0.0) or 0.0)
        redemption_fee = float(getattr(row, "redemption_fee", 0.0) or 0.0)
        holding_days = float(getattr(row, "min_holding_days", 0.0) or 0.0)
        tradability_penalty = purchase_fee + redemption_fee + max(0, holding_days - 7) / 3650
        fund_score = (
            float(sector_row["score"])
            + 2.0 * float(latest_trend.get(code, 0.0))
            - 0.25 * float(latest_vol.get(code, 0.0))
            - 5.0 * tradability_penalty
        )
        rows.append(
            {
                "fund_code": code,
                "fund_name": row.fund_name,
                "sector": row.sector,
                "sector_action": sector_row["action"],
                "sector_score": float(sector_row["score"]),
                "fund_score": fund_score,
                "annual_volatility": float(latest_vol.get(code, np.nan)),
                "purchase_fee": purchase_fee,
                "redemption_fee": redemption_fee,
                "min_holding_days": holding_days,
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["rank_in_sector"] = result.groupby("sector")["fund_score"].rank(
        method="first", ascending=False
    )
    return (
        result[result["rank_in_sector"] <= top_per_sector]
        .sort_values(["sector_score", "fund_score"], ascending=False)
        .reset_index(drop=True)
    )
