"""Free point-in-time fund structure data from public Eastmoney F10 pages."""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd

try:
    import requests
except ImportError:
    requests = None


def _read_f10_tables(fund_code: str, data_type: str) -> list[pd.DataFrame]:
    if requests is None:
        raise RuntimeError("requests is required for fund metadata")
    url = "https://fundf10.eastmoney.com/FundArchivesDatas.aspx"
    response = requests.get(
        url,
        params={"type": data_type, "code": fund_code},
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=20,
    )
    response.raise_for_status()
    return pd.read_html(StringIO(response.text))


def _find_column(frame: pd.DataFrame, keywords: tuple[str, ...]) -> object | None:
    for column in frame.columns:
        text = str(column)
        if all(keyword in text for keyword in keywords):
            return column
    return None


def fetch_fund_structure_history(fund_code: str) -> pd.DataFrame:
    """Fetch scale/share and holder structure by report date."""
    scale_tables = _read_f10_tables(fund_code, "gmbd")
    holder_tables = _read_f10_tables(fund_code, "cyrjg")
    scale = max(scale_tables, key=len).copy()
    holders = max(holder_tables, key=len).copy()

    scale_date = _find_column(scale, ("截止", "日期"))
    scale_asset = _find_column(scale, ("净资产",))
    scale_share = _find_column(scale, ("基金份额",))
    holder_date = _find_column(holders, ("截止", "日期"))
    institution = _find_column(holders, ("机构", "比例"))
    holder_count = _find_column(holders, ("持有人户数",))
    if scale_date is None or holder_date is None:
        raise ValueError("Unexpected Eastmoney fund-structure table")

    scale_result = pd.DataFrame(
        {"report_date": pd.to_datetime(scale[scale_date], errors="coerce")}
    )
    if scale_asset is not None:
        scale_result["scale_billion_cny"] = (
            pd.to_numeric(
                scale[scale_asset].astype(str).str.replace("亿元", "", regex=False),
                errors="coerce",
            )
            / 10.0
        )
    if scale_share is not None:
        shares = pd.to_numeric(
            scale[scale_share].astype(str).str.replace("亿份", "", regex=False),
            errors="coerce",
        )
        scale_result["quarterly_share_growth"] = shares.pct_change(-1)

    holder_result = pd.DataFrame(
        {"report_date": pd.to_datetime(holders[holder_date], errors="coerce")}
    )
    if institution is not None:
        holder_result["institution_ratio"] = (
            pd.to_numeric(
                holders[institution].astype(str).str.replace("%", "", regex=False),
                errors="coerce",
            )
            / 100
        )
    if holder_count is not None:
        holder_result["holder_count"] = pd.to_numeric(
            holders[holder_count].astype(str).str.replace(",", "", regex=False),
            errors="coerce",
        )
    return (
        scale_result.merge(holder_result, on="report_date", how="outer")
        .dropna(subset=["report_date"])
        .sort_values("report_date")
        .reset_index(drop=True)
    )


def cache_fund_structure_history(
    fund_code: str,
    cache_dir: str | Path = "data/fund_structure",
) -> pd.DataFrame:
    result = fetch_fund_structure_history(fund_code)
    path = Path(cache_dir) / f"{fund_code}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(path, index=False, encoding="utf-8-sig")
    return result
