"""Best-effort public external factor loaders for fund timing research."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


ASSET_EXTERNAL_MAP = {
    "cpo_communication": {
        "etf": "515880",
        "csindex": "931160",
        "us_proxy": "ANET",
    },
    "memory_semiconductor_proxy": {
        "etf": "159995",
        "csindex": "931865",
        "us_proxy": "SOXX",
    },
    "artificial_intelligence": {
        "etf": "515070",
        "csindex": "930713",
        "us_proxy": "XLK",
    },
    "pcb_proxy_consumer_electronics": {
        "etf": "561100",
        "csindex": "931494",
        "us_proxy": "XLK",
    },
    "green_power": {
        "etf": "159669",
        "csindex": "931897",
    },
    "chemical": {
        "etf": "516020",
        "csindex": "000813",
        "commodity_proxy": "M00Y",
    },
    "nonferrous": {
        "etf": "512400",
        "csindex": "000819",
        "commodity_proxy": "HG00Y",
    },
}


def _akshare() -> Any | None:
    try:
        import akshare as ak  # type: ignore
    except Exception:
        return None
    return ak


def _date_column(frame: pd.DataFrame) -> str | None:
    for column in frame.columns:
        if str(column) in {"date", "日期", "净值日期"}:
            return str(column)
    return None


def _normalize_date_index(frame: pd.DataFrame) -> pd.DataFrame:
    column = _date_column(frame)
    result = frame.copy()
    if column is None:
        if isinstance(result.index, pd.DatetimeIndex):
            result.index = result.index.normalize()
            return result.sort_index()
        raise ValueError(f"No date column in frame: {list(frame.columns)}")
    result[column] = pd.to_datetime(result[column], errors="coerce")
    result = result.dropna(subset=[column]).set_index(column).sort_index()
    result.index.name = "date"
    result = result[~result.index.duplicated(keep="last")]
    return result


def _numeric(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in result.columns:
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def _load_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()


def _save_cache(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index_label="date", encoding="utf-8-sig")


def fetch_etf_factor_history(
    etf_code: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> pd.DataFrame:
    ak = _akshare()
    if ak is None:
        return pd.DataFrame()
    try:
        raw = ak.fund_etf_hist_em(
            symbol=etf_code,
            period="daily",
            start_date=start.strftime("%Y%m%d"),
            end_date=end.strftime("%Y%m%d"),
            adjust="qfq",
        )
    except Exception:
        exchange = "sh" if etf_code.startswith(("5", "6")) else "sz"
        raw = ak.fund_etf_hist_sina(symbol=f"{exchange}{etf_code}")
    frame = _numeric(_normalize_date_index(raw))
    columns = {
        "收盘": "etf_close",
        "close": "etf_close",
        "成交额": "etf_amount",
        "amount": "etf_amount",
        "volume": "etf_volume",
        "换手率": "etf_turnover",
        "振幅": "etf_amplitude",
        "涨跌幅": "etf_change_pct",
    }
    available = {source: target for source, target in columns.items() if source in frame}
    result = frame[list(available)].rename(columns=available)
    if {"high", "low", "close"}.issubset(frame.columns):
        result["etf_amplitude"] = (frame["high"] - frame["low"]) / frame["close"].shift(1) * 100
        result["etf_change_pct"] = frame["close"].pct_change(fill_method=None) * 100
    return result.loc[(result.index >= start) & (result.index <= end)]


def fetch_csindex_valuation(symbol: str) -> pd.DataFrame:
    ak = _akshare()
    if ak is None:
        return pd.DataFrame()
    raw = ak.stock_zh_index_value_csindex(symbol=symbol)
    frame = _numeric(_normalize_date_index(raw))
    columns = {
        "市盈率1": "pe_1",
        "市盈率2": "pe_2",
        "股息率1": "dividend_yield_1",
        "股息率2": "dividend_yield_2",
    }
    available = {source: target for source, target in columns.items() if source in frame}
    return frame[list(available)].rename(columns=available)


def fetch_global_future_history(symbol: str) -> pd.DataFrame:
    ak = _akshare()
    if ak is None:
        return pd.DataFrame()
    raw = ak.futures_global_hist_em(symbol=symbol)
    frame = _numeric(_normalize_date_index(raw))
    close_column = next(
        (column for column in ("收盘", "最新价", "close") if column in frame.columns),
        None,
    )
    if close_column is None:
        return pd.DataFrame()
    return frame[[close_column]].rename(columns={close_column: "commodity_close"})


def load_external_factor_panel(
    asset: str,
    index: pd.DatetimeIndex,
    cache_dir: str | Path = "data/external_factors",
    refresh: bool = False,
) -> pd.DataFrame:
    """Return an aligned external factor panel; missing sources stay NaN."""
    cache_path = Path(cache_dir) / f"{asset}.csv"
    cached = None if refresh else _load_cache(cache_path)
    if cached is not None:
        return cached.reindex(index).ffill()

    mapping = ASSET_EXTERNAL_MAP.get(asset, {})
    start = pd.Timestamp(index.min()) - pd.DateOffset(days=10)
    end = pd.Timestamp(index.max()) + pd.DateOffset(days=1)
    parts: list[pd.DataFrame] = []
    errors: list[str] = []

    if mapping.get("etf"):
        try:
            parts.append(fetch_etf_factor_history(str(mapping["etf"]), start, end))
        except Exception as exc:
            errors.append(f"etf:{type(exc).__name__}:{exc}")
    if mapping.get("csindex"):
        try:
            parts.append(fetch_csindex_valuation(str(mapping["csindex"])))
        except Exception as exc:
            errors.append(f"csindex:{type(exc).__name__}:{exc}")
    if mapping.get("commodity_proxy"):
        try:
            parts.append(fetch_global_future_history(str(mapping["commodity_proxy"])))
        except Exception as exc:
            errors.append(f"future:{type(exc).__name__}:{exc}")

    if parts:
        panel = pd.concat(parts, axis=1).sort_index()
        panel = panel[~panel.index.duplicated(keep="last")]
    else:
        panel = pd.DataFrame(index=index)
    panel = panel.reindex(index).ffill()
    panel["source_error_count"] = float(len(errors))
    _save_cache(cache_path, panel)
    return panel
