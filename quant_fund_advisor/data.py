"""CSV-first loaders and optional live market-data adapters."""

from __future__ import annotations

from datetime import date
import json
from pathlib import Path
import time
from typing import Any

import pandas as pd

try:
    import requests
except ImportError:
    requests = None


ASSET_TYPES = {"cn_stock", "cn_etf", "us_stock", "open_fund"}

_COLUMN_ALIASES = {
    "\u65e5\u671f": "date",
    "\u51c0\u503c\u65e5\u671f": "date",
    "\u5f00\u76d8": "open",
    "\u6536\u76d8": "close",
    "\u6700\u9ad8": "high",
    "\u6700\u4f4e": "low",
    "\u6210\u4ea4\u91cf": "volume",
    "\u6210\u4ea4\u989d": "amount",
    "\u6da8\u8dcc\u5e45": "change_pct",
    "\u5355\u4f4d\u51c0\u503c": "close",
}


def _normalize_open_fund_index(
    index: pd.Index | pd.DatetimeIndex | pd.Series,
) -> pd.DatetimeIndex:
    """Align public NAV timestamps to Beijing fund dates.

    Eastmoney's JS payload uses UTC millisecond timestamps that land on the
    prior calendar day in naive parsing. Treat all open-fund timestamps as UTC,
    convert to Asia/Shanghai, then normalize to the local date.
    """
    normalized = pd.DatetimeIndex(pd.to_datetime(index, errors="coerce"))
    if normalized.tz is None:
        normalized = normalized.tz_localize("UTC")
    return normalized.tz_convert("Asia/Shanghai").tz_localize(None).normalize()


def _normalize_open_fund_series(series: pd.Series) -> pd.Series:
    result = series.copy()
    result.index = _normalize_open_fund_index(result.index)
    result = (
        result.groupby(level=0).last().sort_index()
    )
    result.index.name = "date"
    return result.dropna()


def load_price_csv(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, parse_dates=["date"])
    if {"asset", "close"}.issubset(frame.columns):
        frame = frame.pivot(index="date", columns="asset", values="close")
    else:
        frame = frame.set_index("date")
    return frame.sort_index().apply(pd.to_numeric, errors="coerce")


def load_fund_universe(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"fund_code": str})
    required = {"fund_code", "fund_name", "sector", "available_on_alipay"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Fund universe missing columns: {sorted(missing)}")
    flag = frame["available_on_alipay"].astype(str).str.lower()
    return frame[flag.isin({"1", "true", "yes", "y"})].copy()


def load_market_manifest(path: str | Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype={"symbol": str}).fillna({"adjust": ""})
    required = {"group", "sector", "asset_type", "symbol"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Market manifest missing columns: {sorted(missing)}")
    unknown = set(frame["asset_type"]) - ASSET_TYPES
    if unknown:
        raise ValueError(f"Unsupported asset types: {sorted(unknown)}")
    duplicated = frame.duplicated(["group", "sector"], keep=False)
    if duplicated.any():
        pairs = frame.loc[duplicated, ["group", "sector"]].drop_duplicates()
        raise ValueError(f"Duplicate group/sector pairs: {pairs.to_dict('records')}")
    return frame


def _akshare() -> Any:
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError(
            'Live data requires AKShare: pip install -e ".[live]"'
        ) from exc
    return ak


def _compact_date(value: str | date | pd.Timestamp | None) -> str:
    if value is None:
        return ""
    return pd.Timestamp(value).strftime("%Y%m%d")


def normalize_history(frame: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """Convert AKShare output into a stable date-indexed OHLCV schema."""
    normalized = frame.rename(columns=lambda col: _COLUMN_ALIASES.get(str(col), str(col).lower()))
    if "date" not in normalized.columns:
        if normalized.index.name and _COLUMN_ALIASES.get(str(normalized.index.name)) == "date":
            index_name = normalized.index.name
            normalized = normalized.reset_index().rename(columns={index_name: "date"})
        else:
            raise ValueError(
                f"Data source returned no date column for {symbol}: "
                f"{list(frame.columns)}"
            )
    if "close" not in normalized.columns:
        raise ValueError(
            f"Data source returned no close/NAV column for {symbol}: "
            f"{list(frame.columns)}"
        )
    normalized["date"] = pd.to_datetime(normalized["date"], errors="coerce")
    normalized = normalized.dropna(subset=["date"]).set_index("date").sort_index()
    normalized = normalized[~normalized.index.duplicated(keep="last")]
    for column in normalized.columns:
        normalized[column] = pd.to_numeric(normalized[column], errors="coerce")
    normalized.index.name = "date"
    normalized.attrs["symbol"] = symbol
    return normalized.dropna(subset=["close"])


def fetch_history(
    asset_type: str,
    symbol: str,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
    adjust: str = "",
) -> pd.DataFrame:
    """Fetch free daily history through AKShare and return normalized data."""
    if asset_type not in ASSET_TYPES:
        raise ValueError(f"Unsupported asset type: {asset_type}")

    ak = _akshare()
    start_date = _compact_date(start) or "19900101"
    end_date = _compact_date(end) or pd.Timestamp.today().strftime("%Y%m%d")

    if asset_type == "cn_stock":
        frame = ak.stock_zh_a_hist(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    elif asset_type == "cn_etf":
        frame = ak.fund_etf_hist_em(
            symbol=symbol,
            period="daily",
            start_date=start_date,
            end_date=end_date,
            adjust=adjust,
        )
    elif asset_type == "us_stock":
        if "." in symbol:
            frame = ak.stock_us_hist(
                symbol=symbol,
                period="daily",
                start_date=start_date,
                end_date=end_date,
                adjust=adjust or "",
            )
        else:
            frame = ak.stock_us_daily(symbol=symbol, adjust=adjust or "")
    else:
        frame = ak.fund_open_fund_info_em(
            symbol=symbol,
            indicator="\u5355\u4f4d\u51c0\u503c\u8d70\u52bf",
        )

    result = normalize_history(frame, symbol)
    if start is not None:
        result = result.loc[result.index >= pd.Timestamp(start)]
    if end is not None:
        result = result.loc[result.index <= pd.Timestamp(end)]
    if result.empty:
        raise ValueError(
            f"No history returned for {asset_type}:{symbol} in the requested range"
        )
    return result


def fetch_open_fund_nav(fund_code: str) -> pd.Series:
    history = fetch_history("open_fund", fund_code)
    return history["close"].rename(fund_code)


def fetch_open_fund_nav_eastmoney(
    fund_code: str,
    timeout: float = 20.0,
) -> pd.Series:
    """Fallback: fetch complete NAV history from Eastmoney's public fund page."""
    if requests is None:
        raise RuntimeError("The Eastmoney fallback requires requests")

    url = f"https://fund.eastmoney.com/pingzhongdata/{fund_code}.js"
    response = requests.get(
        url,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()
    marker = "var Data_netWorthTrend = "
    start = response.text.find(marker)
    if start < 0:
        raise ValueError(f"No NAV series found for fund {fund_code}")
    start += len(marker)
    end = response.text.find(";", start)
    records = json.loads(response.text[start:end])
    frame = pd.DataFrame(records)
    if frame.empty or not {"x", "y"}.issubset(frame.columns):
        raise ValueError(f"Invalid NAV series returned for fund {fund_code}")
    index = _normalize_open_fund_index(pd.to_datetime(frame["x"], unit="ms", utc=True))
    return _normalize_open_fund_series(pd.Series(
        pd.to_numeric(frame["y"], errors="coerce").to_numpy(),
        index=index,
        name=fund_code,
    ))


def load_or_fetch_open_fund_nav(
    fund_code: str,
    cache_dir: str | Path = "data/nav_cache",
    max_age_hours: float = 20.0,
    retries: int = 3,
) -> pd.Series:
    """Use a local cache, AKShare primary source and Eastmoney fallback."""
    cache_path = Path(cache_dir) / f"{fund_code}.csv"
    if cache_path.exists():
        age_hours = (
            pd.Timestamp.now().timestamp() - cache_path.stat().st_mtime
        ) / 3600
        if age_hours <= max_age_hours:
            cached = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")
            cached_series = _normalize_open_fund_series(
                cached["close"].rename(fund_code)
            )
            if not cached_series.index.equals(cached.index):
                cached_series.rename("close").to_csv(
                    cache_path, index_label="date", encoding="utf-8-sig"
                )
            return cached_series

    errors: list[str] = []
    loaders = (
        lambda: fetch_open_fund_nav(fund_code),
        lambda: fetch_open_fund_nav_eastmoney(fund_code),
    )
    for loader in loaders:
        for attempt in range(retries):
            try:
                series = _normalize_open_fund_series(loader())
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                series.rename("close").to_csv(
                    cache_path, index_label="date", encoding="utf-8-sig"
                )
                return series
            except Exception as exc:
                errors.append(f"{type(exc).__name__}: {exc}")
                if attempt + 1 < retries:
                    time.sleep(1.5 * (attempt + 1))

    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")
        cached_series = _normalize_open_fund_series(
            cached["close"].rename(fund_code)
        )
        cached_series.rename("close").to_csv(
            cache_path, index_label="date", encoding="utf-8-sig"
        )
        return cached_series
    raise RuntimeError(
        f"Unable to obtain NAV history for {fund_code}. " + " | ".join(errors)
    )


def fetch_fund_universe_nav(universe: pd.DataFrame) -> pd.DataFrame:
    series = [fetch_open_fund_nav(code) for code in universe["fund_code"]]
    return pd.concat(series, axis=1).sort_index()


def fetch_manifest_prices(
    manifest: pd.DataFrame,
    group: str,
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    selected = manifest.loc[manifest["group"] == group]
    if selected.empty:
        raise ValueError(f"Market manifest has no rows for group: {group}")

    prices: dict[str, pd.Series] = {}
    for row in selected.itertuples(index=False):
        history = fetch_history(
            row.asset_type,
            row.symbol,
            start=start,
            end=end,
            adjust=str(getattr(row, "adjust", "") or ""),
        )
        prices[row.sector] = history["close"].rename(row.sector)
    return pd.concat(prices.values(), axis=1).sort_index()


def fetch_basket_prices(
    assets: list[dict[str, str]],
    start: str | date | pd.Timestamp | None = None,
    end: str | date | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch several assets as a price matrix keyed by each asset's name."""
    prices: dict[str, pd.Series] = {}
    for asset in assets:
        history = fetch_history(
            asset["asset_type"],
            asset["symbol"],
            start=start,
            end=end,
            adjust=asset.get("adjust", ""),
        )
        name = asset.get("name", asset["symbol"])
        prices[name] = history["close"].rename(name)
    return pd.concat(prices.values(), axis=1).sort_index()
