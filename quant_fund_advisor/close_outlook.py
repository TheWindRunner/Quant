"""Generate the 18:30 close review and next-session outlook report."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
from typing import Any

import pandas as pd
import requests

from .forward_test import score_forward_ledger, score_rotation_forward_ledger
from .intraday_estimate import (
    EstimatePolicy,
    adaptive_policy_from_history,
    calibration_metrics,
    fetch_eastmoney_intraday_estimate,
    reconcile_estimate_history,
)
from .run_research import RESEARCH_FUNDS
from .sector_rotation import rotation_configs


FUND_TO_ETF = {
    "007817": ("sh515880", "515880", "通信ETF国泰"),
    "008887": ("sz159995", "159995", "芯片ETF"),
    "008585": ("sh515070", "515070", "人工智能ETF华夏"),
}

FOCUS_BASKETS = {
    "CPO": [
        ("sz300308", "中际旭创"),
        ("sz300502", "新易盛"),
        ("sz300394", "天孚通信"),
        ("sz002281", "光迅科技"),
    ],
    "存储": [
        ("sh603986", "兆易创新"),
        ("sz301308", "江波龙"),
        ("sh688525", "佰维存储"),
        ("sz300223", "北京君正"),
    ],
    "PCB": [
        ("sz300476", "胜宏科技"),
        ("sz002463", "沪电股份"),
        ("sz002916", "深南电路"),
        ("sh600183", "生益科技"),
    ],
    "AI": [
        ("sz300308", "中际旭创"),
        ("sh688041", "海光信息"),
        ("sh603019", "中科曙光"),
        ("sz002230", "科大讯飞"),
    ],
}

OTHER_TECH_SCAN = {
    "消费电子": [
        ("sz300408", "三环集团"),
        ("sh603160", "汇顶科技"),
    ],
    "高端PCB延伸": [
        ("sh603920", "世运电路"),
        ("sz300476", "胜宏科技"),
    ],
}

MARKET_SYMBOLS = [
    "sh000001",
    "sz399001",
    "sz399006",
    "sh515880",
    "sz159995",
    "sh515070",
    "sz300308",
    "sz300502",
    "sz300394",
    "sz002281",
    "sh603986",
    "sz301308",
    "sh688525",
    "sz300223",
    "sz300476",
    "sz002463",
    "sz002916",
    "sh600183",
    "sh688041",
    "sh603019",
    "sz002230",
    "sz300408",
    "sh603920",
    "sh603160",
]

US_SYMBOLS = [
    "hf_ES",
    "hf_NQ",
    "hf_YM",
    "gb_cohr",
    "gb_lite",
    "gb_aaoi",
    "gb_mu",
    "gb_wdc",
    "gb_stx",
    "gb_nvda",
    "gb_smci",
    "gb_dell",
]

CNINFO_COMPANIES = [
    ("szse", "sz", "002281,光迅科技"),
    ("szse", "sz", "300502,新易盛"),
    ("sse", "sh", "688525,佰维存储"),
    ("szse", "sz", "300223,北京君正"),
    ("szse", "sz", "300476,胜宏科技"),
    ("sse", "sh", "600183,生益科技"),
    ("sse", "sh", "603019,中科曙光"),
    ("sse", "sh", "688041,海光信息"),
]

CLS_KEYWORDS = ("收评", "半导体", "存储", "PCB", "AI", "算力", "消费电子", "光模块", "CPO")


@dataclass(frozen=True)
class CnQuote:
    symbol: str
    name: str
    close: float
    previous_close: float
    change_pct: float
    amount_cny: float
    quote_date: str
    quote_time: str


@dataclass(frozen=True)
class UsQuote:
    symbol: str
    name: str
    level: float
    change_pct: float
    timestamp: str
    previous_close: float | None = None


def _http_get(url: str, *, params: dict[str, Any] | None = None) -> requests.Response:
    response = requests.get(
        url,
        params=params,
        timeout=20,
        headers={"User-Agent": "Mozilla/5.0", "Referer": "https://finance.sina.com.cn/"},
    )
    response.raise_for_status()
    return response


def _load_or_fetch_official_nav(fund_code: str) -> pd.Series:
    cache_path = Path("data/nav_cache") / f"{fund_code}.csv"
    if cache_path.exists():
        cached = pd.read_csv(cache_path, parse_dates=["date"]).set_index("date")
    else:
        cached = pd.DataFrame(columns=["close"])
    response = _http_get(f"https://fund.eastmoney.com/pingzhongdata/{fund_code}.js")
    marker = "var Data_netWorthTrend = "
    start = response.text.find(marker)
    if start < 0:
        if cache_path.exists():
            return cached["close"].rename(fund_code)
        raise RuntimeError(f"Unable to locate NAV history for {fund_code}")
    start += len(marker)
    end = response.text.find(";", start)
    records = json.loads(response.text[start:end])
    frame = pd.DataFrame(records)
    index = (
        pd.to_datetime(frame["x"], unit="ms", utc=True)
        .dt.tz_convert("Asia/Shanghai")
        .dt.tz_localize(None)
    )
    series = pd.Series(
        pd.to_numeric(frame["y"], errors="coerce").to_numpy(),
        index=index,
        name=fund_code,
    ).dropna().sort_index()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    series.rename("close").to_csv(cache_path, index_label="date", encoding="utf-8-sig")
    return series


def _fetch_sina_raw(symbols: list[str]) -> dict[str, list[str]]:
    response = _http_get(f"https://hq.sinajs.cn/list={','.join(symbols)}")
    text = response.content.decode("gbk", errors="ignore")
    result: dict[str, list[str]] = {}
    for line in text.strip().splitlines():
        if '="' not in line:
            continue
        prefix, raw = line.split('="', 1)
        symbol = prefix.replace("var hq_str_", "", 1)
        result[symbol] = raw.rstrip('";').split(",")
    return result


def _parse_cn_quote(symbol: str, fields: list[str]) -> CnQuote:
    name = fields[0]
    previous_close = float(fields[2])
    close = float(fields[3])
    amount_cny = float(fields[9])
    if symbol.startswith(("sh000", "sz399")):
        quote_date = fields[30]
        quote_time = fields[31]
    else:
        quote_date = fields[30]
        quote_time = fields[31]
    change_pct = (close / previous_close - 1.0) * 100 if previous_close else 0.0
    return CnQuote(
        symbol=symbol,
        name=name,
        close=close,
        previous_close=previous_close,
        change_pct=change_pct,
        amount_cny=amount_cny,
        quote_date=quote_date,
        quote_time=quote_time,
    )


def _parse_us_quote(symbol: str, fields: list[str]) -> UsQuote:
    if symbol.startswith("hf_"):
        level = float(fields[0])
        previous_close = float(fields[5]) if fields[5] else None
        change_pct = (
            (level / previous_close - 1.0) * 100 if previous_close else 0.0
        )
        timestamp = f"{fields[12]} {fields[6]}"
        name = fields[13]
        return UsQuote(
            symbol=symbol,
            name=name,
            level=level,
            change_pct=change_pct,
            timestamp=timestamp,
            previous_close=previous_close,
        )
    level = float(fields[1])
    change_pct = float(fields[2])
    timestamp = fields[3]
    previous_close = float(fields[-1]) if fields[-1] else None
    return UsQuote(
        symbol=symbol,
        name=fields[0],
        level=level,
        change_pct=change_pct,
        timestamp=timestamp,
        previous_close=previous_close,
    )


def _fetch_cn_quotes() -> dict[str, CnQuote]:
    raw = _fetch_sina_raw(MARKET_SYMBOLS)
    return {symbol: _parse_cn_quote(symbol, raw[symbol]) for symbol in raw}


def _fetch_us_quotes() -> dict[str, UsQuote]:
    raw = _fetch_sina_raw(US_SYMBOLS)
    return {symbol: _parse_us_quote(symbol, raw[symbol]) for symbol in raw}


def _fetch_cninfo_announcements(start_date: str, end_date: str) -> list[dict[str, Any]]:
    url = "http://www.cninfo.com.cn/new/hisAnnouncement/query"
    headers = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "http://www.cninfo.com.cn/",
        "X-Requested-With": "XMLHttpRequest",
    }
    results: list[dict[str, Any]] = []
    for column, plate, stock in CNINFO_COMPANIES:
        response = requests.post(
            url,
            data={
                "pageNum": "1",
                "pageSize": "5",
                "column": column,
                "tabName": "fulltext",
                "plate": plate,
                "stock": stock,
                "searchkey": "",
                "secid": "",
                "category": "",
                "trade": "",
                "seDate": f"{start_date}~{end_date}",
                "sortName": "",
                "sortType": "",
                "isHLtitle": "true",
            },
            headers=headers,
            timeout=20,
        )
        response.raise_for_status()
        payload = response.json()
        results.extend(payload.get("announcements") or [])
    return results


def _fetch_cls_articles(limit: int = 6) -> list[dict[str, str]]:
    homepage = _http_get("https://www.cls.cn").text
    pairs = re.findall(
        r'<a[^>]+href="(/detail/\d+)"[^>]*>(.*?)</a>',
        homepage,
        flags=re.S,
    )
    articles: list[dict[str, str]] = []
    seen: set[str] = set()
    for href, raw_text in pairs:
        title = re.sub(r"<[^>]+>", "", raw_text)
        title = html.unescape(title).strip()
        if not title or href in seen:
            continue
        if not any(keyword in title for keyword in CLS_KEYWORDS):
            continue
        seen.add(href)
        page = _http_get(f"https://www.cls.cn{href}").text
        description_match = re.search(r'"description" content="([^"]+)"', page)
        timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", page)
        description = description_match.group(1) if description_match else title
        articles.append(
            {
                "title": title,
                "url": f"https://www.cls.cn{href}",
                "description": description,
                "timestamp": timestamp_match.group(1) if timestamp_match else "",
            }
        )
        if len(articles) >= limit:
            break
    return articles


def _estimate_history_path() -> Path:
    return Path("data/intraday_estimates/estimate_history.csv")


def _snapshot_path(report_date: str) -> Path:
    return Path("data/intraday_estimates") / f"{report_date}_1430_snapshot.csv"


def _load_estimate_history() -> pd.DataFrame:
    path = _estimate_history_path()
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "date",
                "decision_time",
                "estimate_time",
                "fund_code",
                "previous_nav",
                "estimated_nav",
                "estimated_change_pct",
                "source",
                "estimate_used_for_decision",
                "estimate_reason",
            ]
        )
    return pd.read_csv(path, dtype={"fund_code": str})


def _append_1430_estimates_if_available(
    history: pd.DataFrame,
    report_date: str,
    nav_by_code: dict[str, pd.Series],
    snapshot_file: Path | None = None,
) -> pd.DataFrame:
    snapshot_file = snapshot_file or _snapshot_path(report_date)
    if not snapshot_file.exists():
        return history
    snapshot = pd.read_csv(snapshot_file, dtype={"fund_code": str})
    rows: list[dict[str, Any]] = []
    for row in snapshot.itertuples(index=False):
        change_pct = getattr(row, "eastmoney_fundgz_change_pct", None)
        if pd.isna(change_pct):
            continue
        fund_code = str(row.fund_code).zfill(6)
        nav = nav_by_code[fund_code].sort_index()
        previous = nav.loc[nav.index < pd.Timestamp(report_date)]
        if previous.empty:
            continue
        previous_nav = float(previous.iloc[-1])
        estimate_pct = float(change_pct)
        estimate_time = getattr(row, "eastmoney_fundgz_time", None)
        if pd.isna(estimate_time) or not str(estimate_time).strip():
            estimate_time = f"{report_date} 14:30:00+08:00"
        decision_time = getattr(row, "decision_time", None)
        if pd.isna(decision_time) or not str(decision_time).strip():
            decision_time = f"{report_date} 14:30:00"
        rows.append(
            {
                "date": report_date,
                "decision_time": decision_time,
                "estimate_time": estimate_time,
                "fund_code": fund_code,
                "previous_nav": previous_nav,
                "estimated_nav": previous_nav * (1 + estimate_pct / 100.0),
                "estimated_change_pct": estimate_pct,
                "source": str(
                    getattr(row, "eastmoney_fundgz_source", "eastmoney_fundgz_1430_snapshot")
                )
                or "eastmoney_fundgz_1430_snapshot",
                "estimate_used_for_decision": bool(
                    getattr(row, "estimate_used_for_decision", False)
                ),
                "estimate_reason": str(getattr(row, "estimate_reason", "") or ""),
            }
        )
    if not rows:
        return history
    combined = pd.concat([history, pd.DataFrame(rows)], ignore_index=True)
    return combined.drop_duplicates(
        ["date", "fund_code", "source"], keep="first"
    )


def _append_archived_1430_estimates(
    history: pd.DataFrame,
    nav_by_code: dict[str, pd.Series],
) -> pd.DataFrame:
    snapshots = sorted(Path("data/intraday_estimates").glob("*_1430_snapshot.csv"))
    for snapshot_file in snapshots:
        report_date = snapshot_file.name.replace("_1430_snapshot.csv", "")
        history = _append_1430_estimates_if_available(
            history,
            report_date,
            nav_by_code,
            snapshot_file=snapshot_file,
        )
    return history


def _reconcile_estimates(
    history: pd.DataFrame,
    nav_by_code: dict[str, pd.Series],
) -> tuple[pd.DataFrame, dict[str, float], EstimatePolicy]:
    history = history.copy()
    history["fund_code"] = history["fund_code"].astype(str).str.zfill(6)
    parts = []
    for fund_code, group in history.groupby("fund_code", dropna=False):
        nav = nav_by_code.get(str(fund_code).zfill(6))
        if nav is None:
            parts.append(group)
            continue
        parts.append(reconcile_estimate_history(group, nav))
    reconciled = pd.concat(parts, ignore_index=True) if parts else history
    metrics = calibration_metrics(reconciled)
    policy = adaptive_policy_from_history(reconciled, EstimatePolicy())
    return reconciled, metrics, policy


def _save_estimate_history(history: pd.DataFrame) -> None:
    path = _estimate_history_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    history.to_csv(path, index=False, encoding="utf-8-sig")


def _score_forward_ledgers(nav_by_code: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger_path = Path("output/forward/ledger.csv")
    ledger = pd.read_csv(ledger_path, dtype={"fund_code": str})
    scored = score_forward_ledger(ledger, nav_by_code)
    scored.to_csv(ledger_path, index=False, encoding="utf-8-sig")

    rotation_path = Path("output/forward/rotation_ledger.csv")
    rotation = pd.read_csv(rotation_path)
    nav_by_asset = {
        asset: nav_by_code[definition["code"]]
        for asset, definition in RESEARCH_FUNDS.items()
    }
    scored_rotation = score_rotation_forward_ledger(rotation, nav_by_asset)
    scored_rotation.to_csv(rotation_path, index=False, encoding="utf-8-sig")
    return scored, scored_rotation


def _format_pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "缺失"
    return f"{float(value):.{digits}f}%"


def _format_ratio(value: float | int | None, digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "缺失"
    return f"{float(value):.{digits}f}"


def _format_amt_yi(value: float | None) -> str:
    if value is None or pd.isna(value):
        return "缺失"
    return f"{float(value) / 1e8:.2f}亿元"


def _market_date_from_quotes(quotes: dict[str, CnQuote]) -> str:
    return quotes["sh000001"].quote_date


def _latest_official_nav_date(nav_by_code: dict[str, pd.Series]) -> str:
    return max(series.index.max() for series in nav_by_code.values()).date().isoformat()


def _actual_change_pct_on_date(nav: pd.Series, target_date: str) -> float | None:
    nav = nav.sort_index()
    target = pd.Timestamp(target_date)
    if target not in nav.index:
        return None
    previous = nav.loc[nav.index < target]
    if previous.empty:
        return None
    return (float(nav.loc[target]) / float(previous.iloc[-1]) - 1.0) * 100


def _latest_rotation_weights(rotation_ledger: pd.DataFrame) -> tuple[float, float, float]:
    last = rotation_ledger.sort_values(["signal_date", "generated_at"]).iloc[-1]
    return (
        float(last["candidate_cpo_weight"]),
        float(last["candidate_memory_weight"]),
        float(last["candidate_ai_weight"]),
    )


def _forward_summary(ledger: pd.DataFrame) -> list[str]:
    latest_rows = ledger.loc[ledger["signal_date"] == ledger["signal_date"].max()].copy()
    latest_rows["fund_code"] = latest_rows["fund_code"].astype(str).str.zfill(6)
    latest_rows = latest_rows.sort_values(["fund_code", "generated_at"]).drop_duplicates(
        ["fund_code"], keep="last"
    )
    lines = []
    for row in latest_rows.itertuples(index=False):
        if pd.isna(row.next_nav_return):
            lines.append(f"- `{row.fund_code}` 下一净值表现待正式净值发布后补记分。")
            continue
        lines.append(
            f"- `{row.fund_code}` 下一净值表现已记分：`{_format_pct(float(row.next_nav_return) * 100)}`"
        )
    return lines


def _rotation_summary(rotation_ledger: pd.DataFrame) -> str:
    latest = rotation_ledger.sort_values(["signal_date", "generated_at"]).iloc[-1]
    if pd.isna(latest["candidate_next_return"]) or pd.isna(latest["next_nav_date"]):
        return "最新轮动前视记分待共同下一净值日正式发布后补记，不覆盖旧信号。"
    return (
        "最新轮动前视记分："
        f"候选组合下一净值 `{_format_pct(float(latest['candidate_next_return']) * 100)}`，"
        f"等权部署 `{_format_pct(float(latest['deployed_next_return']) * 100)}`，"
        f"下一共同净值日 `{latest['next_nav_date']}`。"
    )


def _sector_snapshot(quotes: dict[str, CnQuote], basket: list[tuple[str, str]]) -> dict[str, Any]:
    rows = []
    for symbol, name in basket:
        quote = quotes[symbol]
        rows.append({"name": name, "change_pct": quote.change_pct})
    frame = pd.DataFrame(rows)
    return {
        "average_change_pct": float(frame["change_pct"].mean()),
        "positive_count": int((frame["change_pct"] > 0).sum()),
        "sample_size": int(len(frame)),
        "top_mover": frame.sort_values("change_pct", ascending=False).iloc[0]["name"],
    }


def _official_nav_section(nav_by_code: dict[str, pd.Series], quotes: dict[str, CnQuote]) -> list[str]:
    etf_names = {
        "007817": "515880 通信ETF",
        "008887": "159995 芯片ETF",
        "008585": "515070 人工智能ETF",
    }
    lines = []
    for definition in RESEARCH_FUNDS.values():
        code = definition["code"]
        series = nav_by_code[code]
        last_date = series.index.max().date().isoformat()
        last_nav = float(series.iloc[-1])
        prev_nav = float(series.iloc[-2])
        actual_change_pct = (last_nav / prev_nav - 1.0) * 100
        etf_symbol, _, _ = FUND_TO_ETF[code]
        etf_quote = quotes[etf_symbol]
        lines.append(
            f"- `{code} {definition['name']}` 正式净值最新到 `{last_date}`：`{last_nav:.4f}`，"
            f"较上一净值日 `{_format_pct(actual_change_pct)}`；对应 ETF `{etf_names[code]}` "
            f"收盘 `{_format_pct(etf_quote.change_pct)}`。"
        )
    return lines


def _estimate_section(
    report_date: str,
    metrics: dict[str, float],
    policy: EstimatePolicy,
    nav_by_code: dict[str, pd.Series],
    latest_estimates: dict[str, Any],
) -> list[str]:
    snapshot_file = _snapshot_path(report_date)
    if not snapshot_file.exists():
        lines = ["- 当日 `14:30` 归档快照缺失，估值校准未执行。"]
        for code in ("007817", "008887", "008585"):
            estimate = latest_estimates.get(code)
            if estimate is None:
                lines.append(f"- `{code}` 未抓到最新可得估值。")
                continue
            lines.append(
                f"- `{code}` 最新可得估值 `{_format_pct(estimate.estimated_change_pct)}`，"
                f"估值时间 `{estimate.estimate_time}`，仅作收盘后参考。"
            )
        return lines
    snapshot = pd.read_csv(snapshot_file, dtype={"fund_code": str})
    lines = []
    for row in snapshot.itertuples(index=False):
        fund_code = str(row.fund_code).zfill(6)
        nav = nav_by_code[fund_code]
        actual_change = _actual_change_pct_on_date(nav, report_date)
        estimate_change = getattr(row, "eastmoney_fundgz_change_pct", None)
        latest_estimate = latest_estimates.get(fund_code)
        estimate_time_text = ""
        if latest_estimate is not None:
            estimate_time_text = f"，最新可得估值时间 `{latest_estimate.estimate_time}`"
        if actual_change is None:
            if bool(getattr(row, "estimate_used_for_decision", False)) and pd.notna(estimate_change):
                lines.append(
                    f"- `{fund_code}` 14:30 归档估值 `{_format_pct(float(estimate_change))}`{estimate_time_text}，"
                    f"但 `{report_date}` 正式净值尚未发布，暂不计入 MAE/方向准确率。"
                )
            else:
                lines.append(
                    f"- `{fund_code}` 14:30 估值源不可用或未纳入决策；`{report_date}` 正式净值尚未发布{estimate_time_text}。"
                )
            continue
        if bool(getattr(row, "estimate_used_for_decision", False)) and pd.notna(estimate_change):
            error = float(estimate_change) - actual_change
            lines.append(
                f"- `{fund_code}` 14:30 估值 `{_format_pct(float(estimate_change))}`{estimate_time_text}，"
                f"正式净值 `{_format_pct(actual_change)}`，误差 `{_format_pct(abs(error))}`。"
            )
        else:
            lines.append(
                f"- `{fund_code}` 14:30 估值源不可用或未纳入决策；正式净值 `{_format_pct(actual_change)}`{estimate_time_text}，"
                "当时未用于 14:30 决策，但已纳入事后校准的 MAE/方向准确率样本。"
            )
    lines.append(
        "- 当前估值校准汇总："
        f"观测 `{int(metrics['observations'])}`，"
        f"MAE `{_format_ratio(metrics['mae_pct'], 2) if pd.notna(metrics['mae_pct']) else '缺失'}` pct，"
        f"方向准确率 `{_format_pct(metrics['direction_accuracy'] * 100) if pd.notna(metrics['direction_accuracy']) else '缺失'}`，"
        f"动态阈值 `{policy.minimum_abs_change_pct:.2f}% / 强阈值 {policy.strong_abs_change_pct:.2f}%`。"
    )
    lines.append(
        "- 由于可用校准样本不足 `20` 条，估值层仍不得改变次日基础仓位；最多只作为 `10%` 战术加减仓前提。"
    )
    return lines


def _relative_1430_changes(
    report_date: str,
    nav_by_code: dict[str, pd.Series],
    scored_ledger: pd.DataFrame,
    scored_rotation: pd.DataFrame,
) -> list[str]:
    latest_official_date = _latest_official_nav_date(nav_by_code)
    if latest_official_date == report_date:
        parts = []
        for code in ("007817", "008887", "008585"):
            parts.append(f"{code} {_format_pct(_actual_change_pct_on_date(nav_by_code[code], report_date))}")
        nav_line = f"- 正式净值已更新到 `{latest_official_date}`：`{' / '.join(parts)}`。"
    else:
        nav_line = (
            f"- 正式净值最新只到 `{latest_official_date}`；`{report_date}` 当日官方净值尚未发布，"
            "今天 14:30 的基金估值只归档、不校准。"
        )

    scored_dates = []
    ledger_scored = scored_ledger.loc[scored_ledger["next_nav_return"].notna()]
    rotation_scored = scored_rotation.loc[scored_rotation["candidate_next_return"].notna()]
    if not ledger_scored.empty:
        scored_dates.append(str(ledger_scored["signal_date"].max()))
    if not rotation_scored.empty:
        scored_dates.append(str(rotation_scored["signal_date"].max()))
    if scored_dates:
        scored_line = (
            f"- `output/forward/ledger.csv` 与 `rotation_ledger.csv` 已补记到信号日 "
            f"`{max(scored_dates)}` 的下一净值表现，仅新增评分列，未覆盖旧信号。"
        )
    else:
        scored_line = "- 本次没有新增可补记分的官方净值，前视台账旧信号保持原样。"

    if _snapshot_path(report_date).exists():
        estimate_line = "- 若当日估值可靠性不足，次日基础仓位继续完全沿用冻结配置，估值层不参与基线。"
    else:
        estimate_line = "- 当日 `14:30` 归档缺失，次日基础仓位继续只看冻结模型，不让估值层介入。"
    return [nav_line, scored_line, estimate_line]


def _cls_section(articles: list[dict[str, str]]) -> list[str]:
    if not articles:
        return ["- 财联社公开页未抽取到匹配科技主线的可复核条目。"]
    lines = []
    for article in articles[:4]:
        timestamp = f" `{article['timestamp']}`" if article["timestamp"] else ""
        lines.append(
            f"- 财联社{timestamp}：[{article['title']}]({article['url']})。"
        )
    return lines


def _announcement_section(announcements: list[dict[str, Any]], start_date: str, end_date: str) -> list[str]:
    if not announcements:
        return [
            f"- 通过巨潮 `hisAnnouncement/query` 扫描 `{start_date}` 至 `{end_date}` 的代表公司公告，"
            "未检索到 CPO / 存储 / PCB / AI 代表股新增重大披露。"
        ]
    lines = []
    for item in announcements[:6]:
        title = re.sub(r"<[^>]+>", "", str(item.get("announcementTitle", "")))
        path = item.get("adjunctUrl", "")
        url = f"http://static.cninfo.com.cn/{path}" if path else "http://www.cninfo.com.cn/"
        ts = pd.to_datetime(item.get("announcementTime"), unit="ms", errors="coerce")
        lines.append(f"- `{ts}` [{title}]({url})")
    return lines


def _outlook_bias(quotes: dict[str, CnQuote], us_quotes: dict[str, UsQuote]) -> tuple[str, list[str]]:
    strong_cn = (
        quotes["sh515880"].change_pct > 3
        and quotes["sz159995"].change_pct > 3
        and quotes["sh515070"].change_pct > 3
    )
    strong_us = (
        us_quotes["hf_NQ"].change_pct > 0
        and us_quotes["hf_ES"].change_pct > 0
        and us_quotes["gb_mu"].change_pct > 0
    )
    if strong_cn and strong_us:
        return "偏多", [
            "A股科技主线同步放量走强，CPO/存储/PCB/AI 四条线同向。",
            "美股股指期货与半导体、光模块链夜盘继续偏强，次日情绪外溢概率更高。",
            "但短线涨幅过大，执行上仍以“不追涨、不用不可靠估值改基础仓位”为前提。",
        ]
    if strong_cn or strong_us:
        return "中性偏多", [
            "单边数据仍偏强，但中美两侧并未完全共振。",
            "维持基线仓位，等待下一交易日盘中估值与广度确认。",
        ]
    return "中性", [
        "跨市场共振不足，次日更重视 14:30 盘中确认而非收盘情绪延续。",
    ]


def _report_text(
    report_date: str,
    nav_by_code: dict[str, pd.Series],
    cn_quotes: dict[str, CnQuote],
    us_quotes: dict[str, UsQuote],
    estimate_metrics: dict[str, float],
    adaptive_policy: EstimatePolicy,
    scored_ledger: pd.DataFrame,
    scored_rotation: pd.DataFrame,
    latest_estimates: dict[str, Any],
    cls_articles: list[dict[str, str]],
    announcements: list[dict[str, Any]],
) -> str:
    overnight_ts = us_quotes["hf_ES"].timestamp
    cpo_weight, memory_weight, ai_weight = _latest_rotation_weights(scored_rotation)
    total_exposure = 0.90
    plan_weights = {
        "007817": total_exposure * cpo_weight,
        "008887": total_exposure * memory_weight,
        "008585": total_exposure * ai_weight,
        "cash": 1.0 - total_exposure,
    }
    bias, bias_reasons = _outlook_bias(cn_quotes, us_quotes)
    latest_official_date = _latest_official_nav_date(nav_by_code)
    relative_change_lines = _relative_1430_changes(
        report_date, nav_by_code, scored_ledger, scored_rotation
    )
    estimate_lines = _estimate_section(
        report_date, estimate_metrics, adaptive_policy, nav_by_code, latest_estimates
    )

    focus_baskets = {
        "CPO": [
            ("sz300308", "中际旭创"),
            ("sz300502", "新易盛"),
            ("sz300394", "天孚通信"),
            ("sz002281", "光迅科技"),
        ],
        "存储": [
            ("sh603986", "兆易创新"),
            ("sz301308", "江波龙"),
            ("sh688525", "佰维存储"),
            ("sz300223", "北京君正"),
        ],
        "PCB": [
            ("sz300476", "胜宏科技"),
            ("sz002463", "沪电股份"),
            ("sz002916", "深南电路"),
            ("sh600183", "生益科技"),
        ],
        "AI": [
            ("sz300308", "中际旭创"),
            ("sh688041", "海光信息"),
            ("sh603019", "中科曙光"),
            ("sz002230", "科大讯飞"),
        ],
    }
    other_baskets = {
        "消费电子": [
            ("sz300408", "三环集团"),
            ("sh603160", "汇顶科技"),
        ],
        "高端PCB延伸": [
            ("sh603920", "世运电路"),
            ("sz300476", "胜宏科技"),
        ],
    }
    sector_lines = []
    for sector_name, basket in focus_baskets.items():
        snapshot = _sector_snapshot(cn_quotes, basket)
        sector_lines.append(
            f"- `{sector_name}`：样本均涨 `{_format_pct(snapshot['average_change_pct'])}`，"
            f"上涨家数 `{snapshot['positive_count']}/{snapshot['sample_size']}`，"
            f"强势代表 `{snapshot['top_mover']}`。"
        )
    other_lines = []
    for sector_name, basket in other_baskets.items():
        snapshot = _sector_snapshot(cn_quotes, basket)
        other_lines.append(
            f"- `{sector_name}`：样本均涨 `{_format_pct(snapshot['average_change_pct'])}`，"
            f"上涨家数 `{snapshot['positive_count']}/{snapshot['sample_size']}`。"
        )

    short_window = pd.read_csv("output/research_v8/short_window_correlations.csv")
    short_window = short_window.loc[
        short_window["us_lead_trading_closes"] == 1,
        ["pair", "correlation", "direction_accuracy", "confidence"],
    ]
    short_window_lines = [
        f"- `{row.pair}`：相关系数 `{row.correlation:.2f}`，方向命中 `{row.direction_accuracy:.1%}`，置信度 `{row.confidence}`。"
        for row in short_window.itertuples(index=False)
    ]

    research_locked = pd.read_csv("output/research_v8/locked_six_month_test.csv")
    rotation_locked = pd.read_csv("output/research_v8/rotation_locked_comparison.csv")
    rotation_walk = pd.read_csv("output/research_v8/rotation_walk_forward.csv")
    overfit = json.loads(Path("output/research_v8/overfitting_diagnostics.json").read_text(encoding="utf-8"))
    single_pbo = next(item["pbo"] for item in overfit if item["model_group"] == "single_fund_zoo")
    rotation_pbo = next(item["pbo"] for item in overfit if item["model_group"] == "sector_rotation")

    research_lines = []
    for code, asset in (("007817", "cpo_communication"), ("008887", "memory_semiconductor_proxy"), ("008585", "artificial_intelligence")):
        row = research_locked.loc[
            (research_locked["asset"] == asset) & (research_locked["model"] == "buy_hold")
        ].iloc[0]
        research_lines.append(
            f"- `{code}` 近半年含费买入持有 `{row['total_return']:.2%}`，"
            f"年化 `{row['cagr']:.2%}`，最大回撤 `{row['max_drawdown']:.2%}`。"
        )

    rotation_equal = rotation_locked.iloc[0]
    rotation_candidate = rotation_locked.iloc[1]
    walk_forward_beat_ratio = float((rotation_walk["excess_return"] > 0).mean())
    walk_forward_median = float(rotation_walk["excess_return"].median())
    walk_forward_worst = float(rotation_walk["excess_return"].min())

    lines = [
        f"# {report_date} 18:30 收盘复盘与下一交易日预判",
        "",
        f"生成时间：{pd.Timestamp.now(tz='Asia/Shanghai')}",
        f"A股正式收盘口径：`{cn_quotes['sh000001'].quote_date}`",
        f"基金正式净值最新到：`{latest_official_date}`",
        f"美股期指/盘中口径：`{overnight_ts}`",
        "",
        "## 结论",
        "",
        f"- 次日总体判断：`{bias}`",
        f"- 候选基金：`007817 / 008887 / 008585`",
        f"- 计划基础仓位：`007817 {plan_weights['007817']:.0%} / 008887 {plan_weights['008887']:.0%} / 008585 {plan_weights['008585']:.0%} / 现金 {plan_weights['cash']:.0%}`",
        "- 执行原则：这份报告只服务下一交易日局势判断，不替代当日 `14:30` 的盘中操作。",
        "",
        "## 相对 14:30 报告的变化",
        "",
        *relative_change_lines,
        "",
        "## A股收盘",
        "",
        f"- 上证指数 `{_format_pct(cn_quotes['sh000001'].change_pct)}`，成交额 `{_format_amt_yi(cn_quotes['sh000001'].amount_cny)}`。",
        f"- 深证成指 `{_format_pct(cn_quotes['sz399001'].change_pct)}`，成交额 `{_format_amt_yi(cn_quotes['sz399001'].amount_cny)}`。",
        f"- 创业板指 `{_format_pct(cn_quotes['sz399006'].change_pct)}`，成交额 `{_format_amt_yi(cn_quotes['sz399006'].amount_cny)}`。",
        "",
        "## 重点科技板块",
        "",
        *sector_lines,
        "",
        "## 其它科技板块扫描",
        "",
        *other_lines,
        "",
        "## 基金正式净值与ETF映射",
        "",
        *_official_nav_section(nav_by_code, cn_quotes),
        "",
        "## 14:30 估值校准",
        "",
        *estimate_lines,
        "",
        "## 前视台账更新",
        "",
        *_forward_summary(scored_ledger),
        f"- {_rotation_summary(scored_rotation)}",
        "",
        "## 财联社与官方公告",
        "",
        "### 财联社",
        "",
        *_cls_section(cls_articles),
        "",
        "### 官方公告",
        "",
        *_announcement_section(announcements, report_date, overnight_ts[:10]),
        "",
        "## 美股股指期货与相关科技链夜盘",
        "",
        f"- 标普500期指 `{_format_pct(us_quotes['hf_ES'].change_pct)}`，纳指期指 `{_format_pct(us_quotes['hf_NQ'].change_pct)}`，道指期指 `{_format_pct(us_quotes['hf_YM'].change_pct)}`。",
        f"- CPO 链：`COHR {_format_pct(us_quotes['gb_cohr'].change_pct)}` / `LITE {_format_pct(us_quotes['gb_lite'].change_pct)}` / `AAOI {_format_pct(us_quotes['gb_aaoi'].change_pct)}`。",
        f"- 存储链：`MU {_format_pct(us_quotes['gb_mu'].change_pct)}` / `WDC {_format_pct(us_quotes['gb_wdc'].change_pct)}` / `STX {_format_pct(us_quotes['gb_stx'].change_pct)}`。",
        f"- AI 硬件链：`NVDA {_format_pct(us_quotes['gb_nvda'].change_pct)}` / `SMCI {_format_pct(us_quotes['gb_smci'].change_pct)}` / `DELL {_format_pct(us_quotes['gb_dell'].change_pct)}`。",
        "",
        "## 中美短窗相关仅作方向确认",
        "",
        *short_window_lines,
        "",
        "## 次日计划、触发与减仓条件",
        "",
        *[f"- {reason}" for reason in bias_reasons],
        "- `14:30` 若估值源仍不可用，继续只执行基础仓位，不做战术追涨或临时减仓。",
        f"- 只有当单只基金估值绝对变动 `>= {adaptive_policy.minimum_abs_change_pct:.2f}%` 且累计校准样本 `>=20`、MAE/方向准确率达标时，才允许 `10%` 战术加减仓。",
        f"- 在当前样本不足阶段，只有估值绝对变动 `>= {adaptive_policy.strong_abs_change_pct:.2f}%` 且 ETF 与广度双确认，才可讨论 `10%` 战术微调。",
        "- 战术方向仍是逆向：可靠大跌只加 `10%`，可靠大涨只减 `10%`；绝不改变 `54% / 18% / 18% / 10%` 的基础框架。",
        "- 若后续正式净值连续破坏中期趋势，再由次日 `14:30` 报告单独给出减仓动作，不在本报告里提前替代。",
        "",
        "## 近半年回测、历史模拟实盘与过拟合风险",
        "",
        *research_lines,
        f"- 组合轮动近半年含费：等权 `{rotation_equal['total_return']:.2%}`，冻结轮动 `{rotation_candidate['total_return']:.2%}`；"
        "但这段优势主要来自静态超配 CPO。",
        f"- 历史模拟实盘（walk-forward）共 `{len(rotation_walk)}` 段，跑赢比例 `{walk_forward_beat_ratio:.0%}`，"
        f"中位超额 `{walk_forward_median:.2%}`，最差一段 `{walk_forward_worst:.2%}`，未达到可部署门槛。",
        f"- 过拟合风险：单基金模型库 `PBO={single_pbo:.2%}`，组合轮动 `PBO={rotation_pbo:.2%}`，都不能视为低风险可实盘替代。",
        "- 研究结论仍是：模型只能做辅助参考，不能宣称稳定战胜买入持有；任何追涨杀跌都比研究本身更容易失真。",
        "",
        "## 来源",
        "",
        "- 本地冻结研究：`output/research_v8/research_report.md`、`locked_six_month_test.csv`、`rotation_locked_comparison.csv`、`rotation_walk_forward.csv`",
        "- 本地前视台账：`output/forward/ledger.csv`、`output/forward/rotation_ledger.csv`",
        "- 本地 14:30 归档：`data/intraday_estimates/"
        f"{report_date}_1430_snapshot.csv`",
        "- 基金正式净值：`https://fund.eastmoney.com/pingzhongdata/007817.js`、"
        "`https://fund.eastmoney.com/pingzhongdata/008887.js`、"
        "`https://fund.eastmoney.com/pingzhongdata/008585.js`",
        "- 基金估值：`https://fundgz.1234567.com.cn/js/007817.js`、"
        "`https://fundgz.1234567.com.cn/js/008887.js`、"
        "`https://fundgz.1234567.com.cn/js/008585.js`",
        "- 行情：`https://hq.sinajs.cn/list=sh000001,sz399001,sz399006,sh515880,sz159995,sh515070,...`",
        "- 财联社：[https://www.cls.cn/](https://www.cls.cn/)",
        "- 巨潮公告检索接口：`http://www.cninfo.com.cn/new/hisAnnouncement/query`",
    ]
    return "\n".join(lines) + "\n"


def run_close_outlook() -> Path:
    cn_quotes = _fetch_cn_quotes()
    report_date = _market_date_from_quotes(cn_quotes)
    nav_by_code = {
        definition["code"]: _load_or_fetch_official_nav(definition["code"])
        for definition in RESEARCH_FUNDS.values()
    }
    latest_estimates: dict[str, Any] = {}
    for definition in RESEARCH_FUNDS.values():
        code = definition["code"]
        try:
            latest_estimates[code] = fetch_eastmoney_intraday_estimate(code)
        except Exception:
            latest_estimates[code] = None
    history = _load_estimate_history()
    history = _append_archived_1430_estimates(history, nav_by_code)
    reconciled, metrics, policy = _reconcile_estimates(history, nav_by_code)
    _save_estimate_history(reconciled)
    scored_ledger, scored_rotation = _score_forward_ledgers(nav_by_code)
    cls_articles = _fetch_cls_articles()
    us_quotes = _fetch_us_quotes()
    announcements = _fetch_cninfo_announcements(report_date, us_quotes["hf_ES"].timestamp[:10])
    report = _report_text(
        report_date,
        nav_by_code,
        cn_quotes,
        us_quotes,
        metrics,
        policy,
        scored_ledger,
        scored_rotation,
        latest_estimates,
        cls_articles,
        announcements,
    )
    output_dir = Path("daily_reports") / report_date
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "1830_close_outlook.md"
    path.write_text(report, encoding="utf-8")
    return path


def main() -> None:
    path = run_close_outlook()
    print(path.resolve())


if __name__ == "__main__":
    main()
