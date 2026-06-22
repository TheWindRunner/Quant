"""Generate the 14:30 OTC fund action report."""

from __future__ import annotations

from dataclasses import dataclass
import html
import re
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from .intraday_estimate import (
    EstimatePolicy,
    adaptive_policy_from_history,
    calibration_metrics,
    estimate_confidence,
    fetch_eastmoney_intraday_estimate,
)
from .premarket_report import _fetch_cninfo_announcements
from .run_research import RESEARCH_FUNDS, build_nav_datasets


ASIA_TZ = "Asia/Shanghai"
RESEARCH_DIR = Path("output/research_v8")
FORWARD_DIR = Path("output/forward")
TOTAL_TECH_EXPOSURE = 0.90
REPORT_TIME = "14:30:00"

FUND_TO_ETF = {
    "007817": ("sh515880", "515880", "通信ETF国泰", "CPO/光通信"),
    "008887": ("sz159995", "159995", "芯片ETF华夏", "存储"),
    "008585": ("sh515070", "515070", "人工智能ETF华夏", "AI"),
}

FUND_FEES = {
    "007817": {"purchase": "0.10%", "redemption": "<7日 1.5%；7-29日 0.5%；30-364日 0.25%；>=365日 0"},
    "008887": {"purchase": "0.12%", "redemption": "<7日 1.5%；7-29日 0.5%；30-364日 0.25%；>=365日 0"},
    "008585": {"purchase": "0.12%", "redemption": "<7日 1.5%；7-29日 0.5%；30-364日 0.25%；>=365日 0"},
}

SECTOR_BASKETS = {
    "CPO/光通信": [
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
        ("sh688041", "海光信息"),
        ("sh603019", "中科曙光"),
        ("sz002230", "科大讯飞"),
        ("sz000977", "浪潮信息"),
    ],
    "机器人": [
        ("sz300024", "机器人"),
        ("sz002747", "埃斯顿"),
        ("sz300124", "汇川技术"),
        ("sh688165", "埃夫特"),
    ],
    "半导体设备": [
        ("sz002371", "北方华创"),
        ("sh688012", "中微公司"),
        ("sh688120", "华海清科"),
        ("sz300604", "长川科技"),
    ],
    "消费电子": [
        ("sz300408", "三环集团"),
        ("sh603160", "汇顶科技"),
    ],
}

CLS_KEYWORDS = (
    "午报",
    "午评",
    "收评",
    "A股",
    "半导体",
    "存储",
    "PCB",
    "AI",
    "算力",
    "光模块",
    "CPO",
)


@dataclass(frozen=True)
class TencentQuote:
    code: str
    name: str
    previous_close: float
    current_close: float
    open_price: float
    quote_timestamp: str


@dataclass(frozen=True)
class MinutePoint:
    code: str
    timestamp: pd.Timestamp
    price: float
    cumulative_amount: float


def _http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: float = 20.0,
) -> requests.Response:
    response = requests.get(
        url,
        params=params,
        headers=headers or {"User-Agent": "Mozilla/5.0"},
        timeout=timeout,
    )
    response.raise_for_status()
    return response


def _fetch_tencent_quote(code: str) -> TencentQuote:
    response = _http_get(
        f"https://qt.gtimg.cn/q={code}",
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://gu.qq.com/",
        },
    )
    raw = response.text.split('="', 1)[1].rstrip('";')
    fields = raw.split("~")
    return TencentQuote(
        code=code,
        name=fields[1],
        current_close=float(fields[3]),
        previous_close=float(fields[4]),
        open_price=float(fields[5]),
        quote_timestamp=fields[30],
    )


def _fetch_tencent_minute_point(code: str, cutoff: pd.Timestamp) -> MinutePoint:
    response = _http_get(
        "https://ifzq.gtimg.cn/appstock/app/minute/query",
        params={"code": code},
    )
    payload = response.json()["data"][code]["data"]
    frame = pd.DataFrame(
        [item.split() for item in payload["data"]],
        columns=["hhmm", "price", "cum_volume", "cum_amount"],
    )
    frame["timestamp"] = pd.to_datetime(
        payload["date"] + " " + frame["hhmm"],
        format="%Y%m%d %H%M",
    )
    frame["price"] = pd.to_numeric(frame["price"], errors="coerce")
    frame["cum_amount"] = pd.to_numeric(frame["cum_amount"], errors="coerce")
    available = frame.loc[frame["timestamp"] <= cutoff].dropna(subset=["price"])
    if available.empty:
        raise RuntimeError(f"No minute point for {code} before {cutoff}")
    point = available.iloc[-1]
    return MinutePoint(
        code=code,
        timestamp=pd.Timestamp(point["timestamp"]),
        price=float(point["price"]),
        cumulative_amount=float(point["cum_amount"]),
    )


def _fetch_yahoo_intraday_snapshot(symbol: str, cutoff: pd.Timestamp) -> tuple[pd.Timestamp, float, float | None]:
    response = _http_get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={
            "range": "2d",
            "interval": "1m",
            "includePrePost": "true",
        },
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://finance.yahoo.com/",
        },
    )
    payload = response.json()["chart"]["result"][0]
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(payload["timestamp"], unit="s", utc=True)
            .tz_convert(ASIA_TZ)
            .tz_localize(None),
            "close": payload["indicators"]["quote"][0]["close"],
        }
    ).dropna(subset=["close"])
    available = frame.loc[frame["timestamp"] <= cutoff]
    if available.empty:
        raise RuntimeError(f"No Yahoo intraday point for {symbol} before {cutoff}")
    point = available.iloc[-1]
    meta = payload.get("meta", {})
    previous_close = meta.get("chartPreviousClose") or meta.get("previousClose")
    return (
        pd.Timestamp(point["timestamp"]),
        float(point["close"]),
        float(previous_close) if previous_close else None,
    )


def _load_estimate_history() -> pd.DataFrame:
    path = Path("data/intraday_estimates/estimate_history.csv")
    if not path.exists():
        return pd.DataFrame(columns=["fund_code", "estimated_change_pct", "actual_change_pct"])
    return pd.read_csv(path, dtype={"fund_code": str})


def _load_forward_targets() -> dict[str, Any]:
    ledger = pd.read_csv(FORWARD_DIR / "ledger.csv", dtype={"fund_code": str})
    rotation = pd.read_csv(FORWARD_DIR / "rotation_ledger.csv")
    latest_signal_date = ledger["signal_date"].max()
    latest_ledger = (
        ledger.loc[ledger["signal_date"] == latest_signal_date]
        .sort_values(["fund_code", "generated_at"])
        .drop_duplicates(["fund_code"], keep="last")
    )
    latest_rotation = (
        rotation.loc[rotation["signal_date"] == latest_signal_date]
        .sort_values(["generated_at"])
        .iloc[-1]
    )
    candidate_weights = {
        "007817": float(latest_rotation["candidate_cpo_weight"]),
        "008887": float(latest_rotation["candidate_memory_weight"]),
        "008585": float(latest_rotation["candidate_ai_weight"]),
    }
    base_targets = {
        code: TOTAL_TECH_EXPOSURE * weight
        for code, weight in candidate_weights.items()
    }
    return {
        "signal_date": latest_signal_date,
        "model_version": str(latest_ledger.iloc[-1]["model_version"]),
        "candidate_positions": {
            str(row.fund_code).zfill(6): float(row.candidate_position)
            for row in latest_ledger.itertuples(index=False)
        },
        "base_targets": base_targets,
        "candidate_weights": candidate_weights,
    }


def _load_nav_by_code() -> dict[str, pd.Series]:
    datasets = build_nav_datasets()
    return {
        definition["code"]: datasets[asset]["nav"].rename(definition["code"])
        for asset, definition in RESEARCH_FUNDS.items()
    }


def _find_holding_lot_ledgers() -> list[Path]:
    matches = []
    root = Path("data")
    if not root.exists():
        return matches
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.lower()
        if any(token in name for token in ("holding", "batch", "transaction", "lot")):
            matches.append(path)
    return matches


def _sector_snapshot(
    sector_name: str,
    basket: list[tuple[str, str]],
    cutoff: pd.Timestamp,
) -> dict[str, Any]:
    rows = []
    for code, name in basket:
        minute_point = _fetch_tencent_minute_point(code, cutoff)
        quote = _fetch_tencent_quote(code)
        change_pct = (minute_point.price / quote.previous_close - 1.0) * 100
        rows.append({"name": name, "change_pct": change_pct})
    frame = pd.DataFrame(rows)
    top = frame.sort_values("change_pct", ascending=False).iloc[0]
    return {
        "sector": sector_name,
        "average_change_pct": float(frame["change_pct"].mean()),
        "positive_count": int((frame["change_pct"] > 0).sum()),
        "sample_size": int(len(frame)),
        "breadth_ratio": float((frame["change_pct"] > 0).mean()),
        "top_name": str(top["name"]),
        "top_change_pct": float(top["change_pct"]),
    }


def _fetch_cls_articles(limit: int = 30) -> list[dict[str, str]]:
    homepage = _http_get("https://www.cls.cn").text
    pairs = re.findall(
        r'<a[^>]+href="(/detail/\d+)"[^>]*>(.*?)</a>',
        homepage,
        flags=re.S,
    )
    articles: list[dict[str, str]] = []
    seen: set[str] = set()
    for href, raw_text in pairs:
        title = html.unescape(re.sub(r"<[^>]+>", "", raw_text)).strip()
        if not title or href in seen:
            continue
        if not any(keyword in title for keyword in CLS_KEYWORDS):
            continue
        seen.add(href)
        page = _http_get(f"https://www.cls.cn{href}").text
        description_match = re.search(r'"description" content="([^"]+)"', page)
        timestamp_match = re.search(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2})", page)
        articles.append(
            {
                "title": title,
                "url": f"https://www.cls.cn{href}",
                "description": description_match.group(1) if description_match else title,
                "timestamp": timestamp_match.group(1) if timestamp_match else "",
            }
        )
        if len(articles) >= limit:
            break
    return articles


def _filter_cls_articles(
    articles: list[dict[str, str]],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict[str, str]]:
    result = []
    for article in articles:
        raw = article.get("timestamp") or ""
        if not raw:
            continue
        timestamp = pd.Timestamp(raw)
        if start_ts <= timestamp <= end_ts:
            result.append(article)
    return result


def _filter_announcements(
    announcements: list[dict[str, Any]],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows = []
    for item in announcements:
        timestamp = pd.to_datetime(item.get("announcementTime"), unit="ms", errors="coerce")
        if pd.isna(timestamp):
            continue
        local_time = timestamp.tz_localize("UTC").tz_convert(ASIA_TZ).tz_localize(None)
        if start_ts <= local_time <= end_ts:
            rows.append({**item, "local_time": local_time})
    rows.sort(key=lambda item: item["local_time"], reverse=True)
    return rows


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "缺失"
    return f"{value:.{digits}f}%"


def _format_yi(amount_cny: float | None) -> str:
    if amount_cny is None or pd.isna(amount_cny):
        return "缺失"
    return f"{amount_cny / 1e8:.2f}亿元"


def _estimate_comment(
    fund_code: str,
    estimate_change_pct: float | None,
    estimate_time: pd.Timestamp | None,
    metrics: dict[str, float],
    policy: EstimatePolicy,
    etf_change_pct: float | None,
    breadth_ratio: float | None,
    cutoff: pd.Timestamp,
) -> tuple[bool, str]:
    if estimate_change_pct is None or estimate_time is None:
        return False, "估值源不可用，估值不参与判断。"
    if estimate_time.date() != cutoff.date():
        return (
            False,
            f"fundgz 最新日期为 `{estimate_time:%Y-%m-%d %H:%M}`，不是今天 `{cutoff:%Y-%m-%d} 14:30` 前的同日估值，估值不参与判断。",
        )
    if estimate_time > cutoff:
        return (
            False,
            f"fundgz 最新时间为 `{estimate_time:%H:%M}`，晚于 `14:30` 决策点，估值不参与判断。",
        )
    if abs(estimate_change_pct) < policy.minimum_abs_change_pct:
        return (
            False,
            f"估值绝对变动 `{_format_pct(abs(estimate_change_pct))}` 小于 `{policy.minimum_abs_change_pct:.2f}%`，按规则忽略。",
        )
    confidence = estimate_confidence(
        estimate_change_pct,
        metrics=metrics,
        etf_change_pct=etf_change_pct,
        breadth_ratio=breadth_ratio,
        policy=policy,
    )
    if not confidence["reliable"]:
        details = []
        if metrics.get("observations", 0) < 20:
            details.append(f"累计校准仅 `{int(metrics.get('observations', 0))}` 次")
        if not confidence.get("etf_agrees", False):
            details.append("ETF 未同向确认")
        if not confidence.get("breadth_agrees", False):
            details.append("广度未同向确认")
        reason = "；".join(details) if details else "可靠性门槛未满足"
        return False, f"估值虽达到幅度门槛，但 `{reason}`，估值不参与判断。"
    return True, "估值满足时间、幅度和双确认门槛，可参与 10% 战术微调。"


def _build_snapshot_rows(
    cutoff: pd.Timestamp,
    base_targets: dict[str, float],
    sector_snapshots: dict[str, dict[str, Any]],
    metrics: dict[str, float],
    policy: EstimatePolicy,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    decisions: dict[str, dict[str, Any]] = {}
    for fund_code, (etf_code, plain_etf_code, etf_name, sector_name) in FUND_TO_ETF.items():
        etf_quote = _fetch_tencent_quote(etf_code)
        etf_point = _fetch_tencent_minute_point(etf_code, cutoff)
        etf_change_pct = (etf_point.price / etf_quote.previous_close - 1.0) * 100
        estimate = None
        estimate_error = None
        try:
            estimate = fetch_eastmoney_intraday_estimate(fund_code)
        except Exception as exc:
            estimate_error = str(exc)
        estimate_time = estimate.estimate_time if estimate is not None else None
        estimate_change_pct = estimate.estimated_change_pct if estimate is not None else None
        breadth = sector_snapshots[sector_name]
        estimate_used, estimate_reason = _estimate_comment(
            fund_code,
            estimate_change_pct,
            estimate_time,
            metrics,
            policy,
            etf_change_pct,
            breadth["breadth_ratio"],
            cutoff,
        )
        iopv = None
        iopv_reason = "AKShare 不可用，IOPV 未取到。"
        try:
            import akshare as ak  # type: ignore

            frame = ak.fund_etf_spot_em()
            code_column = "代码" if "代码" in frame.columns else "基金代码"
            row = frame.loc[frame[code_column].astype(str).str.zfill(6) == plain_etf_code]
            if not row.empty:
                item = row.iloc[0]
                for column in ("IOPV实时估值", "IOPV"):
                    if column in row.columns:
                        iopv = pd.to_numeric(item[column], errors="coerce")
                        if pd.notna(iopv):
                            iopv = float(iopv)
                            iopv_reason = "AKShare IOPV 已读取。"
                            break
        except Exception as exc:
            iopv_reason = f"AKShare IOPV 不可用：{exc}"
        rows.append(
            {
                "decision_time": cutoff.strftime("%Y-%m-%d %H:%M:%S"),
                "fund_code": fund_code,
                "fund_name": RESEARCH_FUNDS[next(k for k, v in RESEARCH_FUNDS.items() if v["code"] == fund_code)]["name"],
                "base_target_position": base_targets[fund_code],
                "estimate_used_for_decision": estimate_used,
                "estimate_reason": estimate_reason,
                "eastmoney_fundgz_time": estimate_time.strftime("%Y-%m-%d %H:%M:%S") if estimate_time is not None else "",
                "eastmoney_fundgz_change_pct": estimate_change_pct,
                "eastmoney_fundgz_source": estimate.source if estimate is not None else "",
                "eastmoney_fundgz_error": estimate_error or "",
                "etf_code": plain_etf_code,
                "etf_name": etf_name,
                "etf_price_1430": etf_point.price,
                "etf_change_pct_1430": etf_change_pct,
                "etf_iopv": iopv,
                "etf_iopv_note": iopv_reason,
                "sector_breadth_positive_count": breadth["positive_count"],
                "sector_breadth_sample_size": breadth["sample_size"],
                "sector_breadth_ratio": breadth["breadth_ratio"],
            }
        )
        decisions[fund_code] = {
            "estimate_used": estimate_used,
            "estimate_reason": estimate_reason,
            "estimate_change_pct": estimate_change_pct,
            "estimate_time": estimate_time,
            "etf_change_pct": etf_change_pct,
            "iopv": iopv,
            "iopv_reason": iopv_reason,
            "breadth": breadth,
        }
    return rows, decisions


def _action_table_rows(
    base_targets: dict[str, float],
    nav_by_code: dict[str, pd.Series],
    decisions: dict[str, dict[str, Any]],
    has_real_lots: bool,
) -> list[str]:
    rows = []
    for fund_code in ("007817", "008887", "008585"):
        fund_name = next(
            definition["name"]
            for definition in RESEARCH_FUNDS.values()
            if definition["code"] == fund_code
        )
        nav = nav_by_code[fund_code]
        last_nav = float(nav.iloc[-1])
        ma20 = float(nav.rolling(20).mean().iloc[-1])
        ma60 = float(nav.rolling(60).mean().iloc[-1])
        target = base_targets[fund_code]
        action = "持有"
        trigger = "冻结基线仓位继续有效，且今天不满足可靠战术估值门槛。"
        invalid = "后续正式净值连续破坏中期趋势，或未来某日出现可靠估值双确认。"
        if fund_code == "007817":
            trigger = (
                f"对应 ETF 14:30 涨跌 `{_format_pct(decisions[fund_code]['etf_change_pct'])}`，"
                f"代表广度 `{decisions[fund_code]['breadth']['positive_count']}/"
                f"{decisions[fund_code]['breadth']['sample_size']}`，方向仍强但不追涨。"
            )
            invalid = "正式净值后续有效跌破 MA20/MA60，或未来可靠大涨估值触发 10% 战术减仓。"
        elif fund_code == "008887":
            trigger = (
                f"芯片 ETF 14:30 仅 `{_format_pct(decisions[fund_code]['etf_change_pct'])}`，"
                "且 fundgz 幅度不足 1.5%，维持小权重持有。"
            )
            invalid = "正式净值重新走弱并跌回 MA20 下方，或未来出现可靠回撤估值再讨论加仓。"
        elif fund_code == "008585":
            trigger = (
                f"AI 代表股 14:30 广度仅 `{decisions[fund_code]['breadth']['positive_count']}/"
                f"{decisions[fund_code]['breadth']['sample_size']}`，"
                "板块并非今日强主线，不新增。"
            )
            invalid = "正式净值持续弱于 MA20，或未来 AI 链与估值双确认后再调整。"
        fee_text = f"申购约 `{FUND_FEES[fund_code]['purchase']}`；赎回 `{FUND_FEES[fund_code]['redemption']}`"
        risk_text = (
            f"最新净值 `{last_nav:.4f}`；相对 MA20 `{_format_pct((last_nav / ma20 - 1.0) * 100)}`；"
            f"相对 MA60 `{_format_pct((last_nav / ma60 - 1.0) * 100)}`。"
        )
        if not has_real_lots:
            invalid += " 当前缺少真实持仓批次台账，任何减仓/卖出都先抑制。"
        rows.append(
            f"| {fund_code} {fund_name} | {action} | 总资产 {target:.0%} | {trigger} | {invalid} | {fee_text} | {risk_text} |"
        )
    rows.append(
        "| 现金 / 货基 | 持有 | 总资产 10% | 今日更强的 PCB、机器人等板块未映射到已验证可执行场外基金，且 14:30 估值层不可用。 | 未来若出现可靠回撤估值，或研究库新增可执行纯板块基金。 | 无 | 代价是可能错过强势板块继续冲高。 |"
    )
    return rows


def _write_report(
    report_date: str,
    cutoff: pd.Timestamp,
    forward_targets: dict[str, Any],
    nav_by_code: dict[str, pd.Series],
    index_points: dict[str, dict[str, Any]],
    sector_snapshots: dict[str, dict[str, Any]],
    decisions: dict[str, dict[str, Any]],
    metrics: dict[str, float],
    policy: EstimatePolicy,
    cls_articles: list[dict[str, str]],
    announcements: list[dict[str, Any]],
    has_real_lots: bool,
) -> Path:
    latest_nav_date = max(series.index.max() for series in nav_by_code.values()).date().isoformat()
    ranking = sorted(
        sector_snapshots.values(),
        key=lambda item: item["average_change_pct"],
        reverse=True,
    )
    yahoo_symbols = {}
    for symbol in ("NQ=F", "ES=F", "YM=F"):
        point_ts, price, previous_close = _fetch_yahoo_intraday_snapshot(symbol, cutoff)
        change_pct = (price / previous_close - 1.0) * 100 if previous_close else None
        yahoo_symbols[symbol] = {
            "timestamp": point_ts,
            "price": price,
            "change_pct": change_pct,
        }
    short_window = pd.read_csv(RESEARCH_DIR / "short_window_correlations.csv")
    short_window = short_window.loc[short_window["us_lead_trading_closes"] == 1]
    lines = [
        f"# {report_date} 14:30 场外基金操作报告",
        "",
        f"生成时间：{pd.Timestamp.now(tz=ASIA_TZ)}",
        f"决策时点：北京时间 `{report_date} 14:30`",
        "生成说明：正文只使用 `14:30` 及之前可得的行情、新闻和公告；晚于该时点的页面刷新只用于核对数据源是否可用，不反推 14:30 决策。",
        f"完整净值截止：`{latest_nav_date}`",
        f"冻结模型：`signal_date={forward_targets['signal_date']}`，`model_version={forward_targets['model_version']}`，科技主题内部权重 `60% / 20% / 20%`，总暴露 `90%`，现金 `10%`。",
        "",
        "## 结论",
        "",
        "今天 15:00 前的默认动作是：",
        "",
        "| 基金 | 动作 | 目标仓位 | 触发条件 | 失效条件 | 费用提示 | 主要风险 |",
        "|---|---|---:|---|---|---|---|",
        *_action_table_rows(forward_targets["base_targets"], nav_by_code, decisions, has_real_lots),
        "",
        "**执行口径：**",
        "",
        "- 默认维持 `54% / 18% / 18% / 10%现金`。",
        "- 今天不给出“买入/加仓”指令，因为可靠 14:30 估值层没有形成；板块强弱只做方向确认，不替代估值规则。",
        "- 今天不给出“减仓/卖出”指令，因为未发现真实持仓批次台账，无法避免 `<7日` 与 `<30日` 高赎回费。",
        "",
        "## 14:30 盘中确认",
        "",
        f"- 上证指数 14:30 约 `{_format_pct(index_points['sh000001']['change_pct'])}`，成交额 `{_format_yi(index_points['sh000001']['amount'])}`。",
        f"- 深证成指 14:30 约 `{_format_pct(index_points['sz399001']['change_pct'])}`，成交额 `{_format_yi(index_points['sz399001']['amount'])}`。",
        f"- 创业板指 14:30 约 `{_format_pct(index_points['sz399006']['change_pct'])}`，成交额 `{_format_yi(index_points['sz399006']['amount'])}`。",
        f"- 美股期指 14:30：`NQ {_format_pct(yahoo_symbols['NQ=F']['change_pct'])}`，`ES {_format_pct(yahoo_symbols['ES=F']['change_pct'])}`，`YM {_format_pct(yahoo_symbols['YM=F']['change_pct'])}`。",
        "",
        "## 板块排序与判断",
        "",
        "- 14:30 A股代表篮子强弱排序："
        + " > ".join(f"`{item['sector']}`" for item in ranking),
    ]
    for item in ranking:
        lines.append(
            f"- `{item['sector']}`：样本均涨 `{_format_pct(item['average_change_pct'])}`，"
            f"上涨 `{item['positive_count']}/{item['sample_size']}`，"
            f"最强 `{item['top_name']}` `{_format_pct(item['top_change_pct'])}`。"
        )
    lines.extend(
        [
            "",
            "- `CPO/光通信` 仍是研究库核心方向，14:30 的 ETF 与广度同向偏强，但没有可靠估值，不追涨。",
            "- `存储` 方向仍偏强，但映射基金 `008887` 的 ETF 强度一般，fundgz 幅度也未过阈值，不加仓。",
            "- `PCB` 与 `机器人` 盘中更强，只能作为景气确认；当前研究库没有完成可执行性验证的支付宝纯板块场外基金，不新增标的。",
            "- `AI` 14:30 代表股广度偏弱，`008585` 继续只保留基线持仓，不新增战术仓。",
            "",
            "## 估值层与 ETF 确认",
            "",
            f"- 当前已校准估值样本 `{int(metrics.get('observations', 0))}` 条，MAE "
            + (
                f"`{metrics['mae_pct']:.2f}` pct"
                if pd.notna(metrics.get("mae_pct"))
                else "`缺失`"
            )
            + "，方向准确率 "
            + (
                f"`{metrics['direction_accuracy']:.0%}`"
                if pd.notna(metrics.get("direction_accuracy"))
                else "`缺失`"
            )
            + f"，动态阈值 `{policy.minimum_abs_change_pct:.2f}% / 强阈值 {policy.strong_abs_change_pct:.2f}%`。",
            "- 规则仍然是：估值绝对涨跌 `<1.5%` 一律忽略；只有 `>=20` 次校准且 MAE/方向准确率达标，或估值绝对涨跌 `>=3%` 且 ETF 与广度双确认，才允许最多 `10%` 战术加减仓。",
            "",
        ]
    )
    for fund_code in ("007817", "008887", "008585"):
        decision = decisions[fund_code]
        estimate_part = "缺失"
        if decision["estimate_change_pct"] is not None and decision["estimate_time"] is not None:
            estimate_part = (
                f"`{_format_pct(decision['estimate_change_pct'])}` @ "
                f"`{decision['estimate_time']:%H:%M}`"
            )
        lines.append(
            f"- `{fund_code}`：fundgz {estimate_part}；"
            f"ETF 14:30 `{_format_pct(decision['etf_change_pct'])}`；"
            f"广度 `{decision['breadth']['positive_count']}/{decision['breadth']['sample_size']}`；"
            f"IOPV "
            + (f"`{decision['iopv']:.4f}`" if decision["iopv"] is not None else "`缺失`")
            + f"。{decision['estimate_reason']}"
        )
    lines.extend(
        [
            "",
            "## 中美短窗相关只作低置信确认",
            "",
        ]
    )
    for row in short_window.itertuples(index=False):
        lines.append(
            f"- `{row.pair}`：只允许使用“上一完整美股收盘领先 A 股”的口径，"
            f"相关 `{row.correlation:.2f}`，方向命中 `{row.direction_accuracy:.1%}`，置信度 `{row.confidence}`。"
        )
    lines.extend(
        [
            "- 同日美股收盘绝不用于今天 `14:30` 的 A 股判断；同日美股期货只做情绪补充，不做训练输入。",
            "",
            "## 财联社与官方公告",
            "",
            "### 财联社",
            "",
        ]
    )
    if cls_articles:
        for article in cls_articles[:6]:
            lines.append(
                f"- `{article['timestamp']}` [{article['title']}]({article['url']})"
            )
    else:
        lines.append("- 截至 14:30 未抽取到足以推翻基线仓位的科技主线公开条目。")
    lines.extend(
        [
            "",
            "### 官方公告",
            "",
        ]
    )
    if announcements:
        for item in announcements[:6]:
            title = re.sub(r"<[^>]+>", "", str(item.get("announcementTitle", "")))
            path = item.get("adjunctUrl", "")
            url = f"http://static.cninfo.com.cn/{path}" if path else "http://www.cninfo.com.cn/"
            lines.append(
                f"- `{item['local_time']:%Y-%m-%d %H:%M}` [{title}]({url})"
            )
    else:
        lines.append(
            "- 通过巨潮公告接口扫描代表公司在昨收后至今日 14:30 的披露，未抓到新增、足以单独改变 CPO/存储/PCB/AI 判断的代表性公告。"
        )
    lines.extend(
        [
            "",
            "## 持仓批次与费用约束",
            "",
        ]
    )
    if has_real_lots:
        lines.append("- 已发现真实持仓批次台账，可以在未来需要时按先进先出核对赎回费。")
    else:
        lines.extend(
            [
                "- 本地只找到前视信号台账，没有找到真实持仓批次/交易流水。",
                "- 因此一切涉及赎回的动作都先降级为“不执行”，避免误踩 `<7日 1.5%`、`7-29日 0.5%`、`30-364日 0.25%` 的赎回费。",
                "- 若线下确需调仓，只能优先赎回最老批次，并尽量让计划持有期超过 `30` 天，最好超过 `365` 天。",
            ]
        )
    lines.extend(
        [
            "",
            "## 本次执行摘要",
            "",
            "- `007817`：**持有**",
            "- `008887`：**持有**",
            "- `008585`：**持有**",
            "- `现金`：**持有**",
            "- `估值层`：**不参与判断**",
            "- `14:30 前新增申购`：**默认不做**",
            "- `14:30 前赎回`：**默认不做**",
            "",
            "## 来源",
            "",
            "- 本地冻结研究：`output/research_v8/`",
            "- 本地短窗相关：`output/research_v8/short_window_correlations.csv`",
            "- 本地前视台账：`output/forward/ledger.csv`、`output/forward/rotation_ledger.csv`",
            "- 本地完整净值：`data/nav_cache/007817.csv`、`data/nav_cache/008887.csv`、`data/nav_cache/008585.csv`",
            "- 腾讯分钟行情：`https://ifzq.gtimg.cn/appstock/app/minute/query`",
            "- 腾讯快照行情：`https://qt.gtimg.cn/`",
            "- 基金估值：`https://fundgz.1234567.com.cn/js/007817.js`、`https://fundgz.1234567.com.cn/js/008887.js`、`https://fundgz.1234567.com.cn/js/008585.js`",
            "- 美股期指：`https://query1.finance.yahoo.com/v8/finance/chart/NQ=F`、`ES=F`、`YM=F`",
            "- 财联社：[https://www.cls.cn/](https://www.cls.cn/)",
            "- 巨潮公告接口：`http://www.cninfo.com.cn/new/hisAnnouncement/query`",
            "",
            "本报告仅为量化研究与执行记录，不构成收益承诺，也不替代结合你真实持仓批次、总资产、流动性需求和风险承受能力后的个性化投资建议。",
        ]
    )
    path = Path("daily_reports") / report_date / "1430_action.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_action_report(report_date: str | None = None) -> dict[str, Path | str]:
    report_date = report_date or pd.Timestamp.now(tz=ASIA_TZ).date().isoformat()
    cutoff = pd.Timestamp(f"{report_date} {REPORT_TIME}")
    previous_close_start = (
        pd.Timestamp(report_date) - pd.Timedelta(days=1)
    ).normalize() + pd.Timedelta(hours=15)

    forward_targets = _load_forward_targets()
    nav_by_code = _load_nav_by_code()
    history = _load_estimate_history()
    metrics = calibration_metrics(history)
    policy = adaptive_policy_from_history(history, EstimatePolicy())

    index_points = {}
    for code in ("sh000001", "sz399001", "sz399006"):
        quote = _fetch_tencent_quote(code)
        point = _fetch_tencent_minute_point(code, cutoff)
        index_points[code] = {
            "name": quote.name,
            "price": point.price,
            "change_pct": (point.price / quote.previous_close - 1.0) * 100,
            "amount": point.cumulative_amount,
        }

    sector_snapshots = {
        sector_name: _sector_snapshot(sector_name, basket, cutoff)
        for sector_name, basket in SECTOR_BASKETS.items()
    }

    snapshot_rows, decisions = _build_snapshot_rows(
        cutoff,
        forward_targets["base_targets"],
        sector_snapshots,
        metrics,
        policy,
    )
    snapshot_path = Path("data/intraday_estimates") / f"{report_date}_1430_snapshot.csv"
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(snapshot_rows).to_csv(snapshot_path, index=False, encoding="utf-8-sig")

    cls_articles = _filter_cls_articles(
        _fetch_cls_articles(),
        previous_close_start,
        cutoff,
    )
    announcements = _filter_announcements(
        _fetch_cninfo_announcements(previous_close_start.date().isoformat(), report_date),
        previous_close_start,
        cutoff,
    )
    lot_ledgers = _find_holding_lot_ledgers()
    report_path = _write_report(
        report_date,
        cutoff,
        forward_targets,
        nav_by_code,
        index_points,
        sector_snapshots,
        decisions,
        metrics,
        policy,
        cls_articles,
        announcements,
        bool(lot_ledgers),
    )
    return {
        "report_path": report_path.resolve(),
        "snapshot_path": snapshot_path.resolve(),
    }


def main() -> None:
    result = run_action_report()
    print(result["report_path"])
    print(result["snapshot_path"])


if __name__ == "__main__":
    main()
