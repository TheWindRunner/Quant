"""Generate the 08:30 premarket fund strategy report."""

from __future__ import annotations

from dataclasses import dataclass
import html
import json
from pathlib import Path
import re
import time
from typing import Any

import pandas as pd
import requests

from .close_outlook import _fetch_cls_articles
from .forward_test import (
    append_forward_ledger,
    append_rotation_forward_ledger,
    create_forward_snapshots,
    create_rotation_forward_snapshot,
    score_forward_ledger,
    score_rotation_forward_ledger,
)
from .nav_validation import validate_nav_cache
from .run_research import RESEARCH_FUNDS, build_nav_datasets
from .sector_rotation import rotation_configs


ASIA_TZ = "Asia/Shanghai"
RESEARCH_DIR = Path("output/research_v8")
FORWARD_DIR = Path("output/forward")
TOTAL_TECH_EXPOSURE = 0.90
REPORT_TIME = "08:30:00"

CN_INDEX_SYMBOLS = {
    "上证指数": "000001.SS",
    "深证成指": "399001.SZ",
    "创业板指": "399006.SZ",
}

FUND_TO_EXECUTABLE_PROXY = {
    "CPO/光通信": "007817",
    "存储/半导体代理": "008887",
    "AI/计算机代理": "008585",
}

CN_FOCUS_BASKETS = {
    "CPO/光通信": [
        ("300308.SZ", "中际旭创"),
        ("300502.SZ", "新易盛"),
        ("300394.SZ", "天孚通信"),
        ("002281.SZ", "光迅科技"),
    ],
    "存储": [
        ("603986.SS", "兆易创新"),
        ("301308.SZ", "江波龙"),
        ("688525.SS", "佰维存储"),
        ("300223.SZ", "北京君正"),
    ],
    "PCB": [
        ("300476.SZ", "胜宏科技"),
        ("002463.SZ", "沪电股份"),
        ("002916.SZ", "深南电路"),
        ("600183.SS", "生益科技"),
    ],
    "AI": [
        ("688041.SS", "海光信息"),
        ("603019.SS", "中科曙光"),
        ("002230.SZ", "科大讯飞"),
        ("000977.SZ", "浪潮信息"),
    ],
    "机器人": [
        ("300024.SZ", "机器人"),
        ("002747.SZ", "埃斯顿"),
        ("300124.SZ", "汇川技术"),
        ("688165.SS", "埃夫特"),
    ],
    "计算机": [
        ("603019.SS", "中科曙光"),
        ("000977.SZ", "浪潮信息"),
        ("600845.SS", "宝信软件"),
        ("600536.SS", "中国软件"),
    ],
    "半导体设备": [
        ("002371.SZ", "北方华创"),
        ("688012.SS", "中微公司"),
        ("688120.SS", "华海清科"),
        ("300604.SZ", "长川科技"),
    ],
}

US_BASKETS = {
    "CPO/光通信": [
        ("COHR", "Coherent"),
        ("LITE", "Lumentum"),
        ("AAOI", "Applied Optoelectronics"),
    ],
    "存储": [
        ("MU", "Micron"),
        ("WDC", "Western Digital"),
        ("STX", "Seagate"),
    ],
    "AI硬件": [
        ("NVDA", "NVIDIA"),
        ("SMCI", "Super Micro"),
        ("DELL", "Dell"),
    ],
    "半导体设备": [
        ("AMAT", "Applied Materials"),
        ("LRCX", "Lam Research"),
        ("KLAC", "KLA"),
    ],
}

CNINFO_COMPANIES = [
    ("szse", "sz", "002281,光迅科技"),
    ("szse", "sz", "300502,新易盛"),
    ("szse", "sz", "300394,天孚通信"),
    ("sse", "sh", "688525,佰维存储"),
    ("szse", "sz", "300223,北京君正"),
    ("szse", "sz", "300476,胜宏科技"),
    ("sse", "sh", "600183,生益科技"),
    ("sse", "sh", "603019,中科曙光"),
    ("sse", "sh", "688041,海光信息"),
    ("szse", "sz", "002747,埃斯顿"),
    ("szse", "sz", "002371,北方华创"),
    ("sse", "sh", "688012,中微公司"),
]


@dataclass(frozen=True)
class CloseSnapshot:
    symbol: str
    timestamp: pd.Timestamp
    close: float
    previous_close: float

    @property
    def change_pct(self) -> float:
        if not self.previous_close:
            return 0.0
        return self.close / self.previous_close - 1.0


@dataclass(frozen=True)
class IntradaySnapshot:
    symbol: str
    timestamp: pd.Timestamp
    level: float
    previous_close: float | None

    @property
    def change_pct(self) -> float | None:
        if not self.previous_close:
            return None
        return self.level / self.previous_close - 1.0


def _http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                url,
                params=params,
                headers=headers
                or {
                    "User-Agent": "Mozilla/5.0",
                    "Referer": "https://finance.yahoo.com/",
                },
                timeout=20,
            )
            response.raise_for_status()
            return response
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(1.0 + attempt)
    if last_error is None:
        raise RuntimeError(f"Unable to fetch {url}")
    raise last_error


def _yahoo_chart(
    symbol: str,
    *,
    range_value: str,
    interval: str,
    include_pre_post: bool = False,
) -> dict[str, Any]:
    response = _http_get(
        f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
        params={
            "range": range_value,
            "interval": interval,
            "includePrePost": str(include_pre_post).lower(),
        },
    )
    payload = response.json()["chart"]["result"]
    if not payload:
        raise RuntimeError(f"No Yahoo chart payload for {symbol}")
    return payload[0]


def _chart_frame(payload: dict[str, Any]) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                payload["timestamp"], unit="s", utc=True
            )
            .tz_convert(ASIA_TZ)
            .tz_localize(None),
            "close": payload["indicators"]["quote"][0]["close"],
        }
    ).dropna(subset=["close"])
    return frame.sort_values("timestamp")


def _latest_close_snapshot(
    symbol: str,
    cutoff: pd.Timestamp,
    *,
    range_value: str,
    interval: str,
) -> CloseSnapshot:
    payload = _yahoo_chart(symbol, range_value=range_value, interval=interval)
    frame = _chart_frame(payload)
    available = frame.loc[frame["timestamp"] <= cutoff].copy()
    if available.empty:
        raise RuntimeError(f"No close snapshot for {symbol} before {cutoff}")
    available["session_date"] = available["timestamp"].dt.normalize()
    daily_close = (
        available.groupby("session_date", as_index=False)
        .last()
        .sort_values("session_date")
    )
    if len(daily_close) < 2:
        raise RuntimeError(f"Not enough daily close points for {symbol}")
    last = daily_close.iloc[-1]
    prev = daily_close.iloc[-2]
    return CloseSnapshot(
        symbol=symbol,
        timestamp=pd.Timestamp(last["timestamp"]),
        close=float(last["close"]),
        previous_close=float(prev["close"]),
    )


def _intraday_snapshot(
    symbol: str,
    cutoff: pd.Timestamp,
    *,
    range_value: str = "2d",
    interval: str = "1m",
) -> IntradaySnapshot:
    payload = _yahoo_chart(
        symbol,
        range_value=range_value,
        interval=interval,
        include_pre_post=True,
    )
    frame = _chart_frame(payload)
    available = frame.loc[frame["timestamp"] <= cutoff]
    if available.empty:
        raise RuntimeError(f"No intraday snapshot for {symbol} before {cutoff}")
    point = available.iloc[-1]
    meta = payload.get("meta", {})
    previous_close = meta.get("chartPreviousClose") or meta.get("previousClose")
    return IntradaySnapshot(
        symbol=symbol,
        timestamp=pd.Timestamp(point["timestamp"]),
        level=float(point["close"]),
        previous_close=float(previous_close) if previous_close else None,
    )


def _format_pct(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "缺失"
    return f"{value:.{digits}%}"


def _format_num(value: float | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "缺失"
    return f"{value:.{digits}f}"


def _load_selected_models() -> list[str]:
    models = pd.read_csv(RESEARCH_DIR / "selected_models.csv")
    return models["model"].dropna().tolist()


def _load_rotation_config_name() -> str:
    frame = pd.read_csv(RESEARCH_DIR / "rotation_locked_comparison.csv")
    candidates = frame.loc[
        frame["model"] != "equal_weight_buy_hold", "model"
    ].tolist()
    if not candidates:
        raise RuntimeError("No frozen rotation model found in research_v8")
    return str(candidates[0])


def _rotation_config_by_name(name: str):
    config = next((item for item in rotation_configs() if item.name == name), None)
    if config is None:
        raise RuntimeError(f"Rotation config {name} not found")
    return config


def _sector_snapshot(
    basket: list[tuple[str, str]],
    cutoff: pd.Timestamp,
    *,
    range_value: str,
    interval: str,
) -> dict[str, Any]:
    rows = []
    for symbol, name in basket:
        snap = _latest_close_snapshot(
            symbol,
            cutoff,
            range_value=range_value,
            interval=interval,
        )
        rows.append({"name": name, "change_pct": snap.change_pct})
    frame = pd.DataFrame(rows)
    top = frame.sort_values("change_pct", ascending=False).iloc[0]
    return {
        "average_change_pct": float(frame["change_pct"].mean()),
        "positive_count": int((frame["change_pct"] > 0).sum()),
        "sample_size": int(len(frame)),
        "top_name": str(top["name"]),
        "top_change_pct": float(top["change_pct"]),
    }


def _nav_signal_frame(nav_by_code: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for code, series in nav_by_code.items():
        nav = series.sort_index()
        rows.append(
            {
                "fund_code": code,
                "latest_nav": float(nav.iloc[-1]),
                "latest_nav_date": nav.index[-1].date().isoformat(),
                "return_5d": float(nav.iloc[-1] / nav.iloc[-6] - 1.0),
                "return_20d": float(nav.iloc[-1] / nav.iloc[-21] - 1.0),
                "return_60d": float(nav.iloc[-1] / nav.iloc[-61] - 1.0),
                "ma20_gap": float(nav.iloc[-1] / nav.rolling(20).mean().iloc[-1] - 1.0),
                "vol20": float(nav.pct_change(fill_method=None).rolling(20).std().iloc[-1] * (252**0.5)),
            }
        )
    return pd.DataFrame(rows)


def _fund_display_name(code: str) -> str:
    for definition in RESEARCH_FUNDS.values():
        if definition["code"] == code:
            return str(definition["name"])
    return code


def _fund_notice_status() -> list[str]:
    return [
        "- 基金公司公告页的稳定、免登录、可编程分页接口未确认，盘前不编造“无公告”结论。",
        "- 因此 007817 / 008887 / 008585 的暂停申购、限大额申购与实际费率，仍必须在支付宝下单页二次核对。",
    ]


def _fetch_cninfo_announcements(
    start_date: str,
    end_date: str,
) -> list[dict[str, Any]]:
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


def _filter_announcements(
    announcements: list[dict[str, Any]],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict[str, Any]]:
    rows = []
    for item in announcements:
        ts = pd.to_datetime(
            item.get("announcementTime"), unit="ms", errors="coerce"
        )
        if pd.isna(ts):
            continue
        ts = ts.tz_localize("UTC").tz_convert(ASIA_TZ).tz_localize(None)
        if start_ts <= ts <= end_ts:
            rows.append({**item, "local_time": ts})
    rows.sort(key=lambda item: item["local_time"], reverse=True)
    return rows


def _filter_cls_articles(
    articles: list[dict[str, str]],
    start_ts: pd.Timestamp,
    end_ts: pd.Timestamp,
) -> list[dict[str, str]]:
    rows = []
    for article in articles:
        raw = article.get("timestamp") or ""
        if not raw:
            continue
        ts = pd.Timestamp(raw)
        if start_ts <= ts <= end_ts:
            rows.append(article)
    return rows


def _score_ledgers(nav_by_code: dict[str, pd.Series]) -> tuple[pd.DataFrame, pd.DataFrame]:
    ledger_path = FORWARD_DIR / "ledger.csv"
    rotation_path = FORWARD_DIR / "rotation_ledger.csv"
    ledger = pd.read_csv(ledger_path, dtype={"fund_code": str})
    rotation = pd.read_csv(rotation_path)
    scored_ledger = score_forward_ledger(ledger, nav_by_code)
    nav_by_asset = {
        asset: nav_by_code[definition["code"]]
        for asset, definition in RESEARCH_FUNDS.items()
    }
    scored_rotation = score_rotation_forward_ledger(rotation, nav_by_asset)
    scored_ledger.to_csv(ledger_path, index=False, encoding="utf-8-sig")
    scored_rotation.to_csv(rotation_path, index=False, encoding="utf-8-sig")
    return scored_ledger, scored_rotation


def _forward_lines(scored_ledger: pd.DataFrame, signal_date: str) -> list[str]:
    rows = (
        scored_ledger.loc[scored_ledger["signal_date"] == signal_date]
        .copy()
        .sort_values(["fund_code", "generated_at"])
        .drop_duplicates(["fund_code"], keep="last")
    )
    if rows.empty:
        return ["- 当日盘前影子单基金信号尚未写入。"]
    lines = []
    for row in rows.itertuples(index=False):
        lines.append(
            f"- `{str(row.fund_code).zfill(6)}` 影子候选仓位 `{float(row.candidate_position):.0%}`，"
            f"生产冻结仓位 `{float(row.deployed_position):.0%}`，模型版本 `{row.model_version}`。"
        )
    return lines


def _rotation_line(scored_rotation: pd.DataFrame, signal_date: str) -> str:
    rows = (
        scored_rotation.loc[scored_rotation["signal_date"] == signal_date]
        .sort_values(["generated_at"])
        .drop_duplicates(["rotation_model"], keep="last")
    )
    if rows.empty:
        return "- 当日科技组合候选权重尚未写入。"
    row = rows.iloc[-1]
    return (
        "- 科技组合影子候选权重："
        f"`CPO {float(row['candidate_cpo_weight']):.0%} / "
        f"存储代理 {float(row['candidate_memory_weight']):.0%} / "
        f"AI {float(row['candidate_ai_weight']):.0%}`，"
        f"冻结生产权重仍为等权 `{float(row['deployed_cpo_weight']):.0%} / "
        f"{float(row['deployed_memory_weight']):.0%} / "
        f"{float(row['deployed_ai_weight']):.0%}`。"
    )


def _backtest_table() -> str:
    locked = pd.read_csv(RESEARCH_DIR / "locked_six_month_test.csv")
    theme = pd.read_csv(RESEARCH_DIR / "theme_locked_comparison.csv")
    rows = []
    for asset, code in (
        ("cpo_communication", "007817"),
        ("memory_semiconductor_proxy", "008887"),
        ("artificial_intelligence", "008585"),
    ):
        buy_hold = float(
            locked.loc[
                (locked["asset"] == asset) & (locked["model"] == "buy_hold"),
                "total_return",
            ].iloc[0]
        )
        dual_ma = float(
            locked.loc[
                (locked["asset"] == asset) & (locked["model"] == "dual_ma"),
                "total_return",
            ].iloc[0]
        )
        unified = float(
            locked.loc[
                (locked["asset"] == asset)
                & (locked["model"] == "trained_ensemble"),
                "total_return",
            ].iloc[0]
        )
        theme_selected = float(
            theme.loc[
                (theme["asset"] == asset)
                & (theme["model"] == "theme_selected"),
                "total_return",
            ].iloc[0]
        )
        rows.append(
            f"| {code} | {buy_hold:.2%} | {dual_ma:.2%} | {unified:.2%} | {theme_selected:.2%} |"
        )
    return "\n".join(
        [
            "| 基金 | 买入持有 | 20/60均线 | 训练期统一模型 | 逐主题模型 |",
            "|---|---:|---:|---:|---:|",
            *rows,
        ]
    )


def _research_summary_lines() -> list[str]:
    rotation_locked = pd.read_csv(RESEARCH_DIR / "rotation_locked_comparison.csv")
    risk_budget_locked = pd.read_csv(RESEARCH_DIR / "risk_budget_locked_comparison.csv")
    rotation_walk = pd.read_csv(RESEARCH_DIR / "rotation_walk_forward.csv")
    risk_budget_walk = pd.read_csv(RESEARCH_DIR / "risk_budget_walk_forward.csv")
    diagnostics = json.loads(
        (RESEARCH_DIR / "overfitting_diagnostics.json").read_text(encoding="utf-8")
    )
    single_pbo = next(
        item["pbo"] for item in diagnostics if item["model_group"] == "single_fund_zoo"
    )
    rotation_pbo = next(
        item["pbo"] for item in diagnostics if item["model_group"] == "sector_rotation"
    )
    rotation_equal = float(rotation_locked.iloc[0]["total_return"])
    rotation_candidate = float(rotation_locked.iloc[1]["total_return"])
    risk_budget = float(risk_budget_locked.iloc[1]["total_return"])
    return [
        f"- 低频超配组合近半年含费 `{rotation_candidate:.2%}`，等权买入持有 `{rotation_equal:.2%}`，固定风险预算 `{risk_budget:.2%}`。",
        "- 该组合优势来自测试起点前就已确定的 CPO 静态超配，而不是近半年期间成功换仓。",
        f"- 滚动历史模拟实盘共 `{len(rotation_walk)}` 段，跑赢比例 `{(rotation_walk['excess_return'] > 0).mean():.0%}`，"
        f"超额中位数 `{rotation_walk['excess_return'].median():.2%}`；固定风险预算跑赢比例 `{(risk_budget_walk['excess_return'] > 0).mean():.0%}`，"
        f"超额中位数 `{risk_budget_walk['excess_return'].median():.2%}`。",
        f"- PBO 结论：单基金模型库 `{single_pbo:.2%}`，组合轮动 `{rotation_pbo:.2%}`，都不足以支持“稳定超额”表述。",
    ]


def _short_window_lines() -> list[str]:
    frame = pd.read_csv(RESEARCH_DIR / "short_window_correlations.csv")
    frame = frame.loc[frame["us_lead_trading_closes"] == 1]
    lines = []
    for row in frame.itertuples(index=False):
        lines.append(
            f"- `{row.pair}`：相关 `{row.correlation:.2f}`，方向命中 `{row.direction_accuracy:.1%}`，置信度 `{row.confidence}`。"
        )
    lines.append(
        "- 上述短窗结果只作低置信方向确认，不进入训练；长期中美公开相关性序列本轮未取得稳定、可复核数据，继续视为缺失。"
    )
    return lines


def _announcement_lines(announcements: list[dict[str, Any]]) -> list[str]:
    if not announcements:
        return [
            "- 通过巨潮公告接口扫描代表公司在昨收后至今早 08:30 的披露，未抓到新增与 CPO/存储/PCB/AI/机器人/计算机/半导体设备直接相关的代表性公告。"
        ]
    lines = []
    for item in announcements[:6]:
        title = re.sub(r"<[^>]+>", "", str(item.get("announcementTitle", "")))
        path = item.get("adjunctUrl", "")
        url = f"http://static.cninfo.com.cn/{path}" if path else "http://www.cninfo.com.cn/"
        lines.append(
            f"- `{item['local_time']:%Y-%m-%d %H:%M}` [{title}]({url})"
        )
    return lines


def _cls_lines(articles: list[dict[str, str]]) -> list[str]:
    if not articles:
        return ["- 财联社昨收后至今早 08:30 的可提取科技主线条目有限，未发现足以单独推翻基线仓位的新信息。"]
    lines = []
    for article in articles[:5]:
        lines.append(
            f"- `{article['timestamp']}` [{article['title']}]({article['url']})"
        )
    return lines


def _plan_lines(
    signal_date: str,
    model_version: str,
    nav_signals: pd.DataFrame,
    candidate_weights: dict[str, float],
) -> list[str]:
    signal_rows = {
        row["fund_code"]: row
        for row in nav_signals.to_dict("records")
    }
    return [
        "| 基金 | 主题定位 | 14:30前观察条件 | 计划动作 | 目标仓位 | 失效条件 |",
        "|---|---|---|---|---:|---|",
        f"| 007817 | CPO/光通信 | 通信ETF广度不转弱，且盘中回撤后仍守住中期趋势；若追涨过快则只观察 | 以持有为主，仅在可靠回撤确认时小幅加仓 | {TOTAL_TECH_EXPOSURE * candidate_weights['cpo_communication']:.0%} | 正式净值后续有效跌破 MA20 且继续恶化；或 14:30 估值源缺失且板块广度转差 |",
        f"| 008887 | 存储/半导体代理 | 存储与半导体设备需同步偏强，不能只靠单一存储故事 | 只保留小仓/观察，不追高 | {TOTAL_TECH_EXPOSURE * candidate_weights['memory_semiconductor_proxy']:.0%} | 仍低于 MA20 且板块分化扩大；或只见个股强、不见基金代理广度 |",
        f"| 008585 | AI/计算机代理 | AI硬件、算力、计算机广度共振才恢复加仓候选 | 持有，不主动新增 | {TOTAL_TECH_EXPOSURE * candidate_weights['artificial_intelligence']:.0%} | 14:30 共振不成立，或 AI 链转为高位兑现 |",
        f"| 现金 | 风险缓冲 | 估值源不可用、新闻冲突或午后广度走弱时保留 | 不低于基线现金 | {1 - TOTAL_TECH_EXPOSURE:.0%} | 只有可靠盘中确认才动用，不因单条新闻清空 |",
        f"",
        f"- 当日冻结影子信号日期：`{signal_date}`，模型版本：`{model_version}`。",
        f"- 单基金影子信号仍然是：007817 候选 `100%`、008887 候选 `70%`、008585 候选 `70%`；但未通过前视验收，生产仓位不得据此机械切换。",
        "- 14:30 若估值源仍不可得，或可得但未积累到至少 20 次校准样本并达标，禁止把估值层当成基础仓位开关。",
        "- 盘中估值即便可靠，也只允许最多 10% 的战术微调，不得推翻 54%/18%/18%/10% 的基线框架。",
    ]


def _write_report(
    report_date: str,
    report_asof: pd.Timestamp,
    nav_validation: pd.DataFrame,
    nav_by_code: dict[str, pd.Series],
    nav_signals: pd.DataFrame,
    forward_lines: list[str],
    rotation_line: str,
    cn_indices: dict[str, CloseSnapshot],
    us_indices: dict[str, CloseSnapshot],
    vix: CloseSnapshot,
    treasury: CloseSnapshot,
    intraday_points: dict[str, IntradaySnapshot],
    cn_sector_lines: list[str],
    us_sector_lines: list[str],
    cls_lines: list[str],
    announcement_lines: list[str],
    plan_lines: list[str],
) -> Path:
    latest_nav_date = max(series.index.max() for series in nav_by_code.values()).date().isoformat()
    short_window = _short_window_lines()
    research_summary = _research_summary_lines()
    model_version = (
        pd.read_csv(FORWARD_DIR / "ledger.csv", dtype={"fund_code": str})
        .sort_values(["signal_date", "generated_at"])
        .iloc[-1]["model_version"]
    )
    signal_date = (
        pd.read_csv(FORWARD_DIR / "ledger.csv", dtype={"fund_code": str})
        .sort_values(["signal_date", "generated_at"])
        .iloc[-1]["signal_date"]
    )
    lines = [
        f"# {report_date} 08:30盘前基金策略日报",
        "",
        f"生成口径：北京时间 `{report_asof:%Y-%m-%d %H:%M}`",
        f"正式净值截止：`{latest_nav_date}`",
        "",
        "> 严格说明：第一轮近半年锁定测试已被查看，同区间后续结果只是迭代对照；真正未见样本来自不可修改的每日前视台账。未经前视验证的模型，不得宣称稳定超额。",
        "",
        "## 净值校验",
        "",
        "```text",
        nav_validation.to_string(index=False),
        "```",
        "",
        "## 结论",
        "",
        "- 今日盘前基线仍是中性偏多，但执行上只允许在 14:30 之前观察、等待确认，不做开盘追涨。",
        "- 唯一可执行对象仍限定为支付宝可购买的场外基金：`007817 / 008887 / 008585`。",
        "- 盘前主线排序仍是：`CPO/光通信 > 存储/半导体代理 > AI/计算机代理`；PCB、机器人、半导体设备只作景气确认，不单独映射到新的可执行基金。",
        "- 当前建议维持基线目标仓位：`007817 54% / 008887 18% / 008585 18% / 现金 10%`，直到 14:30 再做是否微调的判断。",
        "",
        "## 昨日A股与隔夜外盘",
        "",
        f"- 上证指数昨收 `{cn_indices['上证指数'].close:.2f}`，日涨跌 `{cn_indices['上证指数'].change_pct:.2%}`；深证成指 `{cn_indices['深证成指'].change_pct:.2%}`；创业板指 `{cn_indices['创业板指'].change_pct:.2%}`。",
        f"- 美股隔夜收盘：纳指 `{us_indices['纳指'].change_pct:.2%}`，标普500 `{us_indices['标普500'].change_pct:.2%}`，道指 `{us_indices['道指'].change_pct:.2%}`。",
        f"- VIX 收在 `{vix.close:.2f}`，较前一交易日 `{vix.change_pct:.2%}`；10年美债收益率 `{treasury.close:.3f}`，变动 `{treasury.change_pct:.2%}`。",
        f"- 08:30 可得美股期货：`NQ {intraday_points['NQ=F'].level:.2f}`（{_format_pct(intraday_points['NQ=F'].change_pct)}），"
        f"`ES {intraday_points['ES=F'].level:.2f}`（{_format_pct(intraday_points['ES=F'].change_pct)}），"
        f"`YM {intraday_points['YM=F'].level:.2f}`（{_format_pct(intraday_points['YM=F'].change_pct)}）。",
        f"- 08:30 可得油金与人民币：`WTI {intraday_points['CL=F'].level:.2f}`（{_format_pct(intraday_points['CL=F'].change_pct)}），"
        f"`黄金 {intraday_points['GC=F'].level:.2f}`（{_format_pct(intraday_points['GC=F'].change_pct)}），"
        f"`离岸人民币 {intraday_points['CNH=X'].level:.4f}`（{_format_pct(intraday_points['CNH=X'].change_pct)}）。",
        "",
        "## 板块扫描",
        "",
        *cn_sector_lines,
        *us_sector_lines,
        "",
        "## 冻结影子信号与前视台账",
        "",
        *forward_lines,
        rotation_line,
        "",
        "## 14:30观察条件、候选基金与计划动作",
        "",
        *plan_lines,
        "",
        "## 最新净值信号",
        "",
        "| 基金 | 最新净值 | 5日 | 20日 | 60日 | 相对MA20 | 20日年化波动 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in nav_signals.sort_values("fund_code").itertuples(index=False):
        lines.append(
            f"| {row.fund_code} | {row.latest_nav:.4f} | {row.return_5d:.2%} | {row.return_20d:.2%} | "
            f"{row.return_60d:.2%} | {row.ma20_gap:.2%} | {row.vol20:.2%} |"
        )
    lines.extend(
        [
            "",
            "## 近半年含费对照",
            "",
            _backtest_table(),
            "",
            *research_summary,
            "",
            "## 中美短窗相关性",
            "",
            *short_window,
            "",
            "## 财联社与公告",
            "",
            "### 财联社",
            "",
            *cls_lines,
            "",
            "### 交易所/基金公司公告",
            "",
            *announcement_lines,
            "",
            *_fund_notice_status(),
            "",
            "## 缺口与边界",
            "",
            "- 本报告使用的中美短窗相关性只来自 `output/research_v8/short_window_correlations.csv`，仅用于低置信方向确认，不进入训练。",
            "- 未新增长期中美公开相关性数据；若未来取得稳定、可复核长窗序列，再单独评估。",
            "- 盘前报告不使用 08:30 之后才出现的行情、公告和新闻，不追溯篡改当时可得结论。",
            "",
            "## 来源",
            "",
            "- 本地冻结研究：`output/research_v8/research_report.md`、`locked_six_month_test.csv`、`theme_locked_comparison.csv`、`rotation_locked_comparison.csv`、`rotation_walk_forward.csv`、`risk_budget_locked_comparison.csv`、`risk_budget_walk_forward.csv`、`short_window_correlations.csv`",
            "- 本地前视台账：`output/forward/ledger.csv`、`output/forward/rotation_ledger.csv`",
            "- 官方净值：[007817](https://fund.eastmoney.com/pingzhongdata/007817.js) / [008887](https://fund.eastmoney.com/pingzhongdata/008887.js) / [008585](https://fund.eastmoney.com/pingzhongdata/008585.js)",
            "- 行情快照：[Yahoo Finance Chart API](https://query1.finance.yahoo.com/v8/finance/chart/NQ=F) / [Sina 行情](https://hq.sinajs.cn/)",
            "- 财联社：[首页](https://www.cls.cn/)",
            "- 巨潮公告接口：`http://www.cninfo.com.cn/new/hisAnnouncement/query`",
        ]
    )
    path = Path("daily_reports") / report_date / "0830_premarket.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_premarket_report(report_date: str | None = None) -> Path:
    report_date = report_date or pd.Timestamp.now(tz=ASIA_TZ).date().isoformat()
    report_asof = pd.Timestamp(f"{report_date} {REPORT_TIME}", tz=ASIA_TZ)
    previous_close_window_start = (
        report_asof - pd.Timedelta(days=1)
    ).normalize() + pd.Timedelta(hours=15)

    nav_validation = validate_nav_cache(
        "data/nav_cache",
        tuple(definition["code"] for definition in RESEARCH_FUNDS.values()),
    )
    if not nav_validation["valid"].fillna(False).all():
        raise RuntimeError(
            "NAV cache failed integrity checks:\n" + nav_validation.to_string(index=False)
        )

    selected_models = _load_selected_models()
    rotation_name = _load_rotation_config_name()
    rotation_config = _rotation_config_by_name(rotation_name)

    snapshots = create_forward_snapshots(
        selected_models,
        False,
        generated_at=report_asof,
    )
    append_forward_ledger(snapshots, FORWARD_DIR / "ledger.csv")
    rotation_snapshot = create_rotation_forward_snapshot(
        rotation_config,
        False,
        generated_at=report_asof,
    )
    append_rotation_forward_ledger(
        rotation_snapshot,
        FORWARD_DIR / "rotation_ledger.csv",
    )

    datasets = build_nav_datasets()
    nav_by_code = {
        definition["code"]: datasets[asset]["nav"].rename(definition["code"])
        for asset, definition in RESEARCH_FUNDS.items()
    }
    scored_ledger, scored_rotation = _score_ledgers(nav_by_code)
    signal_date = max(series.index.max() for series in nav_by_code.values()).date().isoformat()
    model_version = snapshots[0].model_version

    nav_signals = _nav_signal_frame(nav_by_code)
    candidate_weights = {
        "cpo_communication": rotation_snapshot.candidate_cpo_weight,
        "memory_semiconductor_proxy": rotation_snapshot.candidate_memory_weight,
        "artificial_intelligence": rotation_snapshot.candidate_ai_weight,
    }

    cn_indices = {
        name: _latest_close_snapshot(
            symbol,
            report_asof.tz_localize(None),
            range_value="5d",
            interval="5m",
        )
        for name, symbol in CN_INDEX_SYMBOLS.items()
    }
    us_indices = {
        "纳指": _latest_close_snapshot(
            "^IXIC",
            report_asof.tz_localize(None),
            range_value="5d",
            interval="5m",
        ),
        "标普500": _latest_close_snapshot(
            "^GSPC",
            report_asof.tz_localize(None),
            range_value="5d",
            interval="5m",
        ),
        "道指": _latest_close_snapshot(
            "^DJI",
            report_asof.tz_localize(None),
            range_value="5d",
            interval="5m",
        ),
    }
    vix = _latest_close_snapshot(
        "^VIX",
        report_asof.tz_localize(None),
        range_value="5d",
        interval="5m",
    )
    treasury = _latest_close_snapshot(
        "^TNX",
        report_asof.tz_localize(None),
        range_value="5d",
        interval="5m",
    )
    intraday_points = {
        symbol: _intraday_snapshot(symbol, report_asof.tz_localize(None))
        for symbol in ("NQ=F", "ES=F", "YM=F", "CL=F", "GC=F", "CNH=X")
    }

    cn_sector_lines = []
    sector_ranking = []
    for sector, basket in CN_FOCUS_BASKETS.items():
        snapshot = _sector_snapshot(
            basket,
            report_asof.tz_localize(None),
            range_value="15d",
            interval="1d",
        )
        sector_ranking.append((sector, snapshot["average_change_pct"]))
        executable = "仅观察"
        if sector in FUND_TO_EXECUTABLE_PROXY:
            executable = FUND_TO_EXECUTABLE_PROXY[sector]
        elif sector in {"PCB", "半导体设备"}:
            executable = "008887仅代理，不纯"
        elif sector in {"计算机", "机器人"}:
            executable = "008585仅代理，不纯"
        cn_sector_lines.append(
            f"- `{sector}`：A股代表样本均涨 `{snapshot['average_change_pct']:.2%}`，"
            f"上涨 `{snapshot['positive_count']}/{snapshot['sample_size']}`，"
            f"最强 `{snapshot['top_name']}` `{snapshot['top_change_pct']:.2%}`；可执行映射 `{executable}`。"
        )
    sector_ranking.sort(key=lambda item: item[1], reverse=True)
    cn_sector_lines.insert(
        0,
        "- A股昨收强弱排序："
        + " > ".join(f"`{name}`" for name, _ in sector_ranking),
    )

    us_sector_lines = []
    for sector, basket in US_BASKETS.items():
        snapshot = _sector_snapshot(
            basket,
            report_asof.tz_localize(None),
            range_value="5d",
            interval="5m",
        )
        us_sector_lines.append(
            f"- `美股{sector}`：隔夜代表样本均涨 `{snapshot['average_change_pct']:.2%}`，"
            f"上涨 `{snapshot['positive_count']}/{snapshot['sample_size']}`，"
            f"最强 `{snapshot['top_name']}` `{snapshot['top_change_pct']:.2%}`。"
        )

    cls_articles = _filter_cls_articles(
        _fetch_cls_articles(limit=16),
        previous_close_window_start.tz_localize(None),
        report_asof.tz_localize(None),
    )
    announcements = _filter_announcements(
        _fetch_cninfo_announcements(
            previous_close_window_start.date().isoformat(),
            report_asof.date().isoformat(),
        ),
        previous_close_window_start.tz_localize(None),
        report_asof.tz_localize(None),
    )

    path = _write_report(
        report_date,
        report_asof,
        nav_validation,
        nav_by_code,
        nav_signals,
        _forward_lines(scored_ledger, signal_date),
        _rotation_line(scored_rotation, signal_date),
        cn_indices,
        us_indices,
        vix,
        treasury,
        intraday_points,
        cn_sector_lines,
        us_sector_lines,
        _cls_lines(cls_articles),
        _announcement_lines(announcements),
        _plan_lines(signal_date, model_version, nav_signals, candidate_weights),
    )
    return path


def main() -> None:
    path = run_premarket_report()
    print(path.resolve())


if __name__ == "__main__":
    main()
