"""Source-aware market news collection and deterministic sentiment scoring."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Iterable

import pandas as pd


POSITIVE_WORDS = (
    "上调", "增长", "回暖", "突破", "增持", "利好", "超预期", "创新高",
    "中标", "扩产", "降息", "支持", "复苏", "盈利",
)
NEGATIVE_WORDS = (
    "下调", "下降", "风险", "制裁", "减持", "亏损", "暴跌", "违约",
    "调查", "处罚", "退市", "收紧", "不及预期", "冲突",
)

SECTOR_KEYWORDS = {
    "technology": ("科技", "软件", "云计算", "人工智能", "AI"),
    "semiconductor": ("半导体", "芯片", "晶圆", "光刻"),
    "communication": ("通信", "5G", "运营商", "光模块"),
    "consumer_discretionary": ("汽车", "家电", "旅游", "消费电子"),
    "consumer_staples": ("食品", "白酒", "农业", "必选消费"),
    "healthcare": ("医药", "医疗", "创新药", "生物"),
    "financials": ("银行", "保险", "券商", "金融"),
    "industrials": ("机械", "工业", "基建", "制造"),
    "materials": ("有色", "钢铁", "化工", "材料"),
    "energy": ("石油", "煤炭", "天然气", "能源"),
    "utilities": ("电力", "公用事业", "水务"),
    "real_estate": ("地产", "房地产", "物业"),
}

SOURCE_WEIGHTS = {
    "财联社": 1.0,
    "新华社": 1.0,
    "中国证券报": 0.9,
    "证券时报": 0.9,
    "上证报": 0.9,
    "其他": 0.6,
}


@dataclass(frozen=True)
class NewsItem:
    published_at: datetime
    title: str
    content: str = ""
    source: str = "其他"
    url: str = ""


def classify_text(text: str) -> tuple[float, list[str]]:
    pos = sum(text.count(word) for word in POSITIVE_WORDS)
    neg = sum(text.count(word) for word in NEGATIVE_WORDS)
    sentiment = (pos - neg) / max(1, pos + neg)
    sectors = [
        sector
        for sector, words in SECTOR_KEYWORDS.items()
        if any(word.lower() in text.lower() for word in words)
    ]
    return float(sentiment), sectors


def aggregate_news(
    items: Iterable[NewsItem],
    as_of: datetime | None = None,
    half_life_hours: float = 36.0,
) -> pd.DataFrame:
    as_of = as_of or datetime.now(timezone.utc)
    totals = {sector: [0.0, 0.0, 0] for sector in SECTOR_KEYWORDS}
    for item in items:
        published = item.published_at
        if published.tzinfo is None:
            published = published.replace(tzinfo=timezone.utc)
        age_hours = max(0.0, (as_of - published).total_seconds() / 3600)
        decay = math.exp(-math.log(2) * age_hours / half_life_hours)
        source_weight = SOURCE_WEIGHTS.get(item.source, SOURCE_WEIGHTS["其他"])
        sentiment, sectors = classify_text(f"{item.title} {item.content}")
        for sector in sectors:
            weight = decay * source_weight
            totals[sector][0] += sentiment * weight
            totals[sector][1] += weight
            totals[sector][2] += 1
    rows = []
    for sector, (weighted, weight, count) in totals.items():
        rows.append(
            {
                "sector": sector,
                "news_score": weighted / weight if weight else 0.0,
                "news_count": count,
            }
        )
    return pd.DataFrame(rows).set_index("sector")


def fetch_cls_news(limit: int = 100) -> list[NewsItem]:
    """Fetch the latest CLS telegraphs through AkShare when installed."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("Install the live extra: pip install -e .[live]") from exc

    frame = ak.stock_info_global_cls()
    if frame.empty:
        return []
    columns = {str(col): col for col in frame.columns}
    title_col = columns.get("标题")
    content_col = columns.get("内容")
    time_col = columns.get("发布时间") or columns.get("时间")
    if not title_col or not time_col:
        raise RuntimeError(f"Unexpected CLS columns: {list(frame.columns)}")
    items = []
    for _, row in frame.head(limit).iterrows():
        ts = pd.to_datetime(row[time_col], errors="coerce")
        if pd.isna(ts):
            continue
        items.append(
            NewsItem(
                published_at=ts.to_pydatetime(),
                title=str(row[title_col]),
                content=str(row[content_col]) if content_col else "",
                source="财联社",
            )
        )
    return items

