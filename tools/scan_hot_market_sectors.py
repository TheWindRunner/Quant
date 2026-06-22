from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from datetime import datetime
import html
import io
import json
from pathlib import Path
import re
import sys
import time
from typing import Iterable

import numpy as np
import pandas as pd
import requests


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "hot_sector_scan"
OUT.mkdir(parents=True, exist_ok=True)


SECTOR_BASKETS = {
    "PCB": [
        ("300476", "胜宏科技"),
        ("002463", "沪电股份"),
        ("002916", "深南电路"),
        ("600183", "生益科技"),
        ("688183", "生益电子"),
        ("301377", "鼎泰高科"),
        ("301200", "大族数控"),
    ],
    "CPO/光通信": [
        ("300308", "中际旭创"),
        ("300502", "新易盛"),
        ("300394", "天孚通信"),
        ("002281", "光迅科技"),
        ("688498", "源杰科技"),
        ("688195", "腾景科技"),
    ],
    "存储/半导体": [
        ("603986", "兆易创新"),
        ("301308", "江波龙"),
        ("688525", "佰维存储"),
        ("300223", "北京君正"),
        ("002371", "北方华创"),
        ("688012", "中微公司"),
    ],
    "AI/算力": [
        ("688041", "海光信息"),
        ("603019", "中科曙光"),
        ("002230", "科大讯飞"),
        ("000977", "浪潮信息"),
        ("300033", "同花顺"),
    ],
    "机器人": [
        ("300024", "机器人"),
        ("002747", "埃斯顿"),
        ("300124", "汇川技术"),
        ("688165", "埃夫特"),
        ("002896", "中大力德"),
    ],
    "消费电子": [
        ("603160", "汇顶科技"),
        ("300408", "三环集团"),
        ("002475", "立讯精密"),
        ("000725", "京东方A"),
        ("002241", "歌尔股份"),
    ],
    "化工": [
        ("600309", "万华化学"),
        ("002601", "龙佰集团"),
        ("600426", "华鲁恒升"),
        ("000301", "东方盛虹"),
        ("600989", "宝丰能源"),
    ],
    "有色": [
        ("601899", "紫金矿业"),
        ("603993", "洛阳钼业"),
        ("600547", "山东黄金"),
        ("000807", "云铝股份"),
        ("002466", "天齐锂业"),
    ],
    "电力/电网": [
        ("600900", "长江电力"),
        ("600406", "国电南瑞"),
        ("600905", "三峡能源"),
        ("601985", "中国核电"),
        ("000400", "许继电气"),
    ],
}


SECTOR_KEYWORDS = {
    "PCB": ["PCB", "印制电路", "覆铜板", "HDI", "算力板"],
    "CPO/光通信": ["CPO", "光模块", "光通信", "硅光", "800G", "1.6T"],
    "存储/半导体": ["存储", "半导体", "芯片", "HBM", "晶圆", "先进封装"],
    "AI/算力": ["AI", "人工智能", "算力", "大模型", "服务器", "液冷"],
    "机器人": ["机器人", "具身智能", "减速器", "伺服", "灵巧手"],
    "消费电子": ["消费电子", "手机", "MR", "AR", "端侧AI", "苹果"],
    "化工": ["化工", "MDI", "TDI", "纯碱", "煤化工", "制冷剂"],
    "有色": ["有色", "铜", "铝", "黄金", "锂", "稀土"],
    "电力/电网": ["电力", "电网", "特高压", "储能", "绿电", "核电"],
}


FUND_MAPPING = {
    "PCB": [
        ("720001", "财通价值动量混合", "PCB主动代理，成立时间长"),
        ("024481", "财通品质甄选混合", "PCB补位，历史较短"),
        ("021523", "财通价值动量混合C", "PCB/制造补位"),
        ("021528", "财通成长优选混合C", "PCB/科技成长补位"),
    ],
    "CPO/光通信": [("007817", "国泰中证全指通信设备ETF联接A", "CPO/通信设备代理")],
    "存储/半导体": [
        ("008887", "华夏国证半导体芯片ETF联接A", "存储/半导体代理"),
        ("006503", "财通集成电路产业股票C", "半导体主动增强"),
    ],
    "AI/算力": [("008585", "华夏中证人工智能主题ETF联接A", "AI主题代理")],
    "化工": [("014942", "鹏华中证细分化工产业主题ETF联接A", "化工主题代理")],
    "有色": [("004432", "南方中证申万有色金属ETF发起联接A", "有色主题代理")],
    "电力/电网": [("018034", "国泰国证绿色电力ETF发起联接A", "电力代理")],
}


@dataclass(frozen=True)
class NewsItem:
    title: str
    content: str
    source: str
    url: str = ""


def safe_call(label: str, func, *args, retries: int = 2, sleep_seconds: float = 1.0, **kwargs):
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                return func(*args, **kwargs), ""
        except Exception as exc:  # noqa: BLE001
            last_error = f"{label} attempt {attempt}: {type(exc).__name__}: {exc}"
            if attempt < retries:
                time.sleep(sleep_seconds)
    return None, last_error


def normalize_code(value: object) -> str:
    digits = re.sub(r"\D", "", str(value))
    return digits[-6:].zfill(6) if digits else ""


def to_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    parsed = pd.to_numeric(str(value).replace("%", "").replace(",", ""), errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def find_col(frame: pd.DataFrame, names: Iterable[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    for col in frame.columns:
        if any(name in str(col) for name in names):
            return str(col)
    return None


def zscore(series: pd.Series) -> pd.Series:
    std = series.std()
    if pd.isna(std) or std == 0:
        return pd.Series(0.0, index=series.index)
    return (series - series.mean()) / std


def fetch_board_frames(ak) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    frames = []
    for source_name, func_name in [
        ("东方财富概念板块", "stock_board_concept_name_em"),
        ("东方财富行业板块", "stock_board_industry_name_em"),
    ]:
        frame, err = safe_call(func_name, getattr(ak, func_name), retries=2)
        if err:
            errors.append(err)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            item = normalize_board_frame(frame, source_name)
            if not item.empty:
                frames.append(item)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), errors


def normalize_board_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    name_col = find_col(frame, ["板块名称", "名称"])
    change_col = find_col(frame, ["涨跌幅"])
    amount_col = find_col(frame, ["成交额"])
    turnover_col = find_col(frame, ["换手率"])
    up_col = find_col(frame, ["上涨家数"])
    down_col = find_col(frame, ["下跌家数"])
    if not name_col or not change_col:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "板块": frame[name_col].astype(str),
            "板块类型": source,
            "板块涨跌幅": frame[change_col].map(to_float),
            "成交额": frame[amount_col].map(to_float) if amount_col else np.nan,
            "换手率": frame[turnover_col].map(to_float) if turnover_col else np.nan,
            "上涨家数": frame[up_col].map(to_float) if up_col else np.nan,
            "下跌家数": frame[down_col].map(to_float) if down_col else np.nan,
        }
    )
    denom = out["上涨家数"] + out["下跌家数"]
    out["板块广度"] = np.where(denom > 0, out["上涨家数"] / denom, np.nan)
    return out.dropna(subset=["板块涨跌幅"])


def fetch_stock_quotes(ak) -> tuple[pd.DataFrame, list[str], str]:
    errors: list[str] = []
    frame, err = safe_call("stock_zh_a_spot_em", ak.stock_zh_a_spot_em, retries=2)
    if err:
        errors.append(err)
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        out = normalize_stock_quote_frame(frame, "东方财富A股行情")
        if not out.empty:
            return out, errors, "东方财富A股行情"

    frame, err = safe_call("stock_zh_a_spot", ak.stock_zh_a_spot, retries=1)
    if err:
        errors.append(err)
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        out = normalize_stock_quote_frame(frame, "新浪A股行情")
        if not out.empty:
            return out, errors, "新浪A股行情"

    return pd.DataFrame(), errors, ""


def normalize_stock_quote_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    code_col = find_col(frame, ["代码", "股票代码", "symbol"])
    name_col = find_col(frame, ["名称", "股票名称"])
    change_col = find_col(frame, ["涨跌幅", "涨幅"])
    amount_col = find_col(frame, ["成交额"])
    if not code_col or not change_col:
        return pd.DataFrame()
    out = pd.DataFrame(
        {
            "股票代码": frame[code_col].map(normalize_code),
            "股票名称": frame[name_col].astype(str) if name_col else "",
            "涨跌幅": frame[change_col].map(to_float),
            "成交额": frame[amount_col].map(to_float) if amount_col else np.nan,
            "行情源": source,
        }
    )
    return out.dropna(subset=["股票代码", "涨跌幅"]).drop_duplicates("股票代码")


def basket_scores(quotes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    quote_map = quotes.set_index("股票代码")
    for sector, members in SECTOR_BASKETS.items():
        matched = []
        for code, name in members:
            norm = normalize_code(code)
            if norm in quote_map.index:
                row = quote_map.loc[norm]
                matched.append(
                    {
                        "股票代码": norm,
                        "股票名称": name,
                        "涨跌幅": float(row["涨跌幅"]),
                        "成交额": float(row["成交额"]) if pd.notna(row["成交额"]) else np.nan,
                    }
                )
        if not matched:
            continue
        data = pd.DataFrame(matched)
        rows.append(
            {
                "板块": sector,
                "板块类型": "代表成分股篮子",
                "板块涨跌幅": float(data["涨跌幅"].mean()),
                "板块广度": float((data["涨跌幅"] > 0).mean()),
                "强势股占比": float((data["涨跌幅"] >= 3).mean()),
                "成交额": float(data["成交额"].sum(skipna=True)) if data["成交额"].notna().any() else np.nan,
                "样本数": len(data),
                "样本明细": "；".join(f"{r.股票名称}{r.涨跌幅:.2f}%" for r in data.itertuples()),
            }
        )
    return pd.DataFrame(rows)


def fetch_news(ak, limit: int, use_akshare_cls: bool = False) -> tuple[list[NewsItem], list[str]]:
    errors: list[str] = []
    items: list[NewsItem] = []
    try:
        response = requests.get("https://www.cls.cn", headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        response.raise_for_status()
        text = response.text
        titles = re.findall(r'<a[^>]+href="(/detail/\d+)"[^>]*>(.*?)</a>', text, flags=re.S)
        seen = set()
        for href, raw_title in titles:
            title = re.sub(r"<[^>]+>", "", raw_title)
            title = html.unescape(title).strip()
            if not title or title in seen:
                continue
            seen.add(title)
            items.append(NewsItem(title=title, content="", source="财联社", url=f"https://www.cls.cn{href}"))
            if len(items) >= limit:
                break
    except Exception as exc:  # noqa: BLE001
        errors.append(f"cls_homepage: {type(exc).__name__}: {exc}")

    if use_akshare_cls and len(items) < max(5, limit // 4):
        frame, err = safe_call("stock_info_global_cls", ak.stock_info_global_cls, retries=1)
        if err:
            errors.append(err)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            title_col = find_col(frame, ["标题"])
            content_col = find_col(frame, ["内容"])
            if title_col:
                for _, row in frame.head(limit).iterrows():
                    items.append(
                        NewsItem(
                            title=str(row[title_col]),
                            content=str(row[content_col]) if content_col else "",
                            source="财联社",
                        )
                    )
    return items, errors


def news_scores(items: list[NewsItem]) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    score_rows = []
    positive_words = ["利好", "增长", "突破", "上调", "涨价", "扩产", "景气", "订单", "创新高", "超预期"]
    negative_words = ["风险", "下调", "下降", "亏损", "减持", "制裁", "调查", "不及预期", "回落"]
    for sector, words in SECTOR_KEYWORDS.items():
        hits = []
        weighted = 0
        for item in items:
            text = f"{item.title} {item.content}"
            if any(word.lower() in text.lower() for word in words):
                sentiment = sum(text.count(w) for w in positive_words) - sum(text.count(w) for w in negative_words)
                hits.append((item, sentiment))
                weighted += sentiment
        score_rows.append({"板块": sector, "新闻条数": len(hits), "新闻分数": weighted})
        for item, sentiment in hits[:5]:
            detail_rows.append({"板块": sector, "来源": item.source, "标题": item.title, "新闻情绪": sentiment, "链接": item.url})
    return pd.DataFrame(score_rows), pd.DataFrame(detail_rows)


def attach_funds(frame: pd.DataFrame) -> pd.DataFrame:
    funds = []
    for sector in frame["板块"]:
        mapped = FUND_MAPPING.get(sector, [])
        funds.append("；".join(f"{code} {name}（{note}）" for code, name, note in mapped))
    frame = frame.copy()
    frame["对应基金候选"] = funds
    frame["是否有可执行基金"] = frame["对应基金候选"].astype(bool)
    return frame


def score_sectors(market: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
    frame = market.copy()
    if "样本数" not in frame:
        frame["样本数"] = np.nan
    if "强势股占比" not in frame:
        frame["强势股占比"] = np.nan
    frame = frame.merge(news, on="板块", how="left")
    frame["新闻条数"] = frame["新闻条数"].fillna(0)
    frame["新闻分数"] = frame["新闻分数"].fillna(0)
    frame["热度分数"] = (
        0.45 * zscore(frame["板块涨跌幅"].fillna(0))
        + 0.20 * zscore(frame["板块广度"].fillna(0))
        + 0.15 * zscore(np.log1p(frame["成交额"].fillna(0)))
        + 0.10 * zscore(frame["强势股占比"].fillna(0))
        + 0.10 * zscore(frame["新闻分数"].fillna(0) + 0.3 * frame["新闻条数"].fillna(0))
    )
    return attach_funds(frame.sort_values("热度分数", ascending=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="扫描当前市场热度最高板块，并映射到可买基金候选。")
    parser.add_argument("--top", type=int, default=20, help="输出前N个板块")
    parser.add_argument("--news-limit", type=int, default=80, help="财联社新闻读取条数")
    parser.add_argument("--include-board", action="store_true", help="尝试读取东方财富全市场行业/概念板块，可能较慢")
    parser.add_argument("--akshare-cls", action="store_true", help="尝试读取 akshare 财联社电报接口，可能较慢")
    parser.add_argument("--tag", default=None, help="输出文件标签")
    return parser.parse_args()


def main() -> None:
    import akshare as ak

    args = parse_args()
    tag = args.tag or datetime.now().strftime("%Y-%m-%d_%H%M%S")
    diagnostics = {"captured_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "errors": []}

    if args.include_board:
        boards, errors = fetch_board_frames(ak)
        diagnostics["errors"].extend(errors)
    else:
        boards = pd.DataFrame()
        diagnostics["errors"].append("skip_board: 默认跳过东方财富全市场板块接口；需要时加 --include-board")
    diagnostics["board_rows"] = int(len(boards))

    quotes, errors, quote_source = fetch_stock_quotes(ak)
    diagnostics["errors"].extend(errors)
    diagnostics["quote_source"] = quote_source
    diagnostics["quote_rows"] = int(len(quotes))

    baskets = basket_scores(quotes) if not quotes.empty else pd.DataFrame()
    diagnostics["basket_rows"] = int(len(baskets))

    market_frames = []
    if not boards.empty:
        market_frames.append(boards)
    if not baskets.empty:
        market_frames.append(baskets)
    if not market_frames:
        raise RuntimeError("未获取到可用板块或代表成分股行情，无法扫描板块热度。")
    market = pd.concat(market_frames, ignore_index=True)

    news_items, errors = fetch_news(ak, args.news_limit, use_akshare_cls=args.akshare_cls)
    diagnostics["errors"].extend(errors)
    diagnostics["news_items"] = len(news_items)
    news_score, news_detail = news_scores(news_items)
    ranked = score_sectors(market, news_score).head(args.top)

    summary_path = OUT / f"{tag}_hot_sectors_cn.csv"
    news_path = OUT / f"{tag}_news_matches_cn.csv"
    diagnostics_path = OUT / f"{tag}_diagnostics.json"
    report_path = OUT / f"{tag}_report_cn.md"

    ranked.to_csv(summary_path, index=False, encoding="utf-8-sig")
    news_detail.to_csv(news_path, index=False, encoding="utf-8-sig")
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    lines = [
        "# 市场热门板块扫描",
        "",
        f"- 生成时间：{diagnostics['captured_at']}",
        f"- 板块行情行数：{diagnostics['board_rows']}；代表篮子行数：{diagnostics['basket_rows']}；新闻条数：{diagnostics['news_items']}",
        "",
        "| 排名 | 板块 | 类型 | 涨跌幅 | 广度 | 热度分数 | 对应基金候选 |",
        "|---:|---|---|---:|---:|---:|---|",
    ]
    for i, row in enumerate(ranked.itertuples(index=False), start=1):
        lines.append(
            f"| {i} | {row.板块} | {row.板块类型} | {row.板块涨跌幅:.2f}% | "
            f"{row.板块广度 if pd.notna(row.板块广度) else 0:.0%} | {row.热度分数:.2f} | {row.对应基金候选 or '无'} |"
        )
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(summary_path.resolve())
    print(ranked[["板块", "板块类型", "板块涨跌幅", "板块广度", "热度分数", "对应基金候选"]].to_string(index=False))


if __name__ == "__main__":
    main()
