from __future__ import annotations

import argparse
import contextlib
from datetime import datetime
import io
import json
from pathlib import Path
import re
import sys
import time
import traceback
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


OUT = ROOT / "data" / "intraday_estimates" / "holding_based"
OUT.mkdir(parents=True, exist_ok=True)

DEFAULT_FUNDS = ["006503", "024481", "021523", "021528"]
DEFAULT_MIN_COVERAGE_PCT = 70.0
DEFAULT_TARGET_COVERAGE_PCT = 100.0


def normalize_stock_code(value: object) -> str:
    text = str(value).strip().lower()
    digits = re.sub(r"\D", "", text)
    return digits[-6:].zfill(6) if digits else ""


def to_float(value: object) -> float | None:
    if pd.isna(value):
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    parsed = pd.to_numeric(text, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def parse_quarter(value: object) -> tuple[int, int]:
    text = str(value)
    match = re.search(r"(\d{4})年(\d)季度", text)
    if match:
        return int(match.group(1)), int(match.group(2))
    return 0, 0


def holding_source_label(year: int, quarter: int) -> str:
    if quarter == 2:
        return f"{year}半年报补足"
    if quarter == 4:
        return f"{year}年报补足"
    return f"{year}Q{quarter}季报"


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


def normalize_holding_frame(frame: pd.DataFrame, fund_code: str) -> tuple[pd.DataFrame, list[str]]:
    errors: list[str] = []
    data = frame.copy()
    required = {"股票代码", "股票名称", "占净值比例", "季度"}
    missing = required - set(data.columns)
    if missing:
        return pd.DataFrame(), [f"{fund_code} 持仓字段缺失: {sorted(missing)}"]
    data["报告年份"] = data["季度"].map(lambda item: parse_quarter(item)[0])
    data["报告季度"] = data["季度"].map(lambda item: parse_quarter(item)[1])
    data["股票代码标准化"] = data["股票代码"].map(normalize_stock_code)
    data["占净值比例"] = data["占净值比例"].map(to_float)
    data = data.dropna(subset=["占净值比例"])
    data = data.loc[data["占净值比例"] > 0].copy()
    if data.empty:
        errors.append(f"{fund_code} 持仓为空或占净值比例均为0")
    return data, errors


def fetch_holding_history(ak, fund_code: str, year: int | None = None) -> tuple[pd.DataFrame, list[str]]:
    current_year = year or datetime.now().year
    errors: list[str] = []
    frames = []
    for query_year in [current_year, current_year - 1]:
        frame, err = safe_call(
            f"fund_portfolio_hold_em({fund_code}, {query_year})",
            ak.fund_portfolio_hold_em,
            symbol=fund_code,
            date=str(query_year),
            retries=2,
        )
        if err:
            errors.append(err)
        if isinstance(frame, pd.DataFrame) and not frame.empty:
            normalized, frame_errors = normalize_holding_frame(frame, fund_code)
            errors.extend(frame_errors)
            if not normalized.empty:
                frames.append(normalized)
    return (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()), errors


def fetch_latest_holdings(
    ak,
    fund_code: str,
    year: int | None = None,
    target_coverage_pct: float = DEFAULT_TARGET_COVERAGE_PCT,
) -> tuple[pd.DataFrame, dict]:
    history, errors = fetch_holding_history(ak, fund_code, year)
    if history.empty:
        return pd.DataFrame(), {"fund_code": fund_code, "errors": errors}

    latest_key = history[["报告年份", "报告季度"]].drop_duplicates().sort_values(
        ["报告年份", "报告季度"], ascending=False
    ).iloc[0]
    latest = history[
        (history["报告年份"] == latest_key["报告年份"])
        & (history["报告季度"] == latest_key["报告季度"])
    ].copy()
    latest = latest.sort_values("占净值比例", ascending=False)
    latest["持仓来源"] = latest.apply(lambda row: holding_source_label(int(row["报告年份"]), int(row["报告季度"])), axis=1)
    latest["原始占净值比例"] = latest["占净值比例"]
    latest["补足使用占比"] = latest["占净值比例"]
    latest["是否补足项"] = "否"

    remaining = max(0.0, target_coverage_pct - float(latest["补足使用占比"].sum()))
    supplement_frames = []
    if remaining > 0:
        supplement = history[
            history["报告季度"].isin([2, 4])
            & (
                (history["报告年份"] < int(latest_key["报告年份"]))
                | (
                    (history["报告年份"] == int(latest_key["报告年份"]))
                    & (history["报告季度"] < int(latest_key["报告季度"]))
                )
            )
        ].copy()
        if not supplement.empty:
            supplement = supplement.sort_values(["报告年份", "报告季度", "占净值比例"], ascending=[False, False, False])
            supplement = supplement.drop_duplicates("股票代码标准化", keep="first")
            existing = set(latest["股票代码标准化"])
            supplement = supplement.loc[~supplement["股票代码标准化"].isin(existing)]
            used_rows = []
            for _, row in supplement.iterrows():
                if remaining <= 1e-9:
                    break
                use_weight = min(float(row["占净值比例"]), remaining)
                item = row.copy()
                item["持仓来源"] = holding_source_label(int(row["报告年份"]), int(row["报告季度"]))
                item["原始占净值比例"] = float(row["占净值比例"])
                item["补足使用占比"] = use_weight
                item["占净值比例"] = use_weight
                item["是否补足项"] = "是"
                used_rows.append(item)
                remaining -= use_weight
            if used_rows:
                supplement_frames.append(pd.DataFrame(used_rows))

    composite = pd.concat([latest, *supplement_frames], ignore_index=True) if supplement_frames else latest
    composite = composite.sort_values(["是否补足项", "占净值比例"], ascending=[True, False])
    meta = {
        "fund_code": fund_code,
        "report_year": int(latest_key["报告年份"]),
        "report_quarter": int(latest_key["报告季度"]),
        "holding_rows": int(len(composite)),
        "latest_holding_rows": int(len(latest)),
        "supplement_rows": int((composite["是否补足项"] == "是").sum()),
        "holding_weight_sum_pct": float(composite["占净值比例"].sum()),
        "target_coverage_pct": float(target_coverage_pct),
        "unfilled_weight_pct": float(max(0.0, target_coverage_pct - composite["占净值比例"].sum())),
        "errors": errors,
    }
    return composite, meta


def fetch_stock_quotes(ak) -> tuple[pd.DataFrame, dict]:
    diagnostics = {"quote_source": "", "errors": []}

    frame, err = safe_call("stock_zh_a_spot_em", ak.stock_zh_a_spot_em, retries=2)
    if err:
        diagnostics["errors"].append(err)
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        quotes = normalize_quote_frame(frame, source="东方财富")
        if not quotes.empty:
            diagnostics["quote_source"] = "东方财富 stock_zh_a_spot_em"
            return quotes, diagnostics

    frame, err = safe_call("stock_zh_a_spot", ak.stock_zh_a_spot, retries=1)
    if err:
        diagnostics["errors"].append(err)
    if isinstance(frame, pd.DataFrame) and not frame.empty:
        quotes = normalize_quote_frame(frame, source="新浪")
        if not quotes.empty:
            diagnostics["quote_source"] = "新浪 stock_zh_a_spot"
            return quotes, diagnostics

    return pd.DataFrame(), diagnostics


def normalize_quote_frame(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    data = frame.copy()
    code_col = find_col(data, ["代码", "股票代码", "symbol", "code"])
    name_col = find_col(data, ["名称", "股票名称", "name"])
    change_col = find_col(data, ["涨跌幅", "涨幅", "change_pct", "pct_chg"])
    price_col = find_col(data, ["最新价", "最新", "price", "现价"])
    time_col = find_col(data, ["时间戳", "时间", "time"])
    if not code_col or not change_col:
        return pd.DataFrame()
    quotes = pd.DataFrame(
        {
            "股票代码标准化": data[code_col].map(normalize_stock_code),
            "行情名称": data[name_col] if name_col else "",
            "最新价": data[price_col].map(to_float) if price_col else None,
            "当日涨跌幅": data[change_col].map(to_float),
            "行情时间": data[time_col] if time_col else "",
            "行情来源": source,
        }
    )
    return quotes.dropna(subset=["股票代码标准化", "当日涨跌幅"]).drop_duplicates("股票代码标准化")


def find_col(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    lower_map = {str(col).lower(): col for col in frame.columns}
    for item in candidates:
        if item in frame.columns:
            return item
        if item.lower() in lower_map:
            return lower_map[item.lower()]
    for col in frame.columns:
        text = str(col)
        if any(item in text for item in candidates):
            return col
    return None


def estimate_one_fund(
    ak,
    fund_code: str,
    quotes: pd.DataFrame,
    year: int | None,
    min_coverage_pct: float,
    target_coverage_pct: float,
) -> tuple[dict, pd.DataFrame]:
    holdings, meta = fetch_latest_holdings(ak, fund_code, year, target_coverage_pct)
    if holdings.empty:
        return {
            "基金代码": fund_code,
            "估算状态": "失败",
            "失败原因": "; ".join(meta.get("errors", [])) or "未获取到持仓",
        }, pd.DataFrame()

    merged = holdings.merge(quotes, on="股票代码标准化", how="left")
    merged["贡献涨跌幅"] = merged["占净值比例"] * merged["当日涨跌幅"] / 100
    matched = merged["当日涨跌幅"].notna()
    estimate_pct = float(merged.loc[matched, "贡献涨跌幅"].sum())
    weight_sum = float(merged["占净值比例"].sum())
    matched_weight = float(merged.loc[matched, "占净值比例"].sum())
    is_reliable = matched_weight >= min_coverage_pct
    confidence = confidence_label(weight_sum, matched_weight, min_coverage_pct)

    detail = merged[
        [
            "股票代码",
            "股票名称",
            "季度",
            "持仓来源",
            "是否补足项",
            "占净值比例",
            "原始占净值比例",
            "持股数",
            "持仓市值",
            "行情名称",
            "最新价",
            "当日涨跌幅",
            "贡献涨跌幅",
            "行情时间",
            "行情来源",
        ]
    ].copy()
    detail.insert(0, "基金代码", fund_code)

    summary = {
        "基金代码": fund_code,
        "估算状态": "成功",
        "报告期": f"{meta['report_year']}Q{meta['report_quarter']}",
        "持仓数量": int(len(holdings)),
        "最新季报持仓数量": int(meta.get("latest_holding_rows", 0)),
        "半年报年报补足数量": int(meta.get("supplement_rows", 0)),
        "匹配行情数量": int(matched.sum()),
        "披露持仓覆盖净值比例": weight_sum,
        "匹配行情覆盖净值比例": matched_weight,
        "目标补足覆盖比例": target_coverage_pct,
        "未补足比例": float(meta.get("unfilled_weight_pct", 0.0)),
        "估算当日涨跌幅": estimate_pct,
        "可信阈值": min_coverage_pct,
        "是否可信": "是" if is_reliable else "否",
        "置信度": confidence,
        "估算说明": "最新季报持仓优先；不足部分用最近半年报/年报股票明细补足。补足项可能已被基金经理调仓，只用于盘中估算近似。",
        "失败原因": "",
    }
    return summary, detail


def confidence_label(weight_sum: float, matched_weight: float, min_coverage_pct: float) -> str:
    if matched_weight >= min_coverage_pct and matched_weight / max(weight_sum, 1e-9) >= 0.9:
        return "高"
    if matched_weight >= min_coverage_pct and matched_weight / max(weight_sum, 1e-9) >= 0.75:
        return "中"
    return "低"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于基金上季度披露持仓与A股当日涨跌估算基金盘中涨跌。")
    parser.add_argument("--funds", default=",".join(DEFAULT_FUNDS), help="逗号分隔基金代码，例如 006503,024481")
    parser.add_argument("--year", type=int, default=None, help="查询持仓年份，默认当前年，不存在时回退上一年")
    parser.add_argument("--min-coverage", type=float, default=DEFAULT_MIN_COVERAGE_PCT, help="可信所需的匹配行情持仓覆盖净值比例，默认70")
    parser.add_argument("--target-coverage", type=float, default=DEFAULT_TARGET_COVERAGE_PCT, help="用半年报/年报补足到的目标持仓覆盖比例，默认100")
    parser.add_argument("--tag", default=None, help="输出文件标签，默认使用当前日期时间")
    return parser.parse_args()


def main() -> None:
    import akshare as ak

    args = parse_args()
    funds = [item.strip().zfill(6) for item in args.funds.split(",") if item.strip()]
    captured_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tag = args.tag or datetime.now().strftime("%Y-%m-%d_%H%M%S")

    quotes, quote_diag = fetch_stock_quotes(ak)
    summaries: list[dict] = []
    details: list[pd.DataFrame] = []

    if quotes.empty:
        summaries.append(
            {
                "基金代码": ",".join(funds),
                "估算状态": "失败",
                "失败原因": "未获取到A股行情: " + "; ".join(quote_diag.get("errors", [])),
            }
        )
    else:
        for fund_code in funds:
            summary, detail = estimate_one_fund(
                ak,
                fund_code,
                quotes,
                args.year,
                args.min_coverage,
                args.target_coverage,
            )
            summaries.append(summary)
            if not detail.empty:
                details.append(detail)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.insert(0, "抓取时间", captured_at)
    summary_frame.insert(1, "行情源", quote_diag.get("quote_source", ""))

    detail_frame = pd.concat(details, ignore_index=True) if details else pd.DataFrame()
    if not detail_frame.empty:
        detail_frame.insert(0, "抓取时间", captured_at)

    summary_path = OUT / f"{tag}_summary_cn.csv"
    detail_path = OUT / f"{tag}_detail_cn.csv"
    diagnostics_path = OUT / f"{tag}_diagnostics.json"
    summary_frame.to_csv(summary_path, index=False, encoding="utf-8-sig")
    detail_frame.to_csv(detail_path, index=False, encoding="utf-8-sig")
    diagnostics_path.write_text(
        json.dumps(
            {
                "captured_at": captured_at,
                "funds": funds,
                "akshare_version": getattr(ak, "__version__", "unknown"),
                "quote_diagnostics": quote_diag,
                "quote_shape": list(quotes.shape) if isinstance(quotes, pd.DataFrame) else None,
                "output_summary": str(summary_path),
                "output_detail": str(detail_path),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(summary_path.resolve())
    print(detail_path.resolve())
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
