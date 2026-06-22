from __future__ import annotations

from datetime import datetime
from pathlib import Path
import json
import traceback

import pandas as pd


OUT = Path("data/intraday_estimates")
OUT.mkdir(parents=True, exist_ok=True)

FUNDS = [
    {
        "fund_code": "006503",
        "fund_name": "财通集成电路产业股票C",
        "theme": "半导体主动/集成电路",
        "linked_etf_code": "159995",
        "linked_etf_name": "芯片ETF",
    },
    {
        "fund_code": "011370",
        "fund_name": "华商均衡成长混合C",
        "theme": "科技成长锚",
        "linked_etf_code": "",
        "linked_etf_name": "",
    },
    {
        "fund_code": "007817",
        "fund_name": "国泰中证全指通信设备ETF联接A",
        "theme": "CPO/通信设备",
        "linked_etf_code": "515880",
        "linked_etf_name": "通信ETF",
    },
    {
        "fund_code": "008887",
        "fund_name": "华夏国证半导体芯片ETF联接A",
        "theme": "存储/半导体代理",
        "linked_etf_code": "159995",
        "linked_etf_name": "芯片ETF",
    },
    {
        "fund_code": "008585",
        "fund_name": "华夏中证人工智能主题ETF联接A",
        "theme": "AI/人工智能",
        "linked_etf_code": "515070",
        "linked_etf_name": "人工智能ETF",
    },
    {
        "fund_code": "720001",
        "fund_name": "财通价值动量混合",
        "theme": "PCB/制造成长代理",
        "linked_etf_code": "",
        "linked_etf_name": "",
    },
]


def safe_call(label: str, func, *args, **kwargs):
    try:
        return func(*args, **kwargs), ""
    except Exception as exc:  # noqa: BLE001 - snapshot script must not fail whole capture.
        return None, f"{label}: {type(exc).__name__}: {exc}"


def normalize_code(value: str) -> str:
    return str(value).strip().zfill(6)


def numeric_value(value):
    if pd.isna(value):
        return None
    text = str(value).strip().replace("%", "").replace(",", "")
    parsed = pd.to_numeric(text, errors="coerce")
    return None if pd.isna(parsed) else float(parsed)


def find_change_column(frame: pd.DataFrame) -> str | None:
    candidates = ["涨跌幅", "涨跌幅%", "change", "change_pct", "涨幅", "最新涨跌幅"]
    for col in candidates:
        if col in frame.columns:
            return col
    for col in frame.columns:
        if "涨跌幅" in str(col) or "涨幅" in str(col):
            return col
    return None


def main() -> None:
    import akshare as ak

    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    stamp = now.strftime("%Y-%m-%d %H:%M:%S")
    errors: list[str] = []

    fund_est, err = safe_call("fund_value_estimation_em", ak.fund_value_estimation_em)
    if err:
        errors.append(err)
        fund_est = pd.DataFrame()

    etf_spot, err = safe_call("fund_etf_spot_em", ak.fund_etf_spot_em)
    if err:
        errors.append(err)
        etf_spot = pd.DataFrame()

    rows = []
    for item in FUNDS:
        row = {
            "decision_date": date_str,
            "captured_at": stamp,
            "fund_code": item["fund_code"],
            "fund_name": item["fund_name"],
            "theme": item["theme"],
            "linked_etf_code": item["linked_etf_code"],
            "linked_etf_name": item["linked_etf_name"],
            "fund_estimate_available": False,
            "fund_estimate_change_pct": None,
            "fund_estimate_time": None,
            "fund_estimate_source_date": None,
            "etf_spot_available": False,
            "etf_change_pct": None,
            "etf_latest_price": None,
            "etf_amount": None,
            "data_notes": "",
        }

        notes = []
        if not fund_est.empty:
            code_cols = [col for col in fund_est.columns if "基金代码" in str(col) or str(col) in {"code", "基金代码"}]
            code_col = code_cols[0] if code_cols else None
            if code_col:
                matched = fund_est[fund_est[code_col].astype(str).map(normalize_code) == item["fund_code"]]
                if not matched.empty:
                    m = matched.iloc[0]
                    row["fund_estimate_available"] = True
                    for col in fund_est.columns:
                        if "估算涨幅" in str(col) or "估算增长率" in str(col) or "估算涨跌幅" in str(col):
                            row["fund_estimate_change_pct"] = numeric_value(m[col])
                            row["fund_estimate_source_date"] = str(col).split("-估算数据")[0]
                        if "估算时间" in str(col) or str(col) == "估算时间":
                            row["fund_estimate_time"] = m[col]
                else:
                    notes.append("fund_estimation_no_match")
            else:
                notes.append("fund_estimation_code_column_missing")
        else:
            notes.append("fund_estimation_unavailable")

        if item["linked_etf_code"] and not etf_spot.empty:
            code_cols = [col for col in etf_spot.columns if "代码" in str(col) or str(col).lower() in {"code", "symbol"}]
            code_col = code_cols[0] if code_cols else None
            if code_col:
                matched = etf_spot[etf_spot[code_col].astype(str).map(normalize_code) == item["linked_etf_code"]]
                if not matched.empty:
                    m = matched.iloc[0]
                    row["etf_spot_available"] = True
                    ch_col = find_change_column(etf_spot)
                    if ch_col:
                        row["etf_change_pct"] = numeric_value(m[ch_col])
                    for col in etf_spot.columns:
                        if str(col) in {"最新价", "最新", "price", "现价"}:
                            row["etf_latest_price"] = numeric_value(m[col])
                        if "成交额" in str(col) or str(col).lower() in {"amount"}:
                            row["etf_amount"] = numeric_value(m[col])
                else:
                    notes.append("etf_spot_no_match")
            else:
                notes.append("etf_spot_code_column_missing")
        elif item["linked_etf_code"]:
            notes.append("etf_spot_unavailable")
        else:
            notes.append("no_linked_etf")

        row["data_notes"] = ";".join(notes)
        rows.append(row)

    frame = pd.DataFrame(rows)
    out_csv = OUT / f"{date_str}_1430_auto_snapshot.csv"
    frame.to_csv(out_csv, index=False, encoding="utf-8-sig")

    diagnostics = {
        "captured_at": stamp,
        "akshare_version": getattr(ak, "__version__", "unknown"),
        "fund_estimation_shape": list(fund_est.shape) if isinstance(fund_est, pd.DataFrame) else None,
        "fund_estimation_columns": list(map(str, fund_est.columns)) if isinstance(fund_est, pd.DataFrame) else [],
        "etf_spot_shape": list(etf_spot.shape) if isinstance(etf_spot, pd.DataFrame) else None,
        "etf_spot_columns": list(map(str, etf_spot.columns)) if isinstance(etf_spot, pd.DataFrame) else [],
        "errors": errors,
    }
    (OUT / f"{date_str}_1430_auto_diagnostics.json").write_text(
        json.dumps(diagnostics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(out_csv.resolve())
    print(frame.to_string(index=False))
    if errors:
        print("ERRORS")
        print("\n".join(errors))


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
