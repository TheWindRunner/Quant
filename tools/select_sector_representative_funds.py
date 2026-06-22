from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "output" / "sector_fund_selector"
OUT.mkdir(parents=True, exist_ok=True)

PERIODS = ["近1周", "近1月", "近3月", "近6月"]

SECTOR_RULES = {
    "PCB": {
        "keywords": ["PCB", "印制电路", "电路板", "电子", "消费电子"],
        "known_codes": ["720001", "024481", "021523", "021528", "015876"],
    },
    "存储": {
        "keywords": ["存储", "半导体", "芯片", "集成电路", "人工智能"],
        "known_codes": ["018816", "008887", "006503", "006502"],
    },
    "CPO": {
        "keywords": ["通信", "光通信", "通信设备", "5G", "信息技术"],
        "known_codes": ["007817", "008326", "008327"],
    },
    "AI": {
        "keywords": ["人工智能", "AI", "智能", "数字", "计算机", "软件"],
        "known_codes": ["008585", "017811", "018816", "019829", "019830"],
    },
    "半导体设备": {
        "keywords": ["半导体", "芯片", "集成电路", "人工智能"],
        "known_codes": ["017811", "006503", "006502", "008887"],
    },
}


def fetch_rank() -> pd.DataFrame:
    import akshare as ak

    frames = []
    for fund_type in ["股票型", "混合型", "指数型"]:
        frame = ak.fund_open_fund_rank_em(symbol=fund_type)
        frame["基金类型"] = fund_type
        frames.append(frame)
    rank = pd.concat(frames, ignore_index=True)
    rank["基金代码"] = rank["基金代码"].astype(str).str.zfill(6)
    for col in PERIODS + ["日增长率", "近1年", "近2年", "近3年", "今年来", "成立来"]:
        if col in rank.columns:
            rank[col] = pd.to_numeric(rank[col], errors="coerce")
    rank = rank.sort_values(PERIODS, ascending=False, na_position="last")
    rank = rank.drop_duplicates("基金代码", keep="first").reset_index(drop=True)
    return rank


def add_global_percentiles(rank: pd.DataFrame) -> pd.DataFrame:
    result = rank.copy()
    for period in PERIODS:
        result[f"{period}全市场分位"] = result[period].rank(pct=True)
    percentile_cols = [f"{period}全市场分位" for period in PERIODS]
    result["四周期平均分位"] = result[percentile_cols].mean(axis=1)
    result["四周期最低分位"] = result[percentile_cols].min(axis=1)
    result["四周期均为正"] = (result[PERIODS] > 0).all(axis=1)
    result["多周期综合分"] = (
        0.35 * result["四周期平均分位"]
        + 0.35 * result["四周期最低分位"]
        + 0.15 * result["近3月"].rank(pct=True)
        + 0.15 * result["近6月"].rank(pct=True)
    )
    return result


def classify_candidates(rank: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector, rule in SECTOR_RULES.items():
        name_match = pd.Series(False, index=rank.index)
        for keyword in rule["keywords"]:
            name_match = name_match | rank["基金简称"].astype(str).str.contains(keyword, case=False, na=False)
        code_match = rank["基金代码"].isin(rule["known_codes"])
        selected = rank.loc[name_match | code_match].copy()
        if selected.empty:
            continue
        selected["板块"] = sector
        selected["入选原因"] = np.where(code_match.loc[selected.index], "已知候选/用户指定", "基金简称关键词")
        selected["是否核心代表候选"] = selected["基金代码"].isin(rule["known_codes"])
        selected = selected.sort_values(
            ["四周期均为正", "多周期综合分", "四周期最低分位"],
            ascending=[False, False, False],
        )
        selected["板块内排名"] = np.arange(1, len(selected) + 1)
        rows.append(selected)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def representative_table(candidates: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector, group in candidates.groupby("板块", sort=False):
        qualified = group.loc[group["四周期均为正"]].copy()
        if qualified.empty:
            qualified = group.copy()
        # Prefer user/known core candidates when their multi-period score is close enough.
        best_score = qualified["多周期综合分"].max()
        core = qualified.loc[
            qualified["是否核心代表候选"] & (qualified["多周期综合分"] >= best_score - 0.08)
        ]
        chosen = (core if not core.empty else qualified).sort_values(
            ["多周期综合分", "四周期最低分位"],
            ascending=False,
        ).iloc[0]
        rows.append(chosen)
    return pd.DataFrame(rows).sort_values("多周期综合分", ascending=False)


def write_report(candidates: pd.DataFrame, reps: pd.DataFrame) -> None:
    lines = [
        "# 板块代表基金自动筛选",
        "",
        "## 方法",
        "",
        "- 数据源：AKShare `fund_open_fund_rank_em`，对应东方财富开放式基金排行。",
        "- 周期：近1周、近1月、近3月、近6月。",
        "- 归类：基金简称关键词 + 已知候选基金代码。公开接口不能保证复刻养基宝的私有板块基金池。",
        "- 代表基金：优先选择四个周期都为正、全市场分位靠前、且属于已知核心候选的基金；如果核心候选明显落后，则选择板块内综合分最高基金。",
        "",
        "## 当前推荐代表",
        "",
        reps[
            ["板块", "基金代码", "基金简称", "基金类型", "日期", *PERIODS, "四周期平均分位", "四周期最低分位", "多周期综合分", "入选原因"]
        ].to_markdown(index=False),
        "",
        "## 使用限制",
        "",
        "- 名称关键词会漏掉持仓真实暴露很强、但名称不含主题词的主动基金，例如部分财通系 PCB 暴露基金。",
        "- 下一步应叠加基金半年报/年报/季报持仓，把重仓股映射到板块，作为第二层校验。",
        "- 该脚本解决的是“选代表基金池”，不是最终交易信号；交易信号仍应由板块轮动模型决定。",
    ]
    (OUT / "sector_fund_selector_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    rank = add_global_percentiles(fetch_rank())
    candidates = classify_candidates(rank)
    reps = representative_table(candidates)
    rank.to_csv(OUT / "all_equity_fund_rank_cn.csv", index=False, encoding="utf-8-sig")
    candidates.to_csv(OUT / "sector_fund_candidates_cn.csv", index=False, encoding="utf-8-sig")
    reps.to_csv(OUT / "sector_representative_funds_cn.csv", index=False, encoding="utf-8-sig")
    write_report(candidates, reps)
    print(reps[["板块", "基金代码", "基金简称", "基金类型", "日期", *PERIODS, "多周期综合分", "入选原因"]].to_string(index=False))


if __name__ == "__main__":
    main()
