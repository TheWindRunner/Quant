from __future__ import annotations

from itertools import combinations
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_fund_advisor.style_timing import read_cached_nav


OUT = ROOT / "output" / "hybrid_generalization_validation"
OUT.mkdir(parents=True, exist_ok=True)

COST = 0.0015
WINDOW_DAYS = 63

FUNDS = {
    "007817": {"name": "CPO/通信设备：国泰中证全指通信设备ETF联接A", "theme": "CPO", "group": "科技"},
    "008887": {"name": "存储/半导体：华夏国证半导体芯片ETF联接A", "theme": "存储", "group": "科技"},
    "008585": {"name": "AI：华夏中证人工智能主题ETF联接A", "theme": "AI", "group": "科技"},
    "015876": {"name": "PCB代理：富国中证消费电子主题ETF发起式联接A", "theme": "PCB", "group": "科技"},
    "720001": {"name": "PCB主动代理：财通价值动量混合", "theme": "PCB主动", "group": "科技"},
    "006503": {"name": "半导体主动：财通集成电路产业股票C", "theme": "半导体主动", "group": "科技"},
    "021528": {"name": "科技成长补位：财通成长优选混合C", "theme": "科技成长", "group": "科技"},
    "021523": {"name": "科技制造补位：财通价值动量混合C", "theme": "科技制造", "group": "科技"},
    "014942": {"name": "化工：鹏华中证细分化工产业主题ETF联接A", "theme": "化工", "group": "非科技"},
    "004432": {"name": "有色：南方中证申万有色金属ETF发起联接A", "theme": "有色", "group": "非科技"},
    "018034": {"name": "电力：国泰国证绿色电力ETF发起联接A", "theme": "电力", "group": "非科技"},
}

PAIR_SETS = [
    ("科技：CPO vs 存储", "007817", "008887"),
    ("科技：CPO vs AI", "007817", "008585"),
    ("科技：存储 vs AI", "008887", "008585"),
    ("科技：CPO vs PCB代理", "007817", "015876"),
    ("科技：006503 vs 021528", "006503", "021528"),
    ("非科技：化工 vs 有色", "014942", "004432"),
    ("非科技：化工 vs 电力", "014942", "018034"),
    ("跨风格：CPO vs 化工", "007817", "014942"),
    ("跨风格：存储 vs 有色", "008887", "004432"),
]

POOL_SETS = {
    "三科技池_CPO_存储_AI": ["007817", "008887", "008585"],
    "四科技池_CPO_存储_AI_PCB": ["007817", "008887", "008585", "015876"],
    "多科技池_含主动和补位": ["007817", "008887", "008585", "015876", "006503", "021528", "021523"],
    "非科技池_化工_有色_电力": ["014942", "004432", "018034"],
    "全混合池_科技加非科技": ["007817", "008887", "008585", "015876", "014942", "004432", "018034"],
}


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def load_navs(codes: list[str]) -> pd.DataFrame:
    return pd.DataFrame({code: read_cached_nav(code) for code in codes}).dropna().sort_index()


def confirm_choice(raw: pd.Series, confirm: int = 5) -> pd.Series:
    values = raw.to_numpy(dtype=object)
    current = values[0]
    pending = current
    count = 0
    output = []
    for value in values:
        if value == current:
            pending = value
            count = 0
        elif value == pending:
            count += 1
            if count >= confirm:
                current = value
                count = 0
        else:
            pending = value
            count = 1
        output.append(current)
    return pd.Series(output, index=raw.index)


def base_choice(navs: pd.DataFrame, anchor: str | None = None) -> pd.Series:
    anchor = anchor or navs.columns[0]
    momentum = navs.pct_change(2, fill_method=None).where(lambda frame: frame > 0)
    columns = np.array(momentum.columns)
    raw = []
    for row in momentum.to_numpy():
        mask = ~np.isnan(row)
        raw.append(columns[mask][np.argmax(row[mask])] if mask.any() else anchor)
    return confirm_choice(pd.Series(raw, index=navs.index), confirm=5)


def symmetric_hybrid_choice(navs: pd.DataFrame, anchor: str | None = None) -> pd.Series:
    choice = base_choice(navs, anchor=anchor)
    ret30 = navs.pct_change(30, fill_method=None)
    dd20 = navs / navs.rolling(20).max() - 1
    above_ma20 = navs > navs.rolling(20).mean()
    regime = (ret30 > 0.12) & (dd20 > -0.03) & above_ma20
    confirmed = regime.copy()
    for code in navs.columns:
        streak = regime[code].astype(int).groupby((regime[code] != regime[code].shift()).cumsum()).cumsum()
        confirmed[code] = streak >= 2
    forced = []
    for date in navs.index:
        active = confirmed.loc[date]
        if bool(active.any()):
            forced.append(ret30.loc[date].where(active).idxmax())
        else:
            forced.append(choice.loc[date])
    return pd.Series(forced, index=navs.index)


def returns_from_choice(navs: pd.DataFrame, choice: pd.Series) -> pd.Series:
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    position = choice.shift(2).fillna(choice.iloc[0]).to_numpy(dtype=object)
    strategy_return = np.zeros(len(position))
    for code in navs.columns:
        strategy_return += (position == code) * returns[code].to_numpy()
    turnover_cost = (choice != choice.shift()).astype(float).shift(1).fillna(0.0).to_numpy() * COST
    return pd.Series(strategy_return - turnover_cost, index=navs.index)


def evaluate_windows(navs: pd.DataFrame, strategy_returns: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    if len(navs) < WINDOW_DAYS + 30:
        return pd.DataFrame()
    for window_id, end_pos in enumerate(range(len(navs) - 30, len(navs)), start=1):
        start_pos = end_pos - WINDOW_DAYS + 1
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        curve = (1 + strategy_returns.loc[start:end]).cumprod()
        strategy_return = float(curve.iloc[-1] - 1)
        all_in = {
            code: float(navs.loc[start:end, code].iloc[-1] / navs.loc[start:end, code].iloc[0] - 1)
            for code in navs.columns
        }
        equal_curve = (navs.loc[start:end] / navs.loc[start]).mean(axis=1)
        best_code = max(all_in, key=all_in.get)
        rows.append(
            {
                "策略组": label,
                "窗口": window_id,
                "开始": start.date(),
                "结束": end.date(),
                "策略收益": strategy_return,
                "策略最大回撤": max_drawdown(curve),
                "等权持有收益": float(equal_curve.iloc[-1] - 1),
                "等权持有最大回撤": max_drawdown(equal_curve),
                "最佳全仓代码": best_code,
                "最佳全仓名称": FUNDS.get(best_code, {}).get("name", best_code),
                "最佳全仓收益": all_in[best_code],
                "最佳全仓最大回撤": max_drawdown(navs.loc[start:end, best_code] / navs.loc[start, best_code]),
                "跑赢等权持有": strategy_return > float(equal_curve.iloc[-1] - 1),
                "跑赢最佳全仓": strategy_return > all_in[best_code],
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame, extra: dict | None = None) -> dict:
    extra = extra or {}
    return {
        **extra,
        "窗口数": len(frame),
        "跑赢等权次数": int(frame["跑赢等权持有"].sum()),
        "跑赢最佳全仓次数": int(frame["跑赢最佳全仓"].sum()),
        "平均策略收益": float(frame["策略收益"].mean()),
        "策略收益标准差": float(frame["策略收益"].std()),
        "平均策略最大回撤": float(frame["策略最大回撤"].mean()),
        "最差策略最大回撤": float(frame["策略最大回撤"].min()),
        "平均最佳全仓收益": float(frame["最佳全仓收益"].mean()),
        "平均等权收益": float(frame["等权持有收益"].mean()),
        "最新目标": extra.get("最新目标", ""),
    }


def validate_pairs() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    details = []
    for name, code_a, code_b in PAIR_SETS:
        navs = load_navs([code_a, code_b])
        choice = symmetric_hybrid_choice(navs, anchor=code_b)
        returns = returns_from_choice(navs, choice)
        frame = evaluate_windows(navs, returns, name)
        if frame.empty:
            continue
        frame["代码A"] = code_a
        frame["代码B"] = code_b
        frame["名称A"] = FUNDS[code_a]["name"]
        frame["名称B"] = FUNDS[code_b]["name"]
        details.append(frame)
        summaries.append(
            summarize(
                frame,
                {
                    "类型": "两两轮动",
                    "策略组": name,
                    "基金池": f"{code_a},{code_b}",
                    "样本开始": navs.index[0].date(),
                    "样本结束": navs.index[-1].date(),
                    "最新目标": choice.iloc[-1],
                    "最新目标名称": FUNDS[choice.iloc[-1]]["name"],
                },
            )
        )
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def validate_pools() -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    details = []
    for name, codes in POOL_SETS.items():
        navs = load_navs(codes)
        choice = symmetric_hybrid_choice(navs, anchor=codes[0])
        returns = returns_from_choice(navs, choice)
        frame = evaluate_windows(navs, returns, name)
        if frame.empty:
            continue
        frame["基金池"] = ",".join(codes)
        details.append(frame)
        summaries.append(
            summarize(
                frame,
                {
                    "类型": "多基金池",
                    "策略组": name,
                    "基金池": ",".join(codes),
                    "样本开始": navs.index[0].date(),
                    "样本结束": navs.index[-1].date(),
                    "最新目标": choice.iloc[-1],
                    "最新目标名称": FUNDS[choice.iloc[-1]]["name"],
                },
            )
        )
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def verdict(row: pd.Series) -> str:
    win_equal = row["跑赢等权次数"] >= 15
    win_best = row["跑赢最佳全仓次数"] >= 15
    if win_equal and win_best:
        return "强成立"
    if win_equal:
        return "弱成立：能胜等权，但不能稳定胜最佳单基金"
    return "不成立"


def main() -> None:
    pair_summary, pair_detail = validate_pairs()
    pool_summary, pool_detail = validate_pools()
    summary = pd.concat([pair_summary, pool_summary], ignore_index=True)
    summary["结论"] = summary.apply(verdict, axis=1)
    summary = summary.sort_values(["类型", "跑赢最佳全仓次数", "跑赢等权次数", "平均策略收益"], ascending=[True, False, False, False])

    summary.to_csv(OUT / "generalization_summary_cn.csv", index=False, encoding="utf-8-sig")
    pair_detail.to_csv(OUT / "pair_rotation_windows_cn.csv", index=False, encoding="utf-8-sig")
    pool_detail.to_csv(OUT / "pool_rotation_windows_cn.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 混合策略泛化验证",
        "",
        "固定参数：2日动量Top1、正收益过滤、5日确认、2日执行延迟、0.15%切换成本。",
        "强趋势覆盖层改为对称版本：任一基金满足30日涨幅>12%、20日回撤>-3%、站上20日均线并连续2日成立，则持有其中30日涨幅最强者。",
        "",
        "判定口径：最近30个三个月滑动窗口；15/30以上视为达到一半门槛。",
        "",
        "| 类型 | 策略组 | 跑赢等权 | 跑赢最佳全仓 | 平均收益 | 平均回撤 | 最新目标 | 结论 |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['类型']} | {row['策略组']} | {int(row['跑赢等权次数'])}/30 | "
            f"{int(row['跑赢最佳全仓次数'])}/30 | {pct(row['平均策略收益'])} | "
            f"{pct(row['平均策略最大回撤'])} | {row['最新目标']} | {row['结论']} |"
        )
    (OUT / "generalization_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
