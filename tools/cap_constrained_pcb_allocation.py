from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_fund_advisor.style_timing import read_cached_nav


OUT = Path("output/cap_constrained_pcb_allocation")
OUT.mkdir(parents=True, exist_ok=True)

COST = 0.0015
WINDOW_DAYS = 63
ANCHOR = "011370"

FUNDS = {
    "006503": "财通集成电路产业股票C",
    "024481": "财通品质甄选混合",
    "021523": "财通价值动量混合C",
    "021528": "财通成长优选混合C",
    "720001": "财通价值动量混合",
    "011370": "华商均衡成长混合C",
}

CAPPED_FUNDS = ["006503", "024481", "021523", "021528"]
VALIDATION_POOL = ["006503", "024481", "021523", "021528", "720001", "011370"]
CAP_PER_FUND = 1000
TOTAL_CAPITAL = 10000


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def confirm_choice(raw: pd.Series, confirm: int) -> pd.Series:
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


def build_base_choice(navs: pd.DataFrame) -> pd.Series:
    momentum = navs.pct_change(2, fill_method=None).where(lambda frame: frame > 0)
    columns = np.array(momentum.columns)
    raw = []
    for row in momentum.to_numpy():
        mask = ~np.isnan(row)
        raw.append(columns[mask][np.argmax(row[mask])] if mask.any() else ANCHOR)
    return confirm_choice(pd.Series(raw, index=navs.index), confirm=5)


def build_hybrid_choice(
    navs: pd.DataFrame,
    base_choice: pd.Series,
    lookback: int = 30,
    momentum_threshold: float = 0.12,
    drawdown_threshold: float = -0.03,
    regime_confirm: int = 2,
) -> pd.Series:
    core = navs["006503"]
    regime = (
        (core.pct_change(lookback, fill_method=None) > momentum_threshold)
        & (core / core.rolling(20).max() - 1 > drawdown_threshold)
        & (core > core.rolling(20).mean())
    ).fillna(False)

    streak = 0
    force_core = []
    for flag in regime.to_numpy():
        streak = streak + 1 if flag else 0
        force_core.append(streak >= regime_confirm)

    choice = base_choice.copy()
    choice.loc[pd.Series(force_core, index=navs.index)] = "006503"
    return choice


def returns_from_choice(navs: pd.DataFrame, choice: pd.Series) -> pd.Series:
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    position = choice.shift(2).fillna(ANCHOR).to_numpy(dtype=object)
    strategy_return = np.zeros(len(position))
    for code in navs.columns:
        strategy_return += (position == code) * returns[code].to_numpy()
    turnover_cost = (choice != choice.shift()).astype(float).shift(1).fillna(0.0).to_numpy() * COST
    return pd.Series(strategy_return - turnover_cost, index=navs.index)


def window_curve(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (1 + series.loc[start:end]).cumprod()


def evaluate_windows(navs: pd.DataFrame, strategy_return: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    for window_id, end_pos in enumerate(range(len(navs) - 30, len(navs)), start=1):
        start_pos = end_pos - WINDOW_DAYS + 1
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        curve = window_curve(strategy_return, start, end)
        all_in = {
            code: float(navs.loc[start:end, code].iloc[-1] / navs.loc[start:end, code].iloc[0] - 1)
            for code in navs.columns
        }
        best_code = max(all_in, key=all_in.get)
        rows.append(
            {
                "策略": label,
                "窗口": window_id,
                "开始": start.date(),
                "结束": end.date(),
                "收益率": float(curve.iloc[-1] - 1),
                "最大回撤": max_drawdown(curve),
                "006503全仓收益": all_in["006503"],
                "最佳全仓代码": best_code,
                "最佳全仓收益": all_in[best_code],
                "是否跑赢006503": float(curve.iloc[-1] - 1) > all_in["006503"],
                "是否跑赢最佳全仓": float(curve.iloc[-1] - 1) > all_in[best_code],
            }
        )
    return pd.DataFrame(rows)


def summarize_windows(frame: pd.DataFrame) -> dict:
    return {
        "策略": frame["策略"].iloc[0],
        "窗口数": len(frame),
        "跑赢006503次数": int(frame["是否跑赢006503"].sum()),
        "跑赢最佳全仓次数": int(frame["是否跑赢最佳全仓"].sum()),
        "平均收益率": float(frame["收益率"].mean()),
        "收益率标准差": float(frame["收益率"].std()),
        "平均最大回撤": float(frame["最大回撤"].mean()),
        "最差最大回撤": float(frame["最大回撤"].min()),
        "最差窗口收益": float(frame["收益率"].min()),
    }


def latest_strength_snapshot(navs: pd.DataFrame, funds: list[str]) -> pd.DataFrame:
    latest = navs.index[-1]
    snap = pd.DataFrame(index=funds)
    snap["基金名称"] = [FUNDS[code] for code in funds]
    snap["最新日期"] = latest.date()
    snap["2日涨幅"] = navs[funds].pct_change(2, fill_method=None).iloc[-1]
    snap["20日涨幅"] = navs[funds].pct_change(20, fill_method=None).iloc[-1]
    snap["60日涨幅"] = navs[funds].pct_change(60, fill_method=None).iloc[-1]
    snap["相对20日均线"] = navs[funds].iloc[-1] / navs[funds].rolling(20).mean().iloc[-1] - 1
    score = (
        0.45 * snap["2日涨幅"].fillna(0.0)
        + 0.35 * snap["20日涨幅"].fillna(0.0)
        + 0.20 * snap["60日涨幅"].fillna(0.0)
    )
    score += 0.02 * (snap["相对20日均线"].fillna(0.0) > 0).astype(float)
    snap["最新强度分数"] = score
    snap = snap.reset_index(names="基金代码").sort_values("最新强度分数", ascending=False)
    return snap


def build_purchase_plan(rank_frame: pd.DataFrame, target_code: str) -> pd.DataFrame:
    remaining = TOTAL_CAPITAL
    ranked = rank_frame["基金代码"].tolist()
    ordered = [target_code] + [code for code in ranked if code != target_code]
    days = []
    day = 1
    while remaining > 0:
        day_rows = []
        for code in ordered:
            if remaining <= 0:
                break
            amount = min(CAP_PER_FUND, remaining)
            day_rows.append(
                {
                    "交易日序号": day,
                    "基金代码": code,
                    "基金名称": FUNDS[code],
                    "买入金额": amount,
                }
            )
            remaining -= amount
        days.extend(day_rows)
        day += 1
    plan = pd.DataFrame(days)
    plan["累计买入"] = plan["买入金额"].cumsum()
    plan["说明"] = ""
    if not plan.empty:
        plan.loc[plan["交易日序号"] == 1, "说明"] = "首日先买当前策略主目标，再按替代强弱补位。"
        plan.loc[plan["交易日序号"] > 1, "说明"] = "次日继续按同一顺序补足，前提是14:30信号未转空。"
    return plan


def main() -> None:
    navs = pd.DataFrame({code: read_cached_nav(code) for code in VALIDATION_POOL}).dropna().sort_index()
    base_choice = build_base_choice(navs)
    hybrid_choice = build_hybrid_choice(navs, base_choice)
    base_returns = returns_from_choice(navs, base_choice)
    hybrid_returns = returns_from_choice(navs, hybrid_choice)

    base_windows = evaluate_windows(navs, base_returns, "扩展池基础短周期Top1")
    hybrid_windows = evaluate_windows(navs, hybrid_returns, "扩展池混合策略")
    summary = pd.DataFrame([summarize_windows(base_windows), summarize_windows(hybrid_windows)])

    latest_signal = pd.DataFrame(
        [
            {
                "最新日期": navs.index[-1].date(),
                "基础策略目标": base_choice.iloc[-1],
                "混合策略目标": hybrid_choice.iloc[-1],
                "混合策略目标名称": FUNDS[hybrid_choice.iloc[-1]],
                "共同起始日期": navs.index[0].date(),
                "共同样本交易日数": len(navs),
            }
        ]
    )

    capped_navs = pd.DataFrame({code: read_cached_nav(code) for code in CAPPED_FUNDS + [ANCHOR]}).dropna().sort_index()
    capped_strength = latest_strength_snapshot(capped_navs, CAPPED_FUNDS)
    purchase_plan = build_purchase_plan(capped_strength, target_code="006503")

    same_day_capacity = CAP_PER_FUND * len(CAPPED_FUNDS)
    deployment_note = pd.DataFrame(
        [
            {
                "总资金": TOTAL_CAPITAL,
                "单基金单日限购": CAP_PER_FUND,
                "纳入限购基金数": len(CAPPED_FUNDS),
                "单日最多可买入": same_day_capacity,
                "一次性当天无法打满": TOTAL_CAPITAL > same_day_capacity,
                "至少需要交易日数": int(np.ceil(TOTAL_CAPITAL / same_day_capacity)),
            }
        ]
    )

    base_windows.to_csv(OUT / "expanded_pool_base_windows_cn.csv", index=False, encoding="utf-8-sig")
    hybrid_windows.to_csv(OUT / "expanded_pool_hybrid_windows_cn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "expanded_pool_summary_cn.csv", index=False, encoding="utf-8-sig")
    latest_signal.to_csv(OUT / "expanded_pool_latest_signal_cn.csv", index=False, encoding="utf-8-sig")
    capped_strength.to_csv(OUT / "capped_funds_strength_cn.csv", index=False, encoding="utf-8-sig")
    purchase_plan.to_csv(OUT / "purchase_plan_10000_cn.csv", index=False, encoding="utf-8-sig")
    deployment_note.to_csv(OUT / "deployment_constraint_cn.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 限购约束下的PCB替代基金校验",
        "",
        "## 口径",
        "",
        "- 策略校验池：006503 / 024481 / 021523 / 021528 / 720001 / 011370。",
        "- 混合策略结构不变：2日动量Top1 + 正收益过滤 + 5日确认；上层保留 006503 强趋势覆盖。",
        "- 30个三个月滑动窗口，比较策略、006503全仓、窗口内最佳全仓。",
        "- 实际执行口径单独处理限购：006503 / 024481 / 021523 / 021528 默认每日各限购1000。",
        "",
        "## 结果摘要",
        "",
        "| 策略 | 跑赢006503次数 | 跑赢最佳全仓次数 | 平均收益率 | 平均最大回撤 | 最差窗口收益 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['策略']} | {int(row['跑赢006503次数'])}/{int(row['窗口数'])} | "
            f"{int(row['跑赢最佳全仓次数'])}/{int(row['窗口数'])} | {pct(row['平均收益率'])} | "
            f"{pct(row['平均最大回撤'])} | {pct(row['最差窗口收益'])} |"
        )
    lines.extend(
        [
            "",
            "## 当前信号",
            "",
            f"- 最新共同净值日期：`{latest_signal.loc[0, '最新日期']}`",
            f"- 基础策略目标：`{latest_signal.loc[0, '基础策略目标']}`",
            f"- 混合策略目标：`{latest_signal.loc[0, '混合策略目标']}`（{latest_signal.loc[0, '混合策略目标名称']}）",
            "",
            "## 10000元执行约束",
            "",
            f"- 4只限购基金单日合计最多只能买入：`{same_day_capacity}` 元。",
            f"- 若总资金是 `10000` 元，严格只买这4只，至少需要 `"
            f"{int(np.ceil(TOTAL_CAPITAL / same_day_capacity))}` 个交易日。",
            "- 因此，当天无法一次性把10000元全部打进这4只基金，这是约束本身决定的，不是策略问题。",
            "",
            "## 建议",
            "",
            "- 如果你坚持只买这4只，就按 `purchase_plan_10000_cn.csv` 分3个交易日建仓。",
            "- 如果后续 14:30 信号转弱，第二天起应停止继续补仓，剩余现金保留，不要机械补满。",
            "- 024481 历史较短，只适合作为补位，不适合当作长期主锚。",
        ]
    )
    (OUT / "report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
