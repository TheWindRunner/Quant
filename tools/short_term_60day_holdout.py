from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists() and str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from tools.backtest_pcb_purchase_limit import (
    INITIAL_CAPITAL,
    PCB_DAILY_LIMIT,
    all_in_pcb_choice,
    simulate,
    setup_chinese_font,
)
from tools.optimize_rotation_with_pcb_limit import (
    benchmark_cache,
    choice_for_spec,
    evaluate_choice,
)
from tools.sector_level_rotation_5themes import load_sector_navs, max_drawdown
from tools.strict_locked_multihorizon_validation import (
    core_weight_for_spec,
    dca_curve,
    preregistered_specs,
    redemption_fee,
    sharpe_ratio,
    strategy_curve,
)


OUT = ROOT / "output" / "short_term_60day_holdout"
HOLDOUT_DAYS = 60
TEST_WINDOW_COUNT = 30
ONE_MONTH_DAYS = 21
TRAIN_ENDPOINT_SPACING = 10
TRAIN_LOOKBACK_DAYS = 252
MIN_SIGNAL_HISTORY = 120
MIN_TRAIN_WIN_RATE = 0.50
MIN_TRAIN_MEAN_EXCESS = 0.0
MIN_TRAIN_MEDIAN_EXCESS = 0.0


def boundaries(navs: pd.DataFrame) -> dict[str, int | str]:
    train_end = len(navs) - HOLDOUT_DAYS - 1
    test_endpoints = list(range(len(navs) - TEST_WINDOW_COUNT, len(navs)))
    earliest_test_start = test_endpoints[0] - ONE_MONTH_DAYS + 1
    return {
        "训练结束位置": train_end,
        "训练结束日期": navs.index[train_end].date().isoformat(),
        "封存区开始日期": navs.index[train_end + 1].date().isoformat(),
        "封存区结束日期": navs.index[-1].date().isoformat(),
        "最早测试窗口开始": navs.index[earliest_test_start].date().isoformat(),
        "最晚测试窗口结束": navs.index[-1].date().isoformat(),
        "训练与最早测试间隔交易日": earliest_test_start - train_end - 1,
    }


def training_endpoints(train_end: int) -> list[int]:
    first_end = max(
        MIN_SIGNAL_HISTORY + ONE_MONTH_DAYS - 1,
        train_end - TRAIN_LOOKBACK_DAYS + 1,
    )
    endpoints = list(range(first_end, train_end + 1, TRAIN_ENDPOINT_SPACING))
    if not endpoints or endpoints[-1] != train_end:
        endpoints.append(train_end)
    return sorted(set(endpoints))


def select_short_model(navs: pd.DataFrame, train_end: int):
    endpoints = training_endpoints(train_end)
    benchmark = benchmark_cache(navs, endpoints, ONE_MONTH_DAYS)
    rows = []
    for spec in preregistered_specs():
        choice = choice_for_spec(navs, spec)
        detail, summary = evaluate_choice(
            navs,
            choice,
            endpoints,
            ONE_MONTH_DAYS,
            benchmark,
            core_weight_for_spec(spec),
        )
        summary_values = list(summary.values())
        excess = detail.iloc[:, 7].astype(float)
        drawdown_improvement = detail.iloc[:, 8].astype(float)
        wins = detail.iloc[:, 9].astype(bool)
        win_rate = float(wins.mean())
        mean_excess = float(excess.mean())
        median_excess = float(excess.median())
        mean_dd_improvement = float(drawdown_improvement.mean())
        score = (
            4.0 * win_rate
            + 40.0 * mean_excess
            + 20.0 * median_excess
            + 5.0 * mean_dd_improvement
        )
        rows.append(
            {
                "模型": spec.name,
                "家族": spec.family,
                "参数": repr(spec.params),
                "训练窗口数": len(endpoints),
                "训练胜出数": int(wins.sum()),
                "训练胜率": win_rate,
                "训练平均收益率": float(summary_values[1]),
                "训练平均超额": mean_excess,
                "训练超额中位数": median_excess,
                "训练平均回撤改善": mean_dd_improvement,
                "通过训练准入": bool(
                    win_rate >= MIN_TRAIN_WIN_RATE
                    and mean_excess > MIN_TRAIN_MEAN_EXCESS
                    and median_excess > MIN_TRAIN_MEDIAN_EXCESS
                ),
                "短期训练评分": score,
            }
        )
    ranking = pd.DataFrame(rows).sort_values("短期训练评分", ascending=False).reset_index(drop=True)
    name = str(ranking.iloc[0]["模型"])
    selected = next(spec for spec in preregistered_specs() if spec.name == name)
    qualified = bool(ranking.iloc[0]["通过训练准入"])
    return selected, ranking, qualified


def test_short_model(navs: pd.DataFrame, spec):
    choice = choice_for_spec(navs, spec)
    endpoints = list(range(len(navs) - TEST_WINDOW_COUNT, len(navs)))
    rows = []
    for window_id, end_pos in enumerate(endpoints, start=1):
        start_pos = end_pos - ONE_MONTH_DAYS + 1
        test = navs.iloc[start_pos : end_pos + 1]
        test_choice = choice.reindex(test.index).ffill()
        rotation_curve, rotation_metrics = strategy_curve(test, test_choice, spec)
        all_in_ledger, all_in_metrics = simulate(
            test,
            all_in_pcb_choice(test.index),
            PCB_DAILY_LIMIT,
        )
        all_in_curve = all_in_ledger.iloc[:, 0] / INITIAL_CAPITAL
        daily_dca = dca_curve(test["PCB"])
        curves = {
            "短期训练锁定轮动": rotation_curve,
            "限购全仓PCB": all_in_curve,
            "每日定投PCB": daily_dca,
        }
        returns = {name: float(curve.iloc[-1] - 1.0) for name, curve in curves.items()}
        winner = max(returns, key=returns.get)
        rotation_values = list(rotation_metrics.values())
        all_in_values = list(all_in_metrics.values())
        for name, curve in curves.items():
            if name == "短期训练锁定轮动":
                total_return = float(rotation_values[0])
                mdd = float(rotation_values[1])
                fee = redemption_fee(rotation_metrics)
            elif name == "限购全仓PCB":
                total_return = float(all_in_values[0])
                mdd = float(all_in_values[1])
                fee = 0.0
            else:
                total_return = float(curve.iloc[-1] - 1.0)
                mdd = max_drawdown(curve)
                fee = 0.0
            rows.append(
                {
                    "窗口": window_id,
                    "开始日期": test.index[0].date().isoformat(),
                    "结束日期": test.index[-1].date().isoformat(),
                    "策略": name,
                    "收益率": total_return,
                    "最大回撤": mdd,
                    "夏普比率": sharpe_ratio(curve),
                    "赎回费占初始资金": fee,
                    "收益第一": name == winner,
                    "轮动相对全仓超额": returns["短期训练锁定轮动"] - returns["限购全仓PCB"],
                }
            )
    detail = pd.DataFrame(rows)
    summaries = []
    for strategy, group in detail.groupby("策略", sort=False):
        summaries.append(
            {
                "策略": strategy,
                "窗口数": len(group),
                "收益第一次数": int(group["收益第一"].sum()),
                "收益第一比例": float(group["收益第一"].mean()),
                "平均收益率": float(group["收益率"].mean()),
                "收益率方差": float(group["收益率"].var(ddof=1)),
                "平均最大回撤": float(group["最大回撤"].mean()),
                "平均夏普比率": float(group["夏普比率"].mean()),
                "平均赎回费": float(group["赎回费占初始资金"].mean()),
                "轮动跑赢全仓次数": (
                    int((group["轮动相对全仓超额"] > 0).sum())
                    if strategy == "短期训练锁定轮动"
                    else np.nan
                ),
                "轮动平均超额": (
                    float(group["轮动相对全仓超额"].mean())
                    if strategy == "短期训练锁定轮动"
                    else np.nan
                ),
            }
        )
    return choice, detail, pd.DataFrame(summaries)


def current_target(spec, choice: pd.Series, qualified: bool) -> str:
    signal = str(choice.iloc[-1])
    executed = str(choice.shift(2).fillna(choice.iloc[0]).iloc[-1])
    core_weight = core_weight_for_spec(spec)
    if core_weight is None:
        target = f"{executed} 100%"
    elif executed == "PCB":
        target = "PCB 100%"
    else:
        target = f"PCB {core_weight:.0%}，{executed} {1.0 - core_weight:.0%}"
    research_signal = f"最新研究信号：{signal}；考虑2个净值日延迟后的研究目标：{target}"
    if not qualified:
        return f"模型未通过训练准入，不据此换仓；{research_signal}"
    return research_signal.replace("研究", "执行")


def make_chart(detail: pd.DataFrame) -> None:
    setup_chinese_font()
    pivot = detail.pivot(index="窗口", columns="策略", values="收益率")
    excess = detail.loc[detail["策略"] == "短期训练锁定轮动"].set_index("窗口")[
        "轮动相对全仓超额"
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), dpi=160, sharex=True)
    pivot.plot(ax=axes[0], linewidth=2.0, marker="o", markersize=3)
    axes[0].set_title("最近60个交易日封存区：30个一月入场窗口收益")
    axes[0].set_ylabel("收益率")
    axes[0].grid(alpha=0.25)
    colors = np.where(excess >= 0, "#d62728", "#2ca02c")
    axes[1].bar(excess.index, excess * 100, color=colors)
    axes[1].axhline(0.0, color="gray", linewidth=1.0)
    axes[1].set_ylabel("轮动相对全仓超额（百分点）")
    axes[1].set_xlabel("连续入场窗口编号")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "短期60日封存30窗口对比_cn.png")
    plt.close(fig)


def make_report(bounds, spec, ranking, choice, summary, qualified: bool) -> None:
    lines = [
        "# 短期策略60交易日封存验证",
        "",
        "## 时间边界",
        "",
        f"- 训练数据截止：{bounds['训练结束日期']}",
        f"- 最近60个净值日封存区：{bounds['封存区开始日期']} 至 {bounds['封存区结束日期']}",
        f"- 30个一月测试窗口覆盖：{bounds['最早测试窗口开始']} 至 {bounds['最晚测试窗口结束']}",
        f"- 训练截止与最早测试窗口之间另有{bounds['训练与最早测试间隔交易日']}个交易日间隔。",
        "",
        "## 模型",
        "",
        f"- 锁定模型：`{spec.name}`",
        f"- 家族：{spec.family}",
        f"- 参数：`{spec.params}`",
        f"- 候选模型数：{len(ranking)}",
        "- 选模目标只使用一月窗口，不使用三月、六月测试结果。",
        f"- 训练准入：{'通过' if qualified else '未通过'}。门槛为训练胜率不低于50%，且平均超额与超额中位数均为正。",
        f"- {current_target(spec, choice, qualified)}",
        "",
        "## 测试结果",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "说明：30个连续窗口用于判断不同入场日的敏感性，窗口之间高度重叠，胜出次数不等于30个独立样本。模型参数只用封存区之前的数据确定，但每日信号仍由当日可见的最新净值因子计算。",
    ]
    (OUT / "短期60日封存策略报告_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    navs = load_sector_navs()
    bounds = boundaries(navs)
    selected, ranking, qualified = select_short_model(navs, int(bounds["训练结束位置"]))
    choice, detail, summary = test_short_model(navs, selected)
    pd.DataFrame([bounds]).to_csv(OUT / "训练与封存边界_cn.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(OUT / "短期训练模型排名_cn.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "短期封存30窗口明细_cn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "短期封存测试汇总_cn.csv", index=False, encoding="utf-8-sig")
    make_chart(detail)
    make_report(bounds, selected, ranking, choice, summary, qualified)
    print(f"锁定模型: {selected.name}")
    print(pd.DataFrame([bounds]).to_string(index=False))
    print(summary.to_string(index=False))
    print(current_target(selected, choice, qualified))


if __name__ == "__main__":
    main()
