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
    simulate_weighted_targets,
    setup_chinese_font,
)
from tools.optimize_rotation_with_pcb_limit import benchmark_cache, evaluate_choice
from tools.optimize_sector_rotation_with_c_fee import ModelSpec, top1_choice
from tools.sector_level_rotation_5themes import load_sector_navs, max_drawdown
from tools.short_term_60day_holdout import (
    ONE_MONTH_DAYS,
    TEST_WINDOW_COUNT,
    boundaries,
    dca_curve,
    redemption_fee,
    sharpe_ratio,
    training_endpoints,
)


OUT = ROOT / "output" / "short_core_satellite_60day"


def candidate_specs() -> list[ModelSpec]:
    specs = []
    for momentum_days in [2, 3, 5]:
        for confirm_days in [3, 5, 7]:
            for min_hold_days in [7, 14, 30]:
                for advantage in [0.30, 0.40, 0.50]:
                    for core_weight in [0.80, 0.90, 0.95]:
                        specs.append(
                            ModelSpec(
                                name=(
                                    f"short_core_m{momentum_days}_c{confirm_days}_h{min_hold_days}_"
                                    f"a{advantage:.2f}_core{core_weight:.2f}"
                                ),
                                family="PCB短期核心卫星",
                                params={
                                    "momentum_days": momentum_days,
                                    "confirm_days": confirm_days,
                                    "min_hold_days": min_hold_days,
                                    "advantage_threshold": advantage,
                                    "positive_threshold": 0.0,
                                    "overlay": 1,
                                    "overlay_ret_days": 30,
                                    "overlay_ret_threshold": 0.12,
                                    "overlay_dd_days": 20,
                                    "overlay_dd_threshold": -0.03,
                                    "overlay_ma_days": 20,
                                    "overlay_confirm": 2,
                                    "advantage_window": 30,
                                    "stop_window": 5,
                                    "stop_loss": -0.08,
                                    "pcb_core_weight": core_weight,
                                },
                            )
                        )
    return specs


def target_weights(navs: pd.DataFrame, choice: pd.Series, core_weight: float) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    weights["PCB"] = core_weight
    for date, sector in choice.reindex(navs.index).ffill().items():
        weights.loc[date, str(sector)] += 1.0 - core_weight
    return weights


def detail_stats(detail: pd.DataFrame) -> dict[str, float]:
    excess = detail.iloc[:, 7].astype(float)
    dd_improvement = detail.iloc[:, 8].astype(float)
    wins = detail.iloc[:, 9].astype(bool)
    return {
        "wins": float(wins.sum()),
        "win_rate": float(wins.mean()),
        "mean_excess": float(excess.mean()),
        "median_excess": float(excess.median()),
        "dd_improvement": float(dd_improvement.mean()),
    }


def rank_training_models(navs: pd.DataFrame, train_end: int):
    endpoints = training_endpoints(train_end)
    split = len(endpoints) // 2
    early_endpoints = endpoints[:split]
    late_endpoints = endpoints[split:]
    benchmark = benchmark_cache(navs, endpoints, ONE_MONTH_DAYS)
    rows = []
    specs = candidate_specs()
    choice_cache: dict[tuple[int, int, int, float], pd.Series] = {}
    for number, spec in enumerate(specs, start=1):
        p = spec.params
        signal_key = (
            int(p["momentum_days"]),
            int(p["confirm_days"]),
            int(p["min_hold_days"]),
            float(p["advantage_threshold"]),
        )
        if signal_key not in choice_cache:
            proxy = ModelSpec(spec.name, "短动量Top1", p)
            choice_cache[signal_key] = top1_choice(navs, proxy)
        choice = choice_cache[signal_key]
        core_weight = float(p["pcb_core_weight"])
        full_detail, _ = evaluate_choice(
            navs, choice, endpoints, ONE_MONTH_DAYS, benchmark, core_weight
        )
        early = full_detail.iloc[:split]
        late = full_detail.iloc[split:]
        full_stats = detail_stats(full_detail)
        early_stats = detail_stats(early)
        late_stats = detail_stats(late)
        worst_half_win_rate = min(early_stats["win_rate"], late_stats["win_rate"])
        worst_half_excess = min(early_stats["mean_excess"], late_stats["mean_excess"])
        score = (
            4.0 * worst_half_win_rate
            + 2.0 * full_stats["win_rate"]
            + 40.0 * full_stats["mean_excess"]
            + 20.0 * full_stats["median_excess"]
            + 20.0 * worst_half_excess
            + 5.0 * full_stats["dd_improvement"]
        )
        rows.append(
            {
                "模型": spec.name,
                "参数": repr(spec.params),
                "PCB核心权重": core_weight,
                "训练窗口数": len(endpoints),
                "训练胜出数": int(full_stats["wins"]),
                "训练胜率": full_stats["win_rate"],
                "训练平均超额": full_stats["mean_excess"],
                "训练超额中位数": full_stats["median_excess"],
                "训练回撤改善": full_stats["dd_improvement"],
                "前半段胜率": early_stats["win_rate"],
                "后半段胜率": late_stats["win_rate"],
                "最差半段胜率": worst_half_win_rate,
                "最差半段平均超额": worst_half_excess,
                "训练评分": score,
            }
        )
        if number % 50 == 0:
            print(f"已训练 {number}/{len(specs)}", flush=True)
    ranking = pd.DataFrame(rows).sort_values("训练评分", ascending=False).reset_index(drop=True)
    selected_name = str(ranking.iloc[0]["模型"])
    selected = next(spec for spec in specs if spec.name == selected_name)
    selected_choice = choice_cache[
        (
            int(selected.params["momentum_days"]),
            int(selected.params["confirm_days"]),
            int(selected.params["min_hold_days"]),
            float(selected.params["advantage_threshold"]),
        )
    ]
    return selected, selected_choice, ranking


def evaluate_locked_test(navs: pd.DataFrame, spec: ModelSpec, choice: pd.Series):
    endpoints = list(range(len(navs) - TEST_WINDOW_COUNT, len(navs)))
    rows = []
    core_weight = float(spec.params["pcb_core_weight"])
    for window_id, end_pos in enumerate(endpoints, start=1):
        start_pos = end_pos - ONE_MONTH_DAYS + 1
        test = navs.iloc[start_pos : end_pos + 1]
        test_choice = choice.reindex(test.index).ffill()
        rotation_ledger, rotation_metrics = simulate_weighted_targets(
            test,
            target_weights(test, test_choice, core_weight),
        )
        all_in_ledger, all_in_metrics = simulate(
            test, all_in_pcb_choice(test.index), PCB_DAILY_LIMIT
        )
        rotation_curve = rotation_ledger.iloc[:, 0] / INITIAL_CAPITAL
        all_in_curve = all_in_ledger.iloc[:, 0] / INITIAL_CAPITAL
        dca = dca_curve(test["PCB"])
        curves = {
            "训练锁定核心卫星": rotation_curve,
            "限购全仓PCB": all_in_curve,
            "每日定投PCB": dca,
        }
        returns = {name: float(curve.iloc[-1] - 1.0) for name, curve in curves.items()}
        winner = max(returns, key=returns.get)
        for name, curve in curves.items():
            if name == "训练锁定核心卫星":
                values = list(rotation_metrics.values())
                total_return = float(values[0])
                mdd = float(values[1])
                fee = redemption_fee(rotation_metrics)
            elif name == "限购全仓PCB":
                values = list(all_in_metrics.values())
                total_return = float(values[0])
                mdd = float(values[1])
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
                    "核心卫星相对全仓超额": returns["训练锁定核心卫星"] - returns["限购全仓PCB"],
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
                "核心卫星跑赢全仓次数": (
                    int((group["核心卫星相对全仓超额"] > 0).sum())
                    if strategy == "训练锁定核心卫星"
                    else np.nan
                ),
                "核心卫星平均超额": (
                    float(group["核心卫星相对全仓超额"].mean())
                    if strategy == "训练锁定核心卫星"
                    else np.nan
                ),
            }
        )
    return detail, pd.DataFrame(summaries)


def make_chart(detail: pd.DataFrame) -> None:
    setup_chinese_font()
    pivot = detail.pivot(index="窗口", columns="策略", values="收益率")
    excess = detail.loc[detail["策略"] == "训练锁定核心卫星"].set_index("窗口")[
        "核心卫星相对全仓超额"
    ]
    fig, axes = plt.subplots(2, 1, figsize=(13, 9), dpi=160, sharex=True)
    pivot.plot(ax=axes[0], linewidth=2.0, marker="o", markersize=3)
    axes[0].set_title("60日隔离：训练锁定PCB核心卫星模型的一月测试")
    axes[0].set_ylabel("收益率")
    axes[0].grid(alpha=0.25)
    axes[1].bar(excess.index, excess * 100, color=np.where(excess >= 0, "#d62728", "#2ca02c"))
    axes[1].axhline(0.0, color="gray", linewidth=1.0)
    axes[1].set_xlabel("连续入场窗口编号")
    axes[1].set_ylabel("相对全仓PCB超额（百分点）")
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "短期核心卫星30窗口测试_cn.png")
    plt.close(fig)


def make_report(bounds, spec, ranking, summary) -> None:
    strategy_row = summary.loc[summary["策略"] == "训练锁定核心卫星"].iloc[0]
    passed = int(strategy_row["核心卫星跑赢全仓次数"]) >= 16
    lines = [
        "# 短期PCB核心卫星60日隔离验证",
        "",
        "## 数据边界",
        "",
        f"- 训练截止：{bounds['训练结束日期']}",
        f"- 60日封存区：{bounds['封存区开始日期']} 至 {bounds['封存区结束日期']}",
        f"- 测试窗口：{bounds['最早测试窗口开始']} 至 {bounds['最晚测试窗口结束']}",
        "- 243个核心卫星候选只在训练区排名；测试区只评估训练第一名。",
        "",
        "## 锁定模型",
        "",
        f"- 模型：`{spec.name}`",
        f"- 参数：`{spec.params}`",
        f"- 测试是否达到过半目标：{'是' if passed else '否'}",
        "",
        "## 测试汇总",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "执行口径：C类赎回按FIFO计算，持有不足7日1.5%、7至30日0.5%、满30日0%；基金转换当日卖旧买新；只有主动回到现金才执行T+2可用；PCB四个申购通道各限1000元，合计每日上限4000元；信号延迟2个净值日执行。",
    ]
    (OUT / "短期核心卫星60日隔离报告_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    navs = load_sector_navs()
    bounds = boundaries(navs)
    selected, choice, ranking = rank_training_models(navs, int(bounds["训练结束位置"]))
    detail, summary = evaluate_locked_test(navs, selected, choice)
    pd.DataFrame([bounds]).to_csv(OUT / "训练测试边界_cn.csv", index=False, encoding="utf-8-sig")
    ranking.to_csv(OUT / "核心卫星训练排名_cn.csv", index=False, encoding="utf-8-sig")
    detail.to_csv(OUT / "核心卫星测试30窗口明细_cn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "核心卫星测试汇总_cn.csv", index=False, encoding="utf-8-sig")
    make_chart(detail)
    make_report(bounds, selected, ranking, summary)
    print(f"训练锁定模型: {selected.name}")
    print(ranking.head(10).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
