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
    simulate_gradual_to_pcb,
    simulate_weighted_targets,
    setup_chinese_font,
)
from tools.optimize_rotation_with_pcb_limit import (
    benchmark_cache,
    choice_for_spec,
    evaluate_choice,
    optimization_specs,
)
from tools.sector_level_rotation_5themes import load_sector_navs, max_drawdown


OUT = ROOT / "output" / "strict_locked_multihorizon"
TEST_ENTRY_COUNT = 30
EMBARGO_DAYS = 60
MONTH_DAYS = {1: 21, 3: 63, 6: 126}
TRAIN_ENDPOINT_SPACING = 21
MIN_SIGNAL_HISTORY = 120


def preregistered_specs():
    """Small, fixed candidate set to limit multiple-testing selection bias."""
    selected = []
    for spec in optimization_specs():
        p = spec.params
        if spec.name.startswith("pcb_core_sat_"):
            if (
                int(p["momentum_days"]) in {2, 5}
                and int(p["confirm_days"]) == 5
                and int(p["min_hold_days"]) == 14
                and float(p["advantage_threshold"]) in {0.4, 0.5}
                and float(p["pcb_core_weight"]) in {0.6, 0.8}
            ):
                selected.append(spec)
        elif spec.name.startswith("cap_top1_"):
            if (
                int(p["momentum_days"]) in {2, 5}
                and int(p["confirm_days"]) == 5
                and int(p["min_hold_days"]) in {14, 30}
                and float(p["advantage_threshold"]) in {0.4, 0.5}
            ):
                selected.append(spec)
        elif spec.name.startswith("pcb_core_l"):
            if (
                int(p["lookback"]) in {5, 10}
                and int(p["confirm_days"]) == 5
                and int(p["min_hold_days"]) == 14
                and float(p["entry_advantage"]) in {0.02, 0.05}
                and float(p["exit_margin"]) == 0.02
                and float(p["pcb_lock_threshold"]) == 0.12
            ):
                selected.append(spec)
    if not selected:
        raise ValueError("预注册候选模型集合为空")
    return selected


def core_weight_for_spec(spec) -> float | None:
    value = spec.params.get("pcb_core_weight")
    return float(value) if value is not None else None


def split_boundaries(navs: pd.DataFrame) -> dict[str, int | str]:
    longest = max(MONTH_DAYS.values())
    first_test_start = len(navs) - longest - TEST_ENTRY_COUNT + 1
    last_test_start = first_test_start + TEST_ENTRY_COUNT - 1
    train_end = first_test_start - EMBARGO_DAYS - 1
    if train_end <= MIN_SIGNAL_HISTORY:
        raise ValueError("共同净值历史不足以建立60日隔离的六个月封存测试")
    return {
        "训练开始位置": 0,
        "训练结束位置": train_end,
        "隔离开始位置": train_end + 1,
        "隔离结束位置": first_test_start - 1,
        "首个测试入场位置": first_test_start,
        "末个测试入场位置": last_test_start,
        "训练结束日期": navs.index[train_end].date().isoformat(),
        "隔离开始日期": navs.index[train_end + 1].date().isoformat(),
        "隔离结束日期": navs.index[first_test_start - 1].date().isoformat(),
        "首个测试入场日期": navs.index[first_test_start].date().isoformat(),
        "末个测试入场日期": navs.index[last_test_start].date().isoformat(),
        "测试结束日期": navs.index[-1].date().isoformat(),
    }


def training_endpoints(train_end: int, window_days: int) -> list[int]:
    first_end = MIN_SIGNAL_HISTORY + window_days - 1
    endpoints = list(range(first_end, train_end + 1, TRAIN_ENDPOINT_SPACING))
    if not endpoints or endpoints[-1] != train_end:
        endpoints.append(train_end)
    return sorted(set(endpoints))


def summary_values(summary: dict[str, float]) -> dict[str, float]:
    values = list(summary.values())
    return {
        "wins": float(values[0]),
        "mean_return": float(values[1]),
        "mean_excess": float(values[2]),
        "mean_mdd": float(values[3]),
        "mean_dd_improvement": float(values[4]),
        "mean_fee": float(values[6]),
    }


def training_score(horizon_rows: list[dict[str, float]]) -> float:
    win_rates = np.array([row["win_rate"] for row in horizon_rows])
    excess = np.array([row["mean_excess"] for row in horizon_rows])
    dd_improvement = np.array([row["mean_dd_improvement"] for row in horizon_rows])
    # Each horizon receives equal weight; the worst horizon prevents a model from
    # winning merely by matching the originally requested one- and three-month targets.
    return float(
        4.0 * win_rates.min()
        + 2.0 * win_rates.mean()
        + 20.0 * excess.mean()
        + 10.0 * excess.min()
        + 5.0 * dd_improvement.mean()
    )


def select_model(navs: pd.DataFrame, train_end: int):
    specs = preregistered_specs()
    endpoint_map = {
        months: training_endpoints(train_end, days)
        for months, days in MONTH_DAYS.items()
    }
    benchmark_map = {
        months: benchmark_cache(navs, endpoints, MONTH_DAYS[months])
        for months, endpoints in endpoint_map.items()
    }
    rows = []
    for number, spec in enumerate(specs, start=1):
        choice = choice_for_spec(navs, spec)
        horizon_rows = []
        output = {"模型": spec.name, "家族": spec.family, "参数": repr(spec.params)}
        for months, days in MONTH_DAYS.items():
            _, raw_summary = evaluate_choice(
                navs,
                choice,
                endpoint_map[months],
                days,
                benchmark_map[months],
                core_weight_for_spec(spec),
            )
            values = summary_values(raw_summary)
            values["win_rate"] = values["wins"] / len(endpoint_map[months])
            horizon_rows.append(values)
            output[f"训练{months}月窗口数"] = len(endpoint_map[months])
            output[f"训练{months}月胜率"] = values["win_rate"]
            output[f"训练{months}月平均超额"] = values["mean_excess"]
            output[f"训练{months}月回撤改善"] = values["mean_dd_improvement"]
        output["训练综合评分"] = training_score(horizon_rows)
        rows.append(output)
        if number % 100 == 0:
            print(f"已完成训练候选 {number}/{len(specs)}", flush=True)
    ranking = pd.DataFrame(rows).sort_values("训练综合评分", ascending=False).reset_index(drop=True)
    selected_name = str(ranking.iloc[0]["模型"])
    selected = next(spec for spec in specs if spec.name == selected_name)
    return selected, ranking


def sharpe_ratio(curve: pd.Series) -> float:
    returns = curve.pct_change(fill_method=None).dropna()
    if len(returns) < 2 or float(returns.std(ddof=1)) <= 1e-12:
        return np.nan
    return float(np.sqrt(252.0) * returns.mean() / returns.std(ddof=1))


def dca_curve(pcb_nav: pd.Series) -> pd.Series:
    cash = INITIAL_CAPITAL
    units = 0.0
    daily_budget = min(PCB_DAILY_LIMIT, INITIAL_CAPITAL / len(pcb_nav))
    values = []
    for price in pcb_nav.astype(float):
        spend = min(cash, daily_budget, PCB_DAILY_LIMIT)
        units += spend / price
        cash -= spend
        values.append(cash + units * price)
    return pd.Series(values, index=pcb_nav.index) / INITIAL_CAPITAL


def strategy_curve(test: pd.DataFrame, choice: pd.Series, spec) -> tuple[pd.Series, dict[str, float]]:
    core_weight = core_weight_for_spec(spec)
    if core_weight is None:
        ledger, metrics = simulate_gradual_to_pcb(test, choice)
    else:
        weights = pd.DataFrame(0.0, index=test.index, columns=test.columns)
        weights["PCB"] = core_weight
        for date, sector in choice.reindex(test.index).ffill().items():
            weights.loc[date, str(sector)] += 1.0 - core_weight
        ledger, metrics = simulate_weighted_targets(test, weights)
    return ledger.iloc[:, 0] / INITIAL_CAPITAL, metrics


def redemption_fee(metrics: dict[str, float]) -> float:
    for key, value in metrics.items():
        if "赎回费" in str(key):
            return float(value)
    return 0.0


def locked_test(navs: pd.DataFrame, spec, bounds: dict[str, int | str]):
    choice = choice_for_spec(navs, spec)
    first_start = int(bounds["首个测试入场位置"])
    starts = list(range(first_start, first_start + TEST_ENTRY_COUNT))
    rows = []
    for months, days in MONTH_DAYS.items():
        for window_id, start_pos in enumerate(starts, start=1):
            test = navs.iloc[start_pos : start_pos + days]
            test_choice = choice.reindex(test.index).ffill()
            rotation_curve, rotation_metrics = strategy_curve(test, test_choice, spec)
            all_in_ledger, all_in_metrics = simulate(
                test, all_in_pcb_choice(test.index), PCB_DAILY_LIMIT
            )
            all_in_curve = all_in_ledger.iloc[:, 0] / INITIAL_CAPITAL
            daily_dca = dca_curve(test["PCB"])
            curves = {
                "训练锁定轮动": rotation_curve,
                "限购全仓PCB": all_in_curve,
                "每日定投PCB": daily_dca,
            }
            metrics_by_strategy = {
                "训练锁定轮动": list(rotation_metrics.values()),
                "限购全仓PCB": list(all_in_metrics.values()),
            }
            returns = {name: float(curve.iloc[-1] - 1.0) for name, curve in curves.items()}
            winner = max(returns, key=returns.get)
            for name, curve in curves.items():
                if name in metrics_by_strategy:
                    values = metrics_by_strategy[name]
                    total_return = float(values[0])
                    mdd = float(values[1])
                    fee = redemption_fee(rotation_metrics) if name == "训练锁定轮动" else 0.0
                else:
                    total_return = float(curve.iloc[-1] - 1.0)
                    mdd = max_drawdown(curve)
                    fee = 0.0
                rows.append(
                    {
                        "周期月数": months,
                        "窗口": window_id,
                        "开始日期": test.index[0].date().isoformat(),
                        "结束日期": test.index[-1].date().isoformat(),
                        "策略": name,
                        "收益率": total_return,
                        "最大回撤": mdd,
                        "夏普比率": sharpe_ratio(curve),
                        "赎回费占初始资金": fee,
                        "本窗口收益第一": name == winner,
                        "轮动相对全仓超额": returns["训练锁定轮动"] - returns["限购全仓PCB"],
                    }
                )
    detail = pd.DataFrame(rows)
    summaries = []
    for (months, strategy), group in detail.groupby(["周期月数", "策略"], sort=True):
        summaries.append(
            {
                "周期月数": months,
                "策略": strategy,
                "窗口数": len(group),
                "收益第一次数": int(group["本窗口收益第一"].sum()),
                "收益第一比例": float(group["本窗口收益第一"].mean()),
                "平均收益率": float(group["收益率"].mean()),
                "收益率方差": float(group["收益率"].var(ddof=1)),
                "平均最大回撤": float(group["最大回撤"].mean()),
                "平均夏普比率": float(group["夏普比率"].mean()),
                "平均赎回费": float(group["赎回费占初始资金"].mean()),
                "轮动跑赢全仓次数": (
                    int((group["轮动相对全仓超额"] > 0).sum())
                    if strategy == "训练锁定轮动"
                    else np.nan
                ),
                "轮动平均超额": (
                    float(group["轮动相对全仓超额"].mean())
                    if strategy == "训练锁定轮动"
                    else np.nan
                ),
            }
        )
    return detail, pd.DataFrame(summaries)


def make_chart(summary: pd.DataFrame) -> None:
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=160)
    fields = ["平均收益率", "平均最大回撤", "平均夏普比率"]
    titles = ["平均收益率", "平均最大回撤", "平均夏普比率"]
    for ax, field, title in zip(axes, fields, titles):
        pivot = summary.pivot(index="周期月数", columns="策略", values=field)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("持有周期（月）")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("60交易日隔离：同一批30个入场日的封存测试")
    fig.tight_layout()
    fig.savefig(OUT / "一三六个月严格封存测试对比_cn.png")
    plt.close(fig)


def make_report(bounds, spec, ranking, summary) -> None:
    lines = [
        "# 一、三、六个月严格封存验证报告",
        "",
        "## 数据隔离",
        "",
        f"- 训练结束：{bounds['训练结束日期']}",
        f"- 60交易日隔离带：{bounds['隔离开始日期']} 至 {bounds['隔离结束日期']}",
        f"- 30个连续测试入场日：{bounds['首个测试入场日期']} 至 {bounds['末个测试入场日期']}",
        f"- 最长六个月测试结束：{bounds['测试结束日期']}",
        "- 三种周期使用完全相同的30个入场日；任何测试结果均未参与选模。",
        "",
        "## 训练锁定模型",
        "",
        f"- 模型：`{spec.name}`",
        f"- 家族：{spec.family}",
        f"- 参数：`{spec.params}`",
        f"- 训练候选数：{len(ranking)}",
        "- 候选集合在打开本次测试结果前固定，并从687组压缩到少量代表规则。",
        "- 训练评分同时等权考虑1、3、6个月，并惩罚最弱周期。",
        "",
        "## 封存测试结果",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "说明：30个连续入场日用于观察起点敏感性，因窗口高度重叠，不应解释为30个独立统计样本。历史数据此前已被研究过程多次查看，因此本结果属于严格时间顺序的伪样本外验证，而不是真正从未见过的数据。真正封存验证需要使用未来新增净值或此前未使用的代理基金。",
    ]
    (OUT / "一三六个月严格封存验证报告_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    navs = load_sector_navs()
    bounds = split_boundaries(navs)
    pd.DataFrame([bounds]).to_csv(OUT / "训练隔离测试边界_cn.csv", index=False, encoding="utf-8-sig")
    selected, ranking = select_model(navs, int(bounds["训练结束位置"]))
    ranking.to_csv(OUT / "仅训练区模型排名_cn.csv", index=False, encoding="utf-8-sig")
    detail, summary = locked_test(navs, selected, bounds)
    detail.to_csv(OUT / "封存测试30入场日明细_cn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "封存测试汇总_cn.csv", index=False, encoding="utf-8-sig")
    make_chart(summary)
    make_report(bounds, selected, ranking, summary)
    print(f"锁定模型: {selected.name}")
    print(pd.DataFrame([bounds]).to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
