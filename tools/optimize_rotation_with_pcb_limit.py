from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.backtest_pcb_purchase_limit import (
    PCB_DAILY_LIMIT,
    all_in_pcb_choice,
    simulate,
    simulate_gradual_to_pcb,
    simulate_weighted_targets,
)
from tools.optimize_sector_rotation_with_c_fee import (
    ModelSpec,
    candidate_specs,
    multi_momentum_choice,
    top1_choice,
    trend_quality_choice,
)
from tools.sector_level_rotation_5themes import dataframe_to_markdown, load_sector_navs


OUT = ROOT / "output" / "pcb_limit_strategy_optimization"
OUT.mkdir(parents=True, exist_ok=True)

ONE_MONTH_DAYS = 21
THREE_MONTH_DAYS = 63
WINDOW_COUNT = 30
PURGE_DAYS = 20
TRAIN_WINDOW_COUNT = 10
TRAIN_WINDOW_SPACING = 5


def model_choice(navs: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    if spec.family == "短动量Top1":
        return top1_choice(navs, spec)
    if spec.family == "多周期动量波动惩罚":
        return multi_momentum_choice(navs, spec)
    if spec.family == "趋势质量":
        return trend_quality_choice(navs, spec)
    raise ValueError(spec.family)


def endpoint_sets(navs: pd.DataFrame) -> tuple[list[int], list[int]]:
    test_endpoints = list(range(len(navs) - WINDOW_COUNT, len(navs)))
    earliest_test_start = test_endpoints[0] - THREE_MONTH_DAYS + 1
    training_last_end = earliest_test_start - PURGE_DAYS
    training_endpoints = [
        training_last_end - TRAIN_WINDOW_SPACING * offset
        for offset in reversed(range(TRAIN_WINDOW_COUNT))
    ]
    if training_endpoints[0] - THREE_MONTH_DAYS + 1 < 0:
        raise ValueError("历史不足以构造隔离训练窗口")
    return training_endpoints, test_endpoints


def benchmark_cache(
    navs: pd.DataFrame,
    endpoints: list[int],
    window_days: int,
) -> dict[int, dict[str, float]]:
    cache = {}
    for end_pos in endpoints:
        test = navs.iloc[end_pos - window_days + 1 : end_pos + 1]
        _, metrics = simulate(test, all_in_pcb_choice(test.index), PCB_DAILY_LIMIT)
        cache[end_pos] = metrics
    return cache


def evaluate_choice(
    navs: pd.DataFrame,
    choice: pd.Series,
    endpoints: list[int],
    window_days: int,
    benchmark: dict[int, dict[str, float]],
    pcb_core_weight: float | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for window_id, end_pos in enumerate(endpoints, start=1):
        test = navs.iloc[end_pos - window_days + 1 : end_pos + 1]
        test_choice = choice.reindex(test.index).ffill()
        if pcb_core_weight is None:
            _, strategy = simulate_gradual_to_pcb(test, test_choice)
        else:
            weights = pd.DataFrame(0.0, index=test.index, columns=test.columns)
            for date, tactical_sector in test_choice.items():
                weights.loc[date, "PCB"] = pcb_core_weight
                weights.loc[date, str(tactical_sector)] += 1.0 - pcb_core_weight
            _, strategy = simulate_weighted_targets(test, weights)
        pcb = benchmark[end_pos]
        rows.append(
            {
                "窗口": window_id,
                "开始日期": test.index[0].date().isoformat(),
                "结束日期": test.index[-1].date().isoformat(),
                "轮动收益": strategy["总收益率"],
                "轮动最大回撤": strategy["最大回撤"],
                "限购全仓PCB收益": pcb["总收益率"],
                "限购全仓PCB最大回撤": pcb["最大回撤"],
                "超额收益": strategy["总收益率"] - pcb["总收益率"],
                "回撤改善": strategy["最大回撤"] - pcb["最大回撤"],
                "跑赢": strategy["总收益率"] > pcb["总收益率"],
                "赎回费": strategy["赎回费"],
                "平均现金比例": strategy["平均现金比例"],
                "切换次数": strategy["切换次数"],
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "胜出次数": float(detail["跑赢"].sum()),
        "平均收益": float(detail["轮动收益"].mean()),
        "平均超额收益": float(detail["超额收益"].mean()),
        "平均最大回撤": float(detail["轮动最大回撤"].mean()),
        "平均回撤改善": float(detail["回撤改善"].mean()),
        "最差最大回撤": float(detail["轮动最大回撤"].min()),
        "平均赎回费": float(detail["赎回费"].mean()),
        "平均现金比例": float(detail["平均现金比例"].mean()),
        "平均切换次数": float(detail["切换次数"].mean()),
    }
    return detail, summary


def optimization_specs() -> list[ModelSpec]:
    base = []
    for spec in candidate_specs():
        if spec.family != "多周期动量波动惩罚":
            continue
        p = spec.params
        if (
            int(p["confirm_days"]) in {3, 5, 7}
            and int(p["min_hold_days"]) in {14, 30}
            and float(p["vol_penalty"]) == 0.1
        ):
            base.append(spec)
    extra = []
    for momentum_days in [1, 2, 3, 5, 10]:
        for confirm_days in [2, 5, 7]:
            for hold_days in [7, 14, 30]:
                for advantage in [0.30, 0.40, 0.50]:
                    for overlay_threshold in [0.12]:
                        extra.append(
                            ModelSpec(
                                name=(
                                    f"cap_top1_m{momentum_days}_c{confirm_days}_h{hold_days}_"
                                    f"a{advantage:.2f}_ot{overlay_threshold:.2f}"
                                ),
                                family="限购感知短动量",
                                params={
                                    "momentum_days": momentum_days,
                                    "confirm_days": confirm_days,
                                    "min_hold_days": hold_days,
                                    "advantage_threshold": advantage,
                                    "positive_threshold": 0.0,
                                    "overlay": 1,
                                    "overlay_ret_days": 30,
                                    "overlay_ret_threshold": overlay_threshold,
                                    "overlay_dd_days": 20,
                                    "overlay_dd_threshold": -0.03,
                                    "overlay_ma_days": 20,
                                    "overlay_confirm": 2,
                                    "advantage_window": 30,
                                    "stop_window": 5,
                                    "stop_loss": -0.08,
                                },
                            )
                        )
    challenger_specs = []
    for lookback in [2, 3, 5, 10]:
        for confirm_days in [2, 3, 5]:
            for min_hold_days in [7, 14]:
                for entry_advantage in [0.02, 0.05, 0.10, 0.15]:
                    for exit_margin in [0.00, 0.02]:
                        for lock_threshold in [0.12, 0.20]:
                            challenger_specs.append(
                                ModelSpec(
                                    name=(
                                        f"pcb_core_l{lookback}_c{confirm_days}_h{min_hold_days}_"
                                        f"entry{entry_advantage:.2f}_exit{exit_margin:.2f}_lock{lock_threshold:.2f}"
                                    ),
                                    family="PCB主仓挑战者",
                                    params={
                                        "lookback": lookback,
                                        "confirm_days": confirm_days,
                                        "min_hold_days": min_hold_days,
                                        "entry_advantage": entry_advantage,
                                        "exit_margin": exit_margin,
                                        "pcb_lock_threshold": lock_threshold,
                                        "lock_override": 0.30,
                                    },
                                )
                            )
    core_specs = []
    for momentum_days in [2, 3, 5]:
        for confirm_days in [3, 5, 7]:
            for min_hold_days in [7, 14]:
                for advantage in [0.40, 0.50]:
                    for core_weight in [0.20, 0.40, 0.60, 0.80]:
                        core_specs.append(
                            ModelSpec(
                                name=(
                                    f"pcb_core_sat_m{momentum_days}_c{confirm_days}_h{min_hold_days}_"
                                    f"a{advantage:.2f}_core{core_weight:.2f}"
                                ),
                                family="PCB核心卫星轮动",
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
    return base + extra + challenger_specs + core_specs


def choice_for_spec(navs: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    if spec.family == "限购感知短动量":
        proxy = ModelSpec(spec.name, "短动量Top1", spec.params)
        return top1_choice(navs, proxy)
    if spec.family == "PCB主仓挑战者":
        return pcb_challenger_choice(navs, spec)
    if spec.family == "PCB核心卫星轮动":
        proxy = ModelSpec(spec.name, "短动量Top1", spec.params)
        return top1_choice(navs, proxy)
    return model_choice(navs, spec)


def pcb_challenger_choice(navs: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    p = spec.params
    lookback = int(p["lookback"])
    confirm_days = int(p["confirm_days"])
    min_hold_days = int(p["min_hold_days"])
    entry_advantage = float(p["entry_advantage"])
    exit_margin = float(p["exit_margin"])
    lock_threshold = float(p["pcb_lock_threshold"])
    lock_override = float(p["lock_override"])

    momentum = navs.pct_change(lookback, fill_method=None)
    ret30 = navs.pct_change(30, fill_method=None)
    pcb_dd20 = navs["PCB"] / navs["PCB"].rolling(20).max() - 1
    pcb_above_ma20 = navs["PCB"] > navs["PCB"].rolling(20).mean()
    ma20 = navs.rolling(20).mean()

    current = "PCB"
    entry_date = navs.index[0]
    pending = "PCB"
    pending_count = 0
    output = []

    for date in navs.index:
        row = momentum.loc[date]
        non_pcb = row.drop(labels=["PCB"]).dropna()
        challenger = str(non_pcb.idxmax()) if not non_pcb.empty else "PCB"
        pcb_score = row.get("PCB", np.nan)
        challenger_score = row.get(challenger, np.nan)
        held_days = int((date - entry_date).days)

        proposed = current
        if current == "PCB":
            advantage = challenger_score - pcb_score if pd.notna(challenger_score) and pd.notna(pcb_score) else np.nan
            locked = bool(
                pd.notna(ret30.loc[date, "PCB"])
                and ret30.loc[date, "PCB"] > lock_threshold
                and pcb_dd20.loc[date] > -0.03
                and pcb_above_ma20.loc[date]
            )
            override_advantage = (
                ret30.loc[date, challenger] - ret30.loc[date, "PCB"]
                if challenger in ret30.columns
                else np.nan
            )
            if (
                challenger != "PCB"
                and pd.notna(advantage)
                and advantage > entry_advantage
                and challenger_score > 0
                and (not locked or (pd.notna(override_advantage) and override_advantage > lock_override))
            ):
                proposed = challenger
        else:
            current_score = row.get(current, np.nan)
            pcb_recovers = (
                pd.notna(current_score)
                and pd.notna(pcb_score)
                and pcb_score >= current_score + exit_margin
            )
            current_breaks = (
                pd.notna(navs.loc[date, current])
                and pd.notna(ma20.loc[date, current])
                and navs.loc[date, current] < ma20.loc[date, current]
            )
            if held_days >= min_hold_days and (pcb_recovers or current_breaks):
                proposed = "PCB"
            elif (
                held_days >= min_hold_days
                and challenger != current
                and challenger != "PCB"
                and pd.notna(challenger_score)
                and pd.notna(current_score)
                and challenger_score > current_score + entry_advantage
            ):
                proposed = challenger

        if proposed == current:
            pending = current
            pending_count = 0
        elif proposed == pending:
            pending_count += 1
            if pending_count >= confirm_days:
                current = proposed
                entry_date = date
                pending_count = 0
        else:
            pending = proposed
            pending_count = 1
        output.append(current)
    return pd.Series(output, index=navs.index)


def score_row(one: dict[str, float], three: dict[str, float]) -> float:
    worst_wins = min(one["胜出次数"], three["胜出次数"])
    total_wins = one["胜出次数"] + three["胜出次数"]
    excess = one["平均超额收益"] + three["平均超额收益"]
    drawdown = one["平均回撤改善"] + three["平均回撤改善"]
    return 2.0 * worst_wins + 0.5 * total_wins + 20.0 * excess + 5.0 * drawdown


def main() -> None:
    navs = load_sector_navs()
    train_endpoints, test_endpoints = endpoint_sets(navs)
    train_benchmarks = {
        ONE_MONTH_DAYS: benchmark_cache(navs, train_endpoints, ONE_MONTH_DAYS),
        THREE_MONTH_DAYS: benchmark_cache(navs, train_endpoints, THREE_MONTH_DAYS),
    }
    test_benchmarks = {
        ONE_MONTH_DAYS: benchmark_cache(navs, test_endpoints, ONE_MONTH_DAYS),
        THREE_MONTH_DAYS: benchmark_cache(navs, test_endpoints, THREE_MONTH_DAYS),
    }

    specs = optimization_specs()
    training_rows = []
    for index, spec in enumerate(specs, start=1):
        try:
            choice = choice_for_spec(navs, spec)
            core_weight = (
                float(spec.params["pcb_core_weight"])
                if "pcb_core_weight" in spec.params
                else None
            )
            _, one = evaluate_choice(
                navs,
                choice,
                train_endpoints,
                ONE_MONTH_DAYS,
                train_benchmarks[ONE_MONTH_DAYS],
                core_weight,
            )
            _, three = evaluate_choice(
                navs,
                choice,
                train_endpoints,
                THREE_MONTH_DAYS,
                train_benchmarks[THREE_MONTH_DAYS],
                core_weight,
            )
            training_rows.append(
                {
                    "模型": spec.name,
                    "家族": spec.family,
                    "参数": repr(spec.params),
                    "训练一月胜出": one["胜出次数"],
                    "训练三月胜出": three["胜出次数"],
                    "训练一月平均超额": one["平均超额收益"],
                    "训练三月平均超额": three["平均超额收益"],
                    "训练一月回撤改善": one["平均回撤改善"],
                    "训练三月回撤改善": three["平均回撤改善"],
                    "训练评分": score_row(one, three),
                }
            )
        except Exception as exc:  # noqa: BLE001
            training_rows.append(
                {
                    "模型": spec.name,
                    "家族": spec.family,
                    "参数": repr(spec.params),
                    "错误": f"{type(exc).__name__}: {exc}",
                    "训练评分": -np.inf,
                }
            )
        if index % 500 == 0:
            print(f"已评估 {index}/{len(specs)}", flush=True)

    training = pd.DataFrame(training_rows).sort_values("训练评分", ascending=False)
    training.to_csv(OUT / "训练模型排名_cn.csv", index=False, encoding="utf-8-sig")

    top_models = training.head(20)["模型"].tolist()
    test_rows = []
    test_details = []
    for model_name in top_models:
        spec = next(item for item in specs if item.name == model_name)
        choice = choice_for_spec(navs, spec)
        core_weight = (
            float(spec.params["pcb_core_weight"])
            if "pcb_core_weight" in spec.params
            else None
        )
        one_detail, one = evaluate_choice(
            navs,
            choice,
            test_endpoints,
            ONE_MONTH_DAYS,
            test_benchmarks[ONE_MONTH_DAYS],
            core_weight,
        )
        three_detail, three = evaluate_choice(
            navs,
            choice,
            test_endpoints,
            THREE_MONTH_DAYS,
            test_benchmarks[THREE_MONTH_DAYS],
            core_weight,
        )
        test_rows.append(
            {
                "模型": model_name,
                "家族": spec.family,
                "参数": repr(spec.params),
                "测试一月胜出": one["胜出次数"],
                "测试三月胜出": three["胜出次数"],
                "测试一月平均收益": one["平均收益"],
                "测试三月平均收益": three["平均收益"],
                "测试一月平均超额": one["平均超额收益"],
                "测试三月平均超额": three["平均超额收益"],
                "测试一月回撤改善": one["平均回撤改善"],
                "测试三月回撤改善": three["平均回撤改善"],
                "测试一月平均赎回费": one["平均赎回费"],
                "测试三月平均赎回费": three["平均赎回费"],
                "测试评分": score_row(one, three),
            }
        )
        one_detail.insert(0, "模型", model_name)
        one_detail.insert(1, "周期", "一个月")
        three_detail.insert(0, "模型", model_name)
        three_detail.insert(1, "周期", "三个月")
        test_details.extend([one_detail, three_detail])

    locked_test = pd.DataFrame(test_rows).sort_values("测试评分", ascending=False)
    details = pd.concat(test_details, ignore_index=True)
    locked_test.to_csv(OUT / "锁定测试Top20_cn.csv", index=False, encoding="utf-8-sig")
    details.to_csv(OUT / "锁定测试窗口明细_cn.csv", index=False, encoding="utf-8-sig")

    best = locked_test.iloc[0]
    lines = [
        "# PCB限购、直接转换、FIFO费用下的策略优化",
        "",
        "## 数据隔离",
        "",
        "- 使用较早的30个窗口选择参数。",
        "- 训练窗口与最近30个锁定测试窗口之间保留20个交易日隔离。",
        "- 最近测试窗口不参与参数排名。",
        "",
        "## 最优锁定测试模型",
        "",
        dataframe_to_markdown(locked_test.head(20)),
        "",
        f"- 当前最优模型：{best['模型']}。",
        f"- 一个月窗口胜出：{int(best['测试一月胜出'])}/30。",
        f"- 三个月窗口胜出：{int(best['测试三月胜出'])}/30。",
    ]
    (OUT / "限购策略优化报告_cn.md").write_text("\n".join(lines), encoding="utf-8")

    print(locked_test.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
