from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.optimize_rotation_with_pcb_limit import (
    ONE_MONTH_DAYS,
    THREE_MONTH_DAYS,
    benchmark_cache,
    choice_for_spec,
    endpoint_sets,
    evaluate_choice,
    optimization_specs,
    score_row,
)
from tools.sector_level_rotation_5themes import dataframe_to_markdown, load_sector_navs


OUT = ROOT / "output" / "pcb_limit_strategy_optimization"


def main() -> None:
    navs = load_sector_navs()
    train_endpoints, test_endpoints = endpoint_sets(navs)
    train_bench = {
        ONE_MONTH_DAYS: benchmark_cache(navs, train_endpoints, ONE_MONTH_DAYS),
        THREE_MONTH_DAYS: benchmark_cache(navs, train_endpoints, THREE_MONTH_DAYS),
    }
    test_bench = {
        ONE_MONTH_DAYS: benchmark_cache(navs, test_endpoints, ONE_MONTH_DAYS),
        THREE_MONTH_DAYS: benchmark_cache(navs, test_endpoints, THREE_MONTH_DAYS),
    }
    specs = [spec for spec in optimization_specs() if spec.family == "PCB核心卫星轮动"]
    training_rows = []
    for spec in specs:
        choice = choice_for_spec(navs, spec)
        core = float(spec.params["pcb_core_weight"])
        _, one = evaluate_choice(navs, choice, train_endpoints, ONE_MONTH_DAYS, train_bench[ONE_MONTH_DAYS], core)
        _, three = evaluate_choice(navs, choice, train_endpoints, THREE_MONTH_DAYS, train_bench[THREE_MONTH_DAYS], core)
        training_rows.append(
            {
                "模型": spec.name,
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
    training = pd.DataFrame(training_rows).sort_values("训练评分", ascending=False)
    training.to_csv(OUT / "核心卫星训练排名_cn.csv", index=False, encoding="utf-8-sig")

    test_rows = []
    details = []
    for model_name in training.head(30)["模型"]:
        spec = next(item for item in specs if item.name == model_name)
        choice = choice_for_spec(navs, spec)
        core = float(spec.params["pcb_core_weight"])
        one_detail, one = evaluate_choice(navs, choice, test_endpoints, ONE_MONTH_DAYS, test_bench[ONE_MONTH_DAYS], core)
        three_detail, three = evaluate_choice(navs, choice, test_endpoints, THREE_MONTH_DAYS, test_bench[THREE_MONTH_DAYS], core)
        test_rows.append(
            {
                "模型": model_name,
                "参数": repr(spec.params),
                "PCB核心比例": core,
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
        details.extend([one_detail, three_detail])

    test = pd.DataFrame(test_rows).sort_values("测试评分", ascending=False)
    detail_frame = pd.concat(details, ignore_index=True)
    test.to_csv(OUT / "核心卫星锁定测试_cn.csv", index=False, encoding="utf-8-sig")
    detail_frame.to_csv(OUT / "核心卫星锁定窗口明细_cn.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# PCB核心卫星策略锁定测试",
        "",
        "参数由较早窗口选择，最近30个一月和三月窗口只用于锁定测试。",
        "",
        dataframe_to_markdown(test),
    ]
    (OUT / "核心卫星锁定测试报告_cn.md").write_text("\n".join(lines), encoding="utf-8")
    print(test.head(30).to_string(index=False))


if __name__ == "__main__":
    main()
