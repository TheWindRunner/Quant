from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEPS = ROOT / ".deps"
if DEPS.exists() and str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from tools.backtest_pcb_purchase_limit import (
    INITIAL_CAPITAL,
    PCB_DAILY_LIMIT,
    all_in_pcb_choice,
    simulate,
    simulate_weighted_targets,
)
from tools.optimize_rotation_with_pcb_limit import (
    ONE_MONTH_DAYS,
    THREE_MONTH_DAYS,
    benchmark_cache,
    choice_for_spec,
    endpoint_sets,
    evaluate_choice,
    optimization_specs,
)
from tools.sector_level_rotation_5themes import (
    dataframe_to_markdown,
    load_sector_navs,
    max_drawdown,
)


OUT = ROOT / "output" / "pcb_limit_core_satellite"
MODEL_NAME = "pcb_core_sat_m2_c5_h7_a0.50_core0.80"
PCB_CORE_WEIGHT = 0.80
EXECUTION_DELAY_DAYS = 2


def setup_chinese_font() -> None:
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC"]:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def selected_model():
    return next(spec for spec in optimization_specs() if spec.name == MODEL_NAME)


def target_weights(navs: pd.DataFrame, tactical_choice: pd.Series) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    weights["PCB"] = PCB_CORE_WEIGHT
    for date, sector in tactical_choice.reindex(navs.index).ffill().items():
        weights.loc[date, str(sector)] += 1.0 - PCB_CORE_WEIGHT
    return weights


def limited_dca_curve(pcb_nav: pd.Series) -> pd.Series:
    cash = INITIAL_CAPITAL
    units = 0.0
    daily_budget = min(PCB_DAILY_LIMIT, INITIAL_CAPITAL / len(pcb_nav))
    values = []
    for date, price in pcb_nav.items():
        spend = min(cash, daily_budget, PCB_DAILY_LIMIT)
        units += spend / float(price)
        cash -= spend
        values.append(cash + units * float(price))
    return pd.Series(values, index=pcb_nav.index) / INITIAL_CAPITAL


def curve_metrics(curve: pd.Series) -> tuple[float, float]:
    return float(curve.iloc[-1] - 1.0), max_drawdown(curve)


def nearest_start(index: pd.DatetimeIndex, months: int) -> pd.Timestamp:
    target = index[-1] - pd.DateOffset(months=months)
    return index[index >= target][0]


def executed_tactical(choice: pd.Series) -> pd.Series:
    return choice.shift(EXECUTION_DELAY_DAYS).fillna(choice.iloc[0])


def plot_entry_case(
    navs: pd.DataFrame,
    tactical_choice: pd.Series,
    months: int,
    label: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    start = nearest_start(navs.index, months)
    sliced = navs.loc[start:]
    choice = tactical_choice.reindex(sliced.index).ffill()
    weights = target_weights(sliced, choice)
    strategy_ledger, strategy_metrics = simulate_weighted_targets(sliced, weights)
    all_in_ledger, all_in_metrics = simulate(
        sliced,
        all_in_pcb_choice(sliced.index),
        PCB_DAILY_LIMIT,
    )
    strategy_curve = strategy_ledger["组合市值"] / INITIAL_CAPITAL
    all_in_curve = all_in_ledger["组合市值"] / INITIAL_CAPITAL
    dca_curve = limited_dca_curve(sliced["PCB"])

    executed = executed_tactical(choice)
    switches = executed.ne(executed.shift()).fillna(True)
    trade_rows = []
    previous = None
    for number, (date, sector) in enumerate(executed[switches].items()):
        action = "初始配置" if previous is None else "卫星切换"
        trade_rows.append(
            {
                "进场口径": label,
                "日期": date.date().isoformat(),
                "动作": action,
                "卖出卫星板块": "" if previous is None else previous,
                "买入卫星板块": str(sector),
                "PCB核心目标权重": PCB_CORE_WEIGHT,
                "卫星目标权重": 1.0 - PCB_CORE_WEIGHT,
                "策略相对净值": float(strategy_curve.loc[date]),
            }
        )
        previous = str(sector)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    ax.plot(strategy_curve, label="PCB核心卫星策略（真实限购及C类赎回费）", linewidth=2.4)
    ax.plot(all_in_curve, label="限购条件下全仓PCB", linewidth=2.0)
    ax.plot(dca_curve, label="每日定投PCB", linewidth=1.8, linestyle="--")
    for number, row in enumerate(trade_rows):
        date = pd.Timestamp(row["日期"])
        y = float(row["策略相对净值"])
        text = (
            f"初始：80% PCB + 20% {row['买入卫星板块']}"
            if row["动作"] == "初始配置"
            else f"卫星：{row['卖出卫星板块']}→{row['买入卫星板块']}"
        )
        ax.scatter(date, y, color="#d62728", s=42, zorder=5)
        ax.annotate(
            text,
            (date, y),
            xytext=(0, 16 if number % 2 == 0 else -25),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.7},
        )

    dca_ret, dca_mdd = curve_metrics(dca_curve)
    ax.set_title(
        f"{label}进场：核心卫星策略 vs 全仓PCB vs 每日定投\n"
        f"核心卫星 {strategy_metrics['总收益率']:.2%}/{strategy_metrics['最大回撤']:.2%}；"
        f"全仓PCB {all_in_metrics['总收益率']:.2%}/{all_in_metrics['最大回撤']:.2%}；"
        f"定投 {dca_ret:.2%}/{dca_mdd:.2%}"
    )
    ax.set_ylabel("账户相对净值（初始资金10000元）")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    chart = OUT / f"{label}进场_核心卫星与全仓定投对比_买卖点_cn.png"
    fig.savefig(chart)
    plt.close(fig)

    metrics = pd.DataFrame(
        [
            {
                "进场口径": label,
                "开始日期": start.date().isoformat(),
                "结束日期": sliced.index[-1].date().isoformat(),
                "策略": "PCB核心卫星策略",
                "总收益率": strategy_metrics["总收益率"],
                "最大回撤": strategy_metrics["最大回撤"],
                "赎回费占初始资金": strategy_metrics["赎回费"],
                "平均待到账比例": strategy_metrics["平均待到账比例"],
            },
            {
                "进场口径": label,
                "开始日期": start.date().isoformat(),
                "结束日期": sliced.index[-1].date().isoformat(),
                "策略": "限购条件下全仓PCB",
                "总收益率": all_in_metrics["总收益率"],
                "最大回撤": all_in_metrics["最大回撤"],
                "赎回费占初始资金": all_in_metrics["赎回费"],
                "平均待到账比例": 0.0,
            },
            {
                "进场口径": label,
                "开始日期": start.date().isoformat(),
                "结束日期": sliced.index[-1].date().isoformat(),
                "策略": "每日定投PCB",
                "总收益率": dca_ret,
                "最大回撤": dca_mdd,
                "赎回费占初始资金": 0.0,
                "平均待到账比例": 0.0,
            },
        ]
    )
    return metrics, pd.DataFrame(trade_rows)


def locked_window_report(
    navs: pd.DataFrame,
    tactical_choice: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _, endpoints = endpoint_sets(navs)
    summaries = []
    details = []
    for days, label in [(ONE_MONTH_DAYS, "一个月"), (THREE_MONTH_DAYS, "三个月")]:
        benchmark = benchmark_cache(navs, endpoints, days)
        detail, summary = evaluate_choice(
            navs,
            tactical_choice,
            endpoints,
            days,
            benchmark,
            PCB_CORE_WEIGHT,
        )
        detail.insert(0, "周期", label)
        details.append(detail)
        summaries.append(
            {
                "周期": label,
                "胜出窗口数": int(summary["胜出次数"]),
                "窗口总数": len(endpoints),
                "胜率": summary["胜出次数"] / len(endpoints),
                "平均收益率": summary["平均收益"],
                "平均超额收益": summary["平均超额收益"],
                "平均最大回撤": summary["平均最大回撤"],
                "平均回撤改善": summary["平均回撤改善"],
                "平均赎回费": summary["平均赎回费"],
            }
        )
    return pd.DataFrame(summaries), pd.concat(details, ignore_index=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    setup_chinese_font()
    navs = load_sector_navs()
    model = selected_model()
    tactical_choice = choice_for_spec(navs, model)

    summary, details = locked_window_report(navs, tactical_choice)
    summary.to_csv(OUT / "锁定测试摘要_cn.csv", index=False, encoding="utf-8-sig")
    details.to_csv(OUT / "锁定测试30窗口明细_cn.csv", index=False, encoding="utf-8-sig")

    fixed_metrics = []
    fixed_trades = []
    for months, label in [(1, "一个月前"), (3, "三个月前"), (6, "半年前")]:
        metrics, trades = plot_entry_case(navs, tactical_choice, months, label)
        fixed_metrics.append(metrics)
        fixed_trades.append(trades)
    metric_frame = pd.concat(fixed_metrics, ignore_index=True)
    trade_frame = pd.concat(fixed_trades, ignore_index=True)
    metric_frame.to_csv(OUT / "一三六个月收益回撤对比_cn.csv", index=False, encoding="utf-8-sig")
    trade_frame.to_csv(OUT / "一三六个月卫星切换点_cn.csv", index=False, encoding="utf-8-sig")

    signal_now = str(tactical_choice.iloc[-1])
    executed_now = str(executed_tactical(tactical_choice).iloc[-1])
    current_weights = {"PCB": PCB_CORE_WEIGHT}
    current_weights[executed_now] = current_weights.get(executed_now, 0.0) + 0.20
    weights_text = "、".join(f"{sector} {weight:.0%}" for sector, weight in current_weights.items())
    report = [
        "# PCB限购核心卫星策略报告",
        "",
        f"固定模型：`{MODEL_NAME}`。参数来自较早训练窗口，最近30个窗口仅作锁定测试。",
        "",
        "## 执行规则",
        "",
        "- 战略核心为80% PCB，20%卫星仓按五板块信号轮动；若卫星也选中PCB，目标为100% PCB。",
        "- 板块基金之间转换：执行日卖出旧基金并同日买入新基金；转入PCB每日最多4000元，未转换部分继续持有旧板块。",
        "- 主动变回现金：T日15:00前赎回，T+1日15:00后到账，T+2才允许重新申购。待到账款计入资产，但不计入可用现金。",
        "- 信号延迟2个净值日执行；C类赎回费按FIFO：不足7天1.5%，7至30天0.5%，满30天0%。",
        "",
        "## 最新模型状态",
        "",
        f"- 最新信号日卫星板块：{signal_now}",
        f"- 当前执行卫星板块（含2净值日延迟）：{executed_now}",
        f"- 当前目标：{weights_text}",
        "",
        "## 锁定窗口结果",
        "",
        dataframe_to_markdown(summary),
        "",
        "## 一、三、六个月固定进场",
        "",
        dataframe_to_markdown(metric_frame),
        "",
        "说明：当前模型没有主动现金信号，因此正常板块轮动不会触发T+2现金等待；现金状态机已为未来止盈清仓或风险空仓信号启用。",
    ]
    (OUT / "PCB限购核心卫星策略报告_cn.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False))
    print(metric_frame.to_string(index=False))


if __name__ == "__main__":
    main()
