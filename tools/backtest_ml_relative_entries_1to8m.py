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
from tools.ml_relative_satellite_60day import (
    feature_panel,
    fit_predict,
    predicted_choice,
)
from tools.sector_level_rotation_5themes import load_sector_navs, max_drawdown
from tools.short_term_60day_holdout import dca_curve, redemption_fee, sharpe_ratio


PCB_BASE_CODE = "006503"
OUT = ROOT / "output" / f"ml_relative_entries_1to8m_pcb_{PCB_BASE_CODE}"
PREDICTION_HORIZON = 10
RIDGE_ALPHA = 10.0
PREDICTION_THRESHOLD = 0.0025
CONFIRM_DAYS = 2
MIN_HOLD_DAYS = 14
PCB_CORE_WEIGHT = 0.80
EXECUTION_DELAY_DAYS = 2


def nearest_start(index: pd.DatetimeIndex, months: int) -> pd.Timestamp:
    target = index[-1] - pd.DateOffset(months=months)
    return index[index >= target][0]


def target_weights(navs: pd.DataFrame, choice: pd.Series) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    weights["PCB"] = PCB_CORE_WEIGHT
    for date, sector in choice.reindex(navs.index).ffill().items():
        weights.loc[date, str(sector)] += 1.0 - PCB_CORE_WEIGHT
    return weights


def fit_as_of(
    navs: pd.DataFrame,
    panel: pd.DataFrame,
    fit_end: pd.Timestamp,
) -> tuple[pd.Series, pd.DataFrame, pd.Timestamp]:
    prediction, _ = fit_predict(
        panel,
        navs,
        PREDICTION_HORIZON,
        RIDGE_ALPHA,
        fit_end,
    )
    choice = predicted_choice(
        navs,
        prediction,
        PREDICTION_THRESHOLD,
        CONFIRM_DAYS,
        MIN_HOLD_DAYS,
    )
    raw_weights = target_weights(navs, choice)
    executed_weights = raw_weights.shift(EXECUTION_DELAY_DAYS).fillna(raw_weights.iloc[0])
    executed_choice = choice.shift(EXECUTION_DELAY_DAYS).fillna(choice.iloc[0])
    fit_end_pos = int(navs.index.get_loc(fit_end))
    label_end = navs.index[fit_end_pos - PREDICTION_HORIZON]
    return executed_choice, executed_weights, label_end


def strategy_simulation(
    navs: pd.DataFrame,
    executed_weights: pd.DataFrame,
) -> tuple[pd.Series, dict[str, float]]:
    ledger, metrics = simulate_weighted_targets(
        navs,
        executed_weights.reindex(navs.index),
        execution_delay_days=0,
    )
    return ledger.iloc[:, 0] / INITIAL_CAPITAL, metrics


def metric_row(
    months: int,
    start: pd.Timestamp,
    end: pd.Timestamp,
    strategy: str,
    curve: pd.Series,
    total_return: float,
    mdd: float,
    fee: float,
    fit_end: pd.Timestamp,
    label_end: pd.Timestamp,
) -> dict[str, float | int | str]:
    return {
        "入场月数": months,
        "开始日期": start.date().isoformat(),
        "结束日期": end.date().isoformat(),
        "模型拟合截止": fit_end.date().isoformat(),
        "最后成熟标签日": label_end.date().isoformat(),
        "策略": strategy,
        "收益率": total_return,
        "最大回撤": mdd,
        "夏普比率": sharpe_ratio(curve),
        "赎回费占初始资金": fee,
    }


def plot_case(
    months: int,
    curves: dict[str, pd.Series],
    trades: pd.DataFrame,
    metrics: pd.DataFrame,
) -> None:
    setup_chinese_font()
    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    for name, curve in curves.items():
        ax.plot(curve, label=name, linewidth=2.2 if name == "多因子核心卫星" else 1.9)
    strategy_curve = curves["多因子核心卫星"]
    for number, row in trades.iterrows():
        date = pd.Timestamp(row["日期"])
        if date not in strategy_curve.index:
            continue
        y = float(strategy_curve.loc[date])
        ax.scatter(date, y, color="#d62728", s=34, zorder=5)
        ax.annotate(
            f"{number + 1}.{row['目标卫星']}",
            (date, y),
            xytext=(0, 12 if number % 2 == 0 else -18),
            textcoords="offset points",
            ha="center",
            fontsize=7,
        )
    metric_text = "；".join(
        f"{row['策略']} {row['收益率']:.2%}/{row['最大回撤']:.2%}/夏普{row['夏普比率']:.2f}"
        for _, row in metrics.iterrows()
    )
    ax.set_title(f"{months}个月前入场至今\n{metric_text}")
    ax.set_ylabel("账户相对净值")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(OUT / f"{months}个月前入场_收益回撤夏普与切换点_cn.png")
    plt.close(fig)


def backtest_entry_cases(navs: pd.DataFrame, panel: pd.DataFrame):
    metric_rows = []
    trade_rows = []
    for months in range(1, 9):
        start = nearest_start(navs.index, months)
        start_pos = int(navs.index.get_loc(start))
        fit_end = navs.index[start_pos - 1]
        executed_choice, executed_weights, label_end = fit_as_of(navs, panel, fit_end)
        test = navs.loc[start:]
        strategy_curve, strategy_metrics = strategy_simulation(
            test, executed_weights.loc[start:]
        )
        all_in_ledger, all_in_metrics = simulate(
            test,
            all_in_pcb_choice(test.index),
            PCB_DAILY_LIMIT,
        )
        all_in_curve = all_in_ledger.iloc[:, 0] / INITIAL_CAPITAL
        daily_dca = dca_curve(test["PCB"])
        strategy_values = list(strategy_metrics.values())
        all_in_values = list(all_in_metrics.values())
        rows = [
            metric_row(
                months,
                start,
                test.index[-1],
                "多因子核心卫星",
                strategy_curve,
                float(strategy_values[0]),
                float(strategy_values[1]),
                redemption_fee(strategy_metrics),
                fit_end,
                label_end,
            ),
            metric_row(
                months,
                start,
                test.index[-1],
                "限购全仓PCB",
                all_in_curve,
                float(all_in_values[0]),
                float(all_in_values[1]),
                0.0,
                fit_end,
                label_end,
            ),
            metric_row(
                months,
                start,
                test.index[-1],
                "每日定投PCB",
                daily_dca,
                float(daily_dca.iloc[-1] - 1.0),
                max_drawdown(daily_dca),
                0.0,
                fit_end,
                label_end,
            ),
        ]
        metric_frame = pd.DataFrame(rows)
        metric_rows.extend(rows)

        sliced_choice = executed_choice.loc[start:]
        switches = sliced_choice.ne(sliced_choice.shift()).fillna(True)
        trades = pd.DataFrame(
            {
                "入场月数": months,
                "日期": sliced_choice.index[switches],
                "目标卫星": sliced_choice.loc[switches].astype(str).to_numpy(),
            }
        )
        trades["PCB核心权重"] = PCB_CORE_WEIGHT
        trades["卫星权重"] = 1.0 - PCB_CORE_WEIGHT
        trade_rows.append(trades)
        curves = {
            "多因子核心卫星": strategy_curve,
            "限购全仓PCB": all_in_curve,
            "每日定投PCB": daily_dca,
        }
        plot_case(months, curves, trades.reset_index(drop=True), metric_frame)
    return pd.DataFrame(metric_rows), pd.concat(trade_rows, ignore_index=True)


def corrected_locked_window_audit(navs: pd.DataFrame, panel: pd.DataFrame) -> pd.DataFrame:
    train_end = navs.index[len(navs) - 60 - 1]
    _, executed_weights, label_end = fit_as_of(navs, panel, train_end)
    rows = []
    for window_id, end_pos in enumerate(range(len(navs) - 30, len(navs)), start=1):
        start_pos = end_pos - 21 + 1
        test = navs.iloc[start_pos : end_pos + 1]
        strategy_curve, strategy_metrics = strategy_simulation(
            test, executed_weights.reindex(test.index)
        )
        all_in_ledger, all_in_metrics = simulate(
            test, all_in_pcb_choice(test.index), PCB_DAILY_LIMIT
        )
        strategy_return = float(list(strategy_metrics.values())[0])
        all_in_return = float(list(all_in_metrics.values())[0])
        rows.append(
            {
                "窗口": window_id,
                "开始日期": test.index[0].date().isoformat(),
                "结束日期": test.index[-1].date().isoformat(),
                "训练截止": train_end.date().isoformat(),
                "最后成熟标签日": label_end.date().isoformat(),
                "策略收益率": strategy_return,
                "全仓PCB收益率": all_in_return,
                "超额收益": strategy_return - all_in_return,
                "跑赢": strategy_return > all_in_return,
                "策略最大回撤": float(list(strategy_metrics.values())[1]),
                "策略夏普": sharpe_ratio(strategy_curve),
            }
        )
    return pd.DataFrame(rows)


def plot_summary(metrics: pd.DataFrame) -> None:
    setup_chinese_font()
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), dpi=160)
    for ax, field, title in zip(
        axes,
        ["收益率", "最大回撤", "夏普比率"],
        ["累计收益率", "最大回撤", "夏普比率"],
    ):
        pivot = metrics.pivot(index="入场月数", columns="策略", values=field)
        pivot.plot(kind="bar", ax=ax)
        ax.set_title(title)
        ax.set_xlabel("距今入场月数")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(fontsize=8)
    fig.suptitle("多因子核心卫星：一至八个月入场结果总览")
    fig.tight_layout()
    fig.savefig(OUT / "一至八个月收益回撤夏普总览_cn.png")
    plt.close(fig)


def make_report(metrics: pd.DataFrame, audit: pd.DataFrame) -> None:
    comparison = metrics.pivot(index="入场月数", columns="策略", values=["收益率", "最大回撤", "夏普比率"])
    strategy = metrics.loc[metrics["策略"] == "多因子核心卫星"].set_index("入场月数")
    all_in = metrics.loc[metrics["策略"] == "限购全仓PCB"].set_index("入场月数")
    summary = pd.DataFrame(
        {
            "入场月数": range(1, 9),
            "策略收益率": strategy["收益率"],
            "全仓PCB收益率": all_in["收益率"],
            "策略相对全仓超额": strategy["收益率"] - all_in["收益率"],
            "策略最大回撤": strategy["最大回撤"],
            "全仓PCB最大回撤": all_in["最大回撤"],
            "策略夏普": strategy["夏普比率"],
            "全仓PCB夏普": all_in["夏普比率"],
        }
    ).reset_index(drop=True)
    summary.to_csv(OUT / "一至八个月核心对比摘要_cn.csv", index=False, encoding="utf-8-sig")
    lines = [
        "# 多因子核心卫星一至八个月入场回测",
        "",
        "## 方法",
        "",
        "- 固定当前策略结构与超参数；每个入场日仅使用此前数据重拟合岭回归系数。",
        "- 预测未来10日相对PCB收益，标签必须成熟；信号在完整历史上延迟2个净值日后再切取回测区间。",
        "- 80% PCB核心仓、20%卫星仓；C类FIFO赎回费、基金当日转换、PCB四通道合计每日4000元限购、现金T+2可用。",
        "",
        "## 结果摘要",
        "",
        summary.to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 窗口边界修正复核",
        "",
        f"修正窗口头两日延迟信号后，原30个一月封存窗口跑赢 {int(audit['跑赢'].sum())}/30，平均超额 {audit['超额收益'].mean():.2%}。",
        "",
        "完整三策略指标见 `一至八个月完整指标_cn.csv`，卫星切换见 `一至八个月卫星切换点_cn.csv`。",
    ]
    (OUT / "一至八个月入场回测报告_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    navs = load_sector_navs(pcb_signal_code=PCB_BASE_CODE)
    panel = feature_panel(navs)
    metrics, trades = backtest_entry_cases(navs, panel)
    audit = corrected_locked_window_audit(navs, panel)
    metrics.to_csv(OUT / "一至八个月完整指标_cn.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(OUT / "一至八个月卫星切换点_cn.csv", index=False, encoding="utf-8-sig")
    audit.to_csv(OUT / "修正延迟边界后的30窗口复核_cn.csv", index=False, encoding="utf-8-sig")
    plot_summary(metrics)
    make_report(metrics, audit)
    print(metrics.to_string(index=False))
    print(f"修正延迟边界后30窗口胜出: {int(audit['跑赢'].sum())}/30")


if __name__ == "__main__":
    main()
