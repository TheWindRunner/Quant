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

from tools.plot_sector_rotation_entry_examples import dca_curve, setup_chinese_font, switch_table
from tools.sector_level_rotation_5themes import (
    backtest_choice,
    build_strategy,
    dataframe_to_markdown,
    load_sector_navs,
    max_drawdown,
)


OUT = ROOT / "output" / "sector_level_rotation_5themes" / "一个月跑赢PCB案例"
OUT.mkdir(parents=True, exist_ok=True)
WINDOW_DAYS = 21
WINDOW_COUNT = 30


def evaluate_windows(navs: pd.DataFrame, choice: pd.Series) -> pd.DataFrame:
    rows = []
    for window_id, end_pos in enumerate(range(len(navs) - WINDOW_COUNT, len(navs)), start=1):
        start_pos = end_pos - WINDOW_DAYS + 1
        test = navs.iloc[start_pos : end_pos + 1]
        test_choice = choice.reindex(test.index).ffill()
        curve, _, executed, fee_metrics = backtest_choice(test, test_choice, use_c_redemption_fee=True)
        curve = curve / curve.iloc[0]
        pcb = test["PCB"] / test["PCB"].iloc[0]
        strategy_return = float(curve.iloc[-1] - 1)
        pcb_return = float(pcb.iloc[-1] - 1)
        rows.append(
            {
                "窗口": window_id,
                "开始日期": test.index[0].date().isoformat(),
                "结束日期": test.index[-1].date().isoformat(),
                "轮动收益": strategy_return,
                "全仓PCB收益": pcb_return,
                "轮动超额收益": strategy_return - pcb_return,
                "轮动最大回撤": max_drawdown(curve),
                "全仓PCB最大回撤": max_drawdown(pcb),
                "初始持仓": str(executed.iloc[0]),
                "期末持仓": str(executed.iloc[-1]),
                "切换次数": int(fee_metrics["switch_count"]),
                "赎回费": float(fee_metrics["redemption_fees"]),
                "跑赢全仓PCB": strategy_return > pcb_return,
            }
        )
    return pd.DataFrame(rows)


def select_examples(windows: pd.DataFrame) -> pd.DataFrame:
    winners = windows.loc[windows["跑赢全仓PCB"]].copy()
    if winners.empty:
        return winners
    selected_ids = [
        int(winners["窗口"].min()),
        int(winners.loc[winners["轮动超额收益"].idxmax(), "窗口"]),
        int(winners["窗口"].max()),
    ]
    return windows.loc[windows["窗口"].isin(selected_ids)].sort_values("窗口")


def plot_window(
    navs: pd.DataFrame,
    choice: pd.Series,
    window_row: pd.Series,
) -> tuple[Path, pd.DataFrame, pd.DataFrame]:
    start = pd.Timestamp(window_row["开始日期"])
    end = pd.Timestamp(window_row["结束日期"])
    test = navs.loc[(navs.index >= start) & (navs.index <= end)]
    test_choice = choice.reindex(test.index).ffill()
    strategy, _, executed, fee_metrics = backtest_choice(test, test_choice, use_c_redemption_fee=True)
    strategy = strategy / strategy.iloc[0]
    pcb = test["PCB"] / test["PCB"].iloc[0]
    pcb_dca = dca_curve(test["PCB"])
    equal_dca = dca_curve(test)
    trades = switch_table(executed, strategy)

    fig, ax = plt.subplots(figsize=(13.5, 7.5), dpi=160)
    ax.plot(strategy.index, strategy, label="五板块轮动（计C类赎回费）", linewidth=2.5)
    ax.plot(pcb.index, pcb, label="全仓PCB", linewidth=2.1)
    ax.plot(pcb_dca.index, pcb_dca, label="每日定投PCB", linewidth=1.7)
    ax.plot(equal_dca.index, equal_dca, label="每日定投五板块等权", linewidth=1.5, linestyle="--")

    for row_number, trade in trades.iterrows():
        date = pd.Timestamp(trade["日期"])
        y = float(trade["轮动相对净值"])
        if row_number == 0:
            annotation = f"买入{trade['买入板块']}"
            color = "#2ca02c"
        else:
            annotation = f"卖{trade['卖出板块']} / 买{trade['买入板块']}"
            color = "#d62728"
        ax.scatter([date], [y], color=color, s=50, zorder=5)
        ax.annotate(
            annotation,
            xy=(date, y),
            xytext=(0, 18 if row_number % 2 == 0 else -25),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": color},
        )

    strategy_return = float(strategy.iloc[-1] - 1)
    pcb_return = float(pcb.iloc[-1] - 1)
    title = (
        f"一个月窗口{int(window_row['窗口'])}：{start.date()}至{end.date()}\n"
        f"轮动 {strategy_return:.2%} / 回撤 {max_drawdown(strategy):.2%}；"
        f"全仓PCB {pcb_return:.2%} / 回撤 {max_drawdown(pcb):.2%}；"
        f"超额 {strategy_return - pcb_return:.2%}"
    )
    ax.set_title(title)
    ax.set_ylabel("相对净值（窗口首日=1）")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    out_path = OUT / f"一个月窗口{int(window_row['窗口']):02d}_{start.date()}至{end.date()}_跑赢PCB.png"
    fig.savefig(out_path)
    plt.close(fig)

    metrics = pd.DataFrame(
        [
            {
                "策略": "五板块轮动",
                "总收益率": strategy_return,
                "最大回撤": max_drawdown(strategy),
                "赎回费": fee_metrics["redemption_fees"],
            },
            {"策略": "全仓PCB", "总收益率": pcb_return, "最大回撤": max_drawdown(pcb), "赎回费": 0.0},
            {
                "策略": "每日定投PCB",
                "总收益率": float(pcb_dca.iloc[-1] - 1),
                "最大回撤": max_drawdown(pcb_dca),
                "赎回费": 0.0,
            },
            {
                "策略": "每日定投五板块等权",
                "总收益率": float(equal_dca.iloc[-1] - 1),
                "最大回撤": max_drawdown(equal_dca),
                "赎回费": 0.0,
            },
        ]
    )
    return out_path, metrics, trades


def main() -> None:
    setup_chinese_font()
    navs = load_sector_navs()
    choice, _, _, _ = build_strategy(navs, use_c_redemption_fee=True, protect_redemption_fee=True)
    windows = evaluate_windows(navs, choice)
    selected = select_examples(windows)

    windows.to_csv(OUT / "30个一个月窗口对比_cn.csv", index=False, encoding="utf-8-sig")
    selected.to_csv(OUT / "入选绘图案例_cn.csv", index=False, encoding="utf-8-sig")

    metric_frames = []
    trade_frames = []
    image_paths = []
    for _, row in selected.iterrows():
        image_path, metrics, trades = plot_window(navs, choice, row)
        metrics.insert(0, "窗口", int(row["窗口"]))
        metrics.insert(1, "开始日期", row["开始日期"])
        metrics.insert(2, "结束日期", row["结束日期"])
        trades.insert(0, "窗口", int(row["窗口"]))
        metric_frames.append(metrics)
        trade_frames.append(trades)
        image_paths.append(image_path)

    all_metrics = pd.concat(metric_frames, ignore_index=True)
    all_trades = pd.concat(trade_frames, ignore_index=True)
    all_metrics.to_csv(OUT / "绘图案例收益回撤_cn.csv", index=False, encoding="utf-8-sig")
    all_trades.to_csv(OUT / "绘图案例买卖点_cn.csv", index=False, encoding="utf-8-sig")

    win_count = int(windows["跑赢全仓PCB"].sum())
    lines = [
        "# 30个一个月窗口：轮动与全仓PCB比较",
        "",
        f"- 一个窗口定义为21个基金净值日。",
        f"- 五板块轮动跑赢全仓PCB：{win_count}/30。",
        f"- 绘图案例选择最早跑赢、超额收益最高、最近一次跑赢三个窗口。",
        "",
        "## 入选案例",
        "",
        dataframe_to_markdown(selected),
        "",
        "## 案例收益与回撤",
        "",
        dataframe_to_markdown(all_metrics),
        "",
        "## 案例买卖点",
        "",
        dataframe_to_markdown(all_trades),
    ]
    (OUT / "一个月跑赢PCB案例报告_cn.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"跑赢窗口：{win_count}/30")
    print(selected.to_string(index=False))
    print(all_metrics.to_string(index=False))
    for path in image_paths:
        print(path)


if __name__ == "__main__":
    main()
