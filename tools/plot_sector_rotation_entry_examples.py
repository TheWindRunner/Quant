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

from tools.sector_level_rotation_5themes import (
    backtest_choice,
    build_strategy,
    dataframe_to_markdown,
    load_sector_navs,
    max_drawdown,
)


OUT = ROOT / "output" / "sector_level_rotation_5themes"


def setup_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


def dca_curve(price: pd.Series | pd.DataFrame) -> pd.Series:
    frame = price.to_frame("asset") if isinstance(price, pd.Series) else price.copy()
    cash = 1.0
    units = pd.Series(0.0, index=frame.columns)
    daily_budget = 1.0 / len(frame)
    values = []
    for date in frame.index:
        spend = min(cash, daily_budget)
        if spend > 0:
            units += (spend / len(frame.columns)) / frame.loc[date]
            cash -= spend
        values.append(cash + float((units * frame.loc[date]).sum()))
    curve = pd.Series(values, index=frame.index)
    return curve / curve.iloc[0]


def metrics(curve: pd.Series) -> tuple[float, float]:
    return float(curve.iloc[-1] - 1), max_drawdown(curve)


def nearest_start(index: pd.DatetimeIndex, months: int) -> pd.Timestamp:
    target = index.max() - pd.DateOffset(months=months)
    return index[index >= target][0]


def switch_table(executed: pd.Series, strategy: pd.Series) -> pd.DataFrame:
    rows = []
    previous = str(executed.iloc[0])
    entry_date = executed.index[0]
    rows.append(
        {
            "日期": entry_date,
            "动作": "初始买入",
            "卖出板块": "",
            "买入板块": previous,
            "上一持仓天数": 0,
            "轮动相对净值": float(strategy.loc[entry_date]),
        }
    )
    for date, current_value in executed.iloc[1:].items():
        current = str(current_value)
        if current == previous:
            continue
        rows.append(
            {
                "日期": date,
                "动作": "切换",
                "卖出板块": previous,
                "买入板块": current,
                "上一持仓天数": int((date - entry_date).days),
                "轮动相对净值": float(strategy.loc[date]),
            }
        )
        previous = current
        entry_date = date
    return pd.DataFrame(rows)


def plot_case(
    navs: pd.DataFrame,
    choice: pd.Series,
    months: int,
    label: str,
) -> dict[str, object]:
    start = nearest_start(navs.index, months)
    sliced = navs.loc[navs.index >= start]
    sliced_choice = choice.reindex(sliced.index).ffill()
    strategy, _, executed, fee_metrics = backtest_choice(
        sliced,
        sliced_choice,
        use_c_redemption_fee=True,
    )
    strategy = strategy / strategy.iloc[0]
    pcb_all_in = sliced["PCB"] / sliced["PCB"].iloc[0]
    pcb_dca = dca_curve(sliced["PCB"])
    equal_dca = dca_curve(sliced)
    trades = switch_table(executed, strategy)

    fig, ax = plt.subplots(figsize=(14, 8), dpi=160)
    ax.plot(strategy.index, strategy, label="五板块轮动（计C类赎回费）", linewidth=2.5)
    ax.plot(pcb_all_in.index, pcb_all_in, label="全仓PCB", linewidth=2.0)
    ax.plot(pcb_dca.index, pcb_dca, label="每日定投PCB", linewidth=1.8)
    ax.plot(equal_dca.index, equal_dca, label="每日定投五板块等权", linewidth=1.6, linestyle="--")

    for row_number, row in trades.iterrows():
        date = pd.Timestamp(row["日期"])
        y = float(row["轮动相对净值"])
        if row_number == 0:
            text = f"买入{row['买入板块']}"
            color = "#2ca02c"
        else:
            text = f"卖{row['卖出板块']} / 买{row['买入板块']}"
            color = "#d62728"
        ax.scatter([date], [y], s=48, color=color, zorder=5)
        ax.annotate(
            text,
            xy=(date, y),
            xytext=(0, 18 if row_number % 2 == 0 else -26),
            textcoords="offset points",
            ha="center",
            fontsize=8,
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": color},
        )

    ret_s, dd_s = metrics(strategy)
    ret_p, dd_p = metrics(pcb_all_in)
    ret_dca, dd_dca = metrics(pcb_dca)
    ret_eqdca, dd_eqdca = metrics(equal_dca)
    title = (
        f"{label}进场：五板块轮动 vs 定投 vs 全仓PCB\n"
        f"轮动 {ret_s:.2%}/{dd_s:.2%}；全仓PCB {ret_p:.2%}/{dd_p:.2%}；"
        f"定投PCB {ret_dca:.2%}/{dd_dca:.2%}；五板块定投 {ret_eqdca:.2%}/{dd_eqdca:.2%}"
    )
    ax.set_title(title)
    ax.set_ylabel("相对净值（进场日=1）")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()

    out_path = OUT / f"五板块轮动_{label}进场_计C类赎回费_收益曲线与买卖点.png"
    fig.savefig(out_path)
    plt.close(fig)

    metric_frame = pd.DataFrame(
        [
            {
                "策略": "五板块轮动（计C类赎回费）",
                "总收益率": ret_s,
                "最大回撤": dd_s,
                "赎回费": fee_metrics["redemption_fees"],
            },
            {"策略": "全仓PCB", "总收益率": ret_p, "最大回撤": dd_p, "赎回费": 0.0},
            {"策略": "每日定投PCB", "总收益率": ret_dca, "最大回撤": dd_dca, "赎回费": 0.0},
            {"策略": "每日定投五板块等权", "总收益率": ret_eqdca, "最大回撤": dd_eqdca, "赎回费": 0.0},
        ]
    )
    return {
        "label": label,
        "start": start,
        "end": sliced.index[-1],
        "initial": str(executed.iloc[0]),
        "plot": out_path,
        "metrics": metric_frame,
        "trades": trades,
    }


def generate_standard_entry_reports(
    navs: pd.DataFrame | None = None,
    choice: pd.Series | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    setup_chinese_font()
    navs = load_sector_navs() if navs is None else navs
    if choice is None:
        choice, _, _, _ = build_strategy(
            navs,
            use_c_redemption_fee=True,
            protect_redemption_fee=True,
        )
    cases = [
        plot_case(navs, choice, 1, "一个月前"),
        plot_case(navs, choice, 3, "三个月前"),
        plot_case(navs, choice, 6, "半年前"),
    ]
    metric_rows = []
    trade_rows = []
    for case in cases:
        metric_frame = case["metrics"].copy()
        metric_frame.insert(0, "进场口径", case["label"])
        metric_frame.insert(1, "进场日期", case["start"].date().isoformat())
        metric_frame.insert(2, "结束日期", case["end"].date().isoformat())
        metric_frame.insert(3, "初始持仓", case["initial"])
        metric_rows.append(metric_frame)

        trades = case["trades"].copy()
        trades.insert(0, "进场口径", case["label"])
        trade_rows.append(trades)

    metrics_all = pd.concat(metric_rows, ignore_index=True)
    trades_all = pd.concat(trade_rows, ignore_index=True)
    metrics_all.to_csv(OUT / "常规_一三六月进场收益回撤对比_cn.csv", index=False, encoding="utf-8-sig")
    trades_all.to_csv(OUT / "常规_一三六月进场买卖点_cn.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 五板块轮动常规进场对比",
        "",
        "每次主策略运行后自动生成一个月、三个月、半年前进场的收益曲线、买卖点、总收益和最大回撤。",
        "",
        "## 收益与回撤",
        "",
        dataframe_to_markdown(metrics_all),
        "",
        "## 买卖点",
        "",
        dataframe_to_markdown(trades_all),
    ]
    (OUT / "常规_一三六月进场对比报告_cn.md").write_text("\n".join(lines), encoding="utf-8")
    return metrics_all, trades_all


def main() -> None:
    metrics_all, trades_all = generate_standard_entry_reports()
    print(metrics_all.to_string(index=False))
    print(trades_all.to_string(index=False))


if __name__ == "__main__":
    main()
