from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from quant_fund_advisor.style_timing import read_cached_nav


OUT = Path("output/hybrid_period_comparison")
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CODES = ["006503", "011370", "007817", "008887", "008585", "720001"]
ANCHOR = "011370"
COST = 0.0015
PERIODS = [
    ("近1个月", 31, "1m"),
    ("近3个月", 93, "3m"),
    ("近1年", 365, "1y"),
]


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


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


def build_hybrid_choice(navs: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    momentum = navs.pct_change(2, fill_method=None).where(lambda frame: frame > 0)
    columns = np.array(momentum.columns)
    raw = []
    for row in momentum.to_numpy():
        mask = ~np.isnan(row)
        raw.append(columns[mask][np.argmax(row[mask])] if mask.any() else ANCHOR)
    base_choice = confirm_choice(pd.Series(raw, index=navs.index), confirm=5)

    core = navs["006503"]
    regime = (
        (core.pct_change(30, fill_method=None) > 0.12)
        & (core / core.rolling(20).max() - 1 > -0.03)
        & (core > core.rolling(20).mean())
    ).fillna(False)
    streak = 0
    force_core = []
    for flag in regime.to_numpy():
        streak = streak + 1 if flag else 0
        force_core.append(streak >= 2)
    hybrid = base_choice.copy()
    hybrid.loc[pd.Series(force_core, index=navs.index)] = "006503"
    return base_choice, hybrid


def returns_from_choice(navs: pd.DataFrame, choice: pd.Series) -> pd.Series:
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    position = choice.shift(2).fillna(ANCHOR).to_numpy(dtype=object)
    strategy_return = np.zeros(len(position))
    for code in navs.columns:
        strategy_return += (position == code) * returns[code].to_numpy()
    turnover_cost = (choice != choice.shift()).astype(float).shift(1).fillna(0.0).to_numpy() * COST
    return pd.Series(strategy_return - turnover_cost, index=navs.index)


def dca_curve(series: pd.Series, start: pd.Timestamp) -> pd.Series:
    window = series.loc[start:]
    shares = 0.0
    invested = 0.0
    values = []
    for date, nav in window.items():
        invested += 1.0
        shares += 1.0 / nav
        values.append((date, shares * nav / invested))
    curve = pd.Series(dict(values), name="006503每日定投")
    return curve / curve.iloc[0]


def trade_points(choice: pd.Series) -> pd.DataFrame:
    shifted = choice.shift()
    trades = choice[choice != shifted].iloc[1:]
    rows = []
    for date, target in trades.items():
        rows.append(
            {
                "date": date,
                "target": target,
                "action": "买入/切到006503" if target == "006503" else f"卖出006503/切到{target}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    navs = pd.DataFrame({code: read_cached_nav(code) for code in CODES}).dropna().sort_index()
    _, hybrid_choice = build_hybrid_choice(navs)
    hybrid_returns = returns_from_choice(navs, hybrid_choice)
    trade_log = trade_points(hybrid_choice)
    trade_log.to_csv(OUT / "hybrid_trade_points_cn.csv", index=False, encoding="utf-8-sig")

    rows = []
    for title, days, slug in PERIODS:
        start = navs.index[-1] - pd.Timedelta(days=days)
        start = navs.index[navs.index >= start][0]
        strategy_curve = (1 + hybrid_returns.loc[start:]).cumprod()
        strategy_curve = strategy_curve / strategy_curve.iloc[0]
        all_in_curve = navs.loc[start:, "006503"] / navs.loc[start, "006503"]
        dca = dca_curve(navs["006503"], start).reindex(strategy_curve.index).ffill()

        curves = pd.DataFrame(
            {
                "混合策略": strategy_curve,
                "006503全仓": all_in_curve.reindex(strategy_curve.index),
                "006503每日定投": dca,
            }
        )
        curves.to_csv(OUT / f"{slug}_curves_cn.csv", encoding="utf-8-sig")

        for column in curves.columns:
            rows.append(
                {
                    "区间": title,
                    "策略": column,
                    "收益率": float(curves[column].iloc[-1] - 1),
                    "最大回撤": max_drawdown(curves[column]),
                }
            )

        trades = trade_log[(trade_log["date"] >= start) & (trade_log["date"] <= curves.index[-1])]
        fig, ax = plt.subplots(figsize=(15, 7))
        for column in curves.columns:
            ax.plot(curves.index, curves[column], lw=1.8, label=column)
        for _, trade in trades.iterrows():
            date = pd.Timestamp(trade["date"])
            if date in curves.index:
                y = curves.loc[date, "混合策略"]
                if trade["target"] == "006503":
                    ax.scatter([date], [y], marker="^", s=90, color="#d62728", edgecolor="white", linewidth=0.6, zorder=5)
                else:
                    ax.scatter([date], [y], marker="v", s=90, color="#2ca02c", edgecolor="white", linewidth=0.6, zorder=5)
        start_target = hybrid_choice.loc[start]
        start_y = curves.iloc[0]["混合策略"]
        start_label = f"区间起点持仓：{start_target}"
        start_color = "#d62728" if start_target == "006503" else "#2ca02c"
        ax.scatter([curves.index[0]], [start_y], marker="o", s=70, color=start_color, edgecolor="white", linewidth=0.6, zorder=5)
        ax.annotate(start_label, (curves.index[0], start_y), xytext=(8, 8), textcoords="offset points", fontsize=9, color=start_color)
        if trades.empty:
            ax.text(
                0.02,
                0.96,
                f"{title}内无新增买卖点，持续持有 {start_target}",
                transform=ax.transAxes,
                fontsize=10,
                va="top",
                ha="left",
                bbox={"boxstyle": "round,pad=0.3", "facecolor": "#fff7e6", "edgecolor": "#d9a441", "alpha": 0.9},
            )
        ax.set_title(f"{title}：混合策略 vs 全仓 vs 定投", fontsize=15)
        ax.set_xlabel("日期")
        ax.set_ylabel("累计净值（起点=1）")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best")
        fig.tight_layout()
        fig.savefig(OUT / f"{slug}_comparison_cn.svg", format="svg")
        plt.close(fig)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "period_summary_cn.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 混合策略分区间对比",
        "",
        "## 口径",
        "",
        "- 混合策略：当前生产候选策略。",
        "- 006503全仓：区间起点一次性买入后不动。",
        "- 006503每日定投：区间内每个净值日等额买入。",
        "- 策略买卖点已经标注在曲线上，红色三角代表切到006503，绿色三角代表卖出006503切到其他基金。",
        "",
        "## 结果",
        "",
        "| 区间 | 策略 | 收益率 | 最大回撤 |",
        "|---|---|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(f"| {row['区间']} | {row['策略']} | {pct(row['收益率'])} | {pct(row['最大回撤'])} |")
    (OUT / "period_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
