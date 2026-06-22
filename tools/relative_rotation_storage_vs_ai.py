from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from quant_fund_advisor.style_timing import read_cached_nav


OUT = Path("output/relative_rotation_storage_vs_ai")
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

STORAGE = "008887"
STORAGE_NAME = "华夏国证半导体芯片ETF联接A（008887，存储/半导体代理）"
AI = "008585"
AI_NAME = "华夏中证人工智能主题ETF联接A（008585，AI/人工智能）"

FAST = 15
SLOW = 40
CONFIRM = 2
COST = 0.0015

PERIODS = [
    ("全共同历史", "full", None),
    ("近1年", "1y", 365),
    ("近6个月", "6m", 183),
    ("近3个月", "3m", 93),
]


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def build_relative_choice(navs: pd.DataFrame):
    relative = navs[STORAGE] / navs[AI]
    fast_ret = relative.pct_change(FAST)
    slow_ret = relative.pct_change(SLOW)

    # True means storage is relatively stronger than AI.
    signal_storage = (fast_ret > 0) & (slow_ret > -0.02)
    streak_group = (signal_storage != signal_storage.shift()).cumsum()
    good = signal_storage.astype(int).groupby(streak_group).cumsum()
    bad = (~signal_storage).astype(int).groupby(streak_group).cumsum()

    current = AI
    choices: list[str] = []
    trades = []
    for date in navs.index:
        previous = current
        if current != STORAGE and good.loc[date] >= CONFIRM:
            current = STORAGE
        elif current != AI and bad.loc[date] >= CONFIRM:
            current = AI
        choices.append(current)
        if current != previous:
            trades.append(
                {
                    "日期": date,
                    "原目标": previous,
                    "新目标": current,
                    "相对净值_存储除以AI": relative.loc[date],
                    "15日相对变化": fast_ret.loc[date],
                    "40日相对变化": slow_ret.loc[date],
                    "动作": "切到存储/半导体代理" if current == STORAGE else "切到AI/人工智能",
                }
            )
    return relative, fast_ret, slow_ret, pd.Series(choices, index=navs.index), pd.DataFrame(trades)


def dca_returns(navs: pd.DataFrame, start=None) -> pd.Series:
    window = navs.copy()
    if start is not None:
        window = window.loc[window.index >= start]
    shares = {STORAGE: 0.0, AI: 0.0}
    invested = 0.0
    values = []
    for date, row in window.iterrows():
        contribution = 1.0
        invested += contribution
        shares[STORAGE] += contribution * 0.5 / row[STORAGE]
        shares[AI] += contribution * 0.5 / row[AI]
        value = shares[STORAGE] * row[STORAGE] + shares[AI] * row[AI]
        values.append((date, value / invested))
    return pd.Series(dict(values), name="每日定投50/50")


def backtest(navs: pd.DataFrame, choice: pd.Series, start=None):
    returns = navs.pct_change(fill_method=None)
    # Signal at T is known after T NAV; new exposure affects later intervals.
    position = choice.shift(2).fillna(AI)

    rows = []
    for i, date in enumerate(navs.index[1:], start=1):
        selected = position.loc[date]
        rotation_ret = returns.loc[date, selected]
        if pd.isna(rotation_ret):
            rotation_ret = 0.0
        if i >= 2 and choice.iloc[i - 1] != choice.iloc[i - 2]:
            rotation_ret -= COST
        rows.append(
            {
                "日期": date,
                "相对轮动策略": float(rotation_ret),
                "全仓存储": float(returns.loc[date, STORAGE]),
                "全仓AI": float(returns.loc[date, AI]),
                "等权买入持有": float(0.5 * returns.loc[date, STORAGE] + 0.5 * returns.loc[date, AI]),
            }
        )

    frame = pd.DataFrame(rows).set_index("日期").fillna(0.0)
    if start is not None:
        frame = frame.loc[frame.index >= start]
    curves = (1 + frame).cumprod()
    dca_curve = dca_returns(navs, start=start).reindex(curves.index).ffill()
    curves["每日定投50/50"] = dca_curve / dca_curve.iloc[0]

    metrics = []
    for column in curves.columns:
        metrics.append(
            {
                "策略": column,
                "收益率": float(curves[column].iloc[-1] - 1),
                "最大回撤": max_drawdown(curves[column]),
                "年化波动": float(frame[column].std() * np.sqrt(252)) if column in frame else np.nan,
            }
        )
    return frame, curves, pd.DataFrame(metrics)


def main() -> None:
    navs = pd.DataFrame({STORAGE: read_cached_nav(STORAGE), AI: read_cached_nav(AI)}).dropna().sort_index()
    relative, fast_ret, slow_ret, choice, trades = build_relative_choice(navs)

    pd.DataFrame(
        {
            "日期": navs.index,
            "存储基金净值": navs[STORAGE].values,
            "AI基金净值": navs[AI].values,
            "相对净值_存储除以AI": relative.values,
            "相对净值指数": (relative / relative.iloc[0]).values,
            "15日相对变化": fast_ret.values,
            "40日相对变化": slow_ret.values,
            "目标基金": choice.values,
        }
    ).to_csv(OUT / "storage_vs_ai_signal_detail_cn.csv", index=False, encoding="utf-8-sig")
    trades.to_csv(OUT / "storage_vs_ai_trade_points_cn.csv", index=False, encoding="utf-8-sig")

    summary_rows = []
    conclusion_rows = []
    for period_name, period_slug, days in PERIODS:
        start = None if days is None else navs.index[-1] - pd.Timedelta(days=days)
        _, curves, metrics = backtest(navs, choice, start=start)
        curves.to_csv(OUT / f"storage_vs_ai_{period_slug}_curves_cn.csv", encoding="utf-8-sig")
        for _, row in metrics.iterrows():
            summary_rows.append(
                {
                    "区间": period_name,
                    "起始日": curves.index[0].date(),
                    "结束日": curves.index[-1].date(),
                    **row.to_dict(),
                }
            )
        period_metrics = {row["策略"]: row for _, row in metrics.iterrows()}
        best_name = max(period_metrics, key=lambda key: period_metrics[key]["收益率"])
        conclusion_rows.append(
            {
                "区间": period_name,
                "最佳策略": best_name,
                "最佳收益率": period_metrics[best_name]["收益率"],
                "轮动收益率": period_metrics["相对轮动策略"]["收益率"],
                "轮动最大回撤": period_metrics["相对轮动策略"]["最大回撤"],
                "轮动相对全仓存储": period_metrics["相对轮动策略"]["收益率"] - period_metrics["全仓存储"]["收益率"],
                "轮动相对全仓AI": period_metrics["相对轮动策略"]["收益率"] - period_metrics["全仓AI"]["收益率"],
                "轮动相对等权持有": period_metrics["相对轮动策略"]["收益率"] - period_metrics["等权买入持有"]["收益率"],
                "轮动相对每日定投": period_metrics["相对轮动策略"]["收益率"] - period_metrics["每日定投50/50"]["收益率"],
            }
        )

        if period_slug in {"1y", "6m", "3m"}:
            fig, (ax1, ax2) = plt.subplots(
                2,
                1,
                figsize=(15, 9),
                sharex=True,
                gridspec_kw={"height_ratios": [2.1, 1]},
            )
            for column in curves.columns:
                ax1.plot(curves.index, curves[column], lw=1.7, label=column)
            window_trades = trades[(trades["日期"] >= curves.index[0]) & (trades["日期"] <= curves.index[-1])]
            if not window_trades.empty:
                strategy_curve = curves["相对轮动策略"]
                for _, trade in window_trades.iterrows():
                    idx = curves.index.searchsorted(trade["日期"])
                    if idx < len(curves.index):
                        x = curves.index[idx]
                        y = strategy_curve.iloc[idx]
                        marker = "^" if trade["新目标"] == STORAGE else "v"
                        color = "#d62728" if trade["新目标"] == STORAGE else "#2ca02c"
                        ax1.scatter([x], [y], marker=marker, s=80, color=color, edgecolor="white", linewidth=0.6, zorder=5)
            ax1.set_title(f"存储/半导体代理 与 AI 相互轮动：收益曲线对比（{period_name}）", fontsize=15)
            ax1.set_ylabel("累计净值（起点=1）")
            ax1.grid(True, alpha=0.25)
            ax1.legend(loc="best")

            relative_changes = pd.DataFrame({"15交易日相对变化": fast_ret * 100, "40交易日相对变化": slow_ret * 100}).loc[
                curves.index[0] : curves.index[-1]
            ]
            ax2.plot(relative_changes.index, relative_changes["15交易日相对变化"], color="#ff7f0e", lw=1.4, label="15交易日相对变化")
            ax2.plot(relative_changes.index, relative_changes["40交易日相对变化"], color="#9467bd", lw=1.4, label="40交易日相对变化")
            ax2.axhline(0, color="#333333", lw=0.8)
            ax2.axhline(-2, color="#9467bd", lw=0.8, ls="--", label="40日阈值 -2%")
            ax2.set_ylabel("存储相对AI变化（%）")
            ax2.set_xlabel("日期")
            ax2.grid(True, alpha=0.25)
            ax2.legend(loc="best")
            fig.tight_layout()
            fig.savefig(OUT / f"storage_vs_ai_{period_slug}_curve_cn.svg", format="svg")
            plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    conclusions = pd.DataFrame(conclusion_rows)
    summary.to_csv(OUT / "storage_vs_ai_backtest_summary_cn.csv", index=False, encoding="utf-8-sig")
    conclusions.to_csv(OUT / "storage_vs_ai_period_conclusions_cn.csv", index=False, encoding="utf-8-sig")

    latest = {
        "最新日期": navs.index[-1].date(),
        "相对净值": relative.iloc[-1],
        "15日相对变化": fast_ret.iloc[-1],
        "40日相对变化": slow_ret.iloc[-1],
        "当前目标": choice.iloc[-1],
        "切换次数": len(trades),
    }

    lines = [
        "# 存储与AI相互轮动回测",
        "",
        "## 规则",
        "",
        f"- 存储侧：`{STORAGE}` {STORAGE_NAME}。",
        f"- AI侧：`{AI}` {AI_NAME}。",
        "- 相对净值 = 存储基金单位净值 / AI基金单位净值。",
        "- 15交易日相对变化 > 0，且 40交易日相对变化 > -2%，连续2日确认后切到存储。",
        "- 条件连续不满足2日后切到AI。",
        "- 官方净值信号延迟执行；每次切换扣0.15%摩擦。",
        "- 对比基准：全仓存储、全仓AI、二者等权买入持有、二者每日等额定投50/50。",
        "",
        "## 最新信号",
        "",
        f"- 最新日期：{latest['最新日期']}",
        f"- 相对净值：{latest['相对净值']:.4f}",
        f"- 15日相对变化：{pct(latest['15日相对变化'])}",
        f"- 40日相对变化：{pct(latest['40日相对变化'])}",
        f"- 当前目标：{latest['当前目标']}",
        f"- 历史切换次数：{latest['切换次数']}",
        "",
        "## 回测摘要",
        "",
        "| 区间 | 策略 | 收益率 | 最大回撤 | 年化波动 |",
        "|---|---|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        vol = "" if pd.isna(row["年化波动"]) else pct(row["年化波动"])
        lines.append(f"| {row['区间']} | {row['策略']} | {pct(row['收益率'])} | {pct(row['最大回撤'])} | {vol} |")

    lines += ["", "## 轮动相对基准", "", "| 区间 | 最佳策略 | 轮动收益 | 轮动最大回撤 | 相对全仓存储 | 相对全仓AI | 相对等权持有 | 相对每日定投 |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for _, row in conclusions.iterrows():
        lines.append(
            f"| {row['区间']} | {row['最佳策略']} | {pct(row['轮动收益率'])} | {pct(row['轮动最大回撤'])} | "
            f"{pct(row['轮动相对全仓存储'])} | {pct(row['轮动相对全仓AI'])} | {pct(row['轮动相对等权持有'])} | {pct(row['轮动相对每日定投'])} |"
        )

    lines += [
        "",
        "## 初步判断",
        "",
        "这组测试更接近“两个主题互相抢仓位”。如果轮动胜出，说明策略确实识别了存储和AI之间的相对强弱；如果只是在某段追上其中一边，说明它仍然是风格过滤器，不是稳定alpha。",
    ]
    (OUT / "storage_vs_ai_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
