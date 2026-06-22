from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from quant_fund_advisor.style_timing import read_cached_nav


OUT = Path("output/relative_rotation_storage_ai")
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

FAST = 15
SLOW = 40
CONFIRM = 2
COST = 0.0015

PAIRS = [
    {
        "asset": "存储/半导体代理",
        "slug": "storage_semiconductor",
        "leader": "008887",
        "leader_name": "华夏国证半导体芯片ETF联接A（008887）",
        "anchor": "011370",
        "anchor_name": "华商均衡成长混合C（011370）",
    },
    {
        "asset": "AI/人工智能",
        "slug": "ai",
        "leader": "008585",
        "leader_name": "华夏中证人工智能主题ETF联接A（008585）",
        "anchor": "011370",
        "anchor_name": "华商均衡成长混合C（011370）",
    },
]

PERIODS = [
    ("全共同历史", "full", None),
    ("近1年", "1y", 365),
    ("近6个月", "6m", 183),
    ("近3个月", "3m", 93),
]


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def build_choice(navs: pd.DataFrame, leader: str, anchor: str):
    rel = navs[leader] / navs[anchor]
    fast_ret = rel.pct_change(FAST)
    slow_ret = rel.pct_change(SLOW)
    signal = (fast_ret > 0) & (slow_ret > -0.02)
    streak_group = (signal != signal.shift()).cumsum()
    good = signal.astype(int).groupby(streak_group).cumsum()
    bad = (~signal).astype(int).groupby(streak_group).cumsum()

    current = anchor
    choices: list[str] = []
    actions = []
    for date in navs.index:
        previous = current
        if current != leader and good.loc[date] >= CONFIRM:
            current = leader
        elif current != anchor and bad.loc[date] >= CONFIRM:
            current = anchor
        choices.append(current)
        if current != previous:
            actions.append(
                {
                    "日期": date,
                    "原目标": previous,
                    "新目标": current,
                    "相对净值": rel.loc[date],
                    "15日相对变化": fast_ret.loc[date],
                    "40日相对变化": slow_ret.loc[date],
                    "动作": "买入/切到进攻基金" if current == leader else "卖出进攻基金/切回锚基金",
                }
            )
    return rel, fast_ret, slow_ret, pd.Series(choices, index=navs.index), pd.DataFrame(actions)


def backtest(navs: pd.DataFrame, choice: pd.Series, leader: str, anchor: str, start):
    returns = navs.pct_change(fill_method=None)
    # choice at T is only known after T NAV. The switch executes at T+1 NAV,
    # and the new holding affects the next interval, so use a two-NAV lag.
    position = choice.shift(2).fillna(anchor)
    rows = []
    for i, date in enumerate(navs.index[1:], start=1):
        selected = position.loc[date]
        ret = returns.loc[date, selected]
        if pd.isna(ret):
            ret = 0.0
        if i >= 2 and choice.iloc[i - 1] != choice.iloc[i - 2]:
            ret -= COST
        rows.append(
            {
                "日期": date,
                "相对轮动策略": float(ret),
                f"{leader}买入持有": float(returns.loc[date, leader]),
                f"{anchor}买入持有": float(returns.loc[date, anchor]),
            }
        )
    frame = pd.DataFrame(rows).set_index("日期").fillna(0.0)
    if start is not None:
        frame = frame.loc[frame.index >= start]
    curves = (1 + frame).cumprod()
    metrics = []
    for column in curves.columns:
        metrics.append(
            {
                "策略": column,
                "收益率": float(curves[column].iloc[-1] - 1),
                "最大回撤": max_drawdown(curves[column]),
                "年化波动": float(frame[column].std() * np.sqrt(252)),
                "日胜率": float((frame[column] > 0).mean()),
            }
        )
    return frame, curves, pd.DataFrame(metrics)


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def main() -> None:
    summary_rows = []
    latest_rows = []
    conclusion_rows = []

    for pair in PAIRS:
        leader = pair["leader"]
        anchor = pair["anchor"]
        navs = pd.DataFrame({leader: read_cached_nav(leader), anchor: read_cached_nav(anchor)}).dropna().sort_index()
        rel, fast_ret, slow_ret, choice, trades = build_choice(navs, leader, anchor)

        trades.to_csv(OUT / f"{pair['slug']}_trade_points_cn.csv", index=False, encoding="utf-8-sig")
        detail = pd.DataFrame(
            {
                "日期": navs.index,
                "进攻基金净值": navs[leader].values,
                "锚基金净值": navs[anchor].values,
                "相对净值": rel.values,
                "相对净值指数": (rel / rel.iloc[0]).values,
                "15日相对变化": fast_ret.values,
                "40日相对变化": slow_ret.values,
                "目标基金": choice.values,
            }
        )
        detail.to_csv(OUT / f"{pair['slug']}_signal_detail_cn.csv", index=False, encoding="utf-8-sig")

        latest_rows.append(
            {
                "板块": pair["asset"],
                "最新日期": navs.index[-1].date(),
                "相对净值": rel.iloc[-1],
                "15日相对变化": fast_ret.iloc[-1],
                "40日相对变化": slow_ret.iloc[-1],
                "当前目标": choice.iloc[-1],
                "切换次数": len(trades),
            }
        )

        for period_name, period_slug, days in PERIODS:
            start = None if days is None else navs.index[-1] - pd.Timedelta(days=days)
            _, curves, metrics = backtest(navs, choice, leader, anchor, start)
            for _, metric in metrics.iterrows():
                summary_rows.append(
                    {
                        "板块": pair["asset"],
                        "进攻基金代码": leader,
                        "锚基金代码": anchor,
                        "区间": period_name,
                        "起始日": curves.index[0].date(),
                        "结束日": curves.index[-1].date(),
                        **metric.to_dict(),
                    }
                )

            if period_slug in {"1y", "6m"}:
                fig, (ax1, ax2) = plt.subplots(
                    2,
                    1,
                    figsize=(15, 9),
                    sharex=True,
                    gridspec_kw={"height_ratios": [2.1, 1]},
                )
                for column in curves.columns:
                    ax1.plot(curves.index, curves[column], lw=1.8, label=column)
                window_trades = trades[(trades["日期"] >= curves.index[0]) & (trades["日期"] <= curves.index[-1])]
                if not window_trades.empty:
                    strategy_curve = curves["相对轮动策略"]
                    for _, trade in window_trades.iterrows():
                        idx = curves.index.searchsorted(trade["日期"])
                        if idx < len(curves.index):
                            x = curves.index[idx]
                            y = strategy_curve.iloc[idx]
                            if trade["新目标"] == leader:
                                ax1.scatter([x], [y], marker="^", s=85, color="#d62728", edgecolor="white", linewidth=0.6, zorder=5)
                            else:
                                ax1.scatter([x], [y], marker="v", s=85, color="#2ca02c", edgecolor="white", linewidth=0.6, zorder=5)
                ax1.set_title(f"{pair['asset']}：相对强弱轮动 vs 买入持有（{period_name}）", fontsize=15)
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
                ax2.set_ylabel("相对变化（%）")
                ax2.set_xlabel("日期")
                ax2.grid(True, alpha=0.25)
                ax2.legend(loc="best")
                fig.tight_layout()
                fig.savefig(OUT / f"{pair['slug']}_{period_slug}_curve_cn.svg", format="svg")
                plt.close(fig)

        for period_name in ["近1年", "近6个月", "近3个月"]:
            block = [row for row in summary_rows if row["板块"] == pair["asset"] and row["区间"] == period_name]
            rotation = next(row for row in block if row["策略"] == "相对轮动策略")
            leader_hold = next(row for row in block if row["策略"] == f"{leader}买入持有")
            anchor_hold = next(row for row in block if row["策略"] == f"{anchor}买入持有")
            conclusion_rows.append(
                {
                    "板块": pair["asset"],
                    "区间": period_name,
                    "轮动收益": rotation["收益率"],
                    "进攻买入持有收益": leader_hold["收益率"],
                    "锚基金买入持有收益": anchor_hold["收益率"],
                    "轮动相对进攻基金超额": rotation["收益率"] - leader_hold["收益率"],
                    "轮动相对锚基金超额": rotation["收益率"] - anchor_hold["收益率"],
                    "轮动最大回撤": rotation["最大回撤"],
                }
            )

    latest = pd.DataFrame(latest_rows)
    summary = pd.DataFrame(summary_rows)
    conclusions = pd.DataFrame(conclusion_rows)
    latest.to_csv(OUT / "latest_signal_cn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "backtest_summary_cn.csv", index=False, encoding="utf-8-sig")
    conclusions.to_csv(OUT / "period_conclusions_cn.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 存储与AI板块相对强弱轮动回测",
        "",
        "## 规则",
        "",
        "本次没有重新调参，直接沿用 CPO 那套相对强弱规则：",
        "",
        "- 相对净值 = 进攻基金单位净值 / 锚基金单位净值。",
        "- 15交易日相对变化 > 0，且 40交易日相对变化 > -2%。",
        "- 条件连续满足 2 个交易日后切到进攻基金；连续不满足 2 个交易日后切回锚基金。",
        "- 用官方净值形成信号，新仓位从下一次净值区间开始生效；切换日仍享受旧持仓当日涨跌。",
        "- 每次切换扣 0.15% 摩擦成本。",
        "",
        "## 基金选择",
        "",
    ]
    for pair in PAIRS:
        lines.append(f"- {pair['asset']}：进攻基金 `{pair['leader']}` {pair['leader_name']}；锚基金 `{pair['anchor']}` {pair['anchor_name']}。")

    lines += [
        "",
        "## 最新信号",
        "",
        "| 板块 | 最新日期 | 相对净值 | 15日相对变化 | 40日相对变化 | 当前目标 | 切换次数 |",
        "|---|---:|---:|---:|---:|---|---:|",
    ]
    for _, row in latest.iterrows():
        lines.append(
            f"| {row['板块']} | {row['最新日期']} | {row['相对净值']:.4f} | {pct(row['15日相对变化'])} | {pct(row['40日相对变化'])} | {row['当前目标']} | {int(row['切换次数'])} |"
        )

    lines += ["", "## 回测摘要", ""]
    for asset in summary["板块"].unique():
        lines += [
            f"### {asset}",
            "",
            "| 区间 | 策略 | 收益率 | 最大回撤 | 年化波动 |",
            "|---|---|---:|---:|---:|",
        ]
        for _, row in summary[summary["板块"] == asset].iterrows():
            lines.append(f"| {row['区间']} | {row['策略']} | {pct(row['收益率'])} | {pct(row['最大回撤'])} | {pct(row['年化波动'])} |")
        lines.append("")

    lines += ["## 初步结论", ""]
    for _, row in conclusions.iterrows():
        lines.append(
            f"- {row['板块']} {row['区间']}：轮动收益 {pct(row['轮动收益'])}，进攻基金买入持有 {pct(row['进攻买入持有收益'])}，"
            f"锚基金买入持有 {pct(row['锚基金买入持有收益'])}；相对进攻基金超额 {pct(row['轮动相对进攻基金超额'])}，"
            f"相对锚基金超额 {pct(row['轮动相对锚基金超额'])}，轮动最大回撤 {pct(row['轮动最大回撤'])}。"
        )
    lines += [
        "",
        "这次迁移的结论偏谨慎：相对强弱轮动能在部分区间打赢对应主题进攻基金，但没有打赢这段时间极强的锚基金 011370。"
        "因此它更像一个主题强弱过滤器，而不是一个可以无脑套到所有科技主题上的最终模型。",
    ]
    (OUT / "relative_rotation_storage_ai_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
