from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
DEPS = ROOT / ".deps"
if DEPS.exists() and str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.backtest_pcb_purchase_limit import (  # noqa: E402
    PCB_DAILY_LIMIT,
    all_in_pcb_choice,
    simulate,
    simulate_weighted_targets,
)
from tools.optimize_rotation_with_pcb_limit import (  # noqa: E402
    choice_for_spec,
    optimization_specs,
)
from tools.pcb_limit_core_satellite_strategy import (  # noqa: E402
    MODEL_NAME,
    PCB_CORE_WEIGHT,
    setup_chinese_font,
    target_weights,
)
from tools.sector_level_rotation_5themes import load_sector_navs  # noqa: E402


OUT = ROOT / "output" / "pcb_limit_core_satellite"
WINDOW_COUNT = 30
TRADING_DAYS_PER_MONTH = 21


def metric_values(metrics: dict[str, float]) -> tuple[float, float, float]:
    """Return total return, max drawdown and redemption fee by stable insertion order."""
    values = list(metrics.values())
    fee = float(values[7]) if len(values) > 7 else 0.0
    return float(values[0]), float(values[1]), fee


def fixed_initial_satellite_weights(
    navs: pd.DataFrame,
    tactical_choice: pd.Series,
) -> pd.DataFrame:
    initial_sector = str(tactical_choice.reindex(navs.index).ffill().iloc[0])
    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    weights["PCB"] = PCB_CORE_WEIGHT
    weights[initial_sector] += 1.0 - PCB_CORE_WEIGHT
    return weights


def effective_sample_size(values: pd.Series) -> tuple[float, float]:
    rho = float(values.autocorr(lag=1)) if len(values) > 2 else np.nan
    if not np.isfinite(rho):
        return rho, float(len(values))
    rho = min(max(rho, -0.99), 0.99)
    n_eff = len(values) * (1.0 - rho) / (1.0 + rho)
    return rho, float(min(len(values), max(1.0, n_eff)))


def evaluate_window(
    navs: pd.DataFrame,
    tactical_choice: pd.Series,
    start_pos: int,
    end_pos: int,
) -> dict[str, float | str | bool]:
    test = navs.iloc[start_pos : end_pos + 1]
    choice = tactical_choice.reindex(test.index).ffill()

    rotation_ledger, rotation_metrics = simulate_weighted_targets(
        test,
        target_weights(test, choice),
    )
    static_ledger, static_metrics = simulate_weighted_targets(
        test,
        fixed_initial_satellite_weights(test, choice),
    )
    all_in_ledger, all_in_metrics = simulate(
        test,
        all_in_pcb_choice(test.index),
        PCB_DAILY_LIMIT,
    )

    rotation_return, rotation_mdd, rotation_fee = metric_values(rotation_metrics)
    static_return, static_mdd, static_fee = metric_values(static_metrics)
    all_in_return, all_in_mdd, _ = metric_values(all_in_metrics)
    pcb_raw_return = float(test["PCB"].iloc[-1] / test["PCB"].iloc[0] - 1.0)

    return {
        "开始日期": test.index[0].date().isoformat(),
        "结束日期": test.index[-1].date().isoformat(),
        "轮动收益率": rotation_return,
        "固定核心卫星收益率": static_return,
        "限购全仓PCB收益率": all_in_return,
        "轮动相对全仓超额": rotation_return - all_in_return,
        "轮动相对固定配置超额": rotation_return - static_return,
        "固定配置相对全仓超额": static_return - all_in_return,
        "轮动最大回撤": rotation_mdd,
        "固定核心卫星最大回撤": static_mdd,
        "限购全仓PCB最大回撤": all_in_mdd,
        "轮动赎回费占初始资金": rotation_fee,
        "固定配置赎回费占初始资金": static_fee,
        "PCB区间原始涨幅": pcb_raw_return,
        "轮动胜出全仓": rotation_return > all_in_return,
        "轮动胜出固定配置": rotation_return > static_return,
        "轮动期末净值": float(rotation_ledger.iloc[-1, 0] / rotation_ledger.iloc[0, 0]),
        "固定配置期末净值": float(static_ledger.iloc[-1, 0] / static_ledger.iloc[0, 0]),
        "全仓PCB期末净值": float(all_in_ledger.iloc[-1, 0] / all_in_ledger.iloc[0, 0]),
    }


def summarize_month(
    navs: pd.DataFrame,
    tactical_choice: pd.Series,
    months: int,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    window_days = months * TRADING_DAYS_PER_MONTH
    endpoints = list(range(len(navs) - WINDOW_COUNT, len(navs)))
    rows = []
    for window_id, end_pos in enumerate(endpoints, start=1):
        start_pos = end_pos - window_days + 1
        row = evaluate_window(navs, tactical_choice, start_pos, end_pos)
        row.update({"月数": months, "窗口": window_id, "窗口交易日数": window_days})
        rows.append(row)
    detail = pd.DataFrame(rows)
    excess = detail["轮动相对全仓超额"]
    rho, n_eff = effective_sample_size(excess)

    non_overlap_endpoints = list(range(len(navs) - 1, window_days - 2, -window_days))
    non_overlap_rows = [
        evaluate_window(navs, tactical_choice, end_pos - window_days + 1, end_pos)
        for end_pos in reversed(non_overlap_endpoints)
    ]
    non_overlap = pd.DataFrame(non_overlap_rows)

    summary = {
        "月数": months,
        "相邻窗口胜出数": int(detail["轮动胜出全仓"].sum()),
        "相邻窗口总数": len(detail),
        "相邻窗口胜率": float(detail["轮动胜出全仓"].mean()),
        "相邻窗口理论重叠率": (window_days - 1) / window_days,
        "超额收益一阶自相关": rho,
        "估算有效样本数": n_eff,
        "平均轮动收益率": float(detail["轮动收益率"].mean()),
        "平均全仓PCB收益率": float(detail["限购全仓PCB收益率"].mean()),
        "平均轮动相对全仓超额": float(excess.mean()),
        "平均轮动相对固定配置超额": float(detail["轮动相对固定配置超额"].mean()),
        "平均固定配置相对全仓超额": float(detail["固定配置相对全仓超额"].mean()),
        "平均轮动最大回撤": float(detail["轮动最大回撤"].mean()),
        "平均全仓PCB最大回撤": float(detail["限购全仓PCB最大回撤"].mean()),
        "平均轮动赎回费": float(detail["轮动赎回费占初始资金"].mean()),
        "超额与PCB涨幅相关系数": float(excess.corr(detail["PCB区间原始涨幅"])),
        "非重叠窗口胜出数": int(non_overlap["轮动胜出全仓"].sum()),
        "非重叠窗口总数": len(non_overlap),
        "非重叠窗口胜率": float(non_overlap["轮动胜出全仓"].mean()),
        "非重叠窗口平均超额": float(non_overlap["轮动相对全仓超额"].mean()),
        "相邻窗口覆盖开始": str(detail["开始日期"].min()),
        "相邻窗口覆盖结束": str(detail["结束日期"].max()),
    }
    return detail, summary


def model_provenance() -> dict[str, float | int | str]:
    ranking_path = ROOT / "output" / "pcb_limit_strategy_optimization" / "训练模型排名_cn.csv"
    ranking = pd.read_csv(ranking_path)
    names = ranking.iloc[:, 0].astype(str)
    matches = np.flatnonzero(names.to_numpy() == MODEL_NAME)
    if len(matches) == 0:
        return {"模型": MODEL_NAME, "训练排名": "未找到"}
    pos = int(matches[0])
    row = ranking.iloc[pos]
    return {
        "模型": MODEL_NAME,
        "候选模型数": len(ranking),
        "训练排名": pos + 1,
        "训练一月胜出": int(row.iloc[3]),
        "训练三月胜出": int(row.iloc[4]),
        "训练一月平均超额": float(row.iloc[5]),
        "训练三月平均超额": float(row.iloc[6]),
        "训练评分": float(row.iloc[9]),
    }


def make_chart(summary: pd.DataFrame) -> None:
    setup_chinese_font()
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), dpi=160, sharex=True)
    x = summary["月数"]
    axes[0].plot(x, summary["相邻窗口胜率"], marker="o", linewidth=2.2, label="30个相邻窗口胜率")
    axes[0].plot(x, summary["非重叠窗口胜率"], marker="s", linewidth=2.0, label="全历史非重叠窗口胜率")
    axes[0].axhline(0.5, color="gray", linestyle="--", linewidth=1.0, label="50%")
    axes[0].set_ylabel("胜率")
    axes[0].set_ylim(-0.03, 1.03)
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    axes[1].plot(
        x,
        summary["平均轮动相对全仓超额"] * 100,
        marker="o",
        linewidth=2.2,
        label="轮动策略 - 全仓PCB",
    )
    axes[1].plot(
        x,
        summary["平均轮动相对固定配置超额"] * 100,
        marker="s",
        linewidth=2.0,
        label="轮动本身 - 固定核心卫星",
    )
    axes[1].plot(
        x,
        summary["平均固定配置相对全仓超额"] * 100,
        marker="^",
        linewidth=2.0,
        label="固定核心卫星 - 全仓PCB",
    )
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("窗口长度（月）")
    axes[1].set_ylabel("平均超额收益（百分点）")
    axes[1].grid(alpha=0.25)
    axes[1].legend()
    fig.suptitle("PCB核心卫星策略：窗口胜率与收益归因审计")
    fig.tight_layout()
    fig.savefig(OUT / "一至八个月泛化与过拟合审计_cn.png")
    plt.close(fig)


def latest_eight_month_monthly_attribution(
    navs: pd.DataFrame,
    tactical_choice: pd.Series,
) -> pd.DataFrame:
    window_days = 8 * TRADING_DAYS_PER_MONTH
    test = navs.iloc[-window_days:]
    choice = tactical_choice.reindex(test.index).ffill()
    rotation_ledger, _ = simulate_weighted_targets(test, target_weights(test, choice))
    static_ledger, _ = simulate_weighted_targets(
        test,
        fixed_initial_satellite_weights(test, choice),
    )
    all_in_ledger, _ = simulate(test, all_in_pcb_choice(test.index), PCB_DAILY_LIMIT)

    curves = pd.DataFrame(
        {
            "轮动策略": rotation_ledger.iloc[:, 0],
            "固定核心卫星": static_ledger.iloc[:, 0],
            "限购全仓PCB": all_in_ledger.iloc[:, 0],
        }
    )
    daily_returns = curves.pct_change(fill_method=None).fillna(0.0)
    monthly_returns = (1.0 + daily_returns).groupby(daily_returns.index.to_period("M")).prod() - 1.0
    monthly_returns.index = monthly_returns.index.astype(str)

    pcb_weight_column = next(
        column for column in rotation_ledger.columns if str(column).startswith("权重_PCB")
    )
    avg_pcb_weight = rotation_ledger[pcb_weight_column].groupby(
        rotation_ledger.index.to_period("M")
    ).mean()
    avg_pcb_weight.index = avg_pcb_weight.index.astype(str)

    fee_column = next(
        column for column in rotation_ledger.columns if "累计赎回费" in str(column)
    )
    month_end_fee = rotation_ledger[fee_column].groupby(
        rotation_ledger.index.to_period("M")
    ).last()
    monthly_fee = month_end_fee.diff().fillna(month_end_fee.iloc[0]) / 10000.0
    monthly_fee.index = monthly_fee.index.astype(str)

    executed_choice = choice.shift(2).fillna(choice.iloc[0])
    non_pcb_ratio = executed_choice.ne("PCB").groupby(executed_choice.index.to_period("M")).mean()
    non_pcb_ratio.index = non_pcb_ratio.index.astype(str)

    result = monthly_returns.reset_index(names="月份")
    result["轮动相对全仓超额"] = result["轮动策略"] - result["限购全仓PCB"]
    result["轮动相对固定配置超额"] = result["轮动策略"] - result["固定核心卫星"]
    result["平均实际PCB权重"] = result["月份"].map(avg_pcb_weight)
    result["卫星信号非PCB比例"] = result["月份"].map(non_pcb_ratio)
    result["当月赎回费占初始资金"] = result["月份"].map(monthly_fee)
    return result


def make_report(
    summary: pd.DataFrame,
    provenance: dict[str, float | int | str],
    monthly: pd.DataFrame,
) -> None:
    one = summary.loc[summary["月数"] == 1].iloc[0]
    three = summary.loc[summary["月数"] == 3].iloc[0]
    eight = summary.loc[summary["月数"] == 8].iloc[0]
    worst_months = monthly.nsmallest(3, "轮动相对全仓超额")
    worst_text = "、".join(
        f"{row['月份']}（{row['轮动相对全仓超额']:.2%}）"
        for _, row in worst_months.iterrows()
    )
    report = f"""# PCB核心卫星策略泛化与过拟合审计

## 结论

当前一月、三月的高胜率不能解释为策略已经稳定泛化。主要原因是参数选择后的测试复用，以及30个窗口之间高度重叠。模型在前置训练区表现很差，却在最近一段行情中突然胜出；这更符合阶段性行情适配，而不是稳定Alpha。

## 参数来源审计

- 固定模型：`{provenance['模型']}`
- 候选模型数：{provenance.get('候选模型数', '未知')}
- 前置训练排名：{provenance.get('训练排名', '未知')}
- 前置训练一月胜出：{provenance.get('训练一月胜出', '未知')}/10，平均超额：{float(provenance.get('训练一月平均超额', np.nan)):.2%}
- 前置训练三月胜出：{provenance.get('训练三月胜出', '未知')}/10，平均超额：{float(provenance.get('训练三月平均超额', np.nan)):.2%}

这意味着该模型并不是由前置训练排名选出的。若它是因为最近30窗口表现好而被改选，那么最近30窗口已经参与选模，不能继续叫作锁定测试。

## 为什么短周期看起来好

1. 一月相邻窗口理论重叠率为 {one['相邻窗口理论重叠率']:.1%}，三月为 {three['相邻窗口理论重叠率']:.1%}。相邻窗口只移动一个交易日，绝大部分收益路径完全相同。
2. 一月30窗口实际只覆盖 {one['相邻窗口覆盖开始']} 至 {one['相邻窗口覆盖结束']}；三月也集中在同一轮PCB强势行情。它们不是30种独立市场环境。
3. 一月超额序列的一阶自相关为 {one['超额收益一阶自相关']:.2f}，估算有效样本仅 {one['估算有效样本数']:.1f}；三月分别为 {three['超额收益一阶自相关']:.2f} 和 {three['估算有效样本数']:.1f}。
4. 80% PCB核心使策略与全仓PCB非常接近。短窗口里的微小胜出容易来自起始两三天的限购现金差异、20%卫星的偶然表现及窗口端点，而不全是持续轮动Alpha。

## 为什么时间拉长后失效

窗口越长，20%卫星偏离PCB所造成的复利差异会累积。在PCB持续主升浪中，卫星没有稳定跑赢PCB，轮动费率也持续扣减，因此长期不是“更不容易错过主升浪”，而是“更长时间维持低于100%的PCB暴露”。

八个月相邻窗口的平均总超额为 {eight['平均轮动相对全仓超额']:.2%}；其中真正由动态轮动相对固定80/20配置带来的平均超额为 {eight['平均轮动相对固定配置超额']:.2%}，固定80/20配置相对全仓PCB的平均差异为 {eight['平均固定配置相对全仓超额']:.2%}，平均赎回费为 {eight['平均轮动赎回费']:.2%}。

最新八个月路径中，拖累最大的三个月是：{worst_text}。结合实际PCB权重与卫星信号可判断，长期失效的主因是卫星仓在PCB主升阶段持续偏离，费用是次要但确定的负贡献。

## 正确解读

- 是，存在明显的过拟合风险；更准确地说，是测试集复用和单一行情阶段选择偏差。
- 30个逐日滑动窗口适合观察入场日敏感性，不适合当作30次独立胜负投票。
- 当前模型应降级为研究候选，不应仅凭20/30与23/30直接用于实盘加仓。
- 下一版优化必须预先固定规则，只在更早数据调参；最近阶段只验一次，并同时报告非重叠窗口和固定核心卫星对照。

## 汇总表

{summary.to_markdown(index=False, floatfmt='.4f')}

## 最新八个月月度归因

{monthly.to_markdown(index=False, floatfmt='.4f')}
"""
    (OUT / "PCB核心卫星策略泛化与过拟合审计_cn.md").write_text(report, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    navs = load_sector_navs()
    spec = next(item for item in optimization_specs() if item.name == MODEL_NAME)
    tactical_choice = choice_for_spec(navs, spec)

    details = []
    summaries = []
    for months in range(1, 9):
        detail, summary = summarize_month(navs, tactical_choice, months)
        details.append(detail)
        summaries.append(summary)

    detail_frame = pd.concat(details, ignore_index=True)
    summary_frame = pd.DataFrame(summaries)
    provenance = model_provenance()
    monthly = latest_eight_month_monthly_attribution(navs, tactical_choice)
    pd.DataFrame([provenance]).to_csv(
        OUT / "模型参数来源审计_cn.csv", index=False, encoding="utf-8-sig"
    )
    detail_frame.to_csv(OUT / "一至八个月30窗口审计明细_cn.csv", index=False, encoding="utf-8-sig")
    summary_frame.to_csv(OUT / "一至八个月泛化审计汇总_cn.csv", index=False, encoding="utf-8-sig")
    monthly.to_csv(OUT / "最新八个月月度收益归因_cn.csv", index=False, encoding="utf-8-sig")
    make_chart(summary_frame)
    make_report(summary_frame, provenance, monthly)
    print(summary_frame.to_string(index=False))


if __name__ == "__main__":
    main()
