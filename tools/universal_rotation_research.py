from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from quant_fund_advisor.style_timing import read_cached_nav


OUT = Path("output/universal_rotation_research")
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

COST = 0.0015
ANCHOR = "011370"

FUNDS = {
    "007817": "CPO/通信设备：国泰中证全指通信设备ETF联接A",
    "008887": "存储/半导体代理：华夏国证半导体芯片ETF联接A",
    "008585": "AI/人工智能：华夏中证人工智能主题ETF联接A",
    "720001": "PCB代理/制造成长：财通价值动量混合",
    "006503": "半导体主动：财通集成电路产业股票C",
    "011370": "科技成长锚：华商均衡成长混合C",
}

PERIODS = [
    ("全共同历史", "full", None),
    ("近1年", "1y", 365),
    ("近6个月", "6m", 183),
    ("近3个月", "3m", 93),
]


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def trend_quality(values: pd.Series) -> float:
    values = np.log(values.dropna().to_numpy(dtype=float))
    if len(values) < 30 or not np.isfinite(values).all():
        return 0.0
    x = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(x, values, 1)
    fitted = slope * x + intercept
    total = ((values - values.mean()) ** 2).sum()
    resid = ((values - fitted) ** 2).sum()
    r2 = 1 - resid / total if total > 0 else 0.0
    return float(np.expm1(slope * 252) * max(0.0, r2))


def zscore_cross_section(frame: pd.DataFrame) -> pd.DataFrame:
    mean = frame.mean(axis=1)
    std = frame.std(axis=1).replace(0, np.nan)
    return frame.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


def load_navs() -> pd.DataFrame:
    navs = pd.DataFrame({code: read_cached_nav(code) for code in FUNDS})
    return navs.dropna().sort_index()


def build_features(navs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    returns = navs.pct_change(fill_method=None)
    ret20 = navs.pct_change(20, fill_method=None)
    ret60 = navs.pct_change(60, fill_method=None)
    ret120 = navs.pct_change(120, fill_method=None)
    vol20 = returns.rolling(20).std() * np.sqrt(252)
    dd20 = navs / navs.rolling(20).max() - 1
    ma20 = navs.rolling(20).mean()
    ma60 = navs.rolling(60).mean()
    tq60 = navs.rolling(60).apply(lambda x: trend_quality(pd.Series(x)), raw=False)
    sharpe20 = ret20 / (vol20.replace(0, np.nan) / np.sqrt(12))
    return {
        "ret20": ret20,
        "ret60": ret60,
        "ret120": ret120,
        "vol20": vol20,
        "dd20": dd20,
        "ma20": ma20,
        "ma60": ma60,
        "tq60": tq60,
        "sharpe20": sharpe20,
    }


def confirmed_choice(raw_choice: pd.Series, confirm: int = 2) -> pd.Series:
    current = raw_choice.iloc[0]
    pending = current
    count = 0
    out = []
    for value in raw_choice:
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
        out.append(current)
    return pd.Series(out, index=raw_choice.index)


def top1_weights(choice: pd.Series, columns: list[str]) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=choice.index, columns=columns)
    for date, code in choice.items():
        weights.loc[date, code] = 1.0
    return weights


def build_strategy_weights(navs: pd.DataFrame, features: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    columns = list(navs.columns)
    ret20 = features["ret20"]
    ret60 = features["ret60"]
    vol20 = features["vol20"]
    dd20 = features["dd20"]
    ma20 = features["ma20"]
    ma60 = features["ma60"]
    tq60 = features["tq60"]
    sharpe20 = features["sharpe20"]

    # 1. Absolute + relative momentum: choose the strongest 60d momentum fund only if
    # its own 20d trend is positive and NAV is above MA20; otherwise use the anchor.
    raw_dual = []
    for date in navs.index:
        eligible = (ret20.loc[date] > 0) & (navs.loc[date] > ma20.loc[date])
        ranked = ret60.loc[date].where(eligible).dropna()
        if not ranked.empty:
            raw_dual.append(ranked.idxmax())
        else:
            raw_dual.append(ANCHOR)
    dual_choice = confirmed_choice(pd.Series(raw_dual, index=navs.index), confirm=2)

    # 2. Multi-factor Top1: cross-sectional score with momentum, trend quality,
    # risk-adjusted momentum, volatility penalty and drawdown penalty.
    score = (
        0.30 * zscore_cross_section(ret20)
        + 0.35 * zscore_cross_section(ret60)
        + 0.15 * zscore_cross_section(sharpe20)
        + 0.15 * zscore_cross_section(tq60)
        - 0.20 * zscore_cross_section(vol20)
        + 0.15 * zscore_cross_section(dd20)
    )
    raw_mf = []
    for date in navs.index:
        eligible = (ret20.loc[date] > -0.02) & (navs.loc[date] > ma60.loc[date] * 0.96)
        ranked = score.loc[date].where(eligible).dropna()
        if not ranked.empty:
            raw_mf.append(ranked.idxmax())
        else:
            raw_mf.append(ANCHOR)
    mf_choice = confirmed_choice(pd.Series(raw_mf, index=navs.index), confirm=2)

    # 3. Multi-factor Top2: split between the two best positive-score funds.
    # This is less heroic than Top1 and usually more robust across themes.
    mf2 = pd.DataFrame(0.0, index=navs.index, columns=columns)
    for date in navs.index:
        eligible = ((ret20.loc[date] > -0.02) & (navs.loc[date] > ma60.loc[date] * 0.96)).fillna(False)
        ranked = score.loc[date].where(eligible).dropna().sort_values(ascending=False)
        ranked = ranked[ranked > 0]
        if len(ranked) == 0:
            mf2.loc[date, ANCHOR] = 1.0
        elif len(ranked) == 1:
            mf2.loc[date, ranked.index[0]] = 1.0
        else:
            mf2.loc[date, ranked.index[:2]] = [0.6, 0.4]

    # 4. Volatility managed multi-factor Top2: same selection, but if 20d average
    # volatility of selected funds is above 45%, keep 25% in anchor.
    vmf2 = mf2.copy()
    for date in navs.index:
        selected = vmf2.loc[date][vmf2.loc[date] > 0].index
        avg_vol = vol20.loc[date, selected].mean() if len(selected) else np.nan
        if pd.notna(avg_vol) and avg_vol > 0.45 and ANCHOR not in selected:
            vmf2.loc[date] *= 0.75
            vmf2.loc[date, ANCHOR] += 0.25

    return {
        "绝对相对动量Top1": top1_weights(dual_choice, columns),
        "多因子Top1": top1_weights(mf_choice, columns),
        "多因子Top2分散": mf2,
        "波动管理Top2": vmf2,
    }


def lag_and_backtest(navs: pd.DataFrame, raw_weights: pd.DataFrame, start=None) -> tuple[pd.Series, pd.DataFrame]:
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    weights = raw_weights.shift(2).fillna(0.0)
    if weights.sum(axis=1).eq(0).any():
        weights.loc[weights.sum(axis=1).eq(0), ANCHOR] = 1.0
    gross = (weights * returns).sum(axis=1)
    turnover = raw_weights.diff().abs().sum(axis=1) / 2
    cost = turnover.shift(1).fillna(0.0) * COST
    strategy_ret = gross - cost
    if start is not None:
        strategy_ret = strategy_ret.loc[strategy_ret.index >= start]
        weights = weights.loc[weights.index >= start]
    curve = (1 + strategy_ret).cumprod()
    return curve, weights


def equal_weight_curve(navs: pd.DataFrame, start=None) -> pd.Series:
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    if start is not None:
        returns = returns.loc[returns.index >= start]
    return (1 + returns.mean(axis=1)).cumprod()


def dca_equal_curve(navs: pd.DataFrame, start=None) -> pd.Series:
    window = navs.copy()
    if start is not None:
        window = window.loc[window.index >= start]
    shares = pd.Series(0.0, index=window.columns)
    invested = 0.0
    values = []
    for date, row in window.iterrows():
        contribution = 1.0
        invested += contribution
        shares += contribution / len(window.columns) / row
        value = float((shares * row).sum())
        values.append((date, value / invested))
    curve = pd.Series(dict(values), name="每日定投等权")
    return curve / curve.iloc[0]


def metric_row(name: str, curve: pd.Series, period: str) -> dict:
    return {
        "区间": period,
        "策略": name,
        "收益率": float(curve.iloc[-1] - 1),
        "最大回撤": max_drawdown(curve),
    }


def main() -> None:
    navs = load_navs()
    features = build_features(navs)
    strategies = build_strategy_weights(navs, features)

    summary_rows = []
    curves_by_period: dict[str, pd.DataFrame] = {}
    for period_name, period_slug, days in PERIODS:
        start = None if days is None else navs.index[-1] - pd.Timedelta(days=days)
        curves = {}
        for name, weights in strategies.items():
            curve, _ = lag_and_backtest(navs, weights, start=start)
            curves[name] = curve
            summary_rows.append(metric_row(name, curve, period_name))
        for code in navs.columns:
            window = navs[code] if start is None else navs.loc[navs.index >= start, code]
            curve = window / window.iloc[0]
            name = f"全仓{code}"
            curves[name] = curve
            summary_rows.append(metric_row(name, curve, period_name))
        eq = equal_weight_curve(navs, start=start)
        dca = dca_equal_curve(navs, start=start).reindex(eq.index).ffill()
        curves["等权买入持有"] = eq
        curves["每日定投等权"] = dca
        summary_rows.append(metric_row("等权买入持有", eq, period_name))
        summary_rows.append(metric_row("每日定投等权", dca, period_name))
        curves_frame = pd.DataFrame(curves).dropna(how="all")
        curves_by_period[period_slug] = curves_frame
        curves_frame.to_csv(OUT / f"universal_{period_slug}_curves_cn.csv", encoding="utf-8-sig")

        if period_slug in {"1y", "6m", "3m"}:
            plot_cols = [
                "绝对相对动量Top1",
                "多因子Top1",
                "多因子Top2分散",
                "波动管理Top2",
                "等权买入持有",
                "每日定投等权",
            ]
            fig, ax = plt.subplots(figsize=(15, 7))
            for col in plot_cols:
                ax.plot(curves_frame.index, curves_frame[col], lw=1.8, label=col)
            ax.set_title(f"通用多基金轮动策略对比（{period_name}）", fontsize=15)
            ax.set_ylabel("累计净值（起点=1）")
            ax.set_xlabel("日期")
            ax.grid(True, alpha=0.25)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(OUT / f"universal_{period_slug}_curves_cn.svg", format="svg")
            plt.close(fig)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUT / "universal_backtest_summary_cn.csv", index=False, encoding="utf-8-sig")

    # Last 30 rolling 3-month windows, one trading day apart.
    window_rows = []
    n = 63
    all_dates = navs.index
    end_positions = range(len(all_dates) - 30, len(all_dates))
    for window_id, end_pos in enumerate(end_positions, start=1):
        if end_pos - n + 1 < 0:
            continue
        start_date = all_dates[end_pos - n + 1]
        end_date = all_dates[end_pos]
        local_summary = []
        for name, weights in strategies.items():
            curve, _ = lag_and_backtest(navs.loc[:end_date], weights.loc[:end_date], start=start_date)
            local_summary.append((name, float(curve.iloc[-1] - 1), max_drawdown(curve)))
        eq = equal_weight_curve(navs.loc[:end_date], start=start_date)
        dca = dca_equal_curve(navs.loc[:end_date], start=start_date).reindex(eq.index).ffill()
        local_summary.append(("等权买入持有", float(eq.iloc[-1] - 1), max_drawdown(eq)))
        local_summary.append(("每日定投等权", float(dca.iloc[-1] - 1), max_drawdown(dca)))
        for code in navs.columns:
            window = navs.loc[start_date:end_date, code]
            curve = window / window.iloc[0]
            local_summary.append((f"全仓{code}", float(curve.iloc[-1] - 1), max_drawdown(curve)))
        best = max(local_summary, key=lambda row: row[1])[0]
        for name, ret, mdd in local_summary:
            window_rows.append(
                {
                    "窗口": window_id,
                    "起始日": start_date.date(),
                    "结束日": end_date.date(),
                    "策略": name,
                    "收益率": ret,
                    "最大回撤": mdd,
                    "是否窗口最佳": name == best,
                }
            )
    rolling = pd.DataFrame(window_rows)
    rolling.to_csv(OUT / "rolling_30_windows_3m_cn.csv", index=False, encoding="utf-8-sig")
    rolling_summary = (
        rolling.groupby("策略")
        .agg(
            窗口数=("收益率", "count"),
            最佳次数=("是否窗口最佳", "sum"),
            平均收益=("收益率", "mean"),
            收益标准差=("收益率", "std"),
            平均最大回撤=("最大回撤", "mean"),
            最差最大回撤=("最大回撤", "min"),
        )
        .reset_index()
        .sort_values(["最佳次数", "平均收益"], ascending=False)
    )
    rolling_summary.to_csv(OUT / "rolling_30_windows_summary_cn.csv", index=False, encoding="utf-8-sig")

    latest_weights = {}
    for name, weights in strategies.items():
        latest_weights[name] = weights.iloc[-1][weights.iloc[-1] > 0].to_dict()
    latest = pd.DataFrame(
        [
            {"策略": name, "目标权重": "；".join(f"{code} {weight:.0%}" for code, weight in weight_map.items())}
            for name, weight_map in latest_weights.items()
        ]
    )
    latest.to_csv(OUT / "latest_universal_targets_cn.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# 通用基金轮动策略研究",
        "",
        "## 目标",
        "",
        "这次不再只研究两只基金相对净值，而是把 CPO、存储/半导体、AI、PCB代理、半导体主动和科技成长锚放进同一个可买基金池，尝试更通用的横截面轮动。",
        "",
        "## 基金池",
        "",
    ]
    for code, name in FUNDS.items():
        lines.append(f"- `{code}` {name}")
    lines += [
        "",
        "## 策略定义",
        "",
        "- 绝对相对动量Top1：先要求20日收益为正且净值高于20日均线，再选择60日动量最强的基金，否则回到011370。",
        "- 多因子Top1：每天计算横截面得分，因子包括20日动量、60日动量、20日风险调整动量、60日趋势质量、20日波动惩罚、20日回撤惩罚，选得分最高者。",
        "- 多因子Top2分散：选正得分前两名，按60%/40%持有，减少单基金误判。",
        "- 波动管理Top2：在Top2基础上，如果所选基金20日年化波动均值超过45%，把25%仓位放到011370。",
        "- 所有策略都用官方净值形成信号，新仓位按两日净值延迟处理；换仓扣0.15%摩擦成本。",
        "",
        "## 最新目标权重",
        "",
        "| 策略 | 目标权重 |",
        "|---|---|",
    ]
    for _, row in latest.iterrows():
        lines.append(f"| {row['策略']} | {row['目标权重']} |")
    lines += ["", "## 回测摘要", "", "| 区间 | 策略 | 收益率 | 最大回撤 |", "|---|---|---:|---:|"]
    display_order = ["近1年", "近6个月", "近3个月", "全共同历史"]
    preferred = ["绝对相对动量Top1", "多因子Top1", "多因子Top2分散", "波动管理Top2", "等权买入持有", "每日定投等权"]
    for period in display_order:
        for name in preferred:
            row = summary[(summary["区间"] == period) & (summary["策略"] == name)]
            if not row.empty:
                row = row.iloc[0]
                lines.append(f"| {period} | {name} | {pct(row['收益率'])} | {pct(row['最大回撤'])} |")
    lines += [
        "",
        "## 近30个三个月滑动窗口",
        "",
        "| 策略 | 最佳次数 | 平均收益 | 收益标准差 | 平均最大回撤 | 最差最大回撤 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, row in rolling_summary.iterrows():
        if row["策略"] in preferred or row["策略"].startswith("全仓"):
            lines.append(
                f"| {row['策略']} | {int(row['最佳次数'])}/{int(row['窗口数'])} | {pct(row['平均收益'])} | "
                f"{pct(row['收益标准差'])} | {pct(row['平均最大回撤'])} | {pct(row['最差最大回撤'])} |"
            )
    lines += [
        "",
        "## 暂定结论",
        "",
        "更通用的方向不是单一均线或一对基金的15/40相对净值，而是“横截面多因子排名 + 趋势过滤 + 分散持仓 + 波动管理”。",
        "如果只追求短期最高收益，Top1会更激进；如果追求可迁移和少过拟合，多因子Top2或波动管理Top2更值得继续优化。",
    ]
    (OUT / "universal_rotation_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
