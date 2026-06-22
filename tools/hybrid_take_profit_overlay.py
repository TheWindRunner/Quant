from __future__ import annotations

from pathlib import Path
import itertools

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from quant_fund_advisor.style_timing import read_cached_nav


OUT = Path("output/hybrid_take_profit_overlay")
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CODES = ["006503", "011370", "007817", "008887", "008585", "720001"]
ANCHOR = "011370"
COST = 0.0015
WINDOW_DAYS = 63
PROXY_MAP = {
    "006503": "memory_semiconductor_proxy",
    "007817": "cpo_communication",
    "008887": "memory_semiconductor_proxy",
    "720001": "pcb_quality_selection",
}


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


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


def build_base_hybrid_choice(navs: pd.DataFrame) -> pd.Series:
    momentum = navs.pct_change(2, fill_method=None).where(lambda frame: frame > 0)
    columns = np.array(momentum.columns)
    raw = []
    for row in momentum.to_numpy():
        mask = ~np.isnan(row)
        raw.append(columns[mask][np.argmax(row[mask])] if mask.any() else ANCHOR)
    short_choice = confirm_choice(pd.Series(raw, index=navs.index), confirm=5)

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

    choice = short_choice.copy()
    choice.loc[pd.Series(force_core, index=navs.index)] = "006503"
    return choice


def load_proxy_frame(asset: str, index: pd.Index) -> pd.DataFrame:
    path = Path("data/external_factors") / f"{asset}.csv"
    if not path.exists():
        return pd.DataFrame(index=index)
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "etf_close" not in frame.columns and "etf_change_pct" in frame.columns:
        frame["etf_close"] = (1 + frame["etf_change_pct"].fillna(0.0) / 100).cumprod()
    return frame.reindex(index).ffill()


def build_proxy_pack(index: pd.Index) -> dict[str, pd.DataFrame]:
    return {code: load_proxy_frame(asset, index) for code, asset in PROXY_MAP.items()}


def overlay_choice(
    navs: pd.DataFrame,
    base_choice: pd.Series,
    proxies: dict[str, pd.DataFrame],
    arm_profit: float,
    trail_drop: float,
    overheat_lookback: int,
    overheat_nav: float,
    overheat_proxy: float,
    reentry_lb: int,
    reentry_mom: float,
) -> tuple[pd.Series, pd.DataFrame]:
    choice = base_choice.copy()
    final = []
    notes = []
    current = choice.iloc[0]
    overlay_exit = False
    entry_nav = navs.loc[navs.index[0], current]
    peak_nav = entry_nav
    armed = False

    for date in navs.index:
        desired = choice.loc[date]
        if desired != current and not overlay_exit:
            current = desired
            entry_nav = navs.loc[date, current]
            peak_nav = entry_nav
            armed = False

        nav = navs.loc[date, current]
        peak_nav = max(peak_nav, nav)
        proxy = proxies.get(current, pd.DataFrame(index=navs.index))
        proxy_ret = 0.0
        proxy_amount_ratio = 1.0
        if not proxy.empty:
            if "etf_change_pct" in proxy.columns:
                proxy_ret = float(proxy.loc[date, "etf_change_pct"] or 0.0) / 100
            if "etf_amount" in proxy.columns:
                amount = proxy["etf_amount"]
                amt = amount.loc[date]
                avg = amount.rolling(20).mean().loc[date]
                if pd.notna(amt) and pd.notna(avg) and avg:
                    proxy_amount_ratio = float(amt / avg)

        runup = nav / entry_nav - 1 if entry_nav else 0.0
        nav_short = navs[current].pct_change(overheat_lookback, fill_method=None).loc[date]
        hot = (
            pd.notna(nav_short)
            and nav_short >= overheat_nav
            and proxy_ret >= overheat_proxy
            and proxy_amount_ratio >= 1.0
        )
        if runup >= arm_profit and hot:
            armed = True

        trend_weak = (
            navs[current].loc[date] < navs[current].rolling(10).mean().loc[date]
            if pd.notna(navs[current].rolling(10).mean().loc[date])
            else False
        )
        draw_from_peak = nav / peak_nav - 1 if peak_nav else 0.0
        exit_signal = armed and draw_from_peak <= -trail_drop and (trend_weak or proxy_ret < 0)

        if not overlay_exit and exit_signal:
            current = ANCHOR
            overlay_exit = True
            armed = False
            entry_nav = navs.loc[date, current]
            peak_nav = entry_nav
            final.append(current)
            notes.append("止盈撤退")
            continue

        if overlay_exit:
            target = desired
            target_series = navs[target]
            reentry = False
            if target != ANCHOR:
                target_mom = target_series.pct_change(reentry_lb, fill_method=None).loc[date]
                ma20 = target_series.rolling(20).mean().loc[date]
                reentry = pd.notna(target_mom) and target_mom >= reentry_mom and target_series.loc[date] > ma20
            if reentry:
                current = target
                overlay_exit = False
                entry_nav = navs.loc[date, current]
                peak_nav = entry_nav
                final.append(current)
                notes.append("重新上车")
                continue
            final.append(current)
            notes.append("撤退持有锚基金")
            continue

        final.append(current)
        notes.append("正常持有")

    return pd.Series(final, index=navs.index), pd.DataFrame({"date": navs.index, "action_note": notes})


def returns_from_choice(navs: pd.DataFrame, choice: pd.Series) -> pd.Series:
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    position = choice.shift(2).fillna(ANCHOR).to_numpy(dtype=object)
    strategy_return = np.zeros(len(position))
    for code in navs.columns:
        strategy_return += (position == code) * returns[code].to_numpy()
    turnover_cost = (choice != choice.shift()).astype(float).shift(1).fillna(0.0).to_numpy() * COST
    return pd.Series(strategy_return - turnover_cost, index=navs.index)


def window_curve(series: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    return (1 + series.loc[start:end]).cumprod()


def evaluate_windows(navs: pd.DataFrame, strategy_return: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    for window_id, end_pos in enumerate(range(len(navs) - 30, len(navs)), start=1):
        start_pos = end_pos - WINDOW_DAYS + 1
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        curve = window_curve(strategy_return, start, end)
        strategy_ret = float(curve.iloc[-1] - 1)
        all_in = {code: float(navs.loc[start:end, code].iloc[-1] / navs.loc[start:end, code].iloc[0] - 1) for code in navs.columns}
        best_code = max(all_in, key=all_in.get)
        rows.append(
            {
                "strategy": label,
                "window": window_id,
                "start": start.date(),
                "end": end.date(),
                "return": strategy_ret,
                "mdd": max_drawdown(curve),
                "all_006503": all_in["006503"],
                "best_all": all_in[best_code],
                "best_code": best_code,
                "win_006503": strategy_ret > all_in["006503"],
                "win_best": strategy_ret > all_in[best_code],
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict:
    return {
        "strategy": frame["strategy"].iloc[0],
        "windows": len(frame),
        "wins_vs_006503": int(frame["win_006503"].sum()),
        "wins_vs_best": int(frame["win_best"].sum()),
        "mean_return": float(frame["return"].mean()),
        "std_return": float(frame["return"].std()),
        "mean_mdd": float(frame["mdd"].mean()),
        "worst_mdd": float(frame["mdd"].min()),
        "min_return": float(frame["return"].min()),
    }


def main() -> None:
    navs = pd.DataFrame({code: read_cached_nav(code) for code in CODES}).dropna().sort_index()
    proxies = build_proxy_pack(navs.index)
    base_choice = build_base_hybrid_choice(navs)
    base_returns = returns_from_choice(navs, base_choice)

    first_test_start = navs.index[len(navs) - 30 - WINDOW_DAYS + 1]
    train_windows = []
    for end_pos in range(WINDOW_DAYS - 1, len(navs)):
        start_pos = end_pos - WINDOW_DAYS + 1
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        if end < first_test_start:
            train_windows.append((start, end))
    train_windows = train_windows[-240:]

    family = list(
        itertools.product(
            [0.22, 0.30],                  # arm_profit
            [0.08, 0.10],                  # trail_drop
            [3],                           # overheat_lookback
            [0.12, 0.18],                  # overheat_nav
            [0.02],                        # overheat_proxy
            [3],                           # reentry_lb
            [0.03, 0.05],                  # reentry_mom
        )
    )

    score_rows = []
    cache = {}
    for params in family:
        choice, notes = overlay_choice(navs, base_choice, proxies, *params)
        series = returns_from_choice(navs, choice)
        cache[params] = (choice, notes, series)
        values = []
        wins = 0
        for start, end in train_windows:
            curve = window_curve(series, start, end)
            ret = float(curve.iloc[-1] - 1)
            values.append(ret)
            all_006503 = float(navs.loc[start:end, "006503"].iloc[-1] / navs.loc[start:end, "006503"].iloc[0] - 1)
            wins += ret > all_006503
        score_rows.append(
            {
                "params": params,
                "train_windows": len(train_windows),
                "train_wins_006503": wins,
                "train_mean_return": float(np.mean(values)),
                "train_std_return": float(np.std(values)),
                "train_score": wins + float(np.mean(values) - 0.25 * np.std(values)),
            }
        )
    score = pd.DataFrame(score_rows).sort_values(["train_wins_006503", "train_mean_return"], ascending=False)
    best_params = tuple(score.iloc[0]["params"])
    score["params_label"] = score["params"].map(
        lambda item: (
            f"止盈{item[0]:.2f}_回撤{item[1]:.2f}_过热窗{item[2]}_"
            f"净值过热{item[3]:.2f}_ETF过热{item[4]:.3f}_回补窗{item[5]}_回补动量{item[6]:.2f}"
        )
    )
    score.to_csv(OUT / "take_profit_training_scores_cn.csv", index=False, encoding="utf-8-sig")

    overlay_choice_best, notes_best, overlay_returns = cache[best_params]
    notes_best.to_csv(OUT / "take_profit_action_notes_cn.csv", index=False, encoding="utf-8-sig")

    base_windows = evaluate_windows(navs, base_returns, "混合策略原版")
    overlay_windows = evaluate_windows(
        navs,
        overlay_returns,
        (
            f"止盈层_止盈{best_params[0]:.2f}_回撤{best_params[1]:.2f}_过热窗{best_params[2]}_"
            f"净值过热{best_params[3]:.2f}_ETF过热{best_params[4]:.3f}_回补窗{best_params[5]}_回补动量{best_params[6]:.2f}"
        ),
    )
    base_windows.to_csv(OUT / "base_hybrid_windows_cn.csv", index=False, encoding="utf-8-sig")
    overlay_windows.to_csv(OUT / "take_profit_windows_cn.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([summarize(base_windows), summarize(overlay_windows)])
    summary.to_csv(OUT / "take_profit_summary_cn.csv", index=False, encoding="utf-8-sig")

    latest = pd.DataFrame(
        [
            {
                "latest_date": navs.index[-1].date(),
                "base_target": base_choice.iloc[-1],
                "overlay_target": overlay_choice_best.iloc[-1],
                "selected_params": score.loc[0, "params_label"],
            }
        ]
    )
    latest.to_csv(OUT / "take_profit_latest_signal_cn.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(base_windows["window"], base_windows["return"], marker="o", lw=1.8, label="混合策略原版")
    ax.plot(overlay_windows["window"], overlay_windows["return"], marker="o", lw=1.8, label="叠加止盈层")
    ax.plot(base_windows["window"], base_windows["all_006503"], color="#d62728", lw=1.8, label="006503全仓")
    ax.plot(base_windows["window"], base_windows["best_all"], color="#333333", lw=1.2, ls="--", label="窗口事后最强全仓")
    ax.set_title("止盈层：30个三个月窗口收益对比", fontsize=15)
    ax.set_xlabel("窗口编号")
    ax.set_ylabel("区间收益率")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "take_profit_windows_curve_cn.svg", format="svg")
    plt.close(fig)

    lines = [
        "# 止盈层测试",
        "",
        "## 逻辑",
        "",
        "止盈层只在持仓已经明显盈利且出现过热迹象时才启动，避免把普通上涨过早卖掉。",
        "",
        "- 启动条件：从入场价累计盈利达到阈值，且短窗口净值涨幅与ETF代理涨幅同时偏热。",
        "- 卖出条件：从峰值回落超过移动止盈阈值，同时短期趋势转弱或ETF代理转负，切到011370。",
        "- 回补条件：原目标基金重新回到20日均线上方，且短窗口动量恢复到阈值以上。",
        "",
        "## 结果",
        "",
        "| 策略 | 跑赢006503 | 跑赢最强全仓 | 平均收益 | 收益标准差 | 平均最大回撤 | 最差最大回撤 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['strategy']} | {int(row['wins_vs_006503'])}/{int(row['windows'])} | {int(row['wins_vs_best'])}/{int(row['windows'])} | "
            f"{pct(row['mean_return'])} | {pct(row['std_return'])} | {pct(row['mean_mdd'])} | {pct(row['worst_mdd'])} |"
        )
    lines += [
        "",
        "## 判断",
        "",
        "如果止盈层的窗口胜率、平均收益或回撤明显改善，就可以把它并入生产候选；如果只是让收益下降、回撤不变，说明当前市场更像强趋势而不是泡沫顶，贸然止盈会伤害主升段收益。",
    ]
    (OUT / "take_profit_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
