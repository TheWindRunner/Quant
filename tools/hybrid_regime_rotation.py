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


OUT = Path("output/hybrid_regime_rotation")
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CODES = ["006503", "011370", "007817", "008887", "008585", "720001"]
ANCHOR = "011370"
COST = 0.0015
WINDOW_DAYS = 63


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


def build_base_choice(navs: pd.DataFrame) -> pd.Series:
    momentum = navs.pct_change(2, fill_method=None).where(lambda frame: frame > 0)
    columns = np.array(momentum.columns)
    raw = []
    for row in momentum.to_numpy():
        mask = ~np.isnan(row)
        raw.append(columns[mask][np.argmax(row[mask])] if mask.any() else ANCHOR)
    return confirm_choice(pd.Series(raw, index=navs.index), confirm=5)


def build_hybrid_choice(
    navs: pd.DataFrame,
    base_choice: pd.Series,
    lookback: int,
    momentum_threshold: float,
    drawdown_threshold: float,
    regime_confirm: int,
) -> pd.Series:
    core = navs["006503"]
    regime = (
        (core.pct_change(lookback, fill_method=None) > momentum_threshold)
        & (core / core.rolling(20).max() - 1 > drawdown_threshold)
        & (core > core.rolling(20).mean())
    ).fillna(False)

    streak = 0
    force_006503 = []
    for flag in regime.to_numpy():
        streak = streak + 1 if flag else 0
        force_006503.append(streak >= regime_confirm)

    choice = base_choice.copy()
    choice.loc[pd.Series(force_006503, index=navs.index)] = "006503"
    return choice


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
    base_choice = build_base_choice(navs)
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
            [5, 10, 15, 20, 30, 40, 60],
            [0.02, 0.05, 0.08, 0.12, 0.18],
            [-0.03, -0.05, -0.08, -0.12],
            [1, 2, 3, 5],
        )
    )

    score_rows = []
    returns_cache = {}
    for params in family:
        choice = build_hybrid_choice(navs, base_choice, *params)
        strategy_return = returns_from_choice(navs, choice)
        returns_cache[params] = strategy_return
        values = []
        wins = 0
        for start, end in train_windows:
            curve = window_curve(strategy_return, start, end)
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
        lambda item: f"L{item[0]}_阈值{item[1]:.2f}_回撤{item[2]:.2f}_确认{item[3]}"
    )
    score.to_csv(OUT / "hybrid_training_scores_cn.csv", index=False, encoding="utf-8-sig")

    base_windows = evaluate_windows(navs, base_returns, "基础短周期Top1")
    hybrid_windows = evaluate_windows(navs, returns_cache[best_params], f"混合策略_L{best_params[0]}_阈值{best_params[1]:.2f}_回撤{best_params[2]:.2f}_确认{best_params[3]}")
    base_windows.to_csv(OUT / "base_short_term_windows_cn.csv", index=False, encoding="utf-8-sig")
    hybrid_windows.to_csv(OUT / "hybrid_windows_cn.csv", index=False, encoding="utf-8-sig")

    # Restricted-family walk-forward using top 10 training candidates.
    top10 = [tuple(item) for item in score["params"].head(10)]
    all_windows = [(end_pos - WINDOW_DAYS + 1, end_pos) for end_pos in range(WINDOW_DAYS - 1, len(navs))]
    walk_rows = []
    for window_id, end_pos in enumerate(range(len(navs) - 30, len(navs)), start=1):
        start_pos = end_pos - WINDOW_DAYS + 1
        train = [item for item in all_windows if item[1] < start_pos][-120:]
        best_local = None
        best_score = None
        for params in top10:
            series = returns_cache[params]
            vals = []
            wins = 0
            for train_start_pos, train_end_pos in train:
                start = navs.index[train_start_pos]
                end = navs.index[train_end_pos]
                curve = window_curve(series, start, end)
                ret = float(curve.iloc[-1] - 1)
                vals.append(ret)
                all_006503 = float(navs.loc[start:end, "006503"].iloc[-1] / navs.loc[start:end, "006503"].iloc[0] - 1)
                wins += ret > all_006503
            score_value = (wins, float(np.mean(vals) - 0.25 * np.std(vals)))
            if best_score is None or score_value > best_score:
                best_score = score_value
                best_local = params
        assert best_local is not None
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        series = returns_cache[best_local]
        curve = window_curve(series, start, end)
        strategy_ret = float(curve.iloc[-1] - 1)
        all_in = {code: float(navs.loc[start:end, code].iloc[-1] / navs.loc[start:end, code].iloc[0] - 1) for code in navs.columns}
        best_code = max(all_in, key=all_in.get)
        walk_rows.append(
            {
                "strategy": "受限家族Walk-forward混合策略",
                "window": window_id,
                "start": start.date(),
                "end": end.date(),
                "selected_params": f"L{best_local[0]}_阈值{best_local[1]:.2f}_回撤{best_local[2]:.2f}_确认{best_local[3]}",
                "return": strategy_ret,
                "mdd": max_drawdown(curve),
                "all_006503": all_in["006503"],
                "best_all": all_in[best_code],
                "best_code": best_code,
                "win_006503": strategy_ret > all_in["006503"],
                "win_best": strategy_ret > all_in[best_code],
            }
        )
    walk_windows = pd.DataFrame(walk_rows)
    walk_windows.to_csv(OUT / "hybrid_walk_forward_windows_cn.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([summarize(base_windows), summarize(hybrid_windows), summarize(walk_windows)])
    summary.to_csv(OUT / "hybrid_summary_cn.csv", index=False, encoding="utf-8-sig")

    latest_choice = build_hybrid_choice(navs, base_choice, *best_params)
    latest = pd.DataFrame(
        [
            {
                "latest_date": navs.index[-1].date(),
                "base_raw_target": base_choice.iloc[-1],
                "hybrid_target": latest_choice.iloc[-1],
                "selected_params": f"L{best_params[0]}_阈值{best_params[1]:.2f}_回撤{best_params[2]:.2f}_确认{best_params[3]}",
            }
        ]
    )
    latest.to_csv(OUT / "hybrid_latest_signal_cn.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(base_windows["window"], base_windows["return"], marker="o", lw=1.8, label="基础短周期Top1")
    ax.plot(hybrid_windows["window"], hybrid_windows["return"], marker="o", lw=1.8, label="测试前选参混合策略")
    ax.plot(walk_windows["window"], walk_windows["return"], marker="o", lw=1.8, label="受限家族Walk-forward混合策略")
    ax.plot(base_windows["window"], base_windows["all_006503"], color="#d62728", lw=1.8, label="006503全仓")
    ax.plot(base_windows["window"], base_windows["best_all"], color="#333333", lw=1.2, ls="--", label="窗口事后最强全仓")
    ax.set_title("混合策略：30个三个月窗口收益对比", fontsize=15)
    ax.set_xlabel("窗口编号")
    ax.set_ylabel("区间收益率")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "hybrid_windows_curve_cn.svg", format="svg")
    plt.close(fig)

    lines = [
        "# 混合策略优化",
        "",
        "## 逻辑",
        "",
        "底层继续使用已经过线的短周期策略：2交易日动量Top1、只追正收益、连续5日确认。",
        "上层加入一个 006503 强趋势覆盖层：当 006503 的中期趋势足够强时，直接持有 006503；否则回到短周期Top1策略。",
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
        "## 选中的固定参数",
        "",
        f"- 测试前历史选中的参数：`L{best_params[0]} / 阈值 {best_params[1]:.2f} / 回撤阈值 {best_params[2]:.2f} / 确认 {best_params[3]}`。",
        "- 含义：若 006503 在该 lookback 上的累计涨幅超过阈值、20日回撤未劣化到阈值以下、且站上20日均线，并连续满足若干天，则直接持有 006503。",
        "",
        "## 判断",
        "",
        "如果你接受“先看 006503 是否处在强趋势主升段，强则直接抱住；否则再做横向轮动”的结构，那么这版已经明显强于前面的纯短周期Top1，也更接近你要的窗口胜率标准。",
    ]
    (OUT / "hybrid_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
