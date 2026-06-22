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


OUT = Path("output/parameter_ensemble_rotation")
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


def choice_for(navs: pd.DataFrame, lookback: int, confirm: int) -> pd.Series:
    momentum = navs.pct_change(lookback, fill_method=None).where(lambda x: x > 0)
    columns = np.array(momentum.columns)
    raw = []
    for row in momentum.to_numpy():
        mask = ~np.isnan(row)
        raw.append(columns[mask][np.argmax(row[mask])] if mask.any() else ANCHOR)
    return confirm_choice(pd.Series(raw, index=navs.index), confirm)


def weights_from_choices(choices: dict[tuple[int, int], pd.Series], params: list[tuple[int, int]], navs: pd.DataFrame, mode: str) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    if mode == "vote_top1":
        for date in navs.index:
            votes = pd.Series([choices[p].loc[date] for p in params]).value_counts()
            weights.loc[date, votes.index[0]] = 1.0
    elif mode == "equal_vote":
        for date in navs.index:
            for p in params:
                weights.loc[date, choices[p].loc[date]] += 1.0 / len(params)
    else:
        raise ValueError(mode)
    return weights


def returns_from_weights(navs: pd.DataFrame, weights: pd.DataFrame) -> pd.Series:
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    effective = weights.shift(2).fillna(0.0)
    empty = effective.sum(axis=1).eq(0)
    effective.loc[empty, ANCHOR] = 1.0
    gross = (effective * returns).sum(axis=1)
    turnover = weights.diff().abs().sum(axis=1) / 2
    return gross - turnover.shift(1).fillna(0.0) * COST


def returns_from_choice(navs: pd.DataFrame, choice: pd.Series) -> pd.Series:
    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    for date, code in choice.items():
        weights.loc[date, code] = 1.0
    return returns_from_weights(navs, weights)


def window_detail(navs: pd.DataFrame, strategy_return: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    for window_id, end_pos in enumerate(range(len(navs) - 30, len(navs)), start=1):
        start_pos = end_pos - WINDOW_DAYS + 1
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        curve = (1 + strategy_return.loc[start:end]).cumprod()
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
    candidates = list(itertools.product([1, 2, 3, 5, 8, 10], [1, 2, 3, 5, 8, 10]))
    choices = {params: choice_for(navs, *params) for params in candidates}
    returns = {params: returns_from_choice(navs, choice) for params, choice in choices.items()}

    first_test_start = navs.index[len(navs) - 30 - WINDOW_DAYS + 1]
    train_windows = []
    for end_pos in range(WINDOW_DAYS - 1, len(navs)):
        start_pos = end_pos - WINDOW_DAYS + 1
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        if end < first_test_start:
            train_windows.append((start, end))
    train_windows = train_windows[-240:]

    scored = []
    for params, series in returns.items():
        vals = []
        wins = 0
        for start, end in train_windows:
            curve = (1 + series.loc[start:end]).cumprod()
            ret = float(curve.iloc[-1] - 1)
            vals.append(ret)
            all_006503 = float(navs.loc[start:end, "006503"].iloc[-1] / navs.loc[start:end, "006503"].iloc[0] - 1)
            wins += ret > all_006503
        scored.append(
            {
                "params": params,
                "train_wins_006503": wins,
                "train_mean_return": float(np.mean(vals)),
                "train_std_return": float(np.std(vals)),
                "score": wins + float(np.mean(vals) - 0.25 * np.std(vals)),
            }
        )
    score = pd.DataFrame(scored).sort_values(["train_wins_006503", "train_mean_return"], ascending=False)
    score["params_label"] = score["params"].map(lambda x: f"L{x[0]}_确认{x[1]}")
    score.to_csv(OUT / "ensemble_training_scores_cn.csv", index=False, encoding="utf-8-sig")

    all_frames = []
    # Single best remains the baseline.
    best_param = tuple(score.iloc[0]["params"])
    all_frames.append(window_detail(navs, returns[best_param], f"单参数最佳：L{best_param[0]}_确认{best_param[1]}"))
    for topk in [3, 5, 8, 10]:
        params = [tuple(item) for item in score["params"].head(topk)]
        for mode, label in [("vote_top1", "多数投票Top1"), ("equal_vote", "参数等权组合")]:
            weights = weights_from_choices(choices, params, navs, mode)
            strategy_return = returns_from_weights(navs, weights)
            all_frames.append(window_detail(navs, strategy_return, f"{label}_Top{topk}"))

    detail = pd.concat(all_frames, ignore_index=True)
    summary = pd.DataFrame([summarize(frame) for _, frame in detail.groupby("strategy", sort=False)])
    summary = summary.sort_values(["wins_vs_006503", "mean_return"], ascending=False)
    detail.to_csv(OUT / "ensemble_window_detail_cn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "ensemble_summary_cn.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(15, 7))
    for label in summary["strategy"].head(6):
        sub = detail[detail["strategy"] == label]
        ax.plot(sub["window"], sub["return"], marker="o", lw=1.5, label=label)
    base = detail[detail["strategy"] == summary["strategy"].iloc[0]]
    ax.plot(base["window"], base["all_006503"], color="#d62728", lw=1.8, label="006503全仓")
    ax.plot(base["window"], base["best_all"], color="#333333", lw=1.2, ls="--", label="窗口事后最强全仓")
    ax.set_title("参数投票/组合策略：30个三个月窗口收益对比", fontsize=15)
    ax.set_xlabel("窗口编号")
    ax.set_ylabel("区间收益率")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "ensemble_windows_curve_cn.svg", format="svg")
    plt.close(fig)

    lines = [
        "# 参数投票与组合策略测试",
        "",
        "## 目的",
        "",
        "单一短周期参数已经过30窗口门槛，但仍可能有参数偶然性。本轮只用测试前历史给参数排序，然后把前K个参数做多数投票或等权组合。",
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
        "## 结论",
        "",
        "如果组合/投票策略的胜率不低于单参数，同时回撤或收益标准差下降，它会比单参数更适合作为通用策略候选；否则说明当前市场里集中押最强短周期基金仍然贡献了主要收益。",
    ]
    (OUT / "ensemble_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
