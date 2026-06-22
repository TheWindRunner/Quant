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


OUT = Path("output/rotation_threshold_audit")
OUT.mkdir(parents=True, exist_ok=True)

FONT_PATH = r"C:\Windows\Fonts\simhei.ttf"
font_manager.fontManager.addfont(FONT_PATH)
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

CODES = ["006503", "011370", "007817", "008887", "008585", "720001"]
NAMES = {
    "006503": "财通集成电路产业股票C",
    "011370": "华商均衡成长混合C",
    "007817": "国泰中证全指通信设备ETF联接A",
    "008887": "华夏国证半导体芯片ETF联接A",
    "008585": "华夏中证人工智能主题ETF联接A",
    "720001": "财通价值动量混合",
}
ANCHOR = "011370"
COST = 0.0015
WINDOW_DAYS = 63


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def confirm_choice(raw: pd.Series, confirm: int) -> pd.Series:
    values = raw.to_numpy(dtype=object)
    out = []
    current = values[0]
    pending = current
    count = 0
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
        out.append(current)
    return pd.Series(out, index=raw.index)


def top1_momentum_choice(navs: pd.DataFrame, lookback: int, confirm: int, positive_only: bool) -> pd.Series:
    momentum = navs.pct_change(lookback, fill_method=None)
    if positive_only:
        momentum = momentum.where(momentum > 0)
    columns = np.array(momentum.columns)
    raw = []
    for row in momentum.to_numpy():
        mask = ~np.isnan(row)
        if mask.any():
            raw.append(columns[mask][np.argmax(row[mask])])
        else:
            raw.append(ANCHOR)
    return confirm_choice(pd.Series(raw, index=navs.index), confirm)


def returns_from_choice(navs: pd.DataFrame, choice: pd.Series) -> pd.Series:
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    position = choice.shift(2).fillna(ANCHOR).to_numpy(dtype=object)
    strategy_return = np.zeros(len(position))
    for code in navs.columns:
        strategy_return += (position == code) * returns[code].to_numpy()
    turnover_cost = (choice != choice.shift()).astype(float).shift(1).fillna(0.0).to_numpy() * COST
    return pd.Series(strategy_return - turnover_cost, index=navs.index)


def window_rows(navs: pd.DataFrame, strategy_return: pd.Series, label: str) -> pd.DataFrame:
    rows = []
    for window_id, end_pos in enumerate(range(len(navs) - 30, len(navs)), start=1):
        start_pos = end_pos - WINDOW_DAYS + 1
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        curve = (1 + strategy_return.loc[start:end]).cumprod()
        strategy_ret = float(curve.iloc[-1] - 1)
        strategy_mdd = max_drawdown(curve)
        all_in = {code: float(navs.loc[start:end, code].iloc[-1] / navs.loc[start:end, code].iloc[0] - 1) for code in navs.columns}
        best_code = max(all_in, key=all_in.get)
        rows.append(
            {
                "策略": label,
                "窗口": window_id,
                "起始日": start.date(),
                "结束日": end.date(),
                "策略收益": strategy_ret,
                "策略最大回撤": strategy_mdd,
                "006503全仓收益": all_in["006503"],
                "最强全仓代码": best_code,
                "最强全仓收益": all_in[best_code],
                "是否跑赢006503": strategy_ret > all_in["006503"],
                "是否跑赢最强全仓": strategy_ret > all_in[best_code],
            }
        )
    return pd.DataFrame(rows)


def evaluate_params(navs: pd.DataFrame, params: tuple[int, int, bool]) -> pd.DataFrame:
    choice = top1_momentum_choice(navs, *params)
    returns = returns_from_choice(navs, choice)
    return window_rows(navs, returns, f"Top1动量_L{params[0]}_确认{params[1]}_只追正收益{params[2]}")


def walk_forward_rows(navs: pd.DataFrame, candidates: list[tuple[int, int, bool]]) -> pd.DataFrame:
    returns_cache = {params: returns_from_choice(navs, top1_momentum_choice(navs, *params)) for params in candidates}
    log_cache = {params: np.log1p(series.clip(lower=-0.999999)).cumsum().to_numpy() for params, series in returns_cache.items()}
    nav_array = navs.to_numpy()
    code_index = {code: idx for idx, code in enumerate(navs.columns)}

    def fast_strategy_return(params: tuple[int, int, bool], start_pos: int, end_pos: int) -> float:
        log_curve = log_cache[params]
        base = log_curve[start_pos - 1] if start_pos > 0 else 0.0
        return float(np.exp(log_curve[end_pos] - base) - 1)

    def all_in_return(code: str, start_pos: int, end_pos: int) -> float:
        idx = code_index[code]
        return float(nav_array[end_pos, idx] / nav_array[start_pos, idx] - 1)

    all_windows = [(end_pos - WINDOW_DAYS + 1, end_pos) for end_pos in range(WINDOW_DAYS - 1, len(navs))]
    rows = []
    for window_id, end_pos in enumerate(range(len(navs) - 30, len(navs)), start=1):
        start_pos = end_pos - WINDOW_DAYS + 1
        train_windows = [item for item in all_windows if item[1] < start_pos][-120:]
        best_score = None
        best_params = None
        for params in candidates:
            values = []
            wins = 0
            for train_start, train_end in train_windows:
                ret = fast_strategy_return(params, train_start, train_end)
                values.append(ret)
                wins += ret > all_in_return("006503", train_start, train_end)
            score = (-999, -999) if not values else (wins, float(np.mean(values) - 0.25 * np.std(values)))
            if best_score is None or score > best_score:
                best_score = score
                best_params = params
        assert best_params is not None
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        strategy_return = returns_cache[best_params]
        curve = (1 + strategy_return.loc[start:end]).cumprod()
        strategy_ret = float(curve.iloc[-1] - 1)
        strategy_mdd = max_drawdown(curve)
        all_in = {code: all_in_return(code, start_pos, end_pos) for code in navs.columns}
        best_code = max(all_in, key=all_in.get)
        rows.append(
            {
                "策略": "Walk-forward选参Top1动量",
                "窗口": window_id,
                "起始日": start.date(),
                "结束日": end.date(),
                "训练选中参数": f"L{best_params[0]}_确认{best_params[1]}_只追正收益{best_params[2]}",
                "策略收益": strategy_ret,
                "策略最大回撤": strategy_mdd,
                "006503全仓收益": all_in["006503"],
                "最强全仓代码": best_code,
                "最强全仓收益": all_in[best_code],
                "是否跑赢006503": strategy_ret > all_in["006503"],
                "是否跑赢最强全仓": strategy_ret > all_in[best_code],
            }
        )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> dict:
    return {
        "策略": frame["策略"].iloc[0],
        "窗口数": len(frame),
        "跑赢006503次数": int(frame["是否跑赢006503"].sum()),
        "跑赢最强全仓次数": int(frame["是否跑赢最强全仓"].sum()),
        "平均收益": float(frame["策略收益"].mean()),
        "收益标准差": float(frame["策略收益"].std()),
        "平均最大回撤": float(frame["策略最大回撤"].mean()),
        "最差最大回撤": float(frame["策略最大回撤"].min()),
        "最低窗口收益": float(frame["策略收益"].min()),
    }


def pretest_selected_positive_short(navs: pd.DataFrame) -> tuple[tuple[int, int, bool], pd.DataFrame]:
    """Select one fixed parameter set using only data before the 30 test windows."""
    first_test_start = navs.index[len(navs) - 30 - WINDOW_DAYS + 1]
    candidates = list(itertools.product([1, 2, 3, 5, 8, 10], [1, 2, 3, 5, 8, 10], [True]))
    returns_cache = {params: returns_from_choice(navs, top1_momentum_choice(navs, *params)) for params in candidates}

    train_windows = []
    for end_pos in range(WINDOW_DAYS - 1, len(navs)):
        start_pos = end_pos - WINDOW_DAYS + 1
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        if end < first_test_start:
            train_windows.append((start, end))
    train_windows = train_windows[-240:]

    scored = []
    for params, strategy_return in returns_cache.items():
        values = []
        wins = 0
        for start, end in train_windows:
            curve = (1 + strategy_return.loc[start:end]).cumprod()
            ret = float(curve.iloc[-1] - 1)
            values.append(ret)
            all_006503 = float(navs.loc[start:end, "006503"].iloc[-1] / navs.loc[start:end, "006503"].iloc[0] - 1)
            wins += ret > all_006503
        scored.append(
            {
                "参数": params,
                "训练窗口数": len(train_windows),
                "训练跑赢006503次数": wins,
                "训练平均收益": float(np.mean(values)),
                "训练收益标准差": float(np.std(values)),
                "训练评分": wins + float(np.mean(values) - 0.25 * np.std(values)),
            }
        )
    score_frame = pd.DataFrame(scored).sort_values(["训练跑赢006503次数", "训练平均收益"], ascending=False)
    selected = tuple(score_frame.iloc[0]["参数"])
    rows = evaluate_params(navs, selected)
    rows["策略"] = f"测试前选参Top1动量_L{selected[0]}_确认{selected[1]}_只追正收益{selected[2]}"
    score_frame["参数"] = score_frame["参数"].map(lambda item: f"L{item[0]}_确认{item[1]}_只追正收益{item[2]}")
    score_frame.to_csv(OUT / "pretest_positive_short_training_scores_cn.csv", index=False, encoding="utf-8-sig")
    return selected, rows


def main() -> None:
    navs = pd.DataFrame({code: read_cached_nav(code) for code in CODES}).dropna().sort_index()

    fixed_params = (2, 5, True)
    fixed_rows = evaluate_params(navs, fixed_params)
    selected_params, pretest_rows = pretest_selected_positive_short(navs)

    candidates = list(itertools.product([1, 2, 3, 5, 8, 10, 15, 20, 30, 40, 60], [1, 2, 3, 5, 8, 10], [False, True]))
    walk_rows = walk_forward_rows(navs, candidates)

    fixed_rows.to_csv(OUT / "fixed_short_momentum_windows_cn.csv", index=False, encoding="utf-8-sig")
    pretest_rows.to_csv(OUT / "pretest_selected_positive_short_windows_cn.csv", index=False, encoding="utf-8-sig")
    walk_rows.to_csv(OUT / "walk_forward_windows_cn.csv", index=False, encoding="utf-8-sig")

    summary = pd.DataFrame([summarize(fixed_rows), summarize(pretest_rows), summarize(walk_rows)])
    summary.to_csv(OUT / "threshold_summary_cn.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(fixed_rows["窗口"], fixed_rows["策略收益"], marker="o", lw=1.8, label="固定短周期Top1动量")
    ax.plot(pretest_rows["窗口"], pretest_rows["策略收益"], marker="o", lw=1.8, label="测试前选参Top1动量")
    ax.plot(walk_rows["窗口"], walk_rows["策略收益"], marker="o", lw=1.8, label="无污染Walk-forward选参")
    ax.plot(fixed_rows["窗口"], fixed_rows["006503全仓收益"], color="#d62728", lw=1.8, label="006503全仓")
    ax.plot(fixed_rows["窗口"], fixed_rows["最强全仓收益"], color="#333333", lw=1.6, ls="--", label="窗口内事后最强全仓")
    ax.axhline(0, color="#999999", lw=0.8)
    ax.set_title("30个三个月滑动窗口：策略收益 vs 全仓基准", fontsize=15)
    ax.set_xlabel("窗口编号")
    ax.set_ylabel("区间收益率")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "threshold_windows_curve_cn.svg", format="svg")
    plt.close(fig)

    lines = [
        "# 三个月滑动窗口门槛审计",
        "",
        "## 验收标准",
        "",
        "你提出的标准是：30个三个月滑动窗口里，策略至少一半窗口要优于全仓。这里我分两层审计：",
        "",
        "- 跑赢 `006503` 全仓：这是当前最强主线基金之一，属于可执行的强基准。",
        "- 跑赢窗口内事后最强全仓：这是更苛刻、带有事后选择优势的基准。",
        "",
        "我额外加入了一版“测试前选参”：候选族预先限定为短周期Top1动量、只追正收益基金，然后只用30个测试窗口开始前的历史窗口选参数。",
        f"这套测试前流程选中的参数是：`{selected_params[0]}交易日动量 + 连续{selected_params[1]}日确认 + 只追正收益={selected_params[2]}`。",
        "",
        "## 两个结果",
        "",
        "| 策略 | 跑赢006503次数 | 跑赢最强全仓次数 | 平均收益 | 收益标准差 | 平均最大回撤 | 最差最大回撤 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for _, row in summary.iterrows():
        lines.append(
            f"| {row['策略']} | {int(row['跑赢006503次数'])}/{int(row['窗口数'])} | {int(row['跑赢最强全仓次数'])}/{int(row['窗口数'])} | "
            f"{pct(row['平均收益'])} | {pct(row['收益标准差'])} | {pct(row['平均最大回撤'])} | {pct(row['最差最大回撤'])} |"
        )
    lines += [
        "",
        "## 解释",
        "",
        "- 固定短周期Top1动量参数为：2交易日动量、只追正收益基金、连续5日确认、两日净值延迟执行、每次切换扣0.15%。它在这30个窗口里跑赢006503达到18/30，满足你说的“一半以上”。",
        "- 但这个参数是看过这批窗口后从候选族里挑出来的，所以不能视为已经通过泛化检验。",
        "- 更严格的Walk-forward版本，每个测试窗口只用之前历史窗口选参数，只有8/30跑赢006503，说明当前还没有真正找到足够通用的无污染策略。",
        "- 跑赢事后最强全仓更难：固定短周期策略为12/30，Walk-forward只有2/30。这个基准本身含有事后最优选择优势，不能要求实盘策略长期稳定超过它。",
        "",
        "## 我的判断",
        "",
        "可以把固定短周期Top1动量作为下一轮候选，但不能直接部署为最终模型。真正值得推进的是：保留它的短周期风格切换能力，再加入不依赖测试窗口调参的稳健约束，例如固定参数族投票、成交拥挤度过滤、板块广度确认和回撤状态机。",
    ]
    (OUT / "threshold_audit_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
