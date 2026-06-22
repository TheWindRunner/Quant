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


OUT = Path("output/factor_gate_rotation")
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
FACTOR_MAP = {
    "006503": "memory_semiconductor_proxy",
    "007817": "cpo_communication",
    "008887": "memory_semiconductor_proxy",
    "720001": "pcb_quality_selection",
}
ANCHOR = "011370"
COST = 0.0015
WINDOW_DAYS = 63


def pct(value: float) -> str:
    return f"{value * 100:.2f}%"


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def load_navs() -> pd.DataFrame:
    return pd.DataFrame({code: read_cached_nav(code) for code in CODES}).dropna().sort_index()


def load_factor(asset: str) -> pd.DataFrame:
    path = Path("data/external_factors") / f"{asset}.csv"
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    if "etf_close" not in frame.columns and "etf_change_pct" in frame.columns:
        change = frame["etf_change_pct"].fillna(0.0) / 100
        frame["etf_close"] = (1 + change).cumprod()
    return frame


def build_factor_gates(navs: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, pd.DataFrame] = {}
    for code, asset in FACTOR_MAP.items():
        factor = load_factor(asset).reindex(navs.index).ffill()
        if factor.empty or "etf_change_pct" not in factor.columns:
            continue
        change = factor["etf_change_pct"].fillna(0.0) / 100
        close = factor["etf_close"].ffill() if "etf_close" in factor.columns else (1 + change).cumprod()
        amount = factor.get("etf_amount", pd.Series(np.nan, index=navs.index))
        amount_ratio = amount / amount.rolling(20).mean()
        gate = pd.DataFrame(index=navs.index)
        gate["mild"] = (
            (change.rolling(2).sum() > -0.01)
            & (close > close.rolling(10).mean() * 0.95)
        ).fillna(False)
        gate["strong"] = (
            (change.rolling(2).sum() > 0)
            | ((close > close.rolling(20).mean()) & (amount_ratio > 0.9))
        ).fillna(False)
        gate["volume_confirm"] = (
            (change.rolling(2).sum() > -0.005)
            & ((amount_ratio > 1.05) | (close > close.rolling(20).mean()))
        ).fillna(False)
        out[code] = gate
    return out


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


def top1_choice(navs: pd.DataFrame, lookback: int, confirm: int, positive_only: bool, gate_name: str | None) -> pd.Series:
    momentum = navs.pct_change(lookback, fill_method=None)
    if positive_only:
        momentum = momentum.where(momentum > 0)
    gates = build_factor_gates(navs) if gate_name else {}
    columns = np.array(navs.columns)
    raw = []
    for date, row in momentum.iterrows():
        values = row.copy()
        if gate_name:
            for code in values.index:
                if code in gates and not bool(gates[code].loc[date, gate_name]):
                    values.loc[code] = np.nan
        valid = values.dropna()
        raw.append(valid.idxmax() if not valid.empty else ANCHOR)
    return confirm_choice(pd.Series(raw, index=navs.index), confirm)


def returns_from_choice(navs: pd.DataFrame, choice: pd.Series) -> pd.Series:
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    position = choice.shift(2).fillna(ANCHOR).to_numpy(dtype=object)
    strategy_return = np.zeros(len(position))
    for code in navs.columns:
        strategy_return += (position == code) * returns[code].to_numpy()
    cost = (choice != choice.shift()).astype(float).shift(1).fillna(0.0).to_numpy() * COST
    return pd.Series(strategy_return - cost, index=navs.index)


def evaluate_windows(navs: pd.DataFrame, strategy_return: pd.Series, label: str) -> pd.DataFrame:
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


def pretest_select(navs: pd.DataFrame, gate_options: list[str | None]) -> tuple[tuple[int, int, bool, str | None], pd.DataFrame]:
    first_test_start = navs.index[len(navs) - 30 - WINDOW_DAYS + 1]
    candidates = list(itertools.product([1, 2, 3, 5, 8, 10], [1, 2, 3, 5, 8, 10], [True], gate_options))
    cache = {params: returns_from_choice(navs, top1_choice(navs, *params)) for params in candidates}

    train_windows = []
    for end_pos in range(WINDOW_DAYS - 1, len(navs)):
        start_pos = end_pos - WINDOW_DAYS + 1
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        if end < first_test_start:
            train_windows.append((start, end))
    train_windows = train_windows[-240:]

    rows = []
    for params, strategy_return in cache.items():
        values = []
        wins = 0
        for start, end in train_windows:
            curve = (1 + strategy_return.loc[start:end]).cumprod()
            ret = float(curve.iloc[-1] - 1)
            values.append(ret)
            all_006503 = float(navs.loc[start:end, "006503"].iloc[-1] / navs.loc[start:end, "006503"].iloc[0] - 1)
            wins += ret > all_006503
        rows.append(
            {
                "params": params,
                "train_windows": len(train_windows),
                "train_wins_006503": wins,
                "train_mean_return": float(np.mean(values)),
                "train_std_return": float(np.std(values)),
                "train_score": wins + float(np.mean(values) - 0.25 * np.std(values)),
            }
        )
    score = pd.DataFrame(rows).sort_values(["train_wins_006503", "train_mean_return"], ascending=False)
    selected = tuple(score.iloc[0]["params"])
    score["params"] = score["params"].map(lambda item: f"L{item[0]}_confirm{item[1]}_positive{item[2]}_gate{item[3] or 'none'}")
    return selected, score


def main() -> None:
    navs = load_navs()
    variants = [
        ("无ETF闸门", (2, 5, True, None)),
        ("温和ETF闸门", (2, 5, True, "mild")),
        ("强ETF闸门", (2, 5, True, "strong")),
        ("放量确认闸门", (2, 5, True, "volume_confirm")),
    ]
    all_rows = []
    summary_rows = []
    for label, params in variants:
        choice = top1_choice(navs, *params)
        rows = evaluate_windows(navs, returns_from_choice(navs, choice), label)
        all_rows.append(rows)
        summary_rows.append(summarize(rows))

    selected, score = pretest_select(navs, [None, "mild", "strong", "volume_confirm"])
    score.to_csv(OUT / "pretest_training_scores_cn.csv", index=False, encoding="utf-8-sig")
    selected_label = f"测试前选参：L{selected[0]} 确认{selected[1]} 闸门{selected[3] or '无'}"
    selected_rows = evaluate_windows(navs, returns_from_choice(navs, top1_choice(navs, *selected)), selected_label)
    all_rows.append(selected_rows)
    summary_rows.append(summarize(selected_rows))

    detail = pd.concat(all_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows).sort_values(["wins_vs_006503", "mean_return"], ascending=False)
    detail.to_csv(OUT / "factor_gate_window_detail_cn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "factor_gate_summary_cn.csv", index=False, encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(15, 7))
    for label in summary["strategy"].tolist():
        sub = detail[detail["strategy"] == label]
        ax.plot(sub["window"], sub["return"], marker="o", lw=1.6, label=label)
    base = detail[detail["strategy"] == summary["strategy"].iloc[0]]
    ax.plot(base["window"], base["all_006503"], color="#d62728", lw=1.8, label="006503全仓")
    ax.plot(base["window"], base["best_all"], color="#333333", lw=1.3, ls="--", label="窗口事后最强全仓")
    ax.set_title("ETF/板块因子闸门：30个三个月窗口收益对比", fontsize=15)
    ax.set_xlabel("窗口编号")
    ax.set_ylabel("区间收益率")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(OUT / "factor_gate_windows_curve_cn.svg", format="svg")
    plt.close(fig)

    lines = [
        "# ETF/板块因子闸门增强测试",
        "",
        "## 目的",
        "",
        "在已经通过门槛的短周期Top1动量策略上，加入ETF/板块代理因子闸门，观察是否能进一步提高30个三个月窗口的胜率或降低回撤。",
        "",
        "核心策略仍是：2交易日动量Top1、只追正收益基金、连续5日确认、两日净值延迟执行、换仓扣0.15%。",
        "",
        "## 闸门定义",
        "",
        "- 温和ETF闸门：对应ETF/板块代理近2日跌幅不超过1%，且价格不低于10日均线的95%。",
        "- 强ETF闸门：近2日ETF/板块代理收益为正，或价格高于20日均线且成交额不弱。",
        "- 放量确认闸门：近2日不明显走弱，同时成交额放量或价格高于20日均线。",
        "- 没有可靠ETF代理的基金不强行加闸门；目前008585的AI外部ETF数据质量不足，因此仍以基金净值为主。",
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
        "如果ETF闸门没有提高胜率，说明当前阶段主要alpha来自基金净值的极短期相对强弱，而不是这些粗粒度ETF代理。接下来应优先补更高质量的14:30估值、成分股广度和主题成交额，而不是继续堆同一类ETF收盘因子。",
    ]
    (OUT / "factor_gate_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
