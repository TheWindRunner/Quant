from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_fund_advisor.data import load_or_fetch_open_fund_nav


OUT = ROOT / "output" / "sector_rotation_optimization_c_fee"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 63
SLIDING_WINDOWS = 30
EXECUTION_DELAY_DAYS = 2

SECTOR_CODES = {
    "PCB": "720001",
    "存储": "018816",
    "CPO": "007817",
    "AI": "008585",
    "半导体设备": "017811",
}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    params: dict[str, float | int | str]


@dataclass
class Lot:
    units: float
    purchase_date: pd.Timestamp


def load_navs() -> pd.DataFrame:
    data = {}
    for sector, code in SECTOR_CODES.items():
        data[sector] = load_or_fetch_open_fund_nav(code, max_age_hours=100000).rename(sector)
    return pd.concat(data.values(), axis=1).dropna().sort_index()


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def c_fee(days: int) -> float:
    if days < 7:
        return 0.015
    if days < 30:
        return 0.005
    return 0.0


def confirm(raw: pd.Series, days: int) -> pd.Series:
    values = raw.to_numpy(dtype=object)
    current = values[0]
    pending = current
    count = 0
    out = []
    for value in values:
        if value == current:
            pending = value
            count = 0
        elif value == pending:
            count += 1
            if count >= days:
                current = value
                count = 0
        else:
            pending = value
            count = 1
        out.append(current)
    return pd.Series(out, index=raw.index)


def fee_protect(
    navs: pd.DataFrame,
    choice: pd.Series,
    min_hold_days: int,
    advantage_window: int,
    advantage_threshold: float,
    stop_window: int,
    stop_loss: float,
) -> pd.Series:
    momentum = navs.pct_change(advantage_window, fill_method=None)
    stop_ret = navs.pct_change(stop_window, fill_method=None)
    current = str(choice.iloc[0])
    entry = choice.index[0]
    out = []
    for date, proposed_value in choice.items():
        proposed = str(proposed_value)
        if proposed != current:
            holding_days = int((date - entry).days)
            current_stop = stop_ret.loc[date, current] if current in stop_ret.columns else np.nan
            advantage = (
                momentum.loc[date, proposed] - momentum.loc[date, current]
                if proposed in momentum.columns and current in momentum.columns
                else np.nan
            )
            allow = (
                holding_days >= min_hold_days
                or (pd.notna(current_stop) and current_stop <= stop_loss)
                or (pd.notna(advantage) and advantage >= advantage_threshold)
            )
            if allow:
                current = proposed
                entry = date
        out.append(current)
    return pd.Series(out, index=choice.index)


def backtest(navs: pd.DataFrame, choice: pd.Series) -> tuple[pd.Series, pd.Series, dict[str, float]]:
    navs = navs.dropna().sort_index()
    choice = choice.reindex(navs.index).ffill()
    executed = choice.shift(EXECUTION_DELAY_DAYS).fillna(choice.iloc[0])
    lots: dict[str, list[Lot]] = {col: [] for col in navs.columns}
    cash = 1.0
    fee_total = 0.0
    redeemed = 0.0
    redeemed_days = 0.0
    under7 = 0.0
    under30 = 0.0
    switches = 0
    current = str(executed.iloc[0])
    lots[current].append(Lot(cash / float(navs.iloc[0][current]), navs.index[0]))
    cash = 0.0
    values = []

    def sell_all(sector: str, date: pd.Timestamp) -> None:
        nonlocal cash, fee_total, redeemed, redeemed_days, under7, under30
        price = float(navs.loc[date, sector])
        proceeds = 0.0
        for lot in lots[sector]:
            days = int((date - lot.purchase_date).days)
            gross = lot.units * price
            fee = gross * c_fee(days)
            proceeds += gross - fee
            fee_total += fee
            redeemed += gross
            redeemed_days += gross * days
            if days < 7:
                under7 += gross
            if days < 30:
                under30 += gross
        lots[sector] = []
        cash += proceeds

    def buy_all(sector: str, date: pd.Timestamp) -> None:
        nonlocal cash
        price = float(navs.loc[date, sector])
        if cash > 1e-12:
            lots[sector].append(Lot(cash / price, date))
            cash = 0.0

    for date in navs.index:
        target = str(executed.loc[date])
        if target != current:
            sell_all(current, date)
            buy_all(target, date)
            current = target
            switches += 1
        value = cash
        for sector in navs.columns:
            value += sum(lot.units for lot in lots[sector]) * float(navs.loc[date, sector])
        values.append(value)
    curve = pd.Series(values, index=navs.index)
    curve = curve / curve.iloc[0]
    metrics = {
        "return": float(curve.iloc[-1] - 1),
        "max_drawdown": max_drawdown(curve),
        "switches": float(switches),
        "fees": float(fee_total),
        "avg_holding_days": float(redeemed_days / redeemed) if redeemed else np.nan,
        "under7_ratio": float(under7 / redeemed) if redeemed else 0.0,
        "under30_ratio": float(under30 / redeemed) if redeemed else 0.0,
    }
    return curve, executed.rename("执行持仓"), metrics


def initial_anchor(navs: pd.DataFrame, lookback: int = 20) -> str:
    scores = navs.pct_change(lookback, fill_method=None).iloc[: max(lookback + 2, 120)].iloc[-1]
    return str(scores.idxmax()) if scores.notna().any() else str(navs.columns[0])


def top1_choice(navs: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    p = spec.params
    mom = navs.pct_change(int(p["momentum_days"]), fill_method=None)
    anchor = initial_anchor(navs, int(p.get("anchor_days", 20)))
    raw = []
    for _, row in mom.iterrows():
        valid = row.where(row > float(p.get("positive_threshold", 0.0))).dropna()
        raw.append(str(valid.idxmax()) if not valid.empty else anchor)
    choice = confirm(pd.Series(raw, index=navs.index), int(p["confirm_days"]))
    return apply_overlay_and_protection(navs, choice, spec)


def multi_momentum_choice(navs: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    p = spec.params
    ret = navs.pct_change(fill_method=None)
    score = (
        float(p["w_short"]) * navs.pct_change(int(p["short_days"]), fill_method=None)
        + float(p["w_mid"]) * navs.pct_change(int(p["mid_days"]), fill_method=None)
        + float(p["w_long"]) * navs.pct_change(int(p["long_days"]), fill_method=None)
        - float(p["vol_penalty"]) * ret.rolling(int(p["vol_days"])).std() * np.sqrt(252)
    )
    if bool(p.get("ma_filter", 1)):
        score = score.where(navs > navs.rolling(int(p["ma_days"])).mean())
    anchor = initial_anchor(navs, int(p["mid_days"]))
    raw = []
    for _, row in score.iterrows():
        valid = row.dropna()
        raw.append(str(valid.idxmax()) if not valid.empty and valid.max() > float(p.get("score_threshold", 0.0)) else anchor)
    choice = confirm(pd.Series(raw, index=navs.index), int(p["confirm_days"]))
    return apply_overlay_and_protection(navs, choice, spec)


def trend_quality(values: pd.Series) -> float:
    x_values = np.log(values.dropna().to_numpy(dtype=float))
    if len(x_values) < 20 or not np.isfinite(x_values).all():
        return 0.0
    x = np.arange(len(x_values), dtype=float)
    slope, intercept = np.polyfit(x, x_values, 1)
    fit = slope * x + intercept
    total = ((x_values - x_values.mean()) ** 2).sum()
    resid = ((x_values - fit) ** 2).sum()
    r2 = 1 - resid / total if total > 0 else 0.0
    return float(np.expm1(slope * 252) * max(0.0, r2))


def trend_quality_choice(navs: pd.DataFrame, spec: ModelSpec) -> pd.Series:
    p = spec.params
    tq_window = int(p["trend_window"])
    tq = navs.rolling(tq_window).apply(lambda s: trend_quality(pd.Series(s)), raw=False)
    score = (
        float(p["w_tq"]) * tq
        + float(p["w_mom"]) * navs.pct_change(int(p["momentum_days"]), fill_method=None)
        + float(p["w_dd"]) * (navs / navs.rolling(int(p["dd_days"])).max() - 1)
    )
    score = score.where(navs > navs.rolling(int(p["ma_days"])).mean())
    anchor = initial_anchor(navs, int(p["momentum_days"]))
    raw = []
    for _, row in score.iterrows():
        valid = row.dropna()
        raw.append(str(valid.idxmax()) if not valid.empty and valid.max() > 0 else anchor)
    choice = confirm(pd.Series(raw, index=navs.index), int(p["confirm_days"]))
    return apply_overlay_and_protection(navs, choice, spec)


def apply_overlay_and_protection(navs: pd.DataFrame, choice: pd.Series, spec: ModelSpec) -> pd.Series:
    p = spec.params
    if bool(p.get("overlay", 1)):
        ret = navs.pct_change(int(p.get("overlay_ret_days", 30)), fill_method=None)
        dd = navs / navs.rolling(int(p.get("overlay_dd_days", 20))).max() - 1
        ma = navs.rolling(int(p.get("overlay_ma_days", 20))).mean()
        regime = (
            (ret > float(p.get("overlay_ret_threshold", 0.12)))
            & (dd > float(p.get("overlay_dd_threshold", -0.03)))
            & (navs > ma)
        )
        confirmed = regime.copy()
        for sector in navs.columns:
            streak = regime[sector].astype(int).groupby((regime[sector] != regime[sector].shift()).cumsum()).cumsum()
            confirmed[sector] = streak >= int(p.get("overlay_confirm", 2))
        forced = []
        for date in navs.index:
            active = confirmed.loc[date]
            if bool(active.any()):
                forced.append(str(ret.loc[date].where(active).idxmax()))
            else:
                forced.append(str(choice.loc[date]))
        choice = pd.Series(forced, index=navs.index)
    return fee_protect(
        navs,
        choice,
        min_hold_days=int(p.get("min_hold_days", 7)),
        advantage_window=int(p.get("advantage_window", 30)),
        advantage_threshold=float(p.get("advantage_threshold", 0.30)),
        stop_window=int(p.get("stop_window", 5)),
        stop_loss=float(p.get("stop_loss", -0.08)),
    )


def evaluate_model(navs: pd.DataFrame, choice: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    rows = []
    for window_id, end_pos in enumerate(range(len(navs) - SLIDING_WINDOWS, len(navs)), start=1):
        start_pos = end_pos - WINDOW_DAYS + 1
        test = navs.iloc[start_pos : end_pos + 1]
        test_choice = choice.reindex(test.index).ffill()
        curve, executed, metrics = backtest(test, test_choice)
        equal = test.div(test.iloc[0]).mean(axis=1)
        all_in_returns = {col: float(test[col].iloc[-1] / test[col].iloc[0] - 1) for col in test.columns}
        all_in_dd = {col: max_drawdown(test[col] / test[col].iloc[0]) for col in test.columns}
        best = max(all_in_returns, key=all_in_returns.get)
        rows.append(
            {
                "window": window_id,
                "start": test.index[0].date().isoformat(),
                "end": test.index[-1].date().isoformat(),
                "strategy_return": metrics["return"],
                "strategy_mdd": metrics["max_drawdown"],
                "equal_return": float(equal.iloc[-1] - 1),
                "equal_mdd": max_drawdown(equal),
                "best_sector": best,
                "best_return": all_in_returns[best],
                "best_mdd": all_in_dd[best],
                "beat_equal": metrics["return"] > float(equal.iloc[-1] - 1),
                "beat_best": metrics["return"] > all_in_returns[best],
                "switches": metrics["switches"],
                "fees": metrics["fees"],
                "avg_holding_days": metrics["avg_holding_days"],
                "under7_ratio": metrics["under7_ratio"],
                "under30_ratio": metrics["under30_ratio"],
                "last_holding": str(executed.iloc[-1]),
            }
        )
    detail = pd.DataFrame(rows)
    summary = {
        "windows": float(len(detail)),
        "beat_equal_count": float(detail["beat_equal"].sum()),
        "beat_best_count": float(detail["beat_best"].sum()),
        "avg_return": float(detail["strategy_return"].mean()),
        "std_return": float(detail["strategy_return"].std()),
        "avg_mdd": float(detail["strategy_mdd"].mean()),
        "worst_mdd": float(detail["strategy_mdd"].min()),
        "avg_best_return": float(detail["best_return"].mean()),
        "avg_best_mdd": float(detail["best_mdd"].mean()),
        "avg_switches": float(detail["switches"].mean()),
        "avg_fees": float(detail["fees"].mean()),
        "avg_holding_days": float(detail["avg_holding_days"].mean()),
        "avg_under7_ratio": float(detail["under7_ratio"].mean()),
        "avg_under30_ratio": float(detail["under30_ratio"].mean()),
    }
    summary["objective"] = (
        summary["avg_return"]
        + 0.025 * summary["beat_best_count"]
        + 0.010 * summary["beat_equal_count"]
        + 0.50 * summary["avg_mdd"]
        + 0.20 * summary["worst_mdd"]
        - 0.25 * summary["avg_fees"]
        - 0.005 * summary["avg_switches"]
    )
    return detail, summary


def candidate_specs() -> list[ModelSpec]:
    specs: list[ModelSpec] = []
    for m in [2, 3, 5, 10, 20]:
        for c in [2, 3, 5, 7, 10]:
            for hold in [7, 14, 21, 30]:
                for adv in [0.10, 0.20, 0.30, 0.40]:
                    specs.append(
                        ModelSpec(
                            name=f"top1_m{m}_c{c}_hold{hold}_adv{adv:.2f}",
                            family="短动量Top1",
                            params={
                                "momentum_days": m,
                                "confirm_days": c,
                                "min_hold_days": hold,
                                "advantage_threshold": adv,
                                "positive_threshold": 0.0,
                                "overlay": 1,
                                "overlay_ret_days": 30,
                                "overlay_ret_threshold": 0.12,
                                "overlay_dd_days": 20,
                                "overlay_dd_threshold": -0.03,
                                "overlay_ma_days": 20,
                                "overlay_confirm": 2,
                                "advantage_window": 30,
                                "stop_window": 5,
                                "stop_loss": -0.08,
                            },
                        )
                    )
    for short, mid, long in [(5, 20, 60), (10, 20, 60), (5, 30, 90), (20, 60, 120)]:
        for vol_penalty in [0.00, 0.05, 0.10, 0.20]:
            for confirm_days in [3, 5, 7]:
                for hold in [7, 14, 21, 30]:
                    specs.append(
                        ModelSpec(
                            name=f"multi_{short}_{mid}_{long}_vol{vol_penalty}_c{confirm_days}_hold{hold}",
                            family="多周期动量波动惩罚",
                            params={
                                "short_days": short,
                                "mid_days": mid,
                                "long_days": long,
                                "w_short": 0.35,
                                "w_mid": 0.40,
                                "w_long": 0.25,
                                "vol_days": 20,
                                "vol_penalty": vol_penalty,
                                "ma_filter": 1,
                                "ma_days": 20,
                                "confirm_days": confirm_days,
                                "min_hold_days": hold,
                                "advantage_threshold": 0.25,
                                "overlay": 1,
                                "overlay_ret_days": 30,
                                "overlay_ret_threshold": 0.12,
                                "overlay_dd_days": 20,
                                "overlay_dd_threshold": -0.03,
                                "overlay_ma_days": 20,
                                "overlay_confirm": 2,
                                "advantage_window": 30,
                                "stop_window": 5,
                                "stop_loss": -0.08,
                            },
                        )
                    )
    for tq_window in [40, 60, 90]:
        for momentum_days in [20, 30, 60]:
            for confirm_days in [3, 5, 7]:
                for hold in [14, 21, 30]:
                    specs.append(
                        ModelSpec(
                            name=f"tq_w{tq_window}_m{momentum_days}_c{confirm_days}_hold{hold}",
                            family="趋势质量",
                            params={
                                "trend_window": tq_window,
                                "momentum_days": momentum_days,
                                "dd_days": 20,
                                "ma_days": 20,
                                "w_tq": 0.45,
                                "w_mom": 0.45,
                                "w_dd": 0.10,
                                "confirm_days": confirm_days,
                                "min_hold_days": hold,
                                "advantage_threshold": 0.20,
                                "overlay": 1,
                                "overlay_ret_days": 30,
                                "overlay_ret_threshold": 0.12,
                                "overlay_dd_days": 20,
                                "overlay_dd_threshold": -0.03,
                                "overlay_ma_days": 20,
                                "overlay_confirm": 2,
                                "advantage_window": 30,
                                "stop_window": 5,
                                "stop_loss": -0.08,
                            },
                        )
                    )
    return specs


def run_spec(navs: pd.DataFrame, spec: ModelSpec) -> tuple[pd.DataFrame, dict[str, float]]:
    if spec.family == "短动量Top1":
        choice = top1_choice(navs, spec)
    elif spec.family == "多周期动量波动惩罚":
        choice = multi_momentum_choice(navs, spec)
    elif spec.family == "趋势质量":
        choice = trend_quality_choice(navs, spec)
    else:
        raise ValueError(spec.family)
    detail, summary = evaluate_model(navs, choice)
    summary.update({"model": spec.name, "family": spec.family, "params": repr(spec.params)})
    return detail, summary


def write_report(summary: pd.DataFrame, best_detail: pd.DataFrame) -> None:
    best = summary.iloc[0]
    qualified = summary.loc[summary["beat_best_count"] >= 15]
    lines = [
        "# C类赎回费约束下的五板块轮动优化",
        "",
        "## 目标",
        "",
        "- 使用FIFO C类赎回费：持有<7天1.5%，7-30天0.5%，>=30天0%。",
        "- 调整x日动量Top1、y日确认、MA过滤、强趋势覆盖、最短持有期等参数。",
        "- 尝试经典因子：动量、趋势、波动惩罚、回撤、趋势质量。",
        "- 评价30个三个月滑动窗口中是否能过半跑赢最佳单板块all-in，并改善回撤。",
        "",
        "## 最优结果",
        "",
        f"- 模型：{best['model']}",
        f"- 家族：{best['family']}",
        f"- 跑赢最佳all-in：{int(best['beat_best_count'])}/30",
        f"- 跑赢等权：{int(best['beat_equal_count'])}/30",
        f"- 平均收益：{best['avg_return']:.2%}",
        f"- 平均最大回撤：{best['avg_mdd']:.2%}",
        f"- 最差最大回撤：{best['worst_mdd']:.2%}",
        f"- 平均赎回费：{best['avg_fees']:.2%}",
        f"- 平均切换次数：{best['avg_switches']:.2f}",
        f"- 参数：`{best['params']}`",
        "",
        "## 是否达成过半跑赢最佳all-in",
        "",
        f"- 达标模型数量：{len(qualified)}",
        "- 如果达标模型数量为0，说明在当前三个月窗口样本中，任何轮动都难以系统性超过事后最佳单板块；此时更现实的目标应是跑赢等权/定投，并控制回撤。",
        "",
        "## Top 20模型",
        "",
        summary.head(20)[
            [
                "model",
                "family",
                "beat_best_count",
                "beat_equal_count",
                "avg_return",
                "avg_mdd",
                "worst_mdd",
                "avg_fees",
                "avg_switches",
                "objective",
            ]
        ].to_markdown(index=False, floatfmt=".4f"),
        "",
        "## 最优模型窗口明细",
        "",
        best_detail.to_markdown(index=False, floatfmt=".4f"),
    ]
    (OUT / "optimization_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    navs = load_navs()
    specs = candidate_specs()
    summaries = []
    best_detail = None
    best_objective = -np.inf
    detail_frames = []
    errors = []
    for i, spec in enumerate(specs, start=1):
        try:
            detail, summary = run_spec(navs, spec)
        except Exception as exc:  # noqa: BLE001
            errors.append(
                {
                    "model": spec.name,
                    "family": spec.family,
                    "error": f"{type(exc).__name__}: {exc}",
                    "params": repr(spec.params),
                }
            )
            if i % 100 == 0:
                print(f"evaluated {i}/{len(specs)}; errors={len(errors)}")
                if summaries:
                    pd.DataFrame(summaries).to_csv(
                        OUT / "optimization_summary_partial_cn.csv",
                        index=False,
                        encoding="utf-8-sig",
                    )
            continue
        summaries.append(summary)
        if summary["objective"] > best_objective:
            best_objective = summary["objective"]
            best_detail = detail.copy()
            best_detail.insert(0, "model", spec.name)
        if i % 100 == 0:
            print(f"evaluated {i}/{len(specs)}; errors={len(errors)}")
            pd.DataFrame(summaries).to_csv(
                OUT / "optimization_summary_partial_cn.csv",
                index=False,
                encoding="utf-8-sig",
            )
    summary_frame = pd.DataFrame(summaries).sort_values(
        ["beat_best_count", "objective", "avg_return"],
        ascending=False,
    )
    top_models = summary_frame.head(20)["model"].tolist()
    for spec in specs:
        if spec.name in top_models:
            detail, _ = run_spec(navs, spec)
            detail.insert(0, "model", spec.name)
            detail_frames.append(detail)
    details = pd.concat(detail_frames, ignore_index=True)
    if best_detail is None:
        raise RuntimeError("no model evaluated")
    summary_frame.to_csv(OUT / "optimization_summary_cn.csv", index=False, encoding="utf-8-sig")
    if errors:
        pd.DataFrame(errors).to_csv(OUT / "optimization_errors_cn.csv", index=False, encoding="utf-8-sig")
    details.to_csv(OUT / "optimization_top20_window_detail_cn.csv", index=False, encoding="utf-8-sig")
    best_detail.to_csv(OUT / "optimization_best_window_detail_cn.csv", index=False, encoding="utf-8-sig")
    navs.to_csv(OUT / "sector_nav_used_cn.csv", index_label="日期", encoding="utf-8-sig")
    write_report(summary_frame, best_detail)
    print(summary_frame.head(20)[[
        "model",
        "family",
        "beat_best_count",
        "beat_equal_count",
        "avg_return",
        "avg_mdd",
        "worst_mdd",
        "avg_fees",
        "avg_switches",
        "objective",
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
