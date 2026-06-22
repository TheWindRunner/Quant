"""Goal-oriented strategy validation for the current fund-timing objective.

The acceptance test is intentionally strict: a candidate must be evaluated on
both the trailing two years and the trailing three months, against daily DCA,
named all-in baselines, and the ex-post strongest all-in fund in the universe.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .action_report import _fetch_tencent_minute_point, _fetch_tencent_quote
from .backtest import performance_metrics
from .intraday_estimate import fetch_eastmoney_intraday_estimate
from .meta_strategy import FUND_CODES
from .style_timing import read_cached_nav, read_external_factors


EXPANDED_FUND_CODES = {
    "huashang_balanced_growth_a": "011369",
    "caitong_integrated_circuit_a": "006502",
    "caitong_value_momentum_a": "720001",
    "caitong_growth_selection_a": "001480",
    "dacheng_tech_innovation_a": "008988",
    "xinao_performance_driver_a": "016370",
}

GOAL_LIVE_FUND_CODE = "006502"
GOAL_LIVE_FUND_NAME = "财通集成电路产业股票A"
GOAL_LIVE_ETF_CODE = "sz159995"
GOAL_LIVE_ETF_SYMBOL = "159995"


def load_goal_navs() -> pd.DataFrame:
    return pd.DataFrame(
        {asset: read_cached_nav(code) for asset, code in FUND_CODES.items()}
    ).dropna()


def load_expanded_goal_navs() -> pd.DataFrame:
    return pd.DataFrame(
        {asset: read_cached_nav(code) for asset, code in EXPANDED_FUND_CODES.items()}
    ).dropna()


def _daily_dca_returns(navs: pd.DataFrame, index: pd.Index) -> pd.Series:
    units = pd.Series(0.0, index=navs.columns)
    values = []
    invested = 0.0
    for date in index:
        contribution = 1.0 / len(index)
        per_asset = contribution / len(navs.columns)
        units += per_asset * (1 - 0.0015) / navs.loc[date]
        invested += contribution
        values.append(float((units * navs.loc[date]).sum() / invested))
    equity = pd.Series(values, index=index)
    return equity.pct_change(fill_method=None).fillna(0.0)


def _topn_momentum_returns(
    navs: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int,
    topn: int,
    fee: float,
) -> tuple[pd.Series, pd.DataFrame]:
    score = navs.pct_change(lookback).shift(1)
    ranks = score.rank(axis=1, ascending=False, method="first")
    weights = (ranks <= topn).astype(float) / topn
    weights = weights.where(score.notna(), 0.0).ffill().fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum())
    strategy_returns = (weights * returns).sum(axis=1) - turnover * fee
    return strategy_returns.rename(f"mom{lookback}_top{topn}"), weights


def _pcb_core_satellite_returns(
    navs: pd.DataFrame,
    returns: pd.DataFrame,
    satellite_weight: float,
    lookback: int,
    threshold: float,
    fee: float,
) -> tuple[pd.Series, pd.DataFrame]:
    satellites = ["cpo", "memory", "ai", "chemical", "nonferrous"]
    relative = navs.div(navs["pcb"], axis=0).pct_change(lookback)[satellites]
    trend = navs.pct_change(lookback)[satellites]
    scope = relative.where(trend > 0)
    ranks = scope.rank(axis=1, ascending=False, method="first")
    best_edge = scope.max(axis=1)
    satellite_mask = (ranks.eq(1)).astype(float).where(best_edge.gt(threshold), 0.0)

    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    weights[satellites] = satellite_mask * satellite_weight
    weights["pcb"] = 1.0 - weights[satellites].sum(axis=1)
    weights = weights.shift(1).ffill().fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum())
    strategy_returns = (weights * returns).sum(axis=1) - turnover * fee
    return (
        strategy_returns.rename(
            f"pcb_core_sat{satellite_weight:g}_look{lookback}_thr{threshold:g}"
        ),
        weights,
    )


def _relative_acceleration_returns(
    navs: pd.DataFrame,
    returns: pd.DataFrame,
    leader: str = "caitong_integrated_circuit_a",
    anchor: str = "huashang_balanced_growth_a",
    fast: int = 10,
    slow: int = 40,
    confirm: int = 3,
    fee: float = 0.003,
) -> tuple[pd.Series, pd.DataFrame]:
    relative = navs[leader] / navs[anchor]
    signal = (relative.pct_change(fast).shift(1) > 0) & (
        relative.pct_change(slow).shift(1) > -0.02
    )
    good_streak = signal.astype(int).groupby((signal != signal.shift()).cumsum()).cumsum()
    bad_streak = (~signal).astype(int).groupby((signal == signal.shift()).cumsum()).cumsum()

    current = anchor
    choices = []
    for date in navs.index:
        if current != leader and good_streak.loc[date] >= confirm:
            current = leader
        elif current != anchor and bad_streak.loc[date] >= confirm:
            current = anchor
        choices.append(current)

    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    for date, asset in zip(navs.index, choices):
        weights.loc[date, asset] = 1.0
    turnover = weights.diff().abs().sum(axis=1).fillna(weights.iloc[0].abs().sum())
    strategy_returns = (weights * returns).sum(axis=1) - turnover * fee
    return (
        strategy_returns.rename(
            f"relative_acceleration_{leader}_vs_{anchor}_f{fast}_s{slow}_c{confirm}"
        ),
        weights,
    )


def _same_day_etf_guard_returns(
    navs: pd.DataFrame,
    returns: pd.DataFrame,
    asset: str = "caitong_integrated_circuit_a",
    factor_asset: str = "memory_semiconductor_proxy",
    down_threshold_pct: float = -3.0,
    low_position: float = 0.0,
    fee: float = 0.003,
) -> tuple[pd.Series, pd.DataFrame]:
    """Use same-day ETF proxy movement as a before-cutoff fund estimate signal.

    This is a 14:30-style strategy: it assumes the same-day ETF proxy has already
    revealed a reliable large-down direction before the open-fund subscription
    cutoff. Historical daily ETF closes are used as a proxy for that intraday
    estimate, so this result should be treated as optimistic until real 14:30
    captures are accumulated.
    """
    factors = read_external_factors(factor_asset).reindex(navs.index).ffill()
    etf_change = factors.get("etf_change_pct", pd.Series(0.0, index=navs.index)).fillna(0.0)
    position = pd.Series(1.0, index=navs.index)
    position[etf_change <= down_threshold_pct] = low_position
    weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    # Open-fund orders submitted before the cutoff are filled at same-day NAV.
    # Existing holdings still earn that same-day NAV move; the new target only
    # affects exposure from the next NAV interval onward. Using same-day target
    # for same-day returns would incorrectly let the strategy dodge a drop it
    # only observes near 14:30.
    decision_weights = pd.DataFrame(0.0, index=navs.index, columns=navs.columns)
    decision_weights[asset] = position.clip(0.0, 1.0)
    weights[asset] = decision_weights[asset].shift(1).fillna(decision_weights[asset].iloc[0])
    turnover = decision_weights.diff().abs().sum(axis=1).fillna(
        decision_weights.iloc[0].abs().sum()
    )
    strategy_returns = (weights * returns).sum(axis=1) - turnover * fee
    return (
        strategy_returns.rename(
            f"same_day_etf_guard_{asset}_down{down_threshold_pct:g}_low{low_position:g}"
        ),
        weights,
    )


def _period_rows(
    name: str,
    returns: pd.Series,
    weights: pd.DataFrame | None,
    navs: pd.DataFrame,
    asset_returns: pd.DataFrame,
    periods: dict[str, pd.Timestamp],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for period, start in periods.items():
        index = navs.index[navs.index >= start]
        strategy_metrics = performance_metrics(returns.reindex(index).fillna(0.0))
        buy_hold = {
            asset: performance_metrics(asset_returns[asset].loc[index])["total_return"]
            for asset in navs.columns
        }
        dca_metrics = performance_metrics(_daily_dca_returns(navs, index))
        row = {
            "strategy": name,
            "period": period,
            "start": index[0],
            "end": index[-1],
            "days": len(index),
            "total_return": strategy_metrics["total_return"],
            "max_drawdown": strategy_metrics["max_drawdown"],
            "sharpe": strategy_metrics["sharpe"],
            "dca_equal_return": dca_metrics["total_return"],
            "cpo_all_in_return": buy_hold.get("cpo", np.nan),
            "pcb_all_in_return": buy_hold.get("pcb", np.nan),
            "best_all_in_asset": max(buy_hold, key=buy_hold.get),
            "best_all_in_return": max(buy_hold.values()),
            "excess_vs_dca": strategy_metrics["total_return"] - dca_metrics["total_return"],
            "excess_vs_cpo_all_in": (
                strategy_metrics["total_return"] - buy_hold["cpo"]
                if "cpo" in buy_hold
                else np.nan
            ),
            "excess_vs_pcb_all_in": (
                strategy_metrics["total_return"] - buy_hold["pcb"]
                if "pcb" in buy_hold
                else np.nan
            ),
            "excess_vs_best_all_in": strategy_metrics["total_return"] - max(buy_hold.values()),
        }
        if weights is not None:
            scoped_weights = weights.reindex(index).fillna(0.0)
            row["avg_pcb_weight"] = float(
                scoped_weights["pcb"].mean() if "pcb" in scoped_weights else np.nan
            )
            row["avg_cpo_weight"] = float(
                scoped_weights["cpo"].mean() if "cpo" in scoped_weights else np.nan
            )
            row["avg_turnover"] = float(scoped_weights.diff().abs().sum(axis=1).mean())
        rows.append(row)
    return rows


def run_goal_research(output_dir: str | Path = "output/goal_research") -> dict[str, Path | pd.DataFrame]:
    navs = load_goal_navs()
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    latest = navs.index.max()
    periods = {
        "trailing_2y": latest - pd.DateOffset(years=2),
        "trailing_3m": latest - pd.DateOffset(months=3),
    }

    strategies: dict[str, tuple[pd.Series, pd.DataFrame | None]] = {}
    for asset in navs.columns:
        strategies[f"all_in_{asset}"] = (returns[asset], None)
    strategies["equal_weight"] = (returns.mean(axis=1), None)

    for lookback in (20, 40, 60, 90):
        for topn in (1, 2, 3):
            strategy_returns, weights = _topn_momentum_returns(
                navs, returns, lookback=lookback, topn=topn, fee=0.003
            )
            strategies[strategy_returns.name] = (strategy_returns, weights)

    for satellite_weight in (0.05, 0.10, 0.20, 0.30):
        for lookback in (10, 20, 40, 60):
            for threshold in (0.0, 0.05, 0.10):
                strategy_returns, weights = _pcb_core_satellite_returns(
                    navs,
                    returns,
                    satellite_weight=satellite_weight,
                    lookback=lookback,
                    threshold=threshold,
                    fee=0.0015,
                )
                strategies[strategy_returns.name] = (strategy_returns, weights)

    rows: list[dict[str, object]] = []
    weight_frames = []
    for name, (strategy_returns, weights) in strategies.items():
        rows.extend(_period_rows(name, strategy_returns, weights, navs, returns, periods))
        if weights is not None:
            frame = weights.copy()
            frame.insert(0, "strategy", name)
            frame.insert(1, "date", frame.index)
            weight_frames.append(frame.reset_index(drop=True))

    detail = pd.DataFrame(rows)
    pivot = detail.pivot_table(
        index="strategy",
        columns="period",
        values=[
            "total_return",
            "max_drawdown",
            "excess_vs_dca",
            "excess_vs_cpo_all_in",
            "excess_vs_pcb_all_in",
            "excess_vs_best_all_in",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{period}" for metric, period in pivot.columns]
    pivot = pivot.reset_index()
    pivot["strict_min_excess_vs_best_all_in"] = pivot[
        ["excess_vs_best_all_in_trailing_2y", "excess_vs_best_all_in_trailing_3m"]
    ].min(axis=1)
    pivot["strict_min_excess_vs_cpo_all_in"] = pivot[
        ["excess_vs_cpo_all_in_trailing_2y", "excess_vs_cpo_all_in_trailing_3m"]
    ].min(axis=1)
    pivot["strict_min_excess_vs_dca"] = pivot[
        ["excess_vs_dca_trailing_2y", "excess_vs_dca_trailing_3m"]
    ].min(axis=1)
    summary = pivot.sort_values(
        ["strict_min_excess_vs_best_all_in", "strict_min_excess_vs_cpo_all_in"],
        ascending=False,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "goal_research_detail.csv"
    summary_path = output / "goal_research_summary.csv"
    weights_path = output / "goal_research_weights.csv"
    report_path = output / "goal_research_report.md"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    if weight_frames:
        pd.concat(weight_frames, ignore_index=True).to_csv(
            weights_path, index=False, encoding="utf-8-sig"
        )

    top = summary.head(12)
    report = [
        "# Goal Research Validation",
        "",
        f"Common fund history: {navs.index.min().date()} to {navs.index.max().date()}",
        "",
        "Acceptance periods:",
        "",
        *[
            f"- {name}: {navs.index[navs.index >= start][0].date()} to {latest.date()}"
            for name, start in periods.items()
        ],
        "",
        "Important: `best_all_in` is the ex-post strongest single fund inside the same universe. "
        "A strategy that cannot beat this line may still beat CPO all-in and DCA, but it has not "
        "satisfied the strict objective.",
        "",
        "## Top Strategies By Strict Excess Vs Best All-In",
        "",
        top.to_markdown(index=False),
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    return {
        "detail": detail,
        "summary": summary,
        "detail_path": detail_path,
        "summary_path": summary_path,
        "weights_path": weights_path,
        "report_path": report_path,
    }


def run_expanded_goal_research(
    output_dir: str | Path = "output/goal_research_expanded",
) -> dict[str, Path | pd.DataFrame]:
    navs = load_expanded_goal_navs()
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    latest = navs.index.max()
    periods = {
        "trailing_2y": latest - pd.DateOffset(years=2),
        "trailing_3m": latest - pd.DateOffset(months=3),
    }

    strategies: dict[str, tuple[pd.Series, pd.DataFrame | None]] = {
        f"all_in_{asset}": (returns[asset], None) for asset in navs.columns
    }
    strategy_returns, weights = _relative_acceleration_returns(navs, returns)
    strategies[strategy_returns.name] = (strategy_returns, weights)
    strategy_returns, weights = _same_day_etf_guard_returns(navs, returns)
    strategies[strategy_returns.name] = (strategy_returns, weights)

    rows: list[dict[str, object]] = []
    for name, (strategy_returns, strategy_weights) in strategies.items():
        rows.extend(
            _period_rows(name, strategy_returns, strategy_weights, navs, returns, periods)
        )

    detail = pd.DataFrame(rows)
    pivot = detail.pivot_table(
        index="strategy",
        columns="period",
        values=[
            "total_return",
            "max_drawdown",
            "excess_vs_dca",
            "excess_vs_best_all_in",
        ],
        aggfunc="first",
    )
    pivot.columns = [f"{metric}_{period}" for metric, period in pivot.columns]
    summary = pivot.reset_index()
    summary["strict_min_excess_vs_best_all_in"] = summary[
        ["excess_vs_best_all_in_trailing_2y", "excess_vs_best_all_in_trailing_3m"]
    ].min(axis=1)
    summary["strict_min_excess_vs_dca"] = summary[
        ["excess_vs_dca_trailing_2y", "excess_vs_dca_trailing_3m"]
    ].min(axis=1)
    summary = summary.sort_values(
        ["strict_min_excess_vs_best_all_in", "strict_min_excess_vs_dca"],
        ascending=False,
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "expanded_goal_research_detail.csv"
    summary_path = output / "expanded_goal_research_summary.csv"
    report_path = output / "expanded_goal_research_report.md"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    report = [
        "# Expanded Goal Research Validation",
        "",
        f"Common fund history: {navs.index.min().date()} to {navs.index.max().date()}",
        "",
        "## Strategy Notes",
        "",
        "- `relative_acceleration_*`: prior-close relative-strength switching between 011369 and 006502.",
        "- `same_day_etf_guard_*`: a conservative 14:30-style semiconductor ETF estimate proxy. The default rule holds 006502, but exits to cash when the semiconductor ETF proxy is down at least 3%; 0.3% switch cost is charged. Historical daily ETF changes are used as a stand-in for the pre-cutoff estimate, so this backtest is still optimistic until real 14:30 capture history is long enough.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
    ]
    report_path.write_text("\n".join(report), encoding="utf-8")
    return {
        "detail": detail,
        "summary": summary,
        "detail_path": detail_path,
        "summary_path": summary_path,
        "report_path": report_path,
    }


def run_etf_guard_robustness(
    output_dir: str | Path = "output/goal_research_robustness",
    simulations: int = 500,
    seed: int = 7,
) -> dict[str, Path | pd.DataFrame]:
    navs = load_expanded_goal_navs()
    returns = navs.pct_change(fill_method=None).fillna(0.0)
    factors = read_external_factors("memory_semiconductor_proxy").reindex(navs.index).ffill()
    etf_change = factors.get("etf_change_pct", pd.Series(0.0, index=navs.index)).fillna(0.0)
    fund_returns = returns["caitong_integrated_circuit_a"]
    latest = navs.index.max()
    periods = {
        "trailing_2y": latest - pd.DateOffset(years=2),
        "trailing_3m": latest - pd.DateOffset(months=3),
    }
    best = {}
    for period, start in periods.items():
        index = navs.index[navs.index >= start]
        best[period] = max(
            performance_metrics(returns[column].loc[index])["total_return"]
            for column in navs.columns
        )

    def evaluate(signal: pd.Series) -> dict[str, float]:
        decision_position = pd.Series(1.0, index=navs.index)
        decision_position[signal] = 0.0
        executed_position = decision_position.shift(1).fillna(decision_position.iloc[0])
        turnover = decision_position.diff().abs().fillna(abs(float(decision_position.iloc[0])))
        strategy_returns = executed_position * fund_returns - turnover * 0.003
        output: dict[str, float] = {}
        for period, start in periods.items():
            index = navs.index[navs.index >= start]
            metrics = performance_metrics(strategy_returns.loc[index])
            output[f"{period}_return"] = metrics["total_return"]
            output[f"{period}_max_drawdown"] = metrics["max_drawdown"]
            output[f"{period}_excess_vs_best_all_in"] = (
                metrics["total_return"] - best[period]
            )
        output["strict_min_excess_vs_best_all_in"] = min(
            output["trailing_2y_excess_vs_best_all_in"],
            output["trailing_3m_excess_vs_best_all_in"],
        )
        return output

    rows: list[dict[str, object]] = []
    for threshold in (-3.0, -3.5, -4.0, -5.0, -6.0):
        signal = etf_change <= threshold
        rows.append(
            {
                "scenario": f"threshold_{threshold:g}",
                "signal_days_2y": int(signal.loc[navs.index >= periods["trailing_2y"]].sum()),
                "signal_days_3m": int(signal.loc[navs.index >= periods["trailing_3m"]].sum()),
                **evaluate(signal),
            }
        )

    base_signal = etf_change <= -3.0
    rng = np.random.default_rng(seed)
    for detect_probability in (0.5, 0.6, 0.7, 0.8, 0.9, 1.0):
        for false_exit_probability in (0.0, 0.01, 0.02, 0.05):
            values = []
            for _ in range(simulations):
                detected = pd.Series(
                    rng.random(len(navs.index)) < detect_probability,
                    index=navs.index,
                )
                false_exit = pd.Series(
                    rng.random(len(navs.index)) < false_exit_probability,
                    index=navs.index,
                )
                signal = (base_signal & detected) | ((~base_signal) & false_exit)
                values.append(evaluate(signal)["strict_min_excess_vs_best_all_in"])
            arr = np.asarray(values, dtype=float)
            rows.append(
                {
                    "scenario": (
                        f"mc_detect{detect_probability:g}"
                        f"_false{false_exit_probability:g}"
                    ),
                    "strict_min_excess_mean": float(arr.mean()),
                    "strict_min_excess_p05": float(np.quantile(arr, 0.05)),
                    "pass_ratio": float((arr > 0).mean()),
                }
            )

    detail = pd.DataFrame(rows)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "etf_guard_robustness.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    return {"detail": detail, "detail_path": detail_path}


def run_goal_live_signal(
    output_dir: str | Path = "output/goal_live_signal",
    cutoff: str = "14:30",
    threshold_pct: float = -3.0,
) -> dict[str, Path | pd.DataFrame]:
    """Generate the executable 14:30 signal for the current goal strategy."""
    now = pd.Timestamp.now(tz="Asia/Shanghai").tz_localize(None)
    cutoff_time = pd.Timestamp(f"{now.date()} {cutoff}")
    quote = _fetch_tencent_quote(GOAL_LIVE_ETF_CODE)
    point = _fetch_tencent_minute_point(GOAL_LIVE_ETF_CODE, cutoff_time)
    etf_change_pct = (point.price / quote.previous_close - 1.0) * 100.0
    fresh_for_cutoff = point.timestamp.date() == cutoff_time.date() and now >= cutoff_time
    if not fresh_for_cutoff:
        target_position = np.nan
        action = "WAIT_FOR_CUTOFF" if now < cutoff_time else "NO_FRESH_DATA"
    else:
        target_position = 0.0 if etf_change_pct <= threshold_pct else 1.0
        action = "SELL_TO_CASH" if target_position == 0.0 else "HOLD_FULL"

    estimate_fields: dict[str, object] = {
        "fund_estimate_change_pct": np.nan,
        "fund_estimate_time": "",
        "fund_estimate_source": "",
        "fund_estimate_error": "",
    }
    try:
        estimate = fetch_eastmoney_intraday_estimate(GOAL_LIVE_FUND_CODE)
        estimate_fields.update(
            {
                "fund_estimate_change_pct": estimate.estimated_change_pct,
                "fund_estimate_time": str(estimate.estimate_time),
                "fund_estimate_source": estimate.source,
            }
        )
    except Exception as exc:
        estimate_fields["fund_estimate_error"] = f"{type(exc).__name__}: {exc}"

    row = {
        "generated_at": now,
        "cutoff": cutoff_time,
        "fund_code": GOAL_LIVE_FUND_CODE,
        "fund_name": GOAL_LIVE_FUND_NAME,
        "etf_code": GOAL_LIVE_ETF_SYMBOL,
        "etf_name": quote.name,
        "etf_previous_close": quote.previous_close,
        "etf_cutoff_timestamp": point.timestamp,
        "etf_cutoff_price": point.price,
        "etf_cutoff_change_pct": etf_change_pct,
        "fresh_for_cutoff": fresh_for_cutoff,
        "threshold_pct": threshold_pct,
        "target_position": target_position,
        "action": action,
        "rule": "hold 006502 unless semiconductor ETF 14:30 proxy is down <= threshold",
        **estimate_fields,
    }
    frame = pd.DataFrame([row])
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / f"{now.date()}_goal_live_signal.csv"
    md_path = output / f"{now.date()}_goal_live_signal.md"
    frame.to_csv(csv_path, index=False, encoding="utf-8-sig")
    md_path.write_text(
        "\n".join(
            [
                "# Goal Live Signal",
                "",
                f"- Generated at: {now}",
                f"- Fund: {GOAL_LIVE_FUND_CODE} {GOAL_LIVE_FUND_NAME}",
                f"- ETF proxy: {GOAL_LIVE_ETF_SYMBOL} {quote.name}",
                f"- Cutoff point: {point.timestamp}, price {point.price:.4f}",
                f"- ETF change vs previous close: {etf_change_pct:.2f}%",
                f"- Fresh for cutoff: {fresh_for_cutoff}",
                f"- Rule threshold: {threshold_pct:.2f}%",
                f"- Action: {action}",
                f"- Target position: {target_position:.0%}",
                "",
                "Note: this is the executable live version of the backtested ETF guard. "
                "The historical backtest used daily ETF changes as a proxy; live use relies on the actual pre-cutoff minute point.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {"detail": frame, "csv_path": csv_path, "report_path": md_path}


def run_corrected_execution_audit(
    output_dir: str | Path = "output/goal_research_corrected_execution",
    threshold_pct: float = -3.0,
    fee: float = 0.003,
) -> dict[str, Path | pd.DataFrame]:
    """Audit the ETF guard with correct open-fund execution timing.

    A sell submitted before the cutoff receives same-day NAV, so existing
    holdings still experience that day's fund return. The sell only removes
    exposure for the next NAV interval.
    """
    navs = load_expanded_goal_navs()
    nav = navs["caitong_integrated_circuit_a"]
    returns = nav.pct_change(fill_method=None).fillna(0.0)
    factors = read_external_factors("memory_semiconductor_proxy").reindex(nav.index).ffill()
    etf_change = factors.get("etf_change_pct", pd.Series(0.0, index=nav.index)).fillna(0.0)
    decision_position = pd.Series(1.0, index=nav.index)
    decision_position[etf_change <= threshold_pct] = 0.0
    executed_position = decision_position.shift(1).fillna(1.0)
    turnover = decision_position.diff().abs().fillna(0.0)
    strategy_returns = executed_position * returns - turnover * fee
    strategy_equity = (1.0 + strategy_returns).cumprod()
    all_in_equity = (1.0 + returns).cumprod()

    # Daily DCA into the same fund, normalized to contributed capital.
    units = 0.0
    invested = 0.0
    dca_values = []
    for date, price in nav.items():
        contribution = 1.0 / len(nav)
        units += contribution * (1 - 0.0015) / float(price)
        invested += contribution
        dca_values.append(units * float(price) / invested)
    dca_equity = pd.Series(dca_values, index=nav.index)
    dca_returns = dca_equity.pct_change(fill_method=None).fillna(0.0)

    periods = {
        "trailing_2y": nav.index.max() - pd.DateOffset(years=2),
        "trailing_3m": nav.index.max() - pd.DateOffset(months=3),
    }
    summary_rows = []
    for period, start in periods.items():
        index = nav.index[nav.index >= start]
        strategy_metrics = performance_metrics(strategy_returns.loc[index])
        all_in_metrics = performance_metrics(returns.loc[index])
        dca_metrics = performance_metrics(dca_returns.loc[index])
        summary_rows.append(
            {
                "period": period,
                "start": index[0],
                "end": index[-1],
                "strategy_return": strategy_metrics["total_return"],
                "all_in_return": all_in_metrics["total_return"],
                "dca_return": dca_metrics["total_return"],
                "excess_vs_all_in": strategy_metrics["total_return"] - all_in_metrics["total_return"],
                "excess_vs_dca": strategy_metrics["total_return"] - dca_metrics["total_return"],
                "strategy_max_drawdown": strategy_metrics["max_drawdown"],
                "all_in_max_drawdown": all_in_metrics["max_drawdown"],
                "dca_max_drawdown": dca_metrics["max_drawdown"],
            }
        )
    summary = pd.DataFrame(summary_rows)

    curves = pd.DataFrame(
        {
            "nav": nav,
            "strategy_equity": strategy_equity,
            "all_in_equity": all_in_equity,
            "dca_equity": dca_equity,
            "etf_change_pct": etf_change,
            "decision_position": decision_position,
            "executed_position": executed_position,
            "strategy_return": strategy_returns,
            "all_in_return": returns,
        }
    )

    actions = []
    previous_decision = float(decision_position.iloc[0])
    last_action_date = nav.index[0]
    last_action_equity = float(strategy_equity.iloc[0])
    last_buy_date = nav.index[0] if previous_decision > 0 else None
    last_buy_nav = float(nav.iloc[0]) if previous_decision > 0 else np.nan
    actions.append(
        {
            "date": nav.index[0],
            "action": "INITIAL_HOLD" if previous_decision > 0 else "INITIAL_CASH",
            "nav": float(nav.iloc[0]),
            "etf_change_pct": float(etf_change.iloc[0]),
            "decision_position": previous_decision,
            "executed_position_next_day": float(decision_position.iloc[0]),
            "strategy_equity": float(strategy_equity.iloc[0]),
            "step_return_since_previous_action": 0.0,
            "completed_trade_return": np.nan,
            "note": "Initial state",
        }
    )
    for date in nav.index[1:]:
        current_decision = float(decision_position.loc[date])
        if current_decision == previous_decision:
            continue
        action = "SELL_TO_CASH" if current_decision < previous_decision else "BUY_FULL"
        step_return = float(strategy_equity.loc[date] / last_action_equity - 1.0)
        completed_return = np.nan
        note = (
            "Sell is filled at same-day NAV; this day's fund return is still included."
            if action == "SELL_TO_CASH"
            else "Buy is filled at same-day NAV; exposure resumes from the next NAV interval."
        )
        if action == "SELL_TO_CASH" and last_buy_date is not None:
            completed_return = float(nav.loc[date] / last_buy_nav - 1.0)
        elif action == "BUY_FULL":
            last_buy_date = date
            last_buy_nav = float(nav.loc[date])
        actions.append(
            {
                "date": date,
                "action": action,
                "nav": float(nav.loc[date]),
                "etf_change_pct": float(etf_change.loc[date]),
                "decision_position": current_decision,
                "executed_position_next_day": current_decision,
                "strategy_equity": float(strategy_equity.loc[date]),
                "step_return_since_previous_action": step_return,
                "completed_trade_return": completed_return,
                "note": note,
            }
        )
        previous_decision = current_decision
        last_action_date = date
        last_action_equity = float(strategy_equity.loc[date])
    # Add final mark-to-market row.
    final_date = nav.index[-1]
    actions.append(
        {
            "date": final_date,
            "action": "FINAL_MARK",
            "nav": float(nav.iloc[-1]),
            "etf_change_pct": float(etf_change.iloc[-1]),
            "decision_position": float(decision_position.iloc[-1]),
            "executed_position_next_day": float(decision_position.iloc[-1]),
            "strategy_equity": float(strategy_equity.iloc[-1]),
            "step_return_since_previous_action": float(strategy_equity.iloc[-1] / last_action_equity - 1.0),
            "completed_trade_return": (
                float(nav.iloc[-1] / last_buy_nav - 1.0)
                if last_buy_date is not None and decision_position.iloc[-1] > 0
                else np.nan
            ),
            "note": "Unrealized return from the last action to the final NAV.",
        }
    )
    action_frame = pd.DataFrame(actions)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / "corrected_execution_summary.csv"
    curves_path = output / "corrected_execution_daily_curve.csv"
    actions_path = output / "corrected_execution_trade_steps.csv"
    chart_path = output / "corrected_execution_chart.svg"
    report_path = output / "corrected_execution_report.md"
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    curves.to_csv(curves_path, index_label="date", encoding="utf-8-sig")
    action_frame.to_csv(actions_path, index=False, encoding="utf-8-sig")

    _write_corrected_execution_svg(curves, action_frame, chart_path)
    report_path.write_text(
        "\n".join(
            [
                "# Corrected Execution Audit",
                "",
                "This audit fixes the open-fund timing issue: a sell submitted before the cutoff receives same-day NAV, so the strategy still bears that day's fund return. The target position affects the next NAV interval.",
                "",
                "## Summary",
                "",
                summary.to_markdown(index=False),
                "",
                "## Trade Steps",
                "",
                action_frame.to_markdown(index=False),
                "",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "summary": summary,
        "curves": curves,
        "actions": action_frame,
        "summary_path": summary_path,
        "curves_path": curves_path,
        "actions_path": actions_path,
        "chart_path": chart_path,
        "report_path": report_path,
    }


def _write_corrected_execution_svg(
    curves: pd.DataFrame,
    actions: pd.DataFrame,
    path: Path,
) -> None:
    width, height = 1200, 720
    margin_left, margin_right, margin_top, margin_bottom = 80, 40, 50, 90
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    index = curves.index
    x_values = np.arange(len(index), dtype=float)
    series_names = ["strategy_equity", "all_in_equity", "dca_equity"]
    y_min = float(curves[series_names].min().min())
    y_max = float(curves[series_names].max().max())
    y_pad = (y_max - y_min) * 0.05 if y_max > y_min else 0.1
    y_min -= y_pad
    y_max += y_pad

    def sx(loc: float) -> float:
        return margin_left + (loc / max(len(index) - 1, 1)) * plot_w

    def sy(value: float) -> float:
        return margin_top + (y_max - value) / (y_max - y_min) * plot_h

    colors = {
        "strategy_equity": "#d62728",
        "all_in_equity": "#1f77b4",
        "dca_equity": "#2ca02c",
    }
    labels = {
        "strategy_equity": "Corrected ETF guard",
        "all_in_equity": "006502 all-in",
        "dca_equity": "Daily DCA",
    }

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width/2}" y="28" text-anchor="middle" font-size="20" font-family="Arial">Corrected Execution: 006502 ETF Guard vs All-In/DCA</text>',
        f'<line x1="{margin_left}" y1="{height-margin_bottom}" x2="{width-margin_right}" y2="{height-margin_bottom}" stroke="#333"/>',
        f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{height-margin_bottom}" stroke="#333"/>',
    ]
    for frac in np.linspace(0, 1, 6):
        value = y_min + frac * (y_max - y_min)
        y = sy(value)
        parts.append(f'<line x1="{margin_left}" y1="{y:.1f}" x2="{width-margin_right}" y2="{y:.1f}" stroke="#eee"/>')
        parts.append(f'<text x="{margin_left-8}" y="{y+4:.1f}" text-anchor="end" font-size="11" font-family="Arial">{value:.2f}</text>')

    for name in series_names:
        points = " ".join(
            f"{sx(i):.1f},{sy(float(value)):.1f}"
            for i, value in enumerate(curves[name].to_numpy())
            if pd.notna(value)
        )
        parts.append(
            f'<polyline points="{points}" fill="none" stroke="{colors[name]}" stroke-width="2"/>'
        )

    date_to_loc = {date: i for i, date in enumerate(index)}
    for row in actions.itertuples(index=False):
        date = pd.Timestamp(row.date)
        if date not in date_to_loc or row.action not in {"SELL_TO_CASH", "BUY_FULL"}:
            continue
        loc = date_to_loc[date]
        x = sx(loc)
        y = sy(float(curves.loc[date, "strategy_equity"]))
        color = "#ff7f0e" if row.action == "SELL_TO_CASH" else "#9467bd"
        shape = "SELL" if row.action == "SELL_TO_CASH" else "BUY"
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4" fill="{color}"/>')
        parts.append(
            f'<text x="{x+5:.1f}" y="{y-5:.1f}" font-size="9" font-family="Arial" fill="{color}">{shape}</text>'
        )

    legend_x = margin_left + 10
    legend_y = margin_top + 10
    for i, name in enumerate(series_names):
        y = legend_y + i * 22
        parts.append(f'<line x1="{legend_x}" y1="{y}" x2="{legend_x+28}" y2="{y}" stroke="{colors[name]}" stroke-width="3"/>')
        parts.append(f'<text x="{legend_x+36}" y="{y+4}" font-size="13" font-family="Arial">{labels[name]}</text>')
    for loc_frac, label in [(0, str(index[0].date())), (0.5, str(index[len(index)//2].date())), (1, str(index[-1].date()))]:
        x = margin_left + loc_frac * plot_w
        parts.append(f'<text x="{x:.1f}" y="{height-45}" text-anchor="middle" font-size="12" font-family="Arial">{label}</text>')
    parts.append("</svg>")
    path.write_text("\n".join(parts), encoding="utf-8")
