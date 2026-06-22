"""Reusable six-fund meta-strategy research runner."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .backtest import performance_metrics
from .style_timing import read_cached_nav


FUND_CODES = {
    "pcb": "720001",
    "cpo": "007817",
    "memory": "008887",
    "ai": "008585",
    "chemical": "014942",
    "nonferrous": "004432",
}


def load_nav_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {asset: read_cached_nav(code) for asset, code in FUND_CODES.items()}
    ).dropna()


def _daily_model_returns(
    rets: pd.DataFrame,
    model_detail: pd.DataFrame,
    switch_fee: float = 0.0065,
) -> pd.Series:
    choices = pd.Series("equal_weight", index=rets.index, dtype=object)
    for row in model_detail.sort_values("start").itertuples():
        choices.loc[choices.index >= row.start] = row.choice
    values = []
    previous = None
    for date, choice in choices.items():
        ret = float(rets.loc[date].mean()) if choice == "equal_weight" else float(rets.loc[date, choice])
        if previous is not None and choice != previous:
            ret -= switch_fee
        values.append(ret)
        previous = choice
    return pd.Series(values, index=rets.index, name="six_fund_logreg_model")


def _daily_dca_equal_returns(
    navs: pd.DataFrame,
    index: pd.Index,
    purchase_fee: float = 0.0015,
) -> pd.Series:
    units = pd.Series(0.0, index=navs.columns)
    values = []
    invested = 0.0
    for date in index:
        contribution = 1.0 / len(index)
        per_asset = contribution / len(navs.columns)
        for asset in navs.columns:
            units[asset] += per_asset * (1 - purchase_fee) / float(navs.loc[date, asset])
        invested += contribution
        values.append(float((units * navs.loc[date]).sum() / invested))
    equity = pd.Series(values, index=index)
    return equity.pct_change(fill_method=None).fillna(0.0)


def _choose_candidate(
    past: pd.DataFrame,
    lookback: int,
    objective: str,
) -> str:
    scoped = past.tail(lookback)
    scores = {}
    for candidate in ("model", "cpo", "equal"):
        values = scoped[candidate]
        if objective == "recent_weighted":
            weights = np.linspace(1, 2, len(values))
            score = float((values * weights).sum() / weights.sum())
        elif objective == "mean_plus_worst":
            score = float(values.mean() + 0.4 * values.min())
        else:
            score = float(values.mean())
        scores[candidate] = score
    return max(scores, key=scores.get)


def run_meta_strategy(
    output_dir: str | Path = "output/meta_strategy",
    model_detail_path: str | Path = "output/expanded_research_v10/six_fund_external_sklearn_detail.csv",
    window_days_list: tuple[int, ...] = (60, 90, 120),
    step_list: tuple[int, ...] = (5, 10, 20),
    warmup_list: tuple[int, ...] = (3, 5, 8),
    lookback_list: tuple[int, ...] = (3, 5, 10),
    objectives: tuple[str, ...] = ("mean", "recent_weighted", "mean_plus_worst"),
) -> dict[str, pd.DataFrame | Path]:
    navs = load_nav_panel()
    rets = navs.pct_change(fill_method=None).fillna(0.0)
    model_detail = pd.read_csv(model_detail_path, parse_dates=["start", "end"])
    model_detail = model_detail[
        (model_detail["model"] == "logreg")
        & (model_detail["train_min"] == 20)
        & (model_detail["guard"] == 0.0)
    ].sort_values("start")
    model_returns = _daily_model_returns(rets, model_detail)

    rows = []
    for window_days in window_days_list:
        for step in step_list:
            starts = list(range(160, len(navs) - window_days, step))
            candidate_rows = []
            for window_id, start in enumerate(starts, 1):
                index = navs.index[start : start + window_days]
                candidate_rows.append(
                    {
                        "window_id": window_id,
                        "start": index[0],
                        "end": index[-1],
                        "model": performance_metrics(model_returns.loc[index])["total_return"],
                        "cpo": performance_metrics(rets["cpo"].loc[index])["total_return"],
                        "equal": performance_metrics(rets.mean(axis=1).loc[index])["total_return"],
                        "pcb": performance_metrics(rets["pcb"].loc[index])["total_return"],
                        "dca": performance_metrics(_daily_dca_equal_returns(navs, index))["total_return"],
                    }
                )
            candidates = pd.DataFrame(candidate_rows)
            for warmup in warmup_list:
                for lookback in lookback_list:
                    for objective in objectives:
                        for row in candidates.itertuples():
                            # With overlapping windows, lower window_id does not imply
                            # the window had completed before the current start date.
                            past = candidates[candidates["end"] < row.start]
                            choice = (
                                "equal"
                                if len(past) < warmup
                                else _choose_candidate(past, lookback, objective)
                            )
                            rows.append(
                                {
                                    "window_days": window_days,
                                    "step": step,
                                    "warmup": warmup,
                                    "lookback": lookback,
                                    "objective": objective,
                                    "window_id": int(row.window_id),
                                    "start": row.start,
                                    "end": row.end,
                                    "choice": choice,
                                    "total_return": float(getattr(row, choice)),
                                    "model_return": float(row.model),
                                    "cpo_return": float(row.cpo),
                                    "equal_return": float(row.equal),
                                    "pcb_return": float(row.pcb),
                                    "dca_return": float(row.dca),
                                    "best_candidate_return": max(float(row.model), float(row.cpo), float(row.equal)),
                                }
                            )

    detail = pd.DataFrame(rows)
    for baseline in ("model", "cpo", "equal", "pcb", "dca", "best_candidate"):
        detail[f"excess_vs_{baseline}"] = detail["total_return"] - detail[f"{baseline}_return"]

    summary = (
        detail.groupby(["window_days", "step", "warmup", "lookback", "objective"])
        .agg(
            windows=("window_id", "count"),
            mean_return=("total_return", "mean"),
            median_return=("total_return", "median"),
            return_var=("total_return", "var"),
            mean_excess_vs_cpo=("excess_vs_cpo", "mean"),
            beat_cpo=("excess_vs_cpo", lambda x: float((x > 0).mean())),
            mean_excess_vs_model=("excess_vs_model", "mean"),
            beat_model=("excess_vs_model", lambda x: float((x > 0).mean())),
            mean_excess_vs_equal=("excess_vs_equal", "mean"),
            beat_equal=("excess_vs_equal", lambda x: float((x > 0).mean())),
            mean_excess_vs_dca=("excess_vs_dca", "mean"),
            beat_dca=("excess_vs_dca", lambda x: float((x > 0).mean())),
            mean_excess_vs_best_candidate=("excess_vs_best_candidate", "mean"),
            choices=("choice", lambda x: dict(x.value_counts())),
        )
        .reset_index()
    )
    param_aggregate = (
        summary.groupby(["warmup", "lookback", "objective"])
        .agg(
            grids=("window_days", "count"),
            avg_mean_return=("mean_return", "mean"),
            avg_excess_vs_cpo=("mean_excess_vs_cpo", "mean"),
            avg_excess_vs_equal=("mean_excess_vs_equal", "mean"),
            avg_excess_vs_dca=("mean_excess_vs_dca", "mean"),
            min_excess_vs_cpo=("mean_excess_vs_cpo", "min"),
            positive_cpo_grid_ratio=("mean_excess_vs_cpo", lambda x: float((x > 0).mean())),
        )
        .reset_index()
        .sort_values("avg_excess_vs_cpo", ascending=False)
    )
    best_by_grid = (
        summary.sort_values("mean_return", ascending=False)
        .groupby(["window_days", "step"])
        .head(3)
    )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    detail_path = output / "meta_strategy_detail.csv"
    summary_path = output / "meta_strategy_summary.csv"
    aggregate_path = output / "meta_strategy_param_aggregate.csv"
    best_path = output / "meta_strategy_best_by_grid.csv"
    detail.to_csv(detail_path, index=False, encoding="utf-8-sig")
    summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    param_aggregate.to_csv(aggregate_path, index=False, encoding="utf-8-sig")
    best_by_grid.to_csv(best_path, index=False, encoding="utf-8-sig")
    return {
        "detail": detail,
        "summary": summary,
        "param_aggregate": param_aggregate,
        "best_by_grid": best_by_grid,
        "detail_path": detail_path,
        "summary_path": summary_path,
        "aggregate_path": aggregate_path,
        "best_path": best_path,
    }
