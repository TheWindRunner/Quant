"""Expanded cross-sector fund research beyond the original tech trio."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import pandas as pd

from .experiment import (
    evaluate_locked_selected_ensemble,
    evaluate_locked_theme_models,
    select_ensemble_on_training,
    select_theme_models_on_training,
)
from .fund_backtest import backtest_open_fund
from .data import load_or_fetch_open_fund_nav
from .external_factors import load_external_factor_panel
from .intraday_estimate import simulate_intraday_estimate_history, SyntheticEstimateConfig
from .strategy_curves import _line_chart_with_markers_svg
from .validation import purged_training_folds


EXPANDED_FUNDS = {
    "cpo_communication": {
        "code": "007817",
        "name": "国泰中证全指通信设备ETF联接A",
        "sector_group": "technology",
        "proxy_note": "pure",
    },
    "memory_semiconductor_proxy": {
        "code": "008887",
        "name": "华夏国证半导体芯片ETF联接A",
        "sector_group": "technology",
        "proxy_note": "proxy",
    },
    "artificial_intelligence": {
        "code": "008585",
        "name": "华夏中证人工智能主题ETF联接A",
        "sector_group": "technology",
        "proxy_note": "pure",
    },
    "pcb_proxy_consumer_electronics": {
        "code": "015876",
        "name": "富国中证消费电子主题ETF发起式联接A",
        "sector_group": "technology_adjacent",
        "proxy_note": "pcb_proxy",
    },
    "green_power": {
        "code": "018034",
        "name": "国泰国证绿色电力ETF发起联接A",
        "sector_group": "utilities",
        "proxy_note": "pure",
    },
    "chemical": {
        "code": "014942",
        "name": "鹏华中证细分化工产业主题ETF联接A",
        "sector_group": "materials",
        "proxy_note": "pure",
    },
    "nonferrous": {
        "code": "004432",
        "name": "南方中证申万有色金属ETF发起联接A",
        "sector_group": "materials",
        "proxy_note": "pure",
    },
}

EXPLORATORY_SHORT_HISTORY = {
    "grid_equipment": {
        "code": "023638",
        "name": "国泰恒生A股电网设备ETF发起联接A",
        "sector_group": "utilities",
        "proxy_note": "pure_short_history",
    }
}


@dataclass(frozen=True)
class AllInTConfig:
    trim_threshold_pct: float
    add_threshold_pct: float
    overbought_buffer_pct: float
    pullback_threshold_pct: float
    minimum_days_between_trades: int
    tactical_step: float = 0.10
    seed: int = 42
    disabled: bool = False

    @property
    def name(self) -> str:
        if self.disabled:
            return "all_in_t_hold"
        return (
            f"all_in_t_trim{self.trim_threshold_pct:.1f}"
            f"_add{abs(self.add_threshold_pct):.1f}"
            f"_buf{self.overbought_buffer_pct:.2f}"
            f"_pb{self.pullback_threshold_pct:.2f}"
            f"_gap{self.minimum_days_between_trades}"
        ).replace("-", "n")


def expanded_nav_datasets() -> dict[str, dict[str, pd.Series | pd.DataFrame | None]]:
    navs = {
        asset: load_or_fetch_open_fund_nav(definition["code"])
        for asset, definition in EXPANDED_FUNDS.items()
    }
    common_index = pd.DatetimeIndex(
        sorted(set.intersection(*(set(series.index) for series in navs.values())))
    )
    peer_frame = pd.DataFrame(
        {asset: series.reindex(common_index) for asset, series in navs.items()}
    )
    market_nav = peer_frame.mean(axis=1)
    return {
        asset: {
            "nav": nav.reindex(common_index).ffill(),
            "market_nav": market_nav,
            "peers": peer_frame.drop(columns=[asset]),
            "external_factors": load_external_factor_panel(asset, common_index),
        }
        for asset, nav in navs.items()
    }


def all_in_t_candidate_configs() -> list[AllInTConfig]:
    configs = [AllInTConfig(99.0, -99.0, 0.99, 0.99, 999, disabled=True)]
    for trim in (2.5, 3.5, 5.0):
        for add in (-1.5, -2.5, -4.0):
            for buffer in (0.01, 0.03):
                for pullback in (0.025, 0.05):
                    for gap in (3, 7):
                        configs.append(
                            AllInTConfig(
                                trim_threshold_pct=trim,
                                add_threshold_pct=add,
                                overbought_buffer_pct=buffer,
                                pullback_threshold_pct=pullback,
                                minimum_days_between_trades=gap,
                            )
                        )
    return configs


def all_in_t_overlay(
    nav: pd.Series,
    config: AllInTConfig,
    reset_date: pd.Timestamp | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    if config.disabled:
        history = simulate_intraday_estimate_history(
            nav,
            SyntheticEstimateConfig(seed=config.seed),
        )
        history["target_position"] = 1.0
        history["action"] = "HOLD"
        history["reason"] = "disabled"
        return pd.Series(1.0, index=nav.index, name=config.name), history
    history = simulate_intraday_estimate_history(
        nav,
        SyntheticEstimateConfig(seed=config.seed),
    ).set_index("date")
    estimated_change = history["estimated_change_pct"].reindex(nav.index)
    actual_change = history["actual_change_pct"].reindex(nav.index)
    ma20 = nav.rolling(20).mean()
    ma60 = nav.rolling(60).mean()
    high20 = nav.rolling(20).max()
    drawdown20 = nav / high20 - 1
    momentum5 = nav.pct_change(5, fill_method=None)
    current = 1.0
    last_trade = -config.minimum_days_between_trades
    targets = []
    actions = []
    reasons = []
    for location, date in enumerate(nav.index):
        if reset_date is not None and date < reset_date:
            targets.append(1.0)
            actions.append("HOLD")
            reasons.append("before_reset")
            continue
        if reset_date is not None and date == reset_date:
            current = 1.0
            last_trade = location - config.minimum_days_between_trades
        can_trade = location - last_trade >= config.minimum_days_between_trades
        estimate = float(estimated_change.loc[date]) if pd.notna(estimated_change.loc[date]) else 0.0
        action = "HOLD"
        reason = "no_signal"
        overbought = (
            pd.notna(ma20.loc[date])
            and nav.loc[date] >= ma20.loc[date] * (1 + config.overbought_buffer_pct)
            and momentum5.loc[date] > 0
            and drawdown20.loc[date] > -0.02
        )
        pullback = (
            pd.notna(ma60.loc[date])
            and nav.loc[date] >= ma60.loc[date] * 0.97
            and drawdown20.loc[date] <= -config.pullback_threshold_pct
            and momentum5.loc[date] < 0
        )
        if can_trade and current >= 0.999 and estimate >= config.trim_threshold_pct and overbought:
            current = max(0.0, 1.0 - config.tactical_step)
            last_trade = location
            action = "TACTICAL_TRIM"
            reason = "extreme_up_and_overbought"
        elif can_trade and current < 0.999 and estimate <= config.add_threshold_pct and pullback:
            current = 1.0
            last_trade = location
            action = "TACTICAL_ADD"
            reason = "extreme_down_after_pullback"
        targets.append(current)
        actions.append(action)
        reasons.append(reason)
    details = history.reindex(nav.index)
    details.index.name = "date"
    details["target_position"] = targets
    details["action"] = actions
    details["reason"] = reasons
    return (
        pd.Series(targets, index=nav.index, name=config.name).clip(0, 1),
        details.reset_index(),
    )


def _evaluate_target(
    nav: pd.Series,
    target: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> dict[str, float]:
    sliced_nav = nav.loc[(nav.index >= start) & (nav.index <= end)]
    prior = target.loc[target.index < start]
    initial = float(prior.iloc[-1]) if len(prior) else 0.0
    return backtest_open_fund(
        sliced_nav,
        target.reindex(sliced_nav.index),
        initial_target=initial,
        liquidate_at_end=True,
    )["metrics"]


def select_all_in_t_on_training(
    datasets: dict[str, dict[str, pd.Series | pd.DataFrame | None]],
    test_start: pd.Timestamp,
) -> tuple[dict[str, AllInTConfig], pd.DataFrame]:
    rows = []
    selected = {}
    for asset, dataset in datasets.items():
        nav = dataset["nav"]
        if not isinstance(nav, pd.Series):
            raise TypeError(f"{asset}.nav must be a Series")
        buy_hold = pd.Series(1.0, index=nav.index)
        asset_rows = []
        for config in all_in_t_candidate_configs():
            for fold_start, fold_end in purged_training_folds(nav.index, test_start):
                target, _ = all_in_t_overlay(nav, config, reset_date=fold_start)
                benchmark = _evaluate_target(nav, buy_hold, fold_start, fold_end)
                candidate = _evaluate_target(nav, target, fold_start, fold_end)
                excess_return = candidate["total_return"] - benchmark["total_return"]
                drawdown_improvement = candidate["max_drawdown"] - benchmark["max_drawdown"]
                objective = (
                    excess_return
                    + 0.70 * drawdown_improvement
                    + 0.03 * min(max(candidate["sharpe"], -2.0), 2.0)
                    - 0.001 * candidate["trade_count"]
                )
                row = {
                    "asset": asset,
                    "model": config.name,
                    "fold_start": fold_start,
                    "fold_end": fold_end,
                    "objective": objective,
                    "excess_return": excess_return,
                    "drawdown_improvement": drawdown_improvement,
                    **candidate,
                }
                rows.append(row)
                asset_rows.append(row)
        asset_frame = pd.DataFrame(asset_rows)
        ranking = (
            asset_frame.groupby("model")
            .agg(
                median_objective=("objective", "median"),
                positive_fold_ratio=("objective", lambda x: float((x > 0).mean())),
                median_excess_return=("excess_return", "median"),
                median_drawdown_improvement=("drawdown_improvement", "median"),
                median_holding_days=("average_holding_days", "median"),
            )
            .sort_values(
                ["positive_fold_ratio", "median_objective", "median_drawdown_improvement"],
                ascending=False,
            )
        )
        chosen_name = ranking.index[0]
        selected[asset] = next(
            config for config in all_in_t_candidate_configs() if config.name == chosen_name
        )
    return selected, pd.DataFrame(rows)


def locked_all_in_t_comparison(
    datasets: dict[str, dict[str, pd.Series | pd.DataFrame | None]],
    selected_configs: dict[str, AllInTConfig],
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    rows = []
    histories: dict[str, pd.DataFrame] = {}
    for asset, dataset in datasets.items():
        nav = dataset["nav"]
        if not isinstance(nav, pd.Series):
            raise TypeError(f"{asset}.nav must be a Series")
        buy_hold_target = pd.Series(1.0, index=nav.index)
        buy_hold = _evaluate_target(nav, buy_hold_target, test_start, test_end)
        config = selected_configs[asset]
        target, history = all_in_t_overlay(nav, config, reset_date=test_start)
        candidate = _evaluate_target(nav, target, test_start, test_end)
        rows.extend(
            [
                {"asset": asset, "model": "buy_hold", **buy_hold},
                {"asset": asset, "model": config.name, **candidate},
            ]
        )
        histories[asset] = history
    return pd.DataFrame(rows), histories


def _chart_all_in_t(
    asset: str,
    nav: pd.Series,
    config: AllInTConfig,
    start: pd.Timestamp,
    end: pd.Timestamp,
    output: Path,
) -> None:
    buy_hold_target = pd.Series(1.0, index=nav.index)
    overlay_target, history = all_in_t_overlay(nav, config, reset_date=start)
    sliced_nav = nav.loc[(nav.index >= start) & (nav.index <= end)]
    buy_hold_result = backtest_open_fund(
        sliced_nav,
        buy_hold_target.reindex(sliced_nav.index),
        initial_target=1.0,
        liquidate_at_end=True,
    )
    prior = overlay_target.loc[overlay_target.index < start]
    overlay_result = backtest_open_fund(
        sliced_nav,
        overlay_target.reindex(sliced_nav.index),
        initial_target=float(prior.iloc[-1]) if len(prior) else 1.0,
        liquidate_at_end=True,
    )
    equity = pd.DataFrame(
        {
            "buy_hold": buy_hold_result["ledger"]["portfolio_value"],
            "all_in_t_overlay": overlay_result["ledger"]["portfolio_value"],
        }
    )
    markers = history.loc[
        (pd.to_datetime(history["date"]) >= start)
        & (pd.to_datetime(history["date"]) <= end)
        & (history["action"] != "HOLD")
    ].copy()
    markers["series"] = "all_in_t_overlay"
    output.write_text(
        _line_chart_with_markers_svg(
            equity,
            f"{asset} all-in T overlay vs buy hold",
            markers,
        ),
        encoding="utf-8",
    )


def run_expanded_research(output_dir: str | Path = "output/expanded_research") -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = expanded_nav_datasets()
    latest = min(dataset["nav"].index.max() for dataset in datasets.values())
    test_start = latest - pd.DateOffset(months=6)
    selected_models, training = select_ensemble_on_training(datasets, test_start)
    locked = evaluate_locked_selected_ensemble(datasets, selected_models, test_start, latest)
    theme_models, theme_training = select_theme_models_on_training(datasets, test_start)
    theme_locked = evaluate_locked_theme_models(datasets, theme_models, test_start, latest)
    all_in_t_models, all_in_t_training = select_all_in_t_on_training(datasets, test_start)
    all_in_t_locked, all_in_t_histories = locked_all_in_t_comparison(
        datasets,
        all_in_t_models,
        test_start,
        latest,
    )
    all_in_t_summary = (
        all_in_t_locked.pivot(index="asset", columns="model", values="total_return")
        .reset_index()
    )
    metadata = pd.DataFrame(
        [
            {
                "asset": asset,
                **definition,
                "history_rows": len(datasets[asset]["nav"]),
                "history_start": datasets[asset]["nav"].index.min().date().isoformat(),
                "history_end": datasets[asset]["nav"].index.max().date().isoformat(),
                "selected_all_in_t_model": all_in_t_models[asset].name,
            }
            for asset, definition in EXPANDED_FUNDS.items()
        ]
    )
    exploratory = []
    for asset, definition in EXPLORATORY_SHORT_HISTORY.items():
        nav = load_or_fetch_open_fund_nav(definition["code"])
        exploratory.append(
            {
                "asset": asset,
                **definition,
                "history_rows": len(nav),
                "history_start": nav.index.min().date().isoformat(),
                "history_end": nav.index.max().date().isoformat(),
                "eligible_for_training": len(nav.loc[nav.index < test_start]) >= 252,
            }
        )
    metadata.to_csv(output / "expanded_fund_metadata.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(exploratory).to_csv(output / "exploratory_short_history.csv", index=False, encoding="utf-8-sig")
    training.to_csv(output / "ensemble_training.csv", index=False, encoding="utf-8-sig")
    locked.to_csv(output / "ensemble_locked.csv", index=False, encoding="utf-8-sig")
    theme_training.to_csv(output / "theme_training.csv", index=False, encoding="utf-8-sig")
    theme_locked.to_csv(output / "theme_locked.csv", index=False, encoding="utf-8-sig")
    all_in_t_training.to_csv(output / "all_in_t_training.csv", index=False, encoding="utf-8-sig")
    all_in_t_locked.to_csv(output / "all_in_t_locked.csv", index=False, encoding="utf-8-sig")
    all_in_t_summary.to_csv(output / "all_in_t_summary.csv", index=False, encoding="utf-8-sig")
    for asset, history in all_in_t_histories.items():
        history.to_csv(output / f"{asset}_all_in_t_history.csv", index=False, encoding="utf-8-sig")
        nav = datasets[asset]["nav"]
        if isinstance(nav, pd.Series):
            _chart_all_in_t(
                asset,
                nav,
                all_in_t_models[asset],
                test_start,
                latest,
                output / f"{asset}_all_in_t_chart.svg",
            )
    report_lines = [
        "# Expanded Sector Research",
        "",
        f"Window: {test_start.date()} to {latest.date()}",
        "",
        "## Included funds",
        "",
        metadata.to_markdown(index=False),
        "",
        "## Exploratory only",
        "",
        pd.DataFrame(exploratory).to_markdown(index=False),
        "",
        "## Locked theme comparison",
        "",
        theme_locked[["asset", "selected_model", "model", "total_return", "max_drawdown", "sharpe", "trade_count"]].to_markdown(index=False),
        "",
        "## Locked all-in T comparison",
        "",
        all_in_t_locked[["asset", "model", "total_return", "max_drawdown", "sharpe", "trade_count"]].to_markdown(index=False),
        "",
    ]
    (output / "expanded_report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return {
        "test_start": test_start,
        "test_end": latest,
        "theme_locked": theme_locked,
        "all_in_t_locked": all_in_t_locked,
        "metadata": metadata,
        "exploratory": pd.DataFrame(exploratory),
        "output_dir": output,
    }
