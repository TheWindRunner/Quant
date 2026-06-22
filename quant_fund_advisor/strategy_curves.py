"""Rebuild locked-test strategy curves and export simple comparison charts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd

from .execution_policy import apply_open_fund_execution_policy
from .experiment import (
    evaluate_locked_theme_models,
    select_ensemble_on_training,
    select_theme_models_on_training,
)
from .fund_backtest import backtest_open_fund
from .intraday_estimate import (
    historical_extreme_move_overlay,
    historical_synthetic_estimate_overlay,
)
from .backtest import performance_metrics
from .model_zoo import ModelSignal, build_model_zoo, robust_ensemble
from .portfolio_backtest import backtest_open_fund_portfolio
from .run_research import build_nav_datasets
from .sector_rotation import (
    equal_weight_targets,
    relative_strength_weights,
    risk_budget_momentum_weights,
    select_rotation_on_training,
)


@dataclass(frozen=True)
class CurveBundle:
    title: str
    equity: pd.DataFrame
    daily_returns: pd.DataFrame
    descriptions: pd.DataFrame


@dataclass(frozen=True)
class TradeOverlayResult:
    summary: pd.DataFrame
    equity: pd.DataFrame
    trade_log: pd.DataFrame


def _locked_window() -> tuple[dict[str, dict[str, pd.Series | pd.DataFrame | None]], pd.Timestamp, pd.Timestamp]:
    datasets = build_nav_datasets()
    latest = min(dataset["nav"].index.max() for dataset in datasets.values())
    test_start = latest - pd.DateOffset(months=6)
    return datasets, test_start, latest


def _single_fund_backtest(
    nav: pd.Series,
    target: pd.Series,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    sliced_nav = nav.loc[(nav.index >= start) & (nav.index <= end)]
    practical = target.reindex(nav.index).ffill().fillna(0.0)
    prior = practical.loc[practical.index < start]
    result = backtest_open_fund(
        sliced_nav,
        practical.reindex(sliced_nav.index),
        initial_target=float(prior.iloc[-1]) if len(prior) else 0.0,
        liquidate_at_end=True,
    )
    equity = result["ledger"]["portfolio_value"].rename("equity")
    returns = result["returns"].rename("daily_return")
    return equity, returns


def _portfolio_backtest(
    navs: pd.DataFrame,
    weights: pd.DataFrame,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.Series, pd.Series]:
    sliced_navs = navs.loc[(navs.index >= start) & (navs.index <= end)]
    prior = weights.loc[weights.index < start]
    result = backtest_open_fund_portfolio(
        sliced_navs,
        weights.reindex(sliced_navs.index).ffill().fillna(0.0),
        initial_weights=prior.iloc[-1] if len(prior) else None,
        liquidate_at_end=True,
    )
    equity = result["ledger"]["portfolio_value"].rename("equity")
    returns = result["returns"].rename("daily_return")
    return equity, returns


def build_single_fund_curve_bundle(asset: str) -> CurveBundle:
    datasets, test_start, test_end = _locked_window()
    dataset = datasets[asset]
    nav = dataset["nav"]
    if not isinstance(nav, pd.Series):
        raise TypeError(f"{asset}.nav must be a Series")

    zoo = {
        model.name: model
        for model in build_model_zoo(
            nav,
            market_nav=dataset.get("market_nav"),
            peers=dataset.get("peers"),
        )
    }
    selected_theme_models, _ = select_theme_models_on_training(datasets, test_start)
    selected_name = selected_theme_models.get(asset, "buy_hold")
    theme_signal = zoo.get(selected_name, zoo["buy_hold"])

    selected_ensemble, _ = select_ensemble_on_training(datasets, test_start)
    ensemble_members = [zoo[name] for name in selected_ensemble if name in zoo]
    if ensemble_members:
        ensemble_signal = ModelSignal(
            "trained_ensemble",
            robust_ensemble(ensemble_members).position,
            "Training-selected median ensemble of qualified model families",
        )
    else:
        ensemble_signal = ModelSignal(
            "trained_ensemble",
            zoo["buy_hold"].position,
            "No qualified tactical ensemble; same target as buy and hold",
        )

    overlay_signal = ModelSignal(
        "extreme_estimate_overlay",
        historical_extreme_move_overlay(nav, base_target=1.0),
        "Only trims after very large up days and restores after large down days in trend",
    )

    signals = [
        ModelSignal("buy_hold", zoo["buy_hold"].position, "Always invested"),
        ModelSignal(
            "dual_ma",
            apply_open_fund_execution_policy(zoo["dual_ma"].position),
            "Invest only when NAV is above MA20 and MA20 is above MA60",
        ),
        ModelSignal(
            "theme_selected",
            apply_open_fund_execution_policy(theme_signal.position),
            f"Frozen per-theme model selected on training data: {selected_name}",
        ),
        ModelSignal(
            "trained_ensemble",
            apply_open_fund_execution_policy(ensemble_signal.position),
            ensemble_signal.description,
        ),
        overlay_signal,
    ]

    equity_parts = []
    return_parts = []
    descriptions = []
    for signal in signals:
        equity, returns = _single_fund_backtest(nav, signal.position, test_start, test_end)
        equity_parts.append(equity.rename(signal.name))
        return_parts.append(returns.rename(signal.name))
        descriptions.append({"strategy": signal.name, "description": signal.description})

    return CurveBundle(
        title=asset,
        equity=pd.concat(equity_parts, axis=1),
        daily_returns=pd.concat(return_parts, axis=1),
        descriptions=pd.DataFrame(descriptions),
    )


def build_portfolio_curve_bundle() -> CurveBundle:
    datasets, test_start, test_end = _locked_window()
    navs = pd.DataFrame(
        {asset: dataset["nav"] for asset, dataset in datasets.items()}
    ).dropna()
    selected_rotation, _ = select_rotation_on_training(navs, test_start)
    rotation_weights = (
        relative_strength_weights(navs, selected_rotation)
        if selected_rotation is not None
        else equal_weight_targets(navs)
    )
    risk_budget_weights = risk_budget_momentum_weights(navs)
    equal_weight = equal_weight_targets(navs)

    strategies: list[tuple[str, pd.DataFrame, str]] = [
        ("equal_weight_buy_hold", equal_weight, "Three-theme equal-weight strategic hold"),
        (
            selected_rotation.name if selected_rotation is not None else "no_qualified_rotation",
            rotation_weights,
            (
                "20-day relative-strength rotation with 60% strategic core and tail overlay"
                if selected_rotation is not None
                else "No qualified rotation model; identical to equal weight"
            ),
        ),
        (
            "risk_budget_momentum",
            risk_budget_weights,
            "Inverse-volatility allocation with 60-day momentum tilt and tail overlay",
        ),
    ]

    equity_parts = []
    return_parts = []
    descriptions = []
    for name, weights, description in strategies:
        equity, returns = _portfolio_backtest(navs, weights, test_start, test_end)
        equity_parts.append(equity.rename(name))
        return_parts.append(returns.rename(name))
        descriptions.append({"strategy": name, "description": description})

    return CurveBundle(
        title="technology_portfolio",
        equity=pd.concat(equity_parts, axis=1),
        daily_returns=pd.concat(return_parts, axis=1),
        descriptions=pd.DataFrame(descriptions),
    )


def _date_ticks(index: pd.Index, count: int = 6) -> list[tuple[int, str]]:
    if len(index) == 0:
        return []
    if len(index) <= count:
        locations = list(range(len(index)))
    else:
        step = max(1, (len(index) - 1) // (count - 1))
        locations = list(range(0, len(index), step))
        if locations[-1] != len(index) - 1:
            locations[-1] = len(index) - 1
    return [(location, pd.Timestamp(index[location]).strftime("%Y-%m-%d")) for location in locations]


def _line_chart_svg(
    frame: pd.DataFrame,
    title: str,
    y_label: str,
    percent: bool,
) -> str:
    width = 1200
    height = 680
    left = 90
    right = 40
    top = 60
    bottom = 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    clean = frame.dropna(how="all")
    if clean.empty:
        raise ValueError(f"No data available for chart {title}")

    values = clean.to_numpy().astype(float)
    y_min = float(values.min())
    y_max = float(values.max())
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding

    colors = [
        "#0b7285",
        "#e8590c",
        "#2b8a3e",
        "#9c36b5",
        "#c92a2a",
        "#1c7ed6",
    ]

    def x_pos(location: int) -> float:
        if len(clean.index) == 1:
            return left + plot_width / 2
        return left + plot_width * location / (len(clean.index) - 1)

    def y_pos(value: float) -> float:
        return top + plot_height * (1 - (value - y_min) / (y_max - y_min))

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f9fa"/>',
        f'<text x="{left}" y="32" font-size="24" font-family="Arial" fill="#212529">{title}</text>',
        f'<text x="28" y="{top + plot_height / 2}" transform="rotate(-90 28 {top + plot_height / 2})" '
        f'font-size="16" font-family="Arial" fill="#495057">{y_label}</text>',
    ]

    for step in range(6):
        fraction = step / 5
        value = y_min + (y_max - y_min) * (1 - fraction)
        y = top + plot_height * fraction
        label = f"{value:.2%}" if percent else f"{value:.2f}"
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#dee2e6" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="13" font-family="Arial" fill="#495057">{label}</text>')

    for location, label in _date_ticks(clean.index):
        x = x_pos(location)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#edf2f7" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 30}" text-anchor="middle" font-size="12" font-family="Arial" fill="#495057">{label}</text>')

    parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#adb5bd" stroke-width="1"/>')

    legend_y = height - 58
    legend_x = left
    for column_index, column in enumerate(clean.columns):
        color = colors[column_index % len(colors)]
        series = clean[column]
        points = " ".join(
            f"{x_pos(location):.2f},{y_pos(float(value)):.2f}"
            for location, value in enumerate(series)
            if pd.notna(value)
        )
        parts.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>'
        )
        x1 = legend_x + column_index * 210
        parts.append(f'<line x1="{x1}" y1="{legend_y}" x2="{x1 + 28}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{x1 + 36}" y="{legend_y + 5}" font-size="13" font-family="Arial" fill="#212529">{column}</text>')

    parts.append("</svg>")
    return "\n".join(parts)


def _line_chart_with_markers_svg(
    frame: pd.DataFrame,
    title: str,
    markers: pd.DataFrame,
) -> str:
    width = 1200
    height = 680
    left = 90
    right = 40
    top = 60
    bottom = 90
    plot_width = width - left - right
    plot_height = height - top - bottom
    clean = frame.dropna(how="all")
    if clean.empty:
        raise ValueError(f"No data available for chart {title}")
    values = clean.to_numpy().astype(float)
    y_min = float(values.min())
    y_max = float(values.max())
    if y_min == y_max:
        y_min -= 1.0
        y_max += 1.0
    padding = (y_max - y_min) * 0.08
    y_min -= padding
    y_max += padding
    colors = {"buy_hold": "#0b7285", "all_in_t_overlay": "#e8590c"}

    def x_pos(location: int) -> float:
        if len(clean.index) == 1:
            return left + plot_width / 2
        return left + plot_width * location / (len(clean.index) - 1)

    def y_pos(value: float) -> float:
        return top + plot_height * (1 - (value - y_min) / (y_max - y_min))

    index_lookup = {pd.Timestamp(date): location for location, date in enumerate(clean.index)}
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#f8f9fa"/>',
        f'<text x="{left}" y="32" font-size="24" font-family="Arial" fill="#212529">{title}</text>',
    ]
    for step in range(6):
        fraction = step / 5
        value = y_min + (y_max - y_min) * (1 - fraction)
        y = top + plot_height * fraction
        parts.append(f'<line x1="{left}" y1="{y:.2f}" x2="{width - right}" y2="{y:.2f}" stroke="#dee2e6" stroke-width="1"/>')
        parts.append(f'<text x="{left - 12}" y="{y + 5:.2f}" text-anchor="end" font-size="13" font-family="Arial" fill="#495057">{value:.2f}</text>')
    for location, label in _date_ticks(clean.index):
        x = x_pos(location)
        parts.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#edf2f7" stroke-width="1"/>')
        parts.append(f'<text x="{x:.2f}" y="{height - 30}" text-anchor="middle" font-size="12" font-family="Arial" fill="#495057">{label}</text>')
    parts.append(f'<rect x="{left}" y="{top}" width="{plot_width}" height="{plot_height}" fill="none" stroke="#adb5bd" stroke-width="1"/>')
    for column in clean.columns:
        color = colors.get(column, "#495057")
        series = clean[column]
        points = " ".join(
            f"{x_pos(location):.2f},{y_pos(float(value)):.2f}"
            for location, value in enumerate(series)
            if pd.notna(value)
        )
        parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}"/>')
    for _, marker in markers.iterrows():
        date = pd.Timestamp(marker["date"])
        if date not in index_lookup:
            continue
        location = index_lookup[date]
        series_name = str(marker["series"])
        if series_name not in clean.columns:
            continue
        value = float(clean.loc[date, series_name])
        x = x_pos(location)
        y = y_pos(value)
        if marker["action"] == "TACTICAL_ADD":
            parts.append(f'<polygon points="{x:.2f},{y-8:.2f} {x-7:.2f},{y+6:.2f} {x+7:.2f},{y+6:.2f}" fill="#2b8a3e"/>')
        elif marker["action"] == "TACTICAL_TRIM":
            parts.append(f'<polygon points="{x:.2f},{y+8:.2f} {x-7:.2f},{y-6:.2f} {x+7:.2f},{y-6:.2f}" fill="#c92a2a"/>')
    legend_y = height - 58
    legend_items = [
        ("buy_hold", "buy_hold", "#0b7285"),
        ("all_in_t_overlay", "all_in_t_overlay", "#e8590c"),
        ("buy", "TACTICAL_ADD", "#2b8a3e"),
        ("sell", "TACTICAL_TRIM", "#c92a2a"),
    ]
    for item_index, (_, label, color) in enumerate(legend_items):
        x1 = left + item_index * 220
        parts.append(f'<line x1="{x1}" y1="{legend_y}" x2="{x1 + 28}" y2="{legend_y}" stroke="{color}" stroke-width="4"/>')
        parts.append(f'<text x="{x1 + 36}" y="{legend_y + 5}" font-size="13" font-family="Arial" fill="#212529">{label}</text>')
    parts.append("</svg>")
    return "\n".join(parts)


def build_all_in_t_strategy(asset: str = "cpo_communication") -> TradeOverlayResult:
    datasets, test_start, test_end = _locked_window()
    dataset = datasets[asset]
    nav = dataset["nav"]
    if not isinstance(nav, pd.Series):
        raise TypeError(f"{asset}.nav must be a Series")
    buy_hold_target = pd.Series(1.0, index=nav.index, name="buy_hold")
    overlay_target, overlay_history = historical_synthetic_estimate_overlay(
        nav,
        base_target=1.0,
        tactical_step=0.10,
    )

    def evaluate(target: pd.Series) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
        sliced_nav = nav.loc[(nav.index >= test_start) & (nav.index <= test_end)]
        prior = target.loc[target.index < test_start]
        result = backtest_open_fund(
            sliced_nav,
            target.reindex(sliced_nav.index),
            initial_target=float(prior.iloc[-1]) if len(prior) else 0.0,
            liquidate_at_end=True,
        )
        ledger = result["ledger"].copy()
        ledger["daily_return"] = result["returns"]
        return ledger, result["returns"], result["metrics"]

    buy_hold_ledger, buy_hold_returns, buy_hold_metrics = evaluate(buy_hold_target)
    overlay_ledger, overlay_returns, overlay_metrics = evaluate(overlay_target)
    equity = pd.DataFrame(
        {
            "buy_hold": buy_hold_ledger["portfolio_value"],
            "all_in_t_overlay": overlay_ledger["portfolio_value"],
        }
    )
    summary = pd.DataFrame(
        [
            {"strategy": "buy_hold", **buy_hold_metrics},
            {"strategy": "all_in_t_overlay", **overlay_metrics},
        ]
    )
    trades = overlay_history.copy()
    trades = trades.loc[
        (pd.to_datetime(trades["date"]) >= test_start)
        & (pd.to_datetime(trades["date"]) <= test_end)
        & (trades["action"] != "HOLD")
    ].copy()
    trades["series"] = "all_in_t_overlay"
    return TradeOverlayResult(summary=summary, equity=equity, trade_log=trades)


def export_strategy_curves(output_dir: str | Path = "output/research_v8/strategy_curves") -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    cpo_bundle = build_single_fund_curve_bundle("cpo_communication")
    portfolio_bundle = build_portfolio_curve_bundle()

    outputs: dict[str, Path] = {}
    for prefix, bundle in (("cpo", cpo_bundle), ("portfolio", portfolio_bundle)):
        equity_path = output / f"{prefix}_equity_curves.csv"
        returns_path = output / f"{prefix}_daily_returns.csv"
        desc_path = output / f"{prefix}_strategy_descriptions.csv"
        equity_chart = output / f"{prefix}_equity_curves.svg"
        returns_chart = output / f"{prefix}_daily_returns.svg"

        bundle.equity.to_csv(equity_path, encoding="utf-8-sig")
        bundle.daily_returns.to_csv(returns_path, encoding="utf-8-sig")
        bundle.descriptions.to_csv(desc_path, index=False, encoding="utf-8-sig")
        equity_chart.write_text(
            _line_chart_svg(bundle.equity, f"{bundle.title} locked-test equity curves", "Portfolio value", False),
            encoding="utf-8",
        )
        returns_chart.write_text(
            _line_chart_svg(bundle.daily_returns, f"{bundle.title} daily return volatility", "Daily return", True),
            encoding="utf-8",
        )
        outputs[f"{prefix}_equity_csv"] = equity_path
        outputs[f"{prefix}_returns_csv"] = returns_path
        outputs[f"{prefix}_descriptions_csv"] = desc_path
        outputs[f"{prefix}_equity_svg"] = equity_chart
        outputs[f"{prefix}_returns_svg"] = returns_chart
    return outputs


def export_all_in_t_strategy(
    output_dir: str | Path = "output/research_v10/all_in_t_strategy",
    asset: str = "cpo_communication",
) -> dict[str, Path]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    result = build_all_in_t_strategy(asset)
    summary_path = output / f"{asset}_summary.csv"
    equity_path = output / f"{asset}_equity.csv"
    trades_path = output / f"{asset}_trade_points.csv"
    chart_path = output / f"{asset}_equity_with_trades.svg"
    result.summary.to_csv(summary_path, index=False, encoding="utf-8-sig")
    result.equity.to_csv(equity_path, encoding="utf-8-sig")
    result.trade_log.to_csv(trades_path, index=False, encoding="utf-8-sig")
    chart_path.write_text(
        _line_chart_with_markers_svg(
            result.equity,
            f"{asset} all-in then T overlay vs buy hold",
            result.trade_log,
        ),
        encoding="utf-8",
    )
    return {
        "summary_csv": summary_path,
        "equity_csv": equity_path,
        "trade_points_csv": trades_path,
        "equity_svg": chart_path,
    }
