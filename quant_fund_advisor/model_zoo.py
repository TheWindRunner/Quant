"""Distinct medium-term model families evaluated under one protocol."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ModelSignal:
    name: str
    position: pd.Series
    description: str


def buy_hold(nav: pd.Series) -> ModelSignal:
    return ModelSignal("buy_hold", pd.Series(1.0, index=nav.index), "Always invested")


def dual_moving_average(nav: pd.Series, fast: int = 20, slow: int = 60) -> ModelSignal:
    fast_ma = nav.rolling(fast).mean()
    slow_ma = nav.rolling(slow).mean()
    position = ((nav > fast_ma) & (fast_ma > slow_ma)).astype(float)
    return ModelSignal("dual_ma", position, f"NAV > MA{fast} > MA{slow}")


def time_series_momentum(nav: pd.Series) -> ModelSignal:
    """Diversify horizons rather than optimizing one lookback."""
    returns = pd.DataFrame(
        {
            "m20": nav.pct_change(20, fill_method=None),
            "m60": nav.pct_change(60, fill_method=None),
            "m120": nav.pct_change(120, fill_method=None),
        }
    )
    vote = (returns > 0).mean(axis=1)
    position = vote.where(vote >= 2 / 3, 0.0)
    return ModelSignal(
        "ts_momentum",
        position,
        "Fraction of positive 20/60/120-day momentum horizons",
    )


def donchian_breakout(nav: pd.Series, entry: int = 60, exit: int = 20) -> ModelSignal:
    upper = nav.shift(1).rolling(entry).max()
    lower = nav.shift(1).rolling(exit).min()
    state = 0.0
    positions = []
    for date in nav.index:
        if nav.loc[date] >= upper.loc[date]:
            state = 1.0
        elif nav.loc[date] <= lower.loc[date]:
            state = 0.0
        positions.append(state)
    return ModelSignal(
        "donchian",
        pd.Series(positions, index=nav.index),
        f"{entry}-day breakout with {exit}-day exit",
    )


def volatility_managed_momentum(
    nav: pd.Series,
    target_volatility: float = 0.20,
) -> ModelSignal:
    returns = nav.pct_change(fill_method=None)
    realized = returns.rolling(20).std() * np.sqrt(252)
    trend = nav.pct_change(60, fill_method=None) > 0
    scaled = (target_volatility / realized).clip(0, 1)
    position = scaled.where(trend, 0.0).fillna(0.0)
    return ModelSignal(
        "vol_managed_momentum",
        position,
        "60-day momentum scaled down when realized volatility rises",
    )


def trend_quality(nav: pd.Series, window: int = 60) -> ModelSignal:
    """Regression slope times R-squared rewards smooth trends."""
    log_nav = np.log(nav)

    def score(values: np.ndarray) -> float:
        if not np.isfinite(values).all():
            return np.nan
        x = np.arange(len(values), dtype=float)
        slope, intercept = np.polyfit(x, values, 1)
        fitted = slope * x + intercept
        total = ((values - values.mean()) ** 2).sum()
        residual = ((values - fitted) ** 2).sum()
        r_squared = 1 - residual / total if total > 0 else 0.0
        annualized_slope = np.expm1(slope * 252)
        return float(annualized_slope * max(0.0, r_squared))

    quality = log_nav.rolling(window).apply(score, raw=True)
    expanding_median = quality.expanding(min_periods=120).median()
    position = ((quality > 0) & (quality > expanding_median)).astype(float)
    return ModelSignal(
        "trend_quality",
        position,
        "Positive log-price regression slope filtered by R-squared",
    )


def regime_adaptive(
    nav: pd.Series,
    market_nav: pd.Series | None = None,
) -> ModelSignal:
    market = nav if market_nav is None else market_nav.reindex(nav.index).ffill()
    market_return = market.pct_change(fill_method=None)
    market_vol = market_return.rolling(20).std() * np.sqrt(252)
    vol_threshold = market_vol.expanding(min_periods=120).quantile(0.7)
    market_trend = market / market.rolling(60).mean() - 1
    fund_momentum = nav.pct_change(60, fill_method=None)
    drawdown = nav / nav.rolling(120, min_periods=20).max() - 1
    normal = (market_trend > 0) & (market_vol <= vol_threshold)
    stressed = market_vol > vol_threshold
    position = pd.Series(0.0, index=nav.index)
    position[normal & (fund_momentum > 0)] = 1.0
    position[stressed & (fund_momentum > 0.08) & (drawdown > -0.08)] = 0.33
    return ModelSignal(
        "regime_adaptive",
        position,
        "Risk-on trend exposure; reduced exposure in high-volatility regimes",
    )


def price_sentiment_regime(
    nav: pd.Series,
    market_nav: pd.Series | None = None,
    peers: pd.DataFrame | None = None,
) -> ModelSignal:
    """Reproducible historical sentiment from breadth, trend and volatility."""
    market = nav if market_nav is None else market_nav.reindex(nav.index).ffill()
    market_return = market.pct_change(fill_method=None)
    market_momentum = market.pct_change(20, fill_method=None)
    market_vol = market_return.rolling(20).std() * np.sqrt(252)
    vol_percentile = market_vol.rolling(252, min_periods=120).rank(pct=True)
    drawdown = market / market.rolling(120, min_periods=20).max() - 1
    if peers is None or peers.empty:
        breadth = (market_momentum > 0).astype(float)
    else:
        breadth = (
            peers.reindex(nav.index)
            .ffill()
            .pct_change(20, fill_method=None)
            .gt(0)
            .mean(axis=1)
        )
    sentiment = (
        0.35 * (breadth * 2 - 1)
        + 0.30 * np.tanh(market_momentum * 8)
        - 0.20 * (vol_percentile * 2 - 1)
        + 0.15 * np.tanh(drawdown * 8)
    )
    fund_momentum = nav.pct_change(60, fill_method=None)
    position = pd.Series(0.0, index=nav.index)
    position[(sentiment > 0) & (fund_momentum > 0)] = 0.5
    position[(sentiment > 0.25) & (fund_momentum > 0)] = 1.0
    return ModelSignal(
        "price_sentiment",
        position,
        "Technology breadth, market momentum, volatility percentile and drawdown",
    )


def crowding_aware_momentum(nav: pd.Series) -> ModelSignal:
    """Reduce hot-theme exposure when return and volatility are jointly extreme."""
    returns = nav.pct_change(fill_method=None)
    momentum = nav.pct_change(60, fill_method=None)
    short_return = nav.pct_change(20, fill_method=None)
    volatility = returns.rolling(20).std() * np.sqrt(252)
    return_pct = short_return.rolling(252, min_periods=120).rank(pct=True)
    vol_pct = volatility.rolling(252, min_periods=120).rank(pct=True)
    drawdown = nav / nav.rolling(60).max() - 1
    base = (momentum > 0).astype(float)
    crowded = (return_pct > 0.90) & (vol_pct > 0.75)
    reversal = drawdown < -0.08
    position = base.copy()
    position[crowded] = 0.5
    position[reversal] = 0.0
    return ModelSignal(
        "crowding_aware",
        position,
        "Momentum with joint return-volatility crowding and drawdown exits",
    )


def hysteresis_trend(
    nav: pd.Series,
    entry_buffer: float = 0.02,
    exit_buffer: float = 0.03,
) -> ModelSignal:
    """Slow state changes reduce whipsaw and short-redemption fees."""
    ma60 = nav.rolling(60).mean()
    ma120 = nav.rolling(120).mean()
    state = 0.0
    positions = []
    for date in nav.index:
        if state == 0.0 and nav.loc[date] > ma60.loc[date] * (1 + entry_buffer):
            state = 1.0
        elif state == 1.0 and nav.loc[date] < ma120.loc[date] * (1 - exit_buffer):
            state = 0.0
        positions.append(state)
    return ModelSignal(
        "hysteresis_trend",
        pd.Series(positions, index=nav.index),
        "Buffered 60-day entry and 120-day exit to reduce whipsaw",
    )


def tail_risk_overlay(nav: pd.Series) -> ModelSignal:
    """Remain invested unless trend, drawdown and volatility jointly deteriorate."""
    returns = nav.pct_change(fill_method=None)
    ma120 = nav.rolling(120).mean()
    ma60 = nav.rolling(60).mean()
    drawdown = nav / nav.rolling(120, min_periods=20).max() - 1
    volatility = returns.rolling(20).std() * np.sqrt(252)
    vol_threshold = volatility.rolling(252, min_periods=120).quantile(0.75)
    defensive = (nav < ma120) & (drawdown < -0.10) & (volatility > vol_threshold)
    recovered = (nav > ma60) & (drawdown > -0.06)
    state = 1.0
    positions = []
    for date in nav.index:
        if defensive.loc[date]:
            state = 0.50
        elif recovered.loc[date]:
            state = 1.0
        positions.append(state)
    return ModelSignal(
        "tail_risk_overlay",
        pd.Series(positions, index=nav.index),
        "Full exposure except joint long-trend, drawdown and volatility stress",
    )


def walk_forward_ridge(
    nav: pd.Series,
    forecast_horizon: int = 20,
    training_window: int = 504,
    retrain_frequency: int = 5,
    ridge_alpha: float = 10.0,
) -> ModelSignal:
    """Fixed-feature online ridge model with label maturity enforced."""
    returns = nav.pct_change(fill_method=None)
    features = pd.DataFrame(
        {
            "ret5": nav.pct_change(5, fill_method=None),
            "ret20": nav.pct_change(20, fill_method=None),
            "ret60": nav.pct_change(60, fill_method=None),
            "vol20": returns.rolling(20).std() * np.sqrt(252),
            "downside20": returns.clip(upper=0).rolling(20).std() * np.sqrt(252),
            "drawdown60": nav / nav.rolling(60).max() - 1,
            "trend20_60": nav.rolling(20).mean() / nav.rolling(60).mean() - 1,
        },
        index=nav.index,
    )
    forward_return = nav.shift(-forecast_horizon) / nav - 1
    predictions = pd.Series(np.nan, index=nav.index)
    beta: np.ndarray | None = None

    for location in range(len(nav)):
        if location % retrain_frequency == 0:
            # At location t, the latest knowable label belongs to t-horizon.
            train_end = location - forecast_horizon
            train_start = max(0, train_end - training_window)
            if train_end - train_start >= 120:
                x_train = features.iloc[train_start:train_end]
                y_train = forward_return.iloc[train_start:train_end]
                valid = x_train.notna().all(axis=1) & y_train.notna()
                x_valid = x_train.loc[valid]
                y_valid = y_train.loc[valid]
                if len(x_valid) >= 120:
                    mean = x_valid.mean()
                    std = x_valid.std().replace(0.0, 1.0)
                    standardized = (x_valid - mean) / std
                    matrix = np.column_stack(
                        [np.ones(len(standardized)), standardized.to_numpy()]
                    )
                    penalty = np.eye(matrix.shape[1]) * ridge_alpha
                    penalty[0, 0] = 0.0
                    beta = np.linalg.solve(
                        matrix.T @ matrix + penalty,
                        matrix.T @ y_valid.to_numpy(),
                    )
                    fitted_mean = mean
                    fitted_std = std
        if beta is not None and features.iloc[location].notna().all():
            row = (features.iloc[location] - fitted_mean) / fitted_std
            predictions.iloc[location] = np.r_[1.0, row.to_numpy()] @ beta

    rolling_noise = forward_return.rolling(252, min_periods=120).std()
    conviction = (predictions / rolling_noise.replace(0.0, np.nan)).clip(0, 1)
    raw_position = conviction.where(predictions > 0, 0.0).fillna(0.0)
    # Forecasts update daily, but subscriptions should not chase tiny changes.
    position = pd.Series(0.0, index=nav.index)
    position[raw_position >= 0.20] = 0.5
    position[raw_position >= 0.60] = 1.0
    return ModelSignal(
        "walk_forward_ridge",
        position,
        "Weekly retrained ridge regression with 20-day label maturity",
    )


def relative_strength(
    nav: pd.Series,
    peers: pd.DataFrame,
) -> ModelSignal:
    peer_momentum = peers.reindex(nav.index).ffill().pct_change(60, fill_method=None)
    fund_momentum = nav.pct_change(60, fill_method=None)
    combined = pd.concat(
        [fund_momentum.rename("__fund__"), peer_momentum], axis=1
    )
    percentile = combined.rank(axis=1, pct=True)["__fund__"]
    position = pd.Series(0.0, index=nav.index)
    position[percentile >= 0.60] = 0.5
    position[percentile >= 0.80] = 1.0
    return ModelSignal(
        "relative_strength",
        position,
        "Fund/theme momentum relative to independent technology peers",
    )


def _rolling_percentile(
    series: pd.Series,
    window: int = 252,
    min_periods: int = 60,
) -> pd.Series:
    def score(values: np.ndarray) -> float:
        current = values[-1]
        valid = values[np.isfinite(values)]
        if not np.isfinite(current) or len(valid) == 0:
            return np.nan
        return float((valid <= current).mean() * 2 - 1)

    return series.rolling(window, min_periods=min_periods).apply(
        score,
        raw=True,
    )


def external_multifactor_timing(
    nav: pd.Series,
    external_factors: pd.DataFrame | None = None,
    core_weight: float = 0.70,
) -> ModelSignal:
    """Use tradable-market, valuation and flow proxies beyond the fund NAV."""
    if external_factors is None or external_factors.empty:
        return ModelSignal(
            "external_multifactor",
            pd.Series(1.0, index=nav.index),
            "No external factor history available; fall back to full exposure",
        )
    factors = external_factors.reindex(nav.index).ffill()
    returns = nav.pct_change(fill_method=None)
    trend = nav.rolling(20).mean() / nav.rolling(60).mean() - 1
    drawdown60 = nav / nav.rolling(60).max() - 1
    vol20 = returns.rolling(20).std() * np.sqrt(252)

    if "etf_amount" in factors:
        amount_impulse = factors["etf_amount"] / factors["etf_amount"].rolling(20).mean() - 1
    else:
        amount_impulse = pd.Series(0.0, index=nav.index)
    if "etf_turnover" in factors:
        turnover_heat = _rolling_percentile(factors["etf_turnover"]).fillna(0.0)
    else:
        turnover_heat = pd.Series(0.0, index=nav.index)
    if "etf_amplitude" in factors:
        amplitude_heat = _rolling_percentile(factors["etf_amplitude"]).fillna(0.0)
    else:
        amplitude_heat = pd.Series(0.0, index=nav.index)

    valuation_parts = []
    for column in ("pe_1", "pe_2", "pb", "pb_1"):
        if column in factors:
            valuation_parts.append(-_rolling_percentile(factors[column]).fillna(0.0))
    for column in ("dividend_yield_1", "dividend_yield_2"):
        if column in factors:
            valuation_parts.append(_rolling_percentile(factors[column]).fillna(0.0))
    valuation = (
        pd.concat(valuation_parts, axis=1).mean(axis=1)
        if valuation_parts
        else pd.Series(0.0, index=nav.index)
    )

    commodity_momentum = (
        factors["commodity_close"].pct_change(20, fill_method=None)
        if "commodity_close" in factors
        else pd.Series(0.0, index=nav.index)
    )
    score = (
        0.30 * np.tanh(trend * 20)
        + 0.20 * np.tanh(amount_impulse * 2)
        + 0.18 * valuation
        + 0.12 * np.tanh(commodity_momentum * 5)
        - 0.12 * turnover_heat.clip(lower=0)
        - 0.08 * amplitude_heat.clip(lower=0)
        + 0.10 * np.tanh(drawdown60 * 8)
    )
    position = pd.Series(core_weight, index=nav.index, dtype=float)
    position[(score > 0.15) & (trend > 0)] = 1.0
    position[(score < -0.10) | ((drawdown60 < -0.08) & (vol20 > vol20.rolling(120, min_periods=60).quantile(0.65)))] = core_weight * 0.50
    position[(score < -0.25) & (drawdown60 < -0.10)] = core_weight * 0.25
    return ModelSignal(
        "external_multifactor",
        position.clip(0, 1).fillna(core_weight),
        "ETF liquidity/turnover/amplitude, index valuation, commodity proxy and NAV risk state",
    )


def drawdown_control_timing(
    nav: pd.Series,
    external_factors: pd.DataFrame | None = None,
    core_weight: float = 0.75,
) -> ModelSignal:
    """Cut exposure during persistent drawdowns and buy back on repair."""
    returns = nav.pct_change(fill_method=None)
    ma10 = nav.rolling(10).mean()
    ma20 = nav.rolling(20).mean()
    ma60 = nav.rolling(60).mean()
    high20 = nav.rolling(20).max()
    drawdown20 = nav / high20 - 1
    drawdown60 = nav / nav.rolling(60).max() - 1
    downside = returns.clip(upper=0).rolling(10).std() * np.sqrt(252)
    if external_factors is not None and not external_factors.empty and "etf_amount" in external_factors:
        amount = external_factors["etf_amount"].reindex(nav.index).ffill()
        volume_confirm = amount > amount.rolling(20).mean()
    else:
        volume_confirm = pd.Series(True, index=nav.index)

    state = 1.0
    positions = []
    for date in nav.index:
        falling = (
            nav.loc[date] < ma20.loc[date]
            and ma10.loc[date] < ma20.loc[date]
            and drawdown20.loc[date] < -0.04
            and downside.loc[date] > downside.rolling(120, min_periods=60).median().loc[date]
        )
        panic = drawdown60.loc[date] < -0.10
        repaired = (
            nav.loc[date] > ma10.loc[date]
            and returns.rolling(3).sum().loc[date] > 0
            and volume_confirm.loc[date]
        )
        trend_recovered = nav.loc[date] > ma60.loc[date] and ma10.loc[date] > ma20.loc[date]
        if state > core_weight * 0.50 and falling:
            state = core_weight * 0.50
        elif state > core_weight * 0.25 and panic:
            state = core_weight * 0.25
        elif state < 1.0 and (repaired or trend_recovered):
            state = min(1.0, state + 0.25)
        positions.append(state)
    return ModelSignal(
        "drawdown_control_t",
        pd.Series(positions, index=nav.index).clip(0, 1),
        "FIFO-friendly T: reduce in persistent drawdown, buy back on repair/volume confirmation",
    )


def regime_relative_strength(
    nav: pd.Series,
    market_nav: pd.Series | None = None,
    peers: pd.DataFrame | None = None,
    core_weight: float = 0.70,
) -> ModelSignal:
    """Switch between attack and defense using market regime plus cross-theme strength."""
    market = nav if market_nav is None else market_nav.reindex(nav.index).ffill()
    market_return = market.pct_change(fill_method=None)
    market_vol = market_return.rolling(20).std() * np.sqrt(252)
    vol_percentile = market_vol.rolling(252, min_periods=120).rank(pct=True)
    market_ma60 = market.rolling(60).mean()
    market_ma120 = market.rolling(120).mean()
    market_momentum20 = market.pct_change(20, fill_method=None)
    market_momentum60 = market.pct_change(60, fill_method=None)
    fund_drawdown = nav / nav.rolling(120, min_periods=20).max() - 1
    fund_momentum20 = nav.pct_change(20, fill_method=None)
    fund_momentum60 = nav.pct_change(60, fill_method=None)

    if peers is None or peers.empty:
        breadth20 = (fund_momentum20 > 0).astype(float)
        percentile = pd.Series(0.5, index=nav.index)
    else:
        peer_frame = peers.reindex(nav.index).ffill()
        breadth20 = peer_frame.pct_change(20, fill_method=None).gt(0).mean(axis=1)
        combined = pd.concat(
            [fund_momentum60.rename("__fund__"), peer_frame.pct_change(60, fill_method=None)],
            axis=1,
        )
        percentile = combined.rank(axis=1, pct=True)["__fund__"].fillna(0.5)

    state = []
    current = "neutral"
    for date in nav.index:
        risk_on = (
            market.loc[date] > market_ma60.loc[date]
            and market_momentum20.loc[date] > 0
            and breadth20.loc[date] >= 0.50
            and vol_percentile.loc[date] <= 0.75
        )
        stressed = (
            market.loc[date] < market_ma120.loc[date]
            and market_momentum60.loc[date] < 0
            and vol_percentile.loc[date] >= 0.75
        )
        if stressed:
            current = "defense"
        elif risk_on:
            current = "attack"
        state.append(current)
    regime = pd.Series(state, index=nav.index)

    position = pd.Series(core_weight, index=nav.index, dtype=float)
    tactical = 1.0 - core_weight
    attack = (
        (regime == "attack")
        & (percentile >= 0.65)
        & (fund_momentum60 > 0)
        & (fund_drawdown > -0.12)
    )
    strong_attack = attack & (percentile >= 0.80) & (fund_momentum20 > 0)
    defense = (
        (regime == "defense")
        & ((fund_drawdown < -0.10) | (fund_momentum20 < 0))
    )
    position[attack] = core_weight + tactical * 0.50
    position[strong_attack] = 1.0
    position[defense] = core_weight * 0.50
    position[(regime == "neutral") & (percentile < 0.50)] = core_weight * 0.75
    return ModelSignal(
        "regime_relative_strength",
        position.clip(0, 1),
        "Market-regime switch with peer-relative strength and drawdown defense",
    )


def market_state_nowcast(
    nav: pd.Series,
    market_nav: pd.Series | None = None,
    peers: pd.DataFrame | None = None,
    training_window: int = 504,
    retrain_frequency: int = 5,
    ridge_alpha: float = 8.0,
) -> ModelSignal:
    """Rolling one-day direction model using only information known by the prior close."""
    market = nav if market_nav is None else market_nav.reindex(nav.index).ffill()
    returns = nav.pct_change(fill_method=None)
    market_returns = market.pct_change(fill_method=None)
    drawdown60 = nav / nav.rolling(60).max() - 1
    volatility20 = returns.rolling(20).std() * np.sqrt(252)
    downside20 = returns.clip(upper=0).rolling(20).std() * np.sqrt(252)
    market_vol20 = market_returns.rolling(20).std() * np.sqrt(252)
    market_drawdown60 = market / market.rolling(60).max() - 1
    if peers is None or peers.empty:
        breadth20 = (nav.pct_change(20, fill_method=None) > 0).astype(float)
        rs60 = pd.Series(0.5, index=nav.index)
    else:
        peer_frame = peers.reindex(nav.index).ffill()
        breadth20 = peer_frame.pct_change(20, fill_method=None).gt(0).mean(axis=1)
        combined = pd.concat(
            [nav.pct_change(60, fill_method=None).rename("__fund__"), peer_frame.pct_change(60, fill_method=None)],
            axis=1,
        )
        rs60 = combined.rank(axis=1, pct=True)["__fund__"].fillna(0.5)
    features = pd.DataFrame(
        {
            "ret1": returns.shift(1),
            "ret5": nav.pct_change(5, fill_method=None).shift(1),
            "ret20": nav.pct_change(20, fill_method=None).shift(1),
            "market_ret1": market_returns.shift(1),
            "market_ret5": market.pct_change(5, fill_method=None).shift(1),
            "market_ret20": market.pct_change(20, fill_method=None).shift(1),
            "vol20": volatility20.shift(1),
            "downside20": downside20.shift(1),
            "market_vol20": market_vol20.shift(1),
            "drawdown60": drawdown60.shift(1),
            "market_drawdown60": market_drawdown60.shift(1),
            "breadth20": breadth20.shift(1),
            "rs60": rs60.shift(1),
            "trend20_60": (nav.rolling(20).mean() / nav.rolling(60).mean() - 1).shift(1),
        },
        index=nav.index,
    )
    label = (returns > 0).astype(float)
    score = pd.Series(np.nan, index=nav.index)
    beta: np.ndarray | None = None
    fitted_mean: pd.Series | None = None
    fitted_std: pd.Series | None = None
    for location in range(len(nav)):
        if location % retrain_frequency == 0:
            train_end = location
            train_start = max(0, train_end - training_window)
            x_train = features.iloc[train_start:train_end]
            y_train = label.iloc[train_start:train_end]
            valid = x_train.notna().all(axis=1) & y_train.notna()
            x_valid = x_train.loc[valid]
            y_valid = y_train.loc[valid]
            if len(x_valid) >= 120:
                fitted_mean = x_valid.mean()
                fitted_std = x_valid.std().replace(0.0, 1.0)
                standardized = (x_valid - fitted_mean) / fitted_std
                matrix = np.column_stack(
                    [np.ones(len(standardized)), standardized.to_numpy()]
                )
                penalty = np.eye(matrix.shape[1]) * ridge_alpha
                penalty[0, 0] = 0.0
                beta = np.linalg.solve(
                    matrix.T @ matrix + penalty,
                    matrix.T @ y_valid.to_numpy(),
                )
        if (
            beta is not None
            and fitted_mean is not None
            and fitted_std is not None
            and features.iloc[location].notna().all()
        ):
            row = (features.iloc[location] - fitted_mean) / fitted_std
            score.iloc[location] = np.r_[1.0, row.to_numpy()] @ beta
    conviction = (score - 0.5).abs()
    position = pd.Series(0.0, index=nav.index, dtype=float)
    position[score >= 0.53] = 0.50
    position[score >= 0.60] = 1.0
    position[(score <= 0.47) & (conviction >= 0.03)] = 0.0
    return ModelSignal(
        "market_state_nowcast",
        position,
        "Rolling next-day direction model from lagged market state, breadth and relative strength",
    )


def robust_ensemble(signals: list[ModelSignal]) -> ModelSignal:
    if not signals:
        raise ValueError("At least one model signal is required")
    matrix = pd.concat(
        [signal.position.rename(signal.name) for signal in signals], axis=1
    ).fillna(0.0)
    # Median limits the influence of one unstable model.
    position = matrix.median(axis=1)
    position = position.where(position >= 0.5, 0.0).clip(0, 1)
    return ModelSignal(
        "robust_ensemble",
        position,
        "Median vote across economically distinct model families",
    )


def core_tactical(
    tactical: ModelSignal,
    core_weight: float = 0.70,
) -> ModelSignal:
    """Keep a strategic core while using the model only for the tactical sleeve."""
    position = core_weight + (1.0 - core_weight) * tactical.position.clip(0, 1)
    return ModelSignal(
        f"core{int(core_weight * 100)}_{tactical.name}",
        position,
        f"{core_weight:.0%} permanent core plus tactical {tactical.name}",
    )


def bull_hold_bear_defense(
    nav: pd.Series,
    market_nav: pd.Series,
    defensive: ModelSignal,
    core_weight: float = 0.70,
) -> ModelSignal:
    """Hold fully in bull regimes and activate defense only after deterioration."""
    market = market_nav.reindex(nav.index).ffill()
    ma60 = market.rolling(60).mean()
    ma120 = market.rolling(120).mean()
    momentum20 = market.pct_change(20, fill_method=None)
    momentum60 = market.pct_change(60, fill_method=None)
    risk_on = True
    states = []
    for date in nav.index:
        if (
            risk_on
            and market.loc[date] < ma120.loc[date]
            and momentum60.loc[date] < 0
        ):
            risk_on = False
        elif (
            not risk_on
            and market.loc[date] > ma60.loc[date]
            and momentum20.loc[date] > 0
        ):
            risk_on = True
        states.append(risk_on)
    state = pd.Series(states, index=nav.index)
    defensive_position = (
        core_weight
        + (1.0 - core_weight) * defensive.position.clip(0, 1)
    )
    position = defensive_position.where(~state, 1.0)
    return ModelSignal(
        f"bull_hold_bear_{defensive.name}",
        position,
        "Full exposure in bull regimes; core plus defensive signal in bear regimes",
    )


def build_model_zoo(
    nav: pd.Series,
    market_nav: pd.Series | None = None,
    peers: pd.DataFrame | None = None,
    external_factors: pd.DataFrame | None = None,
) -> list[ModelSignal]:
    core = [
        buy_hold(nav),
        dual_moving_average(nav),
        time_series_momentum(nav),
        donchian_breakout(nav),
        volatility_managed_momentum(nav),
        trend_quality(nav),
        regime_adaptive(nav, market_nav),
        price_sentiment_regime(nav, market_nav, peers),
        crowding_aware_momentum(nav),
        regime_relative_strength(nav, market_nav, peers),
        hysteresis_trend(nav),
        tail_risk_overlay(nav),
        walk_forward_ridge(nav),
        market_state_nowcast(nav, market_nav, peers),
        external_multifactor_timing(nav, external_factors),
        drawdown_control_timing(nav, external_factors),
    ]
    if peers is not None and not peers.empty:
        core.append(relative_strength(nav, peers))
    tactical_by_name = {
        signal.name: signal for signal in core if signal.name != "buy_hold"
    }
    for name in (
        "donchian",
        "trend_quality",
        "crowding_aware",
        "ts_momentum",
        "walk_forward_ridge",
        "relative_strength",
        "regime_relative_strength",
        "market_state_nowcast",
    ):
        if name in tactical_by_name:
            core.append(core_tactical(tactical_by_name[name], 0.70))
            core.append(core_tactical(tactical_by_name[name], 0.80))
            core.append(core_tactical(tactical_by_name[name], 0.90))
    ensemble_inputs = [signal for signal in core if signal.name != "buy_hold"]
    core.append(robust_ensemble(ensemble_inputs))
    return core
