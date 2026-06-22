"""Walk-forward portfolio backtest with delayed execution and costs."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    top_n: int = 3
    rebalance: str = "W-FRI"
    minimum_score: float = 0.0
    max_asset_weight: float = 0.40
    transaction_cost: float = 0.003
    rebalance_buffer: float = 0.10
    annual_risk_free_rate: float = 0.02


def _target_weights(scores: pd.Series, config: BacktestConfig) -> pd.Series:
    selected = scores[scores >= config.minimum_score].nlargest(config.top_n)
    weights = pd.Series(0.0, index=scores.index)
    if selected.empty:
        return weights
    raw = selected.clip(lower=0.01)
    raw = raw / raw.sum()
    capped = raw.clip(upper=config.max_asset_weight)
    weights.loc[capped.index] = capped
    return weights


def run_backtest(
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    config: BacktestConfig | None = None,
) -> dict[str, pd.DataFrame | pd.Series | dict[str, float]]:
    config = config or BacktestConfig()
    prices = prices.sort_index().ffill()
    scores = scores.reindex(prices.index).ffill().fillna(0.0)
    asset_returns = prices.pct_change(fill_method=None).fillna(0.0)

    rebalance_dates = (
        pd.Series(prices.index, index=prices.index)
        .resample(config.rebalance)
        .last()
        .dropna()
        .tolist()
    )
    decision_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    current = pd.Series(0.0, index=prices.columns)
    for date in prices.index:
        if date in rebalance_dates:
            proposed = _target_weights(scores.loc[date], config)
            if (proposed - current).abs().sum() >= config.rebalance_buffer:
                current = proposed
        decision_weights.loc[date] = current

    # Signals formed at close become executable on the next fund NAV date.
    executed_weights = decision_weights.shift(1).fillna(0.0)
    turnover = executed_weights.diff().abs().sum(axis=1).fillna(
        executed_weights.iloc[0].abs().sum()
    )
    gross_returns = (executed_weights * asset_returns).sum(axis=1)
    net_returns = gross_returns - turnover * config.transaction_cost
    equity = (1.0 + net_returns).cumprod()
    benchmark_returns = asset_returns.mean(axis=1)
    benchmark_equity = (1.0 + benchmark_returns).cumprod()

    metrics = performance_metrics(net_returns, config.annual_risk_free_rate)
    metrics["total_turnover"] = float(turnover.sum())
    return {
        "equity": pd.DataFrame(
            {"strategy": equity, "equal_weight_benchmark": benchmark_equity}
        ),
        "returns": net_returns.rename("strategy_return"),
        "weights": executed_weights,
        "turnover": turnover.rename("turnover"),
        "metrics": metrics,
    }


def performance_metrics(returns: pd.Series, annual_risk_free_rate: float = 0.02) -> dict[str, float]:
    returns = returns.dropna()
    if returns.empty:
        return {}
    equity = (1 + returns).cumprod()
    years = max(len(returns) / 252, 1 / 252)
    cagr = float(equity.iloc[-1] ** (1 / years) - 1)
    volatility = float(returns.std() * np.sqrt(252))
    excess = returns.mean() * 252 - annual_risk_free_rate
    sharpe = float(excess / volatility) if volatility > 0 else 0.0
    drawdown = equity / equity.cummax() - 1
    return {
        "total_return": float(equity.iloc[-1] - 1),
        "cagr": cagr,
        "annual_volatility": volatility,
        "sharpe": sharpe,
        "max_drawdown": float(drawdown.min()),
        "positive_day_ratio": float((returns > 0).mean()),
    }
