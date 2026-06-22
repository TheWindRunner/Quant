"""Daily medium-term theme strategy for one-to-three month holding periods."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import performance_metrics
from .multifactor import MultiFactorConfig


THEMES = {
    "cpo": {
        "fund_code": "007817",
        "fund_name": "国泰中证全指通信设备ETF联接A",
        "cn": [
            {"name": "中际旭创", "asset_type": "cn_stock", "symbol": "300308", "adjust": "qfq"},
            {"name": "新易盛", "asset_type": "cn_stock", "symbol": "300502", "adjust": "qfq"},
            {"name": "天孚通信", "asset_type": "cn_stock", "symbol": "300394", "adjust": "qfq"},
            {"name": "光迅科技", "asset_type": "cn_stock", "symbol": "002281", "adjust": "qfq"},
        ],
        "us": [
            {"name": "COHR", "asset_type": "us_stock", "symbol": "105.COHR", "adjust": "qfq"},
            {"name": "LITE", "asset_type": "us_stock", "symbol": "105.LITE", "adjust": "qfq"},
            {"name": "AAOI", "asset_type": "us_stock", "symbol": "105.AAOI", "adjust": "qfq"},
            {"name": "FN", "asset_type": "us_stock", "symbol": "106.FN", "adjust": "qfq"},
        ],
    },
    "memory": {
        "fund_code": "008887",
        "fund_name": "华夏国证半导体芯片ETF联接A",
        "cn": [
            {"name": "兆易创新", "asset_type": "cn_stock", "symbol": "603986", "adjust": "qfq"},
            {"name": "江波龙", "asset_type": "cn_stock", "symbol": "301308", "adjust": "qfq"},
            {"name": "佰维存储", "asset_type": "cn_stock", "symbol": "688525", "adjust": "qfq"},
            {"name": "北京君正", "asset_type": "cn_stock", "symbol": "300223", "adjust": "qfq"},
        ],
        "us": [
            {"name": "MU", "asset_type": "us_stock", "symbol": "105.MU", "adjust": "qfq"},
            {"name": "WDC", "asset_type": "us_stock", "symbol": "105.WDC", "adjust": "qfq"},
            {"name": "STX", "asset_type": "us_stock", "symbol": "105.STX", "adjust": "qfq"},
        ],
    },
    "pcb": {
        "fund_code": "",
        "fund_name": "待确认高PCB持仓基金",
        "cn": [
            {"name": "胜宏科技", "asset_type": "cn_stock", "symbol": "300476", "adjust": "qfq"},
            {"name": "沪电股份", "asset_type": "cn_stock", "symbol": "002463", "adjust": "qfq"},
            {"name": "深南电路", "asset_type": "cn_stock", "symbol": "002916", "adjust": "qfq"},
            {"name": "生益科技", "asset_type": "cn_stock", "symbol": "600183", "adjust": "qfq"},
        ],
        "us": [
            {"name": "TTMI", "asset_type": "us_stock", "symbol": "105.TTMI", "adjust": "qfq"},
            {"name": "SANM", "asset_type": "us_stock", "symbol": "105.SANM", "adjust": "qfq"},
            {"name": "FLEX", "asset_type": "us_stock", "symbol": "105.FLEX", "adjust": "qfq"},
            {"name": "JBL", "asset_type": "us_stock", "symbol": "106.JBL", "adjust": "qfq"},
        ],
    },
}


@dataclass(frozen=True)
class DailyStrategyConfig:
    fast_ma: int = 10
    medium_ma: int = 20
    slow_ma: int = 60
    momentum_window: int = 20
    volatility_window: int = 20
    max_volatility: float = 0.55
    stop_loss: float = 0.10
    take_profit_trail: float = 0.12
    transaction_cost: float = 0.003
    minimum_hold_days: int = 20


def equal_weight_nav(prices: pd.DataFrame) -> pd.Series:
    returns = prices.sort_index().ffill().pct_change(fill_method=None)
    return (1 + returns.mean(axis=1, skipna=True).fillna(0.0)).cumprod()


def daily_features(
    nav: pd.Series,
    us_nav: pd.Series | None = None,
    config: DailyStrategyConfig | None = None,
) -> pd.DataFrame:
    config = config or DailyStrategyConfig()
    frame = pd.DataFrame(index=nav.index)
    frame["nav"] = nav
    frame["ma10"] = nav.rolling(config.fast_ma).mean()
    frame["ma20"] = nav.rolling(config.medium_ma).mean()
    frame["ma60"] = nav.rolling(config.slow_ma).mean()
    frame["momentum20"] = nav.pct_change(config.momentum_window, fill_method=None)
    frame["momentum60"] = nav.pct_change(config.slow_ma, fill_method=None)
    frame["volatility20"] = (
        nav.pct_change(fill_method=None).rolling(config.volatility_window).std()
        * np.sqrt(252)
    )
    if us_nav is not None:
        us_aligned = us_nav.reindex(nav.index).ffill()
        frame["us_momentum20"] = us_aligned.pct_change(20, fill_method=None).shift(1)
    else:
        frame["us_momentum20"] = 0.0
    return frame


def generate_positions(
    features: pd.DataFrame,
    config: DailyStrategyConfig | None = None,
) -> pd.DataFrame:
    """Generate daily target positions and explicit trade actions."""
    config = config or DailyStrategyConfig()
    result = features.copy()
    positions: list[float] = []
    actions: list[str] = []
    position = 0.0
    entry_price = np.nan
    peak = np.nan
    confirmation_days = 0
    holding_days = 0

    for row in result.itertuples():
        price = float(row.nav)
        trend_ok = (
            price > row.ma20 > row.ma60
            and row.momentum20 > 0
            and row.momentum60 > 0
            and row.volatility20 < config.max_volatility
            and row.us_momentum20 > 0
        )
        exit_trend = price < row.ma60 or row.momentum60 < 0

        if position == 0:
            holding_days = 0
            confirmation_days = confirmation_days + 1 if trend_ok else 0
            if confirmation_days >= 2:
                position = 1 / 3
                entry_price = price
                peak = price
                action = "BUY_1_3"
            else:
                action = "WATCH"
        else:
            holding_days += 1
            peak = max(float(peak), price)
            loss = price / float(entry_price) - 1
            trailing = price / float(peak) - 1
            hard_exit = loss <= -config.stop_loss or trailing <= -config.take_profit_trail
            trend_exit = exit_trend and holding_days >= config.minimum_hold_days
            if hard_exit or trend_exit:
                position = 0.0
                confirmation_days = 0
                entry_price = np.nan
                peak = np.nan
                holding_days = 0
                action = "SELL"
            elif trend_ok and position < 1.0:
                old = position
                position = min(1.0, position + 1 / 3)
                action = "ADD" if position > old else "HOLD"
            elif price < row.ma20 and position > 1 / 3:
                position = max(1 / 3, position - 1 / 3)
                action = "REDUCE"
            else:
                action = "HOLD"
        positions.append(position)
        actions.append(action)

    result["target_position"] = positions
    result["action"] = actions
    return result


def compare_fund_strategies(
    nav: pd.Series,
    us_nav: pd.Series | None = None,
    start: str | pd.Timestamp | None = None,
    config: DailyStrategyConfig | None = None,
) -> pd.DataFrame:
    """Compare buy-and-hold with 20/60 MA and full daily decision strategies."""
    config = config or DailyStrategyConfig()
    features = daily_features(nav, us_nav, config)
    decisions = generate_positions(features, config)
    daily_result = backtest_daily(nav, decisions, start=start, config=config)

    ma_signal = (
        (features["nav"] > features["ma20"])
        & (features["ma20"] > features["ma60"])
        & (features["momentum60"] > 0)
    ).astype(float)
    ma_decisions = features.copy()
    ma_decisions["target_position"] = ma_signal
    ma_result = backtest_daily(nav, ma_decisions, start=start, config=config)

    benchmark_nav = nav.loc[nav.index >= pd.Timestamp(start)] if start else nav
    benchmark_returns = benchmark_nav.pct_change(fill_method=None).fillna(0.0)
    benchmark_metrics = performance_metrics(benchmark_returns)
    benchmark_metrics["trade_count"] = 1.0
    benchmark_metrics["invested_day_ratio"] = 1.0

    rows = {
        "buy_hold": benchmark_metrics,
        "ma20_60": ma_result["metrics"],
        "daily_fund_strategy": daily_result["metrics"],
    }
    return pd.DataFrame(rows).T


def backtest_daily(
    nav: pd.Series,
    decisions: pd.DataFrame,
    start: str | pd.Timestamp | None = None,
    config: DailyStrategyConfig | None = None,
) -> dict[str, object]:
    """Backtest with close signal and next trading-day execution."""
    config = config or DailyStrategyConfig()
    frame = decisions.reindex(nav.index).copy()
    if start is not None:
        frame = frame.loc[frame.index >= pd.Timestamp(start)]
        nav = nav.reindex(frame.index)
    returns = nav.pct_change(fill_method=None).fillna(0.0)
    executed = frame["target_position"].shift(1).fillna(0.0)
    turnover = executed.diff().abs().fillna(executed.iloc[0])
    strategy_returns = executed * returns - turnover * config.transaction_cost
    equity = (1 + strategy_returns).cumprod()
    buy_hold = (1 + returns).cumprod()
    metrics = performance_metrics(strategy_returns)
    metrics["trade_count"] = float((turnover > 0).sum())
    metrics["invested_day_ratio"] = float((executed > 0).mean())
    return {
        "equity": pd.DataFrame({"strategy": equity, "buy_hold": buy_hold}),
        "returns": strategy_returns,
        "position": executed,
        "turnover": turnover,
        "metrics": metrics,
    }


def decisions_from_multifactor(
    scored_panel: pd.DataFrame,
    max_theme_weight: float = 0.20,
) -> pd.DataFrame:
    """Convert portfolio weights to the relative position expected by the backtester."""
    decisions = scored_panel.copy()
    decisions["target_position"] = (
        decisions["target_position"] / max_theme_weight
    ).clip(0, 1)
    return decisions


def lagged_daily_correlation(
    cn_nav: pd.Series,
    us_nav: pd.Series,
    window: int = 120,
    max_lag: int = 5,
) -> pd.DataFrame:
    cn_return = cn_nav.pct_change(fill_method=None)
    us_return = us_nav.pct_change(fill_method=None)
    rows = []
    for lag in range(max_lag + 1):
        pair = pd.concat(
            [cn_return.rename("cn"), us_return.shift(lag).rename("us")], axis=1
        ).dropna().tail(window)
        rows.append(
            {
                "us_lead_days": lag,
                "correlation": pair["cn"].corr(pair["us"]),
                "observations": len(pair),
                "cn_annual_volatility": pair["cn"].std() * np.sqrt(252),
                "us_annual_volatility": pair["us"].std() * np.sqrt(252),
            }
        )
    return pd.DataFrame(rows)
