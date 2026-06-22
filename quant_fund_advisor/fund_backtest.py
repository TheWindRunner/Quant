"""Fee-aware backtesting for open-ended mutual-fund subscriptions/redemptions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .backtest import performance_metrics


@dataclass(frozen=True)
class RedemptionTier:
    max_holding_days: int | None
    fee_rate: float


@dataclass(frozen=True)
class FundFeeSchedule:
    purchase_fee_rate: float = 0.0015
    redemption_tiers: tuple[RedemptionTier, ...] = (
        RedemptionTier(7, 0.015),
        RedemptionTier(30, 0.005),
        RedemptionTier(365, 0.0025),
        RedemptionTier(None, 0.0),
    )


@dataclass
class FundLot:
    units: float
    purchase_date: pd.Timestamp


def redemption_fee_rate(
    holding_days: int,
    schedule: FundFeeSchedule,
) -> float:
    for tier in schedule.redemption_tiers:
        if tier.max_holding_days is None or holding_days < tier.max_holding_days:
            return tier.fee_rate
    return 0.0


def backtest_open_fund(
    nav: pd.Series,
    target_position: pd.Series,
    schedule: FundFeeSchedule | None = None,
    initial_capital: float = 1.0,
    initial_target: float = 0.0,
    liquidate_at_end: bool = True,
    minimum_rebalance_fraction: float = 0.05,
    execution_lag_days: int = 1,
    liquidate_final_trade_count: bool = True,
) -> dict[str, object]:
    """Execute target positions at fund NAV using FIFO redemption lots.

    Close-formed signals use the default one NAV-date lag. Intraday signals
    formed before the fund cutoff can set ``execution_lag_days=0``.
    """
    schedule = schedule or FundFeeSchedule()
    nav = nav.dropna().sort_index()
    signal = target_position.reindex(nav.index).ffill().fillna(0.0).clip(0, 1)
    if execution_lag_days < 0:
        raise ValueError("execution_lag_days must be non-negative")
    executable_target = (
        signal.shift(execution_lag_days).fillna(0.0)
        if execution_lag_days
        else signal.copy()
    )
    if len(executable_target):
        executable_target.iloc[0] = float(np.clip(initial_target, 0, 1))

    cash = float(initial_capital)
    lots: list[FundLot] = []
    records = []
    total_purchase_fees = 0.0
    total_redemption_fees = 0.0
    trade_count = 0
    redeemed_value_days = 0.0
    redeemed_value_total = 0.0
    redeemed_under_7 = 0.0
    redeemed_under_30 = 0.0

    for row_number, (date, price) in enumerate(nav.items()):
        units = sum(lot.units for lot in lots)
        gross_value = cash + units * price
        target_value = gross_value * float(executable_target.loc[date])
        current_fund_value = units * price
        trade_value = target_value - current_fund_value
        if gross_value > 0 and abs(trade_value) / gross_value < minimum_rebalance_fraction:
            trade_value = 0.0

        if trade_value > 1e-12 and cash > 0:
            spend = min(trade_value, cash)
            fee = spend * schedule.purchase_fee_rate
            invested = spend - fee
            if invested > 0:
                lots.append(FundLot(invested / price, date))
                cash -= spend
                total_purchase_fees += fee
                trade_count += 1
        elif trade_value < -1e-12 and lots:
            gross_redeem = min(-trade_value, current_fund_value)
            units_to_sell = gross_redeem / price
            proceeds = 0.0
            redemption_fee = 0.0
            remaining_lots: list[FundLot] = []
            for lot in lots:
                if units_to_sell <= 1e-12:
                    remaining_lots.append(lot)
                    continue
                sold = min(lot.units, units_to_sell)
                holding_days = int((date - lot.purchase_date).days)
                gross = sold * price
                fee = gross * redemption_fee_rate(holding_days, schedule)
                proceeds += gross - fee
                redemption_fee += fee
                redeemed_value_days += gross * holding_days
                redeemed_value_total += gross
                if holding_days < 7:
                    redeemed_under_7 += gross
                if holding_days < 30:
                    redeemed_under_30 += gross
                units_to_sell -= sold
                leftover = lot.units - sold
                if leftover > 1e-12:
                    remaining_lots.append(FundLot(leftover, lot.purchase_date))
            lots = remaining_lots
            cash += proceeds
            total_redemption_fees += redemption_fee
            trade_count += 1

        if liquidate_at_end and row_number == len(nav) - 1 and lots:
            proceeds = 0.0
            redemption_fee = 0.0
            for lot in lots:
                holding_days = int((date - lot.purchase_date).days)
                gross = lot.units * price
                fee = gross * redemption_fee_rate(holding_days, schedule)
                proceeds += gross - fee
                redemption_fee += fee
                redeemed_value_days += gross * holding_days
                redeemed_value_total += gross
                if holding_days < 7:
                    redeemed_under_7 += gross
                if holding_days < 30:
                    redeemed_under_30 += gross
            lots = []
            cash += proceeds
            total_redemption_fees += redemption_fee
            if liquidate_final_trade_count:
                trade_count += 1

        units = sum(lot.units for lot in lots)
        portfolio_value = cash + units * price
        records.append(
            {
                "date": date,
                "portfolio_value": portfolio_value,
                "cash": cash,
                "fund_value": units * price,
                "target_position": executable_target.loc[date],
                "actual_position": units * price / portfolio_value
                if portfolio_value > 0
                else 0.0,
            }
        )

    frame = pd.DataFrame(records).set_index("date")
    returns = frame["portfolio_value"].pct_change(fill_method=None).fillna(0.0)
    metrics = performance_metrics(returns)
    metrics.update(
        {
            "trade_count": float(trade_count),
            "purchase_fees": float(total_purchase_fees),
            "redemption_fees": float(total_redemption_fees),
            "total_fees": float(total_purchase_fees + total_redemption_fees),
            "invested_day_ratio": float((frame["actual_position"] > 0.01).mean()),
            "average_holding_days": float(
                redeemed_value_days / redeemed_value_total
                if redeemed_value_total > 0
                else 0.0
            ),
            "under_7_day_redemption_ratio": float(
                redeemed_under_7 / redeemed_value_total
                if redeemed_value_total > 0
                else 0.0
            ),
            "under_30_day_redemption_ratio": float(
                redeemed_under_30 / redeemed_value_total
                if redeemed_value_total > 0
                else 0.0
            ),
        }
    )
    return {"ledger": frame, "returns": returns, "metrics": metrics}
