"""FIFO, fee-aware backtesting for a portfolio of open-ended funds."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd

from .backtest import performance_metrics
from .fund_backtest import (
    FundFeeSchedule,
    FundLot,
    redemption_fee_rate,
)


def backtest_open_fund_portfolio(
    navs: pd.DataFrame,
    target_weights: pd.DataFrame,
    schedule: FundFeeSchedule | None = None,
    initial_capital: float = 1.0,
    initial_weights: pd.Series | None = None,
    liquidate_at_end: bool = True,
    minimum_rebalance_fraction: float = 0.03,
) -> dict[str, object]:
    """Execute prior-day weights with one shared cash balance and FIFO lots."""
    schedule = schedule or FundFeeSchedule()
    navs = navs.dropna().sort_index()
    targets = (
        target_weights.reindex(index=navs.index, columns=navs.columns)
        .ffill()
        .fillna(0.0)
        .clip(0, 1)
    )
    row_sums = targets.sum(axis=1)
    targets = targets.div(row_sums.where(row_sums > 1, 1.0), axis=0)
    executable = targets.shift(1).fillna(0.0)
    target_changed = targets.ne(targets.shift(1)).any(axis=1)
    if len(target_changed) and initial_weights is not None:
        target_changed.iloc[0] = False
    executable_change = target_changed.shift(1).fillna(False).astype(bool)
    if len(executable) and initial_weights is not None:
        initial = (
            initial_weights.reindex(navs.columns)
            .fillna(0.0)
            .clip(0, 1)
        )
        if initial.sum() > 1:
            initial = initial / initial.sum()
        executable.iloc[0] = initial
        executable_change.iloc[0] = True

    cash = float(initial_capital)
    lots: dict[str, list[FundLot]] = defaultdict(list)
    records: list[dict[str, float | pd.Timestamp]] = []
    purchase_fees = 0.0
    redemption_fees = 0.0
    trade_count = 0
    redeemed_value_days = 0.0
    redeemed_value = 0.0
    redeemed_under_30 = 0.0

    def units(asset: str) -> float:
        return sum(lot.units for lot in lots[asset])

    def sell(asset: str, gross_redeem: float, date: pd.Timestamp, price: float) -> None:
        nonlocal cash, redemption_fees, trade_count
        nonlocal redeemed_value_days, redeemed_value, redeemed_under_30
        if gross_redeem <= 1e-12:
            return
        units_to_sell = min(gross_redeem / price, units(asset))
        proceeds = 0.0
        fees = 0.0
        remaining: list[FundLot] = []
        for lot in lots[asset]:
            if units_to_sell <= 1e-12:
                remaining.append(lot)
                continue
            sold = min(lot.units, units_to_sell)
            holding_days = int((date - lot.purchase_date).days)
            gross = sold * price
            fee = gross * redemption_fee_rate(holding_days, schedule)
            proceeds += gross - fee
            fees += fee
            redeemed_value_days += gross * holding_days
            redeemed_value += gross
            if holding_days < 30:
                redeemed_under_30 += gross
            units_to_sell -= sold
            leftover = lot.units - sold
            if leftover > 1e-12:
                remaining.append(FundLot(leftover, lot.purchase_date))
        lots[asset] = remaining
        cash += proceeds
        redemption_fees += fees
        trade_count += 1

    for row_number, date in enumerate(navs.index):
        prices = navs.loc[date]
        values = pd.Series(
            {asset: units(asset) * float(prices[asset]) for asset in navs.columns}
        )
        gross_value = cash + float(values.sum())
        desired = executable.loc[date] * gross_value
        trade_allowed = bool(executable_change.loc[date])

        # Redemptions settle first so their proceeds can finance subscriptions.
        if trade_allowed:
            for asset in navs.columns:
                reduction = float(values[asset] - desired[asset])
                if (
                    reduction > 1e-12
                    and reduction / gross_value >= minimum_rebalance_fraction
                ):
                    sell(asset, reduction, date, float(prices[asset]))

        values = pd.Series(
            {asset: units(asset) * float(prices[asset]) for asset in navs.columns}
        )
        if trade_allowed:
            for asset in navs.columns:
                increase = float(desired[asset] - values[asset])
                if (
                    increase > 1e-12
                    and increase / gross_value >= minimum_rebalance_fraction
                    and cash > 1e-12
                ):
                    spend = min(increase, cash)
                    fee = spend * schedule.purchase_fee_rate
                    invested = spend - fee
                    if invested > 0:
                        lots[asset].append(
                            FundLot(invested / float(prices[asset]), date)
                        )
                        cash -= spend
                        purchase_fees += fee
                        trade_count += 1

        if liquidate_at_end and row_number == len(navs) - 1:
            for asset in navs.columns:
                sell(
                    asset,
                    units(asset) * float(prices[asset]),
                    date,
                    float(prices[asset]),
                )

        values = pd.Series(
            {asset: units(asset) * float(prices[asset]) for asset in navs.columns}
        )
        portfolio_value = cash + float(values.sum())
        record: dict[str, float | pd.Timestamp] = {
            "date": date,
            "portfolio_value": portfolio_value,
            "cash": cash,
        }
        for asset in navs.columns:
            record[f"weight_{asset}"] = (
                float(values[asset] / portfolio_value)
                if portfolio_value > 0
                else 0.0
            )
        records.append(record)

    ledger = pd.DataFrame(records).set_index("date")
    returns = ledger["portfolio_value"].pct_change(fill_method=None).fillna(0.0)
    metrics = performance_metrics(returns)
    metrics.update(
        {
            "trade_count": float(trade_count),
            "purchase_fees": float(purchase_fees),
            "redemption_fees": float(redemption_fees),
            "total_fees": float(purchase_fees + redemption_fees),
            "average_holding_days": float(
                redeemed_value_days / redeemed_value if redeemed_value else 0.0
            ),
            "under_30_day_redemption_ratio": float(
                redeemed_under_30 / redeemed_value if redeemed_value else 0.0
            ),
        }
    )
    return {"ledger": ledger, "returns": returns, "metrics": metrics}
