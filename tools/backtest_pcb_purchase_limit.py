from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
DEPS = ROOT / ".deps"
if DEPS.exists() and str(DEPS) not in sys.path:
    sys.path.insert(0, str(DEPS))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager

from tools.sector_level_rotation_5themes import (
    EXECUTION_DELAY_DAYS,
    build_strategy,
    c_class_redemption_fee_rate,
    dataframe_to_markdown,
    load_sector_navs,
    max_drawdown,
)


OUT = ROOT / "output" / "pcb_purchase_limit_backtest"
OUT.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 10000.0
PCB_DAILY_LIMIT = 4000.0
ONE_MONTH_DAYS = 21
THREE_MONTH_DAYS = 63
WINDOW_COUNT = 30


def setup_chinese_font() -> None:
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC", "Source Han Sans SC"]
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name]
            break
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Lot:
    units: float
    purchase_date: pd.Timestamp


def delayed_choice(choice: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    aligned = choice.reindex(index).ffill()
    return aligned.shift(EXECUTION_DELAY_DAYS).fillna(aligned.iloc[0])


def simulate(
    navs: pd.DataFrame,
    choice: pd.Series,
    pcb_daily_limit: float | None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Simulate one-sector rotation with FIFO fees and optional PCB buy limit."""
    navs = navs.dropna().sort_index()
    executed = delayed_choice(choice, navs.index)
    lots: dict[str, list[Lot]] = {sector: [] for sector in navs.columns}
    cash = INITIAL_CAPITAL
    current = str(executed.iloc[0])
    total_fees = 0.0
    redeemed_gross = 0.0
    redeemed_days = 0.0
    switches = 0
    pcb_limited_days = 0
    records = []

    def units(sector: str) -> float:
        return sum(lot.units for lot in lots[sector])

    def sell_all(sector: str, date: pd.Timestamp) -> None:
        nonlocal cash, total_fees, redeemed_gross, redeemed_days
        price = float(navs.loc[date, sector])
        proceeds = 0.0
        for lot in lots[sector]:
            holding_days = int((date - lot.purchase_date).days)
            gross = lot.units * price
            fee = gross * c_class_redemption_fee_rate(holding_days)
            proceeds += gross - fee
            total_fees += fee
            redeemed_gross += gross
            redeemed_days += gross * holding_days
        lots[sector] = []
        cash += proceeds

    def buy_target(sector: str, date: pd.Timestamp) -> float:
        nonlocal cash
        if cash <= 1e-9:
            return 0.0
        spend = cash
        if sector == "PCB" and pcb_daily_limit is not None:
            spend = min(spend, pcb_daily_limit)
        price = float(navs.loc[date, sector])
        lots[sector].append(Lot(spend / price, date))
        cash -= spend
        return spend

    for date in navs.index:
        target = str(executed.loc[date])
        if target != current:
            sell_all(current, date)
            current = target
            switches += 1

        bought = buy_target(current, date)
        if current == "PCB" and pcb_daily_limit is not None and cash > 1e-9:
            pcb_limited_days += 1

        asset_values = {
            sector: units(sector) * float(navs.loc[date, sector])
            for sector in navs.columns
        }
        portfolio_value = cash + sum(asset_values.values())
        invested_value = sum(asset_values.values())
        records.append(
            {
                "日期": date,
                "组合市值": portfolio_value,
                "现金": cash,
                "投资市值": invested_value,
                "现金比例": cash / portfolio_value if portfolio_value else 0.0,
                "执行持仓": current,
                "当日买入金额": bought,
                "是否受PCB限购约束": bool(current == "PCB" and pcb_daily_limit is not None and cash > 1e-9),
                "累计赎回费": total_fees,
            }
        )

    ledger = pd.DataFrame(records).set_index("日期")
    curve = ledger["组合市值"] / INITIAL_CAPITAL
    metrics = {
        "总收益率": float(curve.iloc[-1] - 1),
        "最大回撤": max_drawdown(curve),
        "平均现金比例": float(ledger["现金比例"].mean()),
        "最高现金比例": float(ledger["现金比例"].max()),
        "切换次数": float(switches),
        "赎回费": float(total_fees / INITIAL_CAPITAL),
        "平均持有天数": float(redeemed_days / redeemed_gross) if redeemed_gross else np.nan,
        "PCB受限天数": float(pcb_limited_days),
    }
    return ledger, metrics


def simulate_gradual_to_pcb(
    navs: pd.DataFrame,
    choice: pd.Series,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Transition into PCB by selling old holdings gradually instead of holding cash."""
    navs = navs.dropna().sort_index()
    executed = delayed_choice(choice, navs.index)
    lots: dict[str, list[Lot]] = {sector: [] for sector in navs.columns}
    cash = INITIAL_CAPITAL
    target_sector = str(executed.iloc[0])
    total_fees = 0.0
    redeemed_gross = 0.0
    redeemed_days = 0.0
    switches = 0
    pcb_limited_days = 0
    records = []

    def units(sector: str) -> float:
        return sum(lot.units for lot in lots[sector])

    def sector_value(sector: str, date: pd.Timestamp) -> float:
        return units(sector) * float(navs.loc[date, sector])

    def sell_net_amount(sector: str, desired_net: float, date: pd.Timestamp) -> float:
        nonlocal cash, total_fees, redeemed_gross, redeemed_days
        if desired_net <= 1e-9 or not lots[sector]:
            return 0.0
        price = float(navs.loc[date, sector])
        net_raised = 0.0
        remaining_lots: list[Lot] = []
        for lot in lots[sector]:
            if net_raised >= desired_net - 1e-9:
                remaining_lots.append(lot)
                continue
            holding_days = int((date - lot.purchase_date).days)
            fee_rate = c_class_redemption_fee_rate(holding_days)
            net_per_unit = price * (1.0 - fee_rate)
            units_needed = (desired_net - net_raised) / net_per_unit
            sold_units = min(lot.units, units_needed)
            gross = sold_units * price
            fee = gross * fee_rate
            net = gross - fee
            net_raised += net
            total_fees += fee
            redeemed_gross += gross
            redeemed_days += gross * holding_days
            leftover = lot.units - sold_units
            if leftover > 1e-12:
                remaining_lots.append(Lot(leftover, lot.purchase_date))
        lots[sector] = remaining_lots
        cash += net_raised
        return net_raised

    def sell_all(sector: str, date: pd.Timestamp) -> None:
        sell_net_amount(sector, float("inf"), date)

    def buy(sector: str, amount: float, date: pd.Timestamp) -> float:
        nonlocal cash
        spend = min(max(amount, 0.0), cash)
        if spend <= 1e-9:
            return 0.0
        lots[sector].append(Lot(spend / float(navs.loc[date, sector]), date))
        cash -= spend
        return spend

    for date in navs.index:
        new_target = str(executed.loc[date])
        if new_target != target_sector:
            target_sector = new_target
            switches += 1

        bought = 0.0
        if target_sector != "PCB":
            for sector in navs.columns:
                if sector != target_sector and lots[sector]:
                    sell_all(sector, date)
            bought = buy(target_sector, cash, date)
        else:
            budget = PCB_DAILY_LIMIT
            if cash < budget - 1e-9:
                need = budget - cash
                for sector in navs.columns:
                    if sector == "PCB" or not lots[sector]:
                        continue
                    raised = sell_net_amount(sector, need, date)
                    need -= raised
                    if need <= 1e-9:
                        break
            bought = buy("PCB", budget, date)
            non_pcb_value = sum(
                sector_value(sector, date)
                for sector in navs.columns
                if sector != "PCB"
            )
            if cash > 1e-9 or non_pcb_value > 1e-9:
                pcb_limited_days += 1

        asset_values = {
            sector: sector_value(sector, date)
            for sector in navs.columns
        }
        portfolio_value = cash + sum(asset_values.values())
        records.append(
            {
                "日期": date,
                "组合市值": portfolio_value,
                "现金": cash,
                "投资市值": sum(asset_values.values()),
                "现金比例": cash / portfolio_value if portfolio_value else 0.0,
                "执行持仓": target_sector,
                "当日买入金额": bought,
                "是否受PCB限购约束": bool(target_sector == "PCB" and pcb_limited_days > 0),
                "累计赎回费": total_fees,
            }
        )

    ledger = pd.DataFrame(records).set_index("日期")
    curve = ledger["组合市值"] / INITIAL_CAPITAL
    metrics = {
        "总收益率": float(curve.iloc[-1] - 1),
        "最大回撤": max_drawdown(curve),
        "平均现金比例": float(ledger["现金比例"].mean()),
        "最高现金比例": float(ledger["现金比例"].max()),
        "切换次数": float(switches),
        "赎回费": float(total_fees / INITIAL_CAPITAL),
        "平均持有天数": float(redeemed_days / redeemed_gross) if redeemed_gross else np.nan,
        "PCB受限天数": float(pcb_limited_days),
    }
    return ledger, metrics


def simulate_weighted_targets(
    navs: pd.DataFrame,
    target_weights: pd.DataFrame,
    execution_delay_days: int | None = None,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Fee-aware weighted portfolio with realistic conversion and cash settlement.

    Fund-to-fund conversions can finance purchases on the execution date.  A
    redemption whose destination is cash settles after the next NAV date and is
    available for purchases on the second NAV date after the sale (T+2).
    """
    navs = navs.dropna().sort_index()
    targets = target_weights.reindex(index=navs.index, columns=navs.columns).ffill().fillna(0.0)
    delay = EXECUTION_DELAY_DAYS if execution_delay_days is None else execution_delay_days
    targets = targets.shift(delay).fillna(targets.iloc[0])
    lots: dict[str, list[Lot]] = {sector: [] for sector in navs.columns}
    cash = INITIAL_CAPITAL
    pending_cash: list[tuple[int, float]] = []
    total_fees = 0.0
    redeemed_gross = 0.0
    redeemed_days = 0.0
    rebalance_count = 0
    pcb_limited_days = 0
    records = []
    previous_target: pd.Series | None = None
    pcb_fill_pending = False
    allocation_fill_pending = False

    def units(sector: str) -> float:
        return sum(lot.units for lot in lots[sector])

    def value(sector: str, date: pd.Timestamp) -> float:
        return units(sector) * float(navs.loc[date, sector])

    def sell_for_net(
        sector: str,
        desired_net: float,
        max_gross: float,
        date: pd.Timestamp,
    ) -> float:
        nonlocal total_fees, redeemed_gross, redeemed_days
        if desired_net <= 1e-9 or max_gross <= 1e-9:
            return 0.0
        price = float(navs.loc[date, sector])
        net_raised = 0.0
        gross_sold = 0.0
        remaining: list[Lot] = []
        for lot in lots[sector]:
            if net_raised >= desired_net - 1e-9 or gross_sold >= max_gross - 1e-9:
                remaining.append(lot)
                continue
            holding_days = int((date - lot.purchase_date).days)
            fee_rate = c_class_redemption_fee_rate(holding_days)
            max_units_by_gross = (max_gross - gross_sold) / price
            units_by_net = (desired_net - net_raised) / (price * (1.0 - fee_rate))
            sold = min(lot.units, max_units_by_gross, units_by_net)
            gross = sold * price
            fee = gross * fee_rate
            net = gross - fee
            gross_sold += gross
            net_raised += net
            total_fees += fee
            redeemed_gross += gross
            redeemed_days += gross * holding_days
            leftover = lot.units - sold
            if leftover > 1e-12:
                remaining.append(Lot(leftover, lot.purchase_date))
        lots[sector] = remaining
        return net_raised

    for day_number, date in enumerate(navs.index):
        matured = sum(amount for unlock_day, amount in pending_cash if unlock_day <= day_number)
        if matured:
            cash += matured
        pending_cash = [
            (unlock_day, amount)
            for unlock_day, amount in pending_cash
            if unlock_day > day_number
        ]
        unsettled = sum(amount for _, amount in pending_cash)
        before_values = pd.Series({sector: value(sector, date) for sector in navs.columns})
        total_value = cash + unsettled + float(before_values.sum())
        desired = targets.loc[date].clip(lower=0.0)
        if desired.sum() > 1.0:
            desired = desired / desired.sum()
        target_changed = previous_target is None or not np.allclose(
            desired.to_numpy(dtype=float),
            previous_target.to_numpy(dtype=float),
            atol=1e-10,
        )
        if not target_changed and not pcb_fill_pending and not allocation_fill_pending:
            portfolio_value = total_value
            record = {
                "日期": date,
                "组合市值": portfolio_value,
                "现金": cash,
                "待到账赎回款": unsettled,
                "现金比例": cash / portfolio_value if portfolio_value else 0.0,
                "待到账比例": unsettled / portfolio_value if portfolio_value else 0.0,
                "当日买入金额": 0.0,
                "当日转为现金金额": 0.0,
                "累计赎回费": total_fees,
            }
            for sector in navs.columns:
                record[f"权重_{sector}"] = before_values[sector] / portfolio_value if portfolio_value else 0.0
            records.append(record)
            continue
        desired_values = desired * total_value
        desired_cash = max(0.0, total_value - float(desired_values.sum()))
        deficits = (desired_values - before_values).clip(lower=0.0)
        excess = (before_values - desired_values).clip(lower=0.0)

        pcb_deficit = float(deficits.get("PCB", 0.0))
        pcb_buy = min(pcb_deficit, PCB_DAILY_LIMIT)
        if pcb_deficit > PCB_DAILY_LIMIT + 1e-9:
            pcb_limited_days += 1
        planned_buys = deficits.copy()
        planned_buys["PCB"] = pcb_buy
        buy_need = float(planned_buys.sum())

        # Existing cash above the strategic cash target can fund purchases.
        spendable_cash = max(0.0, cash - max(0.0, desired_cash - unsettled))
        cash_needed = max(0.0, buy_need - spendable_cash)
        if cash_needed > 1e-9:
            for sector in excess.sort_values(ascending=False).index:
                if cash_needed <= 1e-9:
                    break
                raised = sell_for_net(
                    sector,
                    cash_needed,
                    float(excess[sector]),
                    date,
                )
                cash += raised  # direct fund conversion: usable on the same date
                cash_needed -= raised

        bought_total = 0.0
        for sector in planned_buys.sort_values(ascending=False).index:
            cash_reserve = max(0.0, desired_cash - unsettled)
            spend = min(float(planned_buys[sector]), max(0.0, cash - cash_reserve))
            if spend <= 1e-9:
                continue
            lots[sector].append(Lot(spend / float(navs.loc[date, sector]), date))
            cash -= spend
            bought_total += spend

        # Only an explicit cash allocation uses the Alipay redemption path.
        # The receivable remains part of wealth but cannot fund a purchase until T+2.
        cash_assets = cash + unsettled
        cash_sale = 0.0
        cash_shortfall = max(0.0, desired_cash - cash_assets)
        if cash_shortfall > 1e-9:
            current_values = pd.Series({sector: value(sector, date) for sector in navs.columns})
            current_excess = (current_values - desired_values).clip(lower=0.0)
            for sector in current_excess.sort_values(ascending=False).index:
                if cash_shortfall <= 1e-9:
                    break
                raised = sell_for_net(
                    sector,
                    cash_shortfall,
                    float(current_excess[sector]),
                    date,
                )
                if raised > 0:
                    pending_cash.append((day_number + 2, raised))
                    unsettled += raised
                    cash_sale += raised
                    cash_shortfall -= raised
        if bought_total > 1e-9 or float(excess.sum()) > 1e-9:
            rebalance_count += 1

        after_values = pd.Series({sector: value(sector, date) for sector in navs.columns})
        portfolio_value = cash + unsettled + float(after_values.sum())
        pcb_fill_pending = bool(
            desired_values.get("PCB", 0.0) - after_values.get("PCB", 0.0) > 1.0
        )
        remaining_deficit = (desired_values - after_values).clip(lower=0.0)
        allocation_fill_pending = bool(
            float(remaining_deficit.sum()) > 1.0 and (unsettled > 1.0 or pcb_fill_pending)
        )
        previous_target = desired.copy()
        record = {
            "日期": date,
            "组合市值": portfolio_value,
            "现金": cash,
            "待到账赎回款": unsettled,
            "现金比例": cash / portfolio_value if portfolio_value else 0.0,
            "待到账比例": unsettled / portfolio_value if portfolio_value else 0.0,
            "当日买入金额": bought_total,
            "当日转为现金金额": cash_sale,
            "累计赎回费": total_fees,
        }
        for sector in navs.columns:
            record[f"权重_{sector}"] = after_values[sector] / portfolio_value if portfolio_value else 0.0
        records.append(record)

    ledger = pd.DataFrame(records).set_index("日期")
    curve = ledger["组合市值"] / INITIAL_CAPITAL
    metrics = {
        "总收益率": float(curve.iloc[-1] - 1),
        "最大回撤": max_drawdown(curve),
        "平均现金比例": float(ledger["现金比例"].mean()),
        "最高现金比例": float(ledger["现金比例"].max()),
        "平均待到账比例": float(ledger["待到账比例"].mean()),
        "最高待到账比例": float(ledger["待到账比例"].max()),
        "切换次数": float(rebalance_count),
        "赎回费": float(total_fees / INITIAL_CAPITAL),
        "平均持有天数": float(redeemed_days / redeemed_gross) if redeemed_gross else np.nan,
        "PCB受限天数": float(pcb_limited_days),
    }
    return ledger, metrics


def all_in_pcb_choice(index: pd.DatetimeIndex) -> pd.Series:
    return pd.Series("PCB", index=index)


def evaluate_window(
    navs: pd.DataFrame,
    choice: pd.Series,
    start_pos: int,
    end_pos: int,
) -> dict[str, float | str | bool]:
    test = navs.iloc[start_pos : end_pos + 1]
    test_choice = choice.reindex(test.index).ffill()
    constrained, constrained_metrics = simulate_gradual_to_pcb(test, test_choice)
    unlimited, unlimited_metrics = simulate(test, test_choice, None)
    pcb_limited, pcb_limited_metrics = simulate(test, all_in_pcb_choice(test.index), PCB_DAILY_LIMIT)
    pcb_unlimited_curve = test["PCB"] / test["PCB"].iloc[0]
    pcb_unlimited_return = float(pcb_unlimited_curve.iloc[-1] - 1)

    return {
        "开始日期": test.index[0].date().isoformat(),
        "结束日期": test.index[-1].date().isoformat(),
        "限购轮动收益": constrained_metrics["总收益率"],
        "限购轮动最大回撤": constrained_metrics["最大回撤"],
        "不限购轮动收益": unlimited_metrics["总收益率"],
        "不限购轮动最大回撤": unlimited_metrics["最大回撤"],
        "限购全仓PCB收益": pcb_limited_metrics["总收益率"],
        "限购全仓PCB最大回撤": pcb_limited_metrics["最大回撤"],
        "理论一次性全仓PCB收益": pcb_unlimited_return,
        "理论一次性全仓PCB最大回撤": max_drawdown(pcb_unlimited_curve),
        "限购轮动平均现金比例": constrained_metrics["平均现金比例"],
        "限购轮动最高现金比例": constrained_metrics["最高现金比例"],
        "限购轮动PCB受限天数": constrained_metrics["PCB受限天数"],
        "限购轮动赎回费": constrained_metrics["赎回费"],
        "限购轮动跑赢限购全仓PCB": constrained_metrics["总收益率"] > pcb_limited_metrics["总收益率"],
        "限购轮动跑赢理论一次性PCB": constrained_metrics["总收益率"] > pcb_unlimited_return,
    }


def evaluate_sliding_windows(
    navs: pd.DataFrame,
    choice: pd.Series,
    window_days: int,
) -> pd.DataFrame:
    rows = []
    for window_id, end_pos in enumerate(range(len(navs) - WINDOW_COUNT, len(navs)), start=1):
        start_pos = end_pos - window_days + 1
        row = evaluate_window(navs, choice, start_pos, end_pos)
        row["窗口"] = window_id
        row["窗口交易日数"] = window_days
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_entry_cases(navs: pd.DataFrame, choice: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    ledgers = []
    for label, months in [("一个月前", 1), ("三个月前", 3), ("半年前", 6)]:
        target = navs.index[-1] - pd.DateOffset(months=months)
        start = navs.index[navs.index >= target][0]
        test = navs.loc[navs.index >= start]
        test_choice = choice.reindex(test.index).ffill()
        for strategy_name, strategy_choice, limit in [
            ("限购直接转换轮动", test_choice, "direct_conversion"),
            ("不限购五板块轮动", test_choice, None),
            ("限购全仓PCB", all_in_pcb_choice(test.index), PCB_DAILY_LIMIT),
            ("理论一次性全仓PCB", all_in_pcb_choice(test.index), None),
        ]:
            if limit == "direct_conversion":
                ledger, metrics = simulate_gradual_to_pcb(test, strategy_choice)
            else:
                ledger, metrics = simulate(test, strategy_choice, limit)
            rows.append(
                {
                    "进场口径": label,
                    "开始日期": start.date().isoformat(),
                    "结束日期": test.index[-1].date().isoformat(),
                    "策略": strategy_name,
                    **metrics,
                }
            )
            temp = ledger.reset_index()
            temp.insert(0, "进场口径", label)
            temp.insert(1, "策略", strategy_name)
            ledgers.append(temp)
    return pd.DataFrame(rows), pd.concat(ledgers, ignore_index=True)


def transaction_table(ledger: pd.DataFrame) -> pd.DataFrame:
    rows = []
    previous = None
    for date, row in ledger.iterrows():
        current = str(row["执行持仓"])
        bought = float(row["当日买入金额"])
        if previous is None:
            action = f"初始买入{current}"
        elif current != previous:
            action = f"卖出{previous}，切换到{current}"
        elif current == "PCB" and bought > 1e-9:
            action = "PCB限购补仓"
        elif bought > 1e-9:
            action = f"买入{current}"
        else:
            previous = current
            continue
        rows.append(
            {
                "日期": date,
                "动作": action,
                "执行板块": current,
                "买入金额": bought,
                "现金余额": float(row["现金"]),
                "现金比例": float(row["现金比例"]),
                "组合市值": float(row["组合市值"]),
            }
        )
        previous = current
    return pd.DataFrame(rows)


def plot_fixed_entry_cases(
    navs: pd.DataFrame,
    choice: pd.Series,
) -> tuple[pd.DataFrame, list[Path]]:
    setup_chinese_font()
    all_transactions = []
    image_paths = []
    for label, months in [("一个月前", 1), ("三个月前", 3), ("半年前", 6)]:
        target = navs.index[-1] - pd.DateOffset(months=months)
        start = navs.index[navs.index >= target][0]
        test = navs.loc[navs.index >= start]
        test_choice = choice.reindex(test.index).ffill()

        constrained, constrained_metrics = simulate_gradual_to_pcb(test, test_choice)
        unlimited, unlimited_metrics = simulate(test, test_choice, None)
        pcb_limited, pcb_limited_metrics = simulate(test, all_in_pcb_choice(test.index), PCB_DAILY_LIMIT)
        pcb_unlimited, pcb_unlimited_metrics = simulate(test, all_in_pcb_choice(test.index), None)

        fig, (ax, cash_ax) = plt.subplots(
            2,
            1,
            figsize=(14, 9),
            dpi=160,
            sharex=True,
            gridspec_kw={"height_ratios": [4, 1]},
        )
        for ledger, name, style in [
            (constrained, "限购直接转换轮动", "-"),
            (unlimited, "不限购五板块轮动", "-"),
            (pcb_limited, "限购全仓PCB", "--"),
            (pcb_unlimited, "理论一次性全仓PCB", ":"),
        ]:
            ax.plot(ledger.index, ledger["组合市值"] / INITIAL_CAPITAL, label=name, linestyle=style, linewidth=2.0)

        trades = transaction_table(constrained)
        for row_number, trade in trades.iterrows():
            date = pd.Timestamp(trade["日期"])
            y = float(trade["组合市值"]) / INITIAL_CAPITAL
            action = str(trade["动作"])
            if action == "PCB限购补仓":
                ax.scatter([date], [y], marker="^", s=32, color="#2ca02c", zorder=5)
            else:
                ax.scatter([date], [y], s=42, color="#d62728", zorder=5)
                ax.annotate(
                    action,
                    xy=(date, y),
                    xytext=(0, 18 if row_number % 2 == 0 else -25),
                    textcoords="offset points",
                    ha="center",
                    fontsize=8,
                    arrowprops={"arrowstyle": "->", "lw": 0.8},
                )

        cash_ax.fill_between(
            constrained.index,
            0,
            constrained["现金比例"],
            color="#7f7f7f",
            alpha=0.35,
            label="限购轮动现金比例",
        )
        cash_ax.set_ylabel("现金比例")
        cash_ax.set_ylim(0, max(0.65, float(constrained["现金比例"].max()) * 1.1))
        cash_ax.grid(alpha=0.2)
        cash_ax.legend(loc="upper left")

        ax.set_title(
            f"{label}进场，初始资金1万元，PCB每日最多买4000元\n"
            f"限购直接转换 {constrained_metrics['总收益率']:.2%}/{constrained_metrics['最大回撤']:.2%}；"
            f"不限购轮动 {unlimited_metrics['总收益率']:.2%}/{unlimited_metrics['最大回撤']:.2%}；"
            f"限购PCB {pcb_limited_metrics['总收益率']:.2%}/{pcb_limited_metrics['最大回撤']:.2%}"
        )
        ax.set_ylabel("相对净值（初始=1）")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.autofmt_xdate()
        fig.tight_layout()
        image_path = OUT / f"{label}进场_PCB每日4000限购_收益曲线与交易点.png"
        fig.savefig(image_path)
        plt.close(fig)

        trades.insert(0, "进场口径", label)
        all_transactions.append(trades)
        image_paths.append(image_path)
    return pd.concat(all_transactions, ignore_index=True), image_paths


def summarize_windows(frame: pd.DataFrame, label: str) -> dict[str, float | str]:
    return {
        "窗口口径": label,
        "窗口数量": len(frame),
        "限购轮动跑赢限购全仓PCB次数": int(frame["限购轮动跑赢限购全仓PCB"].sum()),
        "限购轮动跑赢理论一次性PCB次数": int(frame["限购轮动跑赢理论一次性PCB"].sum()),
        "限购轮动平均收益": float(frame["限购轮动收益"].mean()),
        "不限购轮动平均收益": float(frame["不限购轮动收益"].mean()),
        "限购全仓PCB平均收益": float(frame["限购全仓PCB收益"].mean()),
        "理论一次性PCB平均收益": float(frame["理论一次性全仓PCB收益"].mean()),
        "限购轮动平均最大回撤": float(frame["限购轮动最大回撤"].mean()),
        "限购全仓PCB平均最大回撤": float(frame["限购全仓PCB最大回撤"].mean()),
        "限购轮动平均现金比例": float(frame["限购轮动平均现金比例"].mean()),
        "限购轮动平均赎回费": float(frame["限购轮动赎回费"].mean()),
    }


def main() -> None:
    navs = load_sector_navs()
    choice, _, _, _ = build_strategy(navs, use_c_redemption_fee=True, protect_redemption_fee=True)

    one_month = evaluate_sliding_windows(navs, choice, ONE_MONTH_DAYS)
    three_month = evaluate_sliding_windows(navs, choice, THREE_MONTH_DAYS)
    summary = pd.DataFrame(
        [
            summarize_windows(one_month, "30个一个月窗口"),
            summarize_windows(three_month, "30个三个月窗口"),
        ]
    )
    fixed_metrics, fixed_ledgers = fixed_entry_cases(navs, choice)
    transactions, image_paths = plot_fixed_entry_cases(navs, choice)

    one_month.to_csv(OUT / "30个一个月窗口_cn.csv", index=False, encoding="utf-8-sig")
    three_month.to_csv(OUT / "30个三个月窗口_cn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "窗口汇总_cn.csv", index=False, encoding="utf-8-sig")
    fixed_metrics.to_csv(OUT / "一三六月进场对比_cn.csv", index=False, encoding="utf-8-sig")
    fixed_ledgers.to_csv(OUT / "一三六月每日资金明细_cn.csv", index=False, encoding="utf-8-sig")
    transactions.to_csv(OUT / "一三六月交易与PCB补仓点_cn.csv", index=False, encoding="utf-8-sig")

    lines = [
        "# PCB每日4000元限购下的五板块轮动回测",
        "",
        "## 假设",
        "",
        "- 初始资金：10000元。",
        "- PCB执行层只有4个申购通道，每只每日最多1000元，合计每日最多买入4000元。",
        "- 720001仅作为长历史信号代理，不额外增加申购容量。",
        "- 板块之间采用直接转换：执行日当日卖出旧板块，并在同一日买入新板块，不设置资金到账等待。",
        "- 若新板块为PCB，每日最多把4000元旧板块直接转换到PCB；未转换部分继续持有旧板块，不提前变成现金。",
        "- 初始资金本来就是现金时，进入PCB仍需约3个净值日，每日最多申购4000元。",
        "- 只有策略主动变回现金时才使用支付宝结算规则：T日15点前卖出，T+1日15点后到账，T+2才能再次申购。当前策略没有主动空仓信号，因此本轮回测未触发该等待。",
        "- 其他板块暂按不限购处理。",
        "- 赎回继续采用C类FIFO费率。",
        "",
        "## 滑动窗口汇总",
        "",
        dataframe_to_markdown(summary),
        "",
        "## 一个月、三个月、半年前进场",
        "",
        dataframe_to_markdown(fixed_metrics),
    ]
    (OUT / "PCB限购回测报告_cn.md").write_text("\n".join(lines), encoding="utf-8")

    print(summary.to_string(index=False))
    print(fixed_metrics.to_string(index=False))
    for path in image_paths:
        print(path)


if __name__ == "__main__":
    main()
