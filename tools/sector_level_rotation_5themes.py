from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from quant_fund_advisor.data import load_or_fetch_open_fund_nav


OUT = ROOT / "output" / "sector_level_rotation_5themes"
OUT.mkdir(parents=True, exist_ok=True)

WINDOW_DAYS = 63
SLIDING_WINDOWS = 30
INITIAL_CAPITAL = 10000.0
EXECUTION_DELAY_DAYS = 2
MIN_HOLD_DAYS_FOR_FEE_PROTECTION = 7
MOMENTUM_ADVANTAGE_TO_BREAK_HOLD = 0.40
CURRENT_SECTOR_5D_STOP_LOSS = -0.08

# C类常见赎回费率：持有<7天 1.5%；7-30天 0.5%；>=30天 0%。
C_CLASS_FEE_SOURCE = "通用C类赎回费率；AKShare fund_fee_em 对当前样本基金未返回可用赎回费率。"


@dataclass
class FundLot:
    units: float
    purchase_date: pd.Timestamp


SECTORS = {
    "PCB": {
        "signal_code": "720001",
        "signal_name": "财通价值动量混合A",
        "execution": [
            {"code": "021523", "name": "财通价值动量混合C", "role": "C类优先执行/PCB核心", "daily_cap": 1000.0},
            {"code": "024481", "name": "财通品质甄选混合C", "role": "PCB替代/限购补位", "daily_cap": 1000.0},
            {"code": "021528", "name": "财通成长优选混合C", "role": "PCB替代/限购补位", "daily_cap": 1000.0},
            {"code": "720001", "name": "财通价值动量混合A", "role": "A类信号代表/备选执行", "daily_cap": np.nan},
        ],
    },
    "存储": {
        "signal_code": "018816",
        "signal_name": "方正富邦核心优势混合C",
        "execution": [
            {"code": "018816", "name": "方正富邦核心优势混合C", "role": "C类优先执行/存储代理", "daily_cap": np.nan},
            {"code": "006503", "name": "财通集成电路产业股票C", "role": "半导体/存储补位", "daily_cap": 1000.0},
            {"code": "008887", "name": "华夏国证半导体芯片ETF联接A", "role": "A类芯片补位", "daily_cap": np.nan},
        ],
    },
    "CPO": {
        "signal_code": "007817",
        "signal_name": "国泰中证全指通信设备ETF联接A",
        "execution": [
            {"code": "008327", "name": "东财通信C", "role": "C类优先执行/CPO通信代理", "daily_cap": np.nan},
            {"code": "007817", "name": "国泰中证全指通信设备ETF联接A", "role": "A类信号代表/备选执行", "daily_cap": np.nan},
        ],
    },
    "AI": {
        "signal_code": "008585",
        "signal_name": "华夏中证人工智能主题ETF联接A",
        "execution": [
            {"code": "019830", "name": "华夏数字产业混合C", "role": "C类优先执行/AI补位", "daily_cap": np.nan},
            {"code": "008585", "name": "华夏中证人工智能主题ETF联接A", "role": "A类信号代表/备选执行", "daily_cap": np.nan},
        ],
    },
    "半导体设备": {
        "signal_code": "017811",
        "signal_name": "东方人工智能主题混合C",
        "execution": [
            {"code": "017811", "name": "东方人工智能主题混合C", "role": "C类优先执行/半导体设备代理", "daily_cap": np.nan},
            {"code": "006503", "name": "财通集成电路产业股票C", "role": "设备方向补位", "daily_cap": 1000.0},
        ],
    },
}


def pct(value: float) -> str:
    return "" if pd.isna(value) else f"{value * 100:.2f}%"


def dataframe_to_markdown(frame: pd.DataFrame) -> str:
    columns = [str(col) for col in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in frame.itertuples(index=False, name=None):
        values = []
        for value in row:
            if pd.isna(value):
                values.append("")
            elif isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def max_drawdown(curve: pd.Series) -> float:
    return float((curve / curve.cummax() - 1).min())


def c_class_redemption_fee_rate(holding_days: int) -> float:
    if holding_days < 7:
        return 0.015
    if holding_days < 30:
        return 0.005
    return 0.0


def load_sector_navs(pcb_signal_code: str | None = None) -> pd.DataFrame:
    data = {}
    for sector, meta in SECTORS.items():
        signal_code = str(meta["signal_code"])
        if sector == "PCB" and pcb_signal_code is not None:
            signal_code = str(pcb_signal_code)
        nav = load_or_fetch_open_fund_nav(signal_code, max_age_hours=100000)
        data[sector] = nav.rename(sector)
    return pd.concat(data.values(), axis=1).dropna().sort_index()


def confirm_choice(raw: pd.Series, confirm_days: int = 5) -> pd.Series:
    values = raw.to_numpy(dtype=object)
    current = values[0]
    pending = current
    pending_count = 0
    output = []
    for value in values:
        if value == current:
            pending = value
            pending_count = 0
        elif value == pending:
            pending_count += 1
            if pending_count >= confirm_days:
                current = value
                pending_count = 0
        else:
            pending = value
            pending_count = 1
        output.append(current)
    return pd.Series(output, index=raw.index, name="确认后信号")


def base_top1_choice(navs: pd.DataFrame) -> pd.Series:
    momentum2 = navs.pct_change(2, fill_method=None)
    initial_scores = navs.pct_change(20, fill_method=None).iloc[:120].iloc[-1]
    anchor = str(initial_scores.idxmax()) if initial_scores.notna().any() else str(navs.columns[0])
    raw = []
    for _, row in momentum2.iterrows():
        valid = row.where(row > 0).dropna()
        raw.append(str(valid.idxmax()) if not valid.empty else anchor)
    return confirm_choice(pd.Series(raw, index=navs.index, name="基础Top1信号"), confirm_days=5)


def strong_trend_overlay(navs: pd.DataFrame, base: pd.Series) -> tuple[pd.Series, pd.DataFrame]:
    ret30 = navs.pct_change(30, fill_method=None)
    dd20 = navs / navs.rolling(20).max() - 1
    above_ma20 = navs > navs.rolling(20).mean()
    raw_regime = (ret30 > 0.12) & (dd20 > -0.03) & above_ma20
    confirmed = raw_regime.copy()
    for sector in navs.columns:
        streak = raw_regime[sector].astype(int).groupby(
            (raw_regime[sector] != raw_regime[sector].shift()).cumsum()
        ).cumsum()
        confirmed[sector] = streak >= 2

    final = []
    reason = []
    for date in navs.index:
        active = confirmed.loc[date]
        if bool(active.any()):
            final.append(str(ret30.loc[date].where(active).idxmax()))
            reason.append("强趋势覆盖")
        else:
            final.append(str(base.loc[date]))
            reason.append("2日动量Top1+5日确认")
    diagnostics = pd.DataFrame({"基础信号": base, "最终信号": final, "信号原因": reason}, index=navs.index)
    return pd.Series(final, index=navs.index, name="最终信号"), diagnostics


def fee_protected_choice(navs: pd.DataFrame, choice: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Reduce switches that would trigger short-holding C-class redemption fees."""
    ret30 = navs.pct_change(30, fill_method=None)
    ret5 = navs.pct_change(5, fill_method=None)
    current = str(choice.iloc[0])
    entry_date = choice.index[0]
    output = []
    reasons = []
    for date, raw_value in choice.items():
        proposed = str(raw_value)
        if proposed == current:
            reasons.append("保持原板块")
        else:
            holding_days = int((date - entry_date).days)
            current_ret5 = ret5.loc[date, current] if current in ret5.columns else np.nan
            advantage = (
                ret30.loc[date, proposed] - ret30.loc[date, current]
                if proposed in ret30.columns and current in ret30.columns
                else np.nan
            )
            allow = (
                holding_days >= MIN_HOLD_DAYS_FOR_FEE_PROTECTION
                or (pd.notna(current_ret5) and current_ret5 <= CURRENT_SECTOR_5D_STOP_LOSS)
                or (pd.notna(advantage) and advantage >= MOMENTUM_ADVANTAGE_TO_BREAK_HOLD)
            )
            if allow:
                current = proposed
                entry_date = date
                reasons.append("允许切换：满足持有期或强动量例外")
            else:
                reasons.append("赎回费保护：延后切换")
        output.append(current)
    return (
        pd.Series(output, index=choice.index, name="费用保护后信号"),
        pd.Series(reasons, index=choice.index, name="费用保护说明"),
    )


def choice_to_weights(choice: pd.Series, columns: pd.Index) -> pd.DataFrame:
    weights = pd.DataFrame(0.0, index=choice.index, columns=columns)
    for date, sector in choice.items():
        weights.loc[date, sector] = 1.0
    return weights


def delayed_executed_choice(choice: pd.Series) -> pd.Series:
    return choice.shift(EXECUTION_DELAY_DAYS).fillna(choice.iloc[0]).rename("执行持仓")


def backtest_choice(
    navs: pd.DataFrame,
    choice: pd.Series,
    use_c_redemption_fee: bool = True,
) -> tuple[pd.Series, pd.Series, pd.Series, dict[str, float]]:
    """Single-sector rotation backtest with FIFO redemption fee.

    The strategy is always fully invested in one selected sector. When switching,
    old lots are sold FIFO and proceeds are used to buy the new sector at the same
    NAV date. This matches daily open-fund reallocation at net-value close.
    """
    navs = navs.dropna().sort_index()
    executed = delayed_executed_choice(choice.reindex(navs.index).ffill())
    lots: dict[str, list[FundLot]] = {sector: [] for sector in navs.columns}
    cash = 1.0
    records = []
    total_redemption_fees = 0.0
    redeemed_value = 0.0
    redeemed_value_days = 0.0
    redeemed_under_7 = 0.0
    redeemed_under_30 = 0.0
    switch_count = 0

    def sector_value(sector: str, date: pd.Timestamp) -> float:
        return sum(lot.units for lot in lots[sector]) * float(navs.loc[date, sector])

    def portfolio_value(date: pd.Timestamp) -> float:
        return cash + sum(sector_value(sector, date) for sector in navs.columns)

    def sell_all(sector: str, date: pd.Timestamp) -> None:
        nonlocal cash, total_redemption_fees, redeemed_value, redeemed_value_days
        nonlocal redeemed_under_7, redeemed_under_30
        price = float(navs.loc[date, sector])
        proceeds = 0.0
        for lot in lots[sector]:
            holding_days = int((date - lot.purchase_date).days)
            gross = lot.units * price
            fee = gross * c_class_redemption_fee_rate(holding_days) if use_c_redemption_fee else 0.0
            proceeds += gross - fee
            total_redemption_fees += fee
            redeemed_value += gross
            redeemed_value_days += gross * holding_days
            if holding_days < 7:
                redeemed_under_7 += gross
            if holding_days < 30:
                redeemed_under_30 += gross
        lots[sector] = []
        cash += proceeds

    def buy_all(sector: str, date: pd.Timestamp) -> None:
        nonlocal cash
        price = float(navs.loc[date, sector])
        if cash > 1e-12:
            lots[sector].append(FundLot(cash / price, date))
            cash = 0.0

    current = str(executed.iloc[0])
    buy_all(current, navs.index[0])
    for i, date in enumerate(navs.index):
        target = str(executed.loc[date])
        if target != current:
            sell_all(current, date)
            buy_all(target, date)
            current = target
            switch_count += 1
        value = portfolio_value(date)
        records.append(
            {
                "date": date,
                "portfolio_value": value,
                "cash": cash,
                "执行持仓": current,
                "累计赎回费": total_redemption_fees,
            }
        )

    ledger = pd.DataFrame(records).set_index("date")
    curve = ledger["portfolio_value"] / ledger["portfolio_value"].iloc[0]
    returns = curve.pct_change(fill_method=None).fillna(0.0)
    metrics = {
        "total_return": float(curve.iloc[-1] - 1),
        "max_drawdown": max_drawdown(curve),
        "switch_count": float(switch_count),
        "redemption_fees": float(total_redemption_fees),
        "average_holding_days": float(redeemed_value_days / redeemed_value) if redeemed_value else np.nan,
        "under_7_day_redemption_ratio": float(redeemed_under_7 / redeemed_value) if redeemed_value else 0.0,
        "under_30_day_redemption_ratio": float(redeemed_under_30 / redeemed_value) if redeemed_value else 0.0,
    }
    return curve, returns, ledger["执行持仓"], metrics


def daily_dca_equal_curve(navs: pd.DataFrame) -> pd.Series:
    cash = 1.0
    units = pd.Series(0.0, index=navs.columns)
    daily_budget = 1.0 / len(navs)
    values = []
    for date in navs.index:
        spend = min(cash, daily_budget)
        if spend > 0:
            units += (spend / len(navs.columns)) / navs.loc[date]
            cash -= spend
        values.append(cash + float((units * navs.loc[date]).sum()))
    curve = pd.Series(values, index=navs.index)
    return curve / curve.iloc[0]


def build_strategy(
    navs: pd.DataFrame,
    use_c_redemption_fee: bool = True,
    protect_redemption_fee: bool = True,
) -> tuple[pd.Series, pd.DataFrame, pd.Series, dict[str, float]]:
    base = base_top1_choice(navs)
    raw_choice, diagnostics = strong_trend_overlay(navs, base)
    diagnostics["费用保护前信号"] = raw_choice
    if protect_redemption_fee:
        final_choice, protection_reason = fee_protected_choice(navs, raw_choice)
        diagnostics["最终信号"] = final_choice
        diagnostics["费用保护说明"] = protection_reason
    else:
        final_choice = raw_choice
        diagnostics["费用保护说明"] = "未启用"
    curve, _, executed, metrics = backtest_choice(navs, final_choice, use_c_redemption_fee=use_c_redemption_fee)
    diagnostics["执行持仓"] = executed
    diagnostics["策略净值"] = curve
    return final_choice, diagnostics, curve, metrics


def evaluate_windows(navs: pd.DataFrame, choice: pd.Series, use_c_redemption_fee: bool = True) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    curves = []
    if len(navs) < WINDOW_DAYS + SLIDING_WINDOWS + 30:
        raise ValueError("共同净值历史不足，无法做30个三个月滑动窗口。")

    for window_id, end_pos in enumerate(range(len(navs) - SLIDING_WINDOWS, len(navs)), start=1):
        start_pos = end_pos - WINDOW_DAYS + 1
        start = navs.index[start_pos]
        end = navs.index[end_pos]
        test_navs = navs.loc[start:end]
        test_choice = choice.loc[start:end]
        strategy_curve, _, executed, metrics = backtest_choice(
            test_navs,
            test_choice,
            use_c_redemption_fee=use_c_redemption_fee,
        )
        strategy_curve = strategy_curve / strategy_curve.iloc[0]
        equal_curve = test_navs.div(test_navs.iloc[0]).mean(axis=1)
        dca_curve = daily_dca_equal_curve(test_navs)
        all_in_returns = {
            sector: float(test_navs[sector].iloc[-1] / test_navs[sector].iloc[0] - 1)
            for sector in test_navs.columns
        }
        best_sector = max(all_in_returns, key=all_in_returns.get)
        best_curve = test_navs[best_sector] / test_navs[best_sector].iloc[0]
        rows.append(
            {
                "窗口": window_id,
                "开始日期": start.date().isoformat(),
                "结束日期": end.date().isoformat(),
                "轮动收益": float(strategy_curve.iloc[-1] - 1),
                "轮动最大回撤": max_drawdown(strategy_curve),
                "等权收益": float(equal_curve.iloc[-1] - 1),
                "等权最大回撤": max_drawdown(equal_curve),
                "每日定投收益": float(dca_curve.iloc[-1] - 1),
                "每日定投最大回撤": max_drawdown(dca_curve),
                "最佳单板块": best_sector,
                "最佳单板块收益": float(best_curve.iloc[-1] - 1),
                "最佳单板块最大回撤": max_drawdown(best_curve),
                "跑赢等权": float(strategy_curve.iloc[-1] - 1) > float(equal_curve.iloc[-1] - 1),
                "跑赢每日定投": float(strategy_curve.iloc[-1] - 1) > float(dca_curve.iloc[-1] - 1),
                "跑赢最佳单板块": float(strategy_curve.iloc[-1] - 1) > float(best_curve.iloc[-1] - 1),
                "窗口首日执行持仓": str(executed.iloc[0]),
                "窗口末日执行持仓": str(executed.iloc[-1]),
                "窗口内切换次数": int(metrics["switch_count"]),
                "赎回费": float(metrics["redemption_fees"]),
                "平均持有天数": float(metrics["average_holding_days"]) if pd.notna(metrics["average_holding_days"]) else np.nan,
                "7天内赎回占比": float(metrics["under_7_day_redemption_ratio"]),
                "30天内赎回占比": float(metrics["under_30_day_redemption_ratio"]),
            }
        )
        curves.append(
            pd.DataFrame(
                {
                    "日期": test_navs.index,
                    "窗口": window_id,
                    "轮动": strategy_curve.to_numpy(),
                    "等权": equal_curve.to_numpy(),
                    "每日定投": dca_curve.to_numpy(),
                    "最佳单板块": best_curve.to_numpy(),
                    "执行持仓": executed.to_numpy(),
                }
            )
        )
    return pd.DataFrame(rows), pd.concat(curves, ignore_index=True)


def summarize_windows(windows: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "窗口数量": len(windows),
                "跑赢等权次数": int(windows["跑赢等权"].sum()),
                "跑赢每日定投次数": int(windows["跑赢每日定投"].sum()),
                "跑赢最佳单板块次数": int(windows["跑赢最佳单板块"].sum()),
                "轮动平均收益": float(windows["轮动收益"].mean()),
                "轮动收益标准差": float(windows["轮动收益"].std()),
                "轮动平均最大回撤": float(windows["轮动最大回撤"].mean()),
                "轮动最差最大回撤": float(windows["轮动最大回撤"].min()),
                "等权平均收益": float(windows["等权收益"].mean()),
                "每日定投平均收益": float(windows["每日定投收益"].mean()),
                "最佳单板块平均收益": float(windows["最佳单板块收益"].mean()),
                "最佳单板块平均最大回撤": float(windows["最佳单板块最大回撤"].mean()),
                "平均切换次数": float(windows["窗口内切换次数"].mean()),
                "平均赎回费": float(windows["赎回费"].mean()),
                "平均持有天数": float(windows["平均持有天数"].mean()),
                "平均7天内赎回占比": float(windows["7天内赎回占比"].mean()),
                "平均30天内赎回占比": float(windows["30天内赎回占比"].mean()),
            }
        ]
    )


def latest_signal_table(navs: pd.DataFrame, choice: pd.Series, diagnostics: pd.DataFrame) -> pd.DataFrame:
    weights = choice_to_weights(choice, navs.columns)
    last = navs.index[-1]
    prev = navs.index[-2]
    rows = []
    for sector in navs.columns:
        rows.append(
            {
                "板块": sector,
                "代表基金代码": SECTORS[sector]["signal_code"],
                "代表基金名称": SECTORS[sector]["signal_name"],
                "最新日期": last.date().isoformat(),
                "信号目标权重": float(weights.loc[last, sector]),
                "前一日信号目标权重": float(weights.loc[prev, sector]),
                "执行持仓权重": 1.0 if diagnostics.loc[last, "执行持仓"] == sector else 0.0,
                "20日涨跌幅": float(navs[sector].pct_change(20, fill_method=None).loc[last]),
                "30日涨跌幅": float(navs[sector].pct_change(30, fill_method=None).loc[last]),
                "60日涨跌幅": float(navs[sector].pct_change(60, fill_method=None).loc[last]),
                "信号原因": diagnostics.loc[last, "信号原因"] if diagnostics.loc[last, "最终信号"] == sector else "",
                "按10000元目标金额": float(weights.loc[last, sector] * INITIAL_CAPITAL),
            }
        )
    return pd.DataFrame(rows).sort_values(["信号目标权重", "执行持仓权重"], ascending=False)


def execution_candidate_table(latest_targets: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector, meta in SECTORS.items():
        target_amount = float(latest_targets.loc[latest_targets["板块"] == sector, "按10000元目标金额"].iloc[0])
        remaining = target_amount
        for candidate in meta["execution"]:
            cap = candidate["daily_cap"]
            suggested = max(0.0, remaining) if pd.isna(cap) else min(max(0.0, remaining), float(cap))
            remaining -= suggested
            rows.append(
                {
                    "板块": sector,
                    "基金代码": candidate["code"],
                    "基金名称": candidate["name"],
                    "执行角色": candidate["role"],
                    "单日限购假设": "" if pd.isna(cap) else float(cap),
                    "按10000元组合今日建议买入上限": round(suggested, 2),
                    "费用口径": C_CLASS_FEE_SOURCE,
                }
            )
    return pd.DataFrame(rows)


def compare_fee_impact(navs: pd.DataFrame, raw_choice: pd.Series, protected_choice: pd.Series) -> pd.DataFrame:
    rows = []
    for label, selected_choice, use_fee in [
        ("原始Top1，不计赎回费", raw_choice, False),
        ("原始Top1，计C类FIFO赎回费", raw_choice, True),
        ("赎回费保护后，计C类FIFO赎回费", protected_choice, True),
    ]:
        windows, _ = evaluate_windows(navs, selected_choice, use_c_redemption_fee=use_fee)
        s = summarize_windows(windows).iloc[0].to_dict()
        s["费用口径"] = label
        rows.append(s)
    return pd.DataFrame(rows)


def write_report(
    navs: pd.DataFrame,
    diagnostics: pd.DataFrame,
    latest: pd.DataFrame,
    summary: pd.DataFrame,
    fee_impact: pd.DataFrame,
    execution: pd.DataFrame,
) -> None:
    s = summary.iloc[0]
    last = navs.index[-1]
    latest_signal = str(diagnostics.loc[last, "最终信号"])
    executed = str(diagnostics.loc[last, "执行持仓"])
    lines = [
        "# 五板块Top1轮动回测报告",
        "",
        f"- 数据截止：{last.date().isoformat()}",
        "- 板块口径：PCB / 存储 / CPO / AI / 半导体设备",
        f"- 最新信号目标：{latest_signal}",
        f"- 当前实际执行持仓：{executed}",
        f"- 费用口径：{C_CLASS_FEE_SOURCE}",
        "",
        "## 策略规则",
        "",
        "1. 单个板块持仓允许从0%到100%。",
        "2. 基础信号：2日动量Top1，只追2日收益为正的板块。",
        "3. 切换信号必须连续5个交易日确认。",
        "4. 赎回费保护：原则上至少持有7天；只有当前板块5日跌幅<=-8%，或新板块30日动量领先当前板块>=30个百分点，才允许提前切换。",
        "5. 信号产生后按2个基金净值日延迟执行。",
        "6. 强趋势覆盖层：30日涨幅>12%、20日回撤>-3%、站上MA20、连续2日确认。",
        "7. 交易成本使用FIFO赎回费：持有<7天扣1.5%，7-30天扣0.5%，>=30天扣0%。",
        "",
        "## 30个三个月滑动窗口，计C类FIFO赎回费",
        "",
        f"- 跑赢等权：{int(s['跑赢等权次数'])}/{int(s['窗口数量'])}",
        f"- 跑赢每日定投：{int(s['跑赢每日定投次数'])}/{int(s['窗口数量'])}",
        f"- 跑赢最佳单板块：{int(s['跑赢最佳单板块次数'])}/{int(s['窗口数量'])}",
        f"- 轮动平均收益：{pct(s['轮动平均收益'])}，收益标准差：{pct(s['轮动收益标准差'])}",
        f"- 轮动平均最大回撤：{pct(s['轮动平均最大回撤'])}，最差最大回撤：{pct(s['轮动最差最大回撤'])}",
        f"- 平均切换次数：{s['平均切换次数']:.2f}",
        f"- 平均赎回费：{pct(s['平均赎回费'])}",
        f"- 平均持有天数：{s['平均持有天数']:.2f}",
        f"- 平均7天内赎回占比：{pct(s['平均7天内赎回占比'])}",
        f"- 平均30天内赎回占比：{pct(s['平均30天内赎回占比'])}",
        "",
        "## 费用影响对比",
        "",
        dataframe_to_markdown(fee_impact),
        "",
        "## 当前板块信号",
        "",
        dataframe_to_markdown(latest),
        "",
        "## 执行候选基金",
        "",
        dataframe_to_markdown(execution),
        "",
        "## 结论",
        "",
        "- 这版优先使用C类基金作为执行候选，信号代表基金仍可使用历史更长的A类或主题代表。",
        "- 如果后续能拿到每只基金官方赎回费表，应替换通用C类规则；当前公开接口对样本基金返回为空。",
    ]
    (OUT / "sector_level_report_cn.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    navs = load_sector_navs()
    base = base_top1_choice(navs)
    raw_choice, _ = strong_trend_overlay(navs, base)
    choice, diagnostics, full_curve, full_metrics = build_strategy(
        navs,
        use_c_redemption_fee=True,
        protect_redemption_fee=True,
    )
    latest = latest_signal_table(navs, choice, diagnostics)
    windows, curves = evaluate_windows(navs, choice, use_c_redemption_fee=True)
    summary = summarize_windows(windows)
    execution = execution_candidate_table(latest)
    weights = choice_to_weights(choice, navs.columns)
    fee_impact = compare_fee_impact(navs, raw_choice, choice)

    navs.to_csv(OUT / "sector_proxy_nav_cn.csv", index_label="日期", encoding="utf-8-sig")
    weights.to_csv(OUT / "sector_target_weights_cn.csv", index_label="日期", encoding="utf-8-sig")
    diagnostics.to_csv(OUT / "sector_signal_diagnostics_cn.csv", index_label="日期", encoding="utf-8-sig")
    full_curve.rename("策略净值").to_csv(OUT / "sector_rotation_full_curve_cn.csv", index_label="日期", encoding="utf-8-sig")
    latest.to_csv(OUT / "latest_sector_signal_cn.csv", index=False, encoding="utf-8-sig")
    execution.to_csv(OUT / "sector_execution_candidates_cn.csv", index=False, encoding="utf-8-sig")
    windows.to_csv(OUT / "sector_rotation_windows_cn.csv", index=False, encoding="utf-8-sig")
    curves.to_csv(OUT / "sector_rotation_window_curves_cn.csv", index=False, encoding="utf-8-sig")
    summary.to_csv(OUT / "sector_rotation_summary_cn.csv", index=False, encoding="utf-8-sig")
    fee_impact.to_csv(OUT / "sector_rotation_fee_impact_cn.csv", index=False, encoding="utf-8-sig")
    write_report(navs, diagnostics, latest, summary, fee_impact, execution)

    # Standard post-run artifacts: 1/3/6-month entry curves, trades, returns and drawdowns.
    from tools.plot_sector_rotation_entry_examples import generate_standard_entry_reports

    generate_standard_entry_reports(navs=navs, choice=choice)

    print(summary.to_string(index=False))
    print(fee_impact.to_string(index=False))
    print(latest[["板块", "信号目标权重", "执行持仓权重", "20日涨跌幅", "30日涨跌幅", "60日涨跌幅", "信号原因"]].to_string(index=False))


if __name__ == "__main__":
    main()
