"""Build the daily theme signal and six-month backtest report."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .daily_strategy import (
    THEMES,
    backtest_daily,
    compare_fund_strategies,
    daily_features,
    equal_weight_nav,
    generate_positions,
    lagged_daily_correlation,
)
from .data import fetch_basket_prices, load_or_fetch_open_fund_nav


def run_daily_report(output_dir: str | Path = "output/daily") -> pd.DataFrame:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    end = pd.Timestamp.today().normalize()
    data_start = end - pd.Timedelta(days=420)
    backtest_start = end - pd.DateOffset(months=6)
    summary = []

    for theme, definition in THEMES.items():
        cn_prices = fetch_basket_prices(definition["cn"], data_start, end)
        us_prices = fetch_basket_prices(definition["us"], data_start, end)
        cn_nav = equal_weight_nav(cn_prices)
        us_nav = equal_weight_nav(us_prices)
        correlation = lagged_daily_correlation(cn_nav, us_nav)

        fund_code = definition["fund_code"]
        if fund_code:
            fund_nav = load_or_fetch_open_fund_nav(fund_code).loc[data_start:end]
            signal_nav = fund_nav
        else:
            signal_nav = cn_nav

        features = daily_features(signal_nav, us_nav)
        decisions = generate_positions(features)
        backtest = backtest_daily(signal_nav, decisions, start=backtest_start)
        comparison = compare_fund_strategies(
            signal_nav, us_nav, start=backtest_start
        )
        latest = decisions.iloc[-1]
        best_corr = correlation.loc[correlation["correlation"].abs().idxmax()]
        summary.append(
            {
                "theme": theme,
                "fund_code": fund_code,
                "fund_name": definition["fund_name"],
                "data_date": decisions.index[-1].date().isoformat(),
                "action": latest["action"],
                "target_position": latest["target_position"],
                "momentum20": latest["momentum20"],
                "momentum60": latest["momentum60"],
                "volatility20": latest["volatility20"],
                "best_us_lead_days": int(best_corr["us_lead_days"]),
                "best_correlation": best_corr["correlation"],
                **{f"bt_{key}": value for key, value in backtest["metrics"].items()},
            }
        )
        prefix = output / theme
        cn_prices.to_csv(f"{prefix}_cn_prices.csv", encoding="utf-8-sig")
        us_prices.to_csv(f"{prefix}_us_prices.csv", encoding="utf-8-sig")
        correlation.to_csv(f"{prefix}_correlation.csv", index=False, encoding="utf-8-sig")
        decisions.to_csv(f"{prefix}_decisions.csv", encoding="utf-8-sig")
        backtest["equity"].to_csv(f"{prefix}_backtest.csv", encoding="utf-8-sig")
        comparison.to_csv(f"{prefix}_strategy_comparison.csv", encoding="utf-8-sig")

    result = pd.DataFrame(summary).sort_values(
        ["target_position", "momentum20"], ascending=False
    )
    result.to_csv(output / "daily_summary.csv", index=False, encoding="utf-8-sig")
    return result
