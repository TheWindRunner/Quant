"""Public-data short-window US-to-China theme correlation diagnostics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_public_close(path: str | Path) -> pd.Series:
    frame = pd.read_csv(path, parse_dates=["date"])
    return (
        frame.set_index("date")["close"]
        .astype(float)
        .sort_index()
        .rename(Path(path).stem)
    )


def equal_weight_return_basket(prices: pd.DataFrame) -> pd.Series:
    returns = prices.sort_index().pct_change(fill_method=None)
    basket_return = returns.mean(axis=1, skipna=False)
    return (1.0 + basket_return.fillna(0.0)).cumprod().rename("basket")


def fisher_confidence_interval(
    correlation: float,
    observations: int,
    confidence: float = 0.95,
) -> tuple[float, float]:
    if observations <= 3 or not np.isfinite(correlation):
        return (np.nan, np.nan)
    clipped = float(np.clip(correlation, -0.999999, 0.999999))
    z = np.arctanh(clipped)
    critical = 1.959963984540054 if confidence == 0.95 else 1.6448536269514722
    error = critical / np.sqrt(observations - 3)
    return (float(np.tanh(z - error)), float(np.tanh(z + error)))


def correlation_with_lags(
    cn_nav: pd.Series,
    us_nav: pd.Series,
    maximum_lead: int = 3,
) -> pd.DataFrame:
    """Compare naive same-date and causal prior-US-close relationships."""
    cn_return = cn_nav.sort_index().pct_change(fill_method=None).dropna()
    us_return = us_nav.sort_index().pct_change(fill_method=None).dropna()
    rows: list[dict[str, object]] = []

    same = pd.concat(
        [cn_return.rename("cn"), us_return.rename("us")], axis=1
    ).dropna()
    same_corr = float(same["cn"].corr(same["us"])) if len(same) >= 3 else np.nan
    lower, upper = fisher_confidence_interval(same_corr, len(same))
    same_direction = (
        float((np.sign(same["cn"]) == np.sign(same["us"])).mean())
        if len(same)
        else np.nan
    )
    rows.append(
        {
            "relationship": "same_calendar_date_noncausal",
            "us_lead_trading_closes": 0,
            "observations": len(same),
            "correlation": same_corr,
            "ci95_lower": lower,
            "ci95_upper": upper,
            "direction_accuracy": same_direction,
            "cn_return_after_us_up": (
                float(same.loc[same["us"] > 0, "cn"].mean())
                if (same["us"] > 0).any()
                else np.nan
            ),
            "cn_return_after_us_down": (
                float(same.loc[same["us"] < 0, "cn"].mean())
                if (same["us"] < 0).any()
                else np.nan
            ),
        }
    )

    us_dates = us_return.index
    for lead in range(1, maximum_lead + 1):
        pairs = []
        for date, cn_value in cn_return.items():
            prior = us_dates[us_dates < date]
            if len(prior) < lead:
                continue
            us_date = prior[-lead]
            pairs.append((date, cn_value, float(us_return.loc[us_date])))
        aligned = pd.DataFrame(pairs, columns=["date", "cn", "us"])
        correlation = (
            float(aligned["cn"].corr(aligned["us"]))
            if len(aligned) >= 3
            else np.nan
        )
        lower, upper = fisher_confidence_interval(correlation, len(aligned))
        direction_accuracy = (
            float((np.sign(aligned["cn"]) == np.sign(aligned["us"])).mean())
            if len(aligned)
            else np.nan
        )
        rows.append(
            {
                "relationship": "prior_completed_us_close",
                "us_lead_trading_closes": lead,
                "observations": len(aligned),
                "correlation": correlation,
                "ci95_lower": lower,
                "ci95_upper": upper,
                "direction_accuracy": direction_accuracy,
                "cn_return_after_us_up": (
                    float(aligned.loc[aligned["us"] > 0, "cn"].mean())
                    if (aligned["us"] > 0).any()
                    else np.nan
                ),
                "cn_return_after_us_down": (
                    float(aligned.loc[aligned["us"] < 0, "cn"].mean())
                    if (aligned["us"] < 0).any()
                    else np.nan
                ),
            }
        )
    result = pd.DataFrame(rows)
    result["confidence"] = np.where(
        (result["observations"] >= 60)
        & (result["ci95_lower"] * result["ci95_upper"] > 0),
        "medium_or_higher",
        "low",
    )
    return result


def build_public_correlation_report(
    data_dir: str | Path = "data/public_market",
    nav_dir: str | Path = "data/nav_cache",
    output_dir: str | Path = "output/public_correlation",
) -> pd.DataFrame:
    data_dir = Path(data_dir)
    nav_dir = Path(nav_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    anet = load_public_close(data_dir / "ANET_2026-05-15_2026-06-12.csv")
    cohr = load_public_close(data_dir / "COHR_2026-05-15_2026-06-12.csv")
    mu = load_public_close(data_dir / "MU_2026-05-15_2026-06-12.csv")
    soxx = load_public_close(data_dir / "SOXX_2026-05-15_2026-06-12.csv")
    cpo_us = equal_weight_return_basket(
        pd.concat([anet.rename("ANET"), cohr.rename("COHR")], axis=1)
    )

    def nav(code: str) -> pd.Series:
        frame = pd.read_csv(nav_dir / f"{code}.csv", parse_dates=["date"])
        return frame.set_index("date")["close"].astype(float).sort_index()

    relationships = {
        "cpo_us_basket_vs_007817": (nav("007817"), cpo_us),
        "mu_memory_vs_008887": (nav("008887"), mu),
        "soxx_semiconductor_vs_008887": (nav("008887"), soxx),
        "soxx_semiconductor_vs_008585": (nav("008585"), soxx),
    }
    rows = []
    for name, (cn_series, us_series) in relationships.items():
        result = correlation_with_lags(cn_series, us_series)
        result.insert(0, "pair", name)
        rows.append(result)
    combined = pd.concat(rows, ignore_index=True)
    combined.to_csv(
        output_dir / "short_window_correlations.csv",
        index=False,
        encoding="utf-8-sig",
    )
    lines = [
        "# 中美科技主题短窗口日频相关性",
        "",
        "数据窗口：2026-05-15至2026-06-12，公开美股价格共20个交易日。",
        "",
        "重要限制：收益观测不足20个，所有结果均为低置信，"
        "只可用于盘中方向确认，不可用于参数训练或宣称长期稳定关系。",
        "",
        "同日相关为非因果统计，因为A股收盘早于美股同日收盘；"
        "14:30策略只能使用已经完成的上一美股交易日及更早数据。",
        "",
        "```text",
        combined.to_string(index=False),
        "```",
        "",
        "美股来源：Investing.com公开历史行情页；"
        "中国侧来源：本地完整基金单位净值。",
    ]
    (output_dir / "short_window_report.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return combined
