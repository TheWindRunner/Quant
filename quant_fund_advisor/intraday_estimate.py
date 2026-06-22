"""14:30 mutual-fund estimate overlay with reliability calibration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

try:
    import requests
except ImportError:
    requests = None


@dataclass(frozen=True)
class IntradayEstimate:
    fund_code: str
    fund_name: str
    previous_nav: float
    estimated_nav: float
    estimated_change_pct: float
    estimate_time: pd.Timestamp
    source: str


@dataclass(frozen=True)
class EstimatePolicy:
    minimum_abs_change_pct: float = 1.50
    strong_abs_change_pct: float = 3.00
    maximum_calibration_mae_pct: float = 1.20
    minimum_direction_accuracy: float = 0.70
    tactical_step: float = 0.10
    maximum_tactical_adjustment: float = 0.10


@dataclass(frozen=True)
class SyntheticEstimateConfig:
    noise_band_pct: float = 1.50
    small_move_threshold_pct: float = 1.20
    medium_move_threshold_pct: float = 2.50
    small_move_flip_probability: float = 0.35
    medium_move_flip_probability: float = 0.12
    seed: int = 42


def fetch_eastmoney_intraday_estimate(
    fund_code: str,
    timeout: float = 10.0,
) -> IntradayEstimate:
    """Fetch Eastmoney/Tiantian Fund's public JSONP estimate."""
    if requests is None:
        raise RuntimeError("requests is required for intraday estimates")
    url = f"https://fundgz.1234567.com.cn/js/{fund_code}.js"
    response = requests.get(
        url,
        params={"rt": int(pd.Timestamp.now().timestamp() * 1000)},
        headers={
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://fund.eastmoney.com/",
        },
        timeout=timeout,
    )
    response.raise_for_status()
    match = re.search(r"jsonpgz\((.*)\)\s*;?", response.text)
    if not match:
        raise ValueError(f"Invalid estimate response for {fund_code}")
    payload: dict[str, Any] = json.loads(match.group(1))
    return IntradayEstimate(
        fund_code=str(payload["fundcode"]),
        fund_name=str(payload["name"]),
        previous_nav=float(payload["dwjz"]),
        estimated_nav=float(payload["gsz"]),
        estimated_change_pct=float(payload["gszzl"]),
        estimate_time=pd.Timestamp(payload["gztime"]),
        source="eastmoney_fundgz",
    )


def fetch_akshare_etf_iopv(etf_code: str) -> dict[str, float | str]:
    """Read a linked ETF's exchange quote and IOPV through AKShare."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AKShare is required for ETF IOPV confirmation") from exc
    frame = ak.fund_etf_spot_em()
    code_column = next(
        (
            column
            for column in ("代码", "基金代码")
            if column in frame.columns
        ),
        None,
    )
    if code_column is None:
        raise ValueError("ETF spot response has no code column")
    row = frame.loc[frame[code_column].astype(str).str.zfill(6) == etf_code.zfill(6)]
    if row.empty:
        raise ValueError(f"ETF {etf_code} not found in AKShare spot data")
    record = row.iloc[0]

    def number(*names: str) -> float:
        for name in names:
            if name in record.index:
                value = pd.to_numeric(record[name], errors="coerce")
                if pd.notna(value):
                    return float(value)
        return float("nan")

    return {
        "etf_code": etf_code.zfill(6),
        "etf_name": str(record.get("名称", record.get("基金简称", ""))),
        "price": number("最新价", "最新"),
        "change_pct": number("涨跌幅"),
        "iopv": number("IOPV实时估值", "IOPV"),
        "discount_premium_pct": number("基金折价率", "折溢价率"),
        "source": "akshare_fund_etf_spot_em",
    }


def fetch_akshare_fund_estimate(
    fund_code: str,
    category: str = "ETF联接",
) -> IntradayEstimate:
    """Fetch AKShare's current Eastmoney estimate table as a fallback."""
    try:
        import akshare as ak
    except ImportError as exc:
        raise RuntimeError("AKShare is required for fund estimates") from exc
    frame = ak.fund_value_estimation_em(symbol=category)
    code_column = next(
        (
            column
            for column in ("基金代码", "代码")
            if column in frame.columns
        ),
        None,
    )
    if code_column is None:
        raise ValueError("Fund estimate response has no code column")
    row = frame.loc[
        frame[code_column].astype(str).str.zfill(6) == fund_code.zfill(6)
    ]
    if row.empty:
        raise ValueError(f"Fund {fund_code} not found in estimate table")
    record = row.iloc[0]

    def matching_column(
        required: tuple[str, ...],
        excluded: tuple[str, ...] = (),
    ) -> object:
        for column in frame.columns:
            text = str(column)
            if all(word in text for word in required) and not any(
                word in text for word in excluded
            ):
                return column
        raise ValueError(
            f"No estimate column matching {required} for fund {fund_code}"
        )

    estimated_nav_column = matching_column(("估算数据", "估算值"))
    estimated_change_column = matching_column(("估算数据", "估算增长率"))
    previous_nav_column = matching_column(("公布数据", "单位净值"))

    def number(value: object) -> float:
        cleaned = str(value).replace("%", "").replace(",", "").strip()
        return float(pd.to_numeric(cleaned, errors="raise"))

    return IntradayEstimate(
        fund_code=fund_code.zfill(6),
        fund_name=str(record.get("基金名称", record.get("名称", ""))),
        previous_nav=number(record[previous_nav_column]),
        estimated_nav=number(record[estimated_nav_column]),
        estimated_change_pct=number(record[estimated_change_column]),
        estimate_time=pd.Timestamp.now(tz="Asia/Shanghai"),
        source="akshare_fund_value_estimation_em",
    )


def calibration_metrics(history: pd.DataFrame) -> dict[str, float]:
    """Compare archived 14:30 estimates with final same-day NAV returns."""
    required = {"estimated_change_pct", "actual_change_pct"}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"Estimate history missing columns: {sorted(missing)}")
    clean = history.dropna(subset=list(required)).copy()
    if clean.empty:
        return {"observations": 0.0, "mae_pct": np.nan, "direction_accuracy": np.nan}
    error = clean["estimated_change_pct"] - clean["actual_change_pct"]
    direction = (
        np.sign(clean["estimated_change_pct"])
        == np.sign(clean["actual_change_pct"])
    )
    return {
        "observations": float(len(clean)),
        "mae_pct": float(error.abs().mean()),
        "direction_accuracy": float(direction.mean()),
    }


def estimate_confidence(
    estimate_change_pct: float,
    metrics: dict[str, float] | None = None,
    etf_change_pct: float | None = None,
    breadth_ratio: float | None = None,
    policy: EstimatePolicy | None = None,
) -> dict[str, object]:
    """Require a large estimate and independent directional confirmation."""
    policy = policy or EstimatePolicy()
    absolute = abs(estimate_change_pct)
    large_move = absolute >= policy.minimum_abs_change_pct
    strong_move = absolute >= policy.strong_abs_change_pct
    expected_sign = np.sign(estimate_change_pct)
    etf_agrees = (
        etf_change_pct is not None
        and np.sign(etf_change_pct) == expected_sign
        and abs(etf_change_pct) >= 0.5
    )
    breadth_agrees = (
        breadth_ratio is not None
        and (
            (expected_sign > 0 and breadth_ratio >= 0.65)
            or (expected_sign < 0 and breadth_ratio <= 0.35)
        )
    )
    calibrated = False
    if metrics and metrics.get("observations", 0) >= 20:
        calibrated = (
            metrics.get("mae_pct", np.inf) <= policy.maximum_calibration_mae_pct
            and metrics.get("direction_accuracy", 0) >= policy.minimum_direction_accuracy
        )
    independent_confirmations = int(etf_agrees) + int(breadth_agrees)
    reliable = bool(
        large_move
        and independent_confirmations >= 1
        and (calibrated or (strong_move and independent_confirmations == 2))
    )
    return {
        "reliable": reliable,
        "large_move": large_move,
        "strong_move": strong_move,
        "etf_agrees": etf_agrees,
        "breadth_agrees": breadth_agrees,
        "calibrated": calibrated,
        "independent_confirmations": independent_confirmations,
    }


def apply_intraday_overlay(
    base_target: float,
    estimate_change_pct: float,
    confidence: dict[str, object],
    policy: EstimatePolicy | None = None,
) -> dict[str, object]:
    """Contrarian tactical overlay: large down adds, large up trims."""
    policy = policy or EstimatePolicy()
    base = float(np.clip(base_target, 0, 1))
    if not confidence.get("reliable", False):
        return {
            "target_position": base,
            "overlay": 0.0,
            "action": "IGNORE_ESTIMATE",
        }
    magnitude = (
        policy.maximum_tactical_adjustment
        if abs(estimate_change_pct) >= policy.strong_abs_change_pct
        else policy.tactical_step
    )
    if estimate_change_pct < 0:
        overlay = magnitude
        action = "TACTICAL_ADD"
    else:
        overlay = -magnitude
        action = "TACTICAL_TRIM"
    return {
        "target_position": float(np.clip(base + overlay, 0, 1)),
        "overlay": overlay,
        "action": action,
    }


def archive_estimate(
    estimate: IntradayEstimate,
    path: str,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame(
        [
            {
                "date": estimate.estimate_time.date(),
                "estimate_time": estimate.estimate_time,
                "fund_code": estimate.fund_code,
                "previous_nav": estimate.previous_nav,
                "estimated_nav": estimate.estimated_nav,
                "estimated_change_pct": estimate.estimated_change_pct,
                "source": estimate.source,
            }
        ]
    )
    try:
        existing = pd.read_csv(output)
        combined = pd.concat([existing, row], ignore_index=True)
        combined = combined.drop_duplicates(
            ["fund_code", "estimate_time"], keep="last"
        )
    except FileNotFoundError:
        combined = row
    combined.to_csv(output, index=False, encoding="utf-8-sig")


def reconcile_estimate_history(
    history: pd.DataFrame,
    nav: pd.Series,
) -> pd.DataFrame:
    """Attach realized same-day NAV return after the official NAV is published."""
    result = history.copy()
    result["date"] = pd.to_datetime(result["date"])
    official = nav.sort_index().pct_change(fill_method=None).mul(100)
    result["actual_change_pct"] = result["date"].map(official)
    result["absolute_error_pct"] = (
        result["estimated_change_pct"] - result["actual_change_pct"]
    ).abs()
    result["direction_correct"] = (
        np.sign(result["estimated_change_pct"])
        == np.sign(result["actual_change_pct"])
    )
    return result


def adaptive_policy_from_history(
    history: pd.DataFrame,
    base_policy: EstimatePolicy | None = None,
) -> EstimatePolicy:
    """Use only archived past errors to set a conservative move threshold."""
    base = base_policy or EstimatePolicy()
    metrics = calibration_metrics(history)
    if metrics["observations"] < 20 or not np.isfinite(metrics["mae_pct"]):
        return base
    error_threshold = float(max(base.minimum_abs_change_pct, 2.0 * metrics["mae_pct"]))
    return EstimatePolicy(
        minimum_abs_change_pct=min(error_threshold, 3.0),
        strong_abs_change_pct=max(base.strong_abs_change_pct, error_threshold * 1.5),
        maximum_calibration_mae_pct=base.maximum_calibration_mae_pct,
        minimum_direction_accuracy=base.minimum_direction_accuracy,
        tactical_step=base.tactical_step,
        maximum_tactical_adjustment=base.maximum_tactical_adjustment,
    )


def historical_extreme_move_overlay(
    nav: pd.Series,
    base_target: float = 0.90,
    add_threshold_pct: float = -1.50,
    trim_threshold_pct: float = 3.00,
    tactical_step: float = 0.10,
    require_long_trend_for_add: bool = True,
    minimum_days_between_changes: int = 10,
) -> pd.Series:
    """Optimistic historical ceiling for a 14:30 extreme-estimate overlay.

    It uses final same-day NAV direction as if the intraday estimate were
    correct, but execution remains delayed to the next NAV. Real deployment
    must perform no better than this ceiling before estimation error and noise.
    """
    nav = nav.sort_index().astype(float)
    daily_change_pct = nav.pct_change(fill_method=None) * 100
    ma120 = nav.rolling(120).mean()
    minimum_target = max(0.0, base_target - tactical_step)
    maximum_target = min(1.0, base_target + tactical_step)
    current = float(base_target)
    last_change = -minimum_days_between_changes
    values = []
    for location, date in enumerate(nav.index):
        can_change = location - last_change >= minimum_days_between_changes
        add = daily_change_pct.loc[date] <= add_threshold_pct
        if require_long_trend_for_add:
            add = bool(add and nav.loc[date] >= ma120.loc[date])
        trim = daily_change_pct.loc[date] >= trim_threshold_pct
        if can_change and add and current < maximum_target:
            current = min(maximum_target, current + tactical_step)
            last_change = location
        elif can_change and trim and current > minimum_target:
            current = max(minimum_target, current - tactical_step)
            last_change = location
        values.append(current)
    return pd.Series(
        values,
        index=nav.index,
        name="extreme_estimate_overlay",
    ).clip(0, 1)


def simulate_intraday_estimate_history(
    nav: pd.Series,
    config: SyntheticEstimateConfig | None = None,
) -> pd.DataFrame:
    """Create a reproducible pseudo-14:30 estimate history from final NAV moves.

    Large daily moves usually keep the correct direction but with estimation noise.
    Small daily moves are allowed to flip sign, matching the idea that weak
    intraday estimates can easily point the wrong way.
    """
    config = config or SyntheticEstimateConfig()
    nav = nav.sort_index().astype(float)
    actual_change_pct = nav.pct_change(fill_method=None).mul(100)
    previous_nav = nav.shift(1)
    rng = np.random.default_rng(config.seed)
    estimated_change_pct = []
    estimate_quality = []
    for change in actual_change_pct.fillna(0.0):
        noise = rng.uniform(-config.noise_band_pct, config.noise_band_pct)
        flip_probability = 0.0
        quality = "large_move"
        absolute = abs(float(change))
        if absolute < config.small_move_threshold_pct:
            flip_probability = config.small_move_flip_probability
            quality = "small_move"
        elif absolute < config.medium_move_threshold_pct:
            flip_probability = config.medium_move_flip_probability
            quality = "medium_move"
        estimate = float(change + noise)
        if absolute > 0 and rng.random() < flip_probability:
            estimate = -estimate
        estimated_change_pct.append(estimate)
        estimate_quality.append(quality)
    history = pd.DataFrame(
        {
            "date": nav.index,
            "previous_nav": previous_nav.to_numpy(),
            "actual_nav": nav.to_numpy(),
            "actual_change_pct": actual_change_pct.to_numpy(),
            "estimated_change_pct": estimated_change_pct,
            "estimate_quality": estimate_quality,
        }
    ).dropna(subset=["previous_nav"]).reset_index(drop=True)
    history["estimated_nav"] = history["previous_nav"] * (
        1 + history["estimated_change_pct"] / 100.0
    )
    history["absolute_error_pct"] = (
        history["estimated_change_pct"] - history["actual_change_pct"]
    ).abs()
    history["direction_correct"] = (
        np.sign(history["estimated_change_pct"])
        == np.sign(history["actual_change_pct"])
    )
    return history


def historical_synthetic_estimate_overlay(
    nav: pd.Series,
    base_target: float = 0.90,
    add_threshold_pct: float = -1.50,
    trim_threshold_pct: float = 3.00,
    tactical_step: float = 0.10,
    minimum_days_between_changes: int = 10,
    long_trend_window: int = 120,
    config: SyntheticEstimateConfig | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Backtestable overlay driven by synthetic intraday estimate history."""
    nav = nav.sort_index().astype(float)
    history = simulate_intraday_estimate_history(nav, config)
    estimated_change = history.set_index("date")["estimated_change_pct"].reindex(nav.index)
    ma_long = nav.rolling(long_trend_window).mean()
    current = float(base_target)
    minimum_target = max(0.0, base_target - tactical_step)
    maximum_target = min(1.0, base_target + tactical_step)
    last_change = -minimum_days_between_changes
    targets = []
    actions = []
    for location, date in enumerate(nav.index):
        estimate = float(estimated_change.loc[date]) if pd.notna(estimated_change.loc[date]) else 0.0
        can_change = location - last_change >= minimum_days_between_changes
        action = "HOLD"
        large_down = estimate <= add_threshold_pct
        large_up = estimate >= trim_threshold_pct
        long_trend_ok = bool(nav.loc[date] >= ma_long.loc[date]) if pd.notna(ma_long.loc[date]) else False
        if can_change and large_down and long_trend_ok and current < maximum_target:
            current = min(maximum_target, current + tactical_step)
            last_change = location
            action = "TACTICAL_ADD"
        elif can_change and large_up and current > minimum_target:
            current = max(minimum_target, current - tactical_step)
            last_change = location
            action = "TACTICAL_TRIM"
        targets.append(current)
        actions.append(action)
    details = history.set_index("date").reindex(nav.index)
    details["target_position"] = targets
    details["action"] = actions
    details.index.name = "date"
    return (
        pd.Series(targets, index=nav.index, name="synthetic_estimate_overlay").clip(0, 1),
        details.reset_index(),
    )
