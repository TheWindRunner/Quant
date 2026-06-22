"""Immutable forward-test snapshots for honest post-research validation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

import pandas as pd

from .execution_policy import apply_open_fund_execution_policy
from .model_zoo import build_model_zoo, robust_ensemble
from .run_research import RESEARCH_FUNDS, build_nav_datasets
from .sector_rotation import RotationConfig, relative_strength_weights


@dataclass(frozen=True)
class ForwardSnapshot:
    generated_at: str
    signal_date: str
    model_version: str
    asset: str
    fund_code: str
    fund_name: str
    nav: float
    ma_position: float
    candidate_position: float
    deployed_position: float
    deployment_accepted: bool
    selected_models: str


@dataclass(frozen=True)
class RotationForwardSnapshot:
    generated_at: str
    signal_date: str
    model_version: str
    rotation_model: str
    candidate_cpo_weight: float
    candidate_memory_weight: float
    candidate_ai_weight: float
    deployed_cpo_weight: float
    deployed_memory_weight: float
    deployed_ai_weight: float
    deployment_accepted: bool


def model_version(paths: tuple[str, ...] = (
    "quant_fund_advisor/model_zoo.py",
    "quant_fund_advisor/experiment.py",
    "quant_fund_advisor/fund_backtest.py",
    "quant_fund_advisor/portfolio_backtest.py",
    "quant_fund_advisor/sector_rotation.py",
    "quant_fund_advisor/execution_policy.py",
)) -> str:
    digest = sha256()
    for path in paths:
        file_path = Path(path)
        digest.update(path.encode("utf-8"))
        digest.update(file_path.read_bytes())
    return digest.hexdigest()[:12]


def create_forward_snapshots(
    selected_models: list[str],
    deployment_accepted: bool,
    generated_at: pd.Timestamp | None = None,
) -> list[ForwardSnapshot]:
    datasets = build_nav_datasets()
    generated_at = generated_at or pd.Timestamp.now(tz="Asia/Shanghai")
    version = model_version()
    rows = []
    for asset, dataset in datasets.items():
        nav = dataset["nav"]
        zoo = {
            model.name: model
            for model in build_model_zoo(
                nav,
                market_nav=dataset.get("market_nav"),
                peers=dataset.get("peers"),
            )
        }
        chosen = [zoo[name] for name in selected_models if name in zoo]
        candidate = robust_ensemble(chosen) if chosen else zoo["buy_hold"]
        ma_position = apply_open_fund_execution_policy(
            zoo["dual_ma"].position
        ).iloc[-1]
        candidate_position = apply_open_fund_execution_policy(
            candidate.position
        ).iloc[-1]
        # Failed models are shadow-tested; production stays at strategic baseline.
        deployed_position = candidate_position if deployment_accepted else 1.0
        definition = RESEARCH_FUNDS[asset]
        rows.append(
            ForwardSnapshot(
                generated_at=generated_at.isoformat(),
                signal_date=nav.index[-1].date().isoformat(),
                model_version=version,
                asset=asset,
                fund_code=definition["code"],
                fund_name=definition["name"],
                nav=float(nav.iloc[-1]),
                ma_position=float(ma_position),
                candidate_position=float(candidate_position),
                deployed_position=float(deployed_position),
                deployment_accepted=deployment_accepted,
                selected_models="|".join(selected_models),
            )
        )
    return rows


def append_forward_ledger(
    snapshots: list[ForwardSnapshot],
    path: str | Path = "output/forward/ledger.csv",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame([asdict(snapshot) for snapshot in snapshots])
    if "fund_code" in incoming:
        incoming["fund_code"] = incoming["fund_code"].astype(str).str.zfill(6)
    if output.exists():
        existing = pd.read_csv(output, dtype={"fund_code": str})
        if "fund_code" in existing:
            existing["fund_code"] = existing["fund_code"].astype(str).str.zfill(6)
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.drop_duplicates(
            ["signal_date", "model_version", "asset"], keep="first"
        )
    else:
        combined = incoming
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    manifest = {
        "model_version": snapshots[0].model_version if snapshots else "",
        "created_at": snapshots[0].generated_at if snapshots else "",
        "selected_models": (
            snapshots[0].selected_models.split("|") if snapshots else []
        ),
        "deployment_accepted": (
            snapshots[0].deployment_accepted if snapshots else False
        ),
    }
    output.with_name("model_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output


def create_rotation_forward_snapshot(
    config: RotationConfig,
    deployment_accepted: bool,
    generated_at: pd.Timestamp | None = None,
) -> RotationForwardSnapshot:
    datasets = build_nav_datasets()
    navs = pd.DataFrame(
        {asset: dataset["nav"] for asset, dataset in datasets.items()}
    ).dropna()
    weights = relative_strength_weights(navs, config).iloc[-1]
    deployed = (
        weights
        if deployment_accepted
        else pd.Series(1.0 / len(navs.columns), index=navs.columns)
    )
    generated_at = generated_at or pd.Timestamp.now(tz="Asia/Shanghai")
    return RotationForwardSnapshot(
        generated_at=generated_at.isoformat(),
        signal_date=navs.index[-1].date().isoformat(),
        model_version=model_version(),
        rotation_model=config.name,
        candidate_cpo_weight=float(weights["cpo_communication"]),
        candidate_memory_weight=float(weights["memory_semiconductor_proxy"]),
        candidate_ai_weight=float(weights["artificial_intelligence"]),
        deployed_cpo_weight=float(deployed["cpo_communication"]),
        deployed_memory_weight=float(deployed["memory_semiconductor_proxy"]),
        deployed_ai_weight=float(deployed["artificial_intelligence"]),
        deployment_accepted=deployment_accepted,
    )


def append_rotation_forward_ledger(
    snapshot: RotationForwardSnapshot,
    path: str | Path = "output/forward/rotation_ledger.csv",
) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    incoming = pd.DataFrame([asdict(snapshot)])
    if output.exists():
        existing = pd.read_csv(output)
        combined = pd.concat([existing, incoming], ignore_index=True)
        combined = combined.drop_duplicates(
            ["signal_date", "model_version", "rotation_model"],
            keep="first",
        )
    else:
        combined = incoming
    combined.to_csv(output, index=False, encoding="utf-8-sig")
    return output


def score_forward_ledger(
    ledger: pd.DataFrame,
    nav_by_code: dict[str, pd.Series],
) -> pd.DataFrame:
    """Calculate realized next-NAV returns without changing old snapshots."""
    result = ledger.copy()
    result["next_nav_return"] = pd.NA
    for index, row in result.iterrows():
        nav = nav_by_code.get(str(row["fund_code"]).zfill(6))
        if nav is None:
            continue
        future = nav.loc[nav.index > pd.Timestamp(row["signal_date"])]
        if future.empty:
            continue
        result.loc[index, "next_nav_return"] = (
            float(future.iloc[0]) / float(row["nav"]) - 1
        )
    return result


def score_rotation_forward_ledger(
    ledger: pd.DataFrame,
    nav_by_asset: dict[str, pd.Series],
) -> pd.DataFrame:
    """Score candidate and deployed allocation on the next common NAV date."""
    result = ledger.copy()
    result["next_nav_date"] = pd.NA
    result["candidate_next_return"] = pd.NA
    result["deployed_next_return"] = pd.NA
    asset_columns = {
        "cpo": "cpo_communication",
        "memory": "memory_semiconductor_proxy",
        "ai": "artificial_intelligence",
    }
    for row_index, row in result.iterrows():
        signal_date = pd.Timestamp(row["signal_date"])
        future_dates = None
        for asset in asset_columns.values():
            nav = nav_by_asset.get(asset)
            if nav is None:
                future_dates = set()
                break
            available = set(nav.index[nav.index > signal_date])
            future_dates = (
                available
                if future_dates is None
                else future_dates.intersection(available)
            )
        if not future_dates:
            continue
        next_date = min(future_dates)
        candidate_return = 0.0
        deployed_return = 0.0
        for prefix, asset in asset_columns.items():
            nav = nav_by_asset[asset].sort_index()
            previous = nav.loc[nav.index <= signal_date].iloc[-1]
            asset_return = float(nav.loc[next_date] / previous - 1)
            candidate_return += (
                float(row[f"candidate_{prefix}_weight"]) * asset_return
            )
            deployed_return += (
                float(row[f"deployed_{prefix}_weight"]) * asset_return
            )
        result.loc[row_index, "next_nav_date"] = next_date.date().isoformat()
        result.loc[row_index, "candidate_next_return"] = candidate_return
        result.loc[row_index, "deployed_next_return"] = deployed_return
    return result
