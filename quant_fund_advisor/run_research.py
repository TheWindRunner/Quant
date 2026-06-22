"""Run the fee-aware, leakage-controlled six-month research protocol."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .data import load_or_fetch_open_fund_nav
from .experiment import (
    evaluate_locked_selected_ensemble,
    evaluate_locked_theme_models,
    robustness_summary,
    select_ensemble_on_training,
    select_theme_models_on_training,
)
from .nav_validation import validate_nav_cache
from .fund_backtest import backtest_open_fund
from .intraday_estimate import (
    historical_extreme_move_overlay,
    historical_synthetic_estimate_overlay,
)
from .overfitting import probability_of_backtest_overfitting
from .public_correlation import build_public_correlation_report
from .sector_rotation import (
    equal_weight_targets,
    evaluate_rotation,
    fixed_strategy_fold_validation,
    locked_rotation_comparison,
    risk_budget_momentum_weights,
    select_rotation_on_training,
    walk_forward_rotation_validation,
)
from .stress import (
    bootstrap_model_comparison,
    moving_average_sensitivity,
    summarize_bootstrap,
)
from .validation import model_acceptance
from .validation import purged_training_folds


RESEARCH_FUNDS = {
    "cpo_communication": {
        "code": "007817",
        "name": "国泰中证全指通信设备ETF联接A",
    },
    "memory_semiconductor_proxy": {
        "code": "008887",
        "name": "华夏国证半导体芯片ETF联接A",
    },
    "artificial_intelligence": {
        "code": "008585",
        "name": "华夏中证人工智能主题ETF联接A",
    },
}


def build_nav_datasets() -> dict[str, dict[str, pd.Series | pd.DataFrame | None]]:
    navs = {
        asset: load_or_fetch_open_fund_nav(definition["code"])
        for asset, definition in RESEARCH_FUNDS.items()
    }
    common_index = pd.DatetimeIndex(
        sorted(set.intersection(*(set(series.index) for series in navs.values())))
    )
    peer_frame = pd.DataFrame(
        {asset: series.reindex(common_index) for asset, series in navs.items()}
    )
    market_nav = peer_frame.mean(axis=1)
    return {
        asset: {
            "nav": nav.reindex(common_index).ffill(),
            "market_nav": market_nav,
            "peers": peer_frame.drop(columns=[asset]),
        }
        for asset, nav in navs.items()
    }


def run_research(output_dir: str | Path = "output/research") -> dict[str, object]:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    datasets = build_nav_datasets()
    validation = validate_nav_cache(
        "data/nav_cache",
        tuple(definition["code"] for definition in RESEARCH_FUNDS.values()),
    )
    if not validation["valid"].fillna(False).all():
        raise RuntimeError(
            "NAV cache failed integrity checks:\n"
            + validation.to_string(index=False)
        )

    latest = min(dataset["nav"].index.max() for dataset in datasets.values())
    test_start = latest - pd.DateOffset(months=6)
    selected, training = select_ensemble_on_training(datasets, test_start)
    locked = evaluate_locked_selected_ensemble(
        datasets, selected, test_start, latest
    )
    summary = robustness_summary(locked)
    comparable = locked.copy()
    comparable["strategy"] = comparable["model"].replace(
        {
            "buy_hold": "buy_hold",
            "dual_ma": "ma20_60",
            "trained_ensemble": "multifactor",
            "no_qualified_model": "multifactor",
        }
    )
    acceptance = model_acceptance(comparable)

    theme_models, theme_training = select_theme_models_on_training(
        datasets, test_start
    )
    theme_locked = evaluate_locked_theme_models(
        datasets, theme_models, test_start, latest
    )
    theme_comparable = theme_locked.copy()
    theme_comparable["strategy"] = theme_comparable["model"].replace(
        {
            "buy_hold": "buy_hold",
            "dual_ma": "ma20_60",
            "theme_selected": "multifactor",
        }
    )
    theme_acceptance = model_acceptance(theme_comparable)

    nav_frame = pd.DataFrame(
        {asset: dataset["nav"] for asset, dataset in datasets.items()}
    ).dropna()
    rotation_model, rotation_training = select_rotation_on_training(
        nav_frame, test_start
    )
    rotation_locked = locked_rotation_comparison(
        nav_frame, rotation_model, test_start, latest
    )
    rotation_walk_forward = walk_forward_rotation_validation(
        nav_frame, test_start
    )
    risk_budget_targets = risk_budget_momentum_weights(nav_frame)
    risk_budget_locked = pd.DataFrame(
        [
            {
                "model": "equal_weight_buy_hold",
                **evaluate_rotation(
                    nav_frame,
                    equal_weight_targets(nav_frame),
                    test_start,
                    latest,
                ),
            },
            {
                "model": "risk_budget_momentum",
                **evaluate_rotation(
                    nav_frame,
                    risk_budget_targets,
                    test_start,
                    latest,
                ),
            },
        ]
    )
    risk_budget_walk_forward = fixed_strategy_fold_validation(
        nav_frame,
        risk_budget_targets,
        test_start,
        "risk_budget_momentum",
    )
    risk_budget_acceptance = {
        "accepted": bool(
            float(risk_budget_locked.iloc[1]["total_return"])
            > float(risk_budget_locked.iloc[0]["total_return"])
            and (risk_budget_walk_forward["excess_return"] > 0).mean() >= 0.60
            and risk_budget_walk_forward["excess_return"].median() >= 0
        ),
        "locked_excess_return": float(
            risk_budget_locked.iloc[1]["total_return"]
            - risk_budget_locked.iloc[0]["total_return"]
        ),
        "walk_forward_beat_ratio": float(
            (risk_budget_walk_forward["excess_return"] > 0).mean()
        ),
        "walk_forward_median_excess": float(
            risk_budget_walk_forward["excess_return"].median()
        ),
        "walk_forward_median_drawdown_improvement": float(
            risk_budget_walk_forward["drawdown_improvement"].median()
        ),
    }

    estimate_rows = []
    estimate_training_rows = []
    synthetic_estimate_rows = []
    synthetic_estimate_training_rows = []
    synthetic_estimate_histories = []
    for asset, dataset in datasets.items():
        nav = dataset["nav"]
        estimate_target = historical_extreme_move_overlay(
            nav,
            base_target=1.0,
        )
        synthetic_target, synthetic_history = historical_synthetic_estimate_overlay(
            nav,
            base_target=0.90,
        )
        buy_hold_target = pd.Series(1.0, index=nav.index)

        def evaluate_target(
            target: pd.Series,
            start: pd.Timestamp,
            end: pd.Timestamp,
        ) -> dict[str, float]:
            sliced_nav = nav.loc[(nav.index >= start) & (nav.index <= end)]
            prior = target.loc[target.index < start]
            initial = float(prior.iloc[-1]) if len(prior) else 0.0
            return backtest_open_fund(
                sliced_nav,
                target.reindex(sliced_nav.index),
                initial_target=initial,
                liquidate_at_end=True,
            )["metrics"]

        locked_benchmark = evaluate_target(buy_hold_target, test_start, latest)
        locked_candidate = evaluate_target(estimate_target, test_start, latest)
        locked_synthetic = evaluate_target(synthetic_target, test_start, latest)
        estimate_rows.extend(
            [
                {
                    "asset": asset,
                    "model": "buy_hold",
                    **locked_benchmark,
                },
                {
                    "asset": asset,
                    "model": "extreme_estimate_overlay",
                    **locked_candidate,
                },
            ]
        )
        synthetic_estimate_rows.extend(
            [
                {
                    "asset": asset,
                    "model": "buy_hold",
                    **locked_benchmark,
                },
                {
                    "asset": asset,
                    "model": "synthetic_estimate_overlay",
                    **locked_synthetic,
                },
            ]
        )
        synthetic_history["asset"] = asset
        synthetic_estimate_histories.append(synthetic_history)
        for fold_start, fold_end in purged_training_folds(
            nav.index, test_start
        ):
            benchmark = evaluate_target(
                buy_hold_target, fold_start, fold_end
            )
            candidate = evaluate_target(
                estimate_target, fold_start, fold_end
            )
            estimate_training_rows.append(
                {
                    "asset": asset,
                    "fold_start": fold_start,
                    "fold_end": fold_end,
                    "excess_return": (
                        candidate["total_return"] - benchmark["total_return"]
                    ),
                    "drawdown_improvement": (
                        candidate["max_drawdown"] - benchmark["max_drawdown"]
                    ),
                    **candidate,
                }
            )
            synthetic_candidate = evaluate_target(
                synthetic_target, fold_start, fold_end
            )
            synthetic_estimate_training_rows.append(
                {
                    "asset": asset,
                    "fold_start": fold_start,
                    "fold_end": fold_end,
                    "excess_return": (
                        synthetic_candidate["total_return"] - benchmark["total_return"]
                    ),
                    "drawdown_improvement": (
                        synthetic_candidate["max_drawdown"] - benchmark["max_drawdown"]
                    ),
                    **synthetic_candidate,
                }
            )
    estimate_locked = pd.DataFrame(estimate_rows)
    estimate_training = pd.DataFrame(estimate_training_rows)
    synthetic_estimate_locked = pd.DataFrame(synthetic_estimate_rows)
    synthetic_estimate_training = pd.DataFrame(synthetic_estimate_training_rows)
    synthetic_estimate_history = pd.concat(
        synthetic_estimate_histories, ignore_index=True
    )
    estimate_pivot = estimate_locked.pivot(
        index="asset", columns="model", values="total_return"
    )
    estimate_excess = (
        estimate_pivot["extreme_estimate_overlay"]
        - estimate_pivot["buy_hold"]
    )
    estimate_acceptance = {
        "accepted": bool(
            (estimate_excess > 0).mean() >= 0.60
            and (estimate_training["excess_return"] > 0).mean() >= 0.60
            and estimate_training["excess_return"].median() >= 0
        ),
        "locked_assets_beating_buy_hold": int((estimate_excess > 0).sum()),
        "locked_asset_count": int(len(estimate_excess)),
        "training_beat_ratio": float(
            (estimate_training["excess_return"] > 0).mean()
        ),
        "training_median_excess": float(
            estimate_training["excess_return"].median()
        ),
        "training_median_drawdown_improvement": float(
            estimate_training["drawdown_improvement"].median()
        ),
    }
    synthetic_estimate_pivot = synthetic_estimate_locked.pivot(
        index="asset", columns="model", values="total_return"
    )
    synthetic_estimate_excess = (
        synthetic_estimate_pivot["synthetic_estimate_overlay"]
        - synthetic_estimate_pivot["buy_hold"]
    )
    synthetic_estimate_acceptance = {
        "accepted": bool(
            (synthetic_estimate_excess > 0).mean() >= 0.60
            and (synthetic_estimate_training["excess_return"] > 0).mean() >= 0.60
            and synthetic_estimate_training["excess_return"].median() >= 0
        ),
        "locked_assets_beating_buy_hold": int((synthetic_estimate_excess > 0).sum()),
        "locked_asset_count": int(len(synthetic_estimate_excess)),
        "training_beat_ratio": float(
            (synthetic_estimate_training["excess_return"] > 0).mean()
        ),
        "training_median_excess": float(
            synthetic_estimate_training["excess_return"].median()
        ),
        "training_median_drawdown_improvement": float(
            synthetic_estimate_training["drawdown_improvement"].median()
        ),
    }
    rotation_excess = (
        float(rotation_locked.iloc[1]["total_return"])
        - float(rotation_locked.iloc[0]["total_return"])
    )
    rotation_acceptance = {
        "accepted": bool(
            rotation_excess > 0
            and len(rotation_walk_forward) >= 6
            and (rotation_walk_forward["excess_return"] > 0).mean() >= 0.50
            and rotation_walk_forward["excess_return"].median() >= 0
        ),
        "locked_excess_return": rotation_excess,
        "walk_forward_beat_ratio": float(
            (rotation_walk_forward["excess_return"] > 0).mean()
        ),
        "walk_forward_median_excess": float(
            rotation_walk_forward["excess_return"].median()
        ),
        "walk_forward_worst_excess": float(
            rotation_walk_forward["excess_return"].min()
        ),
    }
    model_pbo = probability_of_backtest_overfitting(training)
    rotation_pbo = probability_of_backtest_overfitting(rotation_training)

    cpo_nav = datasets["cpo_communication"]["nav"]
    sensitivity = moving_average_sensitivity(cpo_nav, test_start, latest)
    bootstrap_raw = bootstrap_model_comparison(
        cpo_nav.loc[cpo_nav.index < test_start],
        simulations=200,
        block_size=10,
    )
    bootstrap_summary = summarize_bootstrap(bootstrap_raw)
    public_correlations = build_public_correlation_report(output_dir=output)

    validation.to_csv(
        output / "nav_validation.csv", index=False, encoding="utf-8-sig"
    )
    training.to_csv(
        output / "training_folds.csv", index=False, encoding="utf-8-sig"
    )
    locked.to_csv(
        output / "locked_six_month_test.csv",
        index=False,
        encoding="utf-8-sig",
    )
    summary.to_csv(output / "robustness_summary.csv", encoding="utf-8-sig")
    pd.Series(selected, name="model").to_csv(
        output / "selected_models.csv", index=False, encoding="utf-8-sig"
    )
    theme_training.to_csv(
        output / "theme_training_selection.csv",
        index=False,
        encoding="utf-8-sig",
    )
    theme_locked.to_csv(
        output / "theme_locked_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rotation_training.to_csv(
        output / "rotation_training_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rotation_locked.to_csv(
        output / "rotation_locked_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    rotation_walk_forward.to_csv(
        output / "rotation_walk_forward.csv",
        index=False,
        encoding="utf-8-sig",
    )
    risk_budget_locked.to_csv(
        output / "risk_budget_locked_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    risk_budget_walk_forward.to_csv(
        output / "risk_budget_walk_forward.csv",
        index=False,
        encoding="utf-8-sig",
    )
    estimate_locked.to_csv(
        output / "estimate_overlay_locked_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    estimate_training.to_csv(
        output / "estimate_overlay_training_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    synthetic_estimate_locked.to_csv(
        output / "synthetic_estimate_overlay_locked_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )
    synthetic_estimate_training.to_csv(
        output / "synthetic_estimate_overlay_training_folds.csv",
        index=False,
        encoding="utf-8-sig",
    )
    synthetic_estimate_history.to_csv(
        output / "synthetic_estimate_history.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(
        [
            {"model_group": "single_fund_zoo", **model_pbo},
            {"model_group": "sector_rotation", **rotation_pbo},
        ]
    ).to_json(
        output / "overfitting_diagnostics.json",
        orient="records",
        force_ascii=False,
        indent=2,
    )
    sensitivity.to_csv(
        output / "cpo_ma_sensitivity.csv", index=False, encoding="utf-8-sig"
    )
    bootstrap_raw.to_csv(
        output / "cpo_training_bootstrap.csv",
        index=False,
        encoding="utf-8-sig",
    )
    bootstrap_summary.to_csv(
        output / "cpo_bootstrap_summary.csv", encoding="utf-8-sig"
    )

    report_path = _write_report(
        output,
        test_start,
        latest,
        selected,
        validation,
        locked,
        summary,
        acceptance,
        theme_models,
        theme_locked,
        theme_acceptance,
        rotation_model.name if rotation_model else "无合格轮动模型",
        rotation_locked,
        rotation_walk_forward,
        rotation_acceptance,
        risk_budget_locked,
        risk_budget_acceptance,
        estimate_locked,
        estimate_acceptance,
        synthetic_estimate_locked,
        synthetic_estimate_acceptance,
        model_pbo,
        rotation_pbo,
        public_correlations,
        sensitivity,
        bootstrap_summary,
    )
    return {
        "test_start": test_start,
        "test_end": latest,
        "selected_models": selected,
        "locked_results": locked,
        "theme_models": theme_models,
        "theme_results": theme_locked,
        "rotation_model": rotation_model,
        "rotation_results": rotation_locked,
        "acceptance": acceptance,
        "theme_acceptance": theme_acceptance,
        "rotation_acceptance": rotation_acceptance,
        "risk_budget_results": risk_budget_locked,
        "risk_budget_acceptance": risk_budget_acceptance,
        "estimate_overlay_results": estimate_locked,
        "estimate_overlay_acceptance": estimate_acceptance,
        "synthetic_estimate_overlay_results": synthetic_estimate_locked,
        "synthetic_estimate_overlay_acceptance": synthetic_estimate_acceptance,
        "model_pbo": model_pbo,
        "rotation_pbo": rotation_pbo,
        "public_correlations": public_correlations,
        "report_path": report_path,
    }


def _format_percent_columns(frame: pd.DataFrame) -> pd.DataFrame:
    display = frame.copy()
    for column in (
        "total_return",
        "cagr",
        "annual_volatility",
        "max_drawdown",
    ):
        if column in display:
            display[column] = display[column].map(lambda value: f"{value:.2%}")
    return display


def _write_report(
    output: Path,
    test_start: pd.Timestamp,
    test_end: pd.Timestamp,
    selected: list[str],
    validation: pd.DataFrame,
    locked: pd.DataFrame,
    summary: pd.DataFrame,
    acceptance: dict[str, object],
    theme_models: dict[str, str],
    theme_locked: pd.DataFrame,
    theme_acceptance: dict[str, object],
    rotation_model: str,
    rotation_locked: pd.DataFrame,
    rotation_walk_forward: pd.DataFrame,
    rotation_acceptance: dict[str, object],
    risk_budget_locked: pd.DataFrame,
    risk_budget_acceptance: dict[str, object],
    estimate_locked: pd.DataFrame,
    estimate_acceptance: dict[str, object],
    synthetic_estimate_locked: pd.DataFrame,
    synthetic_estimate_acceptance: dict[str, object],
    model_pbo: dict[str, object],
    rotation_pbo: dict[str, object],
    public_correlations: pd.DataFrame,
    sensitivity: pd.DataFrame,
    bootstrap_summary: pd.DataFrame,
) -> Path:
    walk_forward_summary = {
        "区间数": len(rotation_walk_forward),
        "跑赢比例": (
            float((rotation_walk_forward["excess_return"] > 0).mean())
            if len(rotation_walk_forward)
            else 0.0
        ),
        "超额收益中位数": (
            float(rotation_walk_forward["excess_return"].median())
            if len(rotation_walk_forward)
            else 0.0
        ),
        "最差区间超额": (
            float(rotation_walk_forward["excess_return"].min())
            if len(rotation_walk_forward)
            else 0.0
        ),
    }
    lines = [
        "# 科技主题基金量化模型研究报告",
        "",
        f"研究区间：{test_start.date()} 至 {test_end.date()}",
        "",
        "> 第一轮锁定测试已被查看，后续同区间结果属于迭代对照；"
        "真正未见样本由每日前视台账积累。",
        "",
        "## 数据完整性",
        "",
        "```text",
        validation.to_string(index=False),
        "```",
        "",
        "## 全市场统一模型",
        "",
        f"训练期入选：{', '.join(selected) if selected else '无'}",
        "",
        "```text",
        _format_percent_columns(locked).to_string(index=False),
        "```",
        "",
        f"部署验收：{acceptance}",
        "",
        "## 逐主题训练选择",
        "",
        f"训练期选择：{theme_models}",
        "",
        "```text",
        _format_percent_columns(theme_locked).to_string(index=False),
        "```",
        "",
        f"部署验收：{theme_acceptance}",
        "",
        "## 科技主题组合轮动",
        "",
        f"训练期选择：{rotation_model}",
        "",
        "```text",
        _format_percent_columns(rotation_locked).to_string(index=False),
        "```",
        "",
        f"历史模拟实盘摘要：{walk_forward_summary}",
        "",
        f"部署验收：{rotation_acceptance}",
        "",
        "## 固定风险预算模型",
        "",
        "规则：每20个净值日按60日逆波动率配置，"
        "叠加固定幅度的60日相对动量倾斜，单主题上限60%，"
        "长期趋势恶化时组合暴露降至50%。",
        "",
        "```text",
        _format_percent_columns(risk_budget_locked).to_string(index=False),
        "```",
        "",
        f"部署验收：{risk_budget_acceptance}",
        "",
        "## 极端估值战术覆盖层",
        "",
        "这是对14:30估值逻辑的乐观上限：默认100%核心仓，"
        "单日大涨后降至90%，之后出现明确大跌且长期趋势未破时恢复100%；"
        "每次变化至少间隔10个净值日，并按下一净值日成交。"
        "历史测试使用最终净值方向，真实估值只会更不准确。",
        "",
        "```text",
        _format_percent_columns(estimate_locked).to_string(index=False),
        "```",
        "",
        f"部署验收：{estimate_acceptance}",
        "",
        "## 合成14:30估值覆盖层",
        "",
        "规则：用真实次日净值涨跌幅生成一个合成盘中估值。若真实涨跌幅为 6%，"
        "则估值涨跌幅在 4.5% 到 7.5% 区间内随机扰动；小涨小跌时允许方向翻转。"
        "再按 14:30 的战术规则回测：大跌加 10%，大涨减 10%，其余不动。",
        "",
        "```text",
        _format_percent_columns(synthetic_estimate_locked).to_string(index=False),
        "```",
        "",
        f"部署验收：{synthetic_estimate_acceptance}",
        "",
        "## 过拟合风险",
        "",
        f"单基金模型库PBO：{model_pbo}",
        "",
        f"组合轮动PBO：{rotation_pbo}",
        "",
        "## 中美短窗口相关性",
        "",
        "公开美股数据仅覆盖2026-05-15至2026-06-12。"
        "所有关系均为低置信，只作盘中方向确认，不进入模型训练。",
        "",
        "```text",
        public_correlations.loc[
            public_correlations["us_lead_trading_closes"] == 1
        ].to_string(index=False),
        "```",
        "",
        "## 跨基金稳健性",
        "",
        "```text",
        summary.to_string(),
        "```",
        "",
        "## CPO均线敏感性",
        "",
        "```text",
        sensitivity.to_string(index=False),
        "```",
        "",
        "## CPO训练期区块自助法",
        "",
        "```text",
        bootstrap_summary.to_string(),
        "```",
        "",
        "## 结论",
        "",
        "- 单基金择时与组合轮动均未通过严格部署验收。",
        "- 存储代理的核心仓加岭回归战术仓可继续影子跟踪。",
        "- 20日相对强弱轮动在近半年迭代对照中表现较好，"
        "但历史模拟实盘未证明稳定超额。",
        "- 实盘建议必须结合基金估值置信度、市场新闻、"
        "持有批次与个人组合上限，不得机械照搬回测。",
        "",
        "所有策略均按下一净值日执行，并计入申购费、"
        "按持有期限分档的FIFO赎回费。",
    ]
    path = output / "research_report.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


if __name__ == "__main__":
    result = run_research()
    print(result["locked_results"].to_string(index=False))
    print("\nSelected:", ", ".join(result["selected_models"]))
