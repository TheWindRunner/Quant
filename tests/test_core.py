import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_fund_advisor.backtest import BacktestConfig, run_backtest
from quant_fund_advisor.correlation import (
    best_lead_relationships,
    lagged_sector_correlations,
)
from quant_fund_advisor.data import (
    fetch_manifest_prices,
    fetch_open_fund_nav_eastmoney,
    normalize_history,
)
from quant_fund_advisor.daily_strategy import (
    backtest_daily,
    daily_features,
    compare_fund_strategies,
    generate_positions,
    lagged_daily_correlation,
)
from quant_fund_advisor.demo import make_demo_prices
from quant_fund_advisor.model import latest_recommendations, score_assets
from quant_fund_advisor.news import NewsItem, aggregate_news
from quant_fund_advisor.multifactor import (
    FundStructure,
    compute_factor_panel,
    factor_contributions,
    score_factor_panel,
    point_in_time_structure_score,
    structure_score,
)
from quant_fund_advisor.selector import rank_funds
from quant_fund_advisor.validation import (
    candidate_configs,
    locked_test_comparison,
    model_acceptance,
    purged_training_folds,
    select_config_on_training,
)
from quant_fund_advisor.experiment import (
    evaluate_locked_selected_ensemble,
    evaluate_locked_theme_models,
    evaluate_model_zoo,
    robustness_summary,
    select_ensemble_on_training,
    select_theme_models_on_training,
)
from quant_fund_advisor.model_zoo import (
    bull_hold_bear_defense,
    build_model_zoo,
    core_tactical,
    crowding_aware_momentum,
    hysteresis_trend,
    market_state_nowcast,
    price_sentiment_regime,
    regime_relative_strength,
    tail_risk_overlay,
    walk_forward_ridge,
)
from quant_fund_advisor.nav_validation import validate_nav_series
from quant_fund_advisor.fund_backtest import (
    FundFeeSchedule,
    RedemptionTier,
    backtest_open_fund,
    redemption_fee_rate,
)
from quant_fund_advisor.stress import (
    block_bootstrap_nav,
    moving_average_sensitivity,
    summarize_bootstrap,
)
from quant_fund_advisor.execution_policy import apply_open_fund_execution_policy
from quant_fund_advisor.action_report import _estimate_comment
from quant_fund_advisor.intraday_estimate import (
    adaptive_policy_from_history,
    apply_intraday_overlay,
    calibration_metrics,
    estimate_confidence,
    fetch_akshare_fund_estimate,
    fetch_akshare_etf_iopv,
    historical_extreme_move_overlay,
    historical_synthetic_estimate_overlay,
    reconcile_estimate_history,
    simulate_intraday_estimate_history,
    SyntheticEstimateConfig,
)
from quant_fund_advisor.forward_test import (
    RotationForwardSnapshot,
    append_rotation_forward_ledger,
    score_forward_ledger,
    score_rotation_forward_ledger,
)
from quant_fund_advisor.fund_metadata import fetch_fund_structure_history
from quant_fund_advisor.portfolio_backtest import backtest_open_fund_portfolio
from quant_fund_advisor.sector_rotation import (
    RotationConfig,
    equal_weight_targets,
    fixed_strategy_fold_validation,
    locked_rotation_comparison,
    portfolio_tail_risk_exposure,
    relative_strength_weights,
    risk_budget_momentum_weights,
    select_rotation_on_training,
    walk_forward_rotation_validation,
)
from quant_fund_advisor.overfitting import probability_of_backtest_overfitting
from quant_fund_advisor.public_correlation import (
    correlation_with_lags,
    fisher_confidence_interval,
)
from quant_fund_advisor.strategy_curves import build_all_in_t_strategy
from quant_fund_advisor.expanded_research import AllInTConfig, all_in_t_overlay


class CoreTests(unittest.TestCase):
    def test_live_history_normalizes_chinese_columns(self):
        raw = pd.DataFrame(
            {
                "\u65e5\u671f": ["2026-01-02", "2026-01-05"],
                "\u5f00\u76d8": ["1.0", "1.1"],
                "\u6536\u76d8": ["1.1", "1.2"],
                "\u6210\u4ea4\u91cf": ["100", "120"],
            }
        )
        result = normalize_history(raw, "159995")
        self.assertEqual(list(result.columns), ["open", "close", "volume"])
        self.assertEqual(result.index.name, "date")
        self.assertEqual(result.iloc[-1]["close"], 1.2)

    def test_manifest_prices_use_sector_names(self):
        manifest = pd.DataFrame(
            [
                {
                    "group": "cn",
                    "sector": "technology",
                    "asset_type": "cn_etf",
                    "symbol": "159995",
                    "adjust": "qfq",
                }
            ]
        )
        history = pd.DataFrame(
            {"close": [1.0, 1.1]},
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )
        with patch("quant_fund_advisor.data.fetch_history", return_value=history):
            result = fetch_manifest_prices(manifest, "cn")
        self.assertEqual(list(result.columns), ["technology"])

    @patch("quant_fund_advisor.data.requests.get", create=True)
    def test_eastmoney_nav_fallback_parses_full_series(self, mock_get):
        first = int(
            pd.Timestamp("2026-01-02", tz="Asia/Shanghai")
            .tz_convert("UTC")
            .timestamp()
            * 1000
        )
        second = int(
            pd.Timestamp("2026-01-05", tz="Asia/Shanghai")
            .tz_convert("UTC")
            .timestamp()
            * 1000
        )
        response = mock_get.return_value
        response.text = (
            f'var Data_netWorthTrend = [{{"x":{first},"y":1.1}},'
            f'{{"x":{second},"y":1.2}}];'
        )
        response.raise_for_status.return_value = None
        result = fetch_open_fund_nav_eastmoney("007817")
        self.assertEqual(result.tolist(), [1.1, 1.2])
        self.assertEqual(
            result.index.strftime("%Y-%m-%d").tolist(),
            ["2026-01-02", "2026-01-05"],
        )

    def test_lag_detection_has_valid_output(self):
        us, cn = make_demo_prices(periods=300)
        result = best_lead_relationships(lagged_sector_correlations(us, cn))
        self.assertEqual(set(result["sector"]), set(us.columns))
        self.assertTrue(result["us_lead_days"].between(0, 5).all())

    def test_scores_and_backtest_are_finite(self):
        _, cn = make_demo_prices(periods=400)
        scores = score_assets(cn)
        recs = latest_recommendations(scores)
        result = run_backtest(cn, scores, BacktestConfig(top_n=2))
        self.assertTrue(np.isfinite(recs["score"]).all())
        self.assertTrue(np.isfinite(result["equity"].to_numpy()).all())
        self.assertLessEqual(result["weights"].sum(axis=1).max(), 1.000001)
        self.assertLessEqual(result["weights"].max().max(), 0.400001)

    def test_news_source_and_decay(self):
        now = pd.Timestamp("2026-06-15", tz="UTC").to_pydatetime()
        items = [
            NewsItem(now, "芯片行业盈利超预期并获政策支持", source="财联社"),
            NewsItem(now, "地产风险上升", source="其他"),
        ]
        scores = aggregate_news(items, as_of=now)
        self.assertGreater(scores.loc["semiconductor", "news_score"], 0)
        self.assertLess(scores.loc["real_estate", "news_score"], 0)

    def test_fund_ranking_uses_only_alipay_whitelist(self):
        dates = pd.bdate_range("2025-01-01", periods=100)
        nav = pd.DataFrame(
            {
                "000001": np.linspace(1.0, 1.3, len(dates)),
                "000002": np.linspace(1.0, 1.1, len(dates)),
            },
            index=dates,
        )
        universe = pd.DataFrame(
            [
                {
                    "fund_code": "000001",
                    "fund_name": "eligible",
                    "sector": "technology",
                    "available_on_alipay": True,
                    "purchase_fee": 0.001,
                    "redemption_fee": 0.005,
                    "min_holding_days": 7,
                },
                {
                    "fund_code": "000002",
                    "fund_name": "excluded",
                    "sector": "technology",
                    "available_on_alipay": False,
                    "purchase_fee": 0.0,
                    "redemption_fee": 0.0,
                    "min_holding_days": 0,
                },
            ]
        )
        sectors = pd.DataFrame(
            {"score": [0.8], "action": ["BUY"]}, index=["technology"]
        )
        ranked = rank_funds(universe, nav, sectors)
        self.assertEqual(ranked["fund_code"].tolist(), ["000001"])

    def test_daily_strategy_uses_next_day_execution(self):
        dates = pd.bdate_range("2025-01-01", periods=160)
        nav = pd.Series(np.linspace(1.0, 1.8, len(dates)), index=dates)
        us_nav = pd.Series(np.linspace(1.0, 1.5, len(dates)), index=dates)
        features = daily_features(nav, us_nav)
        decisions = generate_positions(features)
        result = backtest_daily(nav, decisions, start=dates[-120])
        first_signal = decisions.index[decisions["target_position"] > 0][0]
        self.assertEqual(result["position"].loc[first_signal], 0.0)
        self.assertGreater(result["metrics"]["trade_count"], 0)

    def test_daily_lagged_correlation(self):
        dates = pd.bdate_range("2025-01-01", periods=180)
        rng = np.random.default_rng(7)
        us_returns = rng.normal(0, 0.01, len(dates))
        cn_returns = np.roll(us_returns, 1)
        cn_returns[0] = 0
        us_nav = pd.Series((1 + us_returns).cumprod(), index=dates)
        cn_nav = pd.Series((1 + cn_returns).cumprod(), index=dates)
        result = lagged_daily_correlation(cn_nav, us_nav)
        best = result.loc[result["correlation"].idxmax()]
        self.assertEqual(best["us_lead_days"], 1)

    def test_public_correlation_uses_only_prior_us_close_for_causal_lag(self):
        dates = pd.bdate_range("2026-01-01", periods=80)
        rng = np.random.default_rng(41)
        us_return = rng.normal(0, 0.01, len(dates))
        cn_return = np.r_[0.0, us_return[:-1]]
        us_nav = pd.Series((1 + us_return).cumprod(), index=dates)
        cn_nav = pd.Series((1 + cn_return).cumprod(), index=dates)
        result = correlation_with_lags(cn_nav, us_nav, maximum_lead=2)
        causal = result.loc[result["us_lead_trading_closes"] == 1].iloc[0]
        self.assertGreater(causal["correlation"], 0.95)
        lower, upper = fisher_confidence_interval(0.5, 80)
        self.assertLess(lower, 0.5)
        self.assertGreater(upper, 0.5)

    def test_strategy_comparison_has_three_methods(self):
        dates = pd.bdate_range("2025-01-01", periods=180)
        nav = pd.Series(np.linspace(1.0, 1.7, len(dates)), index=dates)
        us_nav = pd.Series(np.linspace(1.0, 1.4, len(dates)), index=dates)
        result = compare_fund_strategies(nav, us_nav, start=dates[-120])
        self.assertEqual(
            set(result.index), {"buy_hold", "ma20_60", "daily_fund_strategy"}
        )

    def test_multifactor_panel_is_bounded_and_explainable(self):
        dates = pd.bdate_range("2024-01-01", periods=320)
        rng = np.random.default_rng(17)
        nav = pd.Series(
            (1 + rng.normal(0.0005, 0.012, len(dates))).cumprod(), index=dates
        )
        constituents = pd.DataFrame(
            {
                "a": (1 + rng.normal(0.0006, 0.015, len(dates))).cumprod(),
                "b": (1 + rng.normal(0.0004, 0.014, len(dates))).cumprod(),
                "c": (1 + rng.normal(0.0003, 0.013, len(dates))).cumprod(),
            },
            index=dates,
        )
        structure = FundStructure(
            scale_billion_cny=2,
            holder_count=50_000,
            top10_concentration=0.60,
            tracking_error=0.012,
        )
        panel = compute_factor_panel(nav, constituents, structure=structure)
        scored = score_factor_panel(panel)
        contributions = factor_contributions(scored)
        self.assertTrue(scored["score"].between(-1, 1).all())
        self.assertLessEqual(scored["target_position"].max(), 0.20)
        self.assertEqual(set(contributions.columns), {
            "trend", "momentum", "quality", "breadth", "cross_market",
            "risk_appetite", "news", "flow", "structure"
        })

    def test_extreme_fund_structure_is_penalized(self):
        good, _ = structure_score(
            FundStructure(
                scale_billion_cny=2,
                holder_count=20_000,
                institution_ratio=0.3,
                top10_concentration=0.5,
                quarterly_share_growth=0.1,
                tracking_error=0.01,
            )
        )
        crowded, _ = structure_score(
            FundStructure(
                scale_billion_cny=20,
                holder_count=800_000,
                institution_ratio=0.9,
                top10_concentration=0.85,
                quarterly_share_growth=1.5,
                tracking_error=0.05,
            )
        )
        self.assertGreater(good, crowded)

    def test_structure_history_respects_publication_lag(self):
        index = pd.date_range("2025-03-01", "2025-07-15", freq="D")
        history = pd.DataFrame(
            {
                "report_date": [pd.Timestamp("2025-03-31")],
                "scale_billion_cny": [2.0],
                "holder_count": [20_000],
            }
        )
        score = point_in_time_structure_score(history, index, publication_lag_days=90)
        self.assertEqual(score.loc["2025-06-28"], 0.0)
        self.assertNotEqual(score.loc["2025-06-29"], 0.0)

    @patch("quant_fund_advisor.fund_metadata._read_f10_tables")
    def test_fund_scale_is_converted_from_100m_to_billion(self, mock_tables):
        scale = pd.DataFrame(
            {
                "截止日期": ["2026-03-31"],
                "净资产（亿元）": ["9.52亿元"],
                "基金份额（亿份）": ["8.00亿份"],
            }
        )
        holders = pd.DataFrame(
            {
                "截止日期": ["2026-03-31"],
                "机构持有比例": ["20%"],
                "持有人户数": ["10,000"],
            }
        )
        mock_tables.side_effect = [[scale], [holders]]
        result = fetch_fund_structure_history("007817")
        self.assertAlmostEqual(result.loc[0, "scale_billion_cny"], 0.952)

    def test_walk_forward_selection_never_uses_test_period(self):
        dates = pd.bdate_range("2022-01-01", periods=900)
        rng = np.random.default_rng(19)
        datasets = {}
        for asset in ("cpo", "memory", "broad_tech"):
            nav = pd.Series(
                (1 + rng.normal(0.0004, 0.012, len(dates))).cumprod(),
                index=dates,
            )
            constituents = pd.DataFrame(
                {
                    "a": (1 + rng.normal(0.0004, 0.014, len(dates))).cumprod(),
                    "b": (1 + rng.normal(0.0003, 0.013, len(dates))).cumprod(),
                },
                index=dates,
            )
            panel = compute_factor_panel(nav, constituents)
            datasets[asset] = (nav, panel)
        test_start = dates[-126]
        folds = purged_training_folds(dates, test_start)
        self.assertTrue(all(end < test_start for _, end in folds))
        chosen, diagnostics = select_config_on_training(
            datasets, test_start, candidate_configs()[:2]
        )
        comparison = locked_test_comparison(
            datasets, chosen, test_start, dates[-1]
        )
        acceptance = model_acceptance(comparison)
        self.assertFalse(diagnostics.empty)
        self.assertEqual(set(comparison["strategy"]), {
            "buy_hold", "ma20_60", "multifactor"
        })
        self.assertEqual(acceptance["asset_count"], 3)

    def test_model_zoo_evaluates_distinct_models(self):
        dates = pd.bdate_range("2023-01-01", periods=700)
        rng = np.random.default_rng(23)
        datasets = {}
        for asset in ("cpo", "memory", "ai"):
            nav = pd.Series(
                (1 + rng.normal(0.0005, 0.012, len(dates))).cumprod(),
                index=dates,
            )
            datasets[asset] = {
                "nav": nav,
                "market_nav": pd.Series(
                    (1 + rng.normal(0.0002, 0.008, len(dates))).cumprod(),
                    index=dates,
                ),
                "peers": pd.DataFrame(
                    {
                        "p1": (1 + rng.normal(0.0003, 0.01, len(dates))).cumprod(),
                        "p2": (1 + rng.normal(0.0002, 0.011, len(dates))).cumprod(),
                    },
                    index=dates,
                ),
            }
        zoo = build_model_zoo(datasets["cpo"]["nav"])
        self.assertGreaterEqual(len(zoo), 8)
        results = evaluate_model_zoo(
            datasets, dates[-126], dates[-1]
        )
        summary = robustness_summary(results)
        self.assertIn("robust_ensemble", summary.index)
        self.assertEqual(results["asset"].nunique(), 3)

        selected, diagnostics = select_ensemble_on_training(
            datasets, dates[-126], top_n=3
        )
        locked = evaluate_locked_selected_ensemble(
            datasets, selected, dates[-126], dates[-1]
        )
        self.assertLessEqual(len(selected), 3)
        self.assertFalse(diagnostics.empty)
        self.assertTrue(
            set(locked["model"]).issubset(
                {"buy_hold", "dual_ma", "trained_ensemble", "no_qualified_model"}
            )
        )

    def test_nav_validation_detects_usable_history(self):
        dates = pd.bdate_range("2023-01-01", periods=600)
        nav = pd.Series(np.linspace(1.0, 1.5, len(dates)), index=dates)
        result = validate_nav_series(nav)
        self.assertTrue(result["valid"])
        self.assertEqual(result["rows"], 600)

    def test_open_fund_backtest_applies_short_holding_fee(self):
        dates = pd.bdate_range("2026-01-01", periods=10)
        nav = pd.Series(1.0, index=dates)
        target = pd.Series([1, 1, 1, 0, 0, 0, 0, 0, 0, 0], index=dates)
        schedule = FundFeeSchedule(
            purchase_fee_rate=0.001,
            redemption_tiers=(
                RedemptionTier(7, 0.015),
                RedemptionTier(None, 0.0),
            ),
        )
        result = backtest_open_fund(nav, target, schedule)
        self.assertGreater(result["metrics"]["purchase_fees"], 0)
        self.assertGreater(result["metrics"]["redemption_fees"], 0)
        self.assertLess(result["metrics"]["total_return"], 0)
        self.assertGreater(result["metrics"]["under_7_day_redemption_ratio"], 0)
        self.assertEqual(redemption_fee_rate(3, schedule), 0.015)
        self.assertEqual(redemption_fee_rate(10, schedule), 0.0)

    def test_buy_hold_pays_entry_and_final_redemption(self):
        dates = pd.bdate_range("2026-01-01", periods=40)
        nav = pd.Series(1.0, index=dates)
        target = pd.Series(1.0, index=dates)
        result = backtest_open_fund(
            nav,
            target,
            FundFeeSchedule(
                purchase_fee_rate=0.001,
                redemption_tiers=(RedemptionTier(None, 0.002),),
            ),
            initial_target=1.0,
            liquidate_at_end=True,
        )
        self.assertAlmostEqual(result["metrics"]["purchase_fees"], 0.001)
        self.assertGreater(result["metrics"]["redemption_fees"], 0)
        self.assertEqual(result["ledger"].iloc[-1]["fund_value"], 0.0)

    def test_stress_tools_preserve_length_and_parameter_grid(self):
        dates = pd.bdate_range("2023-01-01", periods=300)
        nav = pd.Series(np.linspace(1.0, 1.5, len(dates)), index=dates)
        paths = block_bootstrap_nav(nav, simulations=5, block_size=10)
        sensitivity = moving_average_sensitivity(nav, dates[-126], dates[-1])
        self.assertEqual(len(paths), 5)
        self.assertTrue(all(len(path) == len(nav) for path in paths))
        self.assertEqual(len(sensitivity), 5)

    def test_walk_forward_ridge_has_no_early_predictions(self):
        dates = pd.bdate_range("2022-01-01", periods=500)
        rng = np.random.default_rng(29)
        nav = pd.Series(
            (1 + rng.normal(0.0004, 0.01, len(dates))).cumprod(), index=dates
        )
        signal = walk_forward_ridge(nav)
        self.assertTrue((signal.position.iloc[:140] == 0).all())
        self.assertTrue(signal.position.between(0, 1).all())

    def test_sentiment_and_crowding_models_are_bounded(self):
        dates = pd.bdate_range("2022-01-01", periods=500)
        nav = pd.Series(np.linspace(1.0, 2.0, len(dates)), index=dates)
        peers = pd.DataFrame(
            {"a": np.linspace(1.0, 1.8, len(dates)), "b": np.linspace(1.0, 1.5, len(dates))},
            index=dates,
        )
        sentiment = price_sentiment_regime(nav, nav, peers)
        crowding = crowding_aware_momentum(nav)
        self.assertTrue(sentiment.position.between(0, 1).all())
        self.assertTrue(crowding.position.between(0, 1).all())

    def test_slow_models_are_bounded(self):
        dates = pd.bdate_range("2022-01-01", periods=500)
        nav = pd.Series(np.linspace(1.0, 1.8, len(dates)), index=dates)
        self.assertTrue(hysteresis_trend(nav).position.between(0, 1).all())
        self.assertTrue(tail_risk_overlay(nav).position.between(0, 1).all())

    def test_forward_ledger_only_scores_future_nav(self):
        ledger = pd.DataFrame(
            {
                "signal_date": ["2026-01-02"],
                "fund_code": ["007817"],
                "nav": [1.0],
            }
        )
        nav = pd.Series(
            [1.0, 1.1],
            index=pd.to_datetime(["2026-01-02", "2026-01-05"]),
        )
        result = score_forward_ledger(ledger, {"007817": nav})
        self.assertAlmostEqual(float(result.loc[0, "next_nav_return"]), 0.1)

    def test_rotation_forward_ledger_deduplicates(self):
        import tempfile
        snapshot = RotationForwardSnapshot(
            generated_at="2026-06-15T08:30:00+08:00",
            signal_date="2026-06-12",
            model_version="abc",
            rotation_model="rotation",
            candidate_cpo_weight=0.6,
            candidate_memory_weight=0.2,
            candidate_ai_weight=0.2,
            deployed_cpo_weight=1 / 3,
            deployed_memory_weight=1 / 3,
            deployed_ai_weight=1 / 3,
            deployment_accepted=False,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = f"{directory}/rotation.csv"
            append_rotation_forward_ledger(snapshot, path)
            append_rotation_forward_ledger(snapshot, path)
            result = pd.read_csv(path)
        self.assertEqual(len(result), 1)

    def test_rotation_forward_ledger_scores_next_common_nav(self):
        ledger = pd.DataFrame(
            {
                "signal_date": ["2026-01-02"],
                "candidate_cpo_weight": [0.6],
                "candidate_memory_weight": [0.2],
                "candidate_ai_weight": [0.2],
                "deployed_cpo_weight": [1 / 3],
                "deployed_memory_weight": [1 / 3],
                "deployed_ai_weight": [1 / 3],
            }
        )
        dates = pd.to_datetime(["2026-01-02", "2026-01-05"])
        navs = {
            "cpo_communication": pd.Series([1.0, 1.1], index=dates),
            "memory_semiconductor_proxy": pd.Series([1.0, 1.0], index=dates),
            "artificial_intelligence": pd.Series([1.0, 0.9], index=dates),
        }
        result = score_rotation_forward_ledger(ledger, navs)
        self.assertAlmostEqual(
            float(result.loc[0, "candidate_next_return"]),
            0.04,
        )

    def test_open_fund_execution_discretizes_and_throttles_additions(self):
        dates = pd.bdate_range("2026-01-01", periods=8)
        raw = pd.Series([0.3, 0.8, 0.9, 0.1, 0.8, 0.8, 0.8, 0.8], index=dates)
        result = apply_open_fund_execution_policy(raw, 5, confirmation_days=1)
        self.assertEqual(result.iloc[0], 0.25)
        self.assertEqual(result.iloc[1], 0.25)
        self.assertEqual(result.iloc[3], 0.0)
        self.assertEqual(result.iloc[5], 0.8)

    def test_core_tactical_never_drops_below_core(self):
        dates = pd.bdate_range("2026-01-01", periods=5)
        from quant_fund_advisor.model_zoo import ModelSignal
        tactical = ModelSignal(
            "test", pd.Series([0, 1, 0.5, 0, 1], index=dates), "test"
        )
        result = core_tactical(tactical, 0.70)
        self.assertGreaterEqual(result.position.min(), 0.70)
        self.assertEqual(result.position.max(), 1.0)

    def test_bull_hold_bear_defense_is_bounded(self):
        dates = pd.bdate_range("2023-01-01", periods=400)
        market = pd.Series(
            np.r_[np.linspace(1.0, 1.5, 200), np.linspace(1.5, 0.9, 200)],
            index=dates,
        )
        defensive = walk_forward_ridge(market)
        result = bull_hold_bear_defense(
            market, market, defensive, core_weight=0.70
        )
        self.assertTrue(result.position.between(0.70, 1.0).all())
        self.assertEqual(result.position.iloc[150], 1.0)

    def test_model_zoo_has_no_degenerate_85_percent_core(self):
        dates = pd.bdate_range("2023-01-01", periods=400)
        nav = pd.Series(np.linspace(1.0, 1.5, len(dates)), index=dates)
        names = {model.name for model in build_model_zoo(nav)}
        self.assertFalse(any(name.startswith("core85_") for name in names))

    def test_new_market_state_models_are_bounded_and_registered(self):
        dates = pd.bdate_range("2023-01-01", periods=500)
        nav = pd.Series(
            np.r_[np.linspace(1.0, 1.4, 250), np.linspace(1.4, 1.15, 250)],
            index=dates,
        )
        market = pd.Series(
            np.r_[np.linspace(1.0, 1.3, 250), np.linspace(1.3, 1.05, 250)],
            index=dates,
        )
        peers = pd.DataFrame(
            {
                "p1": nav * 0.97,
                "p2": market * 1.02,
            },
            index=dates,
        )
        regime = regime_relative_strength(nav, market, peers)
        nowcast = market_state_nowcast(nav, market, peers)
        self.assertTrue(regime.position.between(0.0, 1.0).all())
        self.assertTrue(nowcast.position.between(0.0, 1.0).all())
        names = {model.name for model in build_model_zoo(nav, market, peers)}
        self.assertIn("regime_relative_strength", names)
        self.assertIn("market_state_nowcast", names)

    def test_theme_selection_uses_training_and_returns_one_model_per_asset(self):
        dates = pd.bdate_range("2022-01-01", periods=900)
        rng = np.random.default_rng(31)
        datasets = {}
        for asset in ("cpo", "memory", "ai"):
            nav = pd.Series(
                (1 + rng.normal(0.0005, 0.012, len(dates))).cumprod(),
                index=dates,
            )
            datasets[asset] = {
                "nav": nav,
                "market_nav": nav,
                "peers": pd.DataFrame(
                    {
                        "p1": (1 + rng.normal(0.0003, 0.01, len(dates))).cumprod(),
                        "p2": (1 + rng.normal(0.0002, 0.011, len(dates))).cumprod(),
                    },
                    index=dates,
                ),
            }
        test_start = dates[-126]
        selected, diagnostics = select_theme_models_on_training(
            datasets, test_start
        )
        locked = evaluate_locked_theme_models(
            datasets, selected, test_start, dates[-1]
        )
        self.assertEqual(set(selected), set(datasets))
        self.assertEqual(len(diagnostics), len(datasets))
        self.assertEqual(
            set(locked["model"]),
            {"buy_hold", "dual_ma", "theme_selected"},
        )

    def test_multi_fund_portfolio_is_fee_aware_and_bounded(self):
        dates = pd.bdate_range("2024-01-01", periods=180)
        navs = pd.DataFrame(
            {
                "a": np.linspace(1.0, 1.4, len(dates)),
                "b": np.linspace(1.0, 1.2, len(dates)),
                "c": np.linspace(1.0, 0.9, len(dates)),
            },
            index=dates,
        )
        config = RotationConfig(20, 1, 0.60)
        targets = relative_strength_weights(navs, config)
        result = backtest_open_fund_portfolio(
            navs,
            targets,
            initial_weights=pd.Series(1 / 3, index=navs.columns),
        )
        weight_columns = [
            column for column in result["ledger"] if column.startswith("weight_")
        ]
        self.assertLessEqual(
            result["ledger"][weight_columns].sum(axis=1).max(),
            1.000001,
        )
        self.assertGreaterEqual(result["metrics"]["total_fees"], 0.0)
        self.assertGreater(
            result["ledger"][weight_columns].iloc[0].sum(),
            0.95,
        )
        exposure = portfolio_tail_risk_exposure(navs)
        self.assertTrue(exposure.between(0.5, 1.0).all())

    def test_multi_fund_buy_hold_does_not_rebalance_constant_target(self):
        dates = pd.bdate_range("2026-01-01", periods=40)
        navs = pd.DataFrame(
            {
                "winner": np.linspace(1.0, 2.0, len(dates)),
                "flat": np.ones(len(dates)),
            },
            index=dates,
        )
        targets = pd.DataFrame(0.5, index=dates, columns=navs.columns)
        result = backtest_open_fund_portfolio(
            navs,
            targets,
            initial_weights=targets.iloc[0],
        )
        self.assertEqual(result["metrics"]["trade_count"], 4.0)
        self.assertGreater(result["metrics"]["total_return"], 0.45)

    def test_rotation_selection_and_locked_comparison(self):
        dates = pd.bdate_range("2021-01-01", periods=1100)
        rng = np.random.default_rng(37)
        navs = pd.DataFrame(
            {
                name: (1 + rng.normal(mu, 0.012, len(dates))).cumprod()
                for name, mu in (
                    ("cpo", 0.0006),
                    ("memory", 0.0004),
                    ("ai", 0.0003),
                )
            },
            index=dates,
        )
        test_start = dates[-126]
        chosen, diagnostics = select_rotation_on_training(navs, test_start)
        comparison = locked_rotation_comparison(
            navs, chosen, test_start, dates[-1]
        )
        self.assertFalse(diagnostics.empty)
        self.assertEqual(len(comparison), 2)
        self.assertTrue(
            np.allclose(equal_weight_targets(navs).sum(axis=1), 1.0)
        )
        walk_forward = walk_forward_rotation_validation(navs, test_start)
        self.assertFalse(walk_forward.empty)
        self.assertTrue(
            (walk_forward["fold_end"] < test_start).all()
        )
        risk_budget = risk_budget_momentum_weights(navs)
        risk_folds = fixed_strategy_fold_validation(
            navs,
            risk_budget,
            test_start,
            "risk_budget_momentum",
        )
        self.assertFalse(risk_folds.empty)
        self.assertLessEqual(risk_budget.sum(axis=1).max(), 1.000001)
        self.assertLessEqual(risk_budget.max().max(), 0.600001)

    def test_probability_of_backtest_overfitting_is_bounded(self):
        diagnostics = pd.DataFrame(
            [
                {
                    "fold_end": fold,
                    "model": model,
                    "excess_return": score,
                }
                for fold, scores in enumerate(
                    (
                        (0.1, 0.0, -0.1),
                        (-0.1, 0.1, 0.0),
                        (0.0, -0.1, 0.1),
                        (0.2, 0.0, -0.2),
                        (-0.2, 0.2, 0.0),
                        (0.0, -0.2, 0.2),
                    )
                )
                for model, score in zip(("a", "b", "c"), scores)
            ]
        )
        result = probability_of_backtest_overfitting(diagnostics)
        self.assertGreater(result["split_count"], 0)
        self.assertGreaterEqual(result["pbo"], 0.0)
        self.assertLessEqual(result["pbo"], 1.0)

    def test_intraday_estimate_ignores_small_moves_and_confirms_large_moves(self):
        small = estimate_confidence(
            0.4,
            {"observations": 30, "mae_pct": 0.5, "direction_accuracy": 0.8},
            etf_change_pct=0.5,
            breadth_ratio=0.8,
        )
        self.assertFalse(small["reliable"])
        large = estimate_confidence(
            -2.5,
            {"observations": 30, "mae_pct": 0.5, "direction_accuracy": 0.8},
            etf_change_pct=-2.0,
            breadth_ratio=0.2,
        )
        overlay = apply_intraday_overlay(0.5, -2.5, large)
        self.assertTrue(large["reliable"])
        self.assertEqual(overlay["action"], "TACTICAL_ADD")
        self.assertGreater(overlay["target_position"], 0.5)

    def test_historical_extreme_overlay_only_acts_on_large_moves(self):
        dates = pd.bdate_range("2025-01-01", periods=130)
        returns = np.zeros(len(dates))
        returns[121] = -0.02
        returns[124] = 0.04
        nav = pd.Series((1 + returns).cumprod(), index=dates)
        target = historical_extreme_move_overlay(
            nav,
            require_long_trend_for_add=False,
        )
        self.assertEqual(target.iloc[120], 0.9)
        self.assertEqual(target.iloc[121], 1.0)
        self.assertEqual(target.iloc[124], 1.0)

    def test_estimate_calibration_measures_direction_and_error(self):
        history = pd.DataFrame(
            {
                "estimated_change_pct": [2.0, -2.0, 0.5],
                "actual_change_pct": [1.5, -1.0, -0.2],
            }
        )
        metrics = calibration_metrics(history)
        self.assertEqual(metrics["observations"], 3)
        self.assertAlmostEqual(metrics["direction_accuracy"], 2 / 3)

    def test_estimate_history_reconciliation_and_adaptive_threshold(self):
        dates = pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"])
        nav = pd.Series([1.0, 1.02, 1.01], index=dates)
        history = pd.DataFrame(
            {
                "date": dates,
                "estimated_change_pct": [0.0, 1.8, -0.8],
            }
        )
        reconciled = reconcile_estimate_history(history, nav)
        sample = pd.concat([reconciled] * 10, ignore_index=True).dropna()
        policy = adaptive_policy_from_history(sample)
        self.assertIn("actual_change_pct", reconciled.columns)
        self.assertGreaterEqual(policy.minimum_abs_change_pct, 1.5)

    def test_action_report_rejects_previous_day_fundgz(self):
        used, reason = _estimate_comment(
            "007817",
            3.6,
            pd.Timestamp("2026-06-18 15:00:00"),
            {"observations": 30, "mae_pct": 0.5, "direction_accuracy": 0.8},
            adaptive_policy_from_history(
                pd.DataFrame(
                    {
                        "estimated_change_pct": [2.0] * 20,
                        "actual_change_pct": [1.8] * 20,
                    }
                )
            ),
            etf_change_pct=3.2,
            breadth_ratio=1.0,
            cutoff=pd.Timestamp("2026-06-19 14:30:00"),
        )
        self.assertFalse(used)
        self.assertIn("不是今天", reason)

    def test_synthetic_estimate_history_adds_bounded_noise(self):
        dates = pd.bdate_range("2026-01-01", periods=4)
        nav = pd.Series([1.0, 1.06, 1.1236, 1.112364], index=dates)
        history = simulate_intraday_estimate_history(
            nav,
            SyntheticEstimateConfig(seed=7),
        )
        large_move = history.iloc[0]
        self.assertGreaterEqual(large_move["estimated_change_pct"], 4.5)
        self.assertLessEqual(large_move["estimated_change_pct"], 7.5)

    def test_synthetic_estimate_can_flip_small_moves(self):
        dates = pd.bdate_range("2026-01-01", periods=6)
        nav = pd.Series([1.0, 1.005, 1.000, 1.004, 0.999, 1.003], index=dates)
        history = simulate_intraday_estimate_history(
            nav,
            SyntheticEstimateConfig(
                seed=1,
                small_move_threshold_pct=1.2,
                small_move_flip_probability=1.0,
                medium_move_flip_probability=0.0,
            ),
        )
        actual_sign = np.sign(history["actual_change_pct"])
        estimated_sign = np.sign(history["estimated_change_pct"])
        self.assertTrue((actual_sign != estimated_sign).any())

    def test_historical_synthetic_overlay_returns_targets_and_history(self):
        dates = pd.bdate_range("2025-01-01", periods=160)
        nav = pd.Series(
            np.r_[np.linspace(1.0, 1.3, 80), np.linspace(1.3, 1.1, 80)],
            index=dates,
        )
        target, history = historical_synthetic_estimate_overlay(nav)
        self.assertEqual(len(target), len(nav))
        self.assertEqual(len(history), len(nav))
        self.assertTrue(target.between(0.0, 1.0).all())

    def test_all_in_t_strategy_exports_summary_shape(self):
        result = build_all_in_t_strategy("cpo_communication")
        self.assertEqual(set(result.summary["strategy"]), {"buy_hold", "all_in_t_overlay"})
        self.assertIn("buy_hold", result.equity.columns)
        self.assertIn("all_in_t_overlay", result.equity.columns)

    def test_expanded_all_in_t_overlay_is_bounded(self):
        dates = pd.bdate_range("2024-01-01", periods=220)
        nav = pd.Series(
            np.r_[np.linspace(1.0, 1.5, 110), np.linspace(1.5, 1.3, 110)],
            index=dates,
        )
        target, history = all_in_t_overlay(
            nav,
            AllInTConfig(5.0, -3.0, 0.03, 0.04, 7),
        )
        self.assertTrue(target.between(0.0, 1.0).all())
        self.assertEqual(len(history), len(nav))

    @patch.dict(
        "sys.modules",
        {
            "akshare": type(
                "FakeAkshare",
                (),
                {
                    "fund_etf_spot_em": staticmethod(
                        lambda: pd.DataFrame(
                            {
                                "代码": ["515880"],
                                "名称": ["通信ETF"],
                                "最新价": [2.0],
                                "涨跌幅": [-2.1],
                                "IOPV实时估值": [1.99],
                                "基金折价率": [0.5],
                            }
                        )
                    )
                },
            )
        },
    )
    def test_akshare_etf_iopv_confirmation(self):
        result = fetch_akshare_etf_iopv("515880")
        self.assertEqual(result["etf_code"], "515880")
        self.assertAlmostEqual(result["change_pct"], -2.1)

    @patch.dict(
        "sys.modules",
        {
            "akshare": type(
                "FakeEstimateAkshare",
                (),
                {
                    "fund_value_estimation_em": staticmethod(
                        lambda symbol: pd.DataFrame(
                            {
                                "基金代码": ["007817"],
                                "基金名称": ["通信联接"],
                                "2026-06-15-估算数据-估算值": [4.30],
                                "2026-06-15-估算数据-估算增长率": ["-2.33%"],
                                "2026-06-15-公布数据-单位净值": [4.4025],
                            }
                        )
                    )
                },
            )
        },
    )
    def test_akshare_fund_estimate_fallback(self):
        result = fetch_akshare_fund_estimate("007817")
        self.assertEqual(result.fund_code, "007817")
        self.assertAlmostEqual(result.estimated_change_pct, -2.33)


if __name__ == "__main__":
    unittest.main()
