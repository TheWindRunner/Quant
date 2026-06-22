import unittest
from unittest.mock import patch

import pandas as pd

from tools import backtest_pcb_purchase_limit as execution


class FundExecutionRuleTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.bdate_range("2026-01-05", periods=5)
        self.navs = pd.DataFrame(
            {"PCB": 1.0, "AI": 1.0},
            index=self.dates,
        )

    def test_direct_conversion_is_available_on_trade_date(self):
        targets = pd.DataFrame(
            {"PCB": [0, 1, 1, 1, 1], "AI": [1, 0, 0, 0, 0]},
            index=self.dates,
        )
        with patch.object(execution, "EXECUTION_DELAY_DAYS", 0):
            ledger, _ = execution.simulate_weighted_targets(self.navs, targets)

        self.assertAlmostEqual(ledger.loc[self.dates[1], "当日买入金额"], 4000.0)
        self.assertAlmostEqual(ledger.loc[self.dates[1], "待到账赎回款"], 0.0)
        self.assertGreater(ledger.loc[self.dates[1], "权重_AI"], 0.0)

    def test_cash_redemption_cannot_be_reinvested_until_t_plus_2(self):
        targets = pd.DataFrame(
            {
                "PCB": [0, 0, 1, 1, 1],
                "AI": [1, 0, 0, 0, 0],
            },
            index=self.dates,
        )
        with patch.object(execution, "EXECUTION_DELAY_DAYS", 0):
            ledger, _ = execution.simulate_weighted_targets(self.navs, targets)

        self.assertGreater(ledger.loc[self.dates[1], "待到账赎回款"], 0.0)
        self.assertAlmostEqual(ledger.loc[self.dates[2], "当日买入金额"], 0.0)
        self.assertAlmostEqual(ledger.loc[self.dates[3], "当日买入金额"], 4000.0)

    def test_c_class_redemption_fee_schedule(self):
        self.assertEqual(execution.c_class_redemption_fee_rate(6), 0.015)
        self.assertEqual(execution.c_class_redemption_fee_rate(7), 0.005)
        self.assertEqual(execution.c_class_redemption_fee_rate(29), 0.005)
        self.assertEqual(execution.c_class_redemption_fee_rate(30), 0.0)

    def test_four_pcb_channels_enforce_aggregate_daily_limit(self):
        targets = pd.DataFrame(
            {"PCB": [1, 1, 1, 1, 1], "AI": [0, 0, 0, 0, 0]},
            index=self.dates,
        )
        with patch.object(execution, "EXECUTION_DELAY_DAYS", 0):
            ledger, _ = execution.simulate_weighted_targets(self.navs, targets)

        bought = ledger.iloc[:, 5].tolist()
        self.assertEqual(bought[:3], [4000.0, 4000.0, 2000.0])
        self.assertTrue(all(amount <= 4 * 1000.0 for amount in bought))


if __name__ == "__main__":
    unittest.main()
