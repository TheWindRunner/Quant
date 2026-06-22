"""Quantitative fund research toolkit."""

from .backtest import BacktestConfig, run_backtest
from .model import AdvisorConfig, score_assets
from .selector import rank_funds

__all__ = ["AdvisorConfig", "BacktestConfig", "rank_funds", "run_backtest", "score_assets"]
