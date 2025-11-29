"""
Full Pipeline Evaluation Module

Provides trading simulation and metrics calculation for the mean reversion
trading pipeline, evaluating out-of-sample performance on the test day.
"""

from .evaluate import (
    load_test_data,
    compute_trade_direction,
    simulate_trades,
    compute_metrics,
    compute_metrics_by_ticker,
)

__all__ = [
    "load_test_data",
    "compute_trade_direction",
    "simulate_trades",
    "compute_metrics",
    "compute_metrics_by_ticker",
]
