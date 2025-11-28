"""
GBM Quantile Regression Exit Model.

This module implements gradient boosted quantile regressors for predicting
stop-loss (0.1 quantile) and target (0.5 quantile) exit levels for mean
reversion trades.
"""

from .data import load_rf_scores, compute_forward_returns, prepare_gbm_dataset
from .model import train_quantile_gbm, tune_and_train_gbm, save_artifacts
from .evaluate import (
    pinball_loss,
    coverage_rate,
    calibration_metrics,
    compute_all_metrics,
    compute_interval_metrics,
    summarize_cv_results,
)

__all__ = [
    "load_rf_scores",
    "compute_forward_returns",
    "prepare_gbm_dataset",
    "train_quantile_gbm",
    "tune_and_train_gbm",
    "save_artifacts",
    "pinball_loss",
    "coverage_rate",
    "calibration_metrics",
    "compute_all_metrics",
    "compute_interval_metrics",
    "summarize_cv_results",
]
