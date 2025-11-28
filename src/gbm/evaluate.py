"""
Evaluation utilities for GBM quantile regression exit model.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """
    Compute pinball (quantile) loss.

    The pinball loss penalizes:
    - Under-predictions (y_true > y_pred) by tau * residual
    - Over-predictions (y_true < y_pred) by (1 - tau) * |residual|

    Args:
        y_true: Actual values
        y_pred: Predicted values
        tau: Quantile level (0 < tau < 1)

    Returns:
        Mean pinball loss
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residual = y_true - y_pred
    return float(np.mean(np.where(residual >= 0, tau * residual, (tau - 1) * residual)))


def coverage_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute coverage rate: fraction of actual values <= predicted quantile.

    For a well-calibrated tau-quantile model, coverage should be approximately tau.

    Args:
        y_true: Actual values
        y_pred: Predicted quantile values

    Returns:
        Coverage rate (fraction of y_true <= y_pred)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    return float(np.mean(y_true <= y_pred))


def calibration_metrics(
    y_true: np.ndarray,
    predictions: Dict[float, np.ndarray],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Compute calibration metrics across multiple quantiles.

    For each quantile tau, we compute:
    - Expected coverage (tau itself)
    - Actual coverage (fraction of y_true <= y_pred)
    - Calibration error (|actual - expected|)

    Args:
        y_true: Actual values
        predictions: Dictionary mapping quantile tau to predicted values
        logger: Optional logger

    Returns:
        DataFrame with calibration metrics for each quantile
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    results = []
    for tau, y_pred in sorted(predictions.items()):
        actual_cov = coverage_rate(y_true, y_pred)
        calib_error = abs(actual_cov - tau)

        results.append({
            "quantile": tau,
            "expected_coverage": tau,
            "actual_coverage": actual_cov,
            "calibration_error": calib_error,
        })

        logger.info(
            "Quantile %.2f: expected=%.2f, actual=%.3f, error=%.3f",
            tau, tau, actual_cov, calib_error
        )

    return pd.DataFrame(results)


def compute_all_metrics(
    y_true: np.ndarray,
    predictions: Dict[float, np.ndarray],
    split_name: str = "test",
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Compute all evaluation metrics for GBM quantile regression.

    Args:
        y_true: Actual values
        predictions: Dictionary mapping quantile tau to predicted values
        split_name: Name of the split (e.g., "train", "test")
        logger: Optional logger

    Returns:
        Dictionary with all metrics organized by quantile
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    y_true = np.asarray(y_true)
    n_samples = len(y_true)

    metrics = {
        "split": split_name,
        "n_samples": n_samples,
        "y_true_mean": float(np.mean(y_true)),
        "y_true_std": float(np.std(y_true)),
        "y_true_min": float(np.min(y_true)),
        "y_true_max": float(np.max(y_true)),
        "quantile_metrics": {},
    }

    for tau, y_pred in sorted(predictions.items()):
        y_pred = np.asarray(y_pred)

        # Pinball loss
        pb_loss = pinball_loss(y_true, y_pred, tau)

        # Coverage rate
        cov = coverage_rate(y_true, y_pred)

        # Calibration error
        calib_err = abs(cov - tau)

        # Residual statistics
        residuals = y_true - y_pred
        residual_mean = float(np.mean(residuals))
        residual_std = float(np.std(residuals))

        # Prediction statistics
        pred_mean = float(np.mean(y_pred))
        pred_std = float(np.std(y_pred))

        q_metrics = {
            "pinball_loss": pb_loss,
            "coverage_rate": cov,
            "calibration_error": calib_err,
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "pred_mean": pred_mean,
            "pred_std": pred_std,
        }

        metrics["quantile_metrics"][tau] = q_metrics

        logger.info(
            "%s quantile %.2f: pinball=%.6f, coverage=%.3f (expected %.2f), "
            "calib_error=%.3f",
            split_name, tau, pb_loss, cov, tau, calib_err
        )

    return metrics


def compute_interval_metrics(
    y_true: np.ndarray,
    y_lower: np.ndarray,
    y_upper: np.ndarray,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Compute metrics for prediction intervals formed by two quantiles.

    Args:
        y_true: Actual values
        y_lower: Lower quantile predictions (e.g., q10)
        y_upper: Upper quantile predictions (e.g., q50)
        logger: Optional logger

    Returns:
        Dictionary with interval metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    y_true = np.asarray(y_true)
    y_lower = np.asarray(y_lower)
    y_upper = np.asarray(y_upper)

    # Interval width
    interval_width = y_upper - y_lower
    mean_width = float(np.mean(interval_width))

    # Coverage: fraction of y_true within [y_lower, y_upper]
    # Note: For stop-loss/target, we actually want y_true >= y_lower
    # (i.e., we don't hit stop-loss) AND some target condition
    in_interval = (y_true >= y_lower) & (y_true <= y_upper)
    interval_coverage = float(np.mean(in_interval))

    # Below lower (hit stop-loss)
    below_lower = float(np.mean(y_true < y_lower))

    # Above upper (exceeded target)
    above_upper = float(np.mean(y_true > y_upper))

    metrics = {
        "mean_interval_width": mean_width,
        "interval_coverage": interval_coverage,
        "below_lower_rate": below_lower,
        "above_upper_rate": above_upper,
    }

    logger.info(
        "Interval metrics: width=%.6f, coverage=%.3f, below_lower=%.3f, above_upper=%.3f",
        mean_width, interval_coverage, below_lower, above_upper
    )

    return metrics


def summarize_cv_results(
    cv_results: List[Dict],
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Summarize cross-validation results from hyperparameter tuning.

    Args:
        cv_results: List of CV result dictionaries from tune_and_train_gbm
        logger: Optional logger

    Returns:
        DataFrame with CV summary statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    df = pd.DataFrame(cv_results)

    # Sort by mean pinball loss
    if "cv_mean_pinball" in df.columns:
        df = df.sort_values("cv_mean_pinball", ascending=True)

    logger.info("CV results summary: %d configurations evaluated", len(df))

    return df
