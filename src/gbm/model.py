"""
Model utilities for GBM quantile regression exit model.
"""

import json
import logging
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from joblib import dump
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler


def _pinball_loss(y_true: np.ndarray, y_pred: np.ndarray, tau: float) -> float:
    """
    Compute pinball (quantile) loss. Internal function for CV.

    Args:
        y_true: Actual values
        y_pred: Predicted values
        tau: Quantile level (0 < tau < 1)

    Returns:
        Mean pinball loss
    """
    residual = y_true - y_pred
    return float(np.mean(np.where(residual >= 0, tau * residual, (tau - 1) * residual)))


def train_quantile_gbm(
    X: np.ndarray,
    y: np.ndarray,
    quantile: float = 0.5,
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    min_samples_leaf: int = 10,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    """
    Train a single quantile GBM model.

    Args:
        X: Feature matrix (n_samples, n_features)
        y: Target array (n_samples,)
        quantile: Quantile to predict (alpha parameter)
        n_estimators: Number of boosting stages
        max_depth: Maximum depth of individual trees
        learning_rate: Learning rate shrinks contribution of each tree
        min_samples_leaf: Minimum samples required at leaf node
        random_state: Random seed

    Returns:
        Fitted GradientBoostingRegressor
    """
    model = GradientBoostingRegressor(
        loss="quantile",
        alpha=quantile,
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
    )
    model.fit(X, y)
    return model


def tune_and_train_gbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    day_groups: np.ndarray,
    quantile: float = 0.5,
    param_grid: Optional[Dict] = None,
    logger: Optional[logging.Logger] = None,
) -> Tuple[GradientBoostingRegressor, Dict]:
    """
    Tune GBM hyperparameters with Leave-One-Day-Out CV, then fit final model.

    Args:
        X_train: Scaled training features
        y_train: Training targets
        day_groups: Day identifier per sample for LODO CV
        quantile: Target quantile (alpha parameter)
        param_grid: Hyperparameter grid (default provided if None)
        logger: Optional logger

    Returns:
        (fitted_model, summary_dict) containing best params, CV results, and metrics
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5],
            "learning_rate": [0.05, 0.1],
            "min_samples_leaf": [10, 20],
        }

    unique_days = np.unique(day_groups)
    if len(unique_days) < 2:
        raise ValueError("Need at least 2 unique training days for LODO CV")

    logger.info(
        "Tuning GBM (quantile=%.2f) via LODO CV on %d days",
        quantile, len(unique_days)
    )

    grid_results: List[Dict] = []

    # Grid search
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())

    for combo in product(*param_values):
        params = dict(zip(param_names, combo))

        fold_losses: List[float] = []

        for val_day in unique_days:
            val_mask = day_groups == val_day
            train_mask = ~val_mask

            X_tr, y_tr = X_train[train_mask], y_train[train_mask]
            X_val, y_val = X_train[val_mask], y_train[val_mask]

            if len(X_tr) == 0 or len(X_val) == 0:
                continue

            try:
                model = train_quantile_gbm(
                    X_tr, y_tr,
                    quantile=quantile,
                    **params,
                )
                y_pred = model.predict(X_val)
                loss = _pinball_loss(y_val, y_pred, quantile)
                fold_losses.append(loss)
            except Exception as exc:
                logger.warning(
                    "GBM training failed for params=%s on fold %s: %s",
                    params, str(val_day), exc
                )
                fold_losses.append(np.nan)

        # Aggregate CV scores
        cv_losses = np.array(fold_losses, dtype=float)
        cv_mean = float(np.nanmean(cv_losses)) if len(cv_losses) else np.nan
        cv_std = float(np.nanstd(cv_losses)) if len(cv_losses) else np.nan

        grid_results.append({
            **params,
            "quantile": quantile,
            "cv_mean_pinball": cv_mean,
            "cv_std_pinball": cv_std,
        })

    # Sort by mean pinball loss (lower is better)
    grid_results_sorted = sorted(
        grid_results,
        key=lambda x: x.get("cv_mean_pinball", np.inf)
    )

    if not grid_results_sorted:
        raise RuntimeError("No GBM configurations could be evaluated")

    best = grid_results_sorted[0]
    best_params = {k: best[k] for k in param_names}

    logger.info(
        "Best GBM params (q=%.2f): %s | CV mean pinball=%.6f ± %.6f",
        quantile, best_params, best["cv_mean_pinball"], best["cv_std_pinball"]
    )

    # Fit final model on all training data
    final_model = train_quantile_gbm(
        X_train, y_train,
        quantile=quantile,
        **best_params,
    )

    # Compute train loss for reference
    y_train_pred = final_model.predict(X_train)
    train_pinball = _pinball_loss(y_train, y_train_pred, quantile)

    summary = {
        "quantile": quantile,
        "best_params": best_params,
        "cv_mean_pinball": best["cv_mean_pinball"],
        "cv_std_pinball": best["cv_std_pinball"],
        "train_pinball": train_pinball,
        "grid_results": grid_results_sorted,
    }

    return final_model, summary


def save_artifacts(
    models: Dict[str, GradientBoostingRegressor],
    scaler: StandardScaler,
    config: Dict,
    outdir: str = "models",
    logger: Optional[logging.Logger] = None,
) -> Dict[str, str]:
    """
    Save GBM models, scaler, and config JSON.

    Args:
        models: Dictionary mapping quantile names to fitted models
                e.g., {"q10": model_q10, "q50": model_q50}
        scaler: Fitted StandardScaler
        config: Configuration dictionary with metadata
        outdir: Output directory
        logger: Optional logger

    Returns:
        Dictionary of saved file paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    paths = {}

    # Save each model
    for name, model in models.items():
        model_path = out / f"gbm_{name}.joblib"
        dump(model, model_path)
        paths[f"model_{name}"] = str(model_path)
        logger.info(f"Saved GBM model to {model_path}")

    # Save scaler
    scaler_path = out / "gbm_scaler.joblib"
    dump(scaler, scaler_path)
    paths["scaler"] = str(scaler_path)
    logger.info(f"Saved scaler to {scaler_path}")

    # Save config
    config_path = out / "gbm_config.json"
    with open(config_path, "w", encoding="utf-8") as f:
        # Convert numpy types for JSON serialization
        json.dump(config, f, indent=2, default=str)
    paths["config"] = str(config_path)
    logger.info(f"Saved config to {config_path}")

    return paths
