"""
Model utilities for GMM overextension detection.

Functions:
- tune_and_train_gmm: LODO day-level CV on train days, fit final model
- score_loglik: compute per-sample log-likelihoods
- choose_threshold: choose a train-based log-likelihood threshold
- save_artifacts: persist model, scaler, and config for reuse
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from joblib import dump
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


@dataclass
class GMMParams:
    n_components: int
    covariance_type: str
    random_state: int = 42
    max_iter: int = 500
    reg_covar: float = 1e-6


def _fit_one(
    X: np.ndarray,
    params: GMMParams,
) -> GaussianMixture:
    model = GaussianMixture(
        n_components=params.n_components,
        covariance_type=params.covariance_type,
        random_state=params.random_state,
        max_iter=params.max_iter,
        reg_covar=params.reg_covar,
        init_params="kmeans",
    )
    model.fit(X)
    return model


def tune_and_train_gmm(
    X_train: np.ndarray,
    day_groups: Iterable,
    n_components_grid: Iterable[int] = range(2, 9),
    cov_types: Iterable[str] = ("full", "diag"),
    logger: Optional[logging.Logger] = None,
) -> Tuple[GaussianMixture, Dict]:
    """
    Tune GMM hyperparameters with Leave-One-Day-Out CV across the 4 training days,
    then fit the final model on all training samples.

    Args:
        X_train: Standardized train features (n_samples x n_features)
        day_groups: Day identifier per sample (length n_samples)
        n_components_grid: Values for number of mixture components
        cov_types: Covariance types to try (e.g., 'full', 'diag')
        logger: Optional logger

    Returns:
        (fitted_model, summary_dict)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    day_groups = np.asarray(list(day_groups))
    unique_days = np.unique(day_groups)
    if len(unique_days) < 2:
        raise ValueError("Need at least 2 unique training days for LODO CV")

    grid_results: List[Dict] = []

    logger.info("Tuning GMM via LODO CV on days: %s", [str(d) for d in unique_days])

    for cov in cov_types:
        for k in n_components_grid:
            params = GMMParams(n_components=k, covariance_type=cov)
            fold_scores: List[float] = []

            for val_day in unique_days:
                val_mask = day_groups == val_day
                train_mask = ~val_mask

                X_tr = X_train[train_mask]
                X_val = X_train[val_mask]

                if len(X_tr) == 0 or len(X_val) == 0:
                    continue

                try:
                    model = _fit_one(X_tr, params)
                    llk = model.score_samples(X_val)
                    fold_scores.append(float(np.mean(llk)))
                except Exception as exc:
                    logger.warning(
                        "GMM training failed for k=%d, cov=%s on fold %s: %s",
                        k, cov, str(val_day), exc,
                    )
                    fold_scores.append(np.nan)

            # Aggregate CV scores
            cv_scores = np.array(fold_scores, dtype=float)
            cv_mean = float(np.nanmean(cv_scores)) if len(cv_scores) else np.nan
            cv_std = float(np.nanstd(cv_scores)) if len(cv_scores) else np.nan

            # Train a model on all of X_train to compute BIC/AIC for tie-breakers
            train_bic = np.nan
            train_aic = np.nan
            try:
                model_full = _fit_one(X_train, params)
                train_bic = float(model_full.bic(X_train))
                train_aic = float(model_full.aic(X_train))
            except Exception:
                pass

            grid_results.append({
                "n_components": int(k),
                "covariance_type": cov,
                "cv_mean_loglik": cv_mean,
                "cv_std_loglik": cv_std,
                "train_bic": train_bic,
                "train_aic": train_aic,
            })

    # Choose best: highest CV mean LLK; tie-breaker: lower BIC
    def sort_key(item: Dict) -> Tuple:
        # Negate BIC for sorting descending by quality (lower BIC is better)
        bic = item.get("train_bic", np.inf)
        return (item.get("cv_mean_loglik", -np.inf), -np.inf if np.isnan(bic) else -bic)

    grid_results_sorted = sorted(grid_results, key=sort_key, reverse=True)
    if not grid_results_sorted:
        raise RuntimeError("No GMM settings could be evaluated")

    best = grid_results_sorted[0]
    best_params = GMMParams(
        n_components=int(best["n_components"]),
        covariance_type=str(best["covariance_type"]),
    )

    logger.info(
        "Best GMM params: k=%d cov=%s | CV mean LLK=%.4f BIC=%.1f",
        best_params.n_components,
        best_params.covariance_type,
        best.get("cv_mean_loglik", float("nan")),
        best.get("train_bic", float("nan")),
    )

    # Fit final model on all training data
    final_model = _fit_one(X_train, best_params)

    summary = {
        "grid_results": grid_results_sorted,
        "best_params": {
            "n_components": best_params.n_components,
            "covariance_type": best_params.covariance_type,
        },
        "train_bic": float(final_model.bic(X_train)),
        "train_aic": float(final_model.aic(X_train)),
    }

    return final_model, summary


def score_loglik(model: GaussianMixture, X: np.ndarray) -> np.ndarray:
    """Per-sample log-likelihood under the fitted GMM."""
    return model.score_samples(X)


def choose_threshold(
    train_loglik: np.ndarray,
    quantile: float = 0.05,
) -> Tuple[float, Dict]:
    """
    Choose a log-likelihood threshold from training distribution.

    Args:
        train_loglik: Array of train log-likelihoods
        quantile: Lower-tail quantile to flag as overextended

    Returns:
        (threshold_value, metadata)
    """
    q = float(np.clip(quantile, 0.0, 1.0))
    thr = float(np.quantile(train_loglik, q))
    meta = {
        "quantile": q,
        "threshold": thr,
        "train_mean_loglik": float(np.mean(train_loglik)),
        "train_std_loglik": float(np.std(train_loglik)),
    }
    return thr, meta


def fit_scaler(X_train: np.ndarray) -> Tuple[StandardScaler, np.ndarray]:
    """Fit a StandardScaler on training data and return (scaler, X_train_scaled)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    return scaler, Xs


def save_artifacts(
    model: GaussianMixture,
    scaler: StandardScaler,
    config: Dict,
    outdir: str = "models",
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Save model, scaler, and config JSON.

    Files:
      - models/gmm.joblib
      - models/gmm_scaler.joblib
      - models/gmm_config.json
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    model_path = out / "gmm.joblib"
    scaler_path = out / "gmm_scaler.joblib"
    cfg_path = out / "gmm_config.json"

    dump(model, model_path)
    dump(scaler, scaler_path)

    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, default=str)

    logger.info("Saved model to %s, scaler to %s, config to %s", model_path, scaler_path, cfg_path)

    return {
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "config_path": str(cfg_path),
    }


