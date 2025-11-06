"""
Evaluation and visualization utilities for the GMM overextension detector.

Key functions:
- evaluate_gmm: compute LLKs, flags, metrics; persist candidates
- make_plots: generate selection and diagnostic plots
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def evaluate_gmm(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    idx_train: pd.Index,
    idx_test: pd.Index,
    X_train_scaled: np.ndarray,
    X_test_scaled: np.ndarray,
    model: GaussianMixture,
    threshold: float,
    feature_names: Optional[List[str]] = None,
    processed_outdir: str = "data/processed",
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Score train/test sets, flag overextended candidates, compute metrics, and
    persist the combined candidate set for downstream RF training.

    Args:
        df_train, df_test: Source dataframes
        idx_train, idx_test: Indices of rows used in X matrices
        X_train_scaled, X_test_scaled: Scaled features
        model: Fitted GMM
        threshold: Log-likelihood threshold (lower tail)
        feature_names: Optional feature column names for persistence
        processed_outdir: Where to write gmm_candidates.parquet
        logger: Optional logger

    Returns:
        metrics dict
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    # Compute log-likelihoods and components
    train_llk = model.score_samples(X_train_scaled)
    test_llk = model.score_samples(X_test_scaled)
    train_resp = model.predict_proba(X_train_scaled)
    test_resp = model.predict_proba(X_test_scaled)

    train_comp = np.argmax(train_resp, axis=1)
    test_comp = np.argmax(test_resp, axis=1)

    # Attach to dataframes
    tr = df_train.loc[idx_train].copy()
    te = df_test.loc[idx_test].copy()
    tr["gmm_loglik"] = train_llk
    tr["gmm_component"] = train_comp
    tr["gmm_is_candidate"] = tr["gmm_loglik"] <= threshold

    te["gmm_loglik"] = test_llk
    te["gmm_component"] = test_comp
    te["gmm_is_candidate"] = te["gmm_loglik"] <= threshold

    # Metrics
    metrics = {
        "train_mean_loglik": float(np.mean(train_llk)),
        "test_mean_loglik": float(np.mean(test_llk)),
        "train_candidate_rate": float(np.mean(tr["gmm_is_candidate"].astype(float))),
        "test_candidate_rate": float(np.mean(te["gmm_is_candidate"].astype(float))),
        "n_train": int(len(tr)),
        "n_test": int(len(te)),
        "n_components": int(model.n_components),
    }

    # Label diagnostics if present
    if "label" in tr.columns:
        metrics["train_label_dist_flagged"] = tr.loc[tr["gmm_is_candidate"], "label"].value_counts().to_dict()
        metrics["train_label_dist_unflagged"] = tr.loc[~tr["gmm_is_candidate"], "label"].value_counts().to_dict()
    if "label" in te.columns:
        metrics["test_label_dist_flagged"] = te.loc[te["gmm_is_candidate"], "label"].value_counts().to_dict()
        metrics["test_label_dist_unflagged"] = te.loc[~te["gmm_is_candidate"], "label"].value_counts().to_dict()

    # Persist candidates
    out_dir = Path(processed_outdir)
    _ensure_dir(out_dir)

    cand_cols = ["ticker", "date", "t", "t_end", "gmm_loglik", "gmm_is_candidate", "gmm_component"]
    if "label" in tr.columns:
        cand_cols.append("label")

    feature_cols = feature_names if feature_names is not None else [c for c in tr.columns if c.startswith("feat_")]
    cand_cols_extended = cand_cols + [c for c in feature_cols if c in tr.columns]

    candidates = pd.concat([
        tr.loc[:, [c for c in cand_cols_extended if c in tr.columns]],
        te.loc[:, [c for c in cand_cols_extended if c in te.columns]],
    ], ignore_index=True)

    out_path = out_dir / "gmm_candidates.parquet"
    candidates.to_parquet(out_path, index=False)
    logger.info("Saved GMM candidates to %s (%d rows, %.1f%% flagged)", out_path, len(candidates), 100 * candidates["gmm_is_candidate"].mean())

    return metrics


def make_plots(
    grid_summary: Dict,
    tr_loglik: np.ndarray,
    te_loglik: np.ndarray,
    X_test_scaled: np.ndarray,
    test_flags: np.ndarray,
    reports_dir: str = "reports/gmm",
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Generate a set of plots for the report.

    Args:
        grid_summary: Output of tune_and_train_gmm (with grid_results)
        tr_loglik, te_loglik: Train/test log-likelihood arrays
        X_test_scaled: Scaled test features for PCA scatter
        test_flags: Boolean flags for test set
        reports_dir: Where to write images
        logger: Optional logger

    Returns:
        dict of written figure paths
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    out = Path(reports_dir)
    _ensure_dir(out)
    written = {}

    # 1) Model selection curve (CV mean LLK vs k by cov type)
    try:
        df_grid = pd.DataFrame(grid_summary.get("grid_results", []))
        if not df_grid.empty:
            plt.figure(figsize=(10, 6))
            for cov, sub in df_grid.groupby("covariance_type"):
                sub_sorted = sub.sort_values("n_components")
                plt.plot(sub_sorted["n_components"], sub_sorted["cv_mean_loglik"], marker="o", label=f"cov={cov}")
            plt.xlabel("n_components")
            plt.ylabel("CV mean log-likelihood")
            plt.title("GMM model selection (LODO CV)")
            plt.legend()
            path = out / "model_selection_cv_llk.png"
            plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
            written["model_selection"] = str(path)
    except Exception as exc:
        logger.warning("Failed to plot model selection: %s", exc)

    # 2) Log-likelihood distributions (train vs test)
    try:
        plt.figure(figsize=(10, 6))
        sns.kdeplot(tr_loglik, fill=True, alpha=0.4, label="train")
        sns.kdeplot(te_loglik, fill=True, alpha=0.4, label="test")
        plt.xlabel("log-likelihood")
        plt.title("Train vs Test log-likelihood")
        plt.legend()
        path = out / "llk_train_vs_test.png"
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        written["llk_dist"] = str(path)
    except Exception as exc:
        logger.warning("Failed to plot LLK distributions: %s", exc)

    # 3) PCA scatter of test colored by flag
    try:
        pca = PCA(n_components=2, random_state=42)
        Z = pca.fit_transform(X_test_scaled)
        plt.figure(figsize=(10, 6))
        plt.scatter(Z[~test_flags, 0], Z[~test_flags, 1], s=6, c="#4c78a8", alpha=0.5, label="not flagged")
        plt.scatter(Z[test_flags, 0], Z[test_flags, 1], s=10, c="#f58518", alpha=0.8, label="flagged")
        plt.xlabel("PCA 1")
        plt.ylabel("PCA 2")
        plt.title("Test PCA (colored by GMM candidate flag)")
        plt.legend()
        path = out / "pca_test_flags.png"
        plt.tight_layout(); plt.savefig(path, dpi=150); plt.close()
        written["pca_flags"] = str(path)
    except Exception as exc:
        logger.warning("Failed to plot PCA scatter: %s", exc)

    return written


