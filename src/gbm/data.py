"""
Data loading and preparation for GBM quantile regression exit model.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_rf_scores(
    data_path: str = "data/randomForest/rf_opportunity_scores.parquet",
    processed_dir: str = "data/processed",
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load RF opportunity scores parquet file and merge with processed data
    to get ewm_std for computing forward returns.

    Args:
        data_path: Path to the parquet file with RF scores
        processed_dir: Directory containing processed parquet files per ticker
        logger: Optional logger

    Returns:
        DataFrame with RF scores, features, and ewm_std
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"RF scores file not found: {path}")

    df = pd.read_parquet(path)
    logger.info(f"Loaded RF scores: {len(df):,} rows from {path}")

    # Check if ewm_std is already present
    if "ewm_std" in df.columns:
        logger.info("ewm_std column already present in RF scores")
        return df

    # Need to load ewm_std from processed data
    logger.info("Loading ewm_std from processed data...")

    processed_path = Path(processed_dir)
    if not processed_path.exists():
        raise FileNotFoundError(f"Processed data directory not found: {processed_path}")

    # Load all processed parquet files
    processed_dfs = []
    for ticker_dir in processed_path.iterdir():
        if ticker_dir.is_dir():
            for pq_file in ticker_dir.glob("*_processed.parquet"):
                try:
                    pq_df = pd.read_parquet(pq_file)
                    # Only keep columns needed for merge
                    keep_cols = ["ticker", "t", "ewm_std"]
                    if "date" in pq_df.columns:
                        keep_cols.append("date")
                    pq_df = pq_df[[c for c in keep_cols if c in pq_df.columns]].copy()
                    processed_dfs.append(pq_df)
                except Exception as exc:
                    logger.warning(f"Failed to load {pq_file}: {exc}")

    if not processed_dfs:
        raise FileNotFoundError(f"No processed parquet files found in {processed_path}")

    processed_all = pd.concat(processed_dfs, ignore_index=True)
    logger.info(f"Loaded {len(processed_all):,} rows from processed data")

    # Merge on ticker and timestamp
    if "t" in df.columns:
        df["t"] = pd.to_datetime(df["t"])
    if "t" in processed_all.columns:
        processed_all["t"] = pd.to_datetime(processed_all["t"])

    # Merge
    df = df.merge(
        processed_all[["ticker", "t", "ewm_std"]].drop_duplicates(subset=["ticker", "t"]),
        on=["ticker", "t"],
        how="left",
    )

    n_missing = df["ewm_std"].isna().sum()
    if n_missing > 0:
        logger.warning(f"{n_missing} rows have missing ewm_std after merge")

    logger.info(f"Merged ewm_std: {df['ewm_std'].notna().sum():,} rows have ewm_std")

    return df


def compute_forward_returns(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Compute forward returns as the target variable for quantile regression.

    The forward return approximates the realized return from entry to exit:
    - For upper barrier hits (label=+1): return = +ewm_std 
    - For lower barrier hits (label=-1): return = -ewm_std 
    - For time barrier hits (label=0): return = 0

    This approximation is valid because the triple-barrier method sets
    upper/lower barriers at +/- 1x ewm_std from entry price.

    Args:
        df: DataFrame with label and ewm_std columns
        logger: Optional logger

    Returns:
        DataFrame with forward_return column added
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    df = df.copy()

    # Check required columns
    required_cols = ["label", "ewm_std"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Compute forward return based on label and volatility
    # Upper barrier = +1 vol unit, Lower = -1 vol unit, Time = 0
    df["forward_return"] = df["label"] * df["ewm_std"]

    # Log summary statistics
    logger.info(
        "Forward return stats: mean=%.6f, std=%.6f, min=%.6f, max=%.6f",
        df["forward_return"].mean(),
        df["forward_return"].std(),
        df["forward_return"].min(),
        df["forward_return"].max(),
    )

    # Distribution by label
    for lbl in sorted(df["label"].unique()):
        subset = df[df["label"] == lbl]["forward_return"]
        logger.info(
            "  Label %+d: n=%d, mean=%.6f, std=%.6f",
            lbl, len(subset), subset.mean(), subset.std()
        )

    return df


def prepare_gbm_dataset(
    df: pd.DataFrame,
    rf_proba_threshold: float = 0.5,
    feature_cols: Optional[List[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict:
    """
    Prepare dataset for GBM training: filter by RF probability, split by day, scale features.

    Args:
        df: DataFrame with features, RF probability, and forward returns
        rf_proba_threshold: Minimum RF probability to include (default 0.5)
        feature_cols: List of feature columns (default: all cols starting with 'feat_')
        logger: Optional logger

    Returns:
        Dictionary with:
        - X_train, y_train: Training features and targets (scaled)
        - X_test, y_test: Test features and targets (scaled)
        - X_train_raw, X_test_raw: Unscaled features
        - scaler: Fitted StandardScaler
        - train_days, test_day: Day identifiers
        - train_day_groups: Day array for LODO CV
        - feature_names: List of feature column names
        - df_train, df_test: Filtered DataFrames
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    df = df.copy()

    # Ensure date column is datetime
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = df["date"].dt.date
    elif "day" not in df.columns:
        raise ValueError("DataFrame must have 'date' or 'day' column")

    # Filter to high-probability reversion events
    if "rf_proba_reversion" not in df.columns:
        raise ValueError("Missing 'rf_proba_reversion' column")

    n_before = len(df)
    df = df[df["rf_proba_reversion"] >= rf_proba_threshold].copy()
    logger.info(
        "Filtered to RF proba >= %.2f: %d -> %d rows (%.1f%% retained)",
        rf_proba_threshold, n_before, len(df), 100 * len(df) / n_before
    )

    # Determine feature columns
    if feature_cols is None:
        feature_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feature_cols:
        raise ValueError("No feature columns found")

    logger.info(f"Using {len(feature_cols)} features: {feature_cols}")

    # Check for target column
    if "forward_return" not in df.columns:
        raise ValueError("Missing 'forward_return' column - run compute_forward_returns first")

    # Drop rows with NaN in features or target
    required = feature_cols + ["forward_return"]
    n_before = len(df)
    df = df.dropna(subset=required).copy()
    if len(df) < n_before:
        logger.warning(
            "Dropped %d rows with NaN in features/target; remaining %d",
            n_before - len(df), len(df)
        )

    # Train/test split by day (first 4 days train, 5th day test)
    unique_days = sorted(df["day"].unique())
    if len(unique_days) < 2:
        raise ValueError(f"Need at least 2 unique days, got {len(unique_days)}")

    train_days = unique_days[:-1]
    test_day = unique_days[-1]

    logger.info(f"Train days: {[str(d) for d in train_days]} | Test day: {test_day}")

    train_mask = df["day"].isin(train_days)
    test_mask = df["day"] == test_day

    df_train = df[train_mask].copy()
    df_test = df[test_mask].copy()

    logger.info(f"Train samples: {len(df_train):,} | Test samples: {len(df_test):,}")

    # Extract features and targets
    X_train_raw = df_train[feature_cols].values.astype(float)
    X_test_raw = df_test[feature_cols].values.astype(float)
    y_train = df_train["forward_return"].values.astype(float)
    y_test = df_test["forward_return"].values.astype(float)

    # Standardize features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    logger.info("Standardized features using StandardScaler (fit on train)")

    # Day groups for LODO CV
    train_day_groups = df_train["day"].values

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_test": X_test,
        "y_test": y_test,
        "X_train_raw": X_train_raw,
        "X_test_raw": X_test_raw,
        "scaler": scaler,
        "train_days": train_days,
        "test_day": test_day,
        "train_day_groups": train_day_groups,
        "feature_names": feature_cols,
        "df_train": df_train,
        "df_test": df_test,
    }
