"""
Data helpers for the GMM overextension detector.

Responsibilities:
- Load processed bar data from data/processed/<TICKER>/*_processed.parquet
- Split into train/test by day (first 4 days train, 5th day test)
- Build feature matrices from columns prefixed with 'feat_'
"""

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd


def load_processed(
    processed_dir: str = "data/processed",
    tickers: Optional[Iterable[str]] = None,
    logger: Optional[logging.Logger] = None,
) -> pd.DataFrame:
    """
    Load all processed parquet files produced by the processing pipeline.

    Args:
        processed_dir: Root directory containing per-ticker processed files
        tickers: Optional subset of tickers to include
        logger: Optional logger

    Returns:
        Combined DataFrame of all processed bars.
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    root = Path(processed_dir)
    if not root.exists():
        raise FileNotFoundError(f"Processed directory not found: {processed_dir}")

    all_frames: List[pd.DataFrame] = []

    for subdir in root.iterdir():
        if not subdir.is_dir():
            continue

        ticker = subdir.name
        if tickers is not None and ticker not in tickers:
            continue

        parquet_files = sorted(subdir.glob("*_processed.parquet"))
        for pf in parquet_files:
            try:
                df = pd.read_parquet(pf)
                if 't' in df.columns:
                    df['t'] = pd.to_datetime(df['t'])
                if 't_end' in df.columns:
                    df['t_end'] = pd.to_datetime(df['t_end'])
                if 'date' in df.columns:
                    df['date'] = pd.to_datetime(df['date']).dt.date
                df['ticker'] = df.get('ticker', ticker)
                all_frames.append(df)
            except Exception as exc:
                logger.warning(f"Failed reading {pf}: {exc}")

    if not all_frames:
        raise RuntimeError(f"No processed parquet files found under {processed_dir}")

    combined = pd.concat(all_frames, ignore_index=True)

    core_cols = [
        'ticker', 'date', 't', 't_end',
        'open', 'high', 'low', 'close', 'volume', 'vwap',
        'label', 'label_t_end', 'label_barrier_hit'
    ]
    present_core = [c for c in core_cols if c in combined.columns]
    feature_cols = [c for c in combined.columns if c.startswith('feat_')]

    ordered_cols = present_core + [c for c in combined.columns if c not in present_core]
    combined = combined[ordered_cols]

    non_feature_cols = [c for c in combined.columns if not c.startswith('feat_')]
    combined = combined[non_feature_cols + feature_cols]

    return combined


def split_train_test_by_day(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, List[pd.Timestamp], pd.Timestamp]:
    """
    Split dataframe into train/test by day: first 4 unique days for training, 5th for testing.

    Args:
        df: Combined dataframe with a 'date' column
        logger: Optional logger

    Returns:
        (df_train, df_test, train_days, test_day)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    if 'date' not in df.columns:
        raise ValueError("Column 'date' is required in dataframe")

    # Normalize date to pandas Timestamp (date only)
    dates = pd.to_datetime(df['date']).dt.date
    unique_days = sorted(pd.unique(dates))

    if len(unique_days) < 5:
        raise ValueError(f"Expected at least 5 unique days, found {len(unique_days)}")

    train_days = unique_days[:4]
    test_day = unique_days[4]

    train_mask = dates.isin(train_days)
    test_mask = dates == test_day

    df_train = df.loc[train_mask].copy()
    df_test = df.loc[test_mask].copy()

    logger.info(
        "Train/Test split: %d train rows across %s | %d test rows on %s",
        len(df_train), [str(d) for d in train_days], len(df_test), str(test_day)
    )

    # Cast back to Timestam
    df_train['date'] = pd.to_datetime(df_train['date']).dt.date
    df_test['date'] = pd.to_datetime(df_test['date']).dt.date

    # Convert times
    if 't' in df_train.columns:
        df_train['t'] = pd.to_datetime(df_train['t'])
    if 't' in df_test.columns:
        df_test['t'] = pd.to_datetime(df_test['t'])
    if 't_end' in df_train.columns:
        df_train['t_end'] = pd.to_datetime(df_train['t_end'])
    if 't_end' in df_test.columns:
        df_test['t_end'] = pd.to_datetime(df_test['t_end'])

    # Return python date objects
    return df_train, df_test, train_days, test_day


def get_feature_matrix(
    df: pd.DataFrame,
    logger: Optional[logging.Logger] = None,
) -> Tuple[np.ndarray, List[str], pd.Index]:
    """
    Build a clean feature matrix from columns prefixed with 'feat_'.

    Drops rows containing NaN/Inf in any feature and returns the original indices
    of the retained rows for alignment with the source dataframe.

    Args:
        df: DataFrame containing feature columns
        logger: Optional logger

    Returns:
        (X, feature_names, retained_index)
    """
    if logger is None:
        logger = logging.getLogger(__name__)

    feature_cols = [c for c in df.columns if c.startswith('feat_')]
    if not feature_cols:
        raise ValueError("No feature columns found (expected columns starting with 'feat_')")

    X_df = df[feature_cols].copy()

    finite_mask = np.isfinite(X_df.to_numpy()).all(axis=1)
    cleaned = X_df.loc[finite_mask]

    dropped = (~finite_mask).sum()
    if dropped > 0:
        logger.warning("Dropped %d rows with NaN/Inf in features", dropped)

    X = cleaned.to_numpy(dtype=np.float64)
    feature_names = feature_cols
    retained_index = cleaned.index

    return X, feature_names, retained_index


