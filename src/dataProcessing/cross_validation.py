"""
Purged K-Fold Cross-Validation with Embargo
Implements LODO (Leave-One-Day-Out).
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def create_cv_splits(
    df_all: pd.DataFrame,
    embargo_minutes: int = 20,
    day_column: str = 'date',
    time_start_column: str = 't',
    time_end_column: str = 'label_t_end',
    logger: logging.Logger = None
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Create purged k-fold CV splits at the day level.
    
    Uses LODO strategy:
    - Each fold validates on 1 day, trains on the other 4 days
    - Purge: Remove training samples where [t, t_end] overlaps validation period
    - Embargo: Remove training samples within embargo_minutes after validation period
    
    Args:
        df_all: DataFrame with all processed data
        embargo_minutes: Embargo period after validation (minutes)
        day_column: Column name for date grouping
        time_start_column: Column name for bar start time
        time_end_column: Column name for label end time (t_end)
        logger: Optional logger instance
        
    Returns:
        (cv_splits, metadata_df)
        - cv_splits: List of dicts with train/val indices and metadata
        - metadata_df: DataFrame with fold metadata
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Creating purged k-fold CV splits (day-level LODO)...")
    
    # Ensure we have necessary columns
    required_cols = [day_column, time_start_column, time_end_column]
    for col in required_cols:
        if col not in df_all.columns:
            raise ValueError(f"Required column '{col}' not found in dataframe")
    
    # Get unique days
    df_all = df_all.copy()
    df_all[day_column] = pd.to_datetime(df_all[day_column])
    unique_days = sorted(df_all[day_column].unique())
    
    n_days = len(unique_days)
    logger.info(f"Found {n_days} unique days: {[d.strftime('%Y-%m-%d') for d in unique_days]}")
    
    cv_splits = []
    
    # Create one fold per day (LODO)
    for fold_idx, test_day in enumerate(unique_days):
        logger.info(f"\nFold {fold_idx + 1}/{n_days}: Test day = {test_day.strftime('%Y-%m-%d')}")
        
        # Validation set: all samples from test_day
        val_mask = df_all[day_column] == test_day
        val_indices = df_all.index[val_mask].tolist()
        
        # Training set: all samples from other days
        train_mask = df_all[day_column] != test_day
        train_indices = df_all.index[train_mask].tolist()
        
        logger.info(f"  Initial: {len(train_indices)} train, {len(val_indices)} val")
        
        # Get validation period boundaries
        val_start = df_all.loc[val_mask, time_start_column].min()
        
        # Handle NaT in label_t_end by using t_end as fallback
        val_end_series = df_all.loc[val_mask, time_end_column]
        # Filter out NaT values before taking max
        valid_val_ends = val_end_series.dropna()
        
        if len(valid_val_ends) > 0:
            val_end = valid_val_ends.max()
        else:
            # Fallback to t_end if all label_t_end are NaT
            val_end = df_all.loc[val_mask, 't_end'].max()
        
        logger.info(f"  Validation period: {val_start} to {val_end}")
        
        # Apply purging: remove training samples where [t, t_end] overlaps [val_start, val_end]
        purged_indices = []
        
        for idx in train_indices:
            sample_start = df_all.loc[idx, time_start_column]
            sample_end = df_all.loc[idx, time_end_column]
            
            # Handle NaT: use t_end as fallback
            if pd.isna(sample_end):
                sample_end = df_all.loc[idx, 't_end']
            
            # Skip if still NaT (shouldn't happen but be safe)
            if pd.isna(sample_end):
                continue
            
            # Check for overlap: [sample_start, sample_end] ∩ [val_start, val_end]
            overlaps = (sample_start <= val_end) and (sample_end >= val_start)
            
            if overlaps:
                purged_indices.append(idx)
        
        # Remove purged samples from training set
        train_indices = [idx for idx in train_indices if idx not in purged_indices]
        
        logger.info(f"  Purged: {len(purged_indices)} samples")
        
        # Apply embargo: remove training samples within embargo_minutes after val_end
        embargo_cutoff = val_end + pd.Timedelta(minutes=embargo_minutes)
        
        embargoed_indices = []
        
        for idx in train_indices:
            sample_start = df_all.loc[idx, time_start_column]
            
            # Check if sample starts within embargo period
            if sample_start <= embargo_cutoff:
                # Only embargo samples that come after validation period
                if sample_start > val_end:
                    embargoed_indices.append(idx)
        
        # Remove embargoed samples from training set
        train_indices = [idx for idx in train_indices if idx not in embargoed_indices]
        
        logger.info(f"  Embargoed: {len(embargoed_indices)} samples")
        logger.info(f"  Final: {len(train_indices)} train, {len(val_indices)} val")
        
        # Store fold information
        fold_info = {
            'fold': fold_idx,
            'test_day': test_day,
            'train_indices': train_indices,
            'val_indices': val_indices,
            'purged_indices': purged_indices,
            'embargoed_indices': embargoed_indices,
            'n_train': len(train_indices),
            'n_val': len(val_indices),
            'n_purged': len(purged_indices),
            'n_embargoed': len(embargoed_indices),
            'val_start': val_start,
            'val_end': val_end,
            'embargo_cutoff': embargo_cutoff
        }
        
        cv_splits.append(fold_info)
    
    # Create metadata DataFrame
    metadata_rows = []
    
    for fold_info in cv_splits:
        metadata_rows.append({
            'fold': fold_info['fold'],
            'test_day': fold_info['test_day'],
            'n_train': fold_info['n_train'],
            'n_val': fold_info['n_val'],
            'n_purged': fold_info['n_purged'],
            'n_embargoed': fold_info['n_embargoed'],
            'val_start': fold_info['val_start'],
            'val_end': fold_info['val_end'],
            'embargo_cutoff': fold_info['embargo_cutoff']
        })
    
    metadata_df = pd.DataFrame(metadata_rows)
    
    logger.info(f"\nCV splits created: {len(cv_splits)} folds")
    logger.info(f"Average train size: {metadata_df['n_train'].mean():.0f}")
    logger.info(f"Average val size: {metadata_df['n_val'].mean():.0f}")
    logger.info(f"Average purged: {metadata_df['n_purged'].mean():.0f}")
    logger.info(f"Average embargoed: {metadata_df['n_embargoed'].mean():.0f}")
    
    return cv_splits, metadata_df


def validate_cv_splits(
    df_all: pd.DataFrame,
    cv_splits: List[Dict],
    logger: logging.Logger = None
) -> Dict:
    """
    Validate CV splits for data leakage and consistency.
    
    Args:
        df_all: DataFrame with all data
        cv_splits: List of CV fold dictionaries
        logger: Optional logger instance
        
    Returns:
        Dictionary with validation statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("Validating CV splits...")
    
    stats = {
        'n_folds': len(cv_splits),
        'total_samples': len(df_all),
        'folds_valid': [],
        'issues': []
    }
    
    for fold_info in cv_splits:
        fold = fold_info['fold']
        train_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        
        # Check 1: No overlap between train and val
        overlap = set(train_idx) & set(val_idx)
        if overlap:
            stats['issues'].append(f"Fold {fold}: Train/val overlap ({len(overlap)} samples)")
        
        # Check 2: All indices are valid
        all_idx = train_idx + val_idx
        invalid_idx = [i for i in all_idx if i not in df_all.index]
        if invalid_idx:
            stats['issues'].append(f"Fold {fold}: Invalid indices ({len(invalid_idx)})")
        
        # Check 3: Coverage (train + val should be substantial)
        coverage = len(all_idx) / len(df_all)
        if coverage < 0.5:
            stats['issues'].append(f"Fold {fold}: Low coverage ({coverage:.1%})")
        
        if not overlap and not invalid_idx:
            stats['folds_valid'].append(fold)
    
    stats['all_folds_valid'] = len(stats['folds_valid']) == stats['n_folds']
    
    if stats['issues']:
        logger.warning(f"CV validation found {len(stats['issues'])} issues:")
        for issue in stats['issues']:
            logger.warning(f"  - {issue}")
    else:
        logger.info("✓ All CV splits are valid")
    
    return stats


def visualize_cv_splits(
    df_all: pd.DataFrame,
    cv_splits: List[Dict],
    output_path: str = None
) -> None:
    """
    Visualize CV splits showing train/val/purge/embargo periods.
    
    Args:
        df_all: DataFrame with all data
        cv_splits: List of CV fold dictionaries
        output_path: Optional path to save figure
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    
    fig, axes = plt.subplots(len(cv_splits), 1, figsize=(14, 2 * len(cv_splits)))
    
    if len(cv_splits) == 1:
        axes = [axes]
    
    for fold_idx, (fold_info, ax) in enumerate(zip(cv_splits, axes)):
        test_day = fold_info['test_day']
        train_idx = fold_info['train_indices']
        val_idx = fold_info['val_indices']
        purged_idx = fold_info['purged_indices']
        embargoed_idx = fold_info['embargoed_indices']
        
        # Plot train samples
        train_times = df_all.loc[train_idx, 't']
        ax.scatter(train_times, [1] * len(train_times), c='blue', alpha=0.3, s=1, label='Train')
        
        # Plot validation samples
        val_times = df_all.loc[val_idx, 't']
        ax.scatter(val_times, [2] * len(val_times), c='green', alpha=0.5, s=2, label='Val')
        
        # Plot purged samples
        if purged_idx:
            purged_times = df_all.loc[purged_idx, 't']
            ax.scatter(purged_times, [1.5] * len(purged_times), c='red', alpha=0.3, s=1, label='Purged')
        
        # Plot embargoed samples
        if embargoed_idx:
            embargoed_times = df_all.loc[embargoed_idx, 't']
            ax.scatter(embargoed_times, [1.3] * len(embargoed_times), c='orange', alpha=0.3, s=1, label='Embargoed')
        
        # Mark embargo cutoff
        ax.axvline(fold_info['embargo_cutoff'], color='orange', linestyle='--', alpha=0.5)
        
        ax.set_ylabel(f"Fold {fold_idx}")
        ax.set_ylim(0.5, 2.5)
        ax.set_yticks([1, 2])
        ax.set_yticklabels(['Train', 'Val'])
        ax.legend(loc='upper right', fontsize=8)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.set_title(f"Fold {fold_idx}: Test Day = {test_day.strftime('%Y-%m-%d')}")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved CV visualization to {output_path}")
    
    plt.show()


def main():
    """Test CV split creation on sample data."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("Cross-Validation Split Test")
    print("=" * 80)
    
    # Load config
    from config_processing import load_processing_config
    from volume_bars import build_volume_bars_by_day
    from labeling import triple_barrier_labels
    
    config = load_processing_config()
    
    # Load multiple days for same ticker
    ticker = "CCL"
    dates = ["10-2-25", "10-3-25", "10-6-25"]
    
    all_bars = []
    
    for date in dates:
        file_path = f"data/clean/{ticker}/{ticker}_{date}_clean.parquet"
        print(f"\nLoading: {file_path}")
        
        df_trades = pd.read_parquet(file_path)
        
        # Build bars
        volume_threshold = config['volume_bars']['thresholds'][ticker]
        df_bars = build_volume_bars_by_day(df_trades, volume_threshold, logger)
        
        # Label
        df_labeled = triple_barrier_labels(
            df_bars,
            k=config['triple_barrier']['k'],
            time_barrier_minutes=config['triple_barrier']['time_barrier_minutes'],
            ewm_halflife=config['triple_barrier']['ewm_halflife'],
            logger=logger
        )
        
        all_bars.append(df_labeled)
    
    # Combine all data
    df_all = pd.concat(all_bars, ignore_index=True)
    print(f"\nTotal bars: {len(df_all)}")
    print(f"Days: {df_all['date'].nunique()}")
    
    # Create CV splits
    print(f"\n{'-' * 80}")
    cv_splits, metadata_df = create_cv_splits(
        df_all,
        embargo_minutes=config['cv']['embargo_minutes'],
        logger=logger
    )
    
    # Validate
    print(f"\n{'-' * 80}")
    validation_stats = validate_cv_splits(df_all, cv_splits, logger)
    
    # Display metadata
    print(f"\n{'-' * 80}")
    print("CV Fold Metadata:")
    print(metadata_df.to_string())
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

