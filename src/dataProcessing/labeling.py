"""
Triple Barrier Labeling
Implements volatility-scaled triple barrier method from scratch.

Labels:
  +1: Upper barrier hit first (profit target)
  -1: Lower barrier hit first (stop loss)
   0: Time barrier hit first (timeout)
"""

import logging
from typing import Dict, Tuple

import numpy as np
import pandas as pd


def triple_barrier_labels(
    df_bars: pd.DataFrame,
    k: float = 1.0,
    time_barrier_minutes: int = 20,
    ewm_halflife: int = 50,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Apply triple barrier labeling to volume bars.
    
    Barriers are volatility-scaled:
    - Upper: p0 * (1 + k * ewm_std)
    - Lower: p0 * (1 - k * ewm_std)
    - Time: t + time_barrier_minutes
    
    Args:
        df_bars: DataFrame with volume bars
        k: Volatility multiplier for price barriers
        time_barrier_minutes: Time horizon for time barrier
        ewm_halflife: Halflife for exponential weighted volatility
        logger: Optional logger instance
        
    Returns:
        DataFrame with added label columns: label, label_t_end, label_holding_period_seconds
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if len(df_bars) == 0:
        logger.warning("No bars provided")
        return df_bars
    
    df = df_bars.copy()
    
    logger.info(f"Applying triple barrier labeling (k={k}, time_barrier={time_barrier_minutes}min)...")
    
    # Ensure timestamps are datetime
    df['t'] = pd.to_datetime(df['t'])
    df['t_end'] = pd.to_datetime(df['t_end'])
    
    # Compute 1-bar returns
    df['ret_1'] = df['close'].pct_change()
    
    # Compute exponentially weighted volatility (shifted to avoid look-ahead)
    ewm_std = df['ret_1'].ewm(halflife=ewm_halflife, min_periods=1).std().shift(1)
    
    # Fill first values with expanding std
    ewm_std.iloc[0] = df['ret_1'].iloc[:10].std() if len(df) >= 10 else 0.01
    ewm_std = ewm_std.fillna(method='bfill')
    
    df['ewm_std'] = ewm_std
    
    # Initialize label columns
    df['label'] = 0
    df['label_t_end'] = pd.NaT
    df['label_holding_period_seconds'] = 0.0
    df['label_barrier_hit'] = ''  # 'upper', 'lower', or 'time'
    
    # Label each bar
    labeled_count = 0
    
    for i in range(len(df)):
        # Get bar entry point
        t_entry = df.loc[i, 't_end']  # Use bar end time as entry
        p0 = df.loc[i, 'close']
        vol = df.loc[i, 'ewm_std']
        
        # Time barrier
        t_time_barrier = t_entry + pd.Timedelta(minutes=time_barrier_minutes)
        
        # Compute price barriers (volatility-scaled)
        if vol > 0:
            upper_barrier = p0 * (1 + k * vol)
            lower_barrier = p0 * (1 - k * vol)
        else:
            # Fallback to small fixed barrier if volatility is zero
            upper_barrier = p0 * 1.005  # 0.5%
            lower_barrier = p0 * 0.995
        
        # Scan forward to find which barrier is hit first
        label_assigned = False
        
        for j in range(i + 1, len(df)):
            bar_time = df.loc[j, 't']
            bar_high = df.loc[j, 'high']
            bar_low = df.loc[j, 'low']
            
            # Check if we've exceeded time barrier
            if bar_time >= t_time_barrier:
                # Time barrier hit
                df.loc[i, 'label'] = 0
                df.loc[i, 'label_t_end'] = t_time_barrier
                df.loc[i, 'label_holding_period_seconds'] = (
                    t_time_barrier - t_entry
                ).total_seconds()
                df.loc[i, 'label_barrier_hit'] = 'time'
                label_assigned = True
                break
            
            # Check if upper barrier hit
            if bar_high >= upper_barrier:
                df.loc[i, 'label'] = 1
                df.loc[i, 'label_t_end'] = bar_time
                df.loc[i, 'label_holding_period_seconds'] = (
                    bar_time - t_entry
                ).total_seconds()
                df.loc[i, 'label_barrier_hit'] = 'upper'
                label_assigned = True
                break
            
            # Check if lower barrier hit
            if bar_low <= lower_barrier:
                df.loc[i, 'label'] = -1
                df.loc[i, 'label_t_end'] = bar_time
                df.loc[i, 'label_holding_period_seconds'] = (
                    bar_time - t_entry
                ).total_seconds()
                df.loc[i, 'label_barrier_hit'] = 'lower'
                label_assigned = True
                break
        
        if label_assigned:
            labeled_count += 1
    
    # Drop temporary columns
    df = df.drop(columns=['ret_1'], errors='ignore')
    
    # Summary statistics
    label_dist = df['label_barrier_hit'].value_counts()
    
    logger.info(f"Labeling complete:")
    logger.info(f"  Total bars: {len(df)}")
    logger.info(f"  Labeled: {labeled_count}")
    logger.info(f"  Label distribution:")
    for barrier, count in label_dist.items():
        pct = count / len(df) * 100
        logger.info(f"    {barrier}: {count} ({pct:.1f}%)")
    
    # Compute average holding period by label
    for label_val in [-1, 0, 1]:
        mask = df['label'] == label_val
        if mask.sum() > 0:
            avg_holding = df.loc[mask, 'label_holding_period_seconds'].mean()
            logger.info(f"  Avg holding period (label={label_val:+d}): {avg_holding:.1f}s")
    
    return df


def validate_labels(
    df: pd.DataFrame,
    logger: logging.Logger = None
) -> Dict:
    """
    Validate triple barrier labels for quality.
    
    Args:
        df: DataFrame with labels
        logger: Optional logger instance
        
    Returns:
        Dictionary with validation statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    stats = {
        'total_bars': len(df),
        'labeled_bars': 0,
        'unlabeled_bars': 0,
        'label_distribution': {},
        'barrier_hit_distribution': {},
        'holding_period_stats': {}
    }
    
    if 'label_barrier_hit' in df.columns:
        # Count labeled vs unlabeled
        labeled_mask = df['label_barrier_hit'].isin(['upper', 'lower', 'time'])
        stats['labeled_bars'] = labeled_mask.sum()
        stats['unlabeled_bars'] = (~labeled_mask).sum()
        
        # Label distribution
        stats['label_distribution'] = df['label'].value_counts().to_dict()
        
        # Barrier hit distribution
        stats['barrier_hit_distribution'] = df['label_barrier_hit'].value_counts().to_dict()
        
        # Holding period statistics
        if 'label_holding_period_seconds' in df.columns:
            holding_periods = df.loc[labeled_mask, 'label_holding_period_seconds']
            if len(holding_periods) > 0:
                stats['holding_period_stats'] = {
                    'mean': float(holding_periods.mean()),
                    'median': float(holding_periods.median()),
                    'min': float(holding_periods.min()),
                    'max': float(holding_periods.max()),
                    'std': float(holding_periods.std())
                }
    
    # Check for label imbalance
    if stats['label_distribution']:
        label_counts = [stats['label_distribution'].get(i, 0) for i in [-1, 0, 1]]
        total_labeled = sum(label_counts)
        if total_labeled > 0:
            max_imbalance = max(label_counts) / total_labeled
            if max_imbalance > 0.6:
                logger.warning(f"Label imbalance detected: {max_imbalance:.1%} in dominant class")
    
    return stats


def grid_search_k(
    df_bars: pd.DataFrame,
    k_values: list,
    time_barrier_minutes: int = 20,
    ewm_halflife: int = 50,
    logger: logging.Logger = None
) -> Dict:
    """
    Grid search over k values to find optimal barrier width.
    
    Args:
        df_bars: DataFrame with volume bars
        k_values: List of k values to try
        time_barrier_minutes: Time barrier duration
        ewm_halflife: EWM halflife
        logger: Optional logger instance
        
    Returns:
        Dictionary with results for each k value
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    results = {}
    
    logger.info(f"Grid searching k values: {k_values}")
    
    for k in k_values:
        logger.info(f"\nTrying k={k}...")
        
        df_labeled = triple_barrier_labels(
            df_bars.copy(),
            k=k,
            time_barrier_minutes=time_barrier_minutes,
            ewm_halflife=ewm_halflife,
            logger=logger
        )
        
        validation_stats = validate_labels(df_labeled, logger)
        
        results[k] = {
            'label_distribution': validation_stats['label_distribution'],
            'barrier_hit_distribution': validation_stats['barrier_hit_distribution'],
            'holding_period_stats': validation_stats['holding_period_stats'],
            'labeled_bars': validation_stats['labeled_bars']
        }
    
    return results


def main():
    """Test triple barrier labeling on sample data."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("Triple Barrier Labeling Test")
    print("=" * 80)
    
    # Load config
    from config_processing import load_processing_config
    from volume_bars import build_volume_bars
    
    config = load_processing_config()
    barrier_config = config['triple_barrier']
    
    # Load and build bars
    test_file = "data/clean/CCL/CCL_10-2-25_clean.parquet"
    ticker = "CCL"
    
    print(f"\nLoading trades from: {test_file}")
    df_trades = pd.read_parquet(test_file)
    
    print(f"Building volume bars...")
    volume_threshold = config['volume_bars']['thresholds'][ticker]
    df_bars = build_volume_bars(df_trades, volume_threshold, logger)
    print(f"Built {len(df_bars)} bars")
    
    # Apply labeling
    print(f"\nApplying triple barrier labeling...")
    print(f"  k = {barrier_config['k']}")
    print(f"  time_barrier = {barrier_config['time_barrier_minutes']} minutes")
    print(f"  ewm_halflife = {barrier_config['ewm_halflife']} bars")
    
    df_labeled = triple_barrier_labels(
        df_bars,
        k=barrier_config['k'],
        time_barrier_minutes=barrier_config['time_barrier_minutes'],
        ewm_halflife=barrier_config['ewm_halflife'],
        logger=logger
    )
    
    # Validate
    print(f"\nValidating labels...")
    validation_stats = validate_labels(df_labeled, logger)
    
    print(f"\nValidation Summary:")
    print(f"  Total bars: {validation_stats['total_bars']}")
    print(f"  Labeled bars: {validation_stats['labeled_bars']}")
    print(f"  Unlabeled bars: {validation_stats['unlabeled_bars']}")
    
    print(f"\n  Label distribution:")
    for label, count in sorted(validation_stats['label_distribution'].items()):
        pct = count / validation_stats['total_bars'] * 100
        print(f"    {label:+2d}: {count:4d} ({pct:5.1f}%)")
    
    print(f"\n  Holding period statistics:")
    for stat, value in validation_stats['holding_period_stats'].items():
        print(f"    {stat}: {value:.2f} seconds")
    
    # Display sample
    print(f"\nSample labeled bars:")
    sample_cols = ['t', 'close', 'ewm_std', 'label', 'label_barrier_hit', 'label_holding_period_seconds']
    print(df_labeled[sample_cols].head(10))
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

