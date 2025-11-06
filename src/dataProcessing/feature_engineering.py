"""
Feature Engineering for Volume Bars
Computes features avoiding look-ahead bias through proper shifting.
"""

import logging
from typing import Dict

import numpy as np
import pandas as pd


def engineer_features(
    df_bars: pd.DataFrame,
    config: Dict = None,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Engineer features from volume bars, avoiding look-ahead bias.
    
    Features:
    1. VWAP distance (z-score)
    2. Bollinger position
    3. Short-term momentum (3-bar, 5-bar returns)
    4. Relative volume
    5. Time of day
    6. 5-minute context features
    
    Args:
        df_bars: DataFrame with volume bars
        config: Configuration dictionary
        logger: Optional logger instance
        
    Returns:
        DataFrame with added feature columns
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if len(df_bars) == 0:
        logger.warning("No bars provided")
        return df_bars
    
    if config is None:
        config = {
            'vwap_zscore_window': 20,
            'bollinger_window': 20,
            'bollinger_std': 2.0,
            'momentum_windows': [3, 5],
            'relative_volume_window': 20,
            'context_window_minutes': 5
        }
    
    df = df_bars.copy()
    
    logger.info(f"Engineering features for {len(df)} bars...")
    
    # 1. VWAP Distance (z-score)
    df['vwap_distance'] = df['close'] - df['vwap']
    
    # Rolling std with shift to avoid look-ahead
    vwap_dist_std = df['vwap_distance'].rolling(
        window=config['vwap_zscore_window'], 
        min_periods=1
    ).std().shift(1)
    
    df['feat_vwap_zscore'] = np.where(
        vwap_dist_std > 0,
        df['vwap_distance'] / vwap_dist_std,
        0
    )
    
    # 2. Bollinger Bands Position
    bb_window = config['bollinger_window']
    bb_std_mult = config['bollinger_std']
    
    bb_mid = df['close'].rolling(window=bb_window, min_periods=1).mean().shift(1)
    bb_std = df['close'].rolling(window=bb_window, min_periods=1).std().shift(1)
    
    bb_upper = bb_mid + bb_std_mult * bb_std
    bb_lower = bb_mid - bb_std_mult * bb_std
    
    # Position within bands: -1 (at lower) to +1 (at upper)
    bb_range = bb_upper - bb_lower
    df['feat_bollinger_position'] = np.where(
        bb_range > 0,
        (df['close'] - bb_mid) / (bb_range / 2),
        0
    )
    
    # Clip to reasonable range
    df['feat_bollinger_position'] = df['feat_bollinger_position'].clip(-3, 3)
    
    # 3. Short-term Momentum
    for window in config['momentum_windows']:
        returns = df['close'].pct_change(periods=window)
        df[f'feat_momentum_{window}bar'] = returns
    
    # 4. Relative Volume
    vol_window = config['relative_volume_window']
    
    avg_volume = df['volume'].rolling(window=vol_window, min_periods=1).mean().shift(1)
    
    df['feat_relative_volume'] = np.where(
        avg_volume > 0,
        df['volume'] / avg_volume,
        1.0
    )
    
    # 5. Time of Day
    # Convert to market time (Eastern) and compute minutes since market open
    df_times = pd.to_datetime(df['t']).dt.tz_convert('America/New_York')
    
    market_open = pd.Timestamp('09:35:00').time()
    
    seconds_since_midnight = (
        df_times.dt.hour * 3600 + 
        df_times.dt.minute * 60 + 
        df_times.dt.second
    )
    
    market_open_seconds = 9 * 3600 + 35 * 60
    
    minutes_since_open = (seconds_since_midnight - market_open_seconds) / 60
    
    df['feat_time_of_day'] = (minutes_since_open / 360).clip(0, 1)
    
    # 6. 5-Minute Context Features
    context_window = pd.Timedelta(minutes=config['context_window_minutes'])
    
    # Initialize context features
    df['feat_context_bar_count'] = 0
    df['feat_context_avg_volume'] = df['volume']
    df['feat_context_price_range'] = 0.0
    
    # Compute context features using rolling time window
    for i in range(len(df)):
        current_time = pd.to_datetime(df.loc[i, 't_end'])
        lookback_time = current_time - context_window
        
        # Find bars within lookback window (excluding current bar)
        mask = (
            (pd.to_datetime(df['t_end']) >= lookback_time) &
            (pd.to_datetime(df['t_end']) < current_time)
        )
        
        context_bars = df.loc[mask]
        
        if len(context_bars) > 0:
            df.loc[i, 'feat_context_bar_count'] = len(context_bars)
            df.loc[i, 'feat_context_avg_volume'] = context_bars['volume'].mean()
            
            # Price range as (high - low) / close
            price_range = (context_bars['high'].max() - context_bars['low'].min())
            df.loc[i, 'feat_context_price_range'] = price_range / df.loc[i, 'close']
    
    df = df.drop(columns=['vwap_distance'], errors='ignore')
    
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    for col in feature_cols:
        df[col] = df[col].astype('float32')
    
    logger.info(f"Engineered {len(feature_cols)} features")
    
    return df


def validate_features(
    df: pd.DataFrame,
    logger: logging.Logger = None
) -> Dict:
    """
    Validate engineered features for quality.
    
    Args:
        df: DataFrame with features
        logger: Optional logger instance
        
    Returns:
        Dictionary with validation statistics
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    feature_cols = [col for col in df.columns if col.startswith('feat_')]
    
    stats = {
        'total_features': len(feature_cols),
        'features_with_nan': {},
        'features_with_inf': {},
        'feature_ranges': {}
    }
    
    for col in feature_cols:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            stats['features_with_nan'][col] = nan_count
        
        inf_count = np.isinf(df[col]).sum()
        if inf_count > 0:
            stats['features_with_inf'][col] = inf_count
        
        if len(df) > 0:
            stats['feature_ranges'][col] = {
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'mean': float(df[col].mean()),
                'std': float(df[col].std())
            }
    
    # Log warnings
    if stats['features_with_nan']:
        logger.warning(f"Features with NaN values: {stats['features_with_nan']}")
    
    if stats['features_with_inf']:
        logger.warning(f"Features with inf values: {stats['features_with_inf']}")
    
    return stats


def main():
    """Test feature engineering on sample data."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("Feature Engineering Test")
    print("=" * 80)
    
    # Load config
    from config_processing import load_processing_config
    
    config = load_processing_config()
    feature_config = config['features']
    
    # Load volume bars
    from volume_bars import build_volume_bars
    
    test_file = "data/clean/CCL/CCL_10-2-25_clean.parquet"
    ticker = "CCL"
    
    print(f"\nLoading trades from: {test_file}")
    df_trades = pd.read_parquet(test_file)
    
    print(f"Building volume bars...")
    volume_threshold = config['volume_bars']['thresholds'][ticker]
    df_bars = build_volume_bars(df_trades, volume_threshold, logger)
    
    print(f"Built {len(df_bars)} bars")
    
    # Engineer features
    print(f"\nEngineering features...")
    df_features = engineer_features(df_bars, feature_config, logger)
    
    # Validate
    print(f"\nValidating features...")
    validation_stats = validate_features(df_features, logger)
    
    print(f"\nFeature Summary:")
    print(f"  Total features: {validation_stats['total_features']}")
    print(f"  Features with NaN: {len(validation_stats['features_with_nan'])}")
    print(f"  Features with inf: {len(validation_stats['features_with_inf'])}")
    
    print(f"\nFeature Ranges:")
    for feat, ranges in validation_stats['feature_ranges'].items():
        print(f"  {feat}:")
        print(f"    Range: [{ranges['min']:.4f}, {ranges['max']:.4f}]")
        print(f"    Mean: {ranges['mean']:.4f}, Std: {ranges['std']:.4f}")
    
    # Display sample
    feature_cols = [col for col in df_features.columns if col.startswith('feat_')]
    print(f"\nFirst 5 bars with features:")
    print(df_features[['t', 'close', 'volume'] + feature_cols].head())
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

