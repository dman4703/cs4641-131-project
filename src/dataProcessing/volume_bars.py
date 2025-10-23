"""
Volume Bar Construction
Converts tick data into volume-based bars targeting ~600 bars per day.
"""

import logging
from typing import Tuple

import numpy as np
import pandas as pd


def build_volume_bars(
    df_trades: pd.DataFrame,
    volume_threshold: int,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Construct volume bars from tick data.
    
    A new bar is formed when cumulative volume reaches the threshold.
    Each bar contains: open, high, low, close, volume, vwap, timestamps, trade count.
    
    Args:
        df_trades: DataFrame with columns ['ts', 'price', 'size', 'ticker']
        volume_threshold: Volume required to trigger a new bar
        logger: Optional logger instance
        
    Returns:
        DataFrame with volume bars, one row per bar
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if len(df_trades) == 0:
        logger.warning("No trades provided")
        return pd.DataFrame()
    
    # Ensure sorted by timestamp
    df_trades = df_trades.sort_values('ts').reset_index(drop=True)
    
    bars = []
    
    # Accumulator for current bar
    cum_volume = 0
    cum_dollar_volume = 0
    bar_trades = []
    
    for idx, trade in df_trades.iterrows():
        # Add trade to current bar
        bar_trades.append(trade)
        cum_volume += trade['size']
        cum_dollar_volume += trade['price'] * trade['size']
        
        # Check if bar is complete
        if cum_volume >= volume_threshold:
            # Compute bar statistics
            prices = [t['price'] for t in bar_trades]
            volumes = [t['size'] for t in bar_trades]
            timestamps = [t['ts'] for t in bar_trades]
            
            bar = {
                't': timestamps[0],  # Bar start time
                't_end': timestamps[-1],  # Bar end time
                'open': prices[0],
                'high': max(prices),
                'low': min(prices),
                'close': prices[-1],
                'volume': cum_volume,
                'vwap': cum_dollar_volume / cum_volume if cum_volume > 0 else prices[-1],
                'trade_count': len(bar_trades),
                'ticker': trade['ticker']
            }
            
            bars.append(bar)
            
            # Reset accumulators for next bar
            cum_volume = 0
            cum_dollar_volume = 0
            bar_trades = []
    
    # Handle remaining trades (partial bar at end)
    if bar_trades:
        prices = [t['price'] for t in bar_trades]
        volumes = [t['size'] for t in bar_trades]
        timestamps = [t['ts'] for t in bar_trades]
        
        bar = {
            't': timestamps[0],
            't_end': timestamps[-1],
            'open': prices[0],
            'high': max(prices),
            'low': min(prices),
            'close': prices[-1],
            'volume': sum(volumes),
            'vwap': sum(p * v for p, v in zip(prices, volumes)) / sum(volumes) if sum(volumes) > 0 else prices[-1],
            'trade_count': len(bar_trades),
            'ticker': bar_trades[0]['ticker']
        }
        
        bars.append(bar)
    
    df_bars = pd.DataFrame(bars)
    
    if len(df_bars) > 0:
        # Ensure proper dtypes
        float_cols = ['open', 'high', 'low', 'close', 'vwap']
        for col in float_cols:
            df_bars[col] = df_bars[col].astype('float32')
        
        df_bars['volume'] = df_bars['volume'].astype('int32')
        df_bars['trade_count'] = df_bars['trade_count'].astype('int32')
        
        logger.debug(f"Built {len(df_bars)} volume bars from {len(df_trades)} trades")
    
    return df_bars


def build_volume_bars_by_day(
    df_trades: pd.DataFrame,
    volume_threshold: int,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Build volume bars, ensuring bars don't span multiple days.
    
    Args:
        df_trades: DataFrame with columns ['ts', 'price', 'size', 'ticker']
        volume_threshold: Volume required to trigger a new bar
        logger: Optional logger instance
        
    Returns:
        DataFrame with volume bars
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    if len(df_trades) == 0:
        return pd.DataFrame()
    
    # Extract date for grouping
    df_trades = df_trades.copy()
    df_trades['date'] = pd.to_datetime(df_trades['ts']).dt.date
    
    all_bars = []
    
    # Process each day separately
    for date, day_trades in df_trades.groupby('date'):
        day_bars = build_volume_bars(
            day_trades.drop(columns=['date']),
            volume_threshold,
            logger
        )
        
        if len(day_bars) > 0:
            day_bars['date'] = date
            all_bars.append(day_bars)
    
    if all_bars:
        df_all_bars = pd.concat(all_bars, ignore_index=True)
        logger.info(f"Built {len(df_all_bars)} bars across {len(all_bars)} days")
        return df_all_bars
    else:
        return pd.DataFrame()


def validate_bars(df_bars: pd.DataFrame, logger: logging.Logger = None) -> Tuple[bool, dict]:
    """
    Validate volume bars for quality and consistency.
    
    Args:
        df_bars: DataFrame of volume bars
        logger: Optional logger instance
        
    Returns:
        (is_valid, stats_dict)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    stats = {
        'total_bars': len(df_bars),
        'invalid_ohlc': 0,
        'negative_volume': 0,
        'zero_trades': 0,
        'time_gaps': 0
    }
    
    if len(df_bars) == 0:
        return True, stats
    
    # Check OHLC consistency: low <= open, close, high and high >= open, close, low
    invalid_ohlc = (
        (df_bars['low'] > df_bars['open']) |
        (df_bars['low'] > df_bars['close']) |
        (df_bars['low'] > df_bars['high']) |
        (df_bars['high'] < df_bars['open']) |
        (df_bars['high'] < df_bars['close']) |
        (df_bars['high'] < df_bars['low'])
    )
    stats['invalid_ohlc'] = invalid_ohlc.sum()
    
    # Check for negative volume
    stats['negative_volume'] = (df_bars['volume'] <= 0).sum()
    
    # Check for zero trade count
    stats['zero_trades'] = (df_bars['trade_count'] <= 0).sum()
    
    # Check time ordering
    if 't' in df_bars.columns:
        time_diffs = df_bars['t'].diff()
        stats['time_gaps'] = (time_diffs < pd.Timedelta(0)).sum()
    
    is_valid = all(v == 0 for k, v in stats.items() if k != 'total_bars')
    
    if not is_valid:
        logger.warning(f"Bar validation failed: {stats}")
    else:
        logger.debug(f"Bar validation passed: {stats['total_bars']} bars")
    
    return is_valid, stats


def compute_bar_statistics(df_bars: pd.DataFrame) -> dict:
    """
    Compute summary statistics for volume bars.
    
    Args:
        df_bars: DataFrame of volume bars
        
    Returns:
        Dictionary of statistics
    """
    if len(df_bars) == 0:
        return {}
    
    # Compute bar duration
    if 't' in df_bars.columns and 't_end' in df_bars.columns:
        df_bars['duration_seconds'] = (
            pd.to_datetime(df_bars['t_end']) - pd.to_datetime(df_bars['t'])
        ).dt.total_seconds()
    
    stats = {
        'total_bars': len(df_bars),
        'avg_volume': df_bars['volume'].mean(),
        'median_volume': df_bars['volume'].median(),
        'avg_trade_count': df_bars['trade_count'].mean(),
        'avg_bar_duration_seconds': df_bars['duration_seconds'].mean() if 'duration_seconds' in df_bars.columns else None,
        'median_bar_duration_seconds': df_bars['duration_seconds'].median() if 'duration_seconds' in df_bars.columns else None,
    }
    
    return stats


def main():
    """Test volume bar construction on a sample file."""
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    print("=" * 80)
    print("Volume Bar Construction Test")
    print("=" * 80)
    
    # Load config
    from config_processing import load_processing_config
    
    config = load_processing_config()
    print(f"\nLoaded config (hash: {config['config_hash']})")
    
    # Test on one file
    test_file = "data/clean/CCL/CCL_10-2-25_clean.parquet"
    ticker = "CCL"
    
    print(f"\nTesting on: {test_file}")
    print(f"Volume threshold for {ticker}: {config['volume_bars']['thresholds'][ticker]:,} shares/bar")
    
    # Load trades
    df_trades = pd.read_parquet(test_file)
    print(f"Loaded {len(df_trades):,} trades")
    
    # Build volume bars
    volume_threshold = config['volume_bars']['thresholds'][ticker]
    df_bars = build_volume_bars(df_trades, volume_threshold, logger)
    
    print(f"\nBuilt {len(df_bars)} volume bars")
    
    # Validate
    is_valid, validation_stats = validate_bars(df_bars, logger)
    print(f"\nValidation: {'PASSED' if is_valid else 'FAILED'}")
    for key, value in validation_stats.items():
        print(f"  {key}: {value}")
    
    # Statistics
    stats = compute_bar_statistics(df_bars)
    print(f"\nBar Statistics:")
    for key, value in stats.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.2f}")
            else:
                print(f"  {key}: {value}")
    
    # Display first few bars
    print(f"\nFirst 5 bars:")
    print(df_bars.head())
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()

