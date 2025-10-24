"""
Configuration Management for Data Processing Pipeline
Computes ticker-specific volume thresholds and saves configuration parameters.
"""

import hashlib
import logging
from pathlib import Path
from typing import Dict

import pandas as pd
import yaml


def compute_volume_thresholds(metadata_path: str = "data/clean/metadata.parquet") -> Dict[str, int]:
    """
    Compute ticker-specific volume thresholds for volume bars.
    
    Target: ~600 bars/day
    Formula: median_daily_volume / 600, rounded to nearest 100, clamped to [2000, 100000]
    
    Args:
        metadata_path: Path to cleaned data metadata
        
    Returns:
        Dictionary mapping ticker -> volume_threshold
    """
    logger = logging.getLogger(__name__)
    
    # Load metadata
    metadata_df = pd.read_parquet(metadata_path)
    
    # Filter successful runs only
    metadata_df = metadata_df[metadata_df['status'] == 'success'].copy()
    
    # Compute daily volume per ticker-day from final_rows (trades)
    # Note: final_rows is the number of trades, not volume
    # We need to load the actual parquet files to get volume
    
    logger.info("Computing volume thresholds from cleaned data...")
    
    volume_by_ticker = {}
    
    for ticker in metadata_df['ticker'].unique():
        ticker_files = metadata_df[metadata_df['ticker'] == ticker]
        daily_volumes = []
        
        for _, row in ticker_files.iterrows():
            # Load the cleaned parquet file
            file_path = Path(row['output_path'])
            if file_path.exists():
                df = pd.read_parquet(file_path)
                # Sum all trade sizes for the day
                total_volume = df['size'].sum()
                daily_volumes.append(total_volume)
        
        if daily_volumes:
            # Compute median daily volume
            median_daily_vol = pd.Series(daily_volumes).median()
            
            # Target 600 bars/day
            threshold = median_daily_vol / 600
            
            # Round to nearest 100
            threshold = round(threshold / 100) * 100
            
            # Clamp to [2000, 100000]
            threshold = max(2000, min(100000, threshold))
            
            volume_by_ticker[ticker] = int(threshold)
            
            logger.info(f"{ticker}: median_daily_vol={median_daily_vol:,.0f}, threshold={threshold:,}")
    
    return volume_by_ticker


def create_processing_config(
    volume_thresholds: Dict[str, int],
    output_path: str = "config/processing.yaml"
) -> dict:
    """
    Create and save processing configuration with locked parameters.
    
    Args:
        volume_thresholds: Ticker-specific volume thresholds
        output_path: Where to save the config
        
    Returns:
        Configuration dictionary
    """
    logger = logging.getLogger(__name__)
    
    config = {
        'volume_bars': {
            'target_bars_per_day': 600,
            'thresholds': volume_thresholds,
            'min_threshold': 2000,
            'max_threshold': 100000
        },
        'triple_barrier': {
            'k': 1.0,  # Volatility multiplier for barriers
            'k_grid': [0.75, 1.0, 1.25],  # For potential grid search
            'ewm_halflife': 50,  # bars
            'time_barrier_minutes': 20
        },
        'features': {
            'vwap_zscore_window': 20,
            'bollinger_window': 20,
            'bollinger_std': 2.0,
            'momentum_windows': [3, 5],
            'relative_volume_window': 20,
            'context_window_minutes': 5
        },
        'session': {
            'market_open': '09:35',
            'market_close': '15:35',
            'timezone': 'America/New_York'
        },
        'cv': {
            'n_folds': 5,
            'strategy': 'LODO',  # Leave-One-Day-Out
            'embargo_minutes': 20,
            'purge_method': 'interval_overlap'  # Purge if [t, t_end] overlaps
        },
        'output': {
            'compression': 'zstd',
            'float_dtype': 'float32',
            'int_dtype': 'int32'
        }
    }
    
    # Compute config hash for reproducibility
    config_str = yaml.dump(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()
    config['config_hash'] = config_hash
    
    # Save config
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    logger.info(f"Saved processing config to {output_path}")
    logger.info(f"Config hash: {config_hash}")
    
    return config


def load_processing_config(config_path: str = "config/processing.yaml") -> dict:
    """
    Load processing configuration.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def main():
    """Generate and save processing configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s'
    )
    
    print("=" * 80)
    print("Processing Configuration Generation")
    print("=" * 80)
    
    # Compute volume thresholds
    print("\nComputing volume thresholds...")
    volume_thresholds = compute_volume_thresholds()
    
    print(f"\nVolume thresholds computed for {len(volume_thresholds)} tickers:")
    for ticker, threshold in sorted(volume_thresholds.items()):
        print(f"  {ticker:6s}: {threshold:6,} shares/bar")
    
    # Create and save config
    print("\nGenerating configuration...")
    config = create_processing_config(volume_thresholds)
    
    print("\n" + "=" * 80)
    print(f"Configuration saved to config/processing.yaml")
    print(f"Config hash: {config['config_hash']}")
    print("=" * 80)


if __name__ == "__main__":
    main()

