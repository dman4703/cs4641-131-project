"""
Data Processing Pipeline
volume bars → features → labels → CV → save
"""

import json
import logging
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml

from config_processing import load_processing_config
from volume_bars import build_volume_bars_by_day, validate_bars, compute_bar_statistics
from feature_engineering import engineer_features, validate_features
from labeling import triple_barrier_labels, validate_labels
from cross_validation import create_cv_splits, validate_cv_splits


def process_single_file(
    ticker: str,
    date_str: str,
    config: Dict,
    input_dir: Path,
    output_dir: Path
) -> Dict:
    """
    Process a single ticker-day file through the complete pipeline.
    
    Args:
        ticker: Ticker symbol
        date_str: Date string (e.g., '10-2-25')
        config: Processing configuration
        input_dir: Input directory with cleaned data
        output_dir: Output directory for processed data
        
    Returns:
        Dictionary with processing metadata
    """
    # Setup logger for this process
    logger = logging.getLogger(f"{__name__}.{ticker}_{date_str}")
    
    start_time = time.time()
    metadata = {
        'ticker': ticker,
        'date': date_str,
        'status': 'pending'
    }
    
    try:
        # 1. Load cleaned tick data
        input_file = input_dir / ticker / f"{ticker}_{date_str}_clean.parquet"
        logger.info(f"Processing {ticker} {date_str}...")
        
        if not input_file.exists():
            raise FileNotFoundError(f"Input file not found: {input_file}")
        
        df_trades = pd.read_parquet(input_file)
        metadata['n_trades'] = len(df_trades)
        
        # 2. Build volume bars
        logger.debug(f"  Building volume bars...")
        volume_threshold = config['volume_bars']['thresholds'][ticker]
        
        df_bars = build_volume_bars_by_day(df_trades, volume_threshold, logger)
        
        if len(df_bars) == 0:
            raise ValueError("No volume bars generated")
        
        metadata['n_bars'] = len(df_bars)
        metadata['volume_threshold'] = volume_threshold
        
        is_valid, bar_stats = validate_bars(df_bars, logger)
        if not is_valid:
            logger.warning(f"  Bar validation failed: {bar_stats}")
        
        # Bar statistics
        bar_summary = compute_bar_statistics(df_bars)
        metadata['avg_bar_volume'] = float(bar_summary['avg_volume'])
        metadata['avg_bar_duration_seconds'] = float(bar_summary['avg_bar_duration_seconds'] or 0)
        
        # 3. Engineer features
        logger.debug(f"  Engineering features...")
        df_features = engineer_features(df_bars, config['features'], logger)
        
        # Validate features
        feature_stats = validate_features(df_features, logger)
        metadata['n_features'] = feature_stats['total_features']
        metadata['features_with_nan'] = len(feature_stats['features_with_nan'])
        
        # 4. Compute triple barrier labels
        logger.debug(f"  Computing labels...")
        df_labeled = triple_barrier_labels(
            df_features,
            k=config['triple_barrier']['k'],
            time_barrier_minutes=config['triple_barrier']['time_barrier_minutes'],
            ewm_halflife=config['triple_barrier']['ewm_halflife'],
            logger=logger
        )
        
        # Validate labels
        label_stats = validate_labels(df_labeled, logger)
        metadata['n_labeled'] = label_stats['labeled_bars']
        metadata['n_unlabeled'] = label_stats['unlabeled_bars']
        metadata['label_dist'] = json.dumps(label_stats['label_distribution'])
        
        # 5. Save processed data
        logger.debug(f"  Saving processed data...")
        ticker_output_dir = output_dir / ticker
        ticker_output_dir.mkdir(parents=True, exist_ok=True)
        
        output_file = ticker_output_dir / f"{ticker}_{date_str}_processed.parquet"
        
        df_labeled.to_parquet(
            output_file,
            engine='pyarrow',
            compression=config['output']['compression'],
            index=False
        )
        
        metadata['output_path'] = str(output_file)
        metadata['file_size_mb'] = output_file.stat().st_size / (1024 * 1024)
        
        processing_time = time.time() - start_time
        metadata['processing_time_seconds'] = processing_time
        metadata['status'] = 'success'
        
        logger.info(f"[OK] {ticker} {date_str}: {len(df_labeled)} bars ({processing_time:.2f}s)")
        
        return metadata
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[FAIL] {ticker} {date_str}: {str(e)}", exc_info=True)
        
        metadata['status'] = 'failed'
        metadata['error'] = str(e)
        metadata['processing_time_seconds'] = processing_time
        
        return metadata


def run_pipeline(
    config: Dict,
    input_dir: Path = Path("data/clean"),
    output_dir: Path = Path("data/processed"),
    max_workers: int = None,
    logger: logging.Logger = None
) -> pd.DataFrame:
    """
    Run the complete processing pipeline for all ticker-day files.
    
    Args:
        config: Processing configuration
        input_dir: Input directory with cleaned data
        output_dir: Output directory for processed data
        max_workers: Number of parallel workers (None = auto)
        logger: Logger instance
        
    Returns:
        DataFrame with processing metadata for all files
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Data Processing Pipeline")
    logger.info("=" * 80)
    
    # Discover all cleaned parquet files
    ticker_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    
    files_to_process = []
    for ticker_dir in ticker_dirs:
        ticker = ticker_dir.name
        
        # Skip metadata files
        if ticker == 'metadata.parquet' or ticker == 'metadata.csv':
            continue
        
        parquet_files = list(ticker_dir.glob("*_clean.parquet"))
        
        for file in parquet_files:
            # Extract date from filename
            date_str = file.stem.replace(f"{ticker}_", "").replace("_clean", "")
            files_to_process.append((ticker, date_str))
    
    logger.info(f"Found {len(files_to_process)} files to process")
    
    if len(files_to_process) == 0:
        logger.error("No files found to process!")
        return pd.DataFrame()
    
    # Determine number of workers
    if max_workers is None:
        import os
        max_workers = max(1, (os.cpu_count() or 2) - 1)
    
    logger.info(f"Processing with {max_workers} parallel workers...")
    
    # Process files in parallel
    metadata_list = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_file = {
            executor.submit(
                process_single_file,
                ticker,
                date_str,
                config,
                input_dir,
                output_dir
            ): (ticker, date_str)
            for ticker, date_str in files_to_process
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_file):
            ticker, date_str = future_to_file[future]
            try:
                metadata = future.result()
                metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Unexpected error processing {ticker} {date_str}: {e}")
                metadata_list.append({
                    'ticker': ticker,
                    'date': date_str,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Create metadata DataFrame
    metadata_df = pd.DataFrame(metadata_list)
    
    return metadata_df


def create_cv_metadata(
    processed_dir: Path,
    config: Dict,
    logger: logging.Logger = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load all processed files and create CV splits metadata.
    
    Args:
        processed_dir: Directory with processed parquet files
        config: Processing configuration
        logger: Logger instance
        
    Returns:
        (cv_metadata_df, aggregated_data_df)
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\nCreating CV split metadata...")
    
    # Load all processed files
    all_data = []
    
    for ticker_dir in processed_dir.iterdir():
        if not ticker_dir.is_dir():
            continue
        
        for file in ticker_dir.glob("*_processed.parquet"):
            df = pd.read_parquet(file)
            all_data.append(df)
    
    if not all_data:
        logger.error("No processed files found!")
        return pd.DataFrame(), pd.DataFrame()
    
    # Combine all data
    df_all = pd.concat(all_data, ignore_index=True)
    logger.info(f"Loaded {len(df_all)} total bars from {len(all_data)} files")
    
    # Create CV splits
    cv_splits, cv_metadata = create_cv_splits(
        df_all,
        embargo_minutes=config['cv']['embargo_minutes'],
        logger=logger
    )
    
    # Validate splits
    validation_stats = validate_cv_splits(df_all, cv_splits, logger)
    
    # Add config hash to metadata
    cv_metadata['config_hash'] = config['config_hash']
    cv_metadata['n_folds'] = config['cv']['n_folds']
    cv_metadata['embargo_minutes'] = config['cv']['embargo_minutes']
    
    return cv_metadata, df_all


def save_metadata(
    processing_metadata: pd.DataFrame,
    cv_metadata: pd.DataFrame,
    config: Dict,
    output_dir: Path,
    logger: logging.Logger = None
) -> None:
    """
    Save all metadata to parquet and CSV files.
    
    Args:
        processing_metadata: File processing metadata
        cv_metadata: CV split metadata
        config: Processing configuration
        output_dir: Output directory
        logger: Logger instance
    """
    if logger is None:
        logger = logging.getLogger(__name__)
    
    logger.info("\nSaving metadata...")
    
    # Save processing metadata
    proc_metadata_file = output_dir / "processing_metadata.parquet"
    processing_metadata.to_parquet(proc_metadata_file, index=False)
    processing_metadata.to_csv(output_dir / "processing_metadata.csv", index=False)
    logger.info(f"  Saved processing metadata: {proc_metadata_file}")
    
    # Save CV metadata
    cv_metadata_file = output_dir / "cv_metadata.parquet"
    cv_metadata.to_parquet(cv_metadata_file, index=False)
    cv_metadata.to_csv(output_dir / "cv_metadata.csv", index=False)
    logger.info(f"  Saved CV metadata: {cv_metadata_file}")
    
    # Save config
    config_file = output_dir / "pipeline_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    logger.info(f"  Saved config: {config_file}")


def print_summary(
    processing_metadata: pd.DataFrame,
    cv_metadata: pd.DataFrame,
    config: Dict,
    logger: logging.Logger = None
) -> None:
    """Print pipeline summary statistics."""
    if logger is None:
        logger = logging.getLogger(__name__)
    
    print("\n" + "=" * 80)
    print("PIPELINE SUMMARY")
    print("=" * 80)
    
    # Processing stats
    success_count = (processing_metadata['status'] == 'success').sum()
    failed_count = (processing_metadata['status'] == 'failed').sum()
    
    print(f"\nProcessing Results:")
    print(f"  Total files: {len(processing_metadata)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    
    if success_count > 0:
        success_df = processing_metadata[processing_metadata['status'] == 'success']
        
        print(f"\n  Total volume bars: {success_df['n_bars'].sum():,}")
        print(f"  Average bars per file: {success_df['n_bars'].mean():.0f}")
        print(f"  Average processing time: {success_df['processing_time_seconds'].mean():.2f}s")
        
        # Label distribution
        print(f"\n  Label distribution (across all files):")
        total_labeled = success_df['n_labeled'].sum()
        total_unlabeled = success_df['n_unlabeled'].sum()
        print(f"    Labeled: {total_labeled:,} ({total_labeled/(total_labeled+total_unlabeled)*100:.1f}%)")
        print(f"    Unlabeled: {total_unlabeled:,} ({total_unlabeled/(total_labeled+total_unlabeled)*100:.1f}%)")
    
    # CV stats
    if len(cv_metadata) > 0:
        print(f"\nCross-Validation:")
        print(f"  Number of folds: {len(cv_metadata)}")
        print(f"  Average train size: {cv_metadata['n_train'].mean():.0f}")
        print(f"  Average validation size: {cv_metadata['n_val'].mean():.0f}")
        print(f"  Average purged: {cv_metadata['n_purged'].mean():.0f}")
        print(f"  Average embargoed: {cv_metadata['n_embargoed'].mean():.0f}")
    
    print(f"\nConfiguration:")
    print(f"  Config hash: {config['config_hash']}")
    print(f"  Volume bars target: {config['volume_bars']['target_bars_per_day']} bars/day")
    print(f"  Triple barrier k: {config['triple_barrier']['k']}")
    print(f"  Time barrier: {config['triple_barrier']['time_barrier_minutes']} minutes")
    print(f"  Embargo: {config['cv']['embargo_minutes']} minutes")
    
    print("\n" + "=" * 80)


def main():
    """Main entry point for the processing pipeline."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler('data/processed/pipeline.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    
    # Load configuration
    config = load_processing_config()
    
    # Run pipeline
    processing_metadata = run_pipeline(
        config,
        input_dir=Path("data/clean"),
        output_dir=Path("data/processed"),
        max_workers=None,  # Auto-detect
        logger=logger
    )
    
    # Create CV metadata
    cv_metadata, df_all = create_cv_metadata(
        Path("data/processed"),
        config,
        logger
    )
    
    # Save metadata
    save_metadata(
        processing_metadata,
        cv_metadata,
        config,
        Path("data/processed"),
        logger
    )
    
    # Print summary
    print_summary(processing_metadata, cv_metadata, config, logger)
    
    logger.info("\nPipeline complete!")


if __name__ == "__main__":
    main()

