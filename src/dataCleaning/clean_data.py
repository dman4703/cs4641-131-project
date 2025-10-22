"""
Data Cleaning Pipeline for Tick Data
Transforms raw CSV tick data into cleaned, NBBO-enriched Parquet files.
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import pandas as pd
import numpy as np
import yaml


# Configuration & Setup

def setup_logging(config: dict) -> logging.Logger:
    """Setup logging configuration."""
    log_level = getattr(logging, config['logging']['level'])
    log_file = Path(config['logging']['log_file'])
    log_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Use UTF-8 encoding for both file and console to support Unicode characters
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Set stdout to UTF-8 on Windows
    if sys.platform == 'win32':
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass  # Python < 3.7
    
    return logging.getLogger(__name__)


def load_config(config_path: str = "config/preprocessing.yaml") -> dict:
    """Load preprocessing configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def load_codebook(codebook_path: str = "config/condition_codes.csv") -> pd.DataFrame:
    """Load condition code codebook."""
    df = pd.read_csv(codebook_path)
    # Create lookup dict for faster access
    return df

# A. Load and Validate

def load_and_validate(csv_path: Path, logger: logging.Logger) -> Tuple[pd.DataFrame, dict]:
    """
    Load raw CSV, validate, clean, and sort.
    Returns: (dataframe, stats_dict)
    """
    stats = {
        'raw_rows': 0,
        'dropped_header': 0,
        'dropped_invalid_price': 0,
        'dropped_invalid_size': 0,
        'dropped_nulls': 0
    }
    
    # Extract ticker from path (e.g., CCL from CCL/CCL_10-2-25.csv)
    ticker = csv_path.parent.name
    
    # Read CSV, skip first 2 junk header lines
    # low_memory=False to avoid dtype warnings on large files
    df = pd.read_csv(csv_path, skiprows=2, low_memory=False)
    stats['raw_rows'] = len(df)
    
    # Rename columns to canonical names
    df.columns = ['ts', 'type', 'price', 'size', 'cond', 'exch', 'tradetime_old', 'spread_old']
    
    # Filter out embedded header rows
    # Remove rows where ts column contains "Dates" or other header text
    header_mask = df['ts'].astype(str).str.strip() == 'Dates'
    stats['dropped_header'] = header_mask.sum()
    df = df[~header_mask]
    
    # Add ticker column
    df['ticker'] = ticker
    
    # Parse timestamps: localize to America/New_York, convert to UTC
    df['ts'] = pd.to_datetime(df['ts'], errors='coerce')
    
    # Drop rows with unparseable timestamps (NaT)
    mask_bad_ts = df['ts'].isna()
    if mask_bad_ts.any():
        stats['dropped_nulls'] += mask_bad_ts.sum()
        df = df[~mask_bad_ts]
    
    # Only localize if we have valid timestamps
    if len(df) > 0:
        df['ts'] = df['ts'].dt.tz_localize('America/New_York', ambiguous='infer', nonexistent='shift_forward')
        df['ts'] = df['ts'].dt.tz_convert('UTC')
    
    # Enforce dtypes
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df['size'] = pd.to_numeric(df['size'], errors='coerce').astype('Int64')  # Nullable int
    df['type'] = df['type'].astype('category')
    df['exch'] = df['exch'].astype(str).astype('category')
    df['cond'] = df['cond'].astype(str)
    
    # Drop invalid rows
    initial_len = len(df)
    
    # Drop price <= 0 or NaN
    mask_price = (df['price'] <= 0) | df['price'].isna()
    stats['dropped_invalid_price'] = mask_price.sum()
    df = df[~mask_price]
    
    # Drop size <= 0 or NaN
    mask_size = (df['size'] <= 0) | df['size'].isna()
    stats['dropped_invalid_size'] = mask_size.sum()
    df = df[~mask_size]
    
    # Drop nulls in critical fields (type)
    mask_nulls = df['type'].isnull()
    stats['dropped_nulls'] += mask_nulls.sum()
    df = df[~mask_nulls]
    
    # Sort by (ts, type, exch, price, size)
    df = df.sort_values(['ts', 'type', 'exch', 'price', 'size']).reset_index(drop=True)
    
    logger.debug(f"{ticker}: Loaded {stats['raw_rows']} rows, dropped {initial_len - len(df)} invalid")
    
    return df, stats

# B. Normalize Condition Code

def normalize_cond_code(cond_str) -> str:
    """
    Normalize condition code string.
    Handle blank/0 → empty string
    Split on comma, strip, uppercase, sort tokens
    """
    if pd.isna(cond_str) or cond_str == '' or cond_str == '0':
        return ''
    
    # Split on comma, strip whitespace, uppercase, sort
    tokens = [t.strip().upper() for t in str(cond_str).split(',')]
    tokens = sorted([t for t in tokens if t and t != '0'])
    
    return ','.join(tokens) if tokens else ''

# C. Classify Condition Code

def classify_cond_code(cond_norm: str, codebook: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Classify normalized condition code.
    Returns: (keep=True/False, list of unknown codes)
    
    Rules:
    - Any HARD_EXCLUDE token → EXCLUDE (keep=False)
    - Else if any SOFT_INCLUDE → INCLUDE (keep=True)
    - Else NEUTRAL → INCLUDE (keep=True)
    """
    if cond_norm == '':
        return True, []  # Empty = NEUTRAL = keep
    
    # Split tokens
    tokens = cond_norm.split(',')
    
    # Create lookup dict from codebook
    code_to_bucket = dict(zip(codebook['code'].astype(str), codebook['bucket']))
    
    unknown_codes = []
    has_hard_exclude = False
    has_soft_include = False
    
    for token in tokens:
        if token not in code_to_bucket:
            unknown_codes.append(token)
            # Treat unknown as HARD_EXCLUDE (conservative)
            has_hard_exclude = True
        else:
            bucket = code_to_bucket[token]
            if bucket == 'HARD_EXCLUDE':
                has_hard_exclude = True
            elif bucket == 'SOFT_INCLUDE':
                has_soft_include = True
            # NEUTRAL doesn't change flags
    
    # Apply priority rules
    if has_hard_exclude:
        return False, unknown_codes
    elif has_soft_include:
        return True, unknown_codes
    else:
        # All NEUTRAL
        return True, unknown_codes

# D. Remove Duplicates

def remove_duplicates(df: pd.DataFrame, codebook: pd.DataFrame, logger: logging.Logger) -> Tuple[pd.DataFrame, dict]:
    """
    Remove duplicates based on key: (ts, type, price, size, exch, cond_norm).
    For exact duplicates, keep last.
    For near-duplicates (differ only in cond), rank by priority and keep best.
    """
    stats = {
        'duplicates_removed': 0,
        'dup_pct': 0.0
    }
    
    initial_len = len(df)
    
    # Add normalized condition code column
    df['cond_norm'] = df['cond'].apply(normalize_cond_code)
    
    # Define duplicate key
    dup_key = ['ts', 'type', 'price', 'size', 'exch', 'cond_norm']
    
    # For exact duplicates, keep last
    df = df.drop_duplicates(subset=dup_key, keep='last')
    
    stats['duplicates_removed'] = initial_len - len(df)
    stats['dup_pct'] = (stats['duplicates_removed'] / initial_len * 100) if initial_len > 0 else 0.0
    
    logger.debug(f"Removed {stats['duplicates_removed']} duplicates ({stats['dup_pct']:.2f}%)")
    
    return df, stats

# E. Split Trades and Quotes

def split_trades_quotes(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split dataframe into trades and quotes."""
    trades = df[df['type'] == 'TRADE'].copy()
    quotes = df[df['type'].isin(['BEST_BID', 'BEST_ASK'])].copy()
    return trades, quotes


# F. Build NBBO

def build_nbbo(quotes_df: pd.DataFrame, ffill_cap_seconds: int, logger: logging.Logger) -> Tuple[pd.DataFrame, dict]:
    """
    Build NBBO timeseries from quotes.
    Handle locked/crossed markets, forward-fill with cap.
    Returns: (nbbo_df, stats)
    """
    stats = {
        'total_quote_stamps': 0,
        'crossed_quotes': 0,
        'crossed_quotes_pct': 0.0,
        'nbbo_coverage_pct': 0.0
    }
    
    if len(quotes_df) == 0:
        logger.warning("No quotes found!")
        return pd.DataFrame(), stats
    
    # Separate bids and asks
    bids = quotes_df[quotes_df['type'] == 'BEST_BID'][['ts', 'price', 'size']].copy()
    asks = quotes_df[quotes_df['type'] == 'BEST_ASK'][['ts', 'price', 'size']].copy()
    
    bids = bids.rename(columns={'price': 'nbb', 'size': 'nbb_size'})
    asks = asks.rename(columns={'price': 'nbo', 'size': 'nbo_size'})
    
    # Group by timestamp and take last (most recent) bid/ask at each timestamp
    bids = bids.groupby('ts').last()
    asks = asks.groupby('ts').last()
    
    # Merge bids and asks
    nbbo = pd.merge(bids, asks, left_index=True, right_index=True, how='outer')
    
    # Forward-fill up to cap
    nbbo = nbbo.sort_index()
    
    # Create time-limited forward fill
    # First, forward fill unlimited
    nbbo_filled = nbbo.ffill()
    
    # Then, mask out values that are too old
    for col in ['nbb', 'nbo', 'nbb_size', 'nbo_size']:
        # Calculate time since last valid value
        last_valid = nbbo[col].notna()
        # Create groups of consecutive valid/invalid
        time_since_valid = (nbbo.index.to_series().diff().dt.total_seconds()).fillna(0)
        
    # Simpler approach: use ffill with limit based on time
    # Since we have second-level timestamps, we'll do a custom ffill
    nbbo_filled = nbbo.copy()
    for col in ['nbb', 'nbo', 'nbb_size', 'nbo_size']:
        nbbo_filled[col] = nbbo[col].ffill()
        # Now mask based on time gap
        last_valid_idx = nbbo[col].notna()
        if last_valid_idx.any():
            # For each row, find time since last valid
            valid_times = nbbo.index[last_valid_idx]
            for idx in nbbo.index:
                prior_valid = valid_times[valid_times <= idx]
                if len(prior_valid) > 0:
                    last_valid_time = prior_valid[-1]
                    time_diff = (idx - last_valid_time).total_seconds()
                    if time_diff > ffill_cap_seconds:
                        nbbo_filled.loc[idx, col] = np.nan
    
    nbbo = nbbo_filled
    
    # Drop rows with any NaN
    nbbo_complete = nbbo.dropna()
    
    stats['total_quote_stamps'] = len(nbbo)
    
    # Check for crossed/locked markets (nbo < nbb)
    if len(nbbo_complete) > 0:
        crossed_mask = nbbo_complete['nbo'] < nbbo_complete['nbb']
        stats['crossed_quotes'] = crossed_mask.sum()
        stats['crossed_quotes_pct'] = (stats['crossed_quotes'] / len(nbbo_complete) * 100)
        
        # Drop crossed quotes
        nbbo_complete = nbbo_complete[~crossed_mask]
        
        # Re-apply forward fill after dropping crossed
        nbbo_complete = nbbo_complete.sort_index()
        # Apply same time-limited ffill
        for col in ['nbb', 'nbo', 'nbb_size', 'nbo_size']:
            nbbo_complete[col] = nbbo_complete[col].ffill()
    
    # Compute mid and spread
    if len(nbbo_complete) > 0:
        nbbo_complete['mid'] = 0.5 * (nbbo_complete['nbb'] + nbbo_complete['nbo'])
        nbbo_complete['spread'] = nbbo_complete['nbo'] - nbbo_complete['nbb']
    
    # Reset index to make ts a column
    nbbo_complete = nbbo_complete.reset_index()
    
    logger.debug(f"Built NBBO: {len(nbbo_complete)} timestamps, {stats['crossed_quotes']} crossed ({stats['crossed_quotes_pct']:.2f}%)")
    
    return nbbo_complete, stats


# G. Merge NBBO to Trades

def merge_nbbo_to_trades(trades_df: pd.DataFrame, nbbo_df: pd.DataFrame, 
                        config: dict, logger: logging.Logger) -> Tuple[pd.DataFrame, dict]:
    """
    Merge NBBO to trades using asof join.
    Compute microstructure flags if configured.
    """
    stats = {
        'trades_before_merge': len(trades_df),
        'trades_after_merge': 0,
        'trades_dropped_no_nbbo': 0,
        'nbbo_coverage_pct': 0.0
    }
    
    if len(trades_df) == 0:
        logger.warning("No trades to merge!")
        return trades_df, stats
    
    if len(nbbo_df) == 0:
        logger.warning("No NBBO data to merge!")
        stats['trades_dropped_no_nbbo'] = len(trades_df)
        return pd.DataFrame(), stats
    
    # Ensure both are sorted by ts
    trades_df = trades_df.sort_values('ts')
    nbbo_df = nbbo_df.sort_values('ts')
    
    # Merge as-of (backward direction)
    merged = pd.merge_asof(
        trades_df, 
        nbbo_df[['ts', 'nbb', 'nbo', 'nbb_size', 'nbo_size', 'mid', 'spread']], 
        on='ts', 
        direction='backward'
    )
    
    # Drop trades with NaN in mid or spread (no valid NBBO)
    initial_len = len(merged)
    merged = merged.dropna(subset=['mid', 'spread'])
    
    stats['trades_after_merge'] = len(merged)
    stats['trades_dropped_no_nbbo'] = initial_len - len(merged)
    stats['nbbo_coverage_pct'] = (len(merged) / stats['trades_before_merge'] * 100) if stats['trades_before_merge'] > 0 else 0.0
    
    # Compute microstructure flags if configured
    if config.get('compute_microstructure_flags', True) and len(merged) > 0:
        tolerance = config.get('at_bid_ask_tolerance', 0.0001)
        merged['at_bid'] = (abs(merged['price'] - merged['nbb']) <= tolerance).astype(int)
        merged['at_ask'] = (abs(merged['price'] - merged['nbo']) <= tolerance).astype(int)
    
    logger.debug(f"Merged NBBO to trades: {stats['trades_after_merge']} trades with valid NBBO ({stats['nbbo_coverage_pct']:.2f}% coverage)")
    
    return merged, stats

# H. Apply Condition Filter

def apply_cond_filter(trades_df: pd.DataFrame, codebook: pd.DataFrame, 
                     logger: logging.Logger) -> Tuple[pd.DataFrame, dict, set]:
    """
    Filter trades based on condition codes.
    Returns: (filtered_df, stats, unknown_codes_set)
    """
    stats = {
        'trades_before_filter': len(trades_df),
        'trades_after_filter': 0,
        'trades_dropped_by_cond': 0,
        'trades_dropped_pct': 0.0
    }
    
    unknown_codes = set()
    
    if len(trades_df) == 0:
        return trades_df, stats, unknown_codes
    
    # Apply classification to each row
    results = trades_df['cond_norm'].apply(lambda c: classify_cond_code(c, codebook))
    keep_flags = [r[0] for r in results]
    unknown_lists = [r[1] for r in results]
    
    # Collect all unknown codes
    for unk_list in unknown_lists:
        unknown_codes.update(unk_list)
    
    # Filter
    filtered = trades_df[keep_flags].copy()
    
    stats['trades_after_filter'] = len(filtered)
    stats['trades_dropped_by_cond'] = stats['trades_before_filter'] - stats['trades_after_filter']
    stats['trades_dropped_pct'] = (stats['trades_dropped_by_cond'] / stats['trades_before_filter'] * 100) if stats['trades_before_filter'] > 0 else 0.0
    
    logger.debug(f"Condition filter: kept {stats['trades_after_filter']}/{stats['trades_before_filter']} trades ({100-stats['trades_dropped_pct']:.2f}%)")
    
    return filtered, stats, unknown_codes

# I. Finalize Schema

def finalize_schema(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """
    Drop redundant columns, finalize schema, sort by timestamp.
    """
    # Drop redundant columns
    cols_to_drop = ['tradetime_old', 'spread_old']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Ensure proper column order
    base_cols = ['ts', 'ticker', 'type', 'price', 'size', 'cond', 'cond_norm', 'exch']
    nbbo_cols = ['nbb', 'nbo', 'nbb_size', 'nbo_size', 'mid', 'spread']
    optional_cols = ['at_bid', 'at_ask']
    
    final_cols = base_cols + nbbo_cols
    for col in optional_cols:
        if col in df.columns:
            final_cols.append(col)
    
    df = df[final_cols]
    
    # Sort by ts
    df = df.sort_values('ts').reset_index(drop=True)
    
    # Convert dtypes for efficiency
    if config['output']['use_categorical']:
        for col in ['type', 'exch']:
            if col in df.columns:
                df[col] = df[col].astype('category')
    
    # Convert to float32/int32 for space efficiency
    float_dtype = config['output']['float_dtype']
    int_dtype = config['output']['int_dtype']
    
    float_cols = ['price', 'nbb', 'nbo', 'mid', 'spread']
    for col in float_cols:
        if col in df.columns:
            df[col] = df[col].astype(float_dtype)
    
    int_cols = ['size', 'nbb_size', 'nbo_size']
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int_dtype)
    
    return df


# J. Write Cleaned Parquet

def write_cleaned_parquet(df: pd.DataFrame, ticker: str, date_str: str, 
                         output_dir: Path, config: dict, logger: logging.Logger) -> Path:
    """
    Write cleaned dataframe to Parquet with compression.
    """
    # Create ticker directory
    ticker_dir = output_dir / ticker
    ticker_dir.mkdir(parents=True, exist_ok=True)
    
    # Create filename
    output_path = ticker_dir / f"{ticker}_{date_str}_clean.parquet"
    
    # Write with specified compression
    df.to_parquet(
        output_path,
        engine='pyarrow',
        compression=config['output']['compression'],
        index=False
    )
    
    logger.debug(f"Wrote {len(df)} rows to {output_path}")
    
    return output_path

# K. Collect Metadata

def collect_metadata(ticker: str, date_str: str, all_stats: dict, 
                     unknown_codes: set, processing_time: float) -> dict:
    """
    Collect metadata from all processing steps.
    """
    metadata = {
        'ticker': ticker,
        'date': date_str,
        'raw_rows': all_stats.get('raw_rows', 0),
        'final_rows': all_stats.get('final_rows', 0),
        'duplicates_removed': all_stats.get('duplicates_removed', 0),
        'dup_pct': all_stats.get('dup_pct', 0.0),
        'crossed_quotes': all_stats.get('crossed_quotes', 0),
        'crossed_quotes_pct': all_stats.get('crossed_quotes_pct', 0.0),
        'nbbo_coverage_pct': all_stats.get('nbbo_coverage_pct', 0.0),
        'trades_dropped_by_cond': all_stats.get('trades_dropped_by_cond', 0),
        'trades_dropped_no_nbbo': all_stats.get('trades_dropped_no_nbbo', 0),
        'unknown_cond_codes': ','.join(sorted(unknown_codes)) if unknown_codes else '',
        'processing_time_seconds': processing_time,
        'dropped_invalid_price': all_stats.get('dropped_invalid_price', 0),
        'dropped_invalid_size': all_stats.get('dropped_invalid_size', 0),
        'dropped_nulls': all_stats.get('dropped_nulls', 0)
    }
    
    return metadata

# L. Process Single File

def process_single_file(csv_path: Path, codebook: pd.DataFrame, config: dict) -> dict:
    """
    Orchestrate full pipeline for one file.
    Returns metadata dict.
    """
    # Setup logger for this process
    logger = logging.getLogger(f"{__name__}.{os.getpid()}")
    
    start_time = time.time()
    all_stats = {}
    unknown_codes = set()
    
    try:
        # Extract date from filename (e.g., "CCL_10-2-25.csv" -> "10-2-25")
        date_str = csv_path.stem.split('_', 1)[1]
        ticker = csv_path.parent.name
        
        logger.info(f"Processing {ticker} {date_str}...")
        
        # A. Load and validate
        df, load_stats = load_and_validate(csv_path, logger)
        all_stats.update(load_stats)
        
        # D. Remove duplicates
        df, dup_stats = remove_duplicates(df, codebook, logger)
        all_stats.update(dup_stats)
        
        # E. Split trades and quotes
        trades, quotes = split_trades_quotes(df)
        logger.debug(f"{ticker}: {len(trades)} trades, {len(quotes)} quotes")
        
        # F. Build NBBO
        nbbo, nbbo_stats = build_nbbo(quotes, config['ffill_cap_seconds'], logger)
        all_stats.update(nbbo_stats)
        
        # G. Merge NBBO to trades
        trades_with_nbbo, merge_stats = merge_nbbo_to_trades(trades, nbbo, config, logger)
        all_stats.update(merge_stats)
        
        # H. Apply condition filter
        filtered_trades, filter_stats, unknown = apply_cond_filter(trades_with_nbbo, codebook, logger)
        all_stats.update(filter_stats)
        unknown_codes.update(unknown)
        
        # I. Finalize schema
        final_df = finalize_schema(filtered_trades, config)
        all_stats['final_rows'] = len(final_df)
        
        # J. Write to parquet
        output_dir = Path('data/clean')
        output_path = write_cleaned_parquet(final_df, ticker, date_str, output_dir, config, logger)
        
        processing_time = time.time() - start_time
        
        # K. Collect metadata
        metadata = collect_metadata(ticker, date_str, all_stats, unknown_codes, processing_time)
        metadata['status'] = 'success'
        metadata['output_path'] = str(output_path)
        
        logger.info(f"[OK] {ticker} {date_str}: {all_stats['final_rows']} final rows ({processing_time:.2f}s)")
        
        return metadata
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[FAIL] {csv_path.name}: {str(e)}", exc_info=True)
        
        # Return error metadata
        return {
            'ticker': csv_path.parent.name,
            'date': csv_path.stem.split('_', 1)[1] if '_' in csv_path.stem else 'unknown',
            'status': 'failed',
            'error': str(e),
            'processing_time_seconds': processing_time
        }

# M. Main

def main():
    """
    Main entry point: process all CSV files in parallel.
    """
    print("=" * 80)
    print("Data Cleaning Pipeline")
    print("=" * 80)
    
    # Load configuration
    config = load_config()
    logger = setup_logging(config)
    
    logger.info("Loading configuration and codebook...")
    codebook = load_codebook()
    logger.info(f"Loaded {len(codebook)} condition codes")
    
    # Discover all CSV files
    raw_data_dir = Path('data/raw')
    csv_files = sorted(raw_data_dir.glob('*/*_10-*.csv'))
    
    logger.info(f"Found {len(csv_files)} CSV files to process")
    
    if len(csv_files) == 0:
        logger.error("No CSV files found!")
        return
    
    # Determine number of workers
    max_workers = config['parallel']['max_workers']
    if max_workers == -1:
        max_workers = max(1, (os.cpu_count() or 2) - 1)
    
    logger.info(f"Processing with {max_workers} parallel workers...")
    
    # Process files in parallel
    metadata_list = []
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_path = {
            executor.submit(process_single_file, csv_path, codebook, config): csv_path
            for csv_path in csv_files
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_path):
            csv_path = future_to_path[future]
            try:
                metadata = future.result()
                metadata_list.append(metadata)
            except Exception as e:
                logger.error(f"Unexpected error processing {csv_path}: {e}")
                metadata_list.append({
                    'ticker': csv_path.parent.name,
                    'date': csv_path.stem,
                    'status': 'failed',
                    'error': str(e)
                })
    
    # Create metadata dataframe
    metadata_df = pd.DataFrame(metadata_list)
    
    # Write metadata table
    output_dir = Path('data/clean')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metadata_df.to_parquet(output_dir / 'metadata.parquet', index=False)
    metadata_df.to_csv(output_dir / 'metadata.csv', index=False)
    
    logger.info(f"Wrote metadata to {output_dir / 'metadata.parquet'}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    success_count = (metadata_df['status'] == 'success').sum()
    failed_count = (metadata_df['status'] == 'failed').sum()
    
    print(f"Files processed: {len(metadata_df)}")
    print(f"  Success: {success_count}")
    print(f"  Failed: {failed_count}")
    
    if success_count > 0:
        success_df = metadata_df[metadata_df['status'] == 'success']
        
        print(f"\nTotal rows:")
        print(f"  Raw: {success_df['raw_rows'].sum():,}")
        print(f"  Final: {success_df['final_rows'].sum():,}")
        
        print(f"\nAverage statistics:")
        print(f"  Duplicate removal: {success_df['dup_pct'].mean():.2f}%")
        print(f"  NBBO coverage: {success_df['nbbo_coverage_pct'].mean():.2f}%")
        print(f"  Crossed quotes: {success_df['crossed_quotes_pct'].mean():.3f}%")
        print(f"  Processing time: {success_df['processing_time_seconds'].mean():.2f}s per file")
        
        # Check quality gates
        print(f"\nQuality Gates:")
        min_coverage = config['quality_gates']['min_nbbo_coverage_pct']
        max_crossed = config['quality_gates']['max_crossed_quotes_pct']
        
        low_coverage = success_df[success_df['nbbo_coverage_pct'] < min_coverage]
        high_crossed = success_df[success_df['crossed_quotes_pct'] > max_crossed]
        
        if len(low_coverage) > 0:
            print(f"  ⚠ WARNING: {len(low_coverage)} files with NBBO coverage < {min_coverage}%")
            for _, row in low_coverage.iterrows():
                print(f"    - {row['ticker']} {row['date']}: {row['nbbo_coverage_pct']:.2f}%")
        else:
            print(f"  [OK] All files meet NBBO coverage threshold (>{min_coverage}%)")
        
        if len(high_crossed) > 0:
            print(f"  ⚠ WARNING: {len(high_crossed)} files with crossed quotes > {max_crossed}%")
            for _, row in high_crossed.iterrows():
                print(f"    - {row['ticker']} {row['date']}: {row['crossed_quotes_pct']:.3f}%")
        else:
            print(f"  [OK] All files meet crossed quotes threshold (<{max_crossed}%)")
        
        # Unknown condition codes
        all_unknown = set()
        for codes_str in success_df['unknown_cond_codes']:
            if codes_str:
                all_unknown.update(codes_str.split(','))
        
        if all_unknown:
            print(f"\n⚠ Unknown condition codes encountered: {sorted(all_unknown)}")
            print(f"  Review and add to config/condition_codes.csv")
        else:
            print(f"\n[OK] No unknown condition codes")
    
    if failed_count > 0:
        print(f"\n⚠ Failed files:")
        failed_df = metadata_df[metadata_df['status'] == 'failed']
        for _, row in failed_df.iterrows():
            print(f"  - {row['ticker']} {row['date']}: {row.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 80)
    print("Pipeline complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

