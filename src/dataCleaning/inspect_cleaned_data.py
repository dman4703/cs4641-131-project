"""
Utility to inspect cleaned data and generate statistics.
Use after running the cleaning pipeline.
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np


def inspect_metadata():
    """Display summary statistics from metadata table."""
    metadata_path = Path('data/clean/metadata.parquet')
    
    if not metadata_path.exists():
        print("[X] Metadata file not found. Run cleaning pipeline first.")
        return False
    
    print("=" * 80)
    print("CLEANED DATA INSPECTION")
    print("=" * 80)
    
    # Load metadata
    md = pd.read_parquet(metadata_path)
    
    print(f"\nProcessing Summary:")
    print(f"  Total files: {len(md)}")
    print(f"  Successful: {(md['status'] == 'success').sum()}")
    print(f"  Failed: {(md['status'] == 'failed').sum()}")
    
    success_md = md[md['status'] == 'success']
    
    if len(success_md) == 0:
        print("\n[X] No successful files found.")
        return False
    
    print(f"\nData Volume:")
    print(f"  Raw rows: {success_md['raw_rows'].sum():,}")
    print(f"  Final rows: {success_md['final_rows'].sum():,}")
    print(f"  Retention: {success_md['final_rows'].sum() / success_md['raw_rows'].sum() * 100:.1f}%")
    
    print(f"\nQuality Metrics (Average):")
    print(f"  NBBO coverage: {success_md['nbbo_coverage_pct'].mean():.2f}%")
    print(f"    - Min: {success_md['nbbo_coverage_pct'].min():.2f}%")
    print(f"    - Max: {success_md['nbbo_coverage_pct'].max():.2f}%")
    
    print(f"  Crossed quotes: {success_md['crossed_quotes_pct'].mean():.3f}%")
    print(f"    - Min: {success_md['crossed_quotes_pct'].min():.3f}%")
    print(f"    - Max: {success_md['crossed_quotes_pct'].max():.3f}%")
    
    print(f"  Duplicate removal: {success_md['dup_pct'].mean():.2f}%")
    print(f"  Trades dropped (cond codes): {success_md['trades_dropped_by_cond'].sum():,}")
    print(f"  Trades dropped (no NBBO): {success_md['trades_dropped_no_nbbo'].sum():,}")
    
    print(f"\nPerformance:")
    print(f"  Avg processing time: {success_md['processing_time_seconds'].mean():.2f}s per file")
    print(f"  Total processing time: {success_md['processing_time_seconds'].sum():.1f}s")
    
    # Per ticker summary
    print(f"\nPer-Ticker Summary:")
    ticker_stats = success_md.groupby('ticker').agg({
        'final_rows': 'sum',
        'nbbo_coverage_pct': 'mean',
        'crossed_quotes_pct': 'mean'
    }).round(2)
    ticker_stats.columns = ['Total Rows', 'Avg NBBO Cov %', 'Avg Crossed %']
    ticker_stats = ticker_stats.sort_values('Total Rows', ascending=False)
    print(ticker_stats.to_string())
    
    # Check for issues
    print(f"\nQuality Warnings:")
    low_coverage = success_md[success_md['nbbo_coverage_pct'] < 99.0]
    if len(low_coverage) > 0:
        print(f"  Files with NBBO coverage < 99%: {len(low_coverage)}")
        for _, row in low_coverage.iterrows():
            print(f"    - {row['ticker']} {row['date']}: {row['nbbo_coverage_pct']:.2f}%")
    else:
        print(f"  ✓ All files have NBBO coverage ≥ 99%")
    
    high_crossed = success_md[success_md['crossed_quotes_pct'] > 0.5]
    if len(high_crossed) > 0:
        print(f"  Files with crossed quotes > 0.5%: {len(high_crossed)}")
        for _, row in high_crossed.iterrows():
            print(f"    - {row['ticker']} {row['date']}: {row['crossed_quotes_pct']:.3f}%")
    else:
        print(f"  ✓ All files have crossed quotes ≤ 0.5%")
    
    # Unknown codes
    unknown_codes = set()
    for codes_str in success_md['unknown_cond_codes']:
        if codes_str and pd.notna(codes_str):
            unknown_codes.update(str(codes_str).split(','))
    
    if unknown_codes:
        print(f"\n  Unknown condition codes found: {sorted(unknown_codes)}")
        print(f"  → Add these to config/condition_codes.csv and re-run if needed")
    else:
        print(f"  ✓ No unknown condition codes")
    
    # Failed files
    if (md['status'] == 'failed').any():
        print(f"\nFailed Files:")
        failed_md = md[md['status'] == 'failed']
        for _, row in failed_md.iterrows():
            print(f"  - {row['ticker']} {row['date']}: {row.get('error', 'Unknown error')}")
    
    return True


def inspect_sample_file(ticker: str = None, date: str = None):
    """Inspect a specific cleaned file or pick first available."""
    clean_dir = Path('data/clean')
    
    if ticker and date:
        file_path = clean_dir / ticker / f"{ticker}_{date}_clean.parquet"
    else:
        # Find first available file
        parquet_files = list(clean_dir.glob('*/*_clean.parquet'))
        if not parquet_files:
            print("No cleaned parquet files found.")
            return False
        file_path = parquet_files[0]
        ticker = file_path.parent.name
        date = file_path.stem.replace(f"{ticker}_", "").replace("_clean", "")
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return False
    
    print(f"\n{'=' * 80}")
    print(f"SAMPLE FILE INSPECTION: {ticker} {date}")
    print('=' * 80)
    
    # Load data
    df = pd.read_parquet(file_path)
    
    print(f"\nFile: {file_path}")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Memory: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    print(f"\nSchema:")
    for col in df.columns:
        print(f"  {col:15s} {str(df[col].dtype):20s} (nulls: {df[col].isna().sum()})")
    
    print(f"\nPrice Statistics:")
    print(f"  Price range: ${df['price'].min():.2f} - ${df['price'].max():.2f}")
    print(f"  Mean price: ${df['price'].mean():.2f}")
    print(f"  Median price: ${df['price'].median():.2f}")
    
    print(f"\nNBBO Statistics:")
    print(f"  Mid range: ${df['mid'].min():.2f} - ${df['mid'].max():.2f}")
    print(f"  Mean spread: ${df['spread'].mean():.4f}")
    print(f"  Median spread: ${df['spread'].median():.4f}")
    print(f"  Max spread: ${df['spread'].max():.4f}")
    
    print(f"\nSize Statistics:")
    print(f"  Mean size: {df['size'].mean():.0f} shares")
    print(f"  Median size: {df['size'].median():.0f} shares")
    print(f"  Max size: {df['size'].max():,} shares")
    
    print(f"\nTimestamp Range:")
    print(f"  Start: {df['ts'].min()}")
    print(f"  End: {df['ts'].max()}")
    print(f"  Duration: {(df['ts'].max() - df['ts'].min()).total_seconds() / 3600:.2f} hours")
    
    print(f"\nExchange Distribution:")
    print(df['exch'].value_counts().to_string())
    
    print(f"\nCondition Code Distribution (Top 10):")
    print(df['cond_norm'].value_counts().head(10).to_string())
    
    if 'at_bid' in df.columns and 'at_ask' in df.columns:
        print(f"\nMicrostructure:")
        print(f"  Trades at bid: {df['at_bid'].sum():,} ({df['at_bid'].mean()*100:.1f}%)")
        print(f"  Trades at ask: {df['at_ask'].sum():,} ({df['at_ask'].mean()*100:.1f}%)")
        print(f"  Trades between: {((df['at_bid'] == 0) & (df['at_ask'] == 0)).sum():,}")
    
    print(f"\nData Quality:")
    print(f"  NaNs in price: {df['price'].isna().sum()}")
    print(f"  NaNs in size: {df['size'].isna().sum()}")
    print(f"  NaNs in mid: {df['mid'].isna().sum()}")
    print(f"  NaNs in spread: {df['spread'].isna().sum()}")
    print(f"  Negative spreads: {(df['spread'] < 0).sum()}")
    print(f"  Zero spreads: {(df['spread'] == 0).sum()}")
    print(f"  Timestamps sorted: {df['ts'].is_monotonic_increasing}")
    
    print(f"\nSample Data (first 5 rows):")
    print(df.head().to_string())
    
    print(f"\nSample Data (last 5 rows):")
    print(df.tail().to_string())
    
    return True


def main():
    """Main entry point."""
    # Check metadata first
    if not inspect_metadata():
        sys.exit(1)
    
    # Inspect a sample file
    if not inspect_sample_file():
        sys.exit(1)
    
    print(f"\n{'=' * 80}")
    print("Inspection complete!")
    print('=' * 80)


if __name__ == "__main__":
    main()

