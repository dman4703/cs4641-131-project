"""
Test script for data cleaning pipeline.
Tests on 1-2 sample files before running full batch.
"""

import sys
from pathlib import Path
import pandas as pd

# Import the cleaning functions
from clean_data import (
    load_config, load_codebook, process_single_file,
    setup_logging
)


def test_single_file():
    """Test pipeline on a single file."""
    print("=" * 80)
    print("Testing Data Cleaning Pipeline on Sample Files")
    print("=" * 80)
    
    # Load config and codebook
    config = load_config()
    logger = setup_logging(config)
    codebook = load_codebook()
    
    print(f"\nLoaded {len(codebook)} condition codes")
    print(f"Config: ffill_cap={config['ffill_cap_seconds']}s")
    
    # Find sample files (first 2 files)
    raw_data_dir = Path('data/raw')
    csv_files = sorted(raw_data_dir.glob('*/*_10-*.csv'))[:2]
    
    if len(csv_files) == 0:
        print("ERROR: No CSV files found!")
        return False
    
    print(f"\nTesting on {len(csv_files)} sample files:")
    for f in csv_files:
        print(f"  - {f}")
    
    # Process each test file
    results = []
    for csv_path in csv_files:
        print(f"\n{'=' * 80}")
        print(f"Processing: {csv_path}")
        print('=' * 80)
        
        metadata = process_single_file(csv_path, codebook, config)
        results.append(metadata)
        
        if metadata['status'] == 'success':
            print(f"\n[SUCCESS]")
            print(f"  Raw rows: {metadata['raw_rows']:,}")
            print(f"  Final rows: {metadata['final_rows']:,}")
            print(f"  Duplicates removed: {metadata['duplicates_removed']} ({metadata['dup_pct']:.2f}%)")
            print(f"  NBBO coverage: {metadata['nbbo_coverage_pct']:.2f}%")
            print(f"  Crossed quotes: {metadata['crossed_quotes_pct']:.3f}%")
            print(f"  Trades dropped by cond: {metadata['trades_dropped_by_cond']}")
            print(f"  Trades dropped (no NBBO): {metadata['trades_dropped_no_nbbo']}")
            print(f"  Processing time: {metadata['processing_time_seconds']:.2f}s")
            
            if metadata['unknown_cond_codes']:
                print(f"  ⚠ Unknown codes: {metadata['unknown_cond_codes']}")
            
            # Validate output file
            output_path = Path(metadata['output_path'])
            if output_path.exists():
                print(f"\n  Output file exists: {output_path}")
                
                # Read and inspect
                df = pd.read_parquet(output_path)
                print(f"  Loaded {len(df)} rows from parquet")
                print(f"\n  Schema:")
                for col, dtype in df.dtypes.items():
                    print(f"    {col}: {dtype}")
                
                print(f"\n  First few rows:")
                print(df.head(3).to_string())
                
                print(f"\n  Data quality checks:")
                print(f"    - NaN in mid: {df['mid'].isna().sum()}")
                print(f"    - NaN in spread: {df['spread'].isna().sum()}")
                print(f"    - Negative spreads: {(df['spread'] < 0).sum()}")
                print(f"    - Price <= 0: {(df['price'] <= 0).sum()}")
                print(f"    - Size <= 0: {(df['size'] <= 0).sum()}")
                
                # Check if timestamps are sorted
                is_sorted = df['ts'].is_monotonic_increasing
                print(f"    - Timestamps sorted: {is_sorted}")
                
            else:
                print(f"  [ERROR] Output file not found!")
                return False
        else:
            print(f"\n[FAILED]")
            print(f"  Error: {metadata.get('error', 'Unknown error')}")
            return False
    
    # Summary
    print(f"\n{'=' * 80}")
    print("TEST SUMMARY")
    print('=' * 80)
    
    success_count = sum(1 for r in results if r['status'] == 'success')
    print(f"Files tested: {len(results)}")
    print(f"Success: {success_count}")
    print(f"Failed: {len(results) - success_count}")
    
    if success_count == len(results):
        print("\n[OK] All test files processed successfully!")
        return True
    else:
        print("\n[FAILED] Some test files failed. Fix errors before running full pipeline.")
        return False


if __name__ == "__main__":
    success = test_single_file()
    sys.exit(0 if success else 1)

