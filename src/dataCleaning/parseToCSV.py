import os
from pathlib import Path
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

EXCEL_DIR = Path(r"C:\Users\doqui\OneDrive\Documents\cs4641-131-project\data\excel")
OUTPUT_ROOT = Path(r"C:\Users\doqui\OneDrive\Documents\cs4641-131-project\data\raw")

# Only do these dates (by filename prefix)
TARGET_PREFIXES = ("10_6_", "10_7_", "10_8_")

def process_excel_file(excel_path: Path, output_root: Path) -> str:
    """
    Process one Excel workbook: read each sheet and write CSV to /raw/<Ticker>/<Ticker>_<MM-DD-YY>.csv
    Skips writing if the CSV already exists.
    Returns a short status string for logging.
    """
    parts = excel_path.stem.split("_")
    if len(parts) < 3:
        return f"SKIP (malformed filename): {excel_path.name}"

    month, day, year = parts[0], parts[1], parts[2]
    date_str = f"{month}-{day}-{year}"

    wrote = 0
    skipped = 0

    try:
        with pd.ExcelFile(excel_path, engine="openpyxl") as xls:
            for sheet_name in xls.sheet_names:
                ticker = sheet_name.split()[0]

                ticker_dir = output_root / ticker
                ticker_dir.mkdir(parents=True, exist_ok=True)

                csv_path = ticker_dir / f"{ticker}_{date_str}.csv"
                if csv_path.exists():
                    skipped += 1
                    continue

                try:
                    df = pd.read_excel(xls, sheet_name=sheet_name, engine="openpyxl")
                except Exception as e:
                    # Continue past bad sheets; report at the end
                    return f"ERROR reading sheet '{sheet_name}' in {excel_path.name}: {e}"

                df.to_csv(csv_path, index=False)
                wrote += 1

    except Exception as e:
        return f"ERROR opening {excel_path.name}: {e}"

    return f"OK {excel_path.name}: wrote {wrote}, skipped {skipped}"

def select_target_excels(root: Path):
    """Return only the Excel files for 10_6, 10_7, 10_8."""
    all_xlsx = sorted(root.glob("*_tickDataStatic.xlsx"))
    return [p for p in all_xlsx if p.name.startswith(TARGET_PREFIXES)]

if __name__ == "__main__":
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    excel_files = select_target_excels(EXCEL_DIR)
    if not excel_files:
        print("No matching Excel files for 10_6, 10_7, or 10_8.")
        raise SystemExit(0)

    # Use up to (logical_cores - 1) workers, but not more than number of files.
    max_workers = max(1, min(len(excel_files), (os.cpu_count() or 2) - 1))
    print(f"Processing {len(excel_files)} workbook(s) with {max_workers} worker(s)...")

    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futures = {ex.submit(process_excel_file, p, OUTPUT_ROOT): p for p in excel_files}
        for fut in as_completed(futures):
            msg = fut.result()
            results.append(msg)
            print(msg)

    print("\nSummary:")
    for r in results:
        print(" -", r)
