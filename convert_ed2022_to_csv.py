#!/usr/bin/env python3
"""
Convert and Merge NHAMCS Stata (.dta) Datasets (2018-2022) to a Single Consolidated CSV.

Usage:
    # Convert all 5 years (2018-2022) to a single combined CSV:
    python convert_ed2022_to_csv.py

    # Specify custom input files and output path:
    python convert_ed2022_to_csv.py --input ED2018-stata.dta ED2019-stata.dta ed2020-stata.dta ed2021-stata.dta ed2022-stata.dta --output datasets/nhamcs_2018_2022.csv

    # Convert a single year:
    python convert_ed2022_to_csv.py --input ed2022-stata.dta --output datasets/ed2022.csv

    # Preserve category text labels instead of numeric codes:
    python convert_ed2022_to_csv.py --preserve-labels --output nhamcs_labeled.csv
"""

import argparse
import os
import sys
import time
from typing import List
import pandas as pd


DEFAULT_FILES = [
    "ED2018-stata.dta",
    "ED2019-stata.dta",
    "ed2020-stata.dta",
    "ed2021-stata.dta",
    "ed2022-stata.dta"
]


def convert_multiple_dta_to_single_csv(
    input_files: List[str],
    output_path: str,
    preserve_labels: bool = False,
    compression: str = None
) -> None:
    """
    Reads multiple NHAMCS Stata .dta files, aligns their schemas, and exports
    a single consolidated CSV file.
    """
    valid_files = [f for f in input_files if os.path.exists(f)]
    missing_files = [f for f in input_files if not os.path.exists(f)]

    if not valid_files:
        print(f"Error: None of the specified input files exist: {input_files}", file=sys.stderr)
        sys.exit(1)

    if missing_files:
        print(f"Warning: Skipping {len(missing_files)} missing file(s): {missing_files}", file=sys.stderr)

    start_time = time.time()
    total_input_size_mb = sum(os.path.getsize(f) for f in valid_files) / (1024 * 1024)

    print("=" * 80)
    print("       NHAMCS Multi-Year Stata (.dta) to Single CSV Converter")
    print("=" * 80)
    print(f"  * Total Input Files  : {len(valid_files)} file(s) ({total_input_size_mb:.2f} MB total)")
    print(f"  * Target Output CSV  : {output_path}")
    print(f"  * Preserve Labels    : {preserve_labels}")
    print("-" * 80)

    # Ensure parent output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dfs = []
    file_summaries = []

    for idx, fpath in enumerate(valid_files, start=1):
        f_size_mb = os.path.getsize(fpath) / (1024 * 1024)
        print(f"[{idx}/{len(valid_files)}] Loading '{fpath}' ({f_size_mb:.2f} MB)...", end=" ", flush=True)
        
        t0 = time.time()
        df = pd.read_stata(fpath, convert_categoricals=preserve_labels)
        t_load = time.time() - t0

        # Normalize column names to uppercase for clean alignment
        df.columns = [col.upper() for col in df.columns]

        # Extract year if present, or infer from filename
        year_val = "Unknown"
        if "YEAR" in df.columns:
            years_in_data = df["YEAR"].dropna().unique().tolist()
            if years_in_data:
                year_val = ", ".join(str(y) for y in sorted(years_in_data))

        # Check IMMEDR counts if present
        valid_immedr_count = 0
        if "IMMEDR" in df.columns:
            if preserve_labels:
                valid_immedr_count = df["IMMEDR"].isin(["Immediate", "Emergent", "Urgent", "Semi-urgent", "Nonurgent"]).sum()
            else:
                valid_immedr_count = df["IMMEDR"].isin([1, 2, 3, 4, 5]).sum()

        dfs.append(df)
        file_summaries.append({
            "File": os.path.basename(fpath),
            "Year": year_val,
            "Rows": len(df),
            "Columns": len(df.columns),
            "Valid_ESI_Visits": valid_immedr_count,
            "Load_Time_s": round(t_load, 2)
        })
        print(f"✓ Done ({len(df):,} rows x {len(df.columns)} cols in {t_load:.2f}s)")

    print("-" * 80)
    print("Concatenating datasets into a unified dataframe...")
    combined_df = pd.concat(dfs, axis=0, ignore_index=True, sort=False)
    total_rows, total_cols = combined_df.shape
    print(f"Unified Dataframe Created: {total_rows:,} total rows x {total_cols:,} unified columns")

    print(f"Writing consolidated CSV to '{output_path}' ...", end=" ", flush=True)
    t0_write = time.time()
    combined_df.to_csv(output_path, index=False, compression=compression)
    t_write = time.time() - t0_write
    print(f"✓ Done ({t_write:.2f}s)")

    elapsed_total = time.time() - start_time
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    # Print summary table
    summary_df = pd.DataFrame(file_summaries)
    print("\n" + "=" * 80)
    print("                      CONVERSION & MERGE BREAKDOWN BY YEAR")
    print("=" * 80)
    print(summary_df.to_string(index=False))
    print("-" * 80)
    print(f"✓ Consolidated Output Summary:")
    print(f"  * Total Merged Visits     : {total_rows:,} records")
    print(f"  * Total Unified Variables : {total_cols:,} columns")
    print(f"  * Total Valid ESI (1-5)   : {summary_df['Valid_ESI_Visits'].sum():,} visits")
    print(f"  * Final CSV File Size     : {out_size_mb:.2f} MB")
    print(f"  * Destination File Path   : {os.path.abspath(output_path)}")
    print(f"  * Total Execution Time    : {elapsed_total:.2f} seconds")
    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Convert and merge NHAMCS Stata (.dta) datasets (2018-2022) into a single CSV."
    )
    parser.add_argument(
        "--input", "-i",
        nargs="+",
        default=DEFAULT_FILES,
        help="One or more paths to .dta Stata files (default: ED2018 through ed2022 Stata files)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="datasets/nhamcs_2018_2022.csv",
        help="Destination path for single merged CSV (default: datasets/nhamcs_2018_2022.csv)"
    )
    parser.add_argument(
        "--preserve-labels", "-p",
        action="store_true",
        help="Convert numerical category codes to text labels (e.g. 'Male' vs 1)"
    )
    parser.add_argument(
        "--gzip",
        action="store_true",
        help="Compress the output directly as .csv.gz"
    )

    args = parser.parse_args()

    out_path = args.output
    compression = None
    if args.gzip and not out_path.endswith('.gz'):
        out_path += '.gz'
        compression = 'gzip'

    convert_multiple_dta_to_single_csv(
        input_files=args.input,
        output_path=out_path,
        preserve_labels=args.preserve_labels,
        compression=compression
    )


if __name__ == "__main__":
    main()
