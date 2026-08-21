#!/usr/bin/env python3
"""
Convert NHAMCS Stata (.dta) Dataset to CSV.

Usage:
    python convert_ed2022_to_csv.py
    python convert_ed2022_to_csv.py --input ed2022-stata.dta --output datasets/ed2022.csv
    python convert_ed2022_to_csv.py --input ed2022-stata.dta --output ed2022_labeled.csv --preserve-labels
"""

import argparse
import os
import sys
import time
import pandas as pd


def convert_dta_to_csv(
    input_path: str,
    output_path: str,
    preserve_labels: bool = False,
    chunksize: int = 10000,
    compression: str = None
) -> None:
    """
    Reads a Stata .dta file and exports it to CSV in chunks for memory efficiency.
    """
    if not os.path.exists(input_path):
        print(f"Error: Input file '{input_path}' does not exist.", file=sys.stderr)
        sys.exit(1)

    start_time = time.time()
    in_size_mb = os.path.getsize(input_path) / (1024 * 1024)
    print("=" * 70)
    print(f"  NHAMCS Stata (.dta) to CSV Converter")
    print("=" * 70)
    print(f"  * Input File         : {input_path} ({in_size_mb:.2f} MB)")
    print(f"  * Output File        : {output_path}")
    print(f"  * Preserve Labels    : {preserve_labels}")
    print(f"  * Chunk Size         : {chunksize:,} rows")
    print("-" * 70)

    # Ensure parent output directory exists
    out_dir = os.path.dirname(output_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    print("Reading and converting dataset in chunks...")
    total_rows = 0
    total_cols = 0

    try:
        reader = pd.read_stata(
            input_path,
            convert_categoricals=preserve_labels,
            iterator=True,
            chunksize=chunksize
        )

        first_chunk = True
        for i, chunk in enumerate(reader, start=1):
            if first_chunk:
                total_cols = chunk.shape[1]
                # Write header on first chunk
                chunk.to_csv(output_path, mode='w', index=False, header=True, compression=compression)
                first_chunk = False
            else:
                # Append subsequent chunks without header
                chunk.to_csv(output_path, mode='a', index=False, header=False, compression=compression)

            total_rows += len(chunk)
            print(f"  -> Processed Chunk {i:2d} ({total_rows:,} total rows converted)...")

    except Exception as e:
        print(f"\nChunked reading failed ({e}). Falling back to full memory read...", file=sys.stderr)
        df = pd.read_stata(input_path, convert_categoricals=preserve_labels)
        total_rows, total_cols = df.shape
        df.to_csv(output_path, index=False, compression=compression)

    elapsed = time.time() - start_time
    out_size_mb = os.path.getsize(output_path) / (1024 * 1024)

    print("-" * 70)
    print(f"✓ Conversion Complete in {elapsed:.2f} seconds!")
    print(f"  * Total Observations : {total_rows:,}")
    print(f"  * Total Variables    : {total_cols:,}")
    print(f"  * CSV File Size      : {out_size_mb:.2f} MB")
    print(f"  * Saved Location     : {os.path.abspath(output_path)}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Convert NHAMCS Stata (.dta) dataset to standard CSV format."
    )
    parser.add_argument(
        "--input", "-i",
        type=str,
        default="ed2022-stata.dta",
        help="Path to input .dta Stata file (default: ed2022-stata.dta)"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="ed2022.csv",
        help="Path to destination .csv file (default: ed2022.csv)"
    )
    parser.add_argument(
        "--preserve-labels", "-p",
        action="store_true",
        help="Convert numerical category codes to their text labels (e.g. 'Male' instead of 1)"
    )
    parser.add_argument(
        "--chunksize", "-c",
        type=int,
        default=10000,
        help="Number of rows per chunk for streaming memory efficiency (default: 10000)"
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

    convert_dta_to_csv(
        input_path=args.input,
        output_path=out_path,
        preserve_labels=args.preserve_labels,
        chunksize=args.chunksize,
        compression=compression
    )


if __name__ == "__main__":
    main()
