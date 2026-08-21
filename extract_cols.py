"""
extract_cols.py — Extract all column names from datasets/5v_cleandf.RData

Uses Rscript (subprocess) to load the RData file, extract column names from
the largest dataframe, and save them as a JSON array to val_columns.json.
"""

import subprocess
import json
import os
import sys

def main():
    rdata_path = "datasets/5v_cleandf.RData"

    # Resolve path relative to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    rdata_full = os.path.join(script_dir, rdata_path)

    if not os.path.exists(rdata_full):
        print(f"Error: {rdata_full} not found.")
        sys.exit(1)

    # R script to extract column names as JSON array
    r_code = f"""
data_env <- new.env()
load("{rdata_full}", envir = data_env)

df_names <- ls(data_env)[sapply(ls(data_env), function(x) is.data.frame(get(x, envir = data_env)))]
df_sizes <- sapply(df_names, function(x) ncol(get(x, envir = data_env)))
target_df <- get(df_names[which.max(df_sizes)], envir = data_env)

col_names <- names(target_df)
cat(paste0('[', paste0('"', col_names, '"', collapse = ','), ']'))
"""

    print(f"Loading RData from: {rdata_full}")
    result = subprocess.run(
        ["Rscript", "-e", r_code],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        print(f"Rscript error:\n{result.stderr}")
        sys.exit(1)

    # Parse the JSON array output from R
    columns = json.loads(result.stdout.strip())

    # Save to val_columns.json
    out_path = os.path.join(script_dir, "val_columns.json")
    with open(out_path, "w") as f:
        json.dump(columns, f, indent=2)

    print(f"Extracted {len(columns)} column names from {rdata_path}")
    print(f"Saved to: {out_path}")

if __name__ == "__main__":
    main()
