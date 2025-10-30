# In scripts/preprocess_raw_data.py

import os
import glob
import pandas as pd
from tqdm import tqdm
import sys

# Add project root to path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config

# The definitive, correct 22-column header of the original IoT-23 CSVs.
# This will be enforced on files that have inconsistent headers.
EXPECTED_COLUMNS = [
    "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
    "proto", "service", "duration", "orig_bytes", "resp_bytes",
    "conn_state", "local_orig", "local_resp", "missed_bytes", "history",
    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes", "label",
    "detailed-label"
]


def standardize_csv_files():
    """
    Reads all raw CSVs from the source directory, standardizes them, and saves them
    to a new destination. This script is idempotent and will only process missing files.
    """
    print("=" * 80)
    print("🚀 STAGE 0: STANDARDIZING RAW CSV FILES 🚀")
    print("=" * 80)

    # ✅ MODIFIED: The source is now the messy, original directory
    source_dir = Config.RAW_DIR_MESSY
    # The destination is the clean directory that the rest of the pipeline will use
    dest_dir = Config.RAW_DIR_ORIGINAL

    if not os.path.exists(source_dir):
        print(f"❌ Source directory for messy data not found: {source_dir}")
        return

    os.makedirs(dest_dir, exist_ok=True)
    print(f"Source (Messy) Directory:      {source_dir}")
    print(f"Destination (Clean) Directory: {dest_dir}")

    raw_csv_files = glob.glob(os.path.join(source_dir, '*.csv'))
    if not raw_csv_files:
        print("❌ No raw CSV files found to process.")
        return

    files_to_process = []
    files_skipped = 0

    # ✅ NEW LOGIC: Check which files already exist in the destination
    for file_path in raw_csv_files:
        filename = os.path.basename(file_path)
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            files_skipped += 1
        else:
            files_to_process.append(file_path)

    print(f"\nFound {len(raw_csv_files)} total raw files.")
    print(f"  - Skipping {files_skipped} files (already standardized).")
    print(f"  - Processing {len(files_to_process)} new or missing files.")

    if not files_to_process:
        print("\n✅ No new files to process. Standardization is up to date.")
        print("=" * 80)
        return

    for file_path in tqdm(sorted(files_to_process), desc="Standardizing Files"):
        filename = os.path.basename(file_path)
        source_folder_value = filename.replace('.csv', '')
        dest_path = os.path.join(dest_dir, filename)

        try:
            # Read the potentially messy CSV.
            df = pd.read_csv(file_path, comment='#', low_memory=False, on_bad_lines='warn', header=0)

            # --- This is the core logic for handling inconsistent headers ---
            if len(df.columns) != len(EXPECTED_COLUMNS):
                tqdm.write(f"  - WARNING: Header mismatch in {filename}. Forcing standard 22-column header.")
                df = pd.read_csv(
                    file_path,
                    comment='#',
                    header=None,
                    skiprows=1,
                    names=EXPECTED_COLUMNS
                )

            # Add the Source_Folder column at the beginning
            if 'Source_Folder' not in df.columns:
                df.insert(0, 'Source_Folder', source_folder_value)
            else:
                # Update existing column if it's wrong
                df['Source_Folder'] = source_folder_value

            # Save the clean, standardized CSV
            df.to_csv(dest_path, index=False)

        except Exception as e:
            tqdm.write(f"\n❌ FAILED to process {filename}: {e}")
            continue

    print("\n" + "=" * 80)
    print("✅ STANDARDIZATION COMPLETE!")
    print(f"Clean files are located in: {dest_dir}")
    print("=" * 80)


if __name__ == "__main__":
    standardize_csv_files()