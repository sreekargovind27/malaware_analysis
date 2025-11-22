# In analysis/stage2/diagnose_raw_data.py

import os
import glob
import pandas as pd
import numpy as np
import sys
from datetime import datetime

# We need to add the project root to the path to import Config
# This is necessary because we are running this script directly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..', '..')))
from config import Config


def check_mixed_type(series):
    """
    Checks if a pandas Series contains mixed data types (numbers and non-numbers).
    Returns True if it finds a mix, False otherwise.
    """
    # Drop NaNs for the check, as they don't indicate a "mix" of intent.
    series = series.dropna()
    if series.empty:
        return False

    # Attempt to convert all values to numeric, coercing errors to NaN.
    numeric_series = pd.to_numeric(series, errors='coerce')

    # If the number of NaNs in the numeric series is greater than zero,
    # it means there was at least one value that could not be converted to a number.
    has_non_numeric = numeric_series.isnull().any()

    # Check if the original series (without NaNs) has any numeric types already.
    # This is a simple way to see if there's a mix.
    has_numeric = any(isinstance(x, (int, float, np.number)) for x in series)

    return has_numeric and has_non_numeric


def main():
    """
    Runs a deep diagnostic on all raw CSV files and logs the output to a file.
    """
    # ✅ MODIFIED: Set up logging to a file in the project root
    log_file_path = os.path.join(Config.PROJECT_ROOT, 'raw_data_diagnostic_report.log')

    # Redirect stdout to the log file
    original_stdout = sys.stdout
    with open(log_file_path, 'w') as f:
        sys.stdout = f

        print("=" * 80)
        print(f"🚀 RAW DATA DIAGNOSTIC REPORT 🚀")
        print(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        csv_files = glob.glob(os.path.join(Config.RAW_DIR_ORIGINAL, '*.csv'))
        if not csv_files:
            print(f"❌ No CSV files found in '{Config.RAW_DIR_ORIGINAL}'. Exiting.")
            return

        print(f"Found {len(csv_files)} files to analyze.\n")

        for file_path in sorted(csv_files):
            print("\n" + "=" * 80)
            print(f"🔬 ANALYZING FILE: {os.path.basename(file_path)}")
            print("=" * 80)
            try:
                df = pd.read_csv(file_path, low_memory=False, on_bad_lines='skip', comment='#')

                for col in df.columns:
                    print(f"\n--- COLUMN: '{col}' ---")
                    series = df[col]

                    print(f"  - Inferred Dtype: {series.dtype}")
                    is_mixed = check_mixed_type(series)
                    print(f"  - Contains Mixed Types (str/num): {is_mixed}")

                    unique_vals = series.dropna().unique()
                    print(f"  - Sample Unique Values: {unique_vals[:10]}")

                    if 'float' in str(series.dtype) or 'int' in str(series.dtype):
                        print("  - Strict Numeric Conversion: ✅ Already numeric.")
                    else:
                        try:
                            pd.to_numeric(series.dropna(), errors='raise')
                            print("  - Strict Numeric Conversion: ✅ SUCCESS (Can be converted safely)")
                        except (ValueError, TypeError) as e:
                            print(f"  - Strict Numeric Conversion: ❌ FAILURE")
                            print(f"    └─ Error on value like: '{e}'")

            except Exception as e:
                print(f"\n❌❌❌ CRITICAL ERROR reading or processing file {os.path.basename(file_path)}: {e} ❌❌❌")

        print("\n" + "=" * 80)
        print("✅ DIAGNOSTIC COMPLETE")
        print(f"📄 Report saved to: {os.path.basename(log_file_path)}")
        print("=" * 80)

    # Restore stdout
    sys.stdout = original_stdout
    print(f"✅ Diagnostic complete. Report saved to: {log_file_path}")


if __name__ == "__main__":
    main()