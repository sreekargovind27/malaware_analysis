"""
Performs a data quality check on the final engineered dataset.
It verifies the file's existence, checks for missing values, and reports on the
distribution of the target labels.
"""
import os

from analysis.stage2.utils import load_engineered_data
from config import Config


def check_engineered_data_quality():
    """Checks the data quality of the final engineered Parquet file."""
    print("=" * 60)
    print("Engineered Data Quality Check")
    print("=" * 60)

    if not os.path.exists(Config.ENGINEERED_DATA_PATH):
        print(f"Engineered data file not found at '{Config.ENGINEERED_DATA_PATH}'")
        print("Please run 'build_dataset.py' first.")
        return

    print(f"\nLoading engineered data...")
    df = load_engineered_data()
    features = Config.get_feature_list()

    print(f"Total rows: {len(df):,}")
    print(f"Total features: {len(features)}")

    # Check for any remaining missing values in the feature columns.
    missing_values = df[features].isna().sum().sum()
    if missing_values > 0:
        print(f"\n⚠ Found {missing_values} missing values in feature columns! This should be zero.")
    else:
        print("\n✓ No missing values found in feature columns.")

    # Check the distribution of labels.
    print("\n" + "=" * 60)
    print("Binary Label Distribution (from 'label' column)")
    print("=" * 60)
    print(df[Config.TARGET_COL].value_counts())

    print("\n" + "=" * 60)
    print("Multi-Class Label Distribution (from 'attack_type' column)")
    print("=" * 60)
    print(df[Config.DETAILED_TARGET_COL].value_counts())

    print("\n" + "=" * 60)
    print("Malware Family Distribution (from 'malware_family' column)")
    print("=" * 60)
    print(df[Config.FAMILY_TARGET_COL].value_counts())

    print("\n" + "=" * 60)
    print("✓ Data quality check complete!")
    print("=" * 60)


if __name__ == "__main__":
    check_engineered_data_quality()
