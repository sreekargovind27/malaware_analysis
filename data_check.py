"""
Quick data quality check on the FINAL ENGINEERED DATASET.
"""
import pandas as pd
import os
from config import Config

def check_engineered_data_quality():
    """Check data quality on the final engineered Parquet file."""
    print("=" * 60)
    print("Engineered Data Quality Check")
    print("=" * 60)

    if not os.path.exists(Config.ENGINEERED_DATA_PATH):
        print(f"Engineered data file not found at '{Config.ENGINEERED_DATA_PATH}'")
        print("Please run 'run_feature_engineering.py' first.")
        return

    print(f"\nChecking '{Config.ENGINEERED_DATA_PATH}'...")
    df = pd.read_parquet(Config.ENGINEERED_DATA_PATH)
    features = Config.get_feature_list()

    print(f"Total rows: {len(df):,}")
    print(f"Total features: {len(features)}")

    # Check for any remaining missing values in feature columns
    missing_values = df[features].isna().sum().sum()
    if missing_values > 0:
        print(f"\n⚠ Found {missing_values} missing values in feature columns! This should be zero.")
    else:
        print("\n✓ No missing values found in feature columns.")

    # Check label distribution
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