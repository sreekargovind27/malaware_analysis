"""
Shared utility functions for Stage 2 data preparation.
"""
import pandas as pd
import numpy as np
from config import Config


def load_flow_features():
    """Load flow-level engineered features."""
    return pd.read_parquet(Config.ENGINEERED_DATA_PATH)


def load_device_features():
    """Load device-level aggregated features."""
    return pd.read_parquet(Config.DEVICE_FEATURES_PATH)


def get_benign_flows(df):
    """Filter to benign flows only."""
    return df[df[Config.TARGET_COL] == 'Benign']


def get_malicious_flows(df):
    """Filter to malicious flows only."""
    return df[df[Config.TARGET_COL] == 'Malicious']


def get_flows_by_family(df, family):
    """Filter flows by malware family."""
    return df[df[Config.FAMILY_TARGET_COL] == family]


def calculate_class_balance(df, label_col=None):
    """Calculate and print class distribution."""
    if label_col is None:
        label_col = Config.TARGET_COL

    counts = df[label_col].value_counts()
    total = len(df)

    print(f"\nClass Distribution ({label_col}):")
    for label, count in counts.items():
        pct = (count / total) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    return counts


def check_missing_values(df, verbose=True):
    """Check for missing values in DataFrame."""
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100

    if verbose:
        print("\nMissing Values:")
        for col in missing[missing > 0].index:
            print(f"  {col}: {missing[col]:,} ({missing_pct[col]:.2f}%)")

    return missing


def standardize_features(df, scaler=None, exclude_cols=None):
    """Standardize numeric features using StandardScaler."""
    from sklearn.preprocessing import StandardScaler

    if exclude_cols is None:
        exclude_cols = [Config.TARGET_COL, Config.DETAILED_TARGET_COL, Config.FAMILY_TARGET_COL]

    # Get numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    if scaler is None:
        scaler = StandardScaler()
        df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    else:
        df[numeric_cols] = scaler.transform(df[numeric_cols])

    return df, scaler


def save_train_test_val_splits(train_df, test_df, val_df=None):
    """Save train/test/val splits to parquet."""
    print("\n💾 Saving data splits...")

    train_df.to_parquet(Config.TRAIN_SET_PATH, index=False)
    print(f"✓ Saved train set: {len(train_df):,} rows")

    test_df.to_parquet(Config.TEST_SET_PATH, index=False)
    print(f"✓ Saved test set: {len(test_df):,} rows")

    if val_df is not None:
        val_df.to_parquet(Config.VAL_SET_PATH, index=False)
        print(f"✓ Saved val set: {len(val_df):,} rows")


def extract_subnet(ip_series):
    """Extract /24 subnet from IP addresses."""
    return ip_series.str.rsplit('.', n=1).str[0]


def extract_port_protocol(df):
    """Create service identifier from port and protocol."""
    return df['id.resp_p'].astype(str) + ':' + df['proto'].astype(str)