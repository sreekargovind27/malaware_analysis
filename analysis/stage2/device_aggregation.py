"""
Device-Level Feature Aggregation

Aggregates flow-level features to device-level (per unique IP address).
Creates behavioral profiles for each IoT device in the network.
"""
import os
import pandas as pd
import numpy as np
from config import Config


def aggregate_devices():
    """Aggregate flow-level data to device-level features."""
    print("\n" + "=" * 70)
    print("DEVICE-LEVEL AGGREGATION")
    print("=" * 70)

    # Load flow-level data
    print(f"📂 Loading flow features from: {Config.ENGINEERED_DATA_PATH}")
    df = pd.read_parquet(Config.ENGINEERED_DATA_PATH)
    print(f"✓ Loaded {len(df):,} flows")

    # Use originating IP as device identifier
    if 'id.orig_h' not in df.columns:
        print("❌ ERROR: 'id.orig_h' column not found")
        return

    device_col = 'id.orig_h'
    print(f"\n🔍 Aggregating by device IP ({device_col})...")

    # Get numeric features only (exclude IPs, labels, etc.)
    features = Config.get_feature_list()
    numeric_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]

    print(f"✓ Found {len(numeric_features)} numeric features to aggregate")

    # Aggregation functions
    agg_dict = {}
    for feat in numeric_features:
        agg_dict[feat] = ['mean', 'std', 'min', 'max', 'sum']

    # Add count of flows per device
    agg_dict[device_col] = 'count'

    print("\n⏳ Computing device statistics...")
    device_stats = df.groupby(device_col).agg(agg_dict)

    # Flatten multi-level columns
    device_stats.columns = ['_'.join(col).strip('_') for col in device_stats.columns.values]
    device_stats = device_stats.rename(columns={f'{device_col}_count': 'flow_count'})
    device_stats = device_stats.reset_index()

    # Add label information (most common label for each device)
    print("⏳ Determining device labels...")
    device_labels = df.groupby(device_col)[Config.TARGET_COL].agg(
        lambda x: x.value_counts().index[0]
    ).reset_index()
    device_labels.columns = [device_col, 'device_label']

    device_stats = device_stats.merge(device_labels, on=device_col, how='left')

    # Add malware family (for malicious devices)
    malicious_devices = df[df[Config.TARGET_COL] == 'Malicious'].groupby(device_col)[Config.FAMILY_TARGET_COL].agg(
        lambda x: x.value_counts().index[0] if len(x) > 0 else 'Unknown'
    ).reset_index()
    malicious_devices.columns = [device_col, 'device_malware_family']

    device_stats = device_stats.merge(malicious_devices, on=device_col, how='left')
    device_stats['device_malware_family'].fillna('Benign', inplace=True)

    # Fill NaN values in statistics
    device_stats.fillna(0, inplace=True)

    print(f"\n✓ Created {len(device_stats):,} device profiles")
    print(f"✓ Features per device: {len(device_stats.columns) - 1}")

    # Save device features
    output_path = Config.DEVICE_FEATURES_PATH
    print(f"\n💾 Saving device features to: {output_path}")
    device_stats.to_parquet(output_path, index=False)
    print("✅ Device aggregation complete!")

    # Summary statistics
    print("\n📊 Device Statistics:")
    print(f"   Total devices: {len(device_stats):,}")
    print(f"   Benign devices: {(device_stats['device_label'] == 'Benign').sum():,}")
    print(f"   Malicious devices: {(device_stats['device_label'] == 'Malicious').sum():,}")
    print(f"   Avg flows per device: {device_stats['flow_count'].mean():.1f}")
    print(f"   Max flows per device: {device_stats['flow_count'].max():,}")

    return device_stats


if __name__ == "__main__":
    aggregate_devices()