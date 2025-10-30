# In analysis/stage2/device_aggregation.py

import os
import sys

import pandas as pd

# Add project root to path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import Config


def aggregate_devices():
    """Aggregate flow-level data to device-level features."""
    print("\n" + "=" * 70)
    print("DEVICE-LEVEL AGGREGATION")
    print("=" * 70)

    print(f"📂 Loading flow features from: {Config.ENGINEERED_DATA_PATH}")
    df = pd.read_parquet(Config.ENGINEERED_DATA_PATH)
    print(f"✓ Loaded {len(df):,} flows")

    device_col = 'id_orig_h'
    if device_col not in df.columns:
        print(f"❌ ERROR: Device identifier column '{device_col}' not found.")
        return

    print(f"\n🔍 Aggregating by device IP ({device_col})...")
    features = Config.get_feature_list()
    numeric_features = [f for f in features if f in df.columns and pd.api.types.is_numeric_dtype(df[f])]
    print(f"✓ Found {len(numeric_features)} numeric features to aggregate.")

    # ✅ FINAL FIX: Perform aggregations in two separate, correct steps.
    print("\n⏳ Computing device statistics...")

    # Step 1: Aggregate only the numeric features
    numeric_agg_dict = {feat: ['mean', 'std', 'min', 'max', 'sum'] for feat in numeric_features}
    device_stats = df.groupby(device_col).agg(numeric_agg_dict)
    device_stats.columns = ['_'.join(col).strip() for col in device_stats.columns]  # Flatten multi-level columns

    # Step 2: Calculate the flow count separately and merge it in
    flow_counts = df.groupby(device_col).size().reset_index(name='flow_count')
    device_stats = device_stats.reset_index().merge(flow_counts, on=device_col, how='left')

    # --- Add labels and clean up ---
    print("⏳ Determining device labels...")
    device_labels = df.groupby(device_col)[Config.TARGET_COL].agg(lambda x: x.value_counts().index[0]).reset_index()
    device_labels.columns = [device_col, 'device_label']
    device_stats = device_stats.merge(device_labels, on=device_col, how='left')

    malicious_devices = df[df[Config.TARGET_COL] == 'Malicious'].groupby(device_col)[Config.FAMILY_TARGET_COL].agg(
        lambda x: x.value_counts().index[0] if not x.empty else 'Unknown').reset_index()
    malicious_devices.columns = [device_col, 'device_malware_family']
    device_stats = device_stats.merge(malicious_devices, on=device_col, how='left')

    device_stats['device_malware_family'].fillna('Benign', inplace=True)
    device_stats.fillna(0, inplace=True)

    device_stats = device_stats.apply(pd.to_numeric, errors='ignore')

    print(f"\n✓ Created {len(device_stats):,} device profiles with {len(device_stats.columns) - 1} features each.")

    print(f"\n💾 Saving device features to: {Config.DEVICE_FEATURES_PATH}")
    device_stats.to_parquet(Config.DEVICE_FEATURES_PATH, index=False)
    print("✅ Device aggregation complete!")


if __name__ == "__main__":
    aggregate_devices()
