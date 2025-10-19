"""
Lightweight data loaders for model training.
All functions read from the final, pre-engineered dataset created by 'build_dataset.py'.
UPDATED: Added detailed timing for all operations.
"""
import os
import time

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

from config import Config


class IoTDataset(Dataset):
    def __init__(self, X, y=None):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def load_engineered_data():
    """Load the pre-engineered dataset with timing."""
    print("\n" + "=" * 70)
    print("📂 LOADING ENGINEERED DATA")
    print("=" * 70)

    t_start = time.time()

    if not os.path.exists(Config.ENGINEERED_DATA_PATH):
        raise FileNotFoundError(
            f"Engineered data not found at '{Config.ENGINEERED_DATA_PATH}'. "
            f"Please run 'build_dataset.py' first."
        )

    file_size_mb = os.path.getsize(Config.ENGINEERED_DATA_PATH) / (1024 * 1024)
    print(f"\n⏳ Loading from: {Config.ENGINEERED_DATA_PATH}")
    print(f"   File size: {file_size_mb:.1f} MB")

    df = pd.read_parquet(Config.ENGINEERED_DATA_PATH)

    t_elapsed = time.time() - t_start

    print(f"\n✅ Data loaded successfully")
    print(f"   Rows: {len(df):,}")
    print(f"   Columns: {len(df.columns)}")
    print(f"⏱️  Load time: {t_elapsed:.2f}s ({file_size_mb / t_elapsed:.1f} MB/s)")
    print("=" * 70)

    return df


def get_data_for_autoencoder():
    """Load and prepare data for Autoencoder (benign only) with timing."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING DATA FOR AUTOENCODER")
    print("=" * 70)

    overall_start = time.time()

    df = load_engineered_data()
    features = Config.get_feature_list()

    print(f"\n⏳ Filtering benign data...")
    t_filter_start = time.time()
    benign_df = df.loc[df[Config.TARGET_COL] == 'Benign', features]
    t_filter = time.time() - t_filter_start
    print(f"✓ Filtered to {len(benign_df):,} benign samples ({t_filter:.2f}s)")

    # Remove constant columns
    print(f"\n⏳ Removing constant columns...")
    t_const_start = time.time()
    non_constant_cols = benign_df.columns[benign_df.std() > 1e-6].tolist()

    if len(non_constant_cols) < len(features):
        removed_cols = set(features) - set(non_constant_cols)
        print(f"   Removed {len(removed_cols)} constant columns")

    if not non_constant_cols:
        print("❌ ERROR: No features with variance found!")
        return None, None, None, []

    t_const = time.time() - t_const_start
    print(f"✓ Kept {len(non_constant_cols)} features with variance ({t_const:.2f}s)")

    # Save feature list
    joblib.dump(non_constant_cols, Config.AUTOENCODER_FEATURE_LIST_PATH)

    # Get data
    X = benign_df[non_constant_cols]

    # Handle NaN BEFORE scaling
    print(f"\n⏳ Checking for missing values...")
    nan_count = X.isna().sum().sum()
    if nan_count > 0:
        print(f"   ⚠️  Found {nan_count:,} NaN values. Filling with 0...")
        X = X.fillna(0)
    else:
        print(f"   ✓ No missing values")

    # Scale
    print(f"\n⏳ Scaling features...")
    t_scale_start = time.time()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    t_scale = time.time() - t_scale_start
    print(f"✓ Scaling complete ({t_scale:.2f}s)")

    # CRITICAL: Handle NaN/Inf AFTER scaling
    nan_count = np.isnan(X_scaled).sum()
    if nan_count > 0:
        print(f"   ⚠️  Found {nan_count:,} NaN after scaling. Replacing with 0...")
        X_scaled = np.nan_to_num(X_scaled, nan=0.0)

    inf_count = np.isinf(X_scaled).sum()
    if inf_count > 0:
        print(f"   ⚠️  Found {inf_count:,} Inf values. Replacing with 0...")
        X_scaled = np.nan_to_num(X_scaled, posinf=0.0, neginf=0.0)

    # Check for extreme values
    max_val = np.abs(X_scaled).max()
    if max_val > 100:
        print(f"   ⚠️  Large values detected (max={max_val:.2f}). Clipping to [-100, 100]...")
        X_scaled = np.clip(X_scaled, -100, 100)

    # Final check
    if np.isnan(X_scaled).any() or np.isinf(X_scaled).any():
        print("   ❌ ERROR: Still have NaN/Inf after cleaning! Setting all to 0...")
        X_scaled = np.zeros_like(X_scaled)

    # Split train/val
    print(f"\n⏳ Creating train/validation split...")
    t_split_start = time.time()

    if len(X_scaled) < 2:
        print("❌ ERROR: Not enough data for train/val split")
        return None, None, None, []

    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=Config.RANDOM_STATE)
    t_split = time.time() - t_split_start
    print(f"✓ Split complete ({t_split:.2f}s)")

    # Create DataLoaders
    print(f"\n⏳ Creating PyTorch DataLoaders...")
    t_loader_start = time.time()

    train_loader = DataLoader(
        IoTDataset(X_train),
        batch_size=Config.AUTOENCODER_BATCH_SIZE,  # Larger batch
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,          # ← ADD THIS
        persistent_workers=True   # ← ADD THIS
    )

    val_loader = DataLoader(
        IoTDataset(X_val),
        batch_size=Config.AUTOENCODER_BATCH_SIZE,
        shuffle=False,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True,          # ← ADD THIS
        persistent_workers=True   # ← ADD THIS
    )

    t_loader = time.time() - t_loader_start
    print(f"✓ DataLoaders created ({t_loader:.2f}s)")

    total_time = time.time() - overall_start

    print(f"\n" + "=" * 70)
    print(f"✅ AUTOENCODER DATA READY")
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Val samples:   {len(X_val):,}")
    print(f"   Features:      {len(non_constant_cols)}")
    print(f"   Batch size:    {Config.AUTOENCODER_BATCH_SIZE}")
    print(f"   Data range:    [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")
    print(f"⏱️  Total prep time: {total_time:.2f}s")
    print("=" * 70)

    return train_loader, val_loader, scaler, non_constant_cols


def get_data_for_binary():
    """Load and prepare data for binary classification with timing."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING DATA FOR BINARY CLASSIFICATION")
    print("=" * 70)

    overall_start = time.time()

    df = load_engineered_data()
    features = Config.get_feature_list()

    print(f"\n⏳ Extracting features and labels...")
    t_extract_start = time.time()
    X = df[features]
    y = (df[Config.TARGET_COL] == 'Malicious').astype(int)
    t_extract = time.time() - t_extract_start
    print(f"✓ Extraction complete ({t_extract:.2f}s)")

    # Show class distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"\n📊 Class distribution:")
    for cls, count in zip(unique, counts):
        label = "Benign" if cls == 0 else "Malicious"
        print(f"   {label}: {count:,} ({count / len(y) * 100:.2f}%)")

    print(f"\n⏳ Creating train/test split (stratified)...")
    t_split_start = time.time()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=y
    )
    t_split = time.time() - t_split_start
    print(f"✓ Split complete ({t_split:.2f}s)")

    total_time = time.time() - overall_start

    print(f"\n" + "=" * 70)
    print(f"✅ BINARY DATA READY")
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Test samples:  {len(X_test):,}")
    print(f"   Features:      {X.shape[1]}")
    print(f"⏱️  Total prep time: {total_time:.2f}s")
    print("=" * 70)

    return X_train, X_test, y_train, y_test


def get_data_for_multiclass():
    """Load and prepare data for multi-class classification with timing."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING DATA FOR MULTI-CLASS CLASSIFICATION")
    print("=" * 70)
    overall_start = time.time()
    df = load_engineered_data()
    features = Config.get_feature_list()

    # --- FIX STARTS HERE ---
    # 1. Identify classes with too few samples BEFORE encoding
    print(f"\n⏳ Filtering classes with sufficient samples for splitting...")
    value_counts = df[Config.DETAILED_TARGET_COL].value_counts()
    to_keep = value_counts[value_counts >= 2].index

    if len(to_keep) < len(value_counts):
        to_remove = value_counts[value_counts < 2].index
        print(f"   ⚠️  Removed {len(to_remove)} classes with only 1 sample: {list(to_remove)}")
        df_filtered = df[df[Config.DETAILED_TARGET_COL].isin(to_keep)]
    else:
        df_filtered = df
        print(f"   ✓ All classes have sufficient samples.")
    # --- FIX ENDS HERE ---

    print(f"\n⏳ Encoding labels...")
    le = LabelEncoder()
    X = df_filtered[features]
    y = le.fit_transform(df_filtered[Config.DETAILED_TARGET_COL])
    print(f"✓ Encoding complete")

    print(f"\n📊 Class distribution (after filtering):")
    for idx, class_name in enumerate(le.classes_):
        count = (y == idx).sum()
        print(f"   {class_name}: {count:,} ({count / len(y) * 100:.2f}%)")

    print(f"\n⏳ Creating train/test split (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_STATE, stratify=y
    )
    print(f"✓ Split complete")

    total_time = time.time() - overall_start
    print(f"\n" + "=" * 70)
    print(f"✅ MULTI-CLASS DATA READY")
    print(
        f"   Train samples: {len(X_train):,}\n   Test samples:  {len(X_test):,}\n   Features:      {X.shape[1]}\n   Classes:       {len(le.classes_)}")
    print(f"⏱️  Total prep time: {total_time:.2f}s")
    print("=" * 70)
    return X_train, X_test, y_train, y_test, le


def get_data_for_virus():
    """Load and prepare data for virus/malware family classification with timing."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING DATA FOR VIRUS CLASSIFICATION")
    print("=" * 70)

    overall_start = time.time()

    df = load_engineered_data()
    features = Config.get_feature_list()

    print(f"\n⏳ Filtering malicious samples...")
    t_filter_start = time.time()
    malicious_df = df[df[Config.TARGET_COL] == 'Malicious'].copy()
    malicious_df = malicious_df[malicious_df[Config.FAMILY_TARGET_COL] != 'Benign']
    t_filter = time.time() - t_filter_start
    print(f"✓ Filtered to {len(malicious_df):,} malicious samples ({t_filter:.2f}s)")

    # Filter out families with only one sample
    print(f"\n⏳ Filtering families with sufficient samples...")
    value_counts = malicious_df[Config.FAMILY_TARGET_COL].value_counts()
    to_keep = value_counts[value_counts >= 2].index
    malicious_df = malicious_df[malicious_df[Config.FAMILY_TARGET_COL].isin(to_keep)]

    print(f"   Kept families with ≥2 samples")

    if malicious_df[Config.FAMILY_TARGET_COL].nunique() < 2:
        raise ValueError("Not enough malware family diversity to train the virus classifier.")

    print(f"\n⏳ Encoding labels...")
    t_encode_start = time.time()
    le = LabelEncoder()
    X = malicious_df[features]
    y = le.fit_transform(malicious_df[Config.FAMILY_TARGET_COL])
    t_encode = time.time() - t_encode_start
    print(f"✓ Encoding complete ({t_encode:.2f}s)")

    # Show family distribution
    print(f"\n📊 Malware family distribution:")
    for idx, family_name in enumerate(le.classes_):
        count = (y == idx).sum()
        print(f"   {family_name}: {count:,} ({count / len(y) * 100:.2f}%)")

    # Perform the split
    print(f"\n⏳ Creating train/test split...")
    t_split_start = time.time()
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE,
            stratify=y
        )
        print(f"✓ Split complete (stratified) ({time.time() - t_split_start:.2f}s)")
    except ValueError:
        print("   ⚠️  Could not stratify due to small class sizes. Using random split.")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=Config.TEST_SIZE,
            random_state=Config.RANDOM_STATE
        )
        print(f"✓ Split complete (random) ({time.time() - t_split_start:.2f}s)")

    total_time = time.time() - overall_start

    print(f"\n" + "=" * 70)
    print(f"✅ VIRUS DATA READY")
    print(f"   Train samples: {len(X_train):,}")
    print(f"   Test samples:  {len(X_test):,}")
    print(f"   Features:      {X.shape[1]}")
    print(f"   Families:      {len(le.classes_)}")
    print(f"⏱️  Total prep time: {total_time:.2f}s")
    print("=" * 70)

    return X_train, X_test, y_train, y_test, le


def get_data_for_clustering():
    """Load and prepare data for clustering (malicious only) with timing."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING DATA FOR CLUSTERING")
    print("=" * 70)

    overall_start = time.time()

    df = load_engineered_data()
    features = Config.get_feature_list()

    print(f"\n⏳ Filtering malicious samples...")
    t_filter_start = time.time()
    X = df[df[Config.TARGET_COL] == 'Malicious'][features]
    t_filter = time.time() - t_filter_start
    print(f"✓ Filtered to {len(X):,} malicious samples ({t_filter:.2f}s)")

    total_time = time.time() - overall_start

    print(f"\n" + "=" * 70)
    print(f"✅ CLUSTERING DATA READY")
    print(f"   Samples:  {len(X):,}")
    print(f"   Features: {X.shape[1]}")
    print(f"⏱️  Total prep time: {total_time:.2f}s")
    print("=" * 70)

    return X
