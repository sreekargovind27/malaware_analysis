"""
Provides data loading and preparation functions for model training.
Classification loaders read from pre-split master datasets to ensure reproducibility,
while unsupervised loaders use the full dataset.
"""
import os

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader

import pickle
import joblib
import xgboost as xgb

from config import Config


class IoTDataset(Dataset):
    """A PyTorch Dataset for the IoT traffic data."""
    def __init__(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X = X.values
        if isinstance(y, pd.Series):
            y = y.values

        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y) if y is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


def load_engineered_data():
    """Loads the main engineered dataset."""
    path = Config.ENGINEERED_DATA_PATH
    print(f"\n📂 Loading FULL engineered data from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Engineered data not found at '{path}'. Please run 'build_dataset.py' first.")
    df = pd.read_parquet(path)
    print(f"✓ Loaded {len(df):,} total samples.")
    return df


def load_master_train_set():
    """Loads the pre-split training dataset."""
    path = 'data/master_splits/train_set.parquet'
    print(f"\n📂 Loading MASTER TRAIN set from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Master train set not found at '{path}'. Please run 'create_splits.py' first.")
    df = pd.read_parquet(path)
    print(f"✓ Loaded {len(df):,} training samples.")
    return df


def load_master_test_set():
    """Loads the pre-split test dataset."""
    path = 'data/master_splits/test_set.parquet'
    print(f"\n📂 Loading MASTER TEST set from: {path}")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Master test set not found at '{path}'. Please run 'create_splits.py' first.")
    df = pd.read_parquet(path)
    print(f"✓ Loaded {len(df):,} test samples.")
    return df


def get_data_for_autoencoder():
    """Loads and prepares data for the Autoencoder, using only benign samples from the full dataset."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING DATA FOR AUTOENCODER (from full dataset)")
    print("=" * 70)
    df = load_engineered_data()
    features = Config.get_feature_list()
    print(f"\n⏳ Filtering benign data...")
    benign_df = df.loc[df[Config.TARGET_COL] == 'Benign', features]
    print(f"✓ Filtered to {len(benign_df):,} benign samples")
    print(f"\n⏳ Removing constant columns...")
    non_constant_cols = benign_df.columns[benign_df.std() > 1e-6].tolist()
    joblib.dump(non_constant_cols, Config.AUTOENCODER_FEATURE_LIST_PATH)
    X = benign_df[non_constant_cols].fillna(0)
    print(f"\n⏳ Scaling features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X.values)
    X_scaled = np.nan_to_num(X_scaled)
    print(f"\n⏳ Creating train/validation split...")
    X_train, X_val = train_test_split(X_scaled, test_size=0.2, random_state=Config.RANDOM_STATE)
    print(f"\n⏳ Creating PyTorch DataLoaders...")
    train_loader = DataLoader(IoTDataset(X_train), batch_size=Config.AUTOENCODER_BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(IoTDataset(X_val), batch_size=Config.AUTOENCODER_BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)
    print(f"✓ Autoencoder data ready.")
    return train_loader, val_loader, scaler, non_constant_cols


def get_data_for_clustering():
    """Loads and prepares data for clustering, using only malicious samples from the full dataset."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING DATA FOR CLUSTERING (from full dataset)")
    print("=" * 70)
    df = load_engineered_data()
    features = Config.get_feature_list()
    X = df[df[Config.TARGET_COL] == 'Malicious'][features]
    print(f"✓ Filtered to {len(X):,} malicious samples.")
    return X


def get_data_for_binary():
    """Prepares data for binary classification from the master train/test splits."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING BINARY CLASSIFICATION DATA")
    print("=" * 70)
    train_df = load_master_train_set()
    test_df = load_master_test_set()
    features = Config.get_feature_list()
    X_train = train_df[features]
    y_train = (train_df[Config.TARGET_COL] == 'Malicious').astype(int).values
    X_test = test_df[features]
    y_test = (test_df[Config.TARGET_COL] == 'Malicious').astype(int).values
    print(f"✓ Binary data ready.\n  - Train samples: {len(X_train):,}\n  - Test samples:  {len(X_test):,}")
    print("=" * 70)
    return X_train, X_test, y_train, y_test


def get_data_for_multiclass():
    """Prepares data for multi-class classification, including class-aware undersampling."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING MULTI-CLASS CLASSIFICATION DATA")
    print("=" * 70)
    train_df = load_master_train_set()
    test_df = load_master_test_set()
    features = Config.get_feature_list()
    print("\n🎯 Performing class-aware undersampling on the TRAINING set...")

    NEW_TARGET_TRAIN_SIZE = int(len(train_df) * 0.30)
    MINORITY_CLASS_THRESHOLD = 100000
    print(f"   Original train size: {len(train_df):,} rows")
    print(f"   New target train size: {NEW_TARGET_TRAIN_SIZE:,} rows")

    value_counts = train_df[Config.DETAILED_TARGET_COL].value_counts()
    small_classes = value_counts[value_counts < MINORITY_CLASS_THRESHOLD].index.tolist()
    large_classes = value_counts[value_counts >= MINORITY_CLASS_THRESHOLD].index.tolist()
    df_minority = train_df[train_df[Config.DETAILED_TARGET_COL].isin(small_classes)]
    df_majority = train_df[train_df[Config.DETAILED_TARGET_COL].isin(large_classes)]
    rows_to_sample_from_majority = NEW_TARGET_TRAIN_SIZE - len(df_minority)

    if rows_to_sample_from_majority > 0 and len(df_majority) > rows_to_sample_from_majority:
        df_majority_sampled = df_majority.sample(n=rows_to_sample_from_majority, random_state=Config.RANDOM_STATE)
        df_train_final = pd.concat([df_minority, df_majority_sampled], ignore_index=True)
    else:
        df_train_final = train_df.sample(n=NEW_TARGET_TRAIN_SIZE, random_state=Config.RANDOM_STATE)

    print(f"✓ Subsampling complete. Intermediate training set size: {len(df_train_final):,}")

    # Filters out classes with only one member after subsampling to prevent errors.
    print("   Final check: Removing any classes with only 1 member after sampling...")
    class_counts = df_train_final[Config.DETAILED_TARGET_COL].value_counts()
    to_keep = class_counts[class_counts >= 2].index
    df_train_final = df_train_final[df_train_final[Config.DETAILED_TARGET_COL].isin(to_keep)]
    print(f"✓ Final training set size: {len(df_train_final):,}")

    le = LabelEncoder()
    X_train = df_train_final[features]
    y_train = le.fit_transform(df_train_final[Config.DETAILED_TARGET_COL])

    # Filter the test set to only include classes present in the final training set.
    test_classes_seen_in_train = [cls for cls in test_df[Config.DETAILED_TARGET_COL].unique() if cls in le.classes_]
    test_df_filtered = test_df[test_df[Config.DETAILED_TARGET_COL].isin(test_classes_seen_in_train)]
    X_test = test_df_filtered[features]
    y_test = le.transform(test_df_filtered[Config.DETAILED_TARGET_COL])

    joblib.dump(le, 'data/master_splits/multiclass_label_encoder.joblib')
    print("✓ Label encoder saved.")
    print(f"\n✓ Multiclass data ready.\n  - Train samples: {len(X_train):,}\n  - Test samples:  {len(X_test):,}")
    print("=" * 70)
    return X_train, X_test, y_train, y_test, le


def get_data_for_virus():
    """Prepares data for malware family classification from the master splits."""
    print("\n" + "=" * 70)
    print("🔧 PREPARING VIRUS CLASSIFICATION DATA")
    print("=" * 70)
    train_df = load_master_train_set()
    test_df = load_master_test_set()
    features = Config.get_feature_list()
    train_malicious = train_df[train_df[Config.TARGET_COL] == 'Malicious'].copy()
    test_malicious = test_df[test_df[Config.TARGET_COL] == 'Malicious'].copy()

    # Filters out families with only one member to prevent training errors.
    print("   Final check: Removing any families with only 1 member...")
    family_counts = train_malicious[Config.FAMILY_TARGET_COL].value_counts()
    to_keep = family_counts[family_counts >= 2].index
    train_malicious = train_malicious[train_malicious[Config.FAMILY_TARGET_COL].isin(to_keep)]
    print(f"✓ Final training set size: {len(train_malicious):,}")

    le = LabelEncoder()
    X_train = train_malicious[features]
    y_train = le.fit_transform(train_malicious[Config.FAMILY_TARGET_COL])
    test_classes_seen_in_train = [cls for cls in test_malicious[Config.FAMILY_TARGET_COL].unique() if
                                  cls in le.classes_]
    test_malicious_filtered = test_malicious[test_malicious[Config.FAMILY_TARGET_COL].isin(test_classes_seen_in_train)]
    X_test = test_malicious_filtered[features]
    y_test = le.transform(test_malicious_filtered[Config.FAMILY_TARGET_COL])

    joblib.dump(le, 'data/master_splits/virus_label_encoder.joblib')
    print("✓ Label encoder saved.")
    print(f"\n✓ Virus data ready.\n  - Train samples: {len(X_train):,}\n  - Test samples:  {len(X_test):,}")
    print("=" * 70)
    return X_train, X_test, y_train, y_test, le
