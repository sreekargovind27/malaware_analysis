"""
This file centralizes all configuration settings for the IoT-23 dataset analysis,
including file paths, model parameters, and feature definitions.
"""
import os

import joblib
import torch


class Config:
    # Set to True to use a smaller dataset for testing, False for the full dataset.
    TEST_MODE = True

    # Paths are dynamically set based on the selected mode.
    RAW_DIR_ORIGINAL = 'data/raw_test/' if TEST_MODE else 'data/raw/'
    ENGINEERED_SPLIT_DIR = 'data/engineered_split_csv/'
    ENGINEERED_DIR = 'data/engineered_features/'
    MODELS_DIR = 'models/'
    RESULTS_DIR = 'results/'

    # Create directories if they do not exist.
    for dir_path in [RAW_DIR_ORIGINAL, ENGINEERED_SPLIT_DIR, ENGINEERED_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    ENGINEERED_DATA_PATH = os.path.join(ENGINEERED_DIR, 'final_features.parquet')
    ENGINEERED_DATA_PATH_CSV = os.path.join(ENGINEERED_DIR, 'final_features.csv')
    FEATURE_LIST_PATH = os.path.join(ENGINEERED_DIR, 'final_feature_list.joblib')
    AUTOENCODER_FEATURE_LIST_PATH = os.path.join(ENGINEERED_DIR, 'autoencoder_feature_list.joblib')

    # Maps filenames to their ground-truth malware family.
    FILENAME_TO_FAMILY_MAP = {
        'CTU-IoT-Malware-Capture-1-1': 'Hide and Seek', 'CTU-IoT-Malware-Capture-3-1': 'Muhstik',
        'CTU-IoT-Malware-Capture-7-1': 'Mirai', 'CTU-IoT-Malware-Capture-8-1': 'Hakai',
        'CTU-IoT-Malware-Capture-9-1': 'Hajime', 'CTU-IoT-Malware-Capture-17-1': 'Kenjiro',
        'CTU-IoT-Malware-Capture-20-1': 'Torii', 'CTU-IoT-Malware-Capture-21-1': 'Torii',
        'CTU-IoT-Malware-Capture-33-1': 'Kenjiro', 'CTU-IoT-Malware-Capture-34-1': 'Mirai',
        'CTU-IoT-Malware-Capture-35-1': 'Mirai', 'CTU-IoT-Malware-Capture-36-1': 'Okiru',
        'CTU-IoT-Malware-Capture-39-1': 'IRCBot', 'CTU-IoT-Malware-Capture-42-1': 'Trojan',
        'CTU-IoT-Malware-Capture-43-1': 'Mirai', 'CTU-IoT-Malware-Capture-44-1': 'Mirai',
        'CTU-IoT-Malware-Capture-48-1': 'Mirai', 'CTU-IoT-Malware-Capture-49-1': 'Mirai',
        'CTU-IoT-Malware-Capture-52-1': 'Mirai', 'CTU-IoT-Malware-Capture-60-1': 'Gagfyt',
        'CTU-Honeypot-Capture-4-1': 'Benign', 'CTU-Honeypot-Capture-5-1': 'Benign',
    }

    # Data processing and sampling settings.
    SAMPLE_SIZE = 50000000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Feature definitions used during data engineering.
    BASE_NUMERICAL_FEATURES = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'orig_ip_bytes',
                               'resp_ip_bytes', 'missed_bytes', 'id.resp_p']
    SKEWED_NUMERICAL_FEATURES = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'orig_ip_bytes',
                                 'resp_ip_bytes']
    CATEGORICAL_LABEL_ENCODE = ['service', 'history']
    CATEGORICAL_ONE_HOT_ENCODE = ['proto', 'conn_state']
    IP_FEATURES_BASE = ['is_private', 'is_broadcast', 'is_multicast', 'ip_first_octet', 'is_localhost']
    ENGINEERED_FEATURES = [
        'is_port_23', 'is_port_22', 'is_S0_state', 'is_telnet',
        'is_unknown_service', 'upload_ratio', 'bytes_per_packet',
        'packet_rate', 'is_scanning_signature', 'suspicious_score'
    ]

    # Target column names.
    TARGET_COL = 'label'
    DETAILED_TARGET_COL = 'attack_type'
    FAMILY_TARGET_COL = 'malware_family'

    @staticmethod
    def get_feature_list():
        if os.path.exists(Config.FEATURE_LIST_PATH):
            return joblib.load(Config.FEATURE_LIST_PATH)
        print(f"Warning: Feature list not found. Run build_dataset.py.")
        return []

    # Settings for handling class imbalance and hyperparameter optimization.
    USE_SMOTE = True
    SMOTE_SAMPLE_THRESHOLD = 100000
    USE_OPTUNA = True
    OPTUNA_N_TRIALS = 20
    OPTUNA_TIMEOUT = 300
    RUN_LOGISTIC_REGRESSION = True  # Enables the baseline model in binary_classifier.py.

    # Hardware and model-specific hyperparameters.
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

    AUTOENCODER_LATENT_DIM = 8
    AUTOENCODER_EPOCHS = 100
    AUTOENCODER_BATCH_SIZE = 32768
    AUTOENCODER_LR = 0.0001
    ANOMALY_THRESHOLD_PERCENTILE = 85
    BINARY_N_ESTIMATORS = 200
    BINARY_LEARNING_RATE = 0.1
    BINARY_NUM_LEAVES = 50
    MULTICLASS_N_ESTIMATORS = 200
    MULTICLASS_LEARNING_RATE = 0.1
    XGB_TREE_METHOD = "hist"
    MULTICLASS_MAX_DEPTH = 10
    KMEANS_N_CLUSTERS = 5
    KMEANS_BATCH_SIZE = 10000
    NUM_WORKERS = 6
    PIN_MEMORY = True
    N_JOBS = 32

    @staticmethod
    def set_seeds():
        """Sets random seeds for reproducibility across libraries."""
        import random
        import numpy as np
        random.seed(Config.RANDOM_STATE)
        np.random.seed(Config.RANDOM_STATE)
        torch.manual_seed(Config.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.RANDOM_STATE)

    @staticmethod
    def print_mode_info():
        """Prints the current configuration mode to the console."""
        print("\n" + "=" * 70)
        print("⚙️  CONFIGURATION MODE")
        print("=" * 70)
        mode = "TEST MODE 🧪" if Config.TEST_MODE else "PRODUCTION MODE 🚀"
        print(f"   Mode: {mode}")
        print(f"   Raw Data Directory: {Config.RAW_DIR_ORIGINAL}")
        print(f"   Device: {Config.DEVICE}")
        print("=" * 70)
