"""
Configuration for IoT-23 Dataset Analysis
"""
import os

import joblib
import torch


class Config:
    # ==================== MODE SETTINGS ====================
    # Set to True for testing with smaller dataset, False for production
    TEST_MODE = False  # <-- TOGGLE THIS

    # ==================== PATHS ====================
    # Dynamically choose raw directory based on test mode
    RAW_DIR_ORIGINAL = 'data/raw_test/' if TEST_MODE else 'data/raw/'

    ENGINEERED_SPLIT_DIR = 'data/engineered_split_csv/'
    ENGINEERED_DIR = 'data/engineered_features/'
    MODELS_DIR = 'models/'
    RESULTS_DIR = 'results/'

    for dir_path in [RAW_DIR_ORIGINAL, ENGINEERED_SPLIT_DIR, ENGINEERED_DIR, MODELS_DIR, RESULTS_DIR]:
        os.makedirs(dir_path, exist_ok=True)

    ENGINEERED_DATA_PATH = os.path.join(ENGINEERED_DIR, 'final_features.parquet')
    ENGINEERED_DATA_PATH_CSV = os.path.join(ENGINEERED_DIR, 'final_features.csv')
    FEATURE_LIST_PATH = os.path.join(ENGINEERED_DIR, 'final_feature_list.joblib')
    AUTOENCODER_FEATURE_LIST_PATH = os.path.join(ENGINEERED_DIR, 'autoencoder_feature_list.joblib')

    # ==================== GROUND TRUTH MAPPING ====================
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

    # ==================== DATA SETTINGS ====================
    SAMPLE_SIZE = 50000000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # ==================== FEATURES ====================
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

    TARGET_COL = 'label'
    DETAILED_TARGET_COL = 'attack_type'
    FAMILY_TARGET_COL = 'malware_family'

    @staticmethod
    def get_feature_list():
        if os.path.exists(Config.FEATURE_LIST_PATH):
            return joblib.load(Config.FEATURE_LIST_PATH)
        print(f"Warning: Feature list not found. Run build_dataset.py.")
        return []

    # ==================== IMBALANCE & OPTIMIZATION ====================
    USE_SMOTE = True
    SMOTE_SAMPLE_THRESHOLD = 100000
    USE_OPTUNA = True
    OPTUNA_N_TRIALS = 20
    OPTUNA_TIMEOUT = 1800  # 30 min
    RUN_LOGISTIC_REGRESSION = True  # Toggle for the baseline model in binary_classifier.py

    # ==================== MODEL SETTINGS ====================
    if torch.cuda.is_available():
        DEVICE = 'cuda'
    elif torch.backends.mps.is_available():
        DEVICE = 'mps'
    else:
        DEVICE = 'cpu'

    AUTOENCODER_LATENT_DIM = 8
    AUTOENCODER_EPOCHS = 100
    AUTOENCODER_BATCH_SIZE = 32768  # 8x larger
    AUTOENCODER_LR = 0.0001
    ANOMALY_THRESHOLD_PERCENTILE = 85
    BINARY_N_ESTIMATORS = 200
    BINARY_LEARNING_RATE = 0.1
    BINARY_NUM_LEAVES = 50
    MULTICLASS_N_ESTIMATORS = 200
    MULTICLASS_LEARNING_RATE = 0.1
    MULTICLASS_MAX_DEPTH = 10
    KMEANS_N_CLUSTERS = 5
    KMEANS_BATCH_SIZE = 10000
    # NUM_WORKERS = 0 if os.name == 'nt' else 16
    NUM_WORKERS = 6
    PIN_MEMORY = True
    N_JOBS = 32

    # ==================== REPRODUCIBILITY ====================
    @staticmethod
    def set_seeds():
        import random
        import numpy as np
        random.seed(Config.RANDOM_STATE)
        np.random.seed(Config.RANDOM_STATE)
        torch.manual_seed(Config.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.RANDOM_STATE)

    # ==================== DISPLAY MODE INFO ====================
    @staticmethod
    def print_mode_info():
        """Print current configuration mode"""
        print("\n" + "=" * 70)
        print("⚙️  CONFIGURATION MODE")
        print("=" * 70)
        mode = "TEST MODE 🧪" if Config.TEST_MODE else "PRODUCTION MODE 🚀"
        print(f"   Mode: {mode}")
        print(f"   Raw Data Directory: {Config.RAW_DIR_ORIGINAL}")
        print(f"   Device: {Config.DEVICE}")
        print("=" * 70)