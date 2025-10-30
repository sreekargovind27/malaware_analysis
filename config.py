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
    PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

    # Data directories (keep separate - raw data is large)
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data')

    # âœ… THE NEW, CLEAN DATA SOURCE FOR THE ENTIRE PIPELINE
    # Raw data directories
    # RAW_DIR_MESSY: Original downloaded data with potential inconsistencies
    # RAW_DIR_ORIGINAL: Standardized, clean data ready for processing
    RAW_DIR_MESSY = os.path.join(DATA_DIR, 'raw_messy_test' if TEST_MODE else 'raw_messy')
    RAW_DIR_ORIGINAL = os.path.join(DATA_DIR, 'raw_test' if TEST_MODE else 'raw')

    # ===== OUTPUTS DIRECTORY (NEW) =====
    OUTPUTS_DIR = os.path.join(PROJECT_ROOT, 'outputs')

    # Stage 1 - Feasibility Analysis
    STAGE1_FEASIBILITY_DIR = os.path.join(OUTPUTS_DIR, 'stage1_feasibility')

    # Stage 2 - Prepared Data
    STAGE2_PREPARED_DIR = os.path.join(OUTPUTS_DIR, 'stage2_prepared')
    STAGE2_QUALITY_DIR = os.path.join(OUTPUTS_DIR, 'stage2_quality')
    ENGINEERED_DIR = STAGE2_PREPARED_DIR
    ENGINEERED_SPLIT_DIR = os.path.join(STAGE2_PREPARED_DIR, 'splits')

    # Splits subdirectory
    SPLITS_DIR = os.path.join(STAGE2_PREPARED_DIR, 'splits')

    # Normalized data subdirectory
    NORMALIZED_DIR = os.path.join(STAGE2_PREPARED_DIR, 'normalized')

    # Graph data subdirectory (for GNN)
    GRAPH_DIR = os.path.join(STAGE2_PREPARED_DIR, 'graph')

    # Trained Models
    MODELS_DIR = os.path.join(OUTPUTS_DIR, 'models_trained')
    TRADITIONAL_MODELS_DIR = os.path.join(MODELS_DIR, 'traditional')
    DL_MODELS_DIR = os.path.join(MODELS_DIR, 'deep_learning')
    GNN_MODELS_DIR = os.path.join(MODELS_DIR, 'gnn')
    UNSUPERVISED_MODELS_DIR = os.path.join(MODELS_DIR, 'unsupervised')

    # Results
    RESULTS_DIR = os.path.join(OUTPUTS_DIR, 'results')
    BINARY_RESULTS_DIR = os.path.join(RESULTS_DIR, 'binary_classification')
    MULTICLASS_RESULTS_DIR = os.path.join(RESULTS_DIR, 'multiclass')
    FAMILY_RESULTS_DIR = os.path.join(RESULTS_DIR, 'malware_family')
    AUTOENCODER_RESULTS_DIR = os.path.join(RESULTS_DIR, 'autoencoder')
    CLUSTERING_RESULTS_DIR = os.path.join(RESULTS_DIR, 'clustering')

    # Logs
    LOGS_DIR = os.path.join(OUTPUTS_DIR, 'logs')

    # Feature files
    ENGINEERED_DATA_PATH = os.path.join(STAGE2_PREPARED_DIR, 'flow_features.parquet')
    ENGINEERED_DATA_PATH_CSV = os.path.join(STAGE2_PREPARED_DIR, 'flow_features.csv')
    DEVICE_FEATURES_PATH = os.path.join(STAGE2_PREPARED_DIR, 'device_features.parquet')
    FEATURE_LIST_PATH = os.path.join(STAGE2_PREPARED_DIR, 'feature_list.joblib')
    AUTOENCODER_FEATURE_LIST_PATH = os.path.join(STAGE2_PREPARED_DIR, 'autoencoder_feature_list.joblib')

    # Split files
    TRAIN_SET_PATH = os.path.join(SPLITS_DIR, 'train_flows.parquet')
    VAL_SET_PATH = os.path.join(SPLITS_DIR, 'val_flows.parquet')
    TEST_SET_PATH = os.path.join(SPLITS_DIR, 'test_flows.parquet')

    # Scaler
    SCALER_PATH = os.path.join(NORMALIZED_DIR, 'flow_scaler.pkl')

    # Graph files (for GNN)
    HETERO_GRAPH_PATH = os.path.join(GRAPH_DIR, 'hetero_graph.pt')
    DEVICE_NODES_PATH = os.path.join(GRAPH_DIR, 'device_nodes.parquet')
    SERVICE_NODES_PATH = os.path.join(GRAPH_DIR, 'service_nodes.parquet')
    SUBNET_NODES_PATH = os.path.join(GRAPH_DIR, 'subnet_nodes.parquet')
    EDGES_PATH = os.path.join(GRAPH_DIR, 'edges.parquet')
    GRAPH_STATS_PATH = os.path.join(GRAPH_DIR, 'graph_stats.json')
    GAN_RESULTS_DIR = os.path.join(RESULTS_DIR, 'gan')

    # Malware family mapping (filename -> family name)
    FILENAME_TO_FAMILY_MAP = {
        'CTU-IoT-Malware-Capture-1-1': 'Mirai',
        'CTU-IoT-Malware-Capture-7-1': 'Mirai',
        'CTU-IoT-Malware-Capture-8-1': 'Mirai',
        'CTU-IoT-Malware-Capture-9-1': 'Mirai',
        'CTU-IoT-Malware-Capture-20-1': 'Mirai',
        'CTU-IoT-Malware-Capture-21-1': 'Mirai',
        'CTU-IoT-Malware-Capture-33-1': 'Mirai',
        'CTU-IoT-Malware-Capture-34-1': 'Mirai',
        'CTU-IoT-Malware-Capture-35-1': 'Mirai',
        'CTU-IoT-Malware-Capture-36-1': 'Mirai',
        'CTU-IoT-Malware-Capture-42-1': 'Mirai',
        'CTU-IoT-Malware-Capture-43-1': 'Mirai',
        'CTU-IoT-Malware-Capture-44-1': 'Mirai',
        'CTU-IoT-Malware-Capture-48-1': 'Mirai',
        'CTU-IoT-Malware-Capture-49-1': 'Mirai',
        'CTU-IoT-Malware-Capture-52-1': 'Mirai',
        'CTU-IoT-Malware-Capture-60-1': 'Mirai',

        'CTU-IoT-Malware-Capture-3-1': 'Kenjiro',
        'CTU-IoT-Malware-Capture-39-1': 'Kenjiro',

        'CTU-IoT-Malware-Capture-5-1': 'Torii',

        'CTU-IoT-Malware-Capture-17-1': 'Gagfyt',
        'CTU-IoT-Malware-Capture-41-1': 'Gagfyt',
        'CTU-IoT-Malware-Capture-51-1': 'Gagfyt',

        'CTU-IoT-Malware-Capture-37-1': 'Okiru',

        'CTU-IoT-Malware-Capture-46-1': 'Muhstik',

        'CTU-IoT-Malware-Capture-40-1': 'Hajime',

        'CTU-IoT-Malware-Capture-53-1': 'Hide and Seek',

        'CTU-IoT-Malware-Capture-54-1': 'Hakai',

        'CTU-IoT-Malware-Capture-55-1': 'IRCBot',

        'CTU-IoT-Malware-Capture-56-1': 'Trojan',

        # Benign captures
        'CTU-Honeypot-Capture-4-1': 'Benign',
        'CTU-Honeypot-Capture-5-1': 'Benign',
        'CTU-Honeypot-Capture-7-1': 'Benign',
    }

    # Data processing and sampling settings.
    SAMPLE_SIZE = 50000000
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    # Feature definitions used during data engineering.
    BASE_NUMERICAL_FEATURES = ['duration', 'orig_bytes', 'resp_bytes', 'orig_pkts', 'resp_pkts', 'orig_ip_bytes',
                               'resp_ip_bytes', 'missed_bytes', 'id_resp_p']
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

    NOISE_START = 0.05  # Starting noise factor
    NOISE_WARMUP_EPOCHS = 40  # Epochs to ramp from start to final noise

    # WGAN-GP semi-supervised settings
    GAN_USE_MALICIOUS = True
    GAN_MALICIOUS_RATIO = 0.1
    GAN_MALICIOUS_WEIGHT = 0.2
    GAN_TEST_INJECTION_RATES = True  # Set True to run grid search

    AUTOENCODER_LATENT_DIM = 8
    AUTOENCODER_EPOCHS = 100
    AUTOENCODER_BATCH_SIZE = 4096  # instead of 32768
    AUTOENCODER_LR = 1e-3  # calm LR, Optuna will tune within [1e-4, 5e-3]
    USE_LATENT_SPARSITY = True
    SPARSITY_WARMUP_EPOCHS = 30
    LATENT_SPARSITY_LAMBDA = 1e-3
    BINARY_N_ESTIMATORS = 200
    BINARY_LEARNING_RATE = 0.1
    BINARY_NUM_LEAVES = 50
    MULTICLASS_N_ESTIMATORS = 200
    MULTICLASS_LEARNING_RATE = 0.1
    XGB_TREE_METHOD = "hist"
    MULTICLASS_MAX_DEPTH = 10
    KMEANS_N_CLUSTERS = 5
    KMEANS_BATCH_SIZE = 10000
    NUM_WORKERS = 2
    PIN_MEMORY = True
    N_JOBS = 32

    @staticmethod
    def ensure_output_dirs():
        """Create output directories if they don't exist. Call this at the start of each script."""
        for dir_path in [
            Config.RAW_DIR_MESSY,
            Config.RAW_DIR_ORIGINAL,
            Config.STAGE1_FEASIBILITY_DIR,
            Config.STAGE2_PREPARED_DIR,
            Config.STAGE2_QUALITY_DIR,
            Config.SPLITS_DIR,
            Config.NORMALIZED_DIR,
            Config.GRAPH_DIR,
            Config.TRADITIONAL_MODELS_DIR,
            Config.DL_MODELS_DIR,
            Config.GNN_MODELS_DIR,
            Config.UNSUPERVISED_MODELS_DIR,
            Config.BINARY_RESULTS_DIR,
            Config.MULTICLASS_RESULTS_DIR,
            Config.FAMILY_RESULTS_DIR,
            Config.AUTOENCODER_RESULTS_DIR,
            Config.CLUSTERING_RESULTS_DIR,
            Config.LOGS_DIR
        ]:
            os.makedirs(dir_path, exist_ok=True)

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
    def get_spark_session(app_name="IoT23-Analysis"):
        """
        Get or create Spark session for analysis.
        Works both locally and on Databricks.
        """
        try:
            # Try to get existing session (works on Databricks)
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            return spark
        except:
            # Create new session (works locally)
            from pyspark.sql import SparkSession
            spark = SparkSession.builder \
                .appName(app_name) \
                .config("spark.driver.memory", "8g") \
                .config("spark.sql.shuffle.partitions", "200") \
                .getOrCreate()
            return spark

    @staticmethod
    def print_mode_info():
        """Prints the current configuration mode to the console."""
        print("\n" + "=" * 70)
        print("Ã¢Å¡â„¢Ã¯Â¸Â  CONFIGURATION MODE")
        print("=" * 70)
        mode = "TEST MODE Ã°Å¸Â§Âª" if Config.TEST_MODE else "PRODUCTION MODE Ã°Å¸Å¡â‚¬"
        print(f"   Mode: {mode}")
        print(f"   Raw Data Directory: {Config.RAW_DIR_ORIGINAL}")
        print(f"   Device: {Config.DEVICE}")
        print("=" * 70)
