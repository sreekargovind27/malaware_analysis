"""
Central configuration for IoT-23 pipeline (data prep, modeling, outputs).
Unified config that works both locally and on Databricks.

Environment Detection:
- Local: Uses local filesystem paths, local[*] Spark
- Databricks: Uses Unity Catalog Volumes paths, cluster Spark with optimized settings
"""

import os
import joblib
import torch


class Config:
    # ==================================================================
    # DATABRICKS CONFIGURATION - ALL SETTINGS IN ONE PLACE
    # ==================================================================

    # Environment-specific Spark settings
    DATABRICKS_SHUFFLE_PARTITIONS = 200  # For 40-50GB data (increase if needed)
    DATABRICKS_REPARTITION_SIZE = 200  # Default repartition size for groupBy operations
    DATABRICKS_COALESCE_PARTITIONS = 100  # For final parquet writes
    DATABRICKS_ENABLE_ADAPTIVE = True  # Adaptive query execution (AQE)
    DATABRICKS_ENABLE_AQE_SKEW = True  # Handle data skew automatically
    DATABRICKS_ENABLE_AQE_COALESCE = True  # Coalesce partitions automatically

    LOCAL_SHUFFLE_PARTITIONS = 8  # For local testing (small data)
    LOCAL_REPARTITION_SIZE = 8  # For local testing
    LOCAL_DRIVER_MEMORY = "8g"  # Local driver memory
    LOCAL_EXECUTOR_MEMORY = "8g"  # Local executor memory

    # Path configuration
    DATABRICKS_BASE_PATH = "/Volumes/workspace/malaware_analysis/iot23_data"  # Unity Catalog Volume path
    
    # Graph building limits (prevent OOM on collect)
    MAX_NODES_TO_COLLECT = 100000  # Max nodes before refusing collect()

    # File write settings
    MAX_RECORDS_PER_FILE = 1000000  # Control output file size
    COMPRESSION_CODEC = "snappy"  # Parquet compression

    # Storage level for persist operations
    PERSIST_STORAGE_LEVEL = "MEMORY_AND_DISK"  # Options: MEMORY_ONLY, MEMORY_AND_DISK, DISK_ONLY

    # ==================================================================
    # ENVIRONMENT DETECTION
    # ==================================================================

    @staticmethod
    def is_databricks():
        """Detect if running on Databricks"""
        return "DATABRICKS_RUNTIME_VERSION" in os.environ

    @staticmethod
    def get_environment():
        """Get current environment name"""
        return "Databricks" if Config.is_databricks() else "Local"

    # ==================================================================
    # MODE / SAMPLING
    # ==================================================================

    TEST_MODE = False  # Set to True for small test data, False for full 40GB run

    # For Spark feature_engineering: take full data or downsample
    # 1.0 = use 100% of rows; 0.1 = 10%; etc.
    DATA_SAMPLE_FRACTION = 1.0

    # Stratified sampling
    USE_STRATIFIED_SAMPLING = True

    # Random seed for splits, sampling, model init, etc.
    RANDOM_STATE = 42

    # ==================================================================
    # PROJECT ROOTS / IO PATHS (Environment-Aware)
    # ==================================================================

    # Determine base path - use Unity Catalog Volume on Databricks, local path otherwise
    BASE_PATH = "/Volumes/workspace/malaware_analysis/iot23_data" if os.path.exists('/dbfs') else os.path.abspath(os.path.join(os.path.dirname(__file__), "."))

    @staticmethod
    def get_base_path():
        """Get base path (already computed)"""
        return Config.BASE_PATH

    # Data directories
    DATA_DIR = os.path.join(BASE_PATH, "data")

    RAW_DIR_MESSY = os.path.join(DATA_DIR, "raw_messy_test" if TEST_MODE else "raw_messy")
    
    # On Databricks with Unity Catalog Volumes, files are directly in the volume
    # On local, they're in a data/raw subdirectory
    RAW_DIR_ORIGINAL = "/Volumes/workspace/malaware_analysis/iot23_data" if os.path.exists('/dbfs') else os.path.join(DATA_DIR, "raw_test" if TEST_MODE else "raw")

    # Outputs
    OUTPUTS_DIR = os.path.join(BASE_PATH, "outputs")

    # Stage 1 (exploration / sanity reports)
    STAGE1_FEASIBILITY_DIR = os.path.join(OUTPUTS_DIR, "stage1_feasibility")

    # Stage 2 (engineered features, splits, graphs, quality checks)
    STAGE2_PREPARED_DIR = os.path.join(OUTPUTS_DIR, "stage2_prepared")
    STAGE2_QUALITY_DIR = os.path.join(OUTPUTS_DIR, "stage2_quality")

    ENGINEERED_DIR = STAGE2_PREPARED_DIR
    SPLITS_DIR = os.path.join(STAGE2_PREPARED_DIR, "splits")
    NORMALIZED_DIR = os.path.join(STAGE2_PREPARED_DIR, "normalized")
    GRAPH_DIR = os.path.join(STAGE2_PREPARED_DIR, "graph")

    # Models (Stage 3+ - for future use, not on Databricks)
    MODELS_DIR = os.path.join(OUTPUTS_DIR, "models_trained")
    TRADITIONAL_MODELS_DIR = os.path.join(MODELS_DIR, "traditional")
    DL_MODELS_DIR = os.path.join(MODELS_DIR, "deep_learning")
    GNN_MODELS_DIR = os.path.join(MODELS_DIR, "gnn")
    UNSUPERVISED_MODELS_DIR = os.path.join(MODELS_DIR, "unsupervised")

    # Results / reports
    RESULTS_DIR = os.path.join(OUTPUTS_DIR, "results")
    BINARY_RESULTS_DIR = os.path.join(RESULTS_DIR, "binary_classification")
    MULTICLASS_RESULTS_DIR = os.path.join(RESULTS_DIR, "multiclass_classification")
    FAMILY_RESULTS_DIR = os.path.join(RESULTS_DIR, "malware_family")
    AUTOENCODER_RESULTS_DIR = os.path.join(RESULTS_DIR, "autoencoder")
    CLUSTERING_RESULTS_DIR = os.path.join(RESULTS_DIR, "clustering")
    GAN_RESULTS_DIR = os.path.join(RESULTS_DIR, "gan")

    # Logs
    LOGS_DIR = os.path.join(OUTPUTS_DIR, "logs")

    # ==================================================================
    # SPECIFIC FILE PATHS
    # ==================================================================

    # Stage 2 outputs
    ENGINEERED_DATA_PATH = os.path.join(STAGE2_PREPARED_DIR, "engineered_flows.parquet")
    DEVICE_FEATURES_PATH = os.path.join(STAGE2_PREPARED_DIR, "device_features.parquet")
    FEATURE_LIST_PATH = os.path.join(STAGE2_PREPARED_DIR, "feature_list.json")


    # Train/Val/Test splits
    TRAIN_PATH = os.path.join(SPLITS_DIR, "train.parquet")
    VAL_PATH = os.path.join(SPLITS_DIR, "val.parquet")
    TEST_PATH = os.path.join(SPLITS_DIR, "test.parquet")

    # Normalized versions (for deep learning)
    TRAIN_NORMALIZED_PATH = os.path.join(NORMALIZED_DIR, "train_normalized.parquet")
    VAL_NORMALIZED_PATH = os.path.join(NORMALIZED_DIR, "val_normalized.parquet")
    TEST_NORMALIZED_PATH = os.path.join(NORMALIZED_DIR, "test_normalized.parquet")
    SCALER_PATH = os.path.join(NORMALIZED_DIR, "scaler.joblib")

    # Graph outputs
    HETERO_GRAPH_PATH = os.path.join(GRAPH_DIR, "hetero_graph.pt")
    GRAPH_STATS_PATH = os.path.join(GRAPH_DIR, "graph_stats.json")

    # Backward compatibility aliases
    TRAIN_SET_PATH = TRAIN_PATH
    VAL_SET_PATH = VAL_PATH
    TEST_SET_PATH = TEST_PATH

    # Stage 1 feasibility report
    FEASIBILITY_SUMMARY = os.path.join(STAGE1_FEASIBILITY_DIR, "feasibility_summary.json")

    # ==================================================================
    # FEATURE DEFINITIONS
    # ==================================================================

    TARGET_COL = "label"  # Binary: Benign / Malicious
    DETAILED_TARGET_COL = "attack_type"  # Multi-class attack type
    FAMILY_TARGET_COL = "malware_family"  # Malware family

    # Core numeric features (time/duration)
    NUMERIC_COLS = [
        "duration", "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
    ]

    # Port features
    PORT_COLS = ["id.orig_p", "id.resp_p"]

    # Categorical features
    CATEGORICAL_COLS = ["proto", "service", "conn_state", "history"]

    @staticmethod
    def get_feature_list():
        """
        Load the feature list from joblib or return a default list.
        This is the final feature set for modeling.
        """
        if os.path.exists(Config.FEATURE_LIST_PATH):
            return joblib.load(Config.FEATURE_LIST_PATH)
        return []

    # ==================================================================
    # SPARK SESSION (Unified for Local + Databricks)
    # ==================================================================

    @staticmethod
    def get_spark_session(app_name="IoT23-DataPipeline"):
        """
        Build or get a SparkSession that works both locally and on Databricks.
        Auto-detects environment and configures appropriately.

        Args:
            app_name: Name for the Spark application

        Returns:
            SparkSession configured for the current environment
        """
        from pyspark.sql import SparkSession

        builder = SparkSession.builder.appName(app_name)

        if Config.is_databricks():
            # ===== DATABRICKS CLUSTER =====
            print("🔧 Detected Databricks environment")
            print(f"   Databricks Runtime: {os.environ.get('DATABRICKS_RUNTIME_VERSION', 'Unknown')}")

            # Databricks manages master, driver memory, executor memory
            # We only configure optimization settings
            builder = (builder
                       .config("spark.sql.shuffle.partitions", str(Config.DATABRICKS_SHUFFLE_PARTITIONS))
                       .config("spark.sql.adaptive.enabled", str(Config.DATABRICKS_ENABLE_ADAPTIVE).lower())
                       .config("spark.sql.adaptive.coalescePartitions.enabled",
                               str(Config.DATABRICKS_ENABLE_AQE_COALESCE).lower())
                       .config("spark.sql.adaptive.skewJoin.enabled", str(Config.DATABRICKS_ENABLE_AQE_SKEW).lower())
                       .config("spark.sql.adaptive.skewJoin.skewedPartitionFactor", "5")
                       .config("spark.sql.adaptive.skewJoin.skewedPartitionThresholdInBytes", "256MB")
                       )

        else:
            # ===== LOCAL DEVELOPMENT =====
            print("🔧 Detected local environment")

            builder = (builder
                       .config("spark.master", "local[*]")
                       .config("spark.driver.memory", Config.LOCAL_DRIVER_MEMORY)
                       .config("spark.executor.memory", Config.LOCAL_EXECUTOR_MEMORY)
                       .config("spark.sql.shuffle.partitions", str(Config.LOCAL_SHUFFLE_PARTITIONS))
                       .config("spark.default.parallelism", str(Config.LOCAL_SHUFFLE_PARTITIONS))
                       )

        # ===== COMMON CONFIGS (Both Environments) =====
        spark = (builder
                 .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
                 .config("spark.sql.execution.arrow.pyspark.enabled", "true")
                 .config("spark.ui.showConsoleProgress", "true")
                 .getOrCreate()
                 )

        # Set log level (not supported on serverless)
        if not Config.is_databricks():
            spark.sparkContext.setLogLevel("WARN")


        print(f"✅ Spark session ready: {spark.version}")
        print(f"   Environment: {Config.get_environment()}")
        print(f"   Base path: {Config.BASE_PATH}")
        print(f"   Shuffle partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")

        return spark

    # ==================================================================
    # TRAINING / MODELING HYPERPARAMS (Not used on Databricks)
    # ==================================================================

    USE_SMOTE = True
    SMOTE_SAMPLE_THRESHOLD = 100_000

    USE_OPTUNA = True
    OPTUNA_N_TRIALS = 20
    OPTUNA_TIMEOUT = 300

    RUN_LOGISTIC_REGRESSION = True

    DEVICE = "cpu"  # For PyTorch models (will be overridden on GPU machines)

    NOISE_START = 0.05
    NOISE_WARMUP_EPOCHS = 40
    GAN_USE_MALICIOUS = True
    GAN_MALICIOUS_RATIO = 0.1
    GAN_MALICIOUS_WEIGHT = 0.2
    GAN_TEST_INJECTION_RATES = True

    AUTOENCODER_LATENT_DIM = 8
    AUTOENCODER_EPOCHS = 100
    AUTOENCODER_BATCH_SIZE = 4096
    AUTOENCODER_LR = 1e-3

    USE_LATENT_SPARSITY = True
    SPARSITY_WARMUP_EPOCHS = 30
    LATENT_SPARSITY_LAMBDA = 1e-3

    BINARY_N_ESTIMATORS = 200
    BINARY_LEARNING_RATE = 0.1
    BINARY_NUM_LEAVES = 50

    MULTICLASS_N_ESTIMATORS = 200
    MULTICLASS_LEARNING_RATE = 0.1
    MULTICLASS_MAX_DEPTH = 10
    XGB_TREE_METHOD = "hist"

    KMEANS_N_CLUSTERS = 5
    KMEANS_BATCH_SIZE = 10_000

    NUM_WORKERS = 2
    PIN_MEMORY = True
    N_JOBS = 32

    # ==================================================================
    # UTILITY FUNCTIONS
    # ==================================================================

    @staticmethod
    def ensure_output_dirs():
        """Create all necessary output directories"""
        if Config.is_databricks():
            # On Databricks with Unity Catalog Volumes, directories are created automatically
            # when writing files. No need to pre-create them.
            print("ℹ️  Running on Databricks - output directories will be created automatically")
            return
        else:
            # Local: use os.makedirs
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
                Config.GAN_RESULTS_DIR,
                Config.LOGS_DIR,
            ]:
                os.makedirs(dir_path, exist_ok=True)

    @staticmethod
    def set_seeds():
        """Set random seeds for reproducibility"""
        import random
        import numpy as np
        random.seed(Config.RANDOM_STATE)
        np.random.seed(Config.RANDOM_STATE)
        torch.manual_seed(Config.RANDOM_STATE)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(Config.RANDOM_STATE)

    @staticmethod
    def print_mode_info():
        """Print current configuration mode"""
        print("\n" + "=" * 70)
        print("CONFIGURATION MODE")
        print("=" * 70)
        mode = "TEST MODE" if Config.TEST_MODE else "PRODUCTION MODE"
        print(f"   Mode: {mode}")
        print(f"   Environment: {Config.get_environment()}")
        print(f"   Raw Data Directory: {Config.RAW_DIR_ORIGINAL}")
        print(f"   Output Directory: {Config.OUTPUTS_DIR}")
        print(f"   Device: {Config.DEVICE}")
        print("=" * 70)