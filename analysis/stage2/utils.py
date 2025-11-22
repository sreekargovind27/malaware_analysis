"""
Stage 2 Utility Functions - PySpark Version
Works both locally and on Databricks.

Provides helper functions for:
- Loading engineered data (Spark)
- Loading splits (Spark)
- Saving/loading feature lists
- Repartitioning utilities for optimization
"""

import os
import joblib
from pyspark.sql import SparkSession, DataFrame
from config import Config


# ==================================================================
# DATA LOADING FUNCTIONS
# ==================================================================

def load_engineered_data_spark(spark: SparkSession) -> DataFrame:
    """
    Load the engineered flow features parquet (Spark DataFrame).

    Args:
        spark: Active SparkSession

    Returns:
        Spark DataFrame with engineered features

    Raises:
        FileNotFoundError: If engineered data doesn't exist
    """
    if not os.path.exists(Config.ENGINEERED_DATA_PATH):
        raise FileNotFoundError(
            f"Engineered data not found at: {Config.ENGINEERED_DATA_PATH}\n"
            f"Please run feature_engineering.py first."
        )

    print(f"📂 Loading engineered data from: {Config.ENGINEERED_DATA_PATH}")
    df = spark.read.parquet(Config.ENGINEERED_DATA_PATH)
    row_count = df.count()
    print(f"✅ Loaded {row_count:,} rows, {len(df.columns)} columns")

    return df


def load_device_features_spark(spark: SparkSession) -> DataFrame:
    """
    Load the device-level aggregated features parquet (Spark DataFrame).

    Args:
        spark: Active SparkSession

    Returns:
        Spark DataFrame with device features

    Raises:
        FileNotFoundError: If device features don't exist
    """
    if not os.path.exists(Config.DEVICE_FEATURES_PATH):
        raise FileNotFoundError(
            f"Device features not found at: {Config.DEVICE_FEATURES_PATH}\n"
            f"Please run device_aggregation.py first."
        )

    print(f"📂 Loading device features from: {Config.DEVICE_FEATURES_PATH}")
    df = spark.read.parquet(Config.DEVICE_FEATURES_PATH)
    row_count = df.count()
    print(f"✅ Loaded {row_count:,} devices, {len(df.columns)} columns")

    return df


def load_split_spark(spark: SparkSession, split_name: str) -> DataFrame:
    """
    Load a train/val/test split parquet (Spark DataFrame).

    Args:
        spark: Active SparkSession
        split_name: One of 'train', 'val', 'test'

    Returns:
        Spark DataFrame for the requested split

    Raises:
        ValueError: If split_name is invalid
        FileNotFoundError: If split doesn't exist
    """
    split_paths = {
        'train': Config.TRAIN_PATH,
        'val': Config.VAL_PATH,
        'test': Config.TEST_PATH
    }

    if split_name not in split_paths:
        raise ValueError(f"Invalid split_name: {split_name}. Must be one of {list(split_paths.keys())}")

    path = split_paths[split_name]

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{split_name.capitalize()} split not found at: {path}\n"
            f"Please run train_test_split.py first."
        )

    print(f"📂 Loading {split_name} split from: {path}")
    df = spark.read.parquet(path)
    row_count = df.count()
    print(f"✅ Loaded {row_count:,} rows")

    return df


# ==================================================================
# FEATURE LIST FUNCTIONS
# ==================================================================

def save_feature_list(feature_list: list, filepath: str = None):
    """
    Save the feature list to a joblib file.

    Args:
        feature_list: List of feature column names
        filepath: Optional custom path (defaults to Config.FEATURE_LIST_PATH)
    """
    if filepath is None:
        filepath = Config.FEATURE_LIST_PATH

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(feature_list, filepath)
    print(f"💾 Feature list saved ({len(feature_list)} features): {filepath}")


def load_feature_list(filepath: str = None) -> list:
    """
    Load the feature list from a joblib file.

    Args:
        filepath: Optional custom path (defaults to Config.FEATURE_LIST_PATH)

    Returns:
        List of feature column names

    Raises:
        FileNotFoundError: If feature list doesn't exist
    """
    if filepath is None:
        filepath = Config.FEATURE_LIST_PATH

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Feature list not found at: {filepath}\n"
            f"Please run feature_engineering.py first."
        )

    features = joblib.load(filepath)
    print(f"📂 Feature list loaded ({len(features)} features)")
    return features


# ==================================================================
# REPARTITIONING UTILITIES (Databricks Optimization)
# ==================================================================

def smart_repartition(df: DataFrame, num_partitions: int = None, partition_cols: list = None) -> DataFrame:
    """
    Intelligently repartition DataFrame based on environment and size.

    Args:
        df: Spark DataFrame to repartition
        num_partitions: Target number of partitions (None = use Config defaults)
        partition_cols: Columns to partition by (for groupBy optimization)

    Returns:
        Repartitioned DataFrame
    """
    if num_partitions is None:
        num_partitions = (Config.DATABRICKS_REPARTITION_SIZE
                          if Config.is_databricks()
                          else Config.LOCAL_REPARTITION_SIZE)

    if partition_cols:
        print(f"📊 Repartitioning by {partition_cols} into {num_partitions} partitions")
        return df.repartition(num_partitions, *partition_cols)
    else:
        print(f"📊 Repartitioning into {num_partitions} partitions")
        return df.repartition(num_partitions)


def smart_coalesce(df: DataFrame, num_partitions: int = None) -> DataFrame:
    """
    Coalesce DataFrame for efficient writing.

    Args:
        df: Spark DataFrame to coalesce
        num_partitions: Target number of partitions (None = use Config defaults)

    Returns:
        Coalesced DataFrame
    """
    if num_partitions is None:
        num_partitions = (Config.DATABRICKS_COALESCE_PARTITIONS
                          if Config.is_databricks()
                          else 4)

    current_partitions = df.rdd.getNumPartitions()

    if current_partitions > num_partitions:
        print(f"📊 Coalescing from {current_partitions} to {num_partitions} partitions")
        return df.coalesce(num_partitions)
    else:
        print(f"📊 Already at {current_partitions} partitions (target: {num_partitions})")
        return df


def smart_persist(df: DataFrame, storage_level: str = None) -> DataFrame:
    """
    Persist DataFrame with appropriate storage level for environment.

    Args:
        df: Spark DataFrame to persist
        storage_level: Storage level string (None = use Config default)

    Returns:
        Persisted DataFrame
    """
    from pyspark.storagelevel import StorageLevel

    if storage_level is None:
        storage_level = Config.PERSIST_STORAGE_LEVEL

    level_map = {
        "MEMORY_ONLY": StorageLevel.MEMORY_ONLY,
        "MEMORY_AND_DISK": StorageLevel.MEMORY_AND_DISK,
        "DISK_ONLY": StorageLevel.DISK_ONLY,
        "MEMORY_ONLY_2": StorageLevel.MEMORY_ONLY_2,
        "MEMORY_AND_DISK_2": StorageLevel.MEMORY_AND_DISK_2
    }

    level = level_map.get(storage_level, StorageLevel.MEMORY_AND_DISK)
    print(f"💾 Persisting with storage level: {storage_level}")

    return df.persist(level)


# ==================================================================
# PARQUET WRITE UTILITIES
# ==================================================================

def write_parquet_optimized(df: DataFrame, path: str, mode: str = "overwrite",
                            coalesce: bool = True, partition_by: list = None):
    """
    Write DataFrame to parquet with optimized settings.

    Args:
        df: Spark DataFrame to write
        path: Output path
        mode: Write mode ('overwrite', 'append', etc.)
        coalesce: Whether to coalesce before writing
        partition_by: Columns to partition output by
    """
    print(f"\n💾 Writing parquet to: {path}")

    # Coalesce if requested (reduces file count)
    if coalesce:
        df = smart_coalesce(df)

    writer = df.write.mode(mode)

    # Add optimizations
    writer = (writer
              .option("compression", Config.COMPRESSION_CODEC)
              .option("maxRecordsPerFile", Config.MAX_RECORDS_PER_FILE)
              )

    # Partition if requested
    if partition_by:
        print(f"   Partitioning by: {partition_by}")
        writer = writer.partitionBy(*partition_by)

    # Write
    writer.parquet(path)
    print(f"✅ Parquet written successfully")


# ==================================================================
# VALIDATION UTILITIES
# ==================================================================

def validate_dataframe(df: DataFrame, name: str = "DataFrame"):
    """
    Validate basic DataFrame properties and print summary.

    Args:
        df: Spark DataFrame to validate
        name: Name for logging
    """
    print(f"\n🔍 Validating {name}...")

    row_count = df.count()
    col_count = len(df.columns)

    print(f"   Rows: {row_count:,}")
    print(f"   Columns: {col_count}")

    if row_count == 0:
        print(f"   ⚠️  WARNING: {name} is empty!")

    if col_count == 0:
        print(f"   ⚠️  WARNING: {name} has no columns!")

    print(f"✅ {name} validation complete")


def check_nulls(df: DataFrame, columns: list = None):
    """
    Check for null values in specified columns.

    Args:
        df: Spark DataFrame
        columns: List of columns to check (None = all columns)

    Returns:
        dict: Column -> null count mapping
    """
    from pyspark.sql import functions as F

    if columns is None:
        columns = df.columns

    print(f"\n🔍 Checking nulls in {len(columns)} columns...")

    null_counts = {}
    for col in columns:
        null_count = df.filter(F.col(col).isNull()).count()
        if null_count > 0:
            null_counts[col] = null_count

    if null_counts:
        print(f"   ⚠️  Found nulls in {len(null_counts)} columns:")
        for col, count in sorted(null_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            print(f"      {col}: {count:,}")
    else:
        print(f"   ✅ No nulls found")

    return null_counts


# ==================================================================
# ENVIRONMENT INFO
# ==================================================================

def print_environment_info(spark: SparkSession):
    """Print current Spark environment configuration."""
    print("\n" + "=" * 70)
    print("SPARK ENVIRONMENT INFO")
    print("=" * 70)
    print(f"Environment: {Config.get_environment()}")
    print(f"Spark Version: {spark.version}")
    print(f"Base Path: {Config.BASE_PATH}")
    print(f"Shuffle Partitions: {spark.conf.get('spark.sql.shuffle.partitions')}")
    print(f"Default Parallelism: {spark.conf.get('spark.default.parallelism', 'Not Set')}")

    if Config.is_databricks():
        print(f"Databricks Runtime: {os.environ.get('DATABRICKS_RUNTIME_VERSION', 'Unknown')}")
        print(f"AQE Enabled: {spark.conf.get('spark.sql.adaptive.enabled', 'Not Set')}")

    print("=" * 70)
