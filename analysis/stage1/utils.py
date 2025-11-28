"""
Stage 1 Utility Functions - PySpark Version
Works on both local and Databricks.

Provides helper functions for:
- Loading raw data (with header normalization)
- Parsing labels (binary, attack_type, malware_family)
- Getting numeric columns
- Saving JSON reports
"""

import os
import json
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark.sql import types as T

from config import Config


# ============================================================================
# RAW DATA LOADING
# ============================================================================

def load_raw_data(spark: SparkSession) -> DataFrame:
    """
    Load raw IoT-23 CSVs with header normalization.
    Tolerant to inconsistent column names across captures.

    Args:
        spark: Active SparkSession

    Returns:
        Spark DataFrame with normalized raw data
    """
    # For Unity Catalog Volumes, don't use os.path.join
    if Config.is_databricks():
        input_glob = f"{Config.RAW_DIR_ORIGINAL}/*.csv"
    else:
        input_glob = os.path.join(Config.RAW_DIR_ORIGINAL, "*.csv")


    print(f"\n📂 Loading raw CSVs from: {input_glob}")

    # Read all CSVs
    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("comment", "#")
        .option("mode", "PERMISSIVE")
        .csv(input_glob)
    )

    # Add source folder tracking (Unity Catalog compatible)
    df = df.withColumn(
        "Source_Folder",
        F.regexp_extract(F.col("_metadata.file_path"), r"([^/]+)\.csv$", 1)
    )


    # Normalize column names (handle dotted vs underscored)
    expected_cols = [
        "ts", "uid",
        "id.orig_h", "id.orig_p",
        "id.resp_h", "id.resp_p",
        "proto", "service",
        "duration",
        "orig_bytes", "resp_bytes",
        "conn_state",
        "local_orig", "local_resp",
        "missed_bytes",
        "history",
        "orig_pkts", "orig_ip_bytes",
        "resp_pkts", "resp_ip_bytes",
        "label", "detailed-label",
    ]

    existing_cols = df.columns
    for col_dot in expected_cols:
        alt = col_dot.replace(".", "_").replace("-", "_")
        if col_dot not in existing_cols and alt in existing_cols:
            df = df.withColumnRenamed(alt, col_dot)

    # Ensure missing columns exist
    for c in expected_cols:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("string"))

    # Cast numeric columns
    numeric_cols = [
        "duration", "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"
    ]

    # Cast numeric columns (safe casting to handle '-' and other invalid values)
    for nc in numeric_cols:
        df = df.withColumn(
            nc,
            F.when(F.col(nc).rlike("^-?[0-9]+\.?[0-9]*$"), 
                F.col(nc).cast(T.DoubleType()))
            .otherwise(None)
        )


    # Cast port columns (safe casting to handle malformed data)
    for port_col in ["id.orig_p", "id.resp_p"]:
        df = df.withColumn(
            port_col,
            F.when(F.col(f"`{port_col}`").rlike("^[0-9]+$"), 
                F.col(f"`{port_col}`").cast(T.IntegerType()))
            .otherwise(None)
        )


    row_count = df.count()
    print(f"✅ Loaded {row_count:,} raw flows from {df.select('Source_Folder').distinct().count()} captures")

    # Sample if requested
    if Config.DATA_SAMPLE_FRACTION < 1.0:
        df = df.sample(False, Config.DATA_SAMPLE_FRACTION, seed=Config.RANDOM_STATE)
        sampled_count = df.count()
        print(f"📊 Sampled down to {sampled_count:,} rows ({Config.DATA_SAMPLE_FRACTION * 100:.1f}%)")

    return df


# ============================================================================
# LABEL PARSING FUNCTIONS
# ============================================================================

def get_combined_label_column(label_col: str = "label", detailed_col: str = "detailed-label"):
    """
    Get combined label column (prefers detailed-label over label).

    Args:
        label_col: Name of label column
        detailed_col: Name of detailed-label column

    Returns:
        Spark Column with combined label
    """
    return F.when(
        F.col(f"`{detailed_col}`").isNotNull() & (F.col(f"`{detailed_col}`") != ""),
        F.col(f"`{detailed_col}`")
    ).otherwise(F.col(label_col))


def parse_binary_label(combined_col):
    """
    Parse binary label (Benign vs Malicious) from combined label column.

    Args:
        combined_col: Spark Column with combined label

    Returns:
        Spark Column with binary label
    """
    return F.when(
        combined_col.rlike("(?i)benign"),
        F.lit("Benign")
    ).otherwise(F.lit("Malicious"))


def parse_attack_type(combined_col):
    """
    Parse attack type (multiclass) from combined label column.

    Args:
        combined_col: Spark Column with combined label

    Returns:
        Spark Column with attack type
    """
    return F.when(
        combined_col.rlike("(?i)benign"),
        F.lit("Benign")
    ).when(
        combined_col.rlike("(?i)ddos"),
        F.lit("DDoS")
    ).when(
        combined_col.rlike("(?i)dos"),
        F.lit("DoS")
    ).when(
        combined_col.rlike("(?i)scan"),
        F.lit("PortScan")
    ).when(
        combined_col.rlike("(?i)c&c"),
        F.lit("C&C")
    ).when(
        combined_col.rlike("(?i)okiru|(?i)partialflows"),
        F.lit("PartialTraffic")
    ).when(
        combined_col.rlike("(?i)filedownload"),
        F.lit("FileDownload")
    ).otherwise(F.lit("Attack"))


def parse_malware_family(combined_col):
    """
    Parse malware family from combined label column.

    Args:
        combined_col: Spark Column with combined label

    Returns:
        Spark Column with malware family
    """
    return F.when(
        combined_col.rlike("(?i)benign"),
        F.lit("Benign")
    ).when(
        combined_col.rlike("(?i)mirai"),
        F.lit("Mirai")
    ).when(
        combined_col.rlike("(?i)torii"),
        F.lit("Torii")
    ).when(
        combined_col.rlike("(?i)gagfyt|(?i)gafgyt"),
        F.lit("Gagfyt")
    ).when(
        combined_col.rlike("(?i)hajime"),
        F.lit("Hajime")
    ).when(
        combined_col.rlike("(?i)kenjiro"),
        F.lit("Kenjiro")
    ).when(
        combined_col.rlike("(?i)okiru"),
        F.lit("Okiru")
    ).when(
        combined_col.rlike("(?i)muhstik"),
        F.lit("Muhstik")
    ).when(
        combined_col.rlike("(?i)hide"),
        F.lit("Hide and Seek")
    ).when(
        combined_col.rlike("(?i)hakai"),
        F.lit("Hakai")
    ).when(
        combined_col.rlike("(?i)irc"),
        F.lit("IRCBot")
    ).otherwise(F.lit("Unknown"))


# ============================================================================
# FEATURE EXTRACTION
# ============================================================================

def get_numeric_columns(df: DataFrame) -> list:
    """
    Get list of numeric column names from DataFrame.

    Args:
        df: Spark DataFrame

    Returns:
        List of numeric column names
    """
    numeric_types = [
        T.ByteType, T.ShortType, T.IntegerType, T.LongType,
        T.FloatType, T.DoubleType, T.DecimalType
    ]

    numeric_cols = []
    for field in df.schema.fields:
        if any(isinstance(field.dataType, t) for t in numeric_types):
            numeric_cols.append(field.name)

    return numeric_cols


# ============================================================================
# REPORT SAVING
# ============================================================================

def save_json_report(data: dict, filepath: str):
    """Save dictionary as formatted JSON file (Volumes-compatible)."""
    import json
    from config import Config
    
    # Convert non-serializable types
    def convert_types(obj):
        if isinstance(obj, (int, float, str, bool, type(None))):
            return obj
        elif isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_types(item) for item in obj]
        else:
            return str(obj)
    
    clean_data = convert_types(data)
    json_str = json.dumps(clean_data, indent=2)
    
    if Config.is_databricks():
        # Import dbutils properly for Serverless
        from pyspark.dbutils import DBUtils
        from pyspark.sql import SparkSession
        spark = SparkSession.getActiveSession()
        dbutils = DBUtils(spark)
        # Write to Volumes using dbutils
        dbutils.fs.put(filepath, json_str, overwrite=True)
    else:
        # Local: use regular file writing
        import os
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(json_str)
            
def load_json_report(filepath: str) -> dict:
    """
    Load JSON report from file.

    Args:
        filepath: Path to JSON file

    Returns:
        Dictionary with report data
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Report not found: {filepath}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    return data


# ============================================================================
# VALIDATION HELPERS
# ============================================================================

def validate_dataframe(df: DataFrame, name: str = "DataFrame"):
    """
    Validate basic DataFrame properties.

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


def check_required_columns(df: DataFrame, required_cols: list):
    """
    Check if required columns exist in DataFrame.

    Args:
        df: Spark DataFrame
        required_cols: List of required column names

    Raises:
        ValueError: If any required columns are missing
    """
    missing_cols = [c for c in required_cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"✅ All required columns present: {required_cols}")


# ============================================================================
# STATISTICS HELPERS
# ============================================================================

def compute_class_distribution(df: DataFrame, label_col: str) -> dict:
    """
    Compute class distribution for a label column.

    Args:
        df: Spark DataFrame
        label_col: Name of label column

    Returns:
        Dictionary mapping class labels to counts
    """
    class_counts = df.groupBy(label_col).count().collect()
    distribution = {row[label_col]: int(row['count']) for row in class_counts}

    return distribution


def compute_missing_percentage(df: DataFrame, columns: list = None) -> dict:
    """
    Compute percentage of missing values per column.

    Args:
        df: Spark DataFrame
        columns: List of columns to check (None = all columns)

    Returns:
        Dictionary mapping column names to missing percentages
    """
    if columns is None:
        columns = df.columns

    total_rows = df.count()

    missing_pcts = {}
    for col in columns:
        null_count = df.filter(F.col(col).isNull()).count()
        missing_pct = (null_count / total_rows * 100) if total_rows > 0 else 0
        missing_pcts[col] = round(missing_pct, 2)

    return missing_pcts


# ============================================================================
# PRINTING HELPERS
# ============================================================================

def print_dict_summary(data: dict, title: str = "Summary", max_items: int = 10):
    """
    Pretty print a dictionary summary.

    Args:
        data: Dictionary to print
        title: Title for the summary
        max_items: Maximum number of items to show
    """
    print(f"\n{title}:")

    items = list(data.items())

    # Sort by value if numeric
    try:
        items = sorted(items, key=lambda x: x[1], reverse=True)
    except:
        pass

    for i, (key, value) in enumerate(items[:max_items]):
        if isinstance(value, float):
            print(f"   {key}: {value:.2f}")
        elif isinstance(value, int):
            print(f"   {key}: {value:,}")
        else:
            print(f"   {key}: {value}")

    if len(items) > max_items:
        print(f"   ... and {len(items) - max_items} more")
