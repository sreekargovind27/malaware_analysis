"""
Stage 2: Feature Engineering - PySpark Version
Works on both local and Databricks with automatic optimization.

Pipeline:
1. Load raw CSVs (tolerant to inconsistent headers)
2. Parse labels (binary, attack_type, malware_family, attack_subtype)
3. Engineer numeric features (log1p, missing flags)
4. Extract IP-derived features
5. Create behavioral features (upload_ratio, packet_rate, etc.)
6. Encode categorical features (StringIndexer, OneHotEncoder)
7. Save engineered parquet + feature list

Optimizations:
- Smart repartitioning for Databricks
- Persist with MEMORY_AND_DISK
- Optimized parquet writes
"""

import os
import sys
import joblib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F
from pyspark.sql import types as T
from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

from config import Config
from analysis.stage2.utils import (
    smart_repartition, smart_persist, write_parquet_optimized,
    save_feature_list, print_environment_info
)


# ============================================================================
# RAW DATA LOADING
# ============================================================================

def load_raw_flows(spark):
    """
    Read ALL CSVs from Config.RAW_DIR_ORIGINAL.
    Tolerant to inconsistent headers across IoT-23 captures.
    Normalizes expected columns, enforces types, and adds Source_Folder.

    Args:
        spark: Active SparkSession

    Returns:
        Spark DataFrame with normalized raw data
    """
    input_glob = os.path.join(Config.RAW_DIR_ORIGINAL, "*.csv")
    print(f"📂 Reading raw CSVs from {input_glob}")

    df = (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("comment", "#")
        .option("mode", "PERMISSIVE")
        .csv(input_glob)
    )

    # Track which capture each row came from
    df = df.withColumn(
        "Source_Folder",
        F.regexp_extract(F.input_file_name(), r"([^/]+)\.csv$", 1)
    )

    # All columns we expect downstream
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

    # Map underscore forms -> dotted forms if needed
    existing_cols = df.columns
    for col_dot in expected_cols:
        alt = col_dot.replace(".", "_").replace("-", "_")
        if col_dot not in existing_cols and alt in existing_cols:
            df = df.withColumnRenamed(alt, col_dot)

    # Ensure missing expected cols exist
    for c in expected_cols:
        if c not in df.columns:
            df = df.withColumn(c, F.lit(None).cast("string"))

    # Helper: safely cast dotted numeric columns
    def safe_cast_numeric(df_in, col_name, target_type):
        """
        Spark hates withColumn('a.b', ...) because it parses as `a`.`b`.
        Use backticks for dotted column names.
        """
        return df_in.withColumn(col_name, F.col(f"`{col_name}`").cast(target_type))

    # Cast port columns
    for port_col in ["id.orig_p", "id.resp_p"]:
        df = safe_cast_numeric(df, port_col, T.IntegerType())

    # Cast numeric columns
    numeric_cols = [
        "duration", "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"
    ]
    for nc in numeric_cols:
        df = safe_cast_numeric(df, nc, T.DoubleType())

    initial_count = df.count()
    print(f"✅ Loaded {initial_count:,} raw flows with {len(df.columns)} columns")

    # Sample if requested
    if Config.DATA_SAMPLE_FRACTION < 1.0:
        df = df.sample(False, Config.DATA_SAMPLE_FRACTION, seed=Config.RANDOM_STATE)
        sampled_count = df.count()
        print(f"📊 Sampled down to {sampled_count:,} rows ({Config.DATA_SAMPLE_FRACTION * 100:.1f}%)")

    return df


# ============================================================================
# LABEL PARSING
# ============================================================================

def parse_labels(df):
    """
    Parse label columns from 'detailed-label' and 'label' fields.
    Creates: label (binary), attack_type, malware_family, attack_subtype

    Args:
        df: Raw flows DataFrame

    Returns:
        DataFrame with parsed label columns
    """
    print("\n🏷️  Parsing labels...")

    # Combine 'label' and 'detailed-label' columns (prefer detailed-label)
    combined_label = F.when(
        F.col("`detailed-label`").isNotNull() & (F.col("`detailed-label`") != ""),
        F.col("`detailed-label`")
    ).otherwise(F.col("label"))

    # Binary label: Benign vs Malicious
    df = df.withColumn(
        "label",
        F.when(
            combined_label.rlike("(?i)benign"),
            F.lit("Benign")
        ).otherwise(F.lit("Malicious"))
    )

    # Attack type (for multiclass)
    df = df.withColumn(
        "attack_type",
        F.when(
            combined_label.rlike("(?i)benign"),
            F.lit("Benign")
        ).when(
            combined_label.rlike("(?i)ddos"),
            F.lit("DDoS")
        ).when(
            combined_label.rlike("(?i)dos"),
            F.lit("DoS")
        ).when(
            combined_label.rlike("(?i)scan"),
            F.lit("PortScan")
        ).when(
            combined_label.rlike("(?i)c&c"),
            F.lit("C&C")
        ).when(
            combined_label.rlike("(?i)okiru|(?i)partialflows"),
            F.lit("PartialTraffic")
        ).when(
            combined_label.rlike("(?i)filedownload"),
            F.lit("FileDownload")
        ).otherwise(F.lit("Attack"))
    )

    # Malware family
    df = df.withColumn(
        "malware_family",
        F.when(
            combined_label.rlike("(?i)benign"),
            F.lit("Benign")
        ).when(
            combined_label.rlike("(?i)mirai"),
            F.lit("Mirai")
        ).when(
            combined_label.rlike("(?i)torii"),
            F.lit("Torii")
        ).when(
            combined_label.rlike("(?i)gagfyt|(?i)gafgyt"),
            F.lit("Gagfyt")
        ).when(
            combined_label.rlike("(?i)hajime"),
            F.lit("Hajime")
        ).when(
            combined_label.rlike("(?i)kenjiro"),
            F.lit("Kenjiro")
        ).when(
            combined_label.rlike("(?i)okiru"),
            F.lit("Okiru")
        ).when(
            combined_label.rlike("(?i)muhstik"),
            F.lit("Muhstik")
        ).when(
            combined_label.rlike("(?i)hide"),
            F.lit("Hide and Seek")
        ).when(
            combined_label.rlike("(?i)hakai"),
            F.lit("Hakai")
        ).when(
            combined_label.rlike("(?i)irc"),
            F.lit("IRCBot")
        ).otherwise(F.lit("Unknown"))
    )

    # Attack subtype (granular)
    df = df.withColumn(
        "attack_subtype",
        F.regexp_extract(combined_label, r"-\s*(.+)$", 1)
    )
    df = df.withColumn(
        "attack_subtype",
        F.when(F.col("attack_subtype") == "", F.lit("Unknown")).otherwise(F.col("attack_subtype"))
    )

    label_counts = df.groupBy("label").count().collect()
    print(f"   Binary distribution: {dict((r['label'], r['count']) for r in label_counts)}")

    return df


# ============================================================================
# NUMERIC FEATURE ENGINEERING
# ============================================================================

def engineer_numeric_features(df):
    """
    Engineer numeric features:
    - Handle missing values (fill with 0, add _was_missing flags)
    - Log1p transforms for skewed distributions

    Args:
        df: DataFrame with parsed labels

    Returns:
        DataFrame with engineered numeric features
    """
    print("\n🔢 Engineering numeric features...")

    numeric_cols = [
        "duration", "orig_bytes", "resp_bytes", "missed_bytes",
        "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"
    ]

    for col in numeric_cols:
        # Missing flag
        df = df.withColumn(
            f"{col}_was_missing",
            F.when(F.col(col).isNull(), 1).otherwise(0)
        )

        # Fill nulls with 0
        df = df.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))

        # Log1p transform
        df = df.withColumn(f"{col}_log1p", F.log1p(F.col(col)))

    print(f"   ✅ Created {len(numeric_cols) * 2} numeric features (log1p + missing flags)")

    return df


# ============================================================================
# IP FEATURE EXTRACTION
# ============================================================================

def extract_ip_features(df):
    """
    Extract IP-derived features:
    - Octets from orig_h and resp_h
    - Port features
    - Device IP (canonical device identifier)

    Args:
        df: DataFrame with numeric features

    Returns:
        DataFrame with IP features
    """
    print("\n🌐 Extracting IP features...")

    # Extract octets from id.orig_h
    for i in range(1, 5):
        df = df.withColumn(
            f"orig_h_octet{i}",
            F.regexp_extract(F.col("`id.orig_h`"), rf"^(\d+)\.(\d+)\.(\d+)\.(\d+)$", i).cast("int")
        )

    # Extract octets from id.resp_h
    for i in range(1, 5):
        df = df.withColumn(
            f"resp_h_octet{i}",
            F.regexp_extract(F.col("`id.resp_h`"), rf"^(\d+)\.(\d+)\.(\d+)\.(\d+)$", i).cast("int")
        )

    # Port features (already cast as int)
    df = df.withColumn("is_port_23", (F.col("`id.resp_p`") == 23).cast("int"))
    df = df.withColumn("is_port_22", (F.col("`id.resp_p`") == 22).cast("int"))
    df = df.withColumn("is_port_80", (F.col("`id.resp_p`") == 80).cast("int"))
    df = df.withColumn("is_port_443", (F.col("`id.resp_p`") == 443).cast("int"))

    # Device IP: canonical device identifier (prefer id.orig_h, fallback to resp_h)
    df = df.withColumn(
        "device_ip",
        F.when(
            F.col("`id.orig_h`").isNotNull() & (F.col("`id.orig_h`") != ""),
            F.col("`id.orig_h`")
        ).otherwise(F.col("`id.resp_h`"))
    )

    print(f"   ✅ Created IP features: 8 octets + 4 port flags + device_ip")

    return df


# ============================================================================
# BEHAVIORAL FEATURE ENGINEERING
# ============================================================================

def engineer_custom_features(df):
    """
    Create behavioral/custom features:
    - Upload/download ratios
    - Packet rates
    - Connection state flags
    - Suspicious score

    Args:
        df: DataFrame with IP features

    Returns:
        DataFrame with behavioral features
    """
    print("\n🧠 Engineering behavioral features...")

    # Upload ratio
    df = df.withColumn(
        "upload_ratio",
        F.when(
            (F.col("orig_bytes") + F.col("resp_bytes")) > 0,
            F.col("orig_bytes") / (F.col("orig_bytes") + F.col("resp_bytes"))
        ).otherwise(0.5)
    )

    # Packet rates
    df = df.withColumn(
        "packet_rate",
        F.when(
            F.col("duration") > 0,
            (F.col("orig_pkts") + F.col("resp_pkts")) / F.col("duration")
        ).otherwise(0.0)
    )

    df = df.withColumn(
        "byte_per_packet",
        F.when(
            (F.col("orig_pkts") + F.col("resp_pkts")) > 0,
            (F.col("orig_bytes") + F.col("resp_bytes")) / (F.col("orig_pkts") + F.col("resp_pkts"))
        ).otherwise(0.0)
    )

    # Connection state flag
    df = df.withColumn("is_S0_state", (F.col("conn_state") == "S0").cast("int"))

    # Suspicious score (heuristic)
    df = df.withColumn(
        "suspicious_score",
        (F.col("is_port_23") * 40) +
        (F.col("is_S0_state") * 30) +
        (F.col("is_port_22") * 25)
    )

    print(f"   ✅ Created behavioral features: ratios, rates, flags, suspicious_score")

    return df


# ============================================================================
# CATEGORICAL FEATURE ENCODING
# ============================================================================

def engineer_categorical_features(df):
    """
    Encode categorical features:
    - service, history -> StringIndexer (label encoding)
    - proto, conn_state -> StringIndexer + OneHotEncoder

    Args:
        df: DataFrame with behavioral features

    Returns:
        DataFrame with encoded categorical features
    """
    print("\n🏷️  Encoding categorical features...")

    # Normalize categorical columns (handle nulls, empty strings)
    def normalize(colname):
        return F.when(
            F.col(colname).isNull() |
            (F.col(colname) == "") |
            (F.col(colname) == "-") |
            (F.col(colname) == "nan") |
            (F.col(colname) == "None"),
            F.lit("unknown")
        ).otherwise(F.col(colname).cast("string"))

    for c in ["proto", "conn_state", "service", "history"]:
        if c in df.columns:
            df = df.withColumn(c, normalize(c))
        else:
            df = df.withColumn(c, F.lit("unknown"))

    stages = []

    # service/history -> index only (high cardinality)
    for c in ["service", "history"]:
        idx_col = f"{c}_idx"
        stages.append(
            StringIndexer(
                inputCol=c,
                outputCol=idx_col,
                handleInvalid="keep"
            )
        )

    # proto/conn_state -> OneHotEncode (low cardinality)
    for c in ["proto", "conn_state"]:
        idx_col = f"{c}_idx"
        vec_col = f"{c}_vec"
        stages.append(
            StringIndexer(
                inputCol=c,
                outputCol=idx_col,
                handleInvalid="keep"
            )
        )
        stages.append(
            OneHotEncoder(
                inputCol=idx_col,
                outputCol=vec_col
            )
        )

    # Fit pipeline
    pipe = Pipeline(stages=stages)
    model = pipe.fit(df)
    df = model.transform(df)

    print(f"   ✅ Encoded categorical features: service_idx, history_idx, proto_vec, conn_state_vec")

    return df


# ============================================================================
# FEATURE LIST BUILDING
# ============================================================================

def build_and_save_feature_list(df):
    """
    Build the final feature list for modeling (excludes label columns, raw fields).
    Saves to Config.FEATURE_LIST_PATH.

    Args:
        df: Fully engineered DataFrame
    """
    print("\n📋 Building feature list...")

    all_cols = set(df.columns)

    # Exclude non-feature columns
    exclude = {
        "ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p",
        "label", "detailed-label", "attack_type", "malware_family", "attack_subtype",
        "Source_Folder", "device_ip",
        "proto", "service", "conn_state", "history",  # Raw categoricals
        "proto_idx", "conn_state_idx"  # Intermediate indices (we use vectors)
    }

    features = sorted(list(all_cols - exclude))

    print(f"   Total columns: {len(all_cols)}")
    print(f"   Feature columns: {len(features)}")
    print(f"   First 10 features: {features[:10]}")

    save_feature_list(features)

    return features


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def build_engineered_dataset_spark(spark):
    """
    Main feature engineering pipeline.
    Reads raw CSVs, engineers features, saves parquet + feature list.

    Args:
        spark: Active SparkSession
    """
    print("\n" + "=" * 70)
    print("STAGE 2: FEATURE ENGINEERING")
    print("=" * 70)

    print_environment_info(spark)

    # 1. Load raw data
    df = load_raw_flows(spark)

    # 2. Repartition early for Databricks
    if Config.is_databricks():
        df = smart_repartition(df)

    # 3. Parse labels
    df = parse_labels(df)

    # 4. Numeric cleanup
    df = engineer_numeric_features(df)

    # 5. IP features
    df = extract_ip_features(df)

    # 6. Behavioral features
    df = engineer_custom_features(df)

    # 7. Categorical encoding
    df = engineer_categorical_features(df)

    # 8. Persist for safety (multiple operations coming)
    df = smart_persist(df)
    final_rows = df.count()
    print(f"\n✅ Engineered {final_rows:,} rows with {len(df.columns)} total columns")

    # 9. Build and save feature list
    features = build_and_save_feature_list(df)

    # 10. Write final parquet (optimized)
    print(f"\n💾 Saving engineered dataset...")
    write_parquet_optimized(
        df,
        Config.ENGINEERED_DATA_PATH,
        mode="overwrite",
        coalesce=True
    )
    print(f"✅ Saved to {Config.ENGINEERED_DATA_PATH}")

    print("\n" + "=" * 70)
    print("✅ FEATURE ENGINEERING COMPLETE!")
    print("=" * 70)
    print(f"📊 Final dataset: {final_rows:,} rows × {len(features)} features")
    print(f"📂 Output: {Config.ENGINEERED_DATA_PATH}")
    print(f"📋 Feature list: {Config.FEATURE_LIST_PATH}")
    print("=" * 70)

    return df


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    Config.ensure_output_dirs()
    spark = Config.get_spark_session("Stage2-FeatureEngineering")

    try:
        build_engineered_dataset_spark(spark)
    finally:
        if not Config.is_databricks():
            spark.stop()
            print("🧹 Spark session stopped")
