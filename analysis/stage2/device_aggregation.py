"""
Stage 2: Device Aggregation - PySpark Version

Goal:
- Take flow-level engineered features (one row per network flow)
- Aggregate them into per-device profiles (one row per originating IP)

This is the Spark equivalent of the pandas `aggregate_devices()` logic.
"""

import os
import sys

from pyspark.sql import functions as F
from pyspark.sql import SparkSession

# Make sure local imports work if run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config


def _get_numeric_feature_cols(df):
    """
    Infer which columns are numeric and safe to aggregate.

    We EXCLUDE:
    - device identity / metadata / targets
      ("device_ip", "uid", "Source_Folder", timestamps, etc.)
    - label columns ("label", "attack_type", "malware_family", "attack_subtype")

    We INCLUDE:
    - anything that's int*/bigint*/long*/float*/double*
      (Spark dtypes like 'int', 'bigint', 'double', etc.)
    """
    BLOCKLIST = {
        "device_ip",
        "uid",
        "Source_Folder",
        "ts",
        "label",
        "attack_type",
        "attack_subtype",
        "malware_family",
        "is_port_23",  # we'll still aggregate these flags, so DO NOT blocklist them
        "is_port_22",  # (remove from blocklist)
        "is_telnet",  # same
        "is_unknown_service",  # same
        "is_S0_state",  # same
        "suspicious_score",  # same
    }

    # fix: pull the flags above back out of BLOCKLIST
    # better approach: create BASE_BLOCKLIST then re-allow
    BASE_BLOCKLIST = {
        "device_ip",
        "uid",
        "Source_Folder",
        "ts",
        "label",
        "attack_type",
        "attack_subtype",
        "malware_family",
    }

    numeric_cols = []
    for field in df.schema:
        name = field.name
        dtype = field.dataType.simpleString()  # e.g. 'double', 'string', 'int', 'bigint'
        if name in BASE_BLOCKLIST:
            continue
        # only aggregate scalars, not vector columns (like proto_vec is VectorUDT)
        # we detect vectors by checking simpleString() doesn't contain 'vector'
        if "vector" in dtype:
            continue
        if (
                dtype.startswith("int")
                or dtype.startswith("bigint")
                or dtype.startswith("long")
                or dtype.startswith("float")
                or dtype.startswith("double")
        ):
            numeric_cols.append(name)

    return numeric_cols


def aggregate_devices_spark(spark):
    """
    Build per-device stats:
      - mean / std / min / max / sum for each numeric flow feature
      - flow_count per device
      - device_label (approx majority label for that device)
      - device_malware_family (dominant malware family if malicious, else 'Benign')

    Saves to Config.DEVICE_FEATURES_PATH.
    """

    print("\n" + "=" * 70)
    print("ðŸ”§ DEVICE-LEVEL AGGREGATION (PySpark)")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Load engineered flow-level features
    # ------------------------------------------------------------------
    print("ðŸ“‚ Loading engineered flow features for device aggregation...")
    print(f"   Path: {Config.ENGINEERED_DATA_PATH}")

    df = spark.read.parquet(Config.ENGINEERED_DATA_PATH)

    total_flows = df.count()
    print(f"âœ“ Loaded {total_flows:,} flows")

    # sanity: we NEED device_ip (we constructed this from id.orig_h in feature_engineering)
    if "device_ip" not in df.columns:
        raise RuntimeError(
            "Device aggregation needs 'device_ip' in the engineered parquet. "
            "This column should come from id.orig_h in feature_engineering."
        )

    # ------------------------------------------------------------------
    # Pick numeric columns to aggregate
    # ------------------------------------------------------------------
    numeric_features = _get_numeric_feature_cols(df)
    print(f"âœ“ Found {len(numeric_features)} numeric features to aggregate")

    # Build aggregations: mean/std/min/max/sum for each numeric feature
    agg_exprs = []
    for feat in numeric_features:
        # Use backticks to escape column names with dots or hyphens
        col_ref = f"`{feat}`"
        agg_exprs.extend([
            F.mean(F.col(col_ref)).alias(f"{feat}_mean"),
            F.stddev(F.col(col_ref)).alias(f"{feat}_std"),
            F.min(F.col(col_ref)).alias(f"{feat}_min"),
            F.max(F.col(col_ref)).alias(f"{feat}_max"),
            F.sum(F.col(col_ref)).alias(f"{feat}_sum"),
        ])

    # Also include flow_count
    agg_exprs.append(F.count(F.lit(1)).alias("flow_count"))

    print("\nâ³ Aggregating numeric stats per device (groupBy device_ip)...")
    device_stats = df.groupBy("device_ip").agg(*agg_exprs)

    # ------------------------------------------------------------------
    # Attach device_label (approximate majority)
    # ------------------------------------------------------------------
    # pandas version: most common label per device.
    # Spark fast path: first() is cheaper and generally fine for us.
    print("â³ Deriving per-device label...")
    device_labels = (
        df.groupBy("device_ip")
        .agg(F.first("label").alias("device_label"))
    )

    device_stats = device_stats.join(device_labels, on="device_ip", how="left")

    # ------------------------------------------------------------------
    # Attach device_malware_family
    # ------------------------------------------------------------------
    # pandas version:
    #   malicious_devices = df[df['label']=="Malicious"].groupby(device)['malware_family'].agg(mode)
    #
    # We'll do:
    #   - filter malicious rows only
    #   - first(malware_family) for each device
    # then join
    if "malware_family" in df.columns:
        malicious_fam = (
            df.filter(F.col("label") == "Malicious")
            .groupBy("device_ip")
            .agg(F.first("malware_family").alias("device_malware_family"))
        )
        device_stats = device_stats.join(malicious_fam, on="device_ip", how="left")
    else:
        # fallback if somehow malware_family missing
        device_stats = device_stats.withColumn("device_malware_family", F.lit(None).cast("string"))

    # Fill Benign for devices without a malware family
    device_stats = device_stats.withColumn(
        "device_malware_family",
        F.when(
            F.col("device_malware_family").isNull() & (F.col("device_label") != "Malicious"),
            F.lit("Benign")
        ).otherwise(F.col("device_malware_family"))
    )

    # ------------------------------------------------------------------
    # Cleanup nulls
    # ------------------------------------------------------------------
    print("â³ Cleaning nulls...")
    # stddev can be null for single-flow devices, fill with 0
    device_stats = device_stats.fillna(0)
    # malware family final fallback
    device_stats = device_stats.fillna({"device_malware_family": "Benign"})

    # Count device rows
    device_count = device_stats.count()
    print(f"\nâœ“ Created {device_count:,} device profiles")

    # ------------------------------------------------------------------
    # Save device-level parquet
    # ------------------------------------------------------------------
    print(f"\nðŸ’¾ Saving device features to: {Config.DEVICE_FEATURES_PATH}")
    (
        device_stats.write
        .mode("overwrite")
        .parquet(Config.DEVICE_FEATURES_PATH, compression="snappy")
    )

    # ------------------------------------------------------------------
    # Small summary stats (collected to driver)
    # ------------------------------------------------------------------
    print("\nðŸ“Š Device Statistics:")

    print(f"   Total devices: {device_count:,}")

    label_counts = (
        device_stats.groupBy("device_label")
        .agg(F.count(F.lit(1)).alias("cnt"))
        .collect()
    )
    for row in label_counts:
        print(f"   {row['device_label']}: {row['cnt']:,}")

    agg_summary = device_stats.agg(
        F.mean("flow_count").alias("avg_flows"),
        F.max("flow_count").alias("max_flows")
    ).collect()[0]

    avg_flows = agg_summary["avg_flows"]
    max_flows = agg_summary["max_flows"]

    print(f"   Avg flows per device: {avg_flows:.1f}")
    print(f"   Max flows per device: {int(max_flows)}")

    print("\nâœ… Device aggregation complete!")
    print("=" * 70)

    return device_stats


if __name__ == "__main__":
    # Standalone debug mode:
    Config.ensure_output_dirs()
    spark = (
        SparkSession.builder
        .appName("Stage2-DeviceAggregation-Standalone")
        .config("spark.master", "local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        aggregate_devices_spark(spark)
    finally:
        spark.stop()
