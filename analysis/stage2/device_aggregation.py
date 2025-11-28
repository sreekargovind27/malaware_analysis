"""
Stage 2: Device-Level Aggregation - PySpark Version
Works on both local and Databricks with automatic optimization.

Pipeline:
1. Load engineered flow features
2. Aggregate per device_ip (statistical features)
3. Create device-level behavioral features
4. Save device_features.parquet

Optimizations:
- Repartition by device_ip before groupBy (critical for 40GB data)
- Smart persist for intermediate results
- Optimized parquet writes
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F
from pyspark.sql import Window

from config import Config
from analysis.stage2.utils import (
    load_engineered_data_spark, smart_repartition, smart_persist,
    write_parquet_optimized, print_environment_info, validate_dataframe
)


# ============================================================================
# DEVICE AGGREGATION
# ============================================================================

def aggregate_devices_spark(spark):
    """
    Aggregate flow-level features to device-level features.

    Creates per-device statistics:
    - Flow counts (total, benign, malicious)
    - Duration statistics (mean, std, min, max)
    - Byte statistics (orig_bytes, resp_bytes)
    - Packet statistics (orig_pkts, resp_pkts)
    - Behavioral ratios (avg upload ratio, packet rate)
    - Port usage patterns
    - Attack type distribution

    Args:
        spark: Active SparkSession
    """
    print("\n" + "=" * 70)
    print("STAGE 2: DEVICE-LEVEL AGGREGATION")
    print("=" * 70)

    print_environment_info(spark)

    # ========================================
    # 1. LOAD ENGINEERED DATA
    # ========================================

    flows = load_engineered_data_spark(spark)

    # Validate required columns exist
    required_cols = ["device_ip", "label", "duration", "orig_bytes", "resp_bytes"]
    missing_cols = [c for c in required_cols if c not in flows.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    print(f"\n📊 Input: {flows.count():,} flows from {flows.select('device_ip').distinct().count():,} devices")

    # ========================================
    # 2. REPARTITION BY DEVICE_IP (CRITICAL FOR PERFORMANCE)
    # ========================================

    print("\n🔄 Optimizing partitioning for aggregation...")

    if Config.is_databricks():
        # Repartition by device_ip to colocate same devices
        # This prevents massive shuffles during groupBy
        flows = smart_repartition(flows, partition_cols=["device_ip"])

    # Persist to avoid re-reading from disk during multiple aggregations
    flows = smart_persist(flows)

    # ========================================
    # 3. DEVICE-LEVEL AGGREGATIONS
    # ========================================

    print("\n🔢 Aggregating features per device...")

    # Build aggregation expressions
    agg_exprs = []

    # --- Flow Counts ---
    agg_exprs.extend([
        F.count("*").alias("total_flows"),
        F.sum(F.when(F.col("label") == "Benign", 1).otherwise(0)).alias("benign_flows"),
        F.sum(F.when(F.col("label") == "Malicious", 1).otherwise(0)).alias("malicious_flows"),
    ])

    # --- Duration Statistics ---
    agg_exprs.extend([
        F.mean("duration").alias("duration_mean"),
        F.stddev("duration").alias("duration_std"),
        F.min("duration").alias("duration_min"),
        F.max("duration").alias("duration_max"),
    ])

    # --- Byte Statistics ---
    for byte_col in ["orig_bytes", "resp_bytes"]:
        agg_exprs.extend([
            F.mean(byte_col).alias(f"{byte_col}_mean"),
            F.stddev(byte_col).alias(f"{byte_col}_std"),
            F.sum(byte_col).alias(f"{byte_col}_total"),
            F.max(byte_col).alias(f"{byte_col}_max"),
        ])

    # --- Packet Statistics ---
    for pkt_col in ["orig_pkts", "resp_pkts"]:
        if pkt_col in flows.columns:
            agg_exprs.extend([
                F.mean(pkt_col).alias(f"{pkt_col}_mean"),
                F.sum(pkt_col).alias(f"{pkt_col}_total"),
            ])

    # --- Behavioral Features ---
    if "upload_ratio" in flows.columns:
        agg_exprs.append(F.mean("upload_ratio").alias("upload_ratio_mean"))

    if "packet_rate" in flows.columns:
        agg_exprs.append(F.mean("packet_rate").alias("packet_rate_mean"))

    if "suspicious_score" in flows.columns:
        agg_exprs.extend([
            F.mean("suspicious_score").alias("suspicious_score_mean"),
            F.max("suspicious_score").alias("suspicious_score_max"),
        ])

    # --- Port Usage Patterns ---
    for port_flag in ["is_port_23", "is_port_22", "is_port_80", "is_port_443"]:
        if port_flag in flows.columns:
            agg_exprs.append(F.sum(port_flag).alias(f"{port_flag}_count"))

    # --- Attack Type Distribution ---
    if "attack_type" in flows.columns:
        # Most common attack type
        agg_exprs.append(F.first("attack_type").alias("most_common_attack_type"))

    # --- Malware Family ---
    if "malware_family" in flows.columns:
        agg_exprs.append(F.first("malware_family").alias("most_common_malware_family"))

    # Perform aggregation
    device_stats = (
        flows
        .groupBy("device_ip", "label")
        .agg(*agg_exprs)
    )

    print(f"   ✅ Aggregated {len(agg_exprs)} features per (device_ip, label)")

    # ========================================
    # 4. DEVICE-LEVEL DERIVED FEATURES
    # ========================================

    print("\n🧮 Creating derived features...")

    # Malicious flow ratio
    device_stats = device_stats.withColumn(
        "malicious_flow_ratio",
        F.when(
            F.col("total_flows") > 0,
            F.col("malicious_flows") / F.col("total_flows")
        ).otherwise(0.0)
    )

    # Average bytes per flow
    device_stats = device_stats.withColumn(
        "avg_bytes_per_flow",
        F.when(
            F.col("total_flows") > 0,
            (F.col("orig_bytes_total") + F.col("resp_bytes_total")) / F.col("total_flows")
        ).otherwise(0.0)
    )

    # Port diversity (number of unique ports used)
    port_diversity = 0
    for port_flag in ["is_port_23", "is_port_22", "is_port_80", "is_port_443"]:
        col_name = f"{port_flag}_count"
        if col_name in device_stats.columns:
            port_diversity += F.when(F.col(col_name) > 0, 1).otherwise(0)

    device_stats = device_stats.withColumn("port_diversity", port_diversity)

    print(f"   ✅ Created derived features: malicious_ratio, avg_bytes_per_flow, port_diversity")

    # ========================================
    # 5. FILL NULLS & VALIDATION
    # ========================================

    print("\n🔍 Handling nulls and validating...")

    # Fill null statistics with 0 (only numeric columns)
    # Exclude string columns from numeric fill
    string_cols = {"most_common_attack_type", "most_common_malware_family"}
    stat_cols = [c for c in device_stats.columns if c not in ["device_ip", "label"] and c not in string_cols]

    for col in stat_cols:
        device_stats = device_stats.withColumn(col, F.coalesce(F.col(col), F.lit(0.0)))

    # Fill string columns with "Unknown"
    for col in string_cols:
        if col in device_stats.columns:
            device_stats = device_stats.withColumn(col, F.coalesce(F.col(col), F.lit("Unknown")))

    # Validate
    validate_dataframe(device_stats, "Device Statistics")

    # ========================================
    # 6. SAVE DEVICE FEATURES
    # ========================================

    print("\n💾 Saving device features...")

    # Persist before counting (will be used for write)
    device_stats = smart_persist(device_stats)

    device_count = device_stats.count()
    feature_count = len(device_stats.columns)

    print(f"   Devices: {device_count:,}")
    print(f"   Features: {feature_count}")

    # Write parquet
    write_parquet_optimized(
        device_stats,
        Config.DEVICE_FEATURES_PATH,
        mode="overwrite",
        coalesce=True
    )

    print(f"✅ Saved to {Config.DEVICE_FEATURES_PATH}")

    # ========================================
    # 7. SUMMARY STATISTICS
    # ========================================

    print("\n" + "=" * 70)
    print("📊 DEVICE AGGREGATION SUMMARY")
    print("=" * 70)

    # Label distribution at device level
    label_dist = device_stats.groupBy("label").count().collect()
    print("\nDevice-level label distribution:")
    for row in label_dist:
        print(f"   {row['label']}: {row['count']:,} devices")

    # Top malicious devices
    print("\nTop 5 devices by malicious flow count:")
    top_malicious = (
        device_stats
        .filter(F.col("malicious_flows") > 0)
        .orderBy(F.desc("malicious_flows"))
        .limit(5)
        .select("device_ip", "malicious_flows", "total_flows", "malicious_flow_ratio")
        .collect()
    )

    for row in top_malicious:
        print(
            f"   {row['device_ip']}: {row['malicious_flows']:,} malicious / {row['total_flows']:,} total ({row['malicious_flow_ratio']:.2%})")

    print("\n" + "=" * 70)
    print("✅ DEVICE AGGREGATION COMPLETE!")
    print("=" * 70)
    print(f"📊 Output: {device_count:,} devices × {feature_count} features")
    print(f"📂 Saved: {Config.DEVICE_FEATURES_PATH}")
    print("=" * 70)

    return device_stats


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    Config.ensure_output_dirs()
    spark = Config.get_spark_session("Stage2-DeviceAggregation")

    try:
        aggregate_devices_spark(spark)
    finally:
        if not Config.is_databricks():
            spark.stop()
            print("🧹 Spark session stopped")
