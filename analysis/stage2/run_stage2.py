#!/usr/bin/env python3
"""
Stage 2 Runner - PySpark Version
Orchestrates the complete Stage 2 workflow:
1. Flow-level feature engineering          (raw CSVs -> flow_features.parquet)
2. Device-level aggregation                (per device_ip -> device_features.parquet)
3. Train/val/test splitting                (stratified 70/10/20 parquet splits)
"""

import sys
import time
from pyspark.sql import SparkSession

# Early banner
print("=" * 80, flush=True)
print("STAGE 2 STARTING NOW", flush=True)
print("=" * 80, flush=True)

# Make sure local package imports work when run from repo root
sys.path.insert(0, ".")
sys.path.insert(0, "./analysis")
sys.path.insert(0, "./analysis/stage2")

from config import Config


def get_spark_session():
    """
    Create SparkSession tuned for local dev but OK for tens/hundreds of millions
    of rows when scaled out. If you're on a real cluster, override spark.master,
    memory, and shuffle partitions with cluster settings.
    """
    print("\n⚙️  Initializing Spark session ...", flush=True)

    spark = (
        SparkSession.builder
        .appName("Stage2-Pipeline")
        # local mode; remove/override on cluster
        .config("spark.master", "local[*]")
        # memory tuning for local box
        .config("spark.driver.memory", "8g")
        .config("spark.executor.memory", "8g")
        # fewer shuffle partitions for local dev
        .config("spark.sql.shuffle.partitions", "8")
        .config("spark.default.parallelism", "8")
        # Arrow can speed Pandas interop
        .config("spark.sql.execution.arrow.pyspark.enabled", "true")
        # Kryo serializer is faster than Java serializer
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
        # show progress bars
        .config("spark.ui.showConsoleProgress", "true")
        .getOrCreate()
    )

    spark.sparkContext.setLogLevel("WARN")
    print("✅ Spark session ready.\n", flush=True)
    return spark


def main():
    """Run complete Stage 2 pipeline with PySpark."""
    print("\n" + "=" * 70)
    print("🚀 STAGE 2: FEATURE ENGINEERING & DATA PREPARATION (PySpark)")
    print("=" * 70)
    print(f"Mode: {'TEST' if Config.TEST_MODE else 'PRODUCTION'}")
    print(f"Input data dir:   {Config.RAW_DIR_ORIGINAL}")
    print(f"Output base dir:  {Config.STAGE2_PREPARED_DIR}")
    print("=" * 70)

    overall_start = time.time()

    # Ensure output directory structure exists
    Config.ensure_output_dirs()

    # Spin up Spark once and reuse it
    spark = get_spark_session()

    # ---------------------------------------------------------
    # STEP 1: FLOW-LEVEL FEATURE ENGINEERING
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 1/3: FEATURE ENGINEERING (flows → features)")
    print("=" * 70)
    try:
        from analysis.stage2.feature_engineering import build_engineered_dataset_spark
        build_engineered_dataset_spark(spark)
        print("✅ Feature engineering complete!")
    except Exception as e:
        print(f"❌ ERROR in feature engineering: {e}")
        import traceback
        traceback.print_exc()
        spark.stop()
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 2: DEVICE-LEVEL AGGREGATION
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2/3: DEVICE AGGREGATION (per device_ip)")
    print("=" * 70)
    try:
        from analysis.stage2.device_aggregation import aggregate_devices_spark
        aggregate_devices_spark(spark)
        print("✅ Device aggregation complete!")
    except Exception as e:
        print(f"❌ ERROR in device aggregation: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  Continuing without device features (device_features.parquet will be missing)")

    # ---------------------------------------------------------
    # STEP 2.5: GRAPH BUILDING (optional, for GNN models)
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 2.5/4: GRAPH BUILDING (heterogeneous graph)")
    print("=" * 70)
    try:
        from analysis.stage2.build_graph import build_heterogeneous_graph_spark
        build_heterogeneous_graph_spark(spark)
        print("✅ Graph building complete!")
    except Exception as e:
        print(f"❌ ERROR in graph building: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  Continuing without graph (GNN models will be unavailable)")

    # ---------------------------------------------------------
    # STEP 3: TRAIN / VAL / TEST SPLITTING
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 3/3: TRAIN/VAL/TEST SPLITTING (70/10/20)")
    print("=" * 70)
    try:
        from analysis.stage2.train_test_split import create_and_save_master_splits_spark
        create_and_save_master_splits_spark(spark)
        print("✅ Data splitting complete!")
    except Exception as e:
        print(f"❌ ERROR in data splitting: {e}")
        import traceback
        traceback.print_exc()
        # Splits are critical for downstream training, so stop hard.
        spark.stop()
        sys.exit(1)

    # ---------------------------------------------------------
    # STEP 4: QUALITY VALIDATION
    # ---------------------------------------------------------
    print("\n" + "=" * 70)
    print("STEP 4/4: QUALITY VALIDATION")
    print("=" * 70)
    try:
        from analysis.stage2.stage2_quality_report import validate_stage2_outputs
        validate_stage2_outputs()
        print("✅ Quality validation complete!")
    except Exception as e:
        print(f"❌ ERROR in quality validation: {e}")
        import traceback
        traceback.print_exc()
        print("⚠️  Continuing without quality report")

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------
    total_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("🎉 STAGE 2 COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f"📁 Outputs saved under: {Config.STAGE2_PREPARED_DIR}")
    print("\n📊 Artifacts:")
    print(f"   • Flow features parquet:    {Config.ENGINEERED_DATA_PATH}")
    print(f"   • Device features parquet:  {Config.DEVICE_FEATURES_PATH}")
    print(f"   • Train set parquet:        {Config.TRAIN_SET_PATH}")
    print(f"   • Val set parquet:          {Config.VAL_SET_PATH}")
    print(f"   • Test set parquet:         {Config.TEST_SET_PATH}")
    print("\n🚀 Ready for Stage 3: model training / evaluation")
    print("=" * 70)

    spark.stop()
    print("🧹 Spark session closed.")


if __name__ == "__main__":
    main()
