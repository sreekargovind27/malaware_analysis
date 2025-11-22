#!/usr/bin/env python3
"""
Stage 2 Runner - PySpark Version
Works on both local and Databricks.

Orchestrates the complete Stage 2 workflow:
1. Flow-level feature engineering (raw CSVs → engineered_flows.parquet)
2. Device-level aggregation (per device_ip → device_features.parquet)
3. Train/val/test splitting (stratified 70/10/20 splits)
4. Graph building (heterogeneous graph for GNN models)
5. Quality validation (comprehensive validation report)

Usage:
    Local: python analysis/stage2/run_stage2.py
    Databricks: %run ./analysis/stage2/run_stage2
"""

import sys
import time

from config import Config


def main():
    """Run complete Stage 2 pipeline."""

    print("\n" + "=" * 70)
    print("🚀 STAGE 2: FEATURE ENGINEERING & DATA PREPARATION")
    print("=" * 70)
    print(f"Environment: {Config.get_environment()}")
    print(f"Mode: {'TEST' if Config.TEST_MODE else 'PRODUCTION'}")
    print(f"Input data dir:   {Config.RAW_DIR_ORIGINAL}")
    print(f"Output base dir:  {Config.STAGE2_PREPARED_DIR}")
    print("=" * 70)

    overall_start = time.time()

    # Ensure output directories exist
    Config.ensure_output_dirs()

    # Get Spark session (reuses existing session on Databricks)
    spark = Config.get_spark_session("Stage2-Pipeline")

    # =========================================================================
    # STEP 1: FLOW-LEVEL FEATURE ENGINEERING
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 1/5: FEATURE ENGINEERING (flows → features)")
    print("=" * 70)

    try:
        from analysis.stage2.feature_engineering import build_engineered_dataset_spark
        build_engineered_dataset_spark(spark)
        print("✅ Feature engineering complete!")
    except Exception as e:
        print(f"❌ ERROR in feature engineering: {e}")
        import traceback
        traceback.print_exc()

        if not Config.is_databricks():
            spark.stop()
        sys.exit(1)

    # =========================================================================
    # STEP 2: DEVICE-LEVEL AGGREGATION
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 2/5: DEVICE AGGREGATION (per device_ip)")
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

    # =========================================================================
    # STEP 3: TRAIN/VAL/TEST SPLITTING
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 3/5: TRAIN/VAL/TEST SPLITTING (70/10/20)")
    print("=" * 70)

    try:
        from analysis.stage2.train_test_split import create_and_save_master_splits_spark
        create_and_save_master_splits_spark(spark)
        print("✅ Data splitting complete!")
    except Exception as e:
        print(f"❌ ERROR in data splitting: {e}")
        import traceback
        traceback.print_exc()
        # Splits are critical for downstream training, so stop hard

        if not Config.is_databricks():
            spark.stop()
        sys.exit(1)

    # =========================================================================
    # STEP 4: GRAPH BUILDING (Optional - for GNN models)
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 4/5: GRAPH BUILDING (heterogeneous graph)")
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

    # =========================================================================
    # STEP 5: QUALITY VALIDATION
    # =========================================================================

    print("\n" + "=" * 70)
    print("STEP 5/5: QUALITY VALIDATION")
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

    # =========================================================================
    # SUMMARY
    # =========================================================================

    total_time = time.time() - overall_start

    print("\n" + "=" * 70)
    print("🎉 STAGE 2 COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f"🌍 Environment: {Config.get_environment()}")
    print(f"📁 Outputs saved under: {Config.STAGE2_PREPARED_DIR}")

    print("\n📊 Artifacts:")
    print(f"   • Flow features:        {Config.ENGINEERED_DATA_PATH}")
    print(f"   • Device features:      {Config.DEVICE_FEATURES_PATH}")
    print(f"   • Train split:          {Config.TRAIN_PATH}")
    print(f"   • Val split:            {Config.VAL_PATH}")
    print(f"   • Test split:           {Config.TEST_PATH}")
    print(f"   • Graph:                {Config.HETERO_GRAPH_PATH}")
    print(f"   • Validation report:    {Config.STAGE2_QUALITY_DIR}/stage2_validation_report.json")

    print("\n🚀 Ready for Stage 3: Model training & evaluation")
    print("   (Run model training on your GPU cluster/RunPod)")
    print("=" * 70)

    # Only stop Spark on local (Databricks manages cluster)
    if not Config.is_databricks():
        spark.stop()
        print("🧹 Spark session closed.")


if __name__ == "__main__":
    main()
