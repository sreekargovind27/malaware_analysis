"""
Stage 2: Feature Engineering & Data Preparation Pipeline

Orchestrates the complete Stage 2 workflow:
1. Feature engineering from raw CSVs
2. Device-level aggregation
3. Graph construction for GNN
4. Train/test/val splitting
5. Quality validation
"""
import sys
import time

from config import Config


def main():
    """Run complete Stage 2 pipeline."""
    print("\n" + "=" * 70)
    print("🚀 STAGE 2: FEATURE ENGINEERING & DATA PREPARATION")
    print("=" * 70)
    print(f"Mode: {'TEST' if Config.TEST_MODE else 'PRODUCTION'}")
    print(f"Input: {Config.RAW_DIR_ORIGINAL}")
    print(f"Output: {Config.STAGE2_PREPARED_DIR}")
    print("=" * 70)

    overall_start = time.time()

    # Ensure directories exist
    Config.ensure_output_dirs()

    # Step 1: Feature Engineering (Flow-level)
    print("\n" + "=" * 70)
    print("STEP 1/5: FEATURE ENGINEERING (Flow-level)")
    print("=" * 70)
    try:
        from analysis.stage2.feature_engineering import build_engineered_dataset
        build_engineered_dataset()
        print("✅ Feature engineering complete!")
    except Exception as e:
        print(f"❌ ERROR in feature engineering: {e}")
        sys.exit(1)

    # Step 2: Device Aggregation
    print("\n" + "=" * 70)
    print("STEP 2/5: DEVICE AGGREGATION")
    print("=" * 70)
    try:
        from analysis.stage2.device_aggregation import aggregate_devices
        aggregate_devices()
        print("✅ Device aggregation complete!")
    except Exception as e:
        print(f"❌ ERROR in device aggregation: {e}")
        print("⚠️  Continuing without device features (GNN will be unavailable)")

    # Step 3: Graph Construction
    print("\n" + "=" * 70)
    print("STEP 3/5: GRAPH CONSTRUCTION")
    print("=" * 70)
    try:
        from analysis.stage2.build_graph import build_heterogeneous_graph
        build_heterogeneous_graph()
        print("✅ Graph construction complete!")
    except Exception as e:
        print(f"❌ ERROR in graph construction: {e}")
        print("⚠️  Continuing without graph (GNN will be unavailable)")

    # Step 4: Train/Test Split
    print("\n" + "=" * 70)
    print("STEP 4/5: TRAIN/TEST/VAL SPLITTING")
    print("=" * 70)
    try:
        from analysis.stage2.train_test_split import create_and_save_master_splits
        create_and_save_master_splits()
        print("✅ Data splitting complete!")
    except Exception as e:
        print(f"❌ ERROR in data splitting: {e}")
        sys.exit(1)

    # Step 5: Quality Check
    print("\n" + "=" * 70)
    print("STEP 5/5: QUALITY VALIDATION")
    print("=" * 70)
    try:
        from analysis.stage2.quality_check import check_engineered_data_quality
        check_engineered_data_quality()
        print("✅ Quality check complete!")
    except Exception as e:
        print(f"❌ ERROR in quality check: {e}")
        print("⚠️  Continuing (non-critical)")

    # Step 6: Comprehensive Validation
    print("\n" + "=" * 70)
    print("STEP 6/6: COMPREHENSIVE VALIDATION")
    print("=" * 70)
    try:
        from analysis.stage2.stage2_quality_report import validate_stage2_outputs
        report = validate_stage2_outputs()
        if report['overall_status'] != 'PASS':
            print("\n⚠️  WARNING: Some validation checks failed!")
            print("   Review the report before proceeding to Stage 3.")
    except Exception as e:
        print(f"❌ ERROR in validation: {e}")
        print("⚠️  Continuing (non-critical)")

    # Summary
    total_time = time.time() - overall_start
    print("\n" + "=" * 70)
    print("🎉 STAGE 2 COMPLETE!")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print(f"📁 Outputs saved to: {Config.STAGE2_PREPARED_DIR}")
    print("\n📊 Generated files:")
    print(f"   ✓ Flow features: {Config.ENGINEERED_DATA_PATH}")
    print(f"   ✓ Device features: {Config.DEVICE_FEATURES_PATH}")
    print(f"   ✓ Train set: {Config.TRAIN_SET_PATH}")
    print(f"   ✓ Test set: {Config.TEST_SET_PATH}")
    print(f"   ✓ Heterogeneous graph: {Config.HETERO_GRAPH_PATH}")
    print("\n🚀 Ready for Stage 3: Model Training")
    print("=" * 70)


if __name__ == "__main__":
    main()
