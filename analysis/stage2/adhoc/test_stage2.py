#!/usr/bin/env python3
"""
Test Stage 2 PySpark implementation with incremental data sizes
"""
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import Config


def test_stage2(sample_fraction=0.01):
    """Test stage2 with specified sample fraction."""

    print("\n" + "=" * 70)
    print(f"🧪 TESTING STAGE 2 with {sample_fraction * 100:.1f}% of data")
    print("=" * 70)

    # Temporarily override config
    original_fraction = Config.DATA_SAMPLE_FRACTION
    Config.DATA_SAMPLE_FRACTION = sample_fraction

    try:
        # Run stage2
        from analysis.stage2.run_stage2 import main
        main()

        # Verify outputs
        print("\n" + "=" * 70)
        print("✅ VERIFICATION")
        print("=" * 70)

        spark = Config.get_spark_session("Verification")

        # Check engineered data
        df_eng = spark.read.parquet(Config.ENGINEERED_DATA_PATH)
        eng_count = df_eng.count()
        print(f"✓ Engineered data: {eng_count:,} rows")

        # Check device data
        df_dev = spark.read.parquet(Config.DEVICE_FEATURES_PATH)
        dev_count = df_dev.count()
        print(f"✓ Device data: {dev_count:,} devices")

        # Check splits
        df_train = spark.read.parquet(Config.TRAIN_SET_PATH)
        df_test = spark.read.parquet(Config.TEST_SET_PATH)
        train_count = df_train.count()
        test_count = df_test.count()

        print(f"✓ Train set: {train_count:,} rows")
        print(f"✓ Test set: {test_count:,} rows")

        # Verify features
        import joblib
        features = joblib.load(Config.FEATURE_LIST_PATH)
        print(f"✓ Feature list: {len(features)} features saved")

        print("\n✅ ALL CHECKS PASSED!")

        # Stop spark
        spark.stop()

    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        # Restore config
        Config.DATA_SAMPLE_FRACTION = original_fraction


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Test Stage 2 PySpark implementation')
    parser.add_argument('--sample', type=float, default=0.01,
                        help='Sample fraction (0.01 = 1%%, 0.1 = 10%%, 1.0 = 100%%)')

    args = parser.parse_args()

    print("\n🧪 Stage 2 PySpark Test Suite")
    print("=" * 70)
    print(f"Sample fraction: {args.sample * 100:.1f}%")
    print("=" * 70)

    test_stage2(args.sample)
