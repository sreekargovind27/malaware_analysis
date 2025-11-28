"""
Stage 2: Train/Val/Test Splitting - PySpark Version
Works on both local and Databricks with automatic optimization.

Pipeline:
1. Load engineered flow features
2. Create stratified splits (70% train, 10% val, 20% test)
3. Validate class distributions
4. Save splits as parquet files

Optimizations:
- Uses Spark's native stratified sampling (sampleBy)
- Smart repartitioning for efficient writes
- Validation of class balance
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F

from config import Config
from analysis.stage2.utils import (
    load_engineered_data_spark, write_parquet_optimized,
    print_environment_info, validate_dataframe
)


# ============================================================================
# STRATIFIED SPLITTING
# ============================================================================

def create_stratified_splits(df, label_col="label", train_ratio=0.7, val_ratio=0.1, test_ratio=0.2, seed=None):
    """
    Create stratified train/val/test splits using random assignment.

    Args:
        df: Spark DataFrame with features
        label_col: Column name for stratification
        train_ratio: Fraction for training set (default: 0.7)
        val_ratio: Fraction for validation set (default: 0.1)
        test_ratio: Fraction for test set (default: 0.2)
        seed: Random seed for reproducibility

    Returns:
        tuple: (train_df, val_df, test_df)
    """
    if seed is None:
        seed = Config.RANDOM_STATE

    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {train_ratio + val_ratio + test_ratio}")

    print(f"\n🔀 Creating stratified splits ({train_ratio:.0%}/{val_ratio:.0%}/{test_ratio:.0%})...")

    # Get class distribution
    label_counts = df.groupBy(label_col).count().collect()
    class_dict = {row[label_col]: row['count'] for row in label_counts}

    print(f"   Class distribution:")
    for label, count in class_dict.items():
        print(f"      {label}: {count:,} samples")

    # Add a random column for splitting (ensures no overlap)
    df_with_rand = df.withColumn("_rand_split", F.rand(seed))

    # Calculate thresholds
    train_threshold = train_ratio
    val_threshold = train_ratio + val_ratio

    # Split based on random value (guarantees no overlap)
    print(f"\n   Assigning splits based on random values...")
    
    train_df = df_with_rand.filter(F.col("_rand_split") < train_threshold).drop("_rand_split")
    val_df = df_with_rand.filter(
        (F.col("_rand_split") >= train_threshold) & 
        (F.col("_rand_split") < val_threshold)
    ).drop("_rand_split")
    test_df = df_with_rand.filter(F.col("_rand_split") >= val_threshold).drop("_rand_split")

    return train_df, val_df, test_df

# ============================================================================
# SPLIT VALIDATION
# ============================================================================

def validate_splits(train_df, val_df, test_df, label_col="label"):
    """
    Validate that splits have reasonable class distributions.

    Args:
        train_df: Training set
        val_df: Validation set
        test_df: Test set
        label_col: Label column name
    """
    print("\n🔍 Validating splits...")

    # Count each split
    train_count = train_df.count()
    val_count = val_df.count()
    test_count = test_df.count()
    total = train_count + val_count + test_count

    print(f"\n📊 Split sizes:")
    print(f"   Train: {train_count:,} ({train_count / total:.1%})")
    print(f"   Val:   {val_count:,} ({val_count / total:.1%})")
    print(f"   Test:  {test_count:,} ({test_count / total:.1%})")
    print(f"   Total: {total:,}")

    # Check class distributions
    print(f"\n📊 Class distributions per split:")

    for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        label_dist = split_df.groupBy(label_col).count().collect()
        split_total = split_df.count()

        print(f"\n   {split_name}:")
        for row in sorted(label_dist, key=lambda x: x['count'], reverse=True):
            label = row[label_col]
            count = row['count']
            pct = (count / split_total * 100) if split_total > 0 else 0
            print(f"      {label}: {count:,} ({pct:.1f}%)")

    # Check for data leakage (overlapping rows)
    print(f"\n🔍 Checking for data leakage...")

    train_val_overlap = train_df.intersect(val_df).count()
    train_test_overlap = train_df.intersect(test_df).count()
    val_test_overlap = val_df.intersect(test_df).count()

    if train_val_overlap > 0 or train_test_overlap > 0 or val_test_overlap > 0:
        print(f"   ⚠️  WARNING: Data leakage detected!")
        print(f"      Train-Val overlap: {train_val_overlap:,}")
        print(f"      Train-Test overlap: {train_test_overlap:,}")
        print(f"      Val-Test overlap: {val_test_overlap:,}")
    else:
        print(f"   ✅ No data leakage (splits are disjoint)")

    print(f"\n✅ Split validation complete")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def create_and_save_master_splits_spark(spark):
    """
    Main train/val/test splitting pipeline.
    Loads engineered data, creates stratified splits, saves parquet files.

    Args:
        spark: Active SparkSession
    """
    print("\n" + "=" * 70)
    print("STAGE 2: TRAIN/VAL/TEST SPLITTING")
    print("=" * 70)

    print_environment_info(spark)

    # ========================================
    # 1. LOAD ENGINEERED DATA
    # ========================================

    flows = load_engineered_data_spark(spark)

    # Validate required columns
    if Config.TARGET_COL not in flows.columns:
        raise ValueError(f"Target column '{Config.TARGET_COL}' not found in data")

    total_rows = flows.count()
    print(f"\n📊 Input: {total_rows:,} flows")

    # ========================================
    # 2. CREATE STRATIFIED SPLITS
    # ========================================

    train_df, val_df, test_df = create_stratified_splits(
        flows,
        label_col=Config.TARGET_COL,
        train_ratio=0.7,
        val_ratio=0.1,
        test_ratio=0.2,
        seed=Config.RANDOM_STATE
    )

    # ========================================
    # 3. VALIDATE SPLITS
    # ========================================

    validate_splits(train_df, val_df, test_df, label_col=Config.TARGET_COL)

    # ========================================
    # 4. SAVE SPLITS
    # ========================================

    print("\n💾 Saving splits to parquet...")

    # Determine partition count based on environment
    if Config.is_databricks():
        num_partitions = Config.DATABRICKS_COALESCE_PARTITIONS
    else:
        num_partitions = 4

    # Save train set
    print(f"\n   [1/3] Saving training set...")
    write_parquet_optimized(
        train_df,
        Config.TRAIN_PATH,
        mode="overwrite",
        coalesce=True
    )
    print(f"   ✅ Saved to {Config.TRAIN_PATH}")

    # Save validation set
    print(f"\n   [2/3] Saving validation set...")
    write_parquet_optimized(
        val_df,
        Config.VAL_PATH,
        mode="overwrite",
        coalesce=True
    )
    print(f"   ✅ Saved to {Config.VAL_PATH}")

    # Save test set
    print(f"\n   [3/3] Saving test set...")
    write_parquet_optimized(
        test_df,
        Config.TEST_PATH,
        mode="overwrite",
        coalesce=True
    )
    print(f"   ✅ Saved to {Config.TEST_PATH}")

    # ========================================
    # 5. FINAL SUMMARY
    # ========================================

    print("\n" + "=" * 70)
    print("✅ TRAIN/VAL/TEST SPLITTING COMPLETE!")
    print("=" * 70)

    train_count = train_df.count()
    val_count = val_df.count()
    test_count = test_df.count()

    print(f"\n📊 Split Summary:")
    print(f"   Training:   {train_count:,} samples ({train_count / total_rows:.1%})")
    print(f"   Validation: {val_count:,} samples ({val_count / total_rows:.1%})")
    print(f"   Test:       {test_count:,} samples ({test_count / total_rows:.1%})")
    print(f"   Total:      {total_rows:,} samples")

    print(f"\n📂 Output Files:")
    print(f"   Train: {Config.TRAIN_PATH}")
    print(f"   Val:   {Config.VAL_PATH}")
    print(f"   Test:  {Config.TEST_PATH}")

    print("\n" + "=" * 70)

    return train_df, val_df, test_df


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    Config.ensure_output_dirs()
    spark = Config.get_spark_session("Stage2-TrainValTestSplit")

    try:
        create_and_save_master_splits_spark(spark)
    finally:
        if not Config.is_databricks():
            spark.stop()
            print("🧹 Spark session stopped")
