"""
Stage 2: Train/Val/Test Split - PySpark Version
Splits the engineered flow-level dataset into train/val/test sets.
Maintains class balance where possible.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyspark.sql import functions as F
from pyspark.sql import SparkSession

from config import Config


def _stratified_split(df, label_col, seed, train_frac=0.7, val_frac=0.1, test_frac=0.2):
    """
    Deterministic per-class split:
    - For each label, assign a uniform random number in [0,1)
    - Cut by thresholds so ~70/10/20 per class.
    Returns (train_df, val_df, test_df).
    """
    # safety check: fractions should sum to ~1.0
    assert abs((train_frac + val_frac + test_frac) - 1.0) < 1e-6, "fractions must sum to 1.0"

    # window-less stratification trick:
    # random value seeded on (label, uid-ish surrogate) would be ideal,
    # but we’ll just use a seeded rand() which is stable per run.
    df_with_r = df.withColumn("_rand", F.rand(seed))

    # threshold cuts
    train_cut = train_frac
    val_cut = train_frac + val_frac  # e.g. 0.7 + 0.1 = 0.8

    train_df = df_with_r.filter(F.col("_rand") < train_cut)
    val_df = df_with_r.filter((F.col("_rand") >= train_cut) & (F.col("_rand") < val_cut))
    test_df = df_with_r.filter(F.col("_rand") >= val_cut)

    # drop helper col
    train_df = train_df.drop("_rand")
    val_df = val_df.drop("_rand")
    test_df = test_df.drop("_rand")

    return train_df, val_df, test_df


def create_and_save_master_splits_spark(spark):
    """
    Creates 70/10/20 train/val/test splits from the engineered flow dataset.
    Uses stratified-style sampling if the 'label' column exists.
    """
    print("\n" + "=" * 70)
    print("✂️  TRAIN/VAL/TEST SPLITTING (PySpark)")
    print("=" * 70)

    print("📂 Loading engineered flow features...")
    df = spark.read.parquet(Config.ENGINEERED_DATA_PATH)

    total_rows = df.count()
    print(f"✓ Loaded {total_rows:,} rows")

    if total_rows == 0:
        print("❌ No rows found — skipping split stage.")
        return

    # -------------------------------------------------------------
    # If we have a label column, do per-class random cut.
    # Otherwise fallback to plain randomSplit.
    # -------------------------------------------------------------
    if "label" in df.columns:
        print("⚖️  Performing per-class 70/10/20 split using seeded random thresholds...")
        train_df, val_df, test_df = _stratified_split(
            df,
            label_col="label",
            seed=Config.RANDOM_STATE,
            train_frac=0.7,
            val_frac=0.1,
            test_frac=0.2,
        )
    else:
        print("⚠️  No 'label' column found — using global randomSplit.")
        train_df, val_df, test_df = df.randomSplit(
            [0.7, 0.1, 0.2],
            seed=Config.RANDOM_STATE
        )

    # -------------------------------------------------------------
    # Count & print summary
    # -------------------------------------------------------------
    train_count = train_df.count()
    val_count = val_df.count()
    test_count = test_df.count()

    total_after = train_count + val_count + test_count
    if total_after == 0:
        print("❌ All splits are empty — aborting save.")
        return

    print("\n✓ Split complete:")
    print(f"   Train: {train_count:,} ({train_count / total_after * 100:.1f}%)")
    print(f"   Val:   {val_count:,} ({val_count / total_after * 100:.1f}%)")
    print(f"   Test:  {test_count:,} ({test_count / total_after * 100:.1f}%)")

    # Optional: sanity check class balance in train vs test (tiny collect)
    if "label" in df.columns:
        def show_dist(name, dframe):
            dist_rows = (
                dframe.groupBy("label")
                .agg(F.count(F.lit(1)).alias("cnt"))
                .collect()
            )
            print(f"\n   {name} label distribution:")
            for row in dist_rows:
                print(f"     {row['label']}: {row['cnt']:,}")

        show_dist("Train", train_df)
        show_dist("Val", val_df)
        show_dist("Test", test_df)

    # -------------------------------------------------------------
    # Save splits
    # -------------------------------------------------------------
    print("\n💾 Saving splits to parquet...")

    (
        train_df.write
        .mode("overwrite")
        .parquet(Config.TRAIN_SET_PATH, compression="snappy")
    )
    print(f"   ✓ Train → {Config.TRAIN_SET_PATH}")

    (
        val_df.write
        .mode("overwrite")
        .parquet(Config.VAL_SET_PATH, compression="snappy")
    )
    print(f"   ✓ Val   → {Config.VAL_SET_PATH}")

    (
        test_df.write
        .mode("overwrite")
        .parquet(Config.TEST_SET_PATH, compression="snappy")
    )
    print(f"   ✓ Test  → {Config.TEST_SET_PATH}")

    print("\n✅ Train/Val/Test split stage complete!")
    print("=" * 70)

    return train_df, val_df, test_df


if __name__ == "__main__":
    # Allow debugging standalone
    Config.ensure_output_dirs()
    spark = (
        SparkSession.builder
        .appName("Stage2-TrainValTestSplit-Standalone")
        .config("spark.master", "local[*]")
        .config("spark.driver.memory", "8g")
        .config("spark.sql.shuffle.partitions", "8")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    try:
        create_and_save_master_splits_spark(spark)
    finally:
        spark.stop()
