"""
RUN THIS SCRIPT ONCE.

This script creates the master train and test data splits for the entire project.
It loads the main 50M row engineered dataset, performs a single stratified split,
and saves the resulting train_set.parquet (40M rows) and test_set.parquet (10M rows)
to disk.

All other training scripts will load these pre-split files, ensuring
perfect reproducibility.
"""
import os

from sklearn.model_selection import train_test_split

from config import Config
from data_loader import load_engineered_data


def create_and_save_master_splits():
    print("=" * 70)
    print("🚀 CREATING AND SAVING MASTER TRAIN/TEST SPLITS")
    print("=" * 70)

    # 1. Define the output directory
    output_dir = 'data/master_splits'
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Output directory: {output_dir}")

    # 2. Load the full 50M row dataset
    df = load_engineered_data()

    # 3. Perform the single, master train-test split
    # We stratify on the binary 'label' column to ensure both sets
    # have a representative amount of malicious vs. benign traffic.
    print(f"\n⏳ Performing 80/20 stratified split on {len(df):,} rows...")

    train_df, test_df = train_test_split(
        df,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=df[Config.TARGET_COL]
    )
    print("✓ Split complete.")

    # 4. Save the splits to disk as Parquet files
    train_path = os.path.join(output_dir, 'train_set.parquet')
    test_path = os.path.join(output_dir, 'test_set.parquet')

    print(f"\n💾 Saving training set ({len(train_df):,} rows) to {train_path}...")
    train_df.to_parquet(train_path, index=False)

    print(f"💾 Saving test set ({len(test_df):,} rows) to {test_path}...")
    test_df.to_parquet(test_path, index=False)

    print("\n" + "=" * 70)
    print("🎉 MASTER SPLITS CREATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    create_and_save_master_splits()
