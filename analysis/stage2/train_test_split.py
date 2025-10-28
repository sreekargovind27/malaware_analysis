"""
This script creates and saves the master train and test data splits for the project.
It performs a single, stratified split on the main engineered dataset and saves the
resulting files to disk. This ensures that all subsequent training scripts use the
same data splits for reproducibility.
"""
import os

from sklearn.model_selection import train_test_split

from config import Config
from models.data_loader import load_engineered_data


def create_and_save_master_splits():
    print("=" * 70)
    print("🚀 CREATING AND SAVING MASTER TRAIN/TEST SPLITS")
    print("=" * 70)

    # Define the output directory and create it if it doesn't exist.
    output_dir = Config.SPLITS_DIR
    os.makedirs(output_dir, exist_ok=True)
    print(f"   Output directory: {output_dir}")

    # Load the full engineered dataset.
    df = load_engineered_data()

    # Perform a stratified train-test split to ensure representative class distribution.
    print(f"\n⏳ Performing 80/20 stratified split on {len(df):,} rows...")

    train_df, test_df = train_test_split(
        df,
        test_size=Config.TEST_SIZE,
        random_state=Config.RANDOM_STATE,
        stratify=df[Config.TARGET_COL]
    )
    print("✓ Split complete.")

    # Save the resulting train and test sets as Parquet files.
    train_path = Config.TRAIN_SET_PATH
    test_path = Config.TEST_SET_PATH

    print(f"\n💾 Saving training set ({len(train_df):,} rows) to {train_path}...")
    train_df.to_parquet(train_path, index=False)

    print(f"💾 Saving test set ({len(test_df):,} rows) to {test_path}...")
    test_df.to_parquet(test_path, index=False)

    print("\n" + "=" * 70)
    print("🎉 MASTER SPLITS CREATED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    create_and_save_master_splits()
