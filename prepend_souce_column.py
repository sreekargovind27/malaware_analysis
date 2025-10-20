"""
A utility script to preprocess raw CSV data by adding a 'Source_Folder' column.

This script iterates through CSV files in a specified source directory,
adds a new column at the beginning of each file containing the original filename,
and saves the modified file to a destination directory.
"""

import os

import pandas as pd
from tqdm import tqdm

# Configure the source and destination paths for the data files.
SOURCE_DIRECTORY = 'data/raw_medium/'
DESTINATION_DIRECTORY = 'data/raw/'


def add_source_folder_column():
    """Reads files, adds the source column, and saves the modified versions."""

    print("=" * 60)
    print("Fixing Raw CSVs: Adding 'Source_Folder' Column")
    print("=" * 60)

    os.makedirs(DESTINATION_DIRECTORY, exist_ok=True)

    try:
        csv_files = [f for f in os.listdir(SOURCE_DIRECTORY) if f.endswith('.csv')]
        if not csv_files:
            print(f"Error: No CSV files found in '{SOURCE_DIRECTORY}'.")
            return
    except FileNotFoundError:
        print(f"Error: The source directory '{SOURCE_DIRECTORY}' does not exist.")
        return

    print(f"Found {len(csv_files)} files to process in '{SOURCE_DIRECTORY}'.")
    print(f"Corrected files will be saved in '{DESTINATION_DIRECTORY}'.\n")

    for filename in tqdm(csv_files, desc="Processing files"):
        source_path = os.path.join(SOURCE_DIRECTORY, filename)
        destination_path = os.path.join(DESTINATION_DIRECTORY, filename)

        try:
            df = pd.read_csv(source_path)

            # Use the filename (without extension) as the value for the new column.
            source_folder_value = filename.replace('.csv', '')

            # Insert the new column at the beginning of the DataFrame.
            df.insert(0, 'Source_Folder', source_folder_value)

            df.to_csv(destination_path, index=False)

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue

    print("\n" + "=" * 60)
    print("✓ All files successfully converted.")
    print("=" * 60)


if __name__ == "__main__":
    add_source_folder_column()
