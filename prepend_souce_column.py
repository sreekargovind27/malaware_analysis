"""
One-time utility script to fix the raw data files.

This script reads each CSV file from a source directory (e.g., 'data/raw_medium'),
prepends a new 'Source_Folder' column populated with the filename,
and saves the corrected 23-column file to the final destination directory (e.g., 'data/raw').
"""

import os

import pandas as pd
from tqdm import tqdm

# --- CONFIGURE YOUR PATHS HERE ---
SOURCE_DIRECTORY = 'data/raw_medium/'
DESTINATION_DIRECTORY = 'data/raw/'


# -----------------------------------

def add_source_folder_column():
    """Reads files from SOURCE_DIRECTORY, adds the source column, and saves to DESTINATION_DIRECTORY."""

    print("=" * 60)
    print("Fixing Raw CSVs: Adding 'Source_Folder' Column")
    print("=" * 60)

    # Ensure the destination directory exists
    os.makedirs(DESTINATION_DIRECTORY, exist_ok=True)

    # Find all CSV files in the source directory
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
            # Read the CSV file
            df = pd.read_csv(source_path)

            # The value for the new column is the filename without the '.csv' extension
            source_folder_value = filename.replace('.csv', '')

            # Insert the new 'Source_Folder' column at the beginning (position 0)
            df.insert(0, 'Source_Folder', source_folder_value)

            # Save the corrected DataFrame to the destination
            df.to_csv(destination_path, index=False)

        except Exception as e:
            print(f"\nError processing {filename}: {e}")
            continue

    print("\n" + "=" * 60)
    print("✓ All files successfully converted to the 23-column format.")
    print("=" * 60)


if __name__ == "__main__":
    add_source_folder_column()
