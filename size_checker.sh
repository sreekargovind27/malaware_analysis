#!/bin/bash

# Bash Script to check the Size of our dataset by iterating over folders and returning the size for each file.

# --- Configuration ---
DATA_DIR="iot23_csv_data"
OUTPUT_FILE="size_report.txt"

# --- Main Execution ---

# Check if the target directory exists
if [ ! -d "$DATA_DIR" ]; then
    echo "Error: Directory '$DATA_DIR' not found."
    echo "Please make sure you are in the correct directory and have run the download script."
    exit 1
fi

echo "Calculating disk usage for all scenarios in '$DATA_DIR'..."

# We redirect (>) all of this output into our specified text file.
du -sch "$DATA_DIR"/*/ > "$OUTPUT_FILE"

# Check if the command was successful
if [ $? -eq 0 ]; then
    echo "Success! Report has been saved to '$OUTPUT_FILE'."
    echo ""
    echo "--- Report Preview ---"
    # Display the generated report on the screen for convenience
    cat "$OUTPUT_FILE"
    echo "--------------------"
else
    echo "An error occurred while calculating disk usage."
fi