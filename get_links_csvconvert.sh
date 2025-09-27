#!/bin/bash

# Bash Script to get the Links of the dataset, download them and convert them to .CSV format.

# --- Configuration ---
BASE_URL="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios"
OUTPUT_DIR="iot23_csv_data"
DELAY_SECONDS=5 

# List of all 23 scenarios, it will skip completed one's
SCENARIOS=(
    "CTU-IoT-Malware-Capture-1-1" "CTU-IoT-Malware-Capture-3-1" "CTU-IoT-Malware-Capture-7-1"
    "CTU-IoT-Malware-Capture-8-1" "CTU-IoT-Malware-Capture-9-1" "CTU-IoT-Malware-Capture-17-1"
    "CTU-IoT-Malware-Capture-20-1" "CTU-IoT-Malware-Capture-21-1" "CTU-IoT-Malware-Capture-33-1"
    "CTU-IoT-Malware-Capture-34-1" "CTU-IoT-Malware-Capture-35-1" "CTU-IoT-Malware-Capture-36-1"
    "CTU-IoT-Malware-Capture-39-1" "CTU-IoT-Malware-Capture-42-1" "CTU-IoT-Malware-Capture-43-1"
    "CTU-IoT-Malware-Capture-44-1" "CTU-IoT-Malware-Capture-48-1" "CTU-IoT-Malware-Capture-49-1"
    "CTU-IoT-Malware-Capture-52-1" "CTU-IoT-Malware-Capture-60-1" "CTU-Honeypot-Capture-4-1"
    "CTU-Honeypot-Capture-5-1" "CTU-Honeypot-Capture-7-1"
)

# --- Main Execution ---
echo "--- Starting IoT-23 Dataset Download and Conversion ---"
mkdir -p "$OUTPUT_DIR"

for SCENARIO in "${SCENARIOS[@]}"; do
    echo ""
    echo "===================================================="
    echo "Processing Scenario: $SCENARIO"
    echo "===================================================="

    SCENARIO_DIR="$OUTPUT_DIR/$SCENARIO"
    mkdir -p "$SCENARIO_DIR"
    
    RAW_LOG_PATH="$SCENARIO_DIR/conn.log.labeled"
    CSV_PATH="$SCENARIO_DIR/conn.log.labeled.csv"

    # --- Automatic Resume Feature ---
    if [ -f "$CSV_PATH" ]; then
        echo "--> SUCCESS: CSV file already exists. Skipping."
        continue 
    fi

    LOG_URL="$BASE_URL/$SCENARIO/bro/conn.log.labeled"
    
    # Download the raw log file using wget
    echo "--> Downloading with wget from: $LOG_URL"
    wget -q --show-progress -O "$RAW_LOG_PATH" "$LOG_URL"

    # Check if download was successful
    if [ $? -eq 0 ] && [ -s "$RAW_LOG_PATH" ]; then
        echo "--> Download successful. Converting to CSV using low-memory method..."
        
        # LOW-MEMORY CONVERSION 
        # Define the CSV header
        HEADER="ts,uid,id.orig_h,id.orig_p,id.resp_h,id.resp_p,proto,service,duration,orig_bytes,resp_bytes,conn_state,local_orig,local_resp,missed_bytes,history,orig_pkts,orig_ip_bytes,resp_pkts,resp_ip_bytes,label,detailed-label"
        
        # Write the header to the new CSV file
        echo "$HEADER" > "$CSV_PATH"
        
        grep -v '^#' "$RAW_LOG_PATH" | tr '\t' ',' >> "$CSV_PATH"
        
        echo "--> Successfully converted to $CSV_PATH"

        # Clean up the raw downloaded file
        echo "--> Cleaning up raw log file."
        rm "$RAW_LOG_PATH"
    else
        echo "--> ERROR: Download failed for $SCENARIO. Removing empty file and skipping."
        rm -f "$RAW_LOG_PATH" 
    fi

    # Add a delay to reduce system load
    echo "--> Pausing for $DELAY_SECONDS seconds..."
    sleep $DELAY_SECONDS

done

echo ""
echo "--- All scenarios processed! ---"
echo "Your CSV files are ready in the '$OUTPUT_DIR' folder."