#!/bin/bash

# IoT-23 Dataset Downloader & CSV Converter
# - Outputs flat CSV files into data/raw/
# - Filename = scenario name (e.g., CTU-IoT-Malware-Capture-34-1.csv)
# - Includes Source_Folder column and trims whitespace

# --- Configuration ---
BASE_URL="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios"
OUTPUT_DIR="data/raw"  # <-- Changed to your desired path
MAX_JOBS=4

# Exact list of 20 scenarios
SCENARIOS=(
    "CTU-IoT-Malware-Capture-34-1"
    "CTU-IoT-Malware-Capture-43-1"
    "CTU-IoT-Malware-Capture-44-1"
    "CTU-IoT-Malware-Capture-49-1"
    "CTU-IoT-Malware-Capture-52-1"
    "CTU-IoT-Malware-Capture-20-1"
    "CTU-IoT-Malware-Capture-21-1"
    "CTU-IoT-Malware-Capture-42-1"
    "CTU-IoT-Malware-Capture-60-1"
    "CTU-IoT-Malware-Capture-17-1"
    "CTU-IoT-Malware-Capture-36-1"
    "CTU-IoT-Malware-Capture-33-1"
    "CTU-IoT-Malware-Capture-8-1"
    "CTU-IoT-Malware-Capture-35-1"
    "CTU-IoT-Malware-Capture-48-1"
    "CTU-IoT-Malware-Capture-39-1"
    "CTU-IoT-Malware-Capture-7-1"
    "CTU-IoT-Malware-Capture-9-1"
    "CTU-IoT-Malware-Capture-3-1"
    "CTU-IoT-Malware-Capture-1-1"
)

# Function to process one scenario
process_scenario() {
    local SCENARIO="$1"
    local BASE_URL="$2"
    local OUTPUT_DIR="$3"

    local CSV_PATH="$OUTPUT_DIR/${SCENARIO}.csv"

    if [[ -f "$CSV_PATH" ]]; then
        echo "[SKIP] $SCENARIO: already processed."
        return 0
    fi

    local LOG_URL="$BASE_URL/$SCENARIO/bro/conn.log.labeled"
    local TEMP_LOG="$OUTPUT_DIR/${SCENARIO}.tmp"

    echo "[START] $SCENARIO: downloading..."
    if wget -q --timeout=60 --tries=3 -O "$TEMP_LOG" "$LOG_URL"; then
        if [[ ! -s "$TEMP_LOG" ]]; then
            echo "[ERROR] $SCENARIO: downloaded file is empty."
            rm -f "$TEMP_LOG"
            return 1
        fi

        echo "[CONVERT] $SCENARIO: adding Source_Folder and trimming whitespace..."

        # Full header with Source_Folder
        HEADER="Source_Folder,ts,uid,id.orig_h,id.orig_p,id.resp_h,id.resp_p,proto,service,duration,orig_bytes,resp_bytes,conn_state,local_orig,local_resp,missed_bytes,history,orig_pkts,orig_ip_bytes,resp_pkts,resp_ip_bytes,label,detailed-label"
        echo "$HEADER" > "$CSV_PATH"

        # Process non-comment lines: prepend scenario, trim trailing space, convert tabs to commas
        grep -v '^#' "$TEMP_LOG" | while IFS= read -r line; do
            echo "$SCENARIO,$line" | sed 's/[[:space:]]*$//' | tr '\t' ','
        done >> "$CSV_PATH"

        rm -f "$TEMP_LOG"
        echo "[DONE] $SCENARIO"
    else
        echo "[ERROR] $SCENARIO: download failed."
        rm -f "$TEMP_LOG"
    fi
}

# Export for subshells
export -f process_scenario
export BASE_URL OUTPUT_DIR

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

echo "=== Starting IoT-23 CSV Conversion (output: $OUTPUT_DIR) ==="

if command -v parallel >/dev/null 2>&1; then
    printf '%s\n' "${SCENARIOS[@]}" | parallel -j "$MAX_JOBS" --line-buffer process_scenario {} "$BASE_URL" "$OUTPUT_DIR"
else
    JOB_COUNT=0
    for SCENARIO in "${SCENARIOS[@]}"; do
        while (( JOB_COUNT >= MAX_JOBS )); do
            sleep 1
            JOB_COUNT=$(jobs -r | wc -l)
        done
        process_scenario "$SCENARIO" "$BASE_URL" "$OUTPUT_DIR" &
        ((JOB_COUNT++))
    done
    wait
fi

echo ""
echo "✅ All scenarios processed!"
echo "📁 CSV files saved in: $OUTPUT_DIR/"

# --- OPTIONAL: Merge all CSVs ---
read -p "Do you want to merge all CSVs into a single file? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    MERGED_FILE="$OUTPUT_DIR/iot23_all_merged.csv"
    echo "Merging all CSVs into $MERGED_FILE..."

    head -n1 "$OUTPUT_DIR/${SCENARIOS[0]}.csv" > "$MERGED_FILE"
    for f in "$OUTPUT_DIR"/*.csv; do
        [[ "$f" == "$MERGED_FILE" ]] && continue
        tail -n +2 "$f" >> "$MERGED_FILE"
    done

    echo "✅ Merged file created: $MERGED_FILE"
fi