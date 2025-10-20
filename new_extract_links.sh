#!/bin/bash

# IoT-23 Dataset Downloader & CSV Converter - FULLY FIXED VERSION
# - Handles all different directory structures
# - Uses aria2c for faster multi-connection downloads
# - Parallel processing with increased jobs

# --- Configuration ---
BASE_URL="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios"
OUTPUT_DIR="data/raw"
MAX_JOBS=12

# Standard malicious scenarios (use /bro/ path)
STANDARD_SCENARIOS=(
    "CTU-IoT-Malware-Capture-34-1"
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

# Special case: CTU-43-1 uses /labeled/ instead of /bro/
LABELED_SCENARIOS=(
    "CTU-IoT-Malware-Capture-43-1"
)

# Benign scenarios with their specific subdirectories
declare -A BENIGN_SCENARIOS
BENIGN_SCENARIOS["CTU-Honeypot-Capture-7-1"]="Somfy-01"
BENIGN_SCENARIOS["CTU-Honeypot-Capture-4-1"]="philips-hue"
BENIGN_SCENARIOS["CTU-Honeypot-Capture-5-1"]="amazon-echo"

# Function to process standard malicious scenarios (/bro/ path)
process_standard_scenario() {
    local SCENARIO="$1"
    local BASE_URL="$2"
    local OUTPUT_DIR="$3"

    local CSV_PATH="$OUTPUT_DIR/${SCENARIO}.csv"

    if [[ -f "$CSV_PATH" ]]; then
        echo "[SKIP] $SCENARIO"
        return 0
    fi

    local LOG_URL="$BASE_URL/$SCENARIO/bro/conn.log.labeled"
    local TEMP_LOG="$OUTPUT_DIR/${SCENARIO}.tmp"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[START] $SCENARIO (standard path)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    download_and_process "$SCENARIO" "$LOG_URL" "$TEMP_LOG" "$CSV_PATH"
}

# Function to process labeled scenarios (/labeled/ path)
process_labeled_scenario() {
    local SCENARIO="$1"
    local BASE_URL="$2"
    local OUTPUT_DIR="$3"

    local CSV_PATH="$OUTPUT_DIR/${SCENARIO}.csv"

    if [[ -f "$CSV_PATH" ]]; then
        echo "[SKIP] $SCENARIO"
        return 0
    fi

    local LOG_URL="$BASE_URL/$SCENARIO/labeled/conn.log.labeled"
    local TEMP_LOG="$OUTPUT_DIR/${SCENARIO}.tmp"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[START] $SCENARIO (labeled path)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    download_and_process "$SCENARIO" "$LOG_URL" "$TEMP_LOG" "$CSV_PATH"
}

# Function to process benign scenarios
process_benign_scenario() {
    local SCENARIO="$1"
    local SUBDIR="$2"
    local BASE_URL="$3"
    local OUTPUT_DIR="$4"

    local CSV_PATH="$OUTPUT_DIR/${SCENARIO}.csv"

    if [[ -f "$CSV_PATH" ]]; then
        echo "[SKIP] $SCENARIO"
        return 0
    fi

    local LOG_URL="$BASE_URL/$SCENARIO/$SUBDIR/bro/conn.log.labeled"
    local TEMP_LOG="$OUTPUT_DIR/${SCENARIO}.tmp"

    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[START] $SCENARIO (benign - $SUBDIR)"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    download_and_process "$SCENARIO" "$LOG_URL" "$TEMP_LOG" "$CSV_PATH"
}

# Common download and processing function
download_and_process() {
    local SCENARIO="$1"
    local LOG_URL="$2"
    local TEMP_LOG="$3"
    local CSV_PATH="$4"

    DOWNLOAD_SUCCESS=false

    # Check if aria2c is available, fallback to wget
    if command -v aria2c >/dev/null 2>&1; then
        # Use aria2c with multiple connections for faster download
        if aria2c \
            -x 16 -s 16 \
            --max-tries=5 \
            --retry-wait=3 \
            --connect-timeout=60 \
            --timeout=300 \
            --allow-overwrite=true \
            --auto-file-renaming=false \
            --console-log-level=warn \
            -o "$(basename "$TEMP_LOG")" \
            -d "$(dirname "$TEMP_LOG")" \
            "$LOG_URL" 2>&1 | grep -v "^$"; then

            DOWNLOAD_SUCCESS=true
        fi
    else
        # Fallback to wget if aria2c not available
        if wget --progress=bar:force:noscroll \
                -c \
                --timeout=300 \
                --tries=10 \
                -O "$TEMP_LOG" \
                "$LOG_URL" 2>&1 | \
                grep --line-buffered -E '^[0-9]+%|saved'; then

            DOWNLOAD_SUCCESS=true
        fi
    fi

    if [[ "$DOWNLOAD_SUCCESS" == true ]]; then
        if [[ ! -s "$TEMP_LOG" ]]; then
            echo "[ERROR] $SCENARIO: downloaded file is empty."
            rm -f "$TEMP_LOG"
            return 1
        fi

        echo "[CONVERT] $SCENARIO: processing to CSV..."

        # Optimized awk processing (single pass)
        HEADER="Source_Folder,ts,uid,id.orig_h,id.orig_p,id.resp_h,id.resp_p,proto,service,duration,orig_bytes,resp_bytes,conn_state,local_orig,local_resp,missed_bytes,history,orig_pkts,orig_ip_bytes,resp_pkts,resp_ip_bytes,label,detailed-label"

        awk -v scenario="$SCENARIO" -v header="$HEADER" '
            BEGIN {
                print header
                OFS = ","
            }
            /^#/ { next }
            {
                sub(/[[:space:]]+$/, "")  # trim trailing whitespace
                gsub(/\t/, ",")            # tabs to commas
                print scenario "," $0
            }
        ' "$TEMP_LOG" > "$CSV_PATH"

        rm -f "$TEMP_LOG"

        local FILE_SIZE=$(du -h "$CSV_PATH" | cut -f1)
        echo "✅ [DONE] $SCENARIO (${FILE_SIZE})"
        echo ""
    else
        echo "[ERROR] $SCENARIO: download failed."
        echo "URL tried: $LOG_URL"
        rm -f "$TEMP_LOG"
        echo ""
        return 1
    fi
}

# Export for subshells
export -f process_standard_scenario
export -f process_labeled_scenario
export -f process_benign_scenario
export -f download_and_process
export BASE_URL OUTPUT_DIR

# Ensure output directory exists
mkdir -p "$OUTPUT_DIR"

# Check for aria2c and inform user
if command -v aria2c >/dev/null 2>&1; then
    echo "✓ Using aria2c for faster downloads (16 connections per file)"
else
    echo "⚠ aria2c not found - using wget (slower)"
    echo "  Install aria2c for 2-5x faster downloads:"
    echo "    Ubuntu/Debian: sudo apt-get install aria2"
    echo "    macOS: brew install aria2"
    echo ""
fi

TOTAL_SCENARIOS=$((${#STANDARD_SCENARIOS[@]} + ${#LABELED_SCENARIOS[@]} + ${#BENIGN_SCENARIOS[@]}))

echo "╔════════════════════════════════════════════════════╗"
echo "║  IoT-23 Dataset Downloader - FULLY FIXED          ║"
echo "║  Output: $OUTPUT_DIR                              ║"
echo "║  Parallel Jobs: $MAX_JOBS                          ║"
echo "║  Total Scenarios: $TOTAL_SCENARIOS                 ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Process all scenarios
if command -v parallel >/dev/null 2>&1; then
    echo "✓ Using GNU parallel for job management"

    # Process standard scenarios
    printf '%s\n' "${STANDARD_SCENARIOS[@]}" | \
        parallel -j "$MAX_JOBS" --bar --line-buffer \
        process_standard_scenario {} "$BASE_URL" "$OUTPUT_DIR"

    # Process labeled scenarios
    printf '%s\n' "${LABELED_SCENARIOS[@]}" | \
        parallel -j "$MAX_JOBS" --bar --line-buffer \
        process_labeled_scenario {} "$BASE_URL" "$OUTPUT_DIR"

    # Process benign scenarios
    for SCENARIO in "${!BENIGN_SCENARIOS[@]}"; do
        echo "$SCENARIO|${BENIGN_SCENARIOS[$SCENARIO]}"
    done | parallel -j "$MAX_JOBS" --bar --line-buffer --colsep '|' \
        process_benign_scenario {1} {2} "$BASE_URL" "$OUTPUT_DIR"
else
    echo "⚠ GNU parallel not found - using manual job control"
    echo ""

    JOB_COUNT=0

    # Process standard scenarios
    for SCENARIO in "${STANDARD_SCENARIOS[@]}"; do
        while (( JOB_COUNT >= MAX_JOBS )); do
            sleep 0.5
            JOB_COUNT=$(jobs -r | wc -l)
        done
        process_standard_scenario "$SCENARIO" "$BASE_URL" "$OUTPUT_DIR" &
        ((JOB_COUNT++))
    done

    # Process labeled scenarios
    for SCENARIO in "${LABELED_SCENARIOS[@]}"; do
        while (( JOB_COUNT >= MAX_JOBS )); do
            sleep 0.5
            JOB_COUNT=$(jobs -r | wc -l)
        done
        process_labeled_scenario "$SCENARIO" "$BASE_URL" "$OUTPUT_DIR" &
        ((JOB_COUNT++))
    done

    # Process benign scenarios
    for SCENARIO in "${!BENIGN_SCENARIOS[@]}"; do
        while (( JOB_COUNT >= MAX_JOBS )); do
            sleep 0.5
            JOB_COUNT=$(jobs -r | wc -l)
        done
        process_benign_scenario "$SCENARIO" "${BENIGN_SCENARIOS[$SCENARIO]}" "$BASE_URL" "$OUTPUT_DIR" &
        ((JOB_COUNT++))
    done

    wait
fi

echo ""
echo "╔════════════════════════════════════════════════════╗"
echo "║  ✅ All scenarios processed!                       ║"
echo "║  📁 CSV files saved in: $OUTPUT_DIR/              ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Count successful downloads
SUCCESS_COUNT=$(ls -1 "$OUTPUT_DIR"/*.csv 2>/dev/null | wc -l)
echo "Successfully downloaded: $SUCCESS_COUNT / $TOTAL_SCENARIOS scenarios"
echo ""

# List any failed scenarios
if (( SUCCESS_COUNT < TOTAL_SCENARIOS )); then
    echo "⚠ Failed scenarios:"
    for SCENARIO in "${STANDARD_SCENARIOS[@]}" "${LABELED_SCENARIOS[@]}"; do
        if [[ ! -f "$OUTPUT_DIR/${SCENARIO}.csv" ]]; then
            echo "  - $SCENARIO"
        fi
    done
    for SCENARIO in "${!BENIGN_SCENARIOS[@]}"; do
        if [[ ! -f "$OUTPUT_DIR/${SCENARIO}.csv" ]]; then
            echo "  - $SCENARIO (benign)"
        fi
    done
    echo ""
fi

# --- OPTIONAL: Merge all CSVs ---
read -p "Do you want to merge all CSVs into a single file? (y/N): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    MERGED_FILE="$OUTPUT_DIR/iot23_all_merged.csv"
    echo "Merging $SUCCESS_COUNT CSV files into $MERGED_FILE..."

    # Find first available CSV for header
    FIRST_CSV=$(ls "$OUTPUT_DIR"/*.csv 2>/dev/null | grep -v "merged" | head -n1)

    if [[ -n "$FIRST_CSV" ]]; then
        # Write header
        head -n1 "$FIRST_CSV" > "$MERGED_FILE"

        # Append all data (skip headers)
        for f in "$OUTPUT_DIR"/*.csv; do
            [[ "$f" == "$MERGED_FILE" ]] && continue
            tail -n +2 "$f" >> "$MERGED_FILE"
        done

        MERGED_SIZE=$(du -h "$MERGED_FILE" | cut -f1)
        echo "✅ Merged file created: $MERGED_FILE ($MERGED_SIZE)"
    else
        echo "❌ No CSV files found to merge"
    fi
fi

echo ""
echo "Done! 🎉"