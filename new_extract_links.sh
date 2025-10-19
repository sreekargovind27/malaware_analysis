#!/bin/bash

# IoT-23 Dataset Downloader & CSV Converter - OPTIMIZED VERSION
# - Uses aria2c for faster multi-connection downloads
# - Parallel processing with increased jobs
# - Optimized awk processing
# - Includes all 23 scenarios (20 malicious + 3 benign)

# --- Configuration ---
BASE_URL="https://mcfp.felk.cvut.cz/publicDatasets/IoT-23-Dataset/IndividualScenarios"
OUTPUT_DIR="data/raw"
MAX_JOBS=12  # Increased for better parallelism

# All 23 scenarios (20 malicious + 3 benign)
SCENARIOS=(
    # Malicious scenarios
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
    # Benign scenarios
    "CTU-Honeypot-Capture-7-1"
    "CTU-Honeypot-Capture-4-1"
    "CTU-Honeypot-Capture-5-1"
)

# Function to process one scenario
process_scenario() {
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
    echo "[START] $SCENARIO"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Check if aria2c is available, fallback to wget
    if command -v aria2c >/dev/null 2>&1; then
        # Use aria2c with multiple connections for faster download
        if aria2c \
            -x 16 -s 16 \
            --max-tries=5 \
            --retry-wait=3 \
            --connect-timeout=60 \
            --timeout=180 \
            --allow-overwrite=true \
            --auto-file-renaming=false \
            --console-log-level=warn \
            -d "$OUTPUT_DIR" \
            -o "${SCENARIO}.tmp" \
            "$LOG_URL" 2>&1 | grep -v "^$"; then

            DOWNLOAD_SUCCESS=true
        else
            DOWNLOAD_SUCCESS=false
        fi
    else
        # Fallback to wget if aria2c not available
        if wget --progress=bar:force:noscroll \
                -c \
                --timeout=180 \
                --tries=10 \
                -O "$TEMP_LOG" \
                "$LOG_URL" 2>&1 | \
                grep --line-buffered -E '^[0-9]+%|saved'; then

            DOWNLOAD_SUCCESS=true
        else
            DOWNLOAD_SUCCESS=false
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
        rm -f "$TEMP_LOG"
        echo ""
        return 1
    fi
}

# Export for subshells
export -f process_scenario
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

echo "╔════════════════════════════════════════════════════╗"
echo "║  IoT-23 Dataset Downloader - OPTIMIZED            ║"
echo "║  Output: $OUTPUT_DIR                              ║"
echo "║  Parallel Jobs: $MAX_JOBS                          ║"
echo "║  Total Scenarios: ${#SCENARIOS[@]}                 ║"
echo "╚════════════════════════════════════════════════════╝"
echo ""

# Use GNU parallel if available, otherwise fallback to manual parallelism
if command -v parallel >/dev/null 2>&1; then
    echo "✓ Using GNU parallel for job management"
    printf '%s\n' "${SCENARIOS[@]}" | parallel -j "$MAX_JOBS" --bar --line-buffer process_scenario {} "$BASE_URL" "$OUTPUT_DIR"
else
    echo "⚠ GNU parallel not found - using manual job control"
    echo "  Install for better progress tracking:"
    echo "    Ubuntu/Debian: sudo apt-get install parallel"
    echo "    macOS: brew install parallel"
    echo ""

    JOB_COUNT=0
    for SCENARIO in "${SCENARIOS[@]}"; do
        while (( JOB_COUNT >= MAX_JOBS )); do
            sleep 0.5
            JOB_COUNT=$(jobs -r | wc -l)
        done
        process_scenario "$SCENARIO" "$BASE_URL" "$OUTPUT_DIR" &
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
echo "Successfully downloaded: $SUCCESS_COUNT / ${#SCENARIOS[@]} scenarios"
echo ""

# List any failed scenarios
if (( SUCCESS_COUNT < ${#SCENARIOS[@]} )); then
    echo "⚠ Failed scenarios:"
    for SCENARIO in "${SCENARIOS[@]}"; do
        if [[ ! -f "$OUTPUT_DIR/${SCENARIO}.csv" ]]; then
            echo "  - $SCENARIO"
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