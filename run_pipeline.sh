#!/bin/bash
# IoT-23 ML Pipeline Runner with Logging and PAIRED Parallel Model Training
# Usage: bash run_pipeline.sh

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
LOGS_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
TIMING_FILE="$LOGS_DIR/timing_${TIMESTAMP}.txt"
mkdir -p "$LOGS_DIR"

echo "========================================================================"
echo "🚀 STARTING IoT-23 ML PIPELINE - $TIMESTAMP"
echo "========================================================================"

# --- Function to run a script, log its output, and time it ---
run_script() {
    local script=$1
    local name=$2
    local log_file="$LOGS_DIR/${name}_${TIMESTAMP}.log"

    echo ""
    echo "▶️  STARTING: $name (Log: $log_file)"
    start=$(date +%s)

    # Execute the script, redirecting stdout and stderr to a log file,
    # while also showing the output on the console via 'tee'.
    if python -u "$script" 2>&1 | tee "$log_file"; then
        end=$(date +%s)
        duration=$((end - start))
        echo "✅ FINISHED: $name - Completed in ${duration}s ($(($duration/60))m $(($duration%60))s)"
        # Append timing info to a summary file
        echo "$name: ${duration}s" >> "$TIMING_FILE"
    else
        end=$(date +%s)
        duration=$((end - start))
        echo "========================================================================"
        echo "❌ FAILED: $name - The script failed after ${duration}s."
        echo "   Please check the log file for details: $log_file"
        echo "========================================================================"
        exit 1 # Exit the entire pipeline on failure
    fi
}

# --- Main Pipeline Execution ---
pipeline_start=$(date +%s)
touch "$TIMING_FILE" # Create the timing file

# ========================================================================
# STAGE 1: SEQUENTIAL DATA PREPARATION
# These scripts must run one after another.
# ========================================================================
echo ""
echo "------------------------------------------------------------------------"
echo "STAGE 1: DATA PREPARATION (Sequential)"
echo "------------------------------------------------------------------------"

run_script "build_dataset.py" "01_build_dataset"
run_script "data_check.py" "02_data_check"
run_script "create_splits.py" "03_create_splits"


# ========================================================================
# STAGE 2: PAIRED PARALLEL MODEL TRAINING
# Models are run in pairs. The script waits for each pair to finish
# before starting the next.
# ========================================================================
echo ""
echo "------------------------------------------------------------------------"
echo "STAGE 2: MODEL TRAINING (Paired Parallel Execution)"
echo "------------------------------------------------------------------------"

# --- Pair 1 ---
echo ""
echo "⏳ Launching Pair 1: Binary Classifier & Virus Classifier"
( run_script "binary_classifier.py" "04_binary_classifier" ) &
( run_script "virus_classifier.py" "05_virus_classifier" ) &
wait
echo "✅ Pair 1 finished."

# --- Pair 2 ---
echo ""
echo "⏳ Launching Pair 2: Clustering & Autoencoder"
( run_script "clustering.py" "06_clustering" ) &
( run_script "autoencoder.py" "07_autoencoder" ) &
wait
echo "✅ Pair 2 finished."

# --- Pair 3 (Final Pair) ---
echo ""
echo "⏳ Launching Pair 3: Denoising Autoencoder & Multi-Class Classifier"
( run_script "autoencoder_denoising.py" "08_autoencoder_denoising" ) &
( run_script "multiclass_classifier.py" "09_multiclass_classifier" ) &
wait
echo "✅ Pair 3 finished."


# --- Final Summary ---
pipeline_end=$(date +%s)
total_duration=$((pipeline_end - pipeline_start))

echo ""
echo "========================================================================"
echo "🎉 PIPELINE COMPLETE"
echo "========================================================================"
echo "⏱️  Total Time: ${total_duration}s ($(($total_duration/60))m $(($total_duration%60))s)"
echo "📁 Logs saved in: $LOGS_DIR"
echo "📊 Timing summary: $TIMING_FILE"
echo "========================================================================"

# Save final summary to the timing file
echo "--------------------------" >> "$TIMING_FILE"
echo "Total Pipeline Time: ${total_duration}s" >> "$TIMING_FILE"