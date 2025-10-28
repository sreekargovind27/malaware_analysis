#!/bin/bash
# IoT-23 ML Pipeline Runner with Logging
# Usage: bash run_pipeline.sh

set -e
LOGS_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
mkdir -p "$LOGS_DIR"

echo "========================================================================"
echo "🚀 STARTING IoT-23 ML PIPELINE - $TIMESTAMP"
echo "========================================================================"

run_script() {
    local script=$1
    local name=$2
    local log_file="$LOGS_DIR/${name}_${TIMESTAMP}.log"

    echo ""
    echo "▶️  $name - Started: $(date +"%H:%M:%S")"
    start=$(date +%s)

    if python "$script" 2>&1 | tee "$log_file"; then
        end=$(date +%s)
        duration=$((end - start))
        echo "✅ $name - Completed in ${duration}s ($(($duration/60))m $(($duration%60))s)"
        echo "$name: ${duration}s" >> "$LOGS_DIR/timing_${TIMESTAMP}.txt"
    else
        end=$(date +%s)
        duration=$((end - start))
        echo "❌ $name - Failed after ${duration}s"
        exit 1
    fi
}

pipeline_start=$(date +%s)

# Run all scripts
run_script "build_dataset.py" "01_build_dataset"
run_script "binary_classifier.py" "02_binary_classifier"
run_script "multiclass_classifier.py" "03_multiclass_classifier"
run_script "clustering.py" "04_clustering"
run_script "virus_classifier.py" "05_virus_classifier"
run_script "autoencoder.py" "06_autoencoder"

pipeline_end=$(date +%s)
total_duration=$((pipeline_end - pipeline_start))

echo ""
echo "========================================================================"
echo "🎉 PIPELINE COMPLETE"
echo "========================================================================"
echo "⏱️  Total Time: ${total_duration}s ($(($total_duration/60))m $(($total_duration%60))s)"
echo "📁 Logs saved in: $LOGS_DIR"
echo "📊 Timing summary: $LOGS_DIR/timing_${TIMESTAMP}.txt"
echo "========================================================================"

# Save final summary
echo "PIPELINE COMPLETE - Total: ${total_duration}s" >> "$LOGS_DIR/timing_${TIMESTAMP}.txt"