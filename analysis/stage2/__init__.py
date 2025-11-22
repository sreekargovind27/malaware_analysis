"""
Stage 2: Feature Engineering & Data Preparation

This package contains PySpark-based modules for feature engineering and
data preparation. All modules work on both local and Databricks environments
with automatic optimization for 40GB+ datasets.

Modules:
- feature_engineering: Flow-level feature engineering (raw CSVs → engineered_flows.parquet)
- device_aggregation: Device-level aggregation (per device_ip → device_features.parquet)
- train_test_split: Stratified train/val/test splitting (70/10/20)
- build_graph: Heterogeneous graph construction for GNN models
- stage2_quality_report: Comprehensive validation of Stage 2 outputs
- run_stage2: Main orchestrator for running complete Stage 2 pipeline
- utils: Helper functions for loading data, repartitioning, saving outputs

Key Features:
- Works on local and Databricks (same code)
- Optimized for 40GB+ IoT-23 dataset
- Smart repartitioning (200 partitions on Databricks)
- Persist with MEMORY_AND_DISK strategy
- Optimized parquet writes with compression
- Collect() guards to prevent OOM
- Auto-detects environment (local vs Databricks)

Pipeline:
1. Feature Engineering (raw CSVs → engineered_flows.parquet)
2. Device Aggregation (per device_ip → device_features.parquet)
3. Train/Val/Test Split (stratified 70/10/20)
4. Graph Building (heterogeneous graph for GNN)
5. Quality Validation (comprehensive checks)

Usage:
    Local:
        python analysis/stage2/run_stage2.py

    Databricks:
        %run ./analysis/stage2/run_stage2

    Or import:
        from analysis.stage2.run_stage2 import main
        main()
"""

__version__ = "2.0.0"
__author__ = "Arun Ghontale, Nidhi Rajani, Sreekar Garimella"

# Expose main runner for easy import
from .run_stage2 import main as run_stage2

__all__ = [
    'run_stage2',
]