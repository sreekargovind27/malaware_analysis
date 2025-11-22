"""
Stage 1: Raw Data Feasibility Analysis

This package contains modules for analyzing raw IoT-23 data feasibility
before feature engineering. All modules work with PySpark on both local
and Databricks environments.

Modules:
- data_quality: Data quality checks and statistics
- binary_multiclass: Binary and multiclass classification feasibility
- malware_family: Malware family classification feasibility
- unsupervised_models: Autoencoder, clustering, GAN feasibility
- graph_analysis: Graph structure analysis
- utils: Helper functions for loading data and saving reports
- run_stage1: Main orchestrator for running all analyses
"""

__version__ = "2.0.0"
__author__ = "Arun Ghontale, Nidhi Rajani, Sreekar Garimella"

# Expose main functions for easy import
from .run_stage1 import main as run_stage1

__all__ = [
    'run_stage1',
]