# README.md

# IoT-23 Malware Detection

A comprehensive ML pipeline for detecting and classifying malicious network traffic in IoT environments using the IoT-23 dataset.

## Project Workflow

This project follows a two-stage data pipeline, orchestrated by a single master script:

1.  **Raw Data (`data/raw`)**: The original, untouched CSV files from the IoT-23 dataset. This is the starting point.

2.  **Engineered Data**: The `build_dataset.py` script reads the raw data, fixes all structural errors, applies a full suite of feature engineering, and saves the final, model-ready outputs in two locations:
    *   **`data/engineered_split_csv/`**: Contains an individually engineered CSV for each raw file. Perfect for per-scenario analysis.
    *   **`data/engineered_features/`**: Contains a single master dataset in both `.parquet` (for fast model training) and `.csv` (for manual analysis) formats.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place the 21 original CSV files in the data/raw/ directory.

# 3. Build the engineered dataset (This is the main data preparation step)
# This script reads from data/raw/, fixes errors, engineers all features,
# and saves the final outputs. This may take a few minutes.
python build_dataset.py

# 4. Train your models (Fast and repeatable)
# These scripts now load the pre-engineered data and run much faster.
python binary_classifier.py
python multiclass_classifier.py
python autoencoder.py
python clustering.py
python virus_classifier.py
```

## Models

1.  **Autoencoder** - Anomaly detection on network traffic.
2.  **Binary Classifier** - Classifies traffic as Benign vs. Malicious.
3.  **Multi-Class Classifier** - Identifies the specific type of attack (e.g., C&C, PortScan).
4.  **K-Means Clustering** - Unsupervised grouping of malicious traffic to find patterns.
5.  **Virus Classifier** - Identifies the specific malware family (e.g., Mirai, Torii).

## Configuration

Edit `config.py` to manage paths and model hyperparameters. Key settings include:
-   `SAMPLE_SIZE`: Set to an integer (e.g., `50000`) in `build_dataset.py` for a quick test run, or `None` to process the entire dataset.
-   `DEVICE`: Automatically set to `'cuda'` for GPU or `'cpu'` for CPU.
-   `FILENAME_TO_FAMILY_MAP`: The ground-truth mapping from filenames to malware families.

## File Overview

-   `config.py`: Central configuration, including paths and ground-truth malware family mapping.
-   `build_dataset.py`: **The main script** for all data preparation. It handles cleaning, feature engineering, and saving all final data assets.
-   `data_loader.py`: Contains lightweight functions that load the final, pre-engineered dataset for the models.
-   `autoencoder.py`: Model 1 - Anomaly Detection.
-   `binary_classifier.py`: Model 2 - Benign vs. Malicious.
-   `multiclass_classifier.py`: Model 3 - Attack Type Classification.
-   `clustering.py`: Model 4 - K-Means Clustering.
-   `virus_classifier.py`: Model 5 - Malware Family Classification.