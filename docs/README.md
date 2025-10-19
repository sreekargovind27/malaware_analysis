# IoT-23 Malware Detection

A comprehensive ML pipeline for detecting and classifying malicious network traffic in IoT environments using the IoT-23 dataset. This project is designed to handle the full, large-scale dataset by implementing an efficient, multi-stage data processing and modeling workflow.

## Project Workflow

This project follows a three-phase pipeline to manage the large-scale data and ensure reproducibility:

1.  **Phase 1: Feature Engineering (`build_dataset.py`)**
    *   Reads the massive raw CSV files from `data/raw/`.
    *   Applies a full suite of feature engineering in parallel.
    *   Creates a single, unified, and cleaned dataset of **50 million rows** (`final_features.parquet`) for further processing.

2.  **Phase 2: Master Data Splitting (`create_splits.py`)**
    *   Reads the 50M row dataset created in the previous phase.
    *   Performs a **single, canonical 80/20 train/test split**.
    *   Saves the final `train_set.parquet` (40M rows) and `test_set.parquet` (10M rows) to disk. This ensures all subsequent model training and evaluation is done on the exact same data, guaranteeing reproducible results.

3.  **Phase 3: Model Training & Evaluation**
    *   The individual model scripts (`binary_classifier.py`, `multiclass_classifier.py`, etc.) load the pre-split, pre-processed data.
    *   This makes the training scripts fast, clean, and focused solely on the modeling task.

## Quick Start

Follow these steps to run the entire pipeline from start to finish.

```bash
# 1. Install all dependencies
pip install -r requirements.txt

# 2. Place the original IoT-23 CSV files in the data/raw/ directory.

# 3. PHASE 1: Build the main engineered dataset.
# This is a one-time, long-running step (approx. 20-30 minutes).
# It processes all raw CSVs into a single 50M row dataset.
python build_dataset.py

# 4. PHASE 2: Create the master train/test splits.
# This is also a one-time step, but it is much faster.
# It creates the permanent train and test sets used by all models.
python create_splits.py

# 5. PHASE 3: Train and evaluate all models.
# These scripts can be run in any order and are now much faster.
python binary_classifier.py
python multiclass_classifier.py
python virus_classifier.py
python autoencoder.py
python clustering.py
```

## Models

1.  **Binary Classifier** - Classifies traffic as Benign vs. Malicious using a tuned LightGBM model.
2.  **Multi-Class Classifier** - Identifies the specific type of attack (e.g., C&C, PortScan) using XGBoost and a PyTorch Neural Network.
3.  **Virus Classifier** - Identifies the specific malware family (e.g., Mirai, Torii) using a tuned LightGBM model.
4.  **Autoencoder** - Anomaly detection model trained only on benign data to identify novel threats by reconstruction error.
5.  **K-Means Clustering** - Unsupervised grouping of malicious traffic to discover inherent patterns and families.

## Configuration

Edit `config.py` to manage paths and model hyperparameters. Key settings include:
-   `SAMPLE_SIZE`: The target size for the main engineered dataset created by `build_dataset.py`. Default is `50,000,000`.
-   `TEST_SIZE`: The fraction of data to hold out for the master test set. Default is `0.2`.
-   `USE_OPTUNA` / `OPTUNA_N_TRIALS`: Toggle and configure the hyperparameter search for the classification models.

## File Overview

-   **Data Preparation (Run Once, In Order)**
    -   `build_dataset.py`: **Phase 1.** Cleans and engineers features from raw data into a single master dataset.
    -   `create_splits.py`: **Phase 2.** Performs the master train/test split and saves the results.
-   **Data Loading**
    -   `data_loader.py`: Contains functions that load the final, pre-split datasets for the models. Also handles the class-aware undersampling for the multiclass task.
-   **Model Training (Run After Data Prep)**
    -   `binary_classifier.py`: Benign vs. Malicious classification.
    -   `multiclass_classifier.py`: Attack Type classification.
    -   `virus_classifier.py`: Malware Family classification.
    -   `autoencoder.py` / `autoencoder_denoising.py`: Anomaly Detection models.
    -   `clustering.py`: K-Means Clustering.
-   **Configuration & Utilities**
    -   `config.py`: Central configuration for all scripts.
    -   `requirements.txt`: All project dependencies.