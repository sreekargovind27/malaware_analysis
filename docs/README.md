# 🎯 **README.md**

# IoT-23 Malware Detection Pipeline

A comprehensive machine learning pipeline for detecting and classifying malicious network traffic in IoT environments using the IoT-23 dataset. This project implements **three-stage analysis**: feasibility analysis, feature engineering, and model training with support for both traditional ML and deep learning approaches.

---

## 🏗️ Project Architecture

### **Three-Stage Pipeline**

```
Stage 1: Feasibility Analysis (PySpark)
    ↓
Stage 2: Feature Engineering & Data Preparation (PySpark)
    ↓
Stage 3: Model Training & Evaluation (Scikit-learn, PyTorch, PyG)
```

### **Technology Stack**

| Stage | Technology | Purpose | Scale |
|-------|-----------|---------|-------|
| **Stage 1** | PySpark + Docker | Analyze raw data feasibility | 100M+ rows |
| **Stage 2** | PySpark | Feature engineering & graph construction | 50M+ rows |
| **Stage 3** | Pandas, PyTorch, Scikit-learn | Model training | Memory-optimized |

---

## 🚀 Quick Start

### Prerequisites
```
pip install torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-2.8.0+cpu.html
```

**Local Execution:**
- Python 3.8+
- Java 11 or 17 (required for PySpark)
- 16GB+ RAM recommended

**Docker Execution:**
- Docker Desktop
- Docker Compose

**Check Java Installation:**
```bash
java -version  # Should show Java 11 or 17
```

**Install Java if needed:**
```bash
# Mac
brew install openjdk@11

# Ubuntu/Debian
sudo apt-get install openjdk-11-jdk

# Windows
# Download from https://adoptium.net/
```

**Install Python Dependencies:**
```bash
pip install -r requirements.txt
```

---

## 📊 Complete Pipeline

### **Stage 1: Feasibility Analysis**

**Purpose:** Validates raw data before investing time in feature engineering.

**What it analyzes:**
- ✅ Binary classification feasibility (Benign vs Malicious)
- ✅ Multi-class classification feasibility (Attack types)
- ✅ Malware family classification feasibility
- ✅ Graph structure for GNN (heterogeneous graph analysis)
- ✅ Autoencoder anomaly detection feasibility
- ✅ K-means clustering feasibility
- ✅ GAN augmentation needs (identifies rare classes)

**Run Stage 1:**

```bash
# Option 1: Docker (Recommended)
docker compose up --build

# Option 2: Local Python
python analysis/stage1/run_stage1.py
```

**Output Location:** `outputs/stage1_feasibility/`

**Output Files:**
```
outputs/stage1_feasibility/
├── summary.json                      # ⭐ Overall Go/No-Go recommendations
├── data_quality.json                 # Missing values, duplicates, temporal info
├── binary_feasibility.json           # Binary: Benign vs Malicious
├── multiclass_feasibility.json       # Multi-class: Attack types
├── malware_family_feasibility.json   # Malware families
├── graph_feasibility.json            # GNN graph structure analysis
├── autoencoder_feasibility.json      # Anomaly detection with autoencoders
├── clustering_feasibility.json       # K-means clustering
└── gan_candidates.json               # Classes needing GAN augmentation
```

**Decision Point:** Review `summary.json` to determine which models to train.

---

### **Stage 2: Feature Engineering & Data Preparation**

**Purpose:** Transforms raw network flows into ML-ready features and constructs graph structures.

**What it creates:**
1. **Flow-level features** (for traditional ML and DL)
   - Volume features: bytes, packets, rates
   - Temporal features: duration, inter-arrival times
   - Behavioral features: port diversity, connection states
   - Statistical features: entropy, variance, ratios

2. **Device-level features** (aggregated per IoT device)
   - Device behavior profiles
   - Attack patterns
   - Temporal activity patterns

3. **Heterogeneous graph** (for GNN)
   - Device nodes (2,320 IoT devices)
   - Service nodes (port:protocol combinations)
   - Subnet nodes (target subnets)
   - Three edge types: device→service, device→subnet, service↔service

**Run Stage 2:**

```bash
# Full pipeline (feature engineering + graph construction + splitting)
python analysis/stage2/run_stage2.py

# OR run components individually:
python analysis/stage2/feature_engineering.py    # Flow & device features
python analysis/stage2/build_graph.py            # Heterogeneous graph
python analysis/stage2/train_test_split.py       # Split data
```

**Output Location:** `outputs/stage2_prepared/`

**Output Files:**
```
outputs/stage2_prepared/
├── flow_features.parquet              # 76K flows × 100 features
├── device_features.parquet            # 2.3K devices × 30 features
├── feature_list.joblib                # Feature names
│
├── splits/                            # Train/val/test splits
│   ├── train_flows.parquet           # 60% of data
│   ├── val_flows.parquet             # 20% of data
│   ├── test_flows.parquet            # 20% of data
│   ├── train_device_ids.txt          # Device IDs for GNN
│   ├── val_device_ids.txt
│   └── test_device_ids.txt
│
├── normalized/                        # Scaled features
│   ├── flow_scaler.pkl               # StandardScaler for flows
│   └── device_scaler.pkl             # StandardScaler for devices
│
└── graph/                             # GNN graph data
    ├── hetero_graph.pt               # PyTorch Geometric format
    ├── device_nodes.parquet          # Device node features
    ├── service_nodes.parquet         # Service node features
    ├── subnet_nodes.parquet          # Subnet node features
    ├── edges.parquet                 # All edges
    └── graph_stats.json              # Graph statistics
```

---

### **Stage 3: Model Training**

**Purpose:** Train and evaluate ML/DL models on prepared data.

**Available Models:**

#### **Traditional ML (Supervised)**
```bash
# Binary classification (Benign vs Malicious)
python models/traditional/train_binary.py

# Multi-class classification (Attack types: PortScan, DDoS, C&C, etc.)
python models/traditional/train_multiclass.py

# Malware family classification (Mirai, Gagfyt, Kenjiro)
python models/traditional/train_family.py
```

**Models:** Random Forest, XGBoost, LightGBM

#### **Deep Learning**
```bash
# Multi-layer perceptron (MLP) for classification
python models/deep_learning/train_mlp.py

# Autoencoder for anomaly detection
python models/deep_learning/train_autoencoder.py
```

**Models:** PyTorch MLP, Autoencoder

#### **Graph Neural Networks**
```bash
# Heterogeneous Graph Neural Network for malware family classification
python models/gnn/train_gnn.py
```

**Models:** HGT (Heterogeneous Graph Transformer), R-GCN

#### **Unsupervised Learning**
```bash
# K-means clustering on malicious traffic
python models/unsupervised/train_kmeans.py

# GAN for data augmentation (rare classes)
python models/unsupervised/train_gan.py
```

**Output Location:** `outputs/models_trained/` and `outputs/results/`

**Output Structure:**
```
outputs/
├── models_trained/
│   ├── traditional/
│   │   ├── binary_rf.pkl
│   │   ├── binary_xgboost.pkl
│   │   ├── multiclass_lightgbm.pkl
│   │   └── family_xgboost.pkl
│   ├── deep_learning/
│   │   ├── mlp_binary.pt
│   │   └── autoencoder.pt
│   ├── gnn/
│   │   └── hetero_gnn.pt
│   └── unsupervised/
│       ├── kmeans.pkl
│       └── gan_generator.pt
│
└── results/
    ├── binary_classification/
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   ├── metrics.json
    │   └── predictions.csv
    ├── multiclass/
    │   ├── confusion_matrix_heatmap.png
    │   ├── per_class_metrics.json
    │   └── predictions.csv
    ├── malware_family/
    │   └── ...
    ├── autoencoder/
    │   ├── reconstruction_errors.png
    │   └── anomaly_scores.csv
    └── clustering/
        ├── cluster_visualization.html
        └── cluster_profiles.json
```

---

## 📁 Project Structure

```
dic_phase2/
│
├── config.py                          # ⚙️ Central configuration
├── requirements.txt                   # 📦 Python dependencies
├── docker-compose.yml                 # 🐳 Docker orchestration
├── Dockerfile                         # 🐳 Container definition
│
├── analysis/                          # 📊 Data analysis (Stages 1 & 2)
│   │
│   ├── stage1/                        # Stage 1: Feasibility Analysis
│   │   ├── __init__.py
│   │   ├── run_stage1.py             # Main entry point
│   │   ├── utils.py                  # PySpark utilities
│   │   ├── data_quality.py           # Data quality checks
│   │   ├── binary_multiclass.py      # Binary & multiclass analysis
│   │   ├── malware_family.py         # Family classification analysis
│   │   ├── graph_analysis.py         # Graph structure for GNN
│   │   ├── unsupervised_models.py    # Autoencoder, K-means, GAN analysis
│   │   └── README.md
│   │
│   └── stage2/                        # Stage 2: Feature Engineering
│       ├── __init__.py
│       ├── run_stage2.py             # Main entry point
│       ├── feature_engineering.py    # Flow-level features
│       ├── device_aggregation.py     # Device-level aggregation
│       ├── build_graph.py            # Heterogeneous graph construction
│       ├── train_test_split.py       # Data splitting
│       ├── quality_check.py          # Stage 2 quality validation
│       ├── data_loader.py            # Dataset loading utilities
│       └── utils.py
│
├── models/                            # 🤖 Model training code (Stage 3)
│   ├── traditional/                   # Random Forest, XGBoost, LightGBM
│   │   ├── __init__.py
│   │   ├── train_all.py              # Train all traditional models
│   │   ├── train_binary.py
│   │   ├── train_multiclass.py
│   │   ├── train_family.py
│   │   └── evaluate.py
│   │
│   ├── deep_learning/                 # PyTorch models
│   │   ├── __init__.py
│   │   ├── train_all.py
│   │   ├── train_mlp.py
│   │   ├── train_autoencoder.py
│   │   └── evaluate.py
│   │
│   ├── gnn/                           # Graph Neural Networks
│   │   ├── __init__.py
│   │   ├── train_gnn.py
│   │   ├── architectures.py          # HGT, R-GCN
│   │   ├── data_loader.py
│   │   └── evaluate.py
│   │
│   └── unsupervised/                  # Clustering, GAN
│       ├── __init__.py
│       ├── train_kmeans.py
│       ├── train_gan.py
│       └── evaluate.py
│
├── scripts/                           # 🛠️ Utility scripts
│   ├── migrate_to_outputs.sh         # Migrate old structure to outputs/
│   ├── compare_models.py             # Compare all model results
│   └── visualize_results.py
│
├── data/                              # 💾 Raw data (not in Git)
│   ├── raw/                          # Full dataset CSVs
│   └── raw_test/                     # Test subset CSVs
│
├── outputs/                           # 📁 All outputs (Git-ignored)
│   ├── stage1_feasibility/           # Stage 1 JSON reports
│   ├── stage2_prepared/              # Prepared datasets & graph
│   ├── stage2_quality/               # Stage 2 validation reports
│   ├── models_trained/               # Trained model files
│   ├── results/                      # Metrics, plots, predictions
│   └── logs/                         # Training logs
│
└── notebooks/                         # 📓 Jupyter notebooks (optional)
    └── exploration.ipynb
```

---

## ⚙️ Configuration

Edit `config.py` to customize:

### **Data Settings**
```python
TEST_MODE = True              # Use raw_test/ (subset) or raw/ (full)
SAMPLE_SIZE = 50_000_000     # Target dataset size
TEST_SIZE = 0.2              # Train/test split ratio
RANDOM_STATE = 42            # Reproducibility seed
```

### **Model Hyperparameters**
```python
# Traditional ML
BINARY_N_ESTIMATORS = 200
MULTICLASS_N_ESTIMATORS = 200

# Deep Learning
AUTOENCODER_LATENT_DIM = 8
AUTOENCODER_EPOCHS = 100
AUTOENCODER_BATCH_SIZE = 32768

# Optimization
USE_OPTUNA = True            # Hyperparameter tuning
OPTUNA_N_TRIALS = 20
USE_SMOTE = True             # Handle class imbalance
```

### **Hardware**
```python
DEVICE = 'cuda'              # 'cuda', 'cpu', or 'mps'
N_JOBS = 32                  # Parallel workers
NUM_WORKERS = 6              # DataLoader workers
```

---

## 🐳 Docker Usage

### **What Docker Does**
- Runs Stage 1 feasibility analysis in isolated environment
- Handles PySpark dependencies automatically
- Outputs results to your local machine

### **Docker Commands**

```bash
# Build and run
docker compose up --build

# Run in background
docker compose up -d

# View logs
docker compose logs -f

# Stop and cleanup
docker compose down

# Full cleanup and rebuild
docker compose down && docker builder prune -af && docker compose build --no-cache && docker compose up
```

### **Volume Mounts**

| Host Path | Container Path | Purpose |
|-----------|----------------|---------|
| `./data` | `/app/data` | Raw CSV inputs |
| `./outputs` | `/app/outputs` | All outputs |
| `./analysis` | `/app/analysis` | Live-editable code |
| `./models` | `/app/models` | Model code |
| `./config.py` | `/app/config.py` | Configuration |

---

## 📈 Expected Runtime

| Stage | Task | Time (Full) | Time (Test) |
|-------|------|-------------|-------------|
| **Stage 1** | Feasibility Analysis | 5-10 min | 1-2 min |
| **Stage 2** | Feature Engineering | 20-30 min | 3-5 min |
| **Stage 2** | Graph Construction | 10-15 min | 2-3 min |
| **Stage 2** | Data Splitting | 2-5 min | <1 min |
| **Stage 3** | Binary (RF/XGBoost) | 5-10 min | 1-2 min |
| **Stage 3** | Multiclass | 10-20 min | 2-5 min |
| **Stage 3** | Malware Family | 5-10 min | 1-2 min |
| **Stage 3** | Autoencoder | 10-15 min | 2-3 min |
| **Stage 3** | GNN | 15-30 min | 3-5 min |

---

## 🎯 Model Performance Metrics

After Stage 3, compare models using:

```bash
python scripts/compare_models.py
```

**Outputs:**
- `outputs/results/model_comparison.html` - Interactive comparison
- `outputs/results/model_comparison_table.csv` - Metrics table

**Metrics Tracked:**
- Accuracy, Precision, Recall, F1-score
- ROC-AUC (binary)
- Per-class metrics (multiclass)
- Confusion matrices
- Training time, inference time

---

## 🔧 Troubleshooting

### **Stage 1 Issues**

**"No CSV files found":**
```bash
ls -la data/raw_test/  # Check files exist
grep TEST_MODE config.py  # Check config
```

**PySpark "Java not found":**
```bash
java -version  # Verify Java 11 or 17
export JAVA_HOME=$(/usr/libexec/java_home)  # Mac
export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64  # Linux
```

### **Stage 2 Issues**

**Out of memory:**
```python
# In config.py
SAMPLE_SIZE = 10_000_000  # Reduce
```

**Missing features:**
```bash
# Re-run feature engineering
python analysis/stage2/feature_engineering.py
```

### **Stage 3 Issues**

**"No training data found":**
```bash
# Ensure Stage 2 completed
ls -la outputs/stage2_prepared/splits/
```

**CUDA out of memory:**
```python
# In config.py
AUTOENCODER_BATCH_SIZE = 16384  # Reduce
DEVICE = 'cpu'  # Use CPU
```

---

## 📚 Dataset

**IoT-23 Dataset**
- **Source**: [Stratosphere IPS](https://www.stratosphereips.org/datasets-iot23)
- **Size**: 76,671 network flows (test subset), 20M+ (full)
- **Devices**: 2,423 infected IoT devices
- **Families**: 3 major malware families (Mirai, Gagfyt, Kenjiro)
- **Format**: Zeek/Bro network logs (CSV)

---

## 🤝 Contributing

### **Development Workflow**
1. Fork the repository
2. Create feature branch: `git checkout -b feature/my-feature`
3. Make changes
4. Test: Run all three stages
5. Commit: `git commit -m "Add feature"`
6. Push: `git push origin feature/my-feature`
7. Open Pull Request

### **Code Standards**
- PEP 8 style guide
- Type hints for function signatures
- Docstrings for all public functions
- Comments for complex logic

---

## 📝 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- **IoT-23 Dataset**: Stratosphere IPS
- **PySpark**: Apache Spark community
- **PyTorch Geometric**: PyG team

---

## 📞 Support

**Issues?** Open a GitHub issue with:
- Stage number (1, 2, or 3)
- Error message
- System info (OS, Python version, Java version)
- Log file from `outputs/logs/`

---

**Ready to get started? Run Stage 1!**

```bash
docker compose up --build
```

**Key improvements:**
1. ✅ Reflects actual 3-stage structure
2. ✅ Shows outputs/ folder consolidation
3. ✅ Explains heterogeneous graph construction for GNN
4. ✅ Clear separation: analysis/ (stages 1-2) vs models/ (stage 3)
5. ✅ Complete model list including GNN
6. ✅ Accurate file paths
7. ✅ Proper Docker usage
8. ✅ Expected runtimes
9. ✅ Troubleshooting section
10. ✅ Contributing guidelines