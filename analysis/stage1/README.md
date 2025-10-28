# Stage 1: Raw Data Feasibility Analysis

Analyzes raw IoT-23 data to determine feasibility for all models **before** feature engineering.

## What It Does

Generates feasibility reports for:
1. **Binary Classification** (Benign vs Malicious)
2. **Multi-Class Classification** (Attack types)
3. **Malware Family Classification** 
4. **Autoencoder** (Anomaly detection)
5. **K-Means Clustering**
6. **GAN Augmentation** (Identifies rare classes)
7. **GNN** (Graph structure analysis)
8. **Overall Data Quality**

## Files

```
analysis/stage1/
├── run_stage1.py              # Main runner (execute this)
├── utils.py                   # Shared PySpark utilities
├── data_quality.py            # Data quality checks
├── binary_multiclass.py       # Binary & multiclass analysis
├── malware_family.py          # Family classification analysis
├── graph_analysis.py          # Graph structure for GNN
├── unsupervised_models.py     # Autoencoder, K-means, GAN
└── README.md                  # This file
```

## Requirements

- PySpark >= 3.5.0 (added to requirements.txt)
- Raw CSV files in `Config.RAW_DIR_ORIGINAL`

## Usage

### Local Execution

```bash
# From project root
python analysis/stage1/run_stage1.py
```

### Databricks Execution

1. Upload `analysis/` folder to Databricks workspace
2. Create notebook and run:
   ```python
   %run /Workspace/path/to/analysis/stage1/run_stage1
   ```
3. Or submit as job

## Output

Creates `stage1_feasibility/` directory with:

```
stage1_feasibility/
├── summary.json                      # Overall Go/No-Go recommendations
├── data_quality.json                 # Overall data stats
├── binary_feasibility.json           # Binary classification
├── multiclass_feasibility.json       # Attack type classification
├── malware_family_feasibility.json   # Malware family classification
├── graph_feasibility.json            # GNN graph structure
├── autoencoder_feasibility.json      # Anomaly detection
├── clustering_feasibility.json       # K-means clustering
└── gan_candidates.json               # Rare classes needing GAN
```

## Key Features

✅ **Works locally and on Databricks** (same code)
✅ **Reuses Config.py** (no code duplication)
✅ **PySpark-native** (scales to large datasets)
✅ **Label generation from detailed-label** (flow-level, not file-level)

## Example Output

```
OVERALL RECOMMENDATION: PROCEED
Message: 6 models are feasible. Ready for feature engineering.

Model-by-model status:
  ✅ binary_classification: GO
  ✅ multiclass_classification: GO
  ⚠️  malware_family_classification: WARNING (GAN augmentation needed)
  ✅ autoencoder_anomaly_detection: GO
  ✅ kmeans_clustering: GO
  ⚠️  gan_augmentation: WARNING (3 families need augmentation)
  ✅ gnn_graph_classification: GO
```

## Next Steps

After Stage 1:
1. Review `summary.json` for Go/No-Go decisions
2. Identify rare classes in `gan_candidates.json`
3. Proceed to feature engineering (PySpark)
4. Run Stage 2 quality analysis (after feature engineering)
