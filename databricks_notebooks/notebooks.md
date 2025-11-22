# Databricks Notebooks - Quick Start

## 📓 Notebooks

1. **`nb_stage1.py`** - Run Stage 1 (Data Quality & Feasibility)
2. **`nb_stage2.py`** - Run Stage 2 (Feature Engineering & Preparation)
3. **`nb_master.py`** - Run both Stage 1 + Stage 2

## 🚀 Quick Start

### 1. Upload Your Code to DBFS

```bash
# Upload Python modules to DBFS
databricks fs cp config.py dbfs:/FileStore/dic_phase2/config.py
databricks fs cp -r analysis/ dbfs:/FileStore/dic_phase2/analysis/
```

### 2. Upload Notebooks to Databricks

Import these 3 notebooks into your Databricks workspace:
- `nb_stage1.py`
- `nb_stage2.py`
- `nb_master.py`

### 3. Upload Data

```bash
# Upload CSV files
databricks fs cp -r data/raw_test/ dbfs:/FileStore/dic_phase2/data/raw_test/
```

### 4. Run

Open any notebook and run all cells!

---

## 📁 Required File Structure

**On DBFS:**
```
/FileStore/dic_phase2/
├── config.py
├── analysis/
│   ├── stage1/
│   │   ├── __init__.py
│   │   ├── run_stage1.py
│   │   ├── utils.py
│   │   ├── data_quality.py
│   │   ├── binary_multiclass.py
│   │   ├── malware_family.py
│   │   ├── graph_analysis.py
│   │   └── unsupervised_models.py
│   └── stage2/
│       ├── __init__.py
│       ├── run_stage2.py
│       ├── utils.py
│       ├── feature_engineering.py
│       ├── device_aggregation.py
│       ├── train_test_split.py
│       ├── build_graph.py
│       └── stage2_quality_report.py
└── data/
    └── raw_test/
        └── *.csv
```

---

## ⚙️ What Each Notebook Does

### `nb_stage1.py`
Runs your existing `analysis/stage1/run_stage1.py` which:
- Data quality analysis
- Binary classification feasibility
- Multiclass classification feasibility
- Malware family feasibility
- Clustering feasibility
- Graph feasibility

**Output:** JSON reports in `/outputs/stage1_quality/`

### `nb_stage2.py`
Runs your existing `analysis/stage2/run_stage2.py` which:
- Feature engineering
- Device aggregation
- Train/val/test split
- Graph building
- Quality validation

**Output:** Parquet files in `/outputs/stage2_prepared/`

### `nb_master.py`
Runs both Stage 1 and Stage 2 in sequence.

**Runtime:** ~10-15 minutes

---

## 🔧 Config.py Must Set Paths

Your `config.py` should have:

```python
if cls.is_databricks():
    cls.BASE_DIR = Path("/dbfs/FileStore/dic_phase2")
    cls.RAW_DATA_DIR = cls.BASE_DIR / "data" / "raw_test"
    cls.OUTPUTS_BASE_DIR = cls.BASE_DIR / "outputs"
```

---

## 💡 Tips

1. **First cell in notebook can add path:**
   ```python
   import sys
   sys.path.append('/dbfs/FileStore/dic_phase2')
   ```

2. **Check your paths:**
   ```python
   from config import Config
   print(Config.RAW_DATA_DIR)
   print(Config.STAGE1_QUALITY_DIR)
   ```

3. **View output files:**
   ```python
   dbutils.fs.ls('/FileStore/dic_phase2/outputs/stage1_quality/')
   ```

---

## ✅ That's It!

These notebooks are **dead simple**:
- Import your existing `run_stage1.py` or `run_stage2.py`
- Call `main()`
- Done!

No code duplication. Uses all your existing logic.