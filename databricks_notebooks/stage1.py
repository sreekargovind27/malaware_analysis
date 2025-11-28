# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 1: Data Quality & Feasibility Analysis
# MAGIC
# MAGIC Runs all Stage 1 analyses using existing modules

# COMMAND ----------

!pip install -r ../requirements.txt

# COMMAND ----------

import sys
import os

# Add the repo root to Python path
repo_root = "/Workspace/Users/arun.g.ghontale@gmail.com/malware_analysis"
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

# Verify
print(f"✅ Added {repo_root} to Python path")
print(f"Current working directory: {os.getcwd()}")

# COMMAND ----------

# MAGIC %pwd

# COMMAND ----------

from analysis.stage1.run_stage1 import main

# COMMAND ----------

# Run complete Stage 1 pipeline
main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Results

# COMMAND ----------

import json
from config import Config

# Load feasibility summary
with open(Config.FEASIBILITY_SUMMARY, 'r') as f:
    summary = json.load(f)

print(json.dumps(summary, indent=2))
