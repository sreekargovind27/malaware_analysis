# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 1: Data Quality & Feasibility Analysis
# MAGIC
# MAGIC Runs all Stage 1 analyses using existing modules

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
