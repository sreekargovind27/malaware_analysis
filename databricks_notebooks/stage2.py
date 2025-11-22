# Databricks notebook source
# MAGIC %md
# MAGIC # Stage 2: Feature Engineering & Preparation
# MAGIC
# MAGIC Runs all Stage 2 steps using existing modules

# COMMAND ----------

from analysis.stage2.run_stage2 import main

# COMMAND ----------

# Run complete Stage 2 pipeline
main()

# COMMAND ----------

# MAGIC %md
# MAGIC ## View Quality Report

# COMMAND ----------

import json
import os
from config import Config

# Load quality report
report_path = os.path.join(Config.STAGE2_QUALITY_DIR, "stage2_validation_report.json")

if os.path.exists(report_path):
    with open(report_path, 'r') as f:
        report = json.load(f)
    print(json.dumps(report, indent=2))
else:
    print("Quality report not found")
