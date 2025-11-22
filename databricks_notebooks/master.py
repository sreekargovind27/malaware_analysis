# Databricks notebook source
# MAGIC %md
# MAGIC # Master Pipeline: Stage 1 + Stage 2
# MAGIC
# MAGIC Runs complete pipeline using existing modules

# COMMAND ----------

import time

start_time = time.time()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 1: Data Quality & Feasibility

# COMMAND ----------

from analysis.stage1.run_stage1 import main as run_stage1

print("=" * 70)
print("RUNNING STAGE 1")
print("=" * 70)

run_stage1()

stage1_time = time.time() - start_time
print(f"\nStage 1 completed in {stage1_time:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Stage 2: Feature Engineering & Preparation

# COMMAND ----------

from analysis.stage2.run_stage2 import main as run_stage2

stage2_start = time.time()

print("=" * 70)
print("RUNNING STAGE 2")
print("=" * 70)

run_stage2()

stage2_time = time.time() - stage2_start
print(f"\nStage 2 completed in {stage2_time:.1f}s")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

total_time = time.time() - start_time

print("=" * 70)
print("🎉 FULL PIPELINE COMPLETE!")
print("=" * 70)
print(f"\nTotal time: {total_time:.1f}s ({total_time / 60:.1f} min)")
print(f"  Stage 1: {stage1_time:.1f}s")
print(f"  Stage 2: {stage2_time:.1f}s")
print("\n🚀 Ready for Stage 3: Model Training!")
