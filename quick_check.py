from pyspark.sql import SparkSession
import os

os.chdir('/Users/nidhirajani/Desktop/DIC Phase 3')

spark = SparkSession.builder.appName('check').getOrCreate()

print("="*60)
print("ORIGINAL DATA (from ingestion)")
print("="*60)
df_orig = spark.read.parquet('data/processed/full_labeled_data.parquet')
print(f"Total rows: {df_orig.count()}")
print("\nClass distribution:")
df_orig.groupBy('is_malicious').count().orderBy('is_malicious').show()
print("\nMalware family distribution:")
df_orig.groupBy('malware_family').count().orderBy('count', ascending=False).show()

print("\n" + "="*60)
print("ML-READY DATA (after feature engineering)")
print("="*60)
df_ml = spark.read.parquet('data/processed/ml_ready_data.parquet')
print(f"Total rows: {df_ml.count()}")
print("\nClass distribution:")
df_ml.groupBy('is_malicious').count().orderBy('is_malicious').show()
print("\nMalware family distribution:")
df_ml.groupBy('malware_family').count().orderBy('count', ascending=False).show()

spark.stop()