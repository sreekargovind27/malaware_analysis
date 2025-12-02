"""
Test model with real samples from the dataset
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
import os

spark = SparkSession.builder \
    .appName("Real-Sample-Test") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load model
model_path = "/Users/nidhirajani/Desktop/DIC Phase 3/models/lightgbm_classifier"
model = GBTClassificationModel.load(model_path)


# Load ML-ready data
data_path = "/Users/nidhirajani/Desktop/DIC Phase 3/data/processed/ml_ready_data.parquet"
df = spark.read.parquet(data_path)

print("="*60)
print("Testing with REAL samples from dataset")
print("="*60)

# Get 5 actual malicious samples (PortScan)
print("\n1. Real PortScan samples:")
portscan_samples = df.filter(df.malware_family == "PortScan").limit(5)
predictions = model.transform(portscan_samples)
predictions.select("is_malicious", "prediction", "probability", "malware_family", "orig_port", "resp_port").show(5, truncate=False)

# Get 5 actual benign samples
print("\n2. Real Benign samples:")
benign_samples = df.filter(df.malware_family == "Benign").limit(5)
predictions = model.transform(benign_samples)
predictions.select("is_malicious", "prediction", "probability", "malware_family", "orig_port", "resp_port").show(5, truncate=False)

# Get 5 Okiru samples
print("\n3. Real Okiru samples:")
okiru_samples = df.filter(df.malware_family == "Okiru").limit(5)
predictions = model.transform(okiru_samples)
predictions.select("is_malicious", "prediction", "probability", "malware_family", "orig_port", "resp_port").show(5, truncate=False)

# Get 5 C&C samples
print("\n4. Real C&C samples:")
cc_samples = df.filter(df.malware_family == "C&C").limit(5)
predictions = model.transform(cc_samples)
predictions.select("is_malicious", "prediction", "probability", "malware_family", "orig_port", "resp_port").show(5, truncate=False)

print("="*60)

spark.stop()