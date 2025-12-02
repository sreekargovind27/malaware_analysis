"""
Compare features between real data and manually constructed data
"""

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('compare').getOrCreate()

# Load real ML-ready data
df = spark.read.parquet('/Users/nidhirajani/Desktop/DIC Phase 3/data/processed/ml_ready_data.parquet')

# Get one PortScan sample
portscan = df.filter(df.malware_family == "PortScan").limit(1)

print("="*60)
print("REAL PortScan sample from dataset:")
print("="*60)
portscan.select("is_malicious", "malware_family", "orig_port", "resp_port", "proto", "features").show(1, truncate=False, vertical=True)

# Get one Benign sample
benign = df.filter(df.malware_family == "Benign").limit(1)

print("\n" + "="*60)
print("REAL Benign sample from dataset:")
print("="*60)
benign.select("is_malicious", "malware_family", "orig_port", "resp_port", "proto", "features").show(1, truncate=False, vertical=True)

# Show feature vector values
print("\nPortScan feature vector:")
portscan_features = portscan.select("features").collect()[0][0]
print(portscan_features)

print("\nBenign feature vector:")
benign_features = benign.select("features").collect()[0][0]
print(benign_features)

spark.stop()