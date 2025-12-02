"""
Simple test for MCP tool
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
import os

# Initialize Spark
spark = SparkSession.builder \
    .appName("MCP-Test") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load model
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_path = os.path.join(project_root, "models", "lightgbm_classifier")
model = GBTClassificationModel.load(model_path)

print("✓ Model loaded\n")

# Test sample 1: Malicious traffic (Mirai-like)
print("="*60)
print("TEST 1: Suspicious Telnet Traffic")
print("="*60)

malicious_features = [
    0.5,    # duration
    120.0,  # orig_bytes
    0.0,    # resp_bytes
    3.0,    # orig_pkts
    0.0,    # resp_pkts
    180.0,  # orig_ip_bytes
    0.0,    # resp_ip_bytes
    120.0,  # total_bytes
    3.0,    # total_packets
    40.0,   # bytes_per_packet
    0.0,    # resp_orig_ratio
    6.0,    # packets_per_second
    1,      # is_telnet_port (YES - port 23)
    0,      # is_ssh_port
    0,      # is_http_port
    1,      # proto_encoded (TCP)
    14,     # hour_of_day
    3,      # day_of_week
    54321,  # orig_port
    23      # resp_port (TELNET)
]

df = spark.createDataFrame([Row(features=Vectors.dense(malicious_features))])
result = model.transform(df).select("prediction", "probability").collect()[0]

prediction = "Malicious" if result["prediction"] == 1 else "Benign"
probs = result["probability"].toArray()

print(f"Traffic: TCP connection to Telnet port (23)")
print(f"Bytes: 120 sent, 0 received (one-way)")
print(f"Packets: 3 sent, 0 received\n")
print(f"**PREDICTION: {prediction}**")
print(f"Malicious probability: {probs[1]:.2%}")
print(f"Benign probability: {probs[0]:.2%}")

# Test sample 2: Benign traffic (Normal HTTP)
print("\n" + "="*60)
print("TEST 2: Normal HTTP Traffic")
print("="*60)

benign_features = [
    2.5,    # duration
    500.0,  # orig_bytes
    1500.0, # resp_bytes
    10.0,   # orig_pkts
    12.0,   # resp_pkts
    600.0,  # orig_ip_bytes
    1600.0, # resp_ip_bytes
    2000.0, # total_bytes
    22.0,   # total_packets
    90.9,   # bytes_per_packet
    3.0,    # resp_orig_ratio
    8.8,    # packets_per_second
    0,      # is_telnet_port
    0,      # is_ssh_port
    1,      # is_http_port (YES - port 80)
    1,      # proto_encoded (TCP)
    10,     # hour_of_day
    2,      # day_of_week
    52000,  # orig_port
    80      # resp_port (HTTP)
]

df = spark.createDataFrame([Row(features=Vectors.dense(benign_features))])
result = model.transform(df).select("prediction", "probability").collect()[0]

prediction = "Malicious" if result["prediction"] == 1 else "Benign"
probs = result["probability"].toArray()

print(f"Traffic: TCP connection to HTTP port (80)")
print(f"Bytes: 500 sent, 1500 received (bidirectional)")
print(f"Packets: 10 sent, 12 received\n")
print(f"**PREDICTION: {prediction}**")
print(f"Malicious probability: {probs[1]:.2%}")
print(f"Benign probability: {probs[0]:.2%}")

print("\n" + "="*60)
print("✓ Tests complete!")
print("="*60)

spark.stop()