"""
Manual test of MCP tool with realistic malware patterns
"""

import json
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

# Initialize Spark and load model
spark = SparkSession.builder \
    .appName("MCP-Manual-Test") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

model_path = "/Users/nidhirajani/Desktop/DIC Phase 3/models/lightgbm_classifier"
model = GBTClassificationModel.load(model_path)

print("="*60)
print("MCP Tool: classify_traffic_malicious")
print("="*60)

# Test Case 1: PortScan pattern (from real data)
print("\n1. Testing PortScan pattern (ports 38792→59353):")
print("-" * 60)

traffic_1 = {
    "duration": 0.001,
    "orig_bytes": 40,
    "resp_bytes": 0,
    "orig_pkts": 1,
    "resp_pkts": 0,
    "orig_ip_bytes": 60,
    "resp_ip_bytes": 0,
    "orig_port": 38792,
    "resp_port": 59353,
    "proto": "tcp"
}

# Calculate derived features
traffic_1["total_bytes"] = traffic_1["orig_bytes"] + traffic_1["resp_bytes"]
traffic_1["total_packets"] = traffic_1["orig_pkts"] + traffic_1["resp_pkts"]
traffic_1["bytes_per_packet"] = traffic_1["total_bytes"] / traffic_1["total_packets"] if traffic_1["total_packets"] > 0 else 0
traffic_1["resp_orig_ratio"] = traffic_1["resp_bytes"] / traffic_1["orig_bytes"] if traffic_1["orig_bytes"] > 0 else 0
traffic_1["packets_per_second"] = traffic_1["total_packets"] / traffic_1["duration"] if traffic_1["duration"] > 0 else 0
traffic_1["is_telnet_port"] = 1 if traffic_1["orig_port"] == 23 or traffic_1["resp_port"] == 23 else 0
traffic_1["is_ssh_port"] = 1 if traffic_1["orig_port"] == 22 or traffic_1["resp_port"] == 22 else 0
traffic_1["is_http_port"] = 1 if traffic_1["orig_port"] == 80 or traffic_1["resp_port"] == 80 else 0
traffic_1["proto_encoded"] = 1 if traffic_1["proto"] == "tcp" else 2 if traffic_1["proto"] == "udp" else 0
traffic_1["hour_of_day"] = 12
traffic_1["day_of_week"] = 3

# Build feature vector
feature_order = [
    "duration", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts",
    "orig_ip_bytes", "resp_ip_bytes", "total_bytes", "total_packets",
    "bytes_per_packet", "resp_orig_ratio", "packets_per_second",
    "is_telnet_port", "is_ssh_port", "is_http_port", "proto_encoded",
    "hour_of_day", "day_of_week", "orig_port", "resp_port"
]

features = Vectors.dense([float(traffic_1.get(f, 0)) for f in feature_order])
df = spark.createDataFrame([Row(features=features)])
result = model.transform(df).select("prediction", "probability").collect()[0]

prediction = "Malicious" if result["prediction"] == 1 else "Benign"
probs = result["probability"].toArray()

print(f"Input: {traffic_1['orig_port']} → {traffic_1['resp_port']} (TCP)")
print(f"Bytes: {traffic_1['orig_bytes']} sent, {traffic_1['resp_bytes']} received")
print(f"\n**PREDICTION: {prediction}**")
print(f"Malicious probability: {probs[1]:.2%}")
print(f"Benign probability: {probs[0]:.2%}")

# Test Case 2: Okiru pattern
print("\n2. Testing Okiru pattern (ports 5526→37215):")
print("-" * 60)

traffic_2 = {
    "duration": 0.5,
    "orig_bytes": 120,
    "resp_bytes": 200,
    "orig_pkts": 3,
    "resp_pkts": 4,
    "orig_ip_bytes": 180,
    "resp_ip_bytes": 260,
    "orig_port": 5526,
    "resp_port": 37215,
    "proto": "tcp"
}

# Calculate derived features
traffic_2["total_bytes"] = traffic_2["orig_bytes"] + traffic_2["resp_bytes"]
traffic_2["total_packets"] = traffic_2["orig_pkts"] + traffic_2["resp_pkts"]
traffic_2["bytes_per_packet"] = traffic_2["total_bytes"] / traffic_2["total_packets"]
traffic_2["resp_orig_ratio"] = traffic_2["resp_bytes"] / traffic_2["orig_bytes"]
traffic_2["packets_per_second"] = traffic_2["total_packets"] / traffic_2["duration"]
traffic_2["is_telnet_port"] = 0
traffic_2["is_ssh_port"] = 0
traffic_2["is_http_port"] = 0
traffic_2["proto_encoded"] = 1
traffic_2["hour_of_day"] = 12
traffic_2["day_of_week"] = 3

features = Vectors.dense([float(traffic_2.get(f, 0)) for f in feature_order])
df = spark.createDataFrame([Row(features=features)])
result = model.transform(df).select("prediction", "probability").collect()[0]

prediction = "Malicious" if result["prediction"] == 1 else "Benign"
probs = result["probability"].toArray()

print(f"Input: {traffic_2['orig_port']} → {traffic_2['resp_port']} (TCP)")
print(f"Bytes: {traffic_2['orig_bytes']} sent, {traffic_2['resp_bytes']} received")
print(f"\n**PREDICTION: {prediction}**")
print(f"Malicious probability: {probs[1]:.2%}")
print(f"Benign probability: {probs[0]:.2%}")

# Test Case 3: Benign NTP traffic
print("\n3. Testing Benign NTP pattern (port 123):")
print("-" * 60)

traffic_3 = {
    "duration": 0.1,
    "orig_bytes": 76,
    "resp_bytes": 76,
    "orig_pkts": 1,
    "resp_pkts": 1,
    "orig_ip_bytes": 104,
    "resp_ip_bytes": 104,
    "orig_port": 123,
    "resp_port": 123,
    "proto": "udp"
}

# Calculate derived features
traffic_3["total_bytes"] = traffic_3["orig_bytes"] + traffic_3["resp_bytes"]
traffic_3["total_packets"] = traffic_3["orig_pkts"] + traffic_3["resp_pkts"]
traffic_3["bytes_per_packet"] = traffic_3["total_bytes"] / traffic_3["total_packets"]
traffic_3["resp_orig_ratio"] = traffic_3["resp_bytes"] / traffic_3["orig_bytes"]
traffic_3["packets_per_second"] = traffic_3["total_packets"] / traffic_3["duration"]
traffic_3["is_telnet_port"] = 0
traffic_3["is_ssh_port"] = 0
traffic_3["is_http_port"] = 0
traffic_3["proto_encoded"] = 2  # UDP
traffic_3["hour_of_day"] = 12
traffic_3["day_of_week"] = 3

features = Vectors.dense([float(traffic_3.get(f, 0)) for f in feature_order])
df = spark.createDataFrame([Row(features=features)])
result = model.transform(df).select("prediction", "probability").collect()[0]

prediction = "Malicious" if result["prediction"] == 1 else "Benign"
probs = result["probability"].toArray()

print(f"Input: {traffic_3['orig_port']} → {traffic_3['resp_port']} (UDP)")
print(f"Bytes: {traffic_3['orig_bytes']} sent, {traffic_3['resp_bytes']} received")
print(f"\n**PREDICTION: {prediction}**")
print(f"Malicious probability: {probs[1]:.2%}")
print(f"Benign probability: {probs[0]:.2%}")

print("\n" + "="*60)
print("✓ MCP tool testing complete!")
print("="*60)

spark.stop()