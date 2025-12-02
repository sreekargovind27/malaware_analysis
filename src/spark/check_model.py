"""
Check model performance and data distribution
"""

from pyspark.sql import SparkSession
from pyspark.ml.classification import GBTClassificationModel
import os

spark = SparkSession.builder \
    .appName("Model-Check") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

# Load model
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
model_path = os.path.join(project_root, "models", "lightgbm_classifier")
model = GBTClassificationModel.load(model_path)

print("Model Configuration:")
print(f"Number of trees: {model.getNumTrees}")
print(f"Max depth: {model.getMaxDepth()}")
print(f"Learning rate: {model.getStepSize()}")

# Load test data and check predictions
data_path = os.path.join(project_root, "data", "processed", "ml_ready_data.parquet")
df = spark.read.parquet(data_path)

print(f"\nTotal rows: {df.count()}")
print("\nActual label distribution:")
df.groupBy("is_malicious").count().show()

# Make predictions on sample
sample = df.limit(1000)
predictions = model.transform(sample)

print("\nPredicted distribution (sample of 1000):")
predictions.groupBy("prediction").count().show()

print("\nPrediction vs Actual:")
predictions.groupBy("is_malicious", "prediction").count().show()

# Check feature importance
print("\nFeature Importances:")
print(model.featureImportances)

spark.stop()