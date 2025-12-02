from pyspark.sql import SparkSession

spark = SparkSession.builder.appName('check').getOrCreate()

# Load original processed data
df_orig = spark.read.parquet('data/processed/full_labeled_data.parquet')
print('Original data:')
df_orig.groupBy('is_malicious').count().show()

# Load ML-ready data
df_ml = spark.read.parquet('data/processed/ml_ready_data.parquet')
print('After feature engineering:')
df_ml.groupBy('is_malicious').count().show()

spark.stop()