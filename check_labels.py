from pyspark.sql import SparkSession
from pyspark.sql.functions import col
import glob

spark = SparkSession.builder.appName('check_labels').getOrCreate()

# Load raw data
csv_files = glob.glob('data/raw/*.csv')
df = spark.read.csv(csv_files, header=True, inferSchema=True)

print("Unique values in 'label' column:")
df.select("label").distinct().show(50, truncate=False)

print("\nUnique values in 'detailed-label' column:")
df.select("detailed-label").distinct().show(50, truncate=False)

print("\nSample detailed-label values:")
df.select("detailed-label").filter(col("detailed-label").isNotNull()).show(20, truncate=False)

spark.stop()