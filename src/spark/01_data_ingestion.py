"""
IoT-23 Data Ingestion and Cleaning with PySpark
Loads raw CSV files and creates cleaned Parquet datasets
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, lit, regexp_replace, trim
from pyspark.sql.types import DoubleType, IntegerType, StringType
import os


def create_spark_session():
    """Initialize Spark session with optimized settings"""
    spark = SparkSession.builder \
        .appName("IoT23-DataIngestion") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    print(f"✓ Spark session created: {spark.version}")
    return spark


def load_iot23_data(spark, data_path):
    """Load all IoT-23 CSV files into a single DataFrame"""

    # Find all CSV files
    csv_files = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    print(f"Found {len(csv_files)} CSV files")

    # Load all CSVs
    df = spark.read.csv(
        os.path.join(data_path, "*.csv"),
        header=True,
        inferSchema=False  # We'll cast types manually
    )

    print(f"✓ Loaded {df.count()} total rows")
    return df


def clean_data(df):
    """Clean and preprocess the dataframe"""

    print("Cleaning data...")

    # 1. Cast numeric columns - handle bad values by replacing with NULL first
    df = df.withColumn("duration",
                       when(col("duration").rlike("^[0-9.]+$"), col("duration").cast(DoubleType())).otherwise(
                           lit(None))) \
        .withColumn("`id.orig_p`",
                    when(col("`id.orig_p`").rlike("^[0-9]+$"), col("`id.orig_p`").cast(IntegerType())).otherwise(
                        lit(None))) \
        .withColumn("`id.resp_p`",
                    when(col("`id.resp_p`").rlike("^[0-9]+$"), col("`id.resp_p`").cast(IntegerType())).otherwise(
                        lit(None))) \
        .withColumn("orig_bytes",
                    when(col("orig_bytes").rlike("^[0-9.]+$"), col("orig_bytes").cast(DoubleType())).otherwise(
                        lit(None))) \
        .withColumn("resp_bytes",
                    when(col("resp_bytes").rlike("^[0-9.]+$"), col("resp_bytes").cast(DoubleType())).otherwise(
                        lit(None))) \
        .withColumn("missed_bytes",
                    when(col("missed_bytes").rlike("^[0-9]+$"), col("missed_bytes").cast(IntegerType())).otherwise(
                        lit(None))) \
        .withColumn("orig_pkts",
                    when(col("orig_pkts").rlike("^[0-9]+$"), col("orig_pkts").cast(IntegerType())).otherwise(lit(None))) \
        .withColumn("orig_ip_bytes",
                    when(col("orig_ip_bytes").rlike("^[0-9]+$"), col("orig_ip_bytes").cast(IntegerType())).otherwise(
                        lit(None))) \
        .withColumn("resp_pkts",
                    when(col("resp_pkts").rlike("^[0-9]+$"), col("resp_pkts").cast(IntegerType())).otherwise(lit(None))) \
        .withColumn("resp_ip_bytes",
                    when(col("resp_ip_bytes").rlike("^[0-9]+$"), col("resp_ip_bytes").cast(IntegerType())).otherwise(
                        lit(None)))

    # 2. Handle null values - replace with 0 for numeric columns
    numeric_cols = ["duration", "orig_bytes", "resp_bytes", "missed_bytes",
                    "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes"]

    for col_name in numeric_cols:
        df = df.withColumn(col_name, when(col(col_name).isNull(), 0.0).otherwise(col(col_name)))

    # 3. Clean label columns - remove extra spaces and "(empty)"
    df = df.withColumn("label", trim(regexp_replace(col("label"), r"\(empty\)\s+", "")))
    df = df.withColumn("`detailed-label`", trim(regexp_replace(col("`detailed-label`"), r"\(empty\)\s+", "")))

    # 4. Extract malware family from detailed-label (DO THIS FIRST!)
    # 4. Extract malware family from LABEL column (not detailed-label!)
    df = df.withColumn(
        "malware_family",
        when(col("label").contains("Okiru"), "Okiru")
        .when(col("label").contains("Mirai"), "Mirai")
        .when(col("label").contains("Torii"), "Torii")
        .when(col("label").contains("DDoS"), "DDoS")
        .when(col("label").contains("C&C"), "C&C")
        .when(col("label").contains("PartOfAHorizontalPortScan"), "PortScan")
        .when(col("label").contains("Attack"), "Attack")
        .when(col("label").contains("FileDownload"), "FileDownload")
        .when(col("label").contains("Benign"), "Benign")
        .when(col("label").contains("benign"), "Benign")
        .otherwise("Benign")
    )

    # 5. Create binary label column (Benign vs Malicious) (DO THIS SECOND!)
    df = df.withColumn("is_malicious",
                       when((col("malware_family") == "Benign") |
                            (col("malware_family").isNull()), 0)
                       .otherwise(1))

    # 6. Drop rows with critical nulls
    df = df.dropna(subset=["`id.orig_h`", "`id.resp_h`", "proto"])

    print(f"✓ Cleaned data: {df.count()} rows remaining")
    print(f"✓ Malicious rows: {df.filter(col('is_malicious') == 1).count()}")
    print(f"✓ Benign rows: {df.filter(col('is_malicious') == 0).count()}")

    return df


def create_dataset_splits(df, output_dir):
    """Create three separate datasets for different models"""

    print("\nCreating dataset splits...")

    # 1. Full labeled data (for LightGBM)
    full_data = df
    full_output = os.path.join(output_dir, "full_labeled_data.parquet")
    full_data.write.mode("overwrite").parquet(full_output)
    print(f"✓ Saved full dataset: {full_data.count()} rows → {full_output}")

    # 2. Benign only (for Autoencoder)
    benign_data = df.filter(col("is_malicious") == 0)
    benign_output = os.path.join(output_dir, "benign_only_data.parquet")
    benign_data.write.mode("overwrite").parquet(benign_output)
    print(f"✓ Saved benign dataset: {benign_data.count()} rows → {benign_output}")

    # 3. Malicious only (for K-Means and GAN)
    malicious_data = df.filter(col("is_malicious") == 1)
    malicious_output = os.path.join(output_dir, "malicious_only_data.parquet")
    malicious_data.write.mode("overwrite").parquet(malicious_output)
    print(f"✓ Saved malicious dataset: {malicious_data.count()} rows → {malicious_output}")

    # Print malware family distribution
    print("\nMalware family distribution:")
    malicious_data.groupBy("malware_family").count().orderBy(col("count").desc()).show()


def main():
    """Main execution flow"""

    # Paths - adjust these for your system
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_data_path = os.path.join(project_root, "data", "raw")
    processed_data_path = os.path.join(project_root, "data", "processed")

    # Create output directory if it doesn't exist
    os.makedirs(processed_data_path, exist_ok=True)

    print("=" * 60)
    print("IoT-23 Data Ingestion Pipeline")
    print("=" * 60)

    # Step 1: Create Spark session
    spark = create_spark_session()

    # Step 2: Load data
    df = load_iot23_data(spark, raw_data_path)

    # Step 3: Clean data
    df_cleaned = clean_data(df)

    # Step 4: Create dataset splits
    create_dataset_splits(df_cleaned, processed_data_path)

    print("\n" + "=" * 60)
    print("✓ Data ingestion complete!")
    print("=" * 60)

    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()