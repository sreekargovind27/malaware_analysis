"""
Feature Engineering for IoT-23 Dataset
Creates ML-ready features from cleaned network traffic data
"""

from pyspark.sql import SparkSession
from pyspark.sql.functions import col, when, hour, dayofweek, from_unixtime, lit
from pyspark.sql.types import DoubleType, IntegerType
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline
import os


def create_spark_session():
    """Initialize Spark session"""
    spark = SparkSession.builder \
        .appName("IoT23-FeatureEngineering") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.shuffle.partitions", "8") \
        .getOrCreate()

    print(f"✓ Spark session created: {spark.version}")
    return spark


def load_processed_data(spark, data_path):
    """Load cleaned Parquet data"""

    full_data_path = os.path.join(data_path, "full_labeled_data.parquet")

    df = spark.read.parquet(full_data_path)
    print(f"✓ Loaded {df.count()} rows from processed data")

    return df


def create_features(df):
    """Create engineered features for ML models"""

    print("\nCreating features...")

    # 1. Temporal features from Unix timestamp - handle NULLs gracefully
    df = df.withColumn("ts_double",
                       when(col("ts").rlike("^[0-9.]+$"),
                            col("ts").cast(DoubleType()))
                       .otherwise(lit(None)))

    # Only create time features if timestamp is valid, otherwise use defaults
    df = df.withColumn("timestamp",
                       when(col("ts_double").isNotNull(),
                            from_unixtime(col("ts_double")))
                       .otherwise(lit(None)))

    df = df.withColumn("hour_of_day",
                       when(col("timestamp").isNotNull(),
                            hour(col("timestamp")))
                       .otherwise(12))  # Default to noon

    df = df.withColumn("day_of_week",
                       when(col("timestamp").isNotNull(),
                            dayofweek(col("timestamp")))
                       .otherwise(3))  # Default to Wednesday

    # 2. Traffic volume features
    df = df.withColumn("total_bytes", col("orig_bytes") + col("resp_bytes"))
    df = df.withColumn("total_packets", col("orig_pkts") + col("resp_pkts"))

    # 3. Ratio features (handle division by zero)
    df = df.withColumn("bytes_per_packet",
                       when(col("total_packets") > 0,
                            col("total_bytes") / col("total_packets"))
                       .otherwise(0.0))

    df = df.withColumn("resp_orig_ratio",
                       when(col("orig_bytes") > 0,
                            col("resp_bytes") / col("orig_bytes"))
                       .otherwise(0.0))

    # 4. Connection duration features
    df = df.withColumn("packets_per_second",
                       when(col("duration") > 0,
                            col("total_packets") / col("duration"))
                       .otherwise(0.0))

    # 5. Port-based features - safely cast ports, handle invalid data
    df = df.withColumn("orig_port",
                       when(col("`id.orig_p`").rlike("^[0-9]+$"),
                            col("`id.orig_p`").cast(IntegerType()))
                       .otherwise(0))

    df = df.withColumn("resp_port",
                       when(col("`id.resp_p`").rlike("^[0-9]+$"),
                            col("`id.resp_p`").cast(IntegerType()))
                       .otherwise(0))

    # Create port-based binary features
    df = df.withColumn("is_telnet_port",
                       when((col("orig_port") == 23) | (col("resp_port") == 23), 1)
                       .otherwise(0))

    df = df.withColumn("is_ssh_port",
                       when((col("orig_port") == 22) | (col("resp_port") == 22), 1)
                       .otherwise(0))

    df = df.withColumn("is_http_port",
                       when((col("orig_port") == 80) | (col("resp_port") == 80), 1)
                       .otherwise(0))

    # 6. Protocol encoding (tcp=1, udp=2, icmp=3, other=0)
    df = df.withColumn("proto_encoded",
                       when(col("proto") == "tcp", 1)
                       .when(col("proto") == "udp", 2)
                       .when(col("proto") == "icmp", 3)
                       .otherwise(0))

    print("✓ Created temporal, volume, ratio, and port-based features")

    return df

def prepare_ml_features(df):
    """Prepare feature vectors for ML models"""

    print("\nPreparing ML feature vectors...")

    # Select numeric features for ML
    feature_columns = [
        "duration",
        "orig_bytes",
        "resp_bytes",
        "orig_pkts",
        "resp_pkts",
        "orig_ip_bytes",
        "resp_ip_bytes",
        "total_bytes",
        "total_packets",
        "bytes_per_packet",
        "resp_orig_ratio",
        "packets_per_second",
        "is_telnet_port",
        "is_ssh_port",
        "is_http_port",
        "proto_encoded",
        "hour_of_day",
        "day_of_week",
        "orig_port",
        "resp_port"
    ]

    # Assemble features into vector
    assembler = VectorAssembler(
        inputCols=feature_columns,
        outputCol="features_raw",
        handleInvalid="skip"
    )

    # Scale features
    scaler = StandardScaler(
        inputCol="features_raw",
        outputCol="features",
        withStd=True,
        withMean=True
    )

    # Create pipeline
    pipeline = Pipeline(stages=[assembler, scaler])

    # Fit and transform
    model = pipeline.fit(df)
    df_features = model.transform(df)

    print(f"✓ Created feature vectors with {len(feature_columns)} features")
    print(f"✓ Features: {', '.join(feature_columns[:5])}... (and {len(feature_columns) - 5} more)")

    return df_features, feature_columns


def save_feature_data(df, output_dir):
    """Save feature-engineered data"""

    print("\nSaving feature-engineered data...")

    # Select important columns to keep
    columns_to_keep = [
        "is_malicious",
        "malware_family",
        "`id.orig_h`",
        "`id.resp_h`",
        "orig_port",
        "resp_port",
        "proto",
        "features"
    ]

    df_final = df.select(*columns_to_keep)

    # Save full featured dataset
    output_path = os.path.join(output_dir, "ml_ready_data.parquet")
    df_final.write.mode("overwrite").parquet(output_path)

    count = df_final.count()
    print(f"✓ Saved ML-ready data: {count} rows")
    print(f"✓ Output: {output_path}")

    # Show sample
    print("\nSample of feature-engineered data:")
    df_final.select("is_malicious", "malware_family", "proto", "orig_port", "resp_port").show(5)

    return df_final


def main():
    """Main execution flow"""

    # Paths
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_data_path = os.path.join(project_root, "data", "processed")

    print("=" * 60)
    print("IoT-23 Feature Engineering Pipeline")
    print("=" * 60)

    # Step 1: Create Spark session
    spark = create_spark_session()

    # Step 2: Load processed data
    df = load_processed_data(spark, processed_data_path)

    # Step 3: Create features
    df_features = create_features(df)

    # Step 4: Prepare ML features
    df_ml, feature_list = prepare_ml_features(df_features)

    # Step 5: Save
    df_final = save_feature_data(df_ml, processed_data_path)

    print("\n" + "=" * 60)
    print("✓ Feature engineering complete!")
    print(f"✓ Total features: {len(feature_list)}")
    print("=" * 60)

    # Stop Spark
    spark.stop()


if __name__ == "__main__":
    main()