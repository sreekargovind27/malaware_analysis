"""
FIXED VERSION - Shared utility functions for Stage 1 feasibility analysis (PySpark).
"""

import os

from pyspark.sql import functions as F
from pyspark.sql.types import DoubleType, IntegerType

from config import Config


def load_raw_data(spark):
    """
    Loads all raw CSV files using PySpark, inferring the schema.
    This is for analysis only and is robust to minor inconsistencies.
    """
    print("\n" + "=" * 70)
    print("LOADING RAW DATA (PySpark)")
    print("=" * 70)

    input_dir = Config.RAW_DIR_ORIGINAL

    if not os.path.exists(input_dir):
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        raise ValueError(f"No CSV files found in {input_dir}")

    print(f"Found {len(csv_files)} CSV files in {input_dir}")

    # ✅ FINAL FIX: Use inferSchema=True. This is robust enough for analysis
    # and correctly handles files with slightly different headers.
    df = spark.read.csv(
        f"{input_dir}/*.csv",
        header=True,
        inferSchema=True,
        comment='#'
    )

    # --- Robust Casting and Renaming ---
    print("  Manually casting and renaming columns...")

    # Rename columns with dots to use underscores for consistency
    df = df.withColumnRenamed('id.orig_h', 'id_orig_h') \
        .withColumnRenamed('id.orig_p', 'id_orig_p') \
        .withColumnRenamed('id.resp_h', 'id_resp_h') \
        .withColumnRenamed('id.resp_p', 'id_resp_p')

    # Cast numeric columns that might be read as strings
    df = df.withColumn('ts', F.col('ts').cast(DoubleType()))
    df = df.withColumn('duration', F.col('duration').cast(DoubleType()))
    df = df.withColumn('orig_bytes', F.col('orig_bytes').cast(IntegerType()))
    df = df.withColumn('resp_bytes', F.col('resp_bytes').cast(IntegerType()))
    df = df.withColumn('missed_bytes', F.col('missed_bytes').cast(IntegerType()))
    df = df.withColumn('orig_pkts', F.col('orig_pkts').cast(IntegerType()))
    df = df.withColumn('orig_ip_bytes', F.col('orig_ip_bytes').cast(IntegerType()))
    df = df.withColumn('resp_pkts', F.col('resp_pkts').cast(IntegerType()))
    df = df.withColumn('resp_ip_bytes', F.col('resp_ip_bytes').cast(IntegerType()))
    df = df.withColumn('id_orig_p', F.col('id_orig_p').cast(IntegerType()))
    df = df.withColumn('id_resp_p', F.col('id_resp_p').cast(IntegerType()))

    total_rows = df.count()
    print(f"✓ Total rows loaded: {total_rows:,}")
    return df


def get_combined_label_column(label_col_name, detailed_label_col_name):
    """
    Combine label and detailed-label columns (mimics Stage 2 approach).
    Returns a Column expression with both columns concatenated.
    """
    label_col = F.coalesce(F.col(label_col_name), F.lit(''))
    detailed_col = F.coalesce(F.col(detailed_label_col_name), F.lit(''))
    return F.lower(F.concat(label_col, F.lit(' '), detailed_col))


def parse_binary_label(combined_column_expr):
    """
    Parse binary label (Benign/Malicious) from detailed-label column.
    Returns a Column expression for use in withColumn.
    
    FIXED VERSION:
    - Properly handles NULL/empty values
    - Explicitly checks for 'benign' keyword
    - Returns 'Unknown' for unparseable values
    
    Args:
        column_name: String name of the column (e.g., 'detailed-label')
        
    Returns:
        Column: 'Benign', 'Malicious', or None (for NULL/empty)
        :param combined_column_expr:
    """
    col_ref = combined_column_expr

    return F.when(
        col_ref.isNull() | (F.trim(col_ref) == '') | (col_ref == '-'),
        F.lit(None)  # ✅ Keep NULL as NULL instead of forcing to 'Benign'
    ).when(
        F.lower(col_ref).contains('malicious'),
        F.lit('Malicious')
    ).when(
        F.lower(col_ref).contains('benign'),
        F.lit('Benign')
    ).otherwise(
        F.lit('Unknown')  # ✅ For unparseable values
    )


def parse_attack_type(combined_column_expr):
    """
    Extract attack type from detailed-label column.
    Returns a Column expression for use in withColumn.

    FIXED VERSION:
    - Malware families are NOT attack types - return 'Malware' for them
    - Reordered conditions so generic 'attack' comes LAST
    - Consistent lowercase comparisons
    - Better NULL handling

    Args:
        combined_column_expr: Column expression (result of get_combined_label_column)

    Returns:
        Column: Attack type or 'Benign' or None
    """
    col_ref = combined_column_expr
    detailed_lower = F.lower(col_ref)

    return F.when(
        col_ref.isNull() | (F.trim(col_ref) == '') | (F.lower(col_ref) == '-'),
        F.lit(None)  # ✅ NULL for missing data
    ).when(
        detailed_lower.contains('benign'),
        F.lit('Benign')
    ).when(
        # ✅ CRITICAL FIX: Check for malware family names FIRST and return 'Malware'
        detailed_lower.contains('mirai') |
        detailed_lower.contains('okiru') |
        detailed_lower.contains('torii') |
        detailed_lower.contains('kenjiro') |
        detailed_lower.contains('gagfyt') |
        detailed_lower.contains('muhstik') |
        detailed_lower.contains('hajime') |
        detailed_lower.contains('hide and seek') |
        detailed_lower.contains('hakai') |
        detailed_lower.contains('ircbot') |
        detailed_lower.contains('trojan'),
        F.lit('Malware')  # ✅ Generic label for flows with family names
    ).when(
        # Specific attacks BEFORE generic 'attack' check
        detailed_lower.contains('portscan') | detailed_lower.contains('partofahorizontalportscan'),
        F.lit('PortScan')
    ).when(
        detailed_lower.contains('ddos'),
        F.lit('DDoS')
    ).when(
        detailed_lower.contains('c&c') | detailed_lower.contains('c2'),
        F.lit('C&C')
    ).when(
        detailed_lower.contains('filedownload') | detailed_lower.contains('file download'),
        F.lit('FileDownload')
    ).when(
        detailed_lower.contains('attack'),  # ✅ MOVED TO END - generic catch-all
        F.lit('Attack')
    ).otherwise(
        F.lit('Unknown')  # ✅ For unparseable malicious traffic
    )


def get_malware_family_udf():
    """
    Create UDF to map Source_Folder to malware family.
    Uses Config.FILENAME_TO_FAMILY_MAP.
    
    Returns:
        UDF function
    """
    from pyspark.sql.types import StringType

    family_map = Config.FILENAME_TO_FAMILY_MAP

    def lookup_family(source_folder):
        # ✅ Return None instead of 'Unknown' for missing mappings
        return family_map.get(source_folder, None)

    return F.udf(lookup_family, StringType())


def save_json_report(data, filepath):
    """
    Save analysis results as JSON file.
    
    Args:
        data: Dictionary to save
        filepath: Output file path
    """
    import json

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # Convert int64 to int for JSON serialization
    def convert_types(obj):
        if isinstance(obj, dict):
            return {k: convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(item) for item in obj]
        elif hasattr(obj, 'item'):  # numpy types
            return obj.item()
        else:
            return obj

    data_clean = convert_types(data)

    with open(filepath, 'w') as f:
        json.dump(data_clean, f, indent=2)

    print(f"  ✓ Saved: {os.path.basename(filepath)}")


def calculate_missing_percentage(df, column):
    """
    Calculate percentage of missing values in a column.
    
    Args:
        df: Spark DataFrame
        column: Column name
        
    Returns:
        float: Percentage of missing values (0-100)
    """
    if column not in df.columns:
        return 100.0

    total = df.count()
    if total == 0:
        return 0.0

    missing_count = df.filter(F.col(column).isNull()).count()

    return round((missing_count / total) * 100, 2)


def get_class_distribution(df, column):
    """
    Get distribution of values in a column.
    
    Args:
        df: Spark DataFrame
        column: Column name
        
    Returns:
        dict: {value: count} dictionary
    """
    if column not in df.columns:
        return {}

    distribution = df.groupBy(column).count().collect()
    return {row[column]: int(row['count']) for row in distribution}
