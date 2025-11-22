"""
Stage 1: Data Quality Analysis - PySpark Version
Works on both local and Databricks.

Analyzes raw IoT-23 data quality:
- Basic statistics (rows, columns, data types)
- Missing value analysis
- Duplicate detection
- Label distributions
- Capture-level statistics

Returns comprehensive quality report dictionary.
"""

from pyspark.sql import DataFrame, functions as F
from analysis.stage1.utils import (
    get_combined_label_column, parse_binary_label,
    get_numeric_columns, compute_class_distribution,
    compute_missing_percentage
)


def analyze_data_quality(df: DataFrame) -> dict:
    """
    Comprehensive data quality analysis on raw flows.

    Args:
        df: Spark DataFrame with raw IoT-23 data

    Returns:
        dict: Quality statistics and analysis results
    """
    print("\n" + "=" * 70)
    print("DATA QUALITY ANALYSIS")
    print("=" * 70)

    stats = {
        'total_rows': 0,
        'total_columns': 0,
        'numeric_columns': 0,
        'categorical_columns': 0,
        'missing_values': {},
        'duplicate_info': {},
        'label_distribution': {},
        'capture_statistics': {},
        'data_types': {},
        'feasibility': 'GO',
        'reason': 'Data quality is acceptable'
    }

    # ========================================
    # 1. BASIC STATISTICS
    # ========================================

    print("\n[1/6] Computing basic statistics...")

    total_rows = df.count()
    total_cols = len(df.columns)

    stats['total_rows'] = total_rows
    stats['total_columns'] = total_cols

    print(f"   Total rows: {total_rows:,}")
    print(f"   Total columns: {total_cols}")

    if total_rows == 0:
        stats['feasibility'] = 'NO-GO'
        stats['reason'] = 'Dataset is empty'
        return stats

    # ========================================
    # 2. DATA TYPES
    # ========================================

    print("\n[2/6] Analyzing data types...")

    numeric_cols = get_numeric_columns(df)
    categorical_cols = [c for c in df.columns if c not in numeric_cols]

    stats['numeric_columns'] = len(numeric_cols)
    stats['categorical_columns'] = len(categorical_cols)
    stats['numeric_column_list'] = numeric_cols[:20]  # First 20

    # Data type breakdown
    type_counts = {}
    for field in df.schema.fields:
        dtype = str(field.dataType)
        type_counts[dtype] = type_counts.get(dtype, 0) + 1

    stats['data_types'] = type_counts

    print(f"   Numeric columns: {len(numeric_cols)}")
    print(f"   Categorical columns: {len(categorical_cols)}")

    # ========================================
    # 3. MISSING VALUES
    # ========================================

    print("\n[3/6] Analyzing missing values...")

    # Check all columns for missing values
    missing_counts = {}
    for col in df.columns:
        # Wrap column name in backticks if it contains dots or hyphens
        col_ref = F.col(f"`{col}`") if ('.' in col or '-' in col) else F.col(col)
        null_count = df.filter(col_ref.isNull()).count()
        if null_count > 0:
            missing_pct = (null_count / total_rows) * 100
            missing_counts[col] = {
                'count': null_count,
                'percentage': round(missing_pct, 2)
            }

    stats['missing_values'] = missing_counts
    stats['columns_with_missing'] = len(missing_counts)

    if missing_counts:
        print(f"   Columns with missing values: {len(missing_counts)}")

        # Show top 5 columns with most missing
        sorted_missing = sorted(
            missing_counts.items(),
            key=lambda x: x[1]['count'],
            reverse=True
        )[:5]

        print(f"   Top columns with missing data:")
        for col, info in sorted_missing:
            print(f"      {col}: {info['count']:,} ({info['percentage']:.1f}%)")
    else:
        print(f"   ✅ No missing values found")

    # Check if critical columns have missing values
    critical_cols = ['label', 'detailed-label', 'proto', 'service']
    critical_missing = [c for c in critical_cols if c in missing_counts]

    if critical_missing:
        stats['warnings'] = stats.get('warnings', [])
        stats['warnings'].append(f"Critical columns have missing values: {critical_missing}")

    # ========================================
    # 4. DUPLICATES
    # ========================================

    print("\n[4/6] Checking for duplicates...")

    # Sample-based duplicate check (performance optimization for large datasets)
    if total_rows > 100000:
        sample_size = 10000
        sample_df = df.limit(sample_size)
        distinct_count = sample_df.distinct().count()
        duplicate_count = sample_size - distinct_count
        duplicate_pct = (duplicate_count / sample_size) * 100

        stats['duplicate_info'] = {
            'method': 'sample-based',
            'sample_size': sample_size,
            'duplicates_in_sample': duplicate_count,
            'duplicate_percentage': round(duplicate_pct, 2)
        }

        print(f"   Sample-based check ({sample_size:,} rows):")
        print(f"      Duplicates: {duplicate_count:,} ({duplicate_pct:.1f}%)")

    else:
        # Full duplicate check for smaller datasets
        distinct_count = df.distinct().count()
        duplicate_count = total_rows - distinct_count
        duplicate_pct = (duplicate_count / total_rows) * 100

        stats['duplicate_info'] = {
            'method': 'full',
            'total_rows': total_rows,
            'distinct_rows': distinct_count,
            'duplicates': duplicate_count,
            'duplicate_percentage': round(duplicate_pct, 2)
        }

        print(f"   Full check:")
        print(f"      Distinct rows: {distinct_count:,}")
        print(f"      Duplicates: {duplicate_count:,} ({duplicate_pct:.1f}%)")

    if stats['duplicate_info']['duplicate_percentage'] > 10:
        stats['warnings'] = stats.get('warnings', [])
        stats['warnings'].append(
            f"High duplicate rate: {stats['duplicate_info']['duplicate_percentage']:.1f}%"
        )

    # ========================================
    # 5. LABEL DISTRIBUTIONS
    # ========================================

    print("\n[5/6] Analyzing label distributions...")

    # Parse binary labels
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_labeled = df.withColumn('parsed_binary_label', parse_binary_label(combined_col))

    # Binary distribution
    binary_dist = compute_class_distribution(df_labeled, 'parsed_binary_label')
    stats['label_distribution']['binary'] = binary_dist

    print(f"   Binary distribution:")
    for label, count in binary_dist.items():
        pct = (count / total_rows) * 100
        print(f"      {label}: {count:,} ({pct:.1f}%)")

    # Raw label distribution (from 'label' column)
    if 'label' in df.columns:
        raw_label_dist = df.groupBy('label').count().collect()
        raw_dist = {row['label']: int(row['count']) for row in raw_label_dist if row['label']}
        stats['label_distribution']['raw_label'] = raw_dist

    # Detailed label distribution (from 'detailed-label' column)
    if 'detailed-label' in df.columns:
        detailed_dist = df.groupBy('detailed-label').count().collect()
        detailed_dict = {
            row['detailed-label']: int(row['count'])
            for row in detailed_dist
            if row['detailed-label']
        }
        stats['label_distribution']['detailed_label'] = detailed_dict
        stats['label_distribution']['num_detailed_labels'] = len(detailed_dict)

    # ========================================
    # 6. CAPTURE-LEVEL STATISTICS
    # ========================================

    print("\n[6/6] Analyzing per-capture statistics...")

    if 'Source_Folder' in df.columns:
        capture_stats = df.groupBy('Source_Folder').agg(
            F.count('*').alias('flow_count'),
            F.countDistinct(F.col('`id.orig_h`')).alias('unique_src_ips'),
            F.countDistinct(F.col('`id.resp_h`')).alias('unique_dst_ips')
        ).collect()

        capture_dict = {}
        for row in capture_stats:
            folder = row['Source_Folder']
            capture_dict[folder] = {
                'flows': int(row['flow_count']),
                'unique_src_ips': int(row['unique_src_ips']),
                'unique_dst_ips': int(row['unique_dst_ips'])
            }

        stats['capture_statistics'] = capture_dict
        stats['num_captures'] = len(capture_dict)

        print(f"   Number of captures: {len(capture_dict)}")
        print(
            f"   Flows per capture range: {min(c['flows'] for c in capture_dict.values()):,} - {max(c['flows'] for c in capture_dict.values()):,}")

    # ========================================
    # 7. FEASIBILITY ASSESSMENT
    # ========================================

    print("\n[7/7] Assessing overall feasibility...")

    # Critical issues that would block model training
    if total_rows < 1000:
        stats['feasibility'] = 'NO-GO'
        stats['reason'] = f'Insufficient data: only {total_rows:,} rows (need 1000+)'
    elif len(binary_dist) < 2:
        stats['feasibility'] = 'NO-GO'
        stats['reason'] = 'Only one class present (need both Benign and Malicious)'
    elif len(numeric_cols) < 5:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Very few numeric features: {len(numeric_cols)} (expected 10+)'

    # Warnings (don't block, but noteworthy)
    if 'warnings' not in stats:
        stats['warnings'] = []

    if stats['columns_with_missing'] > total_cols * 0.5:
        stats['warnings'].append(
            f"Over 50% of columns have missing values ({stats['columns_with_missing']}/{total_cols})"
        )

    # ========================================
    # SUMMARY
    # ========================================

    print("\n" + "=" * 70)
    print("DATA QUALITY SUMMARY")
    print("=" * 70)
    print(f"Dataset size: {total_rows:,} rows × {total_cols} columns")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print(f"Columns with missing: {stats['columns_with_missing']}")
    print(f"Duplicate rate: {stats['duplicate_info'].get('duplicate_percentage', 0):.1f}%")
    print(f"\nFeasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"Reason: {stats['reason']}")

    if stats['warnings']:
        print(f"\nWarnings ({len(stats['warnings'])}):")
        for warning in stats['warnings']:
            print(f"   - {warning}")

    print("=" * 70)

    return stats


if __name__ == "__main__":
    """Standalone testing"""
    from config import Config
    from analysis.stage1.utils import load_raw_data

    Config.ensure_output_dirs()
    spark = Config.get_spark_session("Stage1-DataQuality-Test")

    try:
        df = load_raw_data(spark)
        stats = analyze_data_quality(df)

        # Save report
        from analysis.stage1.utils import save_json_report

        output_path = os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'data_quality.json')
        save_json_report(stats, output_path)

    finally:
        if not Config.is_databricks():
            spark.stop()
