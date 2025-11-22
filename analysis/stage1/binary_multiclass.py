"""
Stage 1: Binary and Multiclass Classification Feasibility - PySpark Version
Works on both local and Databricks.

Analyzes:
- Binary classification: Benign vs Malicious
- Multiclass classification: Attack type detection

Returns feasibility assessments (GO/WARNING/NO-GO).
"""

from pyspark.sql import DataFrame, functions as F
from analysis.stage1.utils import (
    get_combined_label_column, parse_binary_label,
    parse_attack_type, get_numeric_columns
)


def analyze_binary_classification(df: DataFrame) -> dict:
    """
    Analyze feasibility for binary classification (Benign vs Malicious).

    Args:
        df: Spark DataFrame with raw IoT-23 data

    Returns:
        dict: Binary classification feasibility stats
    """
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION FEASIBILITY")
    print("=" * 70)

    # Parse labels from detailed-label column
    print("  Parsing binary labels from 'detailed-label' column...")
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_labeled = df.withColumn('parsed_label', parse_binary_label(combined_col))

    # Get class counts
    class_counts = df_labeled.groupBy('parsed_label').count().collect()
    class_dict = {row['parsed_label']: int(row['count']) for row in class_counts}

    benign_count = class_dict.get('Benign', 0)
    malicious_count = class_dict.get('Malicious', 0)
    total = benign_count + malicious_count

    # Get numeric features
    feature_cols = get_numeric_columns(df_labeled)

    stats = {
        'benign_samples': benign_count,
        'malicious_samples': malicious_count,
        'total_samples': total,
        'benign_percentage': round((benign_count / total) * 100, 2) if total > 0 else 0,
        'malicious_percentage': round((malicious_count / total) * 100, 2) if total > 0 else 0,
        'imbalance_ratio': round(
            max(benign_count, malicious_count) / min(benign_count, malicious_count), 2
        ) if min(benign_count, malicious_count) > 0 else float('inf'),
        'missing_per_class': {},
        'numeric_feature_count': len(feature_cols),
        'numeric_features': feature_cols[:20],  # First 20 features
        'feasibility': 'GO',
        'reason': 'Sufficient samples and acceptable balance'
    }

    # Missing values per class
    print("  Calculating missing values per class...")
    for label in ['Benign', 'Malicious']:
        label_df = df_labeled.filter(F.col('parsed_label') == label)
        label_count = label_df.count()

        if label_count > 0:
            # Count nulls across all columns (handle dotted/hyphenated column names)
            null_count = sum([
                label_df.filter(
                    (F.col(f"`{col}`") if ('.' in col or '-' in col) else F.col(col)).isNull()
                ).count()
                for col in df.columns
            ])

            total_cells = label_count * len(df.columns)
            missing_pct = round((null_count / total_cells) * 100, 2) if total_cells > 0 else 0
            stats['missing_per_class'][label] = missing_pct

    # Feasibility checks
    if benign_count < 1000 or malicious_count < 1000:
        stats['feasibility'] = 'NO-GO'
        stats['reason'] = (
            f'Insufficient samples per class (need >1000 each). '
            f'Benign: {benign_count}, Malicious: {malicious_count}'
        )
    elif stats['imbalance_ratio'] > 99:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Severe class imbalance ({stats["imbalance_ratio"]}:1 ratio)'
    elif stats['missing_per_class'] and max(stats['missing_per_class'].values()) > 50:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = 'High missing percentage (>50%) in one or more classes'

    # Summary
    print(f"\n✅ Benign: {benign_count:,} ({stats['benign_percentage']}%)")
    print(f"✅ Malicious: {malicious_count:,} ({stats['malicious_percentage']}%)")
    print(f"✅ Imbalance ratio: {stats['imbalance_ratio']}:1")
    print(f"✅ Numeric features: {len(feature_cols)}")
    print(f"✅ Feasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"   Reason: {stats['reason']}")

    return stats


def analyze_multiclass_classification(df: DataFrame) -> dict:
    """
    Analyze feasibility for multiclass attack type classification.

    Args:
        df: Spark DataFrame with raw IoT-23 data

    Returns:
        dict: Multiclass classification feasibility stats
    """
    print("\n" + "=" * 70)
    print("MULTICLASS ATTACK TYPE FEASIBILITY")
    print("=" * 70)

    # Parse labels
    print("  Parsing attack types from 'detailed-label' column...")
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_labeled = df.withColumn('parsed_label', parse_binary_label(combined_col))
    df_labeled = df_labeled.withColumn('attack_type', parse_attack_type(combined_col))

    # Filter only malicious (multiclass only on malicious traffic)
    malicious_df = df_labeled.filter(F.col('parsed_label') == 'Malicious')
    malicious_count = malicious_df.count()

    print(f"  Filtered to {malicious_count:,} malicious samples")

    if malicious_count == 0:
        return {
            'feasibility': 'NO-GO',
            'reason': 'No malicious samples found',
            'total_malicious_samples': 0
        }

    # Get attack type distribution
    attack_counts = malicious_df.groupBy('attack_type').count().collect()
    attack_dict = {row['attack_type']: int(row['count']) for row in attack_counts}

    # Identify rare classes
    rare_threshold = 100
    rare_classes = {k: v for k, v in attack_dict.items() if v < rare_threshold}

    # Calculate missing values per attack type
    # Calculate missing values per attack type
    print("  Calculating missing values per attack type...")
    missing_per_class = {}
    for attack_type in attack_dict.keys():
        attack_df = malicious_df.filter(F.col('attack_type') == attack_type)
        attack_count = attack_df.count()

        if attack_count > 0:
            null_count = sum([
                attack_df.filter(
                    (F.col(f"`{col}`") if ('.' in col or '-' in col) else F.col(col)).isNull()
                ).count()
                for col in df.columns
            ])

            total_cells = attack_count * len(df.columns)
            missing_pct = round((null_count / total_cells) * 100, 2) if total_cells > 0 else 0
            missing_per_class[attack_type] = missing_pct

    # Get numeric features
    feature_cols = get_numeric_columns(malicious_df)

    stats = {
        'total_malicious_samples': malicious_count,
        'num_attack_types': len(attack_dict),
        'attack_type_distribution': attack_dict,
        'rare_classes': rare_classes,
        'num_rare_classes': len(rare_classes),
        'missing_per_class': missing_per_class,
        'numeric_feature_count': len(feature_cols),
        'numeric_features': feature_cols[:20],
        'feasibility': 'GO',
        'reason': 'Sufficient samples across attack types'
    }

    # Feasibility checks
    if malicious_count < 5000:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Low malicious sample count: {malicious_count:,} (prefer 5000+)'
    elif len(attack_dict) < 3:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Very few attack types: {len(attack_dict)} (prefer 5+)'
    elif len(rare_classes) > len(attack_dict) * 0.5:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = (
            f'{len(rare_classes)} out of {len(attack_dict)} attack types have <{rare_threshold} samples'
        )

    # Summary
    print(f"\n✅ Total malicious samples: {malicious_count:,}")
    print(f"✅ Number of attack types: {len(attack_dict)}")
    print(f"✅ Numeric features: {len(feature_cols)}")

    print(f"\n📊 Attack type distribution:")
    sorted_attacks = sorted(attack_dict.items(), key=lambda x: x[1], reverse=True)
    for attack_type, count in sorted_attacks[:10]:  # Top 10
        pct = (count / malicious_count) * 100
        print(f"   {attack_type}: {count:,} ({pct:.1f}%)")

    if len(sorted_attacks) > 10:
        print(f"   ... and {len(sorted_attacks) - 10} more")

    if rare_classes:
        print(f"\n⚠️  Rare classes (<{rare_threshold} samples): {len(rare_classes)}")
        for attack_type, count in sorted(rare_classes.items(), key=lambda x: x[1]):
            print(f"   {attack_type}: {count}")

    print(f"\n✅ Feasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"   Reason: {stats['reason']}")

    return stats


if __name__ == "__main__":
    """Standalone testing"""
    import os
    from config import Config
    from analysis.stage1.utils import load_raw_data, save_json_report

    Config.ensure_output_dirs()
    spark = Config.get_spark_session("Stage1-BinaryMulticlass-Test")

    try:
        df = load_raw_data(spark)

        # Binary classification analysis
        binary_stats = analyze_binary_classification(df)
        binary_path = os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'binary_feasibility.json')
        save_json_report(binary_stats, binary_path)

        # Multiclass classification analysis
        multiclass_stats = analyze_multiclass_classification(df)
        multiclass_path = os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'multiclass_feasibility.json')
        save_json_report(multiclass_stats, multiclass_path)

    finally:
        if not Config.is_databricks():
            spark.stop()
