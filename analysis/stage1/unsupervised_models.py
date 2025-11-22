"""
Stage 1: Unsupervised Models Feasibility - PySpark Version
Works on both local and Databricks.

Analyzes feasibility for:
- Autoencoder (anomaly detection)
- K-means clustering (pattern discovery)
- GAN (data augmentation for rare classes)

Returns recommendations for each unsupervised approach.
"""

from pyspark.sql import DataFrame, functions as F
from analysis.stage1.utils import (
    get_combined_label_column, parse_binary_label,
    parse_attack_type, parse_malware_family,
    get_numeric_columns
)


def analyze_autoencoder_feasibility(df: DataFrame) -> dict:
    """
    Analyze feasibility for autoencoder-based anomaly detection.

    Args:
        df: Spark DataFrame with raw IoT-23 data

    Returns:
        dict: Autoencoder feasibility stats
    """
    print("\n" + "=" * 70)
    print("AUTOENCODER FEASIBILITY (Anomaly Detection)")
    print("=" * 70)

    # Parse binary labels
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
        'total_samples': total,
        'benign_samples': benign_count,
        'malicious_samples': malicious_count,
        'numeric_feature_count': len(feature_cols),
        'numeric_features': feature_cols[:20],
        'feasibility': 'GO',
        'reason': 'Sufficient data for autoencoder training'
    }

    # Feasibility assessment
    if benign_count < 5000:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = (
            f'Low benign sample count: {benign_count:,}. '
            f'Autoencoders work best with 10K+ benign samples for training.'
        )
    elif len(feature_cols) < 10:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = (
            f'Few numeric features: {len(feature_cols)}. '
            f'Autoencoders work better with 20+ features.'
        )

    # Recommendations
    stats['recommendations'] = []

    if benign_count >= 10000:
        stats['recommendations'].append(
            'Strong candidate: Train autoencoder on benign traffic for anomaly detection'
        )
    elif benign_count >= 5000:
        stats['recommendations'].append(
            'Moderate candidate: Autoencoder possible but may benefit from augmentation'
        )
    else:
        stats['recommendations'].append(
            'Weak candidate: Consider collecting more benign samples'
        )

    # Use case
    stats['use_case'] = (
        'Train autoencoder on benign traffic only. '
        'High reconstruction error indicates potential malicious activity.'
    )

    # Summary
    print(f"\n✅ Total samples: {total:,}")
    print(f"✅ Benign samples: {benign_count:,} (training data)")
    print(f"✅ Malicious samples: {malicious_count:,} (test anomalies)")
    print(f"✅ Numeric features: {len(feature_cols)}")
    print(f"\n✅ Feasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"   Reason: {stats['reason']}")

    print(f"\n💡 Recommendations:")
    for rec in stats['recommendations']:
        print(f"   - {rec}")

    return stats


def analyze_clustering_feasibility(df: DataFrame) -> dict:
    """
    Analyze feasibility for K-means clustering (pattern discovery).

    Args:
        df: Spark DataFrame with raw IoT-23 data

    Returns:
        dict: Clustering feasibility stats
    """
    print("\n" + "=" * 70)
    print("K-MEANS CLUSTERING FEASIBILITY (Pattern Discovery)")
    print("=" * 70)

    total_count = df.count()

    # Get numeric features
    feature_cols = get_numeric_columns(df)

    # Parse labels for reference
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_labeled = df.withColumn('parsed_label', parse_binary_label(combined_col))

    class_counts = df_labeled.groupBy('parsed_label').count().collect()
    class_dict = {row['parsed_label']: int(row['count']) for row in class_counts}

    stats = {
        'total_samples': total_count,
        'numeric_feature_count': len(feature_cols),
        'numeric_features': feature_cols[:20],
        'binary_distribution': class_dict,
        'feasibility': 'GO',
        'reason': 'Sufficient data for clustering analysis'
    }

    # Feasibility assessment
    if total_count < 5000:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = (
            f'Low sample count: {total_count:,}. '
            f'Clustering works better with 10K+ samples.'
        )
    elif len(feature_cols) < 5:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = (
            f'Very few features: {len(feature_cols)}. '
            f'Clustering requires multiple dimensions.'
        )

    # Recommendations
    stats['recommendations'] = [
        'Use K-means to discover natural traffic patterns',
        'Compare clusters against known labels (Benign/Malicious)',
        'Useful for: behavior profiling, anomaly detection, network segmentation'
    ]

    # Suggested K values
    if total_count >= 10000:
        stats['suggested_k_values'] = [3, 5, 7, 10, 15]
        stats['recommendations'].append(
            'Try K values: 3-15. Use elbow method or silhouette score for optimal K.'
        )
    else:
        stats['suggested_k_values'] = [3, 5, 7]
        stats['recommendations'].append(
            'Try K values: 3-7 for smaller dataset.'
        )

    # Summary
    print(f"\n✅ Total samples: {total_count:,}")
    print(f"✅ Numeric features: {len(feature_cols)}")
    print(f"✅ Binary distribution: {class_dict}")
    print(f"\n✅ Feasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"   Reason: {stats['reason']}")

    print(f"\n💡 Recommendations:")
    for rec in stats['recommendations']:
        print(f"   - {rec}")

    return stats


def analyze_gan_feasibility_for_all_tasks(df: DataFrame) -> dict:
    """
    Analyze GAN feasibility for data augmentation across all tasks.

    Checks:
    - Binary classification (Benign/Malicious)
    - Multiclass classification (Attack types)
    - Malware family classification

    Args:
        df: Spark DataFrame with raw IoT-23 data

    Returns:
        dict: Comprehensive GAN feasibility stats for all tasks
    """
    print("\n" + "=" * 70)
    print("GAN FEASIBILITY (Data Augmentation)")
    print("=" * 70)

    comprehensive_stats = {
        'binary_classification': {},
        'multiclass_attack_type': {},
        'malware_family': {},
        'overall_recommendation': ''
    }

    # Parse all labels
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_labeled = df.withColumn('parsed_label', parse_binary_label(combined_col))
    df_labeled = df_labeled.withColumn('attack_type', parse_attack_type(combined_col))
    df_labeled = df_labeled.withColumn('malware_family', parse_malware_family(combined_col))

    # Get numeric features
    feature_cols = get_numeric_columns(df_labeled)

    # ========================================
    # 1. BINARY CLASSIFICATION
    # ========================================

    print("\n  Analyzing for: Binary Classification...")

    binary_counts = df_labeled.groupBy('parsed_label').count().collect()
    binary_dict = {row['parsed_label']: int(row['count']) for row in binary_counts}

    benign_count = binary_dict.get('Benign', 0)
    malicious_count = binary_dict.get('Malicious', 0)

    if benign_count > 0 and malicious_count > 0:
        imbalance_ratio = max(benign_count, malicious_count) / min(benign_count, malicious_count)
    else:
        imbalance_ratio = float('inf')

    comprehensive_stats['binary_classification'] = {
        'benign_samples': benign_count,
        'malicious_samples': malicious_count,
        'imbalance_ratio': round(imbalance_ratio, 2),
        'gan_augmentation_needed': imbalance_ratio > 10,
        'feasibility': 'NOT_APPLICABLE' if imbalance_ratio <= 10 else 'RECOMMENDED',
        'reason': (
            f'Imbalance ratio {imbalance_ratio:.1f}:1. '
            f'GAN recommended for minority class.' if imbalance_ratio > 10 else
            'Classes reasonably balanced. Standard class weighting is sufficient.'
        ),
        'numeric_feature_count': len(feature_cols)
    }

    print(f"    ✅ Feasibility: {'RECOMMENDED' if imbalance_ratio > 10 else 'NOT_APPLICABLE'}")

    # ========================================
    # 2. MULTICLASS ATTACK TYPE
    # ========================================

    print("\n  Analyzing for: Multiclass Attack Type...")

    malicious_df = df_labeled.filter(F.col('parsed_label') == 'Malicious')
    attack_counts = malicious_df.groupBy('attack_type').count().collect()
    attack_dict = {
        row['attack_type']: int(row['count'])
        for row in attack_counts
        if row['attack_type'] and row['attack_type'] != 'Benign'
    }

    if not attack_dict:
        comprehensive_stats['multiclass_attack_type'] = {
            'feasibility': 'NO-GO',
            'reason': 'No malicious attack types found.'
        }
    else:
        critical_rare = {k: v for k, v in attack_dict.items() if v < 10}
        very_rare = {k: v for k, v in attack_dict.items() if 10 <= v < 100}
        rare = {k: v for k, v in attack_dict.items() if 100 <= v < 1000}
        sufficient = {k: v for k, v in attack_dict.items() if v >= 1000}

        needs_gan = len(critical_rare) + len(very_rare) + len(rare) > 0

        feature_cols_mc = get_numeric_columns(malicious_df)

        comprehensive_stats['multiclass_attack_type'] = {
            'gan_augmentation_needed': needs_gan,
            'feasibility': 'RECOMMENDED' if needs_gan else 'GO',
            'reason': (
                f"{len(critical_rare) + len(very_rare) + len(rare)} classes are rare "
                f"and would benefit from augmentation." if needs_gan else
                "All attack types have sufficient samples."
            ),
            'critical_rare_classes': critical_rare,
            'very_rare_classes': very_rare,
            'rare_classes': rare,
            'sufficient_classes': sufficient,
            'numeric_feature_count': len(feature_cols_mc),
            'numeric_features': feature_cols_mc[:20]
        }

        print(f"    ✅ Feasibility: {'RECOMMENDED' if needs_gan else 'GO'}")

    # ========================================
    # 3. MALWARE FAMILY
    # ========================================

    print("\n  Analyzing for: Malware Family Classification...")

    family_counts = malicious_df.groupBy('malware_family').count().collect()
    family_dict = {
        row['malware_family']: int(row['count'])
        for row in family_counts
        if row['malware_family'] and row['malware_family'] not in ['Benign', 'Unknown']
    }

    if not family_dict:
        comprehensive_stats['malware_family'] = {
            'feasibility': 'NO-GO',
            'reason': 'No malware families found.'
        }
    else:
        critical_rare_fam = {k: v for k, v in family_dict.items() if v < 10}
        very_rare_fam = {k: v for k, v in family_dict.items() if 10 <= v < 100}
        rare_fam = {k: v for k, v in family_dict.items() if 100 <= v < 1000}
        sufficient_fam = {k: v for k, v in family_dict.items() if v >= 1000}

        needs_gan_fam = len(critical_rare_fam) + len(very_rare_fam) + len(rare_fam) > 0

        comprehensive_stats['malware_family'] = {
            'gan_augmentation_needed': needs_gan_fam,
            'feasibility': 'RECOMMENDED' if needs_gan_fam else 'GO',
            'reason': (
                f"{len(critical_rare_fam) + len(very_rare_fam) + len(rare_fam)} families "
                f"are rare and would benefit from augmentation." if needs_gan_fam else
                "All malware families have sufficient samples."
            ),
            'critical_rare_families': critical_rare_fam,
            'very_rare_families': very_rare_fam,
            'rare_families': rare_fam,
            'sufficient_families': sufficient_fam,
            'numeric_feature_count': len(feature_cols)
        }

        print(f"    ✅ Feasibility: {'RECOMMENDED' if needs_gan_fam else 'GO'}")

    # ========================================
    # 4. OVERALL RECOMMENDATION
    # ========================================

    binary_needs = comprehensive_stats['binary_classification'].get('gan_augmentation_needed', False)
    multiclass_needs = comprehensive_stats['multiclass_attack_type'].get('gan_augmentation_needed', False)
    family_needs = comprehensive_stats['malware_family'].get('gan_augmentation_needed', False)

    if binary_needs or multiclass_needs or family_needs:
        comprehensive_stats['overall_recommendation'] = (
            'GAN augmentation RECOMMENDED for at least one classification task. '
            'Prioritize augmenting rare classes in multiclass and malware family tasks.'
        )
    else:
        comprehensive_stats['overall_recommendation'] = (
            'GAN augmentation NOT REQUIRED. All classes have sufficient samples. '
            'Consider standard techniques like SMOTE or class weighting if needed.'
        )

    # Summary
    print("\n" + "=" * 70)
    print("GAN FEASIBILITY SUMMARY")
    print("=" * 70)
    print(f"Binary classification:     {'✅ RECOMMENDED' if binary_needs else '❌ NOT NEEDED'}")
    print(f"Multiclass attack type:    {'✅ RECOMMENDED' if multiclass_needs else '❌ NOT NEEDED'}")
    print(f"Malware family:            {'✅ RECOMMENDED' if family_needs else '❌ NOT NEEDED'}")
    print(f"\n💡 Overall: {comprehensive_stats['overall_recommendation']}")
    print("=" * 70)

    return comprehensive_stats


if __name__ == "__main__":
    """Standalone testing"""
    import os
    from config import Config
    from analysis.stage1.utils import load_raw_data, save_json_report

    Config.ensure_output_dirs()
    spark = Config.get_spark_session("Stage1-Unsupervised-Test")

    try:
        df = load_raw_data(spark)

        # Autoencoder analysis
        autoencoder_stats = analyze_autoencoder_feasibility(df)
        ae_path = os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'autoencoder_feasibility.json')
        save_json_report(autoencoder_stats, ae_path)

        # Clustering analysis
        clustering_stats = analyze_clustering_feasibility(df)
        cluster_path = os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'clustering_feasibility.json')
        save_json_report(clustering_stats, cluster_path)

        # GAN analysis
        gan_stats = analyze_gan_feasibility_for_all_tasks(df)
        gan_path = os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'gan_feasibility.json')
        save_json_report(gan_stats, gan_path)

    finally:
        if not Config.is_databricks():
            spark.stop()
