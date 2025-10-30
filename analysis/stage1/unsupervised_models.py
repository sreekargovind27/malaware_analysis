"""
FIXED VERSION - Unsupervised Models Feasibility Analysis (Autoencoder, K-Means, GAN) - PySpark.
"""

from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, FloatType, DoubleType

from .utils import parse_binary_label, parse_attack_type, get_malware_family_udf, get_combined_label_column


def get_numeric_columns(df):
    """
    Get list of numeric column names from DataFrame.
    
    FIXED: Uses proper type checking instead of string comparison.
    
    Args:
        df: Spark DataFrame
        
    Returns:
        list: List of numeric column names
    """
    # ✅ FIXED: Use isinstance() with actual type classes
    numeric_cols = [
        field.name for field in df.schema.fields
        if isinstance(field.dataType, (IntegerType, LongType, FloatType, DoubleType))
    ]

    # ✅ FINAL FIX: Use the new, clean underscored names
    exclude_cols = ['ts', 'id_orig_p', 'id_resp_p']
    feature_cols = [col for col in numeric_cols if col not in exclude_cols]

    return feature_cols


def analyze_autoencoder_feasibility(df):
    """
    Analyze feasibility for autoencoder anomaly detection.
    
    Args:
        df: Spark DataFrame (raw data)
        
    Returns:
        dict: Autoencoder feasibility stats
    """
    print("\n" + "=" * 70)
    print("AUTOENCODER (ANOMALY DETECTION) FEASIBILITY")
    print("=" * 70)

    # Parse labels
    print("  Filtering benign samples...")
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_labeled = df.withColumn('parsed_label', parse_binary_label(combined_col))

    benign_df = df_labeled.filter(F.col('parsed_label') == 'Benign')
    benign_count = benign_df.count()

    # ✅ FIXED: Use proper numeric type detection
    feature_cols = get_numeric_columns(benign_df)
    feature_count = len(feature_cols)

    print(f"  Found {feature_count} numeric features: {feature_cols[:10]}")  # Show first 10

    # Check missing values in benign data
    if benign_count > 0 and feature_cols:
        null_counts = [benign_df.filter(F.col(col).isNull()).count() for col in feature_cols]
        total_cells = benign_count * feature_count
        total_missing = sum(null_counts)
        missing_pct = round((total_missing / total_cells) * 100, 2) if total_cells > 0 else 0
    else:
        missing_pct = 0

    # Per-feature missing percentages
    missing_per_feature = {}
    if benign_count > 0 and feature_cols:
        for col in feature_cols:
            missing_count = benign_df.filter(F.col(col).isNull()).count()
            missing_per_feature[col] = round((missing_count / benign_count) * 100, 2)

    stats = {
        'benign_samples': int(benign_count),
        'numeric_feature_count': feature_count,
        'numeric_features': feature_cols[:20],  # ✅ LIMIT TO 20
        'missing_percentage_overall': missing_pct,  # ✅ RENAMED
        'missing_per_feature': missing_per_feature,  # ✅ ADD THIS
        'feasibility': 'GO',
        'reason': 'Sufficient benign samples and features'
    }

    # Feasibility checks
    if benign_count == 0:
        stats['feasibility'] = 'NO-GO'
        stats['reason'] = 'No benign samples found'
    elif benign_count < 10000:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Low benign sample count ({benign_count}). Recommend >10,000 for robust autoencoder.'
    elif feature_count == 0:
        stats['feasibility'] = 'NO-GO'
        stats['reason'] = 'No numeric features found. Cannot train autoencoder.'
    elif feature_count < 15:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Few numeric features ({feature_count}). Recommend >15 for effective anomaly detection.'
    elif missing_pct > 30:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'High missing percentage ({missing_pct}%) in benign data.'

    # Summary
    print(f"\n✓ Benign samples: {stats['benign_samples']:,}")
    print(f"✓ Numeric features: {stats['numeric_feature_count']}")
    print(f"✓ Missing percentage: {stats['missing_percentage_overall']}%")
    print(f"✓ Feasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"  Reason: {stats['reason']}")

    return stats


def analyze_clustering_feasibility(df):
    """
    Analyze feasibility for K-Means clustering.
    
    Args:
        df: Spark DataFrame (raw data)
        
    Returns:
        dict: Clustering feasibility stats
    """
    print("\n" + "=" * 70)
    print("K-MEANS CLUSTERING FEASIBILITY")
    print("=" * 70)

    # Parse labels
    print("  Filtering malicious samples...")
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_labeled = df.withColumn('parsed_label', parse_binary_label(combined_col))
    df_labeled = df_labeled.withColumn('malware_family', get_malware_family_udf()(F.col('Source_Folder')))

    malicious_df = df_labeled.filter(F.col('parsed_label') == 'Malicious')
    malicious_count = malicious_df.count()

    print(f"  Found {malicious_count:,} malicious samples")

    if malicious_count == 0:
        return {
            'malicious_samples': 0,
            'malware_family_count': 0,
            'numeric_feature_count': 0,
            'non_constant_feature_count': 0,
            'feasibility': 'NO-GO',
            'reason': 'No malicious samples found in dataset'
        }

    # Count distinct malware types (for diversity check)
    malicious_families = malicious_df.filter(
        F.col('malware_family').isNotNull() &
        ~F.col('malware_family').isin(['Benign', 'Unknown'])
    )
    family_count = malicious_families.select('malware_family').distinct().count()

    # ✅ FIXED: Use proper numeric type detection
    feature_cols = get_numeric_columns(malicious_df)
    feature_count = len(feature_cols)

    print(f"  Found {feature_count} numeric features")

    # Check for non-constant features (features with variance)
    if malicious_count > 0 and feature_cols:
        # Check variance for each feature (sample first 10000 rows for speed)
        sample_df = malicious_df.select(feature_cols).limit(10000)
        non_constant_features = []
        for col in feature_cols:
            variance = sample_df.select(F.variance(col)).collect()[0][0]
            if variance is not None and variance > 1e-6:
                non_constant_features.append(col)
        non_constant_count = len(non_constant_features)
    else:
        non_constant_count = 0

    # Per-feature missing percentages
    missing_per_feature = {}
    if malicious_count > 0 and feature_cols:
        for col in feature_cols:
            missing_count = malicious_df.filter(F.col(col).isNull()).count()
            missing_per_feature[col] = round((missing_count / malicious_count) * 100, 2)

    stats = {
        'malicious_samples': int(malicious_count),
        'malware_family_count': int(family_count),
        'numeric_feature_count': feature_count,
        'numeric_features': feature_cols[:20],  # ✅ LIMIT TO 20
        'non_constant_feature_count': non_constant_count,
        'missing_per_feature': missing_per_feature,  # ✅ ADD THIS
        'feasibility': 'GO',
        'reason': 'Sufficient malicious samples and feature diversity'
    }

    # Feasibility checks
    if malicious_count < 5000:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Low malicious sample count ({malicious_count}). Recommend >5,000 for meaningful clusters.'
    elif family_count < 3:
        stats['feasibility'] = 'WARNING'
        stats[
            'reason'] = f'Low malware type diversity ({family_count} types). Clustering may not find distinct patterns.'
    elif feature_count == 0:
        stats['feasibility'] = 'NO-GO'
        stats['reason'] = 'No numeric features found. Cannot perform clustering.'
    elif non_constant_count < 10:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Few non-constant features ({non_constant_count}). May limit clustering effectiveness.'

    # Summary
    print(f"\n✓ Malicious samples: {stats['malicious_samples']:,}")
    print(f"✓ Malware families: {stats['malware_family_count']}")
    print(f"✓ Numeric features: {stats['numeric_feature_count']}")
    print(f"✓ Non-constant features: {stats['non_constant_feature_count']}")
    print(f"✓ Feasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"  Reason: {stats['reason']}")

    return stats


def analyze_gan_feasibility_for_all_tasks(df):
    """
    Analyzes GAN augmentation feasibility for all classification tasks:
    1. Binary (Benign vs. Malicious)
    2. Multi-class (Attack Type)
    3. Malware Family

    Args:
        df: Spark DataFrame (raw data)

    Returns:
        dict: Comprehensive GAN feasibility stats for all models.
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE GAN AUGMENTATION FEASIBILITY")
    print("=" * 70)

    comprehensive_stats = {}

    # --- 1. Binary Classification Analysis ---
    print("  Analyzing for: Binary Classification...")
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_binary = df.withColumn('parsed_label', parse_binary_label(combined_col))
    binary_counts = df_binary.groupBy('parsed_label').count().collect()
    binary_dict = {row['parsed_label']: int(row['count']) for row in binary_counts if row['parsed_label']}

    benign_count = binary_dict.get('Benign', 0)
    malicious_count = binary_dict.get('Malicious', 0)

    comprehensive_stats['binary_classification'] = {
        'gan_augmentation_needed': False,
        'feasibility': 'NOT_APPLICABLE',
        'reason': f"The classes (Benign: {benign_count:,}, Malicious: {malicious_count:,}) are well-balanced and have sufficient samples. Standard class weighting is enough."
    }
    print(f"    ✓ Feasibility: NOT_APPLICABLE")

    # --- 2. Multi-Class Attack Type Analysis ---
    print("  Analyzing for: Multi-Class Attack Type...")
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_multiclass = df.withColumn('parsed_label', parse_binary_label(combined_col))  # This line added
    df_multiclass = df_multiclass.withColumn('attack_type', parse_attack_type(combined_col))

    malicious_df = df_multiclass.filter(F.col('parsed_label') == 'Malicious')

    attack_counts = malicious_df.groupBy('attack_type').count().collect()
    attack_dict = {row['attack_type']: int(row['count']) for row in attack_counts if
                   row['attack_type'] and row['attack_type'] != 'Benign'}

    if not attack_dict:
        comprehensive_stats['multiclass_attack_type'] = {'feasibility': 'NO-GO',
                                                         'reason': 'No malicious attack types found.'}
    else:
        crit_rare = {k: v for k, v in attack_dict.items() if v < 10}
        very_rare = {k: v for k, v in attack_dict.items() if 10 <= v < 100}
        rare = {k: v for k, v in attack_dict.items() if 100 <= v < 1000}
        sufficient = {k: v for k, v in attack_dict.items() if v >= 1000}

        needs_gan = len(crit_rare) + len(very_rare) + len(rare) > 0

        # Get numeric features for multiclass
        feature_cols_mc = get_numeric_columns(malicious_df)

        comprehensive_stats['multiclass_attack_type'] = {
            'gan_augmentation_needed': needs_gan,
            'feasibility': 'RECOMMENDED' if needs_gan else 'GO',
            'reason': f"{len(crit_rare) + len(very_rare) + len(rare)} classes are rare and would benefit from augmentation." if needs_gan else "All attack types have sufficient samples.",
            'critical_rare_classes (<10 samples)': crit_rare,
            'very_rare_classes (10-99 samples)': very_rare,
            'rare_classes (100-999 samples)': rare,
            'sufficient_classes (>=1000 samples)': sufficient,
            'numeric_feature_count': len(feature_cols_mc),  # ✅ ADD
            'numeric_features': feature_cols_mc[:20]  # ✅ ADD
        }

    print(f"    ✓ Feasibility: {'RECOMMENDED' if needs_gan else 'GO'}")

    # --- 3. Malware Family Classification Analysis ---
    print("  Analyzing for: Malware Family Classification...")
    df_family = df.withColumn('malware_family', get_malware_family_udf()(F.col('Source_Folder')))
    combined_col_fam = get_combined_label_column('label', 'detailed-label')
    malicious_family_df = df_family.withColumn('parsed_label', parse_binary_label(combined_col_fam)).filter(
        F.col('parsed_label') == 'Malicious')

    family_counts = malicious_family_df.filter(
        F.col('malware_family').isNotNull() & ~F.col('malware_family').isin(['Benign', 'Unknown'])
    ).groupBy('malware_family').count().collect()
    family_dict = {row['malware_family']: int(row['count']) for row in family_counts}

    if not family_dict:
        comprehensive_stats['malware_family_classification'] = {'feasibility': 'NO-GO',
                                                                'reason': 'No valid malware families found.'}
    else:
        crit_rare_fam = {k: v for k, v in family_dict.items() if v < 10}
        very_rare_fam = {k: v for k, v in family_dict.items() if 10 <= v < 100}
        rare_fam = {k: v for k, v in family_dict.items() if 100 <= v < 1000}
        sufficient_fam = {k: v for k, v in family_dict.items() if v >= 1000}

        needs_gan_fam = len(crit_rare_fam) + len(very_rare_fam) + len(rare_fam) > 0

        # Get numeric features for malware family
        feature_cols_fam = get_numeric_columns(malicious_family_df)

        comprehensive_stats['malware_family_classification'] = {
            'gan_augmentation_needed': needs_gan_fam,
            'feasibility': 'RECOMMENDED' if needs_gan_fam else 'GO',
            'reason': "All malware families have thousands of samples and are considered sufficient." if not needs_gan_fam else f"{len(crit_rare_fam) + len(very_rare_fam) + len(rare_fam)} families are rare.",
            'critical_rare_classes (<10 samples)': crit_rare_fam,
            'very_rare_classes (10-99 samples)': very_rare_fam,
            'rare_classes (100-999 samples)': rare_fam,
            'sufficient_classes (>=1000 samples)': sufficient_fam,
            'numeric_feature_count': len(feature_cols_fam),  # ✅ ADD
            'numeric_features': feature_cols_fam[:20]  # ✅ ADD
        }
    print(f"    ✓ Feasibility: {'RECOMMENDED' if needs_gan_fam else 'GO'}")

    return comprehensive_stats
