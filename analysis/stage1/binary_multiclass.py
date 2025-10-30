"""
Binary and Multi-Class Classification Feasibility Analysis (PySpark).
"""

from pyspark.sql import functions as F

# Get numeric features
from .unsupervised_models import get_numeric_columns
from .utils import get_combined_label_column
from .utils import parse_binary_label, parse_attack_type


def analyze_binary_classification(df):
    """
    Analyze feasibility for binary classification (Benign vs Malicious).
    
    Args:
        df: Spark DataFrame (raw data)
        
    Returns:
        dict: Binary classification feasibility stats
    """
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION FEASIBILITY")
    print("=" * 70)

    # Parse labels from detailed-label column (NOT filename)
    print("  Parsing labels from 'detailed-label' column...")
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_labeled = df.withColumn('parsed_label', parse_binary_label(combined_col))

    # Get class counts
    class_counts = df_labeled.groupBy('parsed_label').count().collect()
    class_dict = {row['parsed_label']: int(row['count']) for row in class_counts}

    benign_count = class_dict.get('Benign', 0)
    malicious_count = class_dict.get('Malicious', 0)
    total = benign_count + malicious_count
    feature_cols = get_numeric_columns(df_labeled)

    stats = {
        'benign_samples': benign_count,
        'malicious_samples': malicious_count,
        'total_samples': total,
        'benign_percentage': round((benign_count / total) * 100, 2) if total > 0 else 0,
        'malicious_percentage': round((malicious_count / total) * 100, 2) if total > 0 else 0,
        'imbalance_ratio': round(max(benign_count, malicious_count) / min(benign_count, malicious_count), 2) if min(
            benign_count, malicious_count) > 0 else float('inf'),
        'missing_per_class': {},
        'numeric_feature_count': len(feature_cols),  # ✅ ADD THIS
        'numeric_features': feature_cols[:20],  # ✅ ADD THIS (first 20)
        'feasibility': 'GO',
        'reason': 'Sufficient samples and acceptable balance'
    }

    # Missing values per class
    print("  Calculating missing values per class...")
    for label in ['Benign', 'Malicious']:
        label_df = df_labeled.filter(F.col('parsed_label') == label)
        label_count = label_df.count()

        if label_count > 0:
            # Count nulls across all columns
            null_count = sum([label_df.filter(F.col(col).isNull()).count() for col in df.columns])
            total_cells = label_count * len(df.columns)
            missing_pct = round((null_count / total_cells) * 100, 2) if total_cells > 0 else 0
            stats['missing_per_class'][label] = missing_pct

    # Feasibility checks
    if benign_count < 1000 or malicious_count < 1000:
        stats['feasibility'] = 'NO-GO'
        stats[
            'reason'] = f'Insufficient samples per class (need >1000 each). Benign: {benign_count}, Malicious: {malicious_count}'
    elif stats['imbalance_ratio'] > 99:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Severe class imbalance ({stats["imbalance_ratio"]}:1 ratio)'
    elif stats['missing_per_class'] and max(stats['missing_per_class'].values()) > 50:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = 'High missing percentage (>50%) in one or more classes'

    # Summary
    print(f"\n✓ Benign: {benign_count:,} ({stats['benign_percentage']}%)")
    print(f"✓ Malicious: {malicious_count:,} ({stats['malicious_percentage']}%)")
    print(f"✓ Imbalance ratio: {stats['imbalance_ratio']}:1")
    print(f"✓ Feasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"  Reason: {stats['reason']}")

    return stats


def analyze_multiclass_classification(df):
    """
    Analyze feasibility for multi-class attack type classification.
    
    Args:
        df: Spark DataFrame (raw data)
        
    Returns:
        dict: Multi-class classification feasibility stats
    """
    print("\n" + "=" * 70)
    print("MULTI-CLASS ATTACK TYPE FEASIBILITY")
    print("=" * 70)

    # Parse labels
    print("  Parsing attack types from 'detailed-label' column...")
    combined_col = get_combined_label_column('label', 'detailed-label')
    df_labeled = df.withColumn('parsed_label', parse_binary_label(combined_col))
    df_labeled = df_labeled.withColumn('attack_type', parse_attack_type(combined_col))

    # Filter only malicious (multi-class only on malicious traffic)
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
    print("  Calculating missing values per attack type...")
    missing_per_class = {}
    for attack_type in attack_dict.keys():
        attack_df = malicious_df.filter(F.col('attack_type') == attack_type)
        attack_count = attack_df.count()

        if attack_count > 0:
            null_count = sum([attack_df.filter(F.col(col).isNull()).count() for col in df.columns])
            total_cells = attack_count * len(df.columns)
            missing_pct = round((null_count / total_cells) * 100, 2) if total_cells > 0 else 0
            missing_per_class[attack_type] = missing_pct

    feature_cols = get_numeric_columns(malicious_df)
    stats = {
        'total_malicious_samples': malicious_count,
        'unique_attack_types': len(attack_dict),
        'attack_type_distribution': attack_dict,
        'rare_classes': rare_classes,
        'rare_class_count': len(rare_classes),
        'min_samples_per_class': min(attack_dict.values()) if attack_dict else 0,
        'max_samples_per_class': max(attack_dict.values()) if attack_dict else 0,
        'avg_samples_per_class': round(sum(attack_dict.values()) / len(attack_dict), 2) if attack_dict else 0,
        'missing_per_class': missing_per_class,  # ✅ ADD THIS
        'numeric_feature_count': len(feature_cols),  # ✅ ADD THIS
        'numeric_features': feature_cols[:20],  # ✅ ADD THIS
        'feasibility': 'GO',
        'reason': 'Sufficient attack types and samples'
    }

    # Feasibility checks
    if stats['unique_attack_types'] < 3:
        stats['feasibility'] = 'NO-GO'
        stats['reason'] = f'Too few attack types ({stats["unique_attack_types"]}). Need at least 3.'
    elif stats['min_samples_per_class'] < 50:
        stats['feasibility'] = 'WARNING'
        stats[
            'reason'] = f'Some classes have very few samples (min: {stats["min_samples_per_class"]}). Consider GAN augmentation.'
    elif len(rare_classes) > len(attack_dict) * 0.5:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Over 50% of classes are rare (<{rare_threshold} samples). GAN augmentation recommended.'

    # Summary
    print(f"\n✓ Total malicious: {stats['total_malicious_samples']:,}")
    print(f"✓ Unique attack types: {stats['unique_attack_types']}")
    print(f"✓ Min samples per class: {stats['min_samples_per_class']}")
    print(f"✓ Rare classes (<{rare_threshold}): {stats['rare_class_count']}")
    print(f"✓ Feasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"  Reason: {stats['reason']}")

    # Show top 10 attack types
    if attack_dict:
        print("\n  Top attack types:")
        sorted_attacks = sorted(attack_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        for attack, count in sorted_attacks:
            print(f"    {attack}: {count:,}")

    return stats
