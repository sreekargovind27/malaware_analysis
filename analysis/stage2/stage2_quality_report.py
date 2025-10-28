"""
Stage 2 Comprehensive Quality Report

Validates Stage 2 outputs before proceeding to Stage 3 training.
Saves validation report to outputs/stage2_quality/
"""
import os
import json
import pandas as pd
from config import Config


def validate_stage2_outputs():
    """Comprehensive validation of Stage 2 outputs."""
    print("\n" + "=" * 70)
    print("STAGE 2 QUALITY VALIDATION REPORT")
    print("=" * 70)

    report = {
        'validation_date': pd.Timestamp.now().isoformat(),
        'files_checked': {},
        'overall_status': 'PASS',
        'issues': [],
        'warnings': []
    }

    # 1. Check flow features
    print("\n[1/6] Validating flow features...")
    if os.path.exists(Config.ENGINEERED_DATA_PATH):
        df = pd.read_parquet(Config.ENGINEERED_DATA_PATH)
        features = Config.get_feature_list()

        report['files_checked']['flow_features'] = {
            'exists': True,
            'row_count': len(df),
            'feature_count': len(features),
            'missing_values': int(df[features].isnull().sum().sum()),
            'duplicate_rows': int(df.duplicated().sum())
        }

        # Check for issues
        if len(df) < 1000:
            report['issues'].append("Flow features: Too few rows (<1000)")
            report['overall_status'] = 'FAIL'
        if len(features) < 10:
            report['issues'].append("Flow features: Too few features (<10)")
            report['overall_status'] = 'FAIL'
        if report['files_checked']['flow_features']['missing_values'] > 0:
            report['warnings'].append(
                f"Flow features: {report['files_checked']['flow_features']['missing_values']} missing values found")

        print(f"   ✓ Rows: {len(df):,}")
        print(f"   ✓ Features: {len(features)}")
    else:
        report['files_checked']['flow_features'] = {'exists': False}
        report['issues'].append("Flow features file not found")
        report['overall_status'] = 'FAIL'
        print("   ❌ File not found")

    # 2. Check device features
    print("\n[2/6] Validating device features...")
    if os.path.exists(Config.DEVICE_FEATURES_PATH):
        device_df = pd.read_parquet(Config.DEVICE_FEATURES_PATH)

        report['files_checked']['device_features'] = {
            'exists': True,
            'device_count': len(device_df),
            'benign_devices': int((device_df['device_label'] == 'Benign').sum()),
            'malicious_devices': int((device_df['device_label'] == 'Malicious').sum())
        }

        if len(device_df) < 100:
            report['warnings'].append("Device features: Very few devices (<100)")

        print(f"   ✓ Devices: {len(device_df):,}")
        print(f"   ✓ Benign: {report['files_checked']['device_features']['benign_devices']:,}")
        print(f"   ✓ Malicious: {report['files_checked']['device_features']['malicious_devices']:,}")
    else:
        report['files_checked']['device_features'] = {'exists': False}
        report['warnings'].append("Device features not found (GNN unavailable)")
        print("   ⚠️  File not found (GNN will be unavailable)")

    # 3. Check train/test splits
    print("\n[3/6] Validating train/test splits...")
    train_exists = os.path.exists(Config.TRAIN_SET_PATH)
    test_exists = os.path.exists(Config.TEST_SET_PATH)

    if train_exists and test_exists:
        train_df = pd.read_parquet(Config.TRAIN_SET_PATH)
        test_df = pd.read_parquet(Config.TEST_SET_PATH)

        report['files_checked']['splits'] = {
            'train_exists': True,
            'test_exists': True,
            'train_count': len(train_df),
            'test_count': len(test_df),
            'split_ratio': round(len(test_df) / (len(train_df) + len(test_df)), 2)
        }

        # Validate split ratio
        if report['files_checked']['splits']['split_ratio'] < 0.15 or report['files_checked']['splits'][
            'split_ratio'] > 0.25:
            report['warnings'].append(f"Unusual split ratio: {report['files_checked']['splits']['split_ratio']}")

        print(f"   ✓ Train: {len(train_df):,} rows")
        print(f"   ✓ Test: {len(test_df):,} rows")
        print(f"   ✓ Split ratio: {report['files_checked']['splits']['split_ratio']}")
    else:
        report['files_checked']['splits'] = {
            'train_exists': train_exists,
            'test_exists': test_exists
        }
        report['issues'].append("Train/test splits missing")
        report['overall_status'] = 'FAIL'
        print("   ❌ Split files not found")

    # 4. Check graph
    print("\n[4/6] Validating heterogeneous graph...")
    if os.path.exists(Config.HETERO_GRAPH_PATH):
        graph_stats_path = Config.GRAPH_STATS_PATH
        if os.path.exists(graph_stats_path):
            with open(graph_stats_path, 'r') as f:
                graph_stats = json.load(f)
            report['files_checked']['graph'] = {
                'exists': True,
                'stats': graph_stats
            }
            print(f"   ✓ Device nodes: {graph_stats['num_device_nodes']:,}")
            print(f"   ✓ Total edges: {graph_stats['total_edges']:,}")
        else:
            report['files_checked']['graph'] = {'exists': True, 'stats': None}
            print("   ✓ Graph exists (no stats file)")
    else:
        report['files_checked']['graph'] = {'exists': False}
        report['warnings'].append("Heterogeneous graph not found (GNN unavailable)")
        print("   ⚠️  Graph not found (GNN will be unavailable)")

    # 5. Check feature list
    print("\n[5/6] Validating feature list...")
    if os.path.exists(Config.FEATURE_LIST_PATH):
        features = Config.get_feature_list()
        report['files_checked']['feature_list'] = {
            'exists': True,
            'count': len(features)
        }
        print(f"   ✓ Feature list: {len(features)} features")
    else:
        report['files_checked']['feature_list'] = {'exists': False}
        report['issues'].append("Feature list file not found")
        report['overall_status'] = 'FAIL'
        print("   ❌ Feature list not found")

    # 6. Check label distributions
    print("\n[6/6] Validating label distributions...")
    if train_exists:
        train_df = pd.read_parquet(Config.TRAIN_SET_PATH)

        binary_dist = train_df[Config.TARGET_COL].value_counts().to_dict()
        multiclass_dist = train_df[Config.DETAILED_TARGET_COL].value_counts().to_dict()
        family_dist = train_df[Config.FAMILY_TARGET_COL].value_counts().to_dict()

        report['files_checked']['label_distributions'] = {
            'binary': binary_dist,
            'multiclass': dict(list(multiclass_dist.items())[:10]),  # Top 10
            'family': family_dist
        }

        # Check for severe imbalance
        if 'Benign' in binary_dist and 'Malicious' in binary_dist:
            ratio = max(binary_dist.values()) / min(binary_dist.values())
            if ratio > 100:
                report['warnings'].append(f"Severe class imbalance: {ratio:.1f}:1 ratio")

        print(f"   ✓ Binary classes: {len(binary_dist)}")
        print(f"   ✓ Attack types: {len(multiclass_dist)}")
        print(f"   ✓ Malware families: {len(family_dist)}")

    # Save report
    report_dir = Config.STAGE2_QUALITY_DIR
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, 'stage2_validation_report.json')

    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)

    # Print summary
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Overall Status: {report['overall_status']}")

    if report['issues']:
        print(f"\n❌ Issues ({len(report['issues'])}):")
        for issue in report['issues']:
            print(f"   - {issue}")

    if report['warnings']:
        print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
        for warning in report['warnings']:
            print(f"   - {warning}")

    if report['overall_status'] == 'PASS' and not report['issues']:
        print("\n✅ All checks passed! Data ready for Stage 3 training.")

    print(f"\n📄 Report saved to: {report_path}")
    print("=" * 70)

    return report


if __name__ == "__main__":
    validate_stage2_outputs()