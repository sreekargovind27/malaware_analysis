"""
Stage 2 Comprehensive Quality Report - PySpark Version
Works on both local and Databricks.

Validates Stage 2 outputs before proceeding to Stage 3 training:
- Engineered flow features
- Device features
- Train/val/test splits
- Heterogeneous graph
- Label distributions

Generates JSON report with issues/warnings.
"""

import json
import os
from datetime import datetime

from pyspark.sql import functions as F

from config import Config


# ============================================================================
# VALIDATION FUNCTIONS
# ============================================================================

def validate_stage2_outputs():
    """
    Comprehensive validation of Stage 2 outputs using PySpark.
    Generates a quality report with issues and warnings.

    Returns:
        dict: Validation report with status, issues, and warnings
    """
    print("\n" + "=" * 70)
    print("STAGE 2 QUALITY VALIDATION REPORT")
    print("=" * 70)

    # Get or create Spark session
    spark = Config.get_spark_session("Stage2-Validation")

    report = {
        "validation_date": datetime.now().isoformat(),
        "environment": Config.get_environment(),
        "files_checked": {},
        "overall_status": "PASS",
        "issues": [],
        "warnings": [],
    }

    # ========================================
    # [1/6] FLOW FEATURES
    # ========================================

    print("\n[1/6] Validating flow features...")

    if os.path.exists(Config.ENGINEERED_DATA_PATH):
        try:
            df = spark.read.parquet(Config.ENGINEERED_DATA_PATH)
            features = Config.get_feature_list()

            row_count = df.count()
            col_count = len(df.columns)

            report["files_checked"]["engineered_flows"] = {
                "path": Config.ENGINEERED_DATA_PATH,
                "exists": True,
                "rows": row_count,
                "columns": col_count,
                "features": len(features)
            }

            # Check for empty dataset
            if row_count == 0:
                report["issues"].append("Engineered flows dataset is empty")
                report["overall_status"] = "FAIL"

            # Check for missing features
            if len(features) == 0:
                report["warnings"].append("No features defined in feature list")

            # Check for nulls in feature columns
            null_counts = {}
            for feat in features[:20]:  # Check first 20 features (sampling)
                if feat in df.columns:
                    null_count = df.filter(F.col(feat).isNull()).count()
                    if null_count > 0:
                        null_counts[feat] = null_count

            if null_counts:
                total_nulls = sum(null_counts.values())
                report["warnings"].append(
                    f"Found {total_nulls:,} null values across {len(null_counts)} features"
                )

            # Check for duplicates (sample-based for performance)
            if row_count > 0:
                sample_size = min(10000, row_count)
                sample_df = df.limit(sample_size)
                distinct_count = sample_df.distinct().count()
                duplicate_pct = ((sample_size - distinct_count) / sample_size) * 100

                if duplicate_pct > 10:
                    report["warnings"].append(
                        f"High duplicate rate in sample: {duplicate_pct:.1f}%"
                    )

            print(f"   ✅ Flow features: {row_count:,} rows, {len(features)} features")

        except Exception as e:
            report["issues"].append(f"Flow features validation error: {e}")
            report["overall_status"] = "FAIL"
            print(f"   ❌ Error: {e}")
    else:
        report["files_checked"]["engineered_flows"] = {
            "path": Config.ENGINEERED_DATA_PATH,
            "exists": False
        }
        report["issues"].append("Engineered flows file not found")
        report["overall_status"] = "FAIL"
        print(f"   ❌ File not found: {Config.ENGINEERED_DATA_PATH}")

    # ========================================
    # [2/6] DEVICE FEATURES
    # ========================================

    print("\n[2/6] Validating device features...")

    if os.path.exists(Config.DEVICE_FEATURES_PATH):
        try:
            df = spark.read.parquet(Config.DEVICE_FEATURES_PATH)
            row_count = df.count()
            col_count = len(df.columns)

            report["files_checked"]["device_features"] = {
                "path": Config.DEVICE_FEATURES_PATH,
                "exists": True,
                "rows": row_count,
                "columns": col_count
            }

            if row_count == 0:
                report["warnings"].append("Device features dataset is empty")

            print(f"   ✅ Device features: {row_count:,} devices, {col_count} features")

        except Exception as e:
            report["warnings"].append(f"Device features validation error: {e}")
            print(f"   ⚠️  Error: {e}")
    else:
        report["files_checked"]["device_features"] = {
            "path": Config.DEVICE_FEATURES_PATH,
            "exists": False
        }
        report["warnings"].append("Device features file not found (optional)")
        print(f"   ⚠️  File not found: {Config.DEVICE_FEATURES_PATH}")

    # ========================================
    # [3/6] TRAIN/VAL/TEST SPLITS
    # ========================================

    print("\n[3/6] Validating train/val/test splits...")

    split_paths = {
        "train": Config.TRAIN_PATH,
        "val": Config.VAL_PATH,
        "test": Config.TEST_PATH
    }

    split_sizes = {}

    for split_name, split_path in split_paths.items():
        if os.path.exists(split_path):
            try:
                df = spark.read.parquet(split_path)
                row_count = df.count()
                split_sizes[split_name] = row_count

                report["files_checked"][f"{split_name}_split"] = {
                    "path": split_path,
                    "exists": True,
                    "rows": row_count
                }

                if row_count == 0:
                    report["issues"].append(f"{split_name.capitalize()} split is empty")
                    report["overall_status"] = "FAIL"

                print(f"   ✅ {split_name.capitalize()} split: {row_count:,} rows")

            except Exception as e:
                report["issues"].append(f"{split_name.capitalize()} split validation error: {e}")
                report["overall_status"] = "FAIL"
                print(f"   ❌ Error in {split_name}: {e}")
        else:
            report["files_checked"][f"{split_name}_split"] = {
                "path": split_path,
                "exists": False
            }
            report["issues"].append(f"{split_name.capitalize()} split file not found")
            report["overall_status"] = "FAIL"
            print(f"   ❌ File not found: {split_path}")

    # Check split ratios
    if len(split_sizes) == 3:
        total = sum(split_sizes.values())
        if total > 0:
            train_pct = (split_sizes["train"] / total) * 100
            val_pct = (split_sizes["val"] / total) * 100
            test_pct = (split_sizes["test"] / total) * 100

            # Expected: 70/10/20 ± 5%
            if not (65 <= train_pct <= 75):
                report["warnings"].append(
                    f"Train split ratio unusual: {train_pct:.1f}% (expected ~70%)"
                )
            if not (5 <= val_pct <= 15):
                report["warnings"].append(
                    f"Val split ratio unusual: {val_pct:.1f}% (expected ~10%)"
                )
            if not (15 <= test_pct <= 25):
                report["warnings"].append(
                    f"Test split ratio unusual: {test_pct:.1f}% (expected ~20%)"
                )

    # ========================================
    # [4/6] GRAPH
    # ========================================

    print("\n[4/6] Validating graph...")

    if os.path.exists(Config.HETERO_GRAPH_PATH):
        try:
            import torch
            data = torch.load(Config.HETERO_GRAPH_PATH, map_location='cpu')

            report["files_checked"]["graph"] = {
                "path": Config.HETERO_GRAPH_PATH,
                "exists": True,
                "node_types": len(data.node_types) if hasattr(data, 'node_types') else 0,
                "edge_types": len(data.edge_types) if hasattr(data, 'edge_types') else 0
            }

            print(f"   ✅ Graph loaded successfully")
            if hasattr(data, 'node_types'):
                print(f"      Node types: {len(data.node_types)}")
            if hasattr(data, 'edge_types'):
                print(f"      Edge types: {len(data.edge_types)}")

        except Exception as e:
            report["warnings"].append(f"Graph validation error: {e}")
            print(f"   ⚠️  Error: {e}")
    else:
        report["files_checked"]["graph"] = {
            "path": Config.HETERO_GRAPH_PATH,
            "exists": False
        }
        report["warnings"].append("Graph file not found (optional for GNN models)")
        print(f"   ⚠️  File not found: {Config.HETERO_GRAPH_PATH}")

    # ========================================
    # [5/6] FEATURE LIST
    # ========================================

    print("\n[5/6] Validating feature list...")

    if os.path.exists(Config.FEATURE_LIST_PATH):
        try:
            features = Config.get_feature_list()

            report["files_checked"]["feature_list"] = {
                "path": Config.FEATURE_LIST_PATH,
                "exists": True,
                "num_features": len(features)
            }

            if len(features) == 0:
                report["warnings"].append("Feature list is empty")
            elif len(features) < 10:
                report["warnings"].append(f"Very few features: {len(features)} (expected 50+)")

            print(f"   ✅ Feature list: {len(features)} features")

        except Exception as e:
            report["warnings"].append(f"Feature list validation error: {e}")
            print(f"   ⚠️  Error: {e}")
    else:
        report["files_checked"]["feature_list"] = {
            "path": Config.FEATURE_LIST_PATH,
            "exists": False
        }
        report["warnings"].append("Feature list file not found")
        print(f"   ⚠️  File not found: {Config.FEATURE_LIST_PATH}")

    # ========================================
    # [6/6] LABEL DISTRIBUTIONS
    # ========================================

    print("\n[6/6] Validating label distributions...")

    if os.path.exists(Config.ENGINEERED_DATA_PATH):
        try:
            df = spark.read.parquet(Config.ENGINEERED_DATA_PATH)

            # Binary label distribution
            if "label" in df.columns:
                binary_dist = df.groupBy("label").count().collect()
                binary_dist = {row["label"]: row["count"] for row in binary_dist}

                report["label_distributions"] = {
                    "binary": binary_dist
                }

                # Check class imbalance
                if len(binary_dist) >= 2:
                    max_c = max(binary_dist.values())
                    min_c = min(binary_dist.values())
                    if min_c > 0:
                        ratio = max_c / min_c
                        if ratio > 100:
                            report["warnings"].append(
                                f"Severe class imbalance: {ratio:.1f}:1 ratio"
                            )

            # Attack type distribution
            if "attack_type" in df.columns:
                attack_type_dist = df.groupBy("attack_type").count().collect()
                attack_type_dist = {row["attack_type"]: row["count"] for row in attack_type_dist}
                report["label_distributions"]["attack_type"] = attack_type_dist

            # Malware family distribution
            if "malware_family" in df.columns:
                family_dist = df.groupBy("malware_family").count().collect()
                family_dist = {row["malware_family"]: row["count"] for row in family_dist}
                report["label_distributions"]["malware_family"] = family_dist

            print(f"   ✅ Label distributions computed")

        except Exception as e:
            report["warnings"].append(f"Label distribution error: {e}")
            print(f"   ⚠️  Error: {e}")

    # ========================================
    # SAVE REPORT
    # ========================================

    os.makedirs(Config.STAGE2_QUALITY_DIR, exist_ok=True)
    report_path = os.path.join(Config.STAGE2_QUALITY_DIR, "stage2_validation_report.json")

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ========================================
    # SUMMARY
    # ========================================

    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)
    print(f"Overall Status: {report['overall_status']}")

    if report["issues"]:
        print(f"\n❌ Issues ({len(report['issues'])}):")
        for issue in report["issues"]:
            print(f"   - {issue}")

    if report["warnings"]:
        print(f"\n⚠️  Warnings ({len(report['warnings'])}):")
        for warning in report["warnings"]:
            print(f"   - {warning}")

    if report["overall_status"] == "PASS" and not report["issues"]:
        print("\n✅ All checks passed! Data ready for Stage 3 training.")

    print(f"\n📊 Report saved to: {report_path}")
    print("=" * 70)

    # Don't stop Spark if on Databricks (cluster managed)
    if not Config.is_databricks():
        spark.stop()

    return report


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    validate_stage2_outputs()
