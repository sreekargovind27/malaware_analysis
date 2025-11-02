"""
Stage 2 Comprehensive Quality Report - PySpark Version

Validates Stage 2 outputs before proceeding to Stage 3 training.
Saves validation report to outputs/stage2_quality/
"""

import json
import os
from datetime import datetime

from pyspark.sql import SparkSession
from pyspark.sql import functions as F

from config import Config


def validate_stage2_outputs():
    """Comprehensive validation of Stage 2 outputs using PySpark."""
    print("\n" + "=" * 70)
    print("STAGE 2 QUALITY VALIDATION REPORT")
    print("=" * 70)

    # Spark session (independent of the pipeline Spark; that's fine)
    spark = SparkSession.builder.appName("Stage2Validation").getOrCreate()
    spark.sparkContext.setLogLevel("WARN")

    report = {
        "validation_date": datetime.now().isoformat(),
        "files_checked": {},
        "overall_status": "PASS",  # we will downgrade if something is actually broken
        "issues": [],
        "warnings": [],
    }

    # -------------------------------------------------
    # [1/6] FLOW FEATURES
    # -------------------------------------------------
    print("\n[1/6] Validating flow features...")
    if os.path.exists(Config.ENGINEERED_DATA_PATH):
        try:
            df = spark.read.parquet(Config.ENGINEERED_DATA_PATH)
            features = Config.get_feature_list()

            row_count = df.count()

            # missing values in model features
            total_nulls = 0
            for feat in features:
                if feat in df.columns:
                    total_nulls += df.filter(F.col(f"`{feat}`").isNull()).count()

            # duplicate groups:
            # We consider rows identical across all "real" columns.
            # NOTE: dataframes with dotted cols are referenced via backticks.
            cols_for_dupe = [c for c in df.columns if not c.startswith("_")]
            spark_cols_for_dupe = [F.col(f"`{c}`") for c in cols_for_dupe]

            dup_groups = (
                df.groupBy(*spark_cols_for_dupe)
                .count()
                .filter(F.col("count") > 1)
                .count()
            )

            report["files_checked"]["flow_features"] = {
                "exists": True,
                "row_count": int(row_count),
                "feature_count": len(features),
                "missing_values": int(total_nulls),
                "duplicate_groups": int(dup_groups),
            }

            # FAIL-level checks
            if row_count < 1000:
                report["issues"].append("Flow features: Too few rows (<1000)")
                report["overall_status"] = "FAIL"
            if len(features) < 10:
                report["issues"].append("Flow features: Too few features (<10)")
                report["overall_status"] = "FAIL"

            # WARN-level checks
            if total_nulls > 0:
                report["warnings"].append(
                    f"Flow features: {total_nulls} missing values found"
                )

            print(f"   ✓ Rows: {row_count:,}")
            print(f"   ✓ Features: {len(features)}")
            print(f"   ✓ Duplicate groups: {dup_groups}")
        except Exception as e:
            report["files_checked"]["flow_features"] = {
                "exists": True,
                "error": str(e),
            }
            report["issues"].append(f"Flow features: Error reading file - {e}")
            report["overall_status"] = "FAIL"
            print(f"   ❌ Error reading file: {e}")
    else:
        report["files_checked"]["flow_features"] = {"exists": False}
        report["issues"].append("Flow features file not found")
        report["overall_status"] = "FAIL"
        print("   ❌ File not found")

    # -------------------------------------------------
    # [2/6] DEVICE FEATURES
    # -------------------------------------------------
    print("\n[2/6] Validating device features...")
    if os.path.exists(Config.DEVICE_FEATURES_PATH):
        try:
            device_df = spark.read.parquet(Config.DEVICE_FEATURES_PATH)
            device_count = device_df.count()

            benign_count = device_df.filter(
                F.col("device_label") == "Benign"
            ).count()
            malicious_count = device_df.filter(
                F.col("device_label") == "Malicious"
            ).count()

            report["files_checked"]["device_features"] = {
                "exists": True,
                "device_count": int(device_count),
                "benign_devices": int(benign_count),
                "malicious_devices": int(malicious_count),
            }

            if device_count < 100:
                report["warnings"].append(
                    "Device features: Very few devices (<100)"
                )

            print(f"   ✓ Devices: {device_count:,}")
            print(f"   ✓ Benign: {benign_count:,}")
            print(f"   ✓ Malicious: {malicious_count:,}")
        except Exception as e:
            report["files_checked"]["device_features"] = {
                "exists": True,
                "error": str(e),
            }
            report["warnings"].append(
                f"Device features: Error reading file - {e}"
            )
            print(f"   ⚠️  Error reading file: {e}")
    else:
        report["files_checked"]["device_features"] = {"exists": False}
        report["warnings"].append(
            "Device features not found (GNN unavailable)"
        )
        print("   ⚠️  File not found (GNN will be unavailable)")

    # -------------------------------------------------
    # [3/6] TRAIN / VAL / TEST SPLITS
    # -------------------------------------------------
    print("\n[3/6] Validating train/val/test splits...")
    train_exists = os.path.exists(Config.TRAIN_SET_PATH)
    val_exists = os.path.exists(Config.VAL_SET_PATH)
    test_exists = os.path.exists(Config.TEST_SET_PATH)

    if train_exists and test_exists:
        try:
            train_df = spark.read.parquet(Config.TRAIN_SET_PATH)
            test_df = spark.read.parquet(Config.TEST_SET_PATH)

            train_count = train_df.count()
            test_count = test_df.count()

            if val_exists:
                val_df = spark.read.parquet(Config.VAL_SET_PATH)
                val_count = val_df.count()
            else:
                val_df = None
                val_count = 0

            total_count = train_count + val_count + test_count
            test_ratio = (
                round(test_count / total_count, 2) if total_count > 0 else 0
            )

            report["files_checked"]["splits"] = {
                "train_exists": True,
                "val_exists": val_exists,
                "test_exists": True,
                "train_count": int(train_count),
                "val_count": int(val_count),
                "test_count": int(test_count),
                "test_ratio": float(test_ratio),
            }

            if test_ratio < 0.15 or test_ratio > 0.25:
                report["warnings"].append(
                    f"Unusual test ratio: {test_ratio}"
                )

            print(f"   ✓ Train: {train_count:,} rows")
            if val_exists:
                print(f"   ✓ Val: {val_count:,} rows")
            print(f"   ✓ Test: {test_count:,} rows")
            print(f"   ✓ Test ratio: {test_ratio}")
        except Exception as e:
            report["files_checked"]["splits"] = {
                "train_exists": train_exists,
                "val_exists": val_exists,
                "test_exists": test_exists,
                "error": str(e),
            }
            report["issues"].append(f"Splits: Error reading files - {e}")
            report["overall_status"] = "FAIL"
            print(f"   ❌ Error reading files: {e}")
    else:
        report["files_checked"]["splits"] = {
            "train_exists": train_exists,
            "val_exists": val_exists,
            "test_exists": test_exists,
        }
        report["issues"].append("Train/test splits missing")
        report["overall_status"] = "FAIL"
        print("   ❌ Split files not found")

    # -------------------------------------------------
    # [4/6] GRAPH / GNN INPUT
    # -------------------------------------------------
    print("\n[4/6] Validating heterogeneous graph...")
    if os.path.exists(Config.HETERO_GRAPH_PATH):
        # graph exists on disk -> must also have stats for us to trust it
        graph_stats_path = Config.GRAPH_STATS_PATH

        if os.path.exists(graph_stats_path):
            try:
                with open(graph_stats_path, "r") as f:
                    graph_stats = json.load(f)

                report["files_checked"]["graph"] = {
                    "exists": True,
                    "stats": graph_stats,
                }

                num_device_nodes = graph_stats.get("num_device_nodes", 0)
                num_service_nodes = graph_stats.get("num_service_nodes", 0)
                num_subnet_nodes = graph_stats.get("num_subnet_nodes", 0)
                total_edges = graph_stats.get("total_edges", 0)

                print(f"   ✓ Device nodes: {num_device_nodes:,}")
                print(f"   ✓ Service nodes: {num_service_nodes:,}")
                print(f"   ✓ Subnet nodes: {num_subnet_nodes:,}")
                print(f"   ✓ Total edges: {total_edges:,}")

                # sanity: if graph saved but has 0 edges something's broken
                if total_edges == 0:
                    report["warnings"].append(
                        "Graph has 0 edges (GNN will be useless)"
                    )

            except Exception as e:
                # stats unreadable -> this is a *real* problem
                report["files_checked"]["graph"] = {
                    "exists": True,
                    "stats": None,
                    "error": str(e),
                }
                report["issues"].append(
                    f"Graph stats unreadable/corrupt - {e}"
                )
                report["overall_status"] = "FAIL"
                print(f"   ❌ Error reading graph stats: {e}")
        else:
            # graph tensor exists but stats missing:
            # call this a warning, not a fail, because training could still load the torch file.
            report["files_checked"]["graph"] = {
                "exists": True,
                "stats": None,
            }
            report["warnings"].append(
                "Graph exists but graph_stats.json missing"
            )
            print("   ⚠️  Graph exists (no stats file)")
    else:
        # graph file NOT there at all:
        # we do NOT fail pipeline here, we just warn.
        report["files_checked"]["graph"] = {"exists": False}
        report["warnings"].append(
            "Heterogeneous graph not found (GNN unavailable)"
        )
        print("   ⚠️  Graph not found (GNN will be unavailable)")

    # -------------------------------------------------
    # [5/6] FEATURE LIST (list of model features)
    # -------------------------------------------------
    print("\n[5/6] Validating feature list...")
    if os.path.exists(Config.FEATURE_LIST_PATH):
        features = Config.get_feature_list()
        report["files_checked"]["feature_list"] = {
            "exists": True,
            "count": len(features),
        }
        print(f"   ✓ Feature list: {len(features)} features")
        if len(features) == 0:
            report["issues"].append("Feature list is empty")
            report["overall_status"] = "FAIL"
    else:
        report["files_checked"]["feature_list"] = {"exists": False}
        report["issues"].append("Feature list file not found")
        report["overall_status"] = "FAIL"
        print("   ❌ Feature list not found")

    # -------------------------------------------------
    # [6/6] LABEL DISTRIBUTIONS
    # -------------------------------------------------
    print("\n[6/6] Validating label distributions...")
    if train_exists and test_exists:
        try:
            train_df = spark.read.parquet(Config.TRAIN_SET_PATH)
            test_df = spark.read.parquet(Config.TEST_SET_PATH)

            splits_to_union = [train_df, test_df]
            if val_exists:
                val_df = spark.read.parquet(Config.VAL_SET_PATH)
                splits_to_union.append(val_df)

            full_df = splits_to_union[0]
            for split_df in splits_to_union[1:]:
                full_df = full_df.unionByName(split_df, allowMissingColumns=True)

            # ---------------------------
            # 1. Binary distribution ("label": Benign / Malicious)
            # ---------------------------
            binary_rows = (
                full_df.groupBy(Config.TARGET_COL)  # "label"
                .count()
                .collect()
            )
            binary_dist = {
                row[Config.TARGET_COL]: int(row["count"])
                for row in binary_rows
                if row[Config.TARGET_COL] is not None
            }

            # ---------------------------
            # 2. Fine-grained attack_type distribution
            #    (PortScan, C&C, DDoS, FileDownload, Attack, Malware, Benign, etc.)
            #    We keep all non-null attack_type values.
            # ---------------------------
            attack_type_rows = (
                full_df
                .filter(F.col(Config.DETAILED_TARGET_COL).isNotNull())
                .groupBy(Config.DETAILED_TARGET_COL)  # "attack_type"
                .count()
                .orderBy(F.col("count").desc())
                .collect()
            )
            attack_type_dist = {
                row[Config.DETAILED_TARGET_COL]: int(row["count"])
                for row in attack_type_rows
                if row[Config.DETAILED_TARGET_COL] is not None
            }

            # ---------------------------
            # 3. Malware family distribution
            #    We ONLY consider malicious rows and non-null families.
            #    This prevents "Benign" or None from inflating the count.
            # ---------------------------
            family_rows = (
                full_df
                .filter(F.col(Config.TARGET_COL) == "Malicious")          # only malicious flows
                .filter(F.col(Config.FAMILY_TARGET_COL).isNotNull())       # must have a family
                .groupBy(Config.FAMILY_TARGET_COL)                        # "malware_family"
                .count()
                .orderBy(F.col("count").desc())
                .collect()
            )
            family_dist = {
                row[Config.FAMILY_TARGET_COL]: int(row["count"])
                for row in family_rows
                if row[Config.FAMILY_TARGET_COL] is not None
            }

            report["files_checked"]["label_distributions"] = {
                "binary": binary_dist,
                "attack_type": attack_type_dist,
                "malware_family": family_dist,
            }

            # imbalance warning (binary Benign vs Malicious)
            if "Benign" in binary_dist and "Malicious" in binary_dist:
                max_c = max(binary_dist.values())
                min_c = min(binary_dist.values())
                if min_c > 0:
                    ratio = max_c / min_c
                    if ratio > 100:
                        report["warnings"].append(
                            f"Severe class imbalance: {ratio:.1f}:1 ratio"
                        )

            total_samples = sum(binary_dist.values())

            print(f"   ✓ Binary classes: {len(binary_dist)}")
            print(f"   ✓ Total samples: {total_samples:,}")
            print(f"   ✓ Attack types: {len(attack_type_dist)}")
            print(f"   ✓ Malware families: {len(family_dist)}")

        except Exception as e:
            report["warnings"].append(
                f"Label distributions: Error - {e}"
            )
            print(f"   ⚠️  Error computing distributions: {e}")

    # -------------------------------------------------
    # Done with Spark
    # -------------------------------------------------
    spark.stop()

    # -------------------------------------------------
    # Save report JSON to disk
    # -------------------------------------------------
    os.makedirs(Config.STAGE2_QUALITY_DIR, exist_ok=True)
    report_path = os.path.join(
        Config.STAGE2_QUALITY_DIR, "stage2_validation_report.json"
    )

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # -------------------------------------------------
    # Final summary to console
    # -------------------------------------------------
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

    return report


if __name__ == "__main__":
    validate_stage2_outputs()
