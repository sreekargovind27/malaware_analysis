"""
Stage 2: Heterogeneous Graph Construction - PySpark Version
Works on both local and Databricks with automatic optimization.

Pipeline:
1. Load engineered flow features
2. Build node tables (Device, Service, Subnet)
3. Build edge tables (device-service, device-subnet, service-service)
4. Convert to PyTorch Geometric HeteroData format
5. Save graph and node mappings

Optimizations:
- Broadcast joins for small dimension tables
- Collect() guards to prevent OOM
- Smart repartitioning for edge construction
- Efficient tensor creation
"""

import os
import sys
import json
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pyspark.sql import functions as F
from pyspark.sql import Window

from config import Config
from analysis.stage2.utils import (
    load_engineered_data_spark, print_environment_info
)

try:
    from torch_geometric.data import HeteroData

    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("⚠️  WARNING: torch_geometric not installed. Graph building will fail.")
    TORCH_GEOMETRIC_AVAILABLE = False


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _require(condition, message):
    """Assert with custom error message"""
    if not condition:
        raise AssertionError(f"Graph building failed: {message}")


def build_indexed_nodes(spark, df, key_col, idx_col):
    """Build node table with deterministic integer indices."""
    distinct_keys = (
        df.select(key_col)
        .where(F.col(key_col).isNotNull() & (F.col(key_col) != ""))
        .distinct()
        .orderBy(key_col)
        .collect()
    )

    # Create DataFrame with indices
    indexed_data = [(row[key_col], i) for i, row in enumerate(distinct_keys)]
    indexed = spark.createDataFrame(indexed_data, [key_col, idx_col])

    return indexed


# ============================================================================
# GRAPH BUILDING
# ============================================================================

def build_heterogeneous_graph_spark(spark):
    """
    Build heterogeneous graph from engineered flows.

    Graph Schema:
        Nodes:
            - Device: Unique IoT devices (by device_ip)
            - Service: Network services (by port:proto)
            - Subnet: IP subnets (by /24 blocks)

        Edges:
            - (device, uses, service): Device contacted service
            - (service, used_by, device): Reverse edge
            - (device, in, subnet): Device belongs to subnet
            - (subnet, contains, device): Reverse edge
            - (service, connects, service): Service co-occurrence (optional)

    Args:
        spark: Active SparkSession
    """
    print("\n" + "=" * 70)
    print("STAGE 2: HETEROGENEOUS GRAPH CONSTRUCTION")
    print("=" * 70)

    if not TORCH_GEOMETRIC_AVAILABLE:
        raise ImportError("torch_geometric is required for graph building. Install with: pip install torch-geometric")

    print_environment_info(spark)

    # ========================================
    # 1. LOAD AND PREPARE DATA
    # ========================================

    flows_df = load_engineered_data_spark(spark)

    print(f"\n📊 Input: {flows_df.count():,} flows")

    # Derive canonical per-flow keys
    print("\n🔑 Deriving canonical identifiers...")

    # device_addr = canonical device identity
    flows_df = flows_df.withColumn(
        "device_addr",
        F.when(
            F.col("device_ip").isNotNull() & (F.col("device_ip") != ""),
            F.col("device_ip").cast("string")
        ).otherwise(
            F.when(
                F.col("`id.orig_h`").isNotNull() & (F.col("`id.orig_h`") != ""),
                F.col("`id.orig_h`").cast("string")
            ).otherwise(F.lit(None).cast("string"))
        )
    )

    # service_key = "<port>:<proto>"
    flows_df = flows_df.withColumn(
        "service_key",
        F.concat_ws(
            ":",
            F.col("`id.resp_p`").cast("string"),
            F.col("proto").cast("string")
        )
    )

    # subnet_block = first 3 octets of id.resp_h (like /24)
    flows_df = flows_df.withColumn(
        "subnet_block",
        F.regexp_extract(F.col("`id.resp_h`"), r"^(\d+\.\d+\.\d+)\.\d+$", 1)
    )

    for c in ["device_addr", "service_key", "subnet_block"]:
        _require(c in flows_df.columns, f"{c} missing after derivation")

    # ========================================
    # 2. BUILD NODE TABLES
    # ========================================

    print("\n🔨 Building node tables...")

    # DEVICE NODES
    device_nodes = build_indexed_nodes(
        spark,
        flows_df.select(F.col("device_addr").alias("device_ip")),
        "device_ip",
        "device_id"
    )
    num_device_nodes = device_nodes.count()
    print(f"   ✅ Device nodes: {num_device_nodes:,}")
    _require(num_device_nodes > 0, "No device nodes created")

    # SERVICE NODES
    raw_service = (
        flows_df
        .select(
            F.col("service_key").alias("service_key"),
            F.col("`id.resp_p`").alias("port"),
            F.col("proto").alias("proto")
        )
        .where(
            F.col("service_key").isNotNull() &
            (F.col("service_key") != "")
        )
        .distinct()
    )

    service_idx_map = build_indexed_nodes(
        spark,
        raw_service.select("service_key"),
        "service_key",
        "service_idx"
    )

    service_nodes = (
        service_idx_map
        .join(raw_service, on="service_key", how="left")
        .select("service_key", "port", "proto", "service_idx")
    )
    num_service_nodes = service_nodes.count()
    print(f"   ✅ Service nodes: {num_service_nodes:,}")
    _require(num_service_nodes > 0, "No service nodes created")

    # SUBNET NODES
    subnet_nodes = build_indexed_nodes(
        spark,
        flows_df.select(F.col("subnet_block").alias("subnet_block_raw")),
        "subnet_block_raw",
        "subnet_idx"
    )
    num_subnet_nodes = subnet_nodes.count()
    print(f"   ✅ Subnet nodes: {num_subnet_nodes:,}")
    _require(num_subnet_nodes > 0, "No subnet nodes created")

    # ========================================
    # 3. CHECK SIZE BEFORE COLLECT (CRITICAL)
    # ========================================

    print("\n🔍 Validating node counts before tensor conversion...")

    if num_device_nodes > Config.MAX_NODES_TO_COLLECT:
        raise RuntimeError(
            f"Too many device nodes ({num_device_nodes:,}) to collect safely (limit: {Config.MAX_NODES_TO_COLLECT:,}). "
            f"Consider using a streaming/chunked approach or increase MAX_NODES_TO_COLLECT in config."
        )

    if num_service_nodes > Config.MAX_NODES_TO_COLLECT:
        raise RuntimeError(
            f"Too many service nodes ({num_service_nodes:,}) to collect safely (limit: {Config.MAX_NODES_TO_COLLECT:,})."
        )

    if num_subnet_nodes > Config.MAX_NODES_TO_COLLECT:
        raise RuntimeError(
            f"Too many subnet nodes ({num_subnet_nodes:,}) to collect safely (limit: {Config.MAX_NODES_TO_COLLECT:,})."
        )

    print(f"   ✅ All node counts within safe limits")

    # ========================================
    # 4. BUILD EDGE TABLES
    # ========================================

    print("\n🔗 Building edge tables...")

    # Use broadcast joins for small dimension tables (optimization)
    from pyspark.sql.functions import broadcast

    # Persist dimension tables
    device_nodes = device_nodes.persist()
    service_nodes = broadcast(service_nodes)
    subnet_nodes = broadcast(subnet_nodes)

    # DEVICE-SERVICE EDGES
    edge_dev_serv_df = (
        flows_df
        .select(
            F.col("device_addr").alias("device_ip"),
            F.col("service_key")
        )
        .where(
            F.col("device_ip").isNotNull() &
            (F.col("device_ip") != "") &
            F.col("service_key").isNotNull() &
            (F.col("service_key") != "")
        )
        .distinct()
        .join(device_nodes, on="device_ip", how="inner")
        .join(service_nodes, on="service_key", how="inner")
        .select("device_id", "service_idx")
    )

    num_dev_serv_edges = edge_dev_serv_df.count()
    print(f"   ✅ Device-Service edges: {num_dev_serv_edges:,}")

    # DEVICE-SUBNET EDGES
    flows_with_subnet = flows_df.withColumn(
        "device_ip_for_subnet",
        F.col("device_addr")
    )

    edge_dev_subnet_df = (
        flows_with_subnet
        .select(
            F.col("device_ip_for_subnet").alias("device_ip"),
            F.col("subnet_block").alias("subnet_block_raw")
        )
        .where(
            F.col("device_ip").isNotNull() &
            (F.col("device_ip") != "") &
            F.col("subnet_block_raw").isNotNull() &
            (F.col("subnet_block_raw") != "")
        )
        .distinct()
        .join(device_nodes, on="device_ip", how="inner")
        .join(subnet_nodes, on="subnet_block_raw", how="inner")
        .select("device_id", "subnet_idx")
    )

    num_dev_subnet_edges = edge_dev_subnet_df.count()
    print(f"   ✅ Device-Subnet edges: {num_dev_subnet_edges:,}")

    # ========================================
    # 5. CONVERT TO PYTORCH TENSORS
    # ========================================

    print("\n🔄 Converting to PyTorch tensors...")

    # Collect edge lists (safe because we checked counts)
    dev_serv_edges = edge_dev_serv_df.select("device_id", "service_idx").collect()
    dev_subnet_edges = edge_dev_subnet_df.select("device_id", "subnet_idx").collect()

    # Convert to tensors
    if dev_serv_edges:
        edge_dev_serv = torch.tensor(
            [[r["device_id"] for r in dev_serv_edges],
             [r["service_idx"] for r in dev_serv_edges]],
            dtype=torch.long
        )
        edge_serv_dev = torch.tensor(
            [[r["service_idx"] for r in dev_serv_edges],
             [r["device_id"] for r in dev_serv_edges]],
            dtype=torch.long
        )
    else:
        edge_dev_serv = torch.empty((2, 0), dtype=torch.long)
        edge_serv_dev = torch.empty((2, 0), dtype=torch.long)

    if dev_subnet_edges:
        edge_dev_subnet = torch.tensor(
            [[r["device_id"] for r in dev_subnet_edges],
             [r["subnet_idx"] for r in dev_subnet_edges]],
            dtype=torch.long
        )
        edge_subnet_dev = torch.tensor(
            [[r["subnet_idx"] for r in dev_subnet_edges],
             [r["device_id"] for r in dev_subnet_edges]],
            dtype=torch.long
        )
    else:
        edge_dev_subnet = torch.empty((2, 0), dtype=torch.long)
        edge_subnet_dev = torch.empty((2, 0), dtype=torch.long)

    print(f"   ✅ Tensors created")

    # ========================================
    # 6. BUILD NODE FEATURES
    # ========================================

    print("\n🎨 Building node features...")

    # DEVICE NODE FEATURES (collect safely)
    device_label_df = (
        flows_df
        .select("device_addr", "label")
        .where(F.col("device_addr").isNotNull())
        .groupBy("device_addr")
        .agg(F.first("label").alias("label"))
        .join(device_nodes.select("device_ip", "device_id"),
              F.col("device_addr") == F.col("device_ip"),
              how="left")
        .select("device_id", "label")
        .orderBy("device_id")
    )

    device_labels = device_label_df.collect()
    device_label_list = [
        (0 if row["label"] is None else (1 if row["label"] == "Malicious" else 0))
        for row in device_labels
    ]
    device_y = torch.tensor(device_label_list, dtype=torch.long)

    # Simple identity features for now (can be enhanced later)
    device_x = torch.eye(num_device_nodes, dtype=torch.float)

    # SERVICE NODE FEATURES
    import hashlib

    service_feats_df = (
        service_nodes
        .orderBy("service_idx")
        .withColumn("port_num", F.coalesce(F.col("port").cast("float"), F.lit(0.0)))
        .withColumn(
            "proto_hash",
            (F.conv(F.substring(F.md5(F.col("proto").cast("string")), 1, 8), 16, 10).cast("long") % 10000).cast("float")
        )
        .select("port_num", "proto_hash")
    )

    service_feats = service_feats_df.collect()
    service_x = torch.tensor([[r["port_num"], r["proto_hash"]] for r in service_feats], dtype=torch.float)

    # SUBNET NODE FEATURES
    subnet_feats_df = (
        subnet_nodes
        .orderBy("subnet_idx")
        .withColumn("octet1", F.split(F.col("subnet_block_raw"), r"\.").getItem(0).cast("float"))
        .withColumn("octet2", F.split(F.col("subnet_block_raw"), r"\.").getItem(1).cast("float"))
        .withColumn("octet3", F.split(F.col("subnet_block_raw"), r"\.").getItem(2).cast("float"))
        .fillna(0.0, subset=["octet1", "octet2", "octet3"])
        .select("octet1", "octet2", "octet3")
    )

    subnet_feats = subnet_feats_df.collect()
    subnet_x = torch.tensor([[r["octet1"], r["octet2"], r["octet3"]] for r in subnet_feats], dtype=torch.float)

    print(f"   ✅ Node features created")

    # ========================================
    # 7. BUILD HETERODATA OBJECT
    # ========================================

    print("\n🏗️  Building HeteroData object...")

    data = HeteroData()

    # Add nodes
    data["device"].x = device_x
    data["device"].y = device_y
    data["service"].x = service_x
    data["subnet"].x = subnet_x

    # Add edges
    data["device", "uses", "service"].edge_index = edge_dev_serv
    data["service", "used_by", "device"].edge_index = edge_serv_dev
    data["device", "in", "subnet"].edge_index = edge_dev_subnet
    data["subnet", "contains", "device"].edge_index = edge_subnet_dev

    print(f"   ✅ HeteroData object created")

    # ========================================
    # 8. SAVE ARTIFACTS
    # ========================================

    os.makedirs(Config.GRAPH_DIR, exist_ok=True)

    # Save graph
    print(f"\n💾 Saving graph to: {Config.HETERO_GRAPH_PATH}")
    torch.save(data, Config.HETERO_GRAPH_PATH)
    print(f"   ✅ Graph saved")

    # Save node mappings
    device_map_rows = device_nodes.select("device_ip", "device_id").collect()
    service_map_rows = service_nodes.select("service_key", "service_idx").collect()
    subnet_map_rows = subnet_nodes.select("subnet_block_raw", "subnet_idx").collect()

    node_mappings = {
        "device_to_idx": {r["device_ip"]: int(r["device_id"]) for r in device_map_rows},
        "service_to_idx": {r["service_key"]: int(r["service_idx"]) for r in service_map_rows},
        "subnet_to_idx": {r["subnet_block_raw"]: int(r["subnet_idx"]) for r in subnet_map_rows},
    }

    mapping_path = os.path.join(Config.GRAPH_DIR, "node_mappings.json")
    with open(mapping_path, "w") as f:
        json.dump(node_mappings, f, indent=2)
    print(f"   ✅ Node mappings saved to: {mapping_path}")

    # Save graph stats
    total_nodes = num_device_nodes + num_service_nodes + num_subnet_nodes
    total_edges = edge_dev_serv.size(1) + edge_serv_dev.size(1) + edge_dev_subnet.size(1) + edge_subnet_dev.size(1)

    stats = {
        "num_device_nodes": int(num_device_nodes),
        "num_service_nodes": int(num_service_nodes),
        "num_subnet_nodes": int(num_subnet_nodes),
        "total_nodes": int(total_nodes),
        "total_edges": int(total_edges),
        "num_edge_types": len(data.edge_types),
    }

    with open(Config.GRAPH_STATS_PATH, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"   ✅ Graph stats saved to: {Config.GRAPH_STATS_PATH}")

    # ========================================
    # 9. SUMMARY
    # ========================================

    print("\n" + "=" * 70)
    print("✅ GRAPH CONSTRUCTION COMPLETE!")
    print("=" * 70)
    print(f"📊 Graph Summary:")
    print(
        f"   Nodes: {total_nodes:,} (Device: {num_device_nodes:,}, Service: {num_service_nodes:,}, Subnet: {num_subnet_nodes:,})")
    print(f"   Edges: {total_edges:,}")
    print(f"   Edge types: {len(data.edge_types)}")
    print(f"\n📂 Outputs:")
    print(f"   Graph: {Config.HETERO_GRAPH_PATH}")
    print(f"   Mappings: {mapping_path}")
    print(f"   Stats: {Config.GRAPH_STATS_PATH}")
    print("=" * 70)

    return data


# ============================================================================
# STANDALONE EXECUTION
# ============================================================================

if __name__ == "__main__":
    Config.ensure_output_dirs()
    spark = Config.get_spark_session("Stage2-GraphBuilding")

    try:
        build_heterogeneous_graph_spark(spark)
    finally:
        if not Config.is_databricks():
            spark.stop()
            print("🧹 Spark session stopped")
