"""
Heterogeneous Graph Construction for GNN - PySpark Version

Node types:
- device: each unique device_ip / orig_h source
- service: each unique (id.resp_p, proto)
- subnet: /24 from id.resp_h

Edge types:
- device -> service        ("uses")
- device -> subnet         ("contacts")
- service <-> service      ("cooccurs")

Outputs:
- Torch Geometric HeteroData saved to Config.HETERO_GRAPH_PATH
- node_mappings.json (device/service/subnet -> idx)
- graph_stats.json
"""

import json
import os

import torch
from pyspark.sql import functions as F
from pyspark.sql import SparkSession

from config import Config


def _require(cond, msg):
    if not cond:
        raise RuntimeError(msg)


def build_indexed_nodes(spark, df_src, value_col, id_col):
    """
    Build mapping df[value_col, id_col] where id_col is 0..N-1
    using distinct -> orderBy -> zipWithIndex() on just the unique values.
    No global Window(), so no WindowExec warnings.
    """
    distinct_sorted_rdd = (
        df_src
        .select(F.col(value_col).alias(value_col))
        .where(F.col(value_col).isNotNull() & (F.col(value_col) != ""))
        .distinct()
        .orderBy(value_col)
        .rdd
        .zipWithIndex()
        .map(lambda pair: (pair[0][value_col], int(pair[1])))
    )

    return spark.createDataFrame(
        distinct_sorted_rdd,
        schema=[value_col, id_col],
    )


def build_heterogeneous_graph_spark(spark: SparkSession):
    print("\n" + "=" * 70)
    print("📡 HETEROGENEOUS GRAPH CONSTRUCTION (PySpark)")
    print("=" * 70)

    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        raise RuntimeError(
            "PyTorch Geometric not installed. Install with: pip install torch-geometric"
        )

    # -------------------------------------------------
    # 0. Load inputs
    # -------------------------------------------------
    flows_df = spark.read.parquet(Config.ENGINEERED_DATA_PATH)
    device_df = spark.read.parquet(Config.DEVICE_FEATURES_PATH)

    flow_cnt = flows_df.count()
    dev_cnt = device_df.count()
    print(f"📂 Loaded flows: {flow_cnt:,}")
    print(f"📂 Loaded devices: {dev_cnt:,}")
    _require(flow_cnt > 0, "No flow data to build graph from")
    _require(dev_cnt > 0, "No device data to build graph from")

    # validate required columns in flows
    for c in ["device_ip", "id.resp_h", "id.resp_p", "proto"]:
        _require(c in flows_df.columns, f"Missing required column {c} in flows_df")

    # -------------------------------------------------
    # 1. Derive canonical per-flow keys
    # -------------------------------------------------
    # device_addr = canonical "device identity per flow"
    flows_df = flows_df.withColumn(
        "device_addr",
        F.when(
            F.col("device_ip").isNotNull() & (F.col("device_ip") != ""),
            F.col("device_ip").cast("string"),
        )
        .otherwise(
            F.when(
                F.col("`id.orig_h`").isNotNull() & (F.col("`id.orig_h`") != ""),
                F.col("`id.orig_h`").cast("string"),
            ).otherwise(F.lit(None).cast("string"))
        ),
    )

    # service_key = "<id.resp_p>:<proto>"
    flows_df = flows_df.withColumn(
        "service_key",
        F.concat_ws(
            ":",
            F.col("`id.resp_p`").cast("string"),
            F.col("proto").cast("string"),
        ),
    )

    # subnet_block = first 3 octets of id.resp_h (like /24-ish)
    flows_df = flows_df.withColumn(
        "subnet_block",
        F.regexp_extract(F.col("`id.resp_h`"), r"^(\d+\.\d+\.\d+)\.\d+$", 1),
    )

    for c in ["device_addr", "service_key", "subnet_block"]:
        _require(c in flows_df.columns, f"{c} missing after derivation")

    # -------------------------------------------------
    # 2. Build node tables with deterministic integer IDs
    # -------------------------------------------------
    # DEVICE NODES
    device_nodes = build_indexed_nodes(
        spark,
        flows_df.select(F.col("device_addr").alias("device_ip")),
        "device_ip",
        "device_id",
    )
    num_device_nodes = device_nodes.count()
    print(f"✓ Device nodes: {num_device_nodes:,}")
    _require(num_device_nodes > 0, "No device nodes created")

    # SERVICE NODES
    raw_service = (
        flows_df
        .select(
            F.col("service_key").alias("service_key"),
            F.col("`id.resp_p`").alias("port"),
            F.col("proto").alias("proto"),
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
        "service_idx",
    )

    service_nodes = (
        service_idx_map
        .join(raw_service, on="service_key", how="left")
        .select("service_key", "port", "proto", "service_idx")
    )
    num_service_nodes = service_nodes.count()
    print(f"✓ Service nodes: {num_service_nodes:,}")
    _require(num_service_nodes > 0, "No service nodes created")

    # SUBNET NODES
    subnet_nodes = build_indexed_nodes(
        spark,
        flows_df.select(F.col("subnet_block").alias("subnet_block_raw")),
        "subnet_block_raw",
        "subnet_idx",
    )
    num_subnet_nodes = subnet_nodes.count()
    print(f"✓ Subnet nodes: {num_subnet_nodes:,}")
    _require(num_subnet_nodes > 0, "No subnet nodes created")

    # -------------------------------------------------
    # 3. Attach node IDs to each flow to build edges
    # -------------------------------------------------
    flows_aug = (
        flows_df
        .join(
            device_nodes
            .withColumnRenamed("device_ip", "jn_device_ip")
            .withColumnRenamed("device_id", "jn_device_id"),
            on=F.col("device_addr") == F.col("jn_device_ip"),
            how="left",
        )
        .join(
            service_nodes
            .withColumnRenamed("service_key", "jn_service_key")
            .withColumnRenamed("service_idx", "jn_service_idx"),
            on=F.col("service_key") == F.col("jn_service_key"),
            how="left",
        )
        .join(
            subnet_nodes
            .withColumnRenamed("subnet_block_raw", "jn_subnet_block")
            .withColumnRenamed("subnet_idx", "jn_subnet_idx"),
            on=F.col("subnet_block") == F.col("jn_subnet_block"),
            how="left",
        )
    )

    _require("jn_device_id" in flows_aug.columns, "jn_device_id missing after join")
    _require("jn_service_idx" in flows_aug.columns, "jn_service_idx missing after join")
    _require("jn_subnet_idx" in flows_aug.columns, "jn_subnet_idx missing after join")

    # -------------------------------------------------
    # 4. Edges
    # -------------------------------------------------
    print("\n🔗 Building edges...")

    # device -> service
    print("   Collecting device→service edges...")
    dev_serv_df = (
        flows_aug
        .select(
            F.col("jn_device_id").alias("device_id"),
            F.col("jn_service_idx").alias("service_idx"),
        )
        .where(
            F.col("device_id").isNotNull() &
            F.col("service_idx").isNotNull()
        )
        .distinct()
    )
    dev_serv_rows = dev_serv_df.collect()
    dev_serv_src = [int(r["device_id"]) for r in dev_serv_rows]
    dev_serv_dst = [int(r["service_idx"]) for r in dev_serv_rows]

    # device -> subnet
    print("   Collecting device→subnet edges...")
    dev_subnet_df = (
        flows_aug
        .select(
            F.col("jn_device_id").alias("device_id"),
            F.col("jn_subnet_idx").alias("subnet_idx"),
        )
        .where(
            F.col("device_id").isNotNull() &
            F.col("subnet_idx").isNotNull()
        )
        .distinct()
    )
    dev_subnet_rows = dev_subnet_df.collect()
    dev_subnet_src = [int(r["device_id"]) for r in dev_subnet_rows]
    dev_subnet_dst = [int(r["subnet_idx"]) for r in dev_subnet_rows]

    # service <-> service co-occurrence
    print("   Computing service co-occurrence edges...")

    TOP_K_SERVICES_PER_DEVICE = 50

    @F.udf("array<struct<src:int,dst:int>>")
    def pairwise_limited(arr):
        if arr is None:
            return []
        uniq = sorted(set([x for x in arr if x is not None]))
        if len(uniq) > TOP_K_SERVICES_PER_DEVICE:
            uniq = uniq[:TOP_K_SERVICES_PER_DEVICE]
        out = []
        n = len(uniq)
        for i in range(n):
            for j in range(i + 1, n):
                out.append({"src": int(uniq[i]), "dst": int(uniq[j])})
        return out

    service_lists_df = (
        flows_aug
        .select(
            F.col("jn_device_id").alias("device_id"),
            F.col("jn_service_idx").alias("service_idx"),
        )
        .where(
            F.col("device_id").isNotNull() &
            F.col("service_idx").isNotNull()
        )
        .distinct()
        .groupBy("device_id")
        .agg(F.collect_set("service_idx").alias("services_for_device"))
    )

    pairs_df = (
        service_lists_df
        .withColumn("pairs", pairwise_limited(F.col("services_for_device")))
        .select(F.explode("pairs").alias("p"))
        .select(
            F.col("p.src").alias("left_service"),
            F.col("p.dst").alias("right_service"),
        )
        .distinct()
    )

    pairs_rows = pairs_df.collect()
    svc_co_src = [int(r["left_service"]) for r in pairs_rows]
    svc_co_dst = [int(r["right_service"]) for r in pairs_rows]

    print(f"✓ Device → Service edges: {len(dev_serv_src):,}")
    print(f"✓ Device → Subnet edges: {len(dev_subnet_src):,}")
    print(f"✓ Service ↔ Service edges: {len(svc_co_src):,}")

    # -------------------------------------------------
    # 5. Create HeteroData and fill node features
    # -------------------------------------------------
    print("\n🏗  Creating HeteroData object...")
    data = HeteroData()

    # DEVICE NODE FEATURES
    # We need every column in device_df except identifiers / string labels,
    # but we must alias with backticks so Spark doesn't split dotted names.
    blocklist = {
        "device_ip",
        "device_label",
        "device_malware_family",
    }

    feature_select_exprs = []
    for c in device_df.columns:
        if c in blocklist:
            continue
        feature_select_exprs.append(F.col(f"`{c}`").alias(c))

    # also pull label (if present), aliased safely
    if "device_label" in device_df.columns:
        label_expr = F.col("`device_label`").alias("device_label_tmp")
    else:
        label_expr = F.lit(None).alias("device_label_tmp")

    # and pull device_ip safely
    device_df_clean = (
        device_df
        .select(
            F.col("`device_ip`").alias("device_ip"),
            *feature_select_exprs,
            label_expr,
        )
    )

    # join to node index, make deterministic order by device_id
    device_joined = (
        device_nodes
        .join(device_df_clean, on="device_ip", how="left")
        .orderBy("device_id")
        .fillna(0)
    )

    feature_cols_for_tensor = [
        c for c in device_joined.columns
        if c not in ["device_ip", "device_id", "device_label_tmp"]
    ]

    dev_rows = device_joined.collect()
    device_feature_matrix = []
    device_label_list = []

    for row in dev_rows:
        row_dict = row.asDict()

        # features
        vec = []
        for fc in feature_cols_for_tensor:
            val = row_dict.get(fc, 0)
            if val is None:
                val = 0.0
            elif isinstance(val, bool):
                val = float(val)
            elif isinstance(val, (int, float)):
                val = float(val)
            else:
                try:
                    val = float(val)
                except Exception:
                    val = float(hash(str(val)) % 10000)
            vec.append(val)
        device_feature_matrix.append(vec)

        # label
        lbl_val = row_dict.get("device_label_tmp", None)
        if lbl_val is None:
            device_label_list.append(None)
        else:
            device_label_list.append(1 if str(lbl_val) == "Malicious" else 0)

    device_x = torch.tensor(device_feature_matrix, dtype=torch.float)
    data["device"].x = device_x

    _require(
        device_x.size(0) == num_device_nodes,
        "Device feature tensor row count != number of device nodes",
    )

    if any(v is not None for v in device_label_list):
        cleaned_labels = [(0 if v is None else int(v)) for v in device_label_list]
        device_y = torch.tensor(cleaned_labels, dtype=torch.long)
        data["device"].y = device_y

    # SERVICE NODE FEATURES
    service_nodes_sorted = service_nodes.orderBy("service_idx").collect()
    service_feats = []
    for row in service_nodes_sorted:
        port_val = row["port"]
        proto_val = row["proto"]
        port_num = float(port_val) if port_val is not None else 0.0
        proto_hash = float(hash(str(proto_val)) % 10000)
        service_feats.append([port_num, proto_hash])
    service_x = torch.tensor(service_feats, dtype=torch.float)
    data["service"].x = service_x
    _require(
        service_x.size(0) == num_service_nodes,
        "Service feature tensor row count != number of service nodes",
    )

    # SUBNET NODE FEATURES
    subnet_nodes_sorted = subnet_nodes.orderBy("subnet_idx").collect()
    subnet_feats = []
    for row in subnet_nodes_sorted:
        block = row["subnet_block_raw"]
        if block is None:
            subnet_feats.append([0.0, 0.0, 0.0])
            continue
        parts = block.split(".")
        vals = []
        for i in range(3):
            if i < len(parts):
                try:
                    vals.append(float(parts[i]))
                except Exception:
                    vals.append(0.0)
            else:
                vals.append(0.0)
        subnet_feats.append(vals[:3])
    subnet_x = torch.tensor(subnet_feats, dtype=torch.float)
    data["subnet"].x = subnet_x
    _require(
        subnet_x.size(0) == num_subnet_nodes,
        "Subnet feature tensor row count != number of subnet nodes",
    )

    # -------------------------------------------------
    # 6. Edges into HeteroData
    # -------------------------------------------------
    # device -> service (and reverse)
    edge_dev_serv = torch.tensor(
        [dev_serv_src, dev_serv_dst], dtype=torch.long
    )
    edge_serv_dev = torch.tensor(
        [dev_serv_dst, dev_serv_src], dtype=torch.long
    )

    # device -> subnet (and reverse)
    edge_dev_subnet = torch.tensor(
        [dev_subnet_src, dev_subnet_dst], dtype=torch.long
    )
    edge_subnet_dev = torch.tensor(
        [dev_subnet_dst, dev_subnet_src], dtype=torch.long
    )

    data["device", "uses", "service"].edge_index = edge_dev_serv
    data["service", "used_by", "device"].edge_index = edge_serv_dev
    data["device", "contacts", "subnet"].edge_index = edge_dev_subnet
    data["subnet", "contacted_by", "device"].edge_index = edge_subnet_dev

    # service <-> service co-occurrence (undirected as two directed)
    if len(svc_co_src) > 0:
        edge_svc_svc = torch.tensor(
            [svc_co_src + svc_co_dst, svc_co_dst + svc_co_src],
            dtype=torch.long,
        )
        data["service", "cooccurs", "service"].edge_index = edge_svc_svc
    else:
        edge_svc_svc = None

    # -------------------------------------------------
    # 7. Save artifacts
    # -------------------------------------------------
    os.makedirs(Config.GRAPH_DIR, exist_ok=True)

    torch.save(data, Config.HETERO_GRAPH_PATH)
    print(f"\n✓ Graph saved to: {Config.HETERO_GRAPH_PATH}")

    # node mappings
    device_map_rows = device_nodes.select("device_ip", "device_id").collect()
    service_map_rows = service_nodes.select("service_key", "service_idx").collect()
    subnet_map_rows = subnet_nodes.select("subnet_block_raw", "subnet_idx").collect()

    node_mappings = {
        "device_to_idx": {
            r["device_ip"]: int(r["device_id"]) for r in device_map_rows
        },
        "service_to_idx": {
            r["service_key"]: int(r["service_idx"]) for r in service_map_rows
        },
        "subnet_to_idx": {
            r["subnet_block_raw"]: int(r["subnet_idx"]) for r in subnet_map_rows
        },
    }

    with open(os.path.join(Config.GRAPH_DIR, "node_mappings.json"), "w") as f:
        json.dump(node_mappings, f, indent=2)
    print("✓ Node mappings saved")

    # final stats
    total_nodes = (
            num_device_nodes +
            num_service_nodes +
            num_subnet_nodes
    )

    # count edges exactly like we stored them
    total_edges = (
            edge_dev_serv.size(1) +
            edge_serv_dev.size(1) +
            edge_dev_subnet.size(1) +
            edge_subnet_dev.size(1) +
            (edge_svc_svc.size(1) if edge_svc_svc is not None else 0)
    )

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
    print("✓ Graph stats saved")

    print("\n📊 Graph Summary:")
    print(f"   Nodes: {total_nodes:,}")
    print(f"   Edges: {total_edges:,}")
    print(f"   Edge types: {len(data.edge_types)}")
    print("=" * 70)


if __name__ == "__main__":
    spark = Config.get_spark_session("GraphBuilding")
    build_heterogeneous_graph_spark(spark)
    spark.stop()
