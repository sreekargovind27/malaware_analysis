"""
Stage 1: Graph Structure Analysis - PySpark Version
Works on both local and Databricks.

Analyzes graph structure feasibility for GNN models:
- Device, Service, Subnet node counts
- Edge connectivity (device-service, device-subnet)
- Graph density and connectivity
- Isolated node detection

Returns feasibility assessment for GNN models (GO/WARNING/NO-GO).
"""

from pyspark.sql import DataFrame, functions as F


def analyze_graph_structure(df: DataFrame) -> dict:
    """
    Analyze graph structure for GNN model feasibility.

    Constructs a conceptual heterogeneous graph:
    - Nodes: Devices (by device_ip), Services (by port:proto), Subnets (by /24 blocks)
    - Edges: device-uses-service, device-in-subnet

    Args:
        df: Spark DataFrame with raw IoT-23 data

    Returns:
        dict: Graph structure feasibility stats
    """
    print("\n" + "=" * 70)
    print("GRAPH STRUCTURE ANALYSIS (GNN Feasibility)")
    print("=" * 70)

    stats = {
        'node_counts': {},
        'edge_counts': {},
        'connectivity': {},
        'feasibility': 'GO',
        'reason': 'Sufficient graph structure for GNN models'
    }

    # ========================================
    # 1. DERIVE GRAPH KEYS
    # ========================================

    print("\n[1/5] Deriving graph keys (device, service, subnet)...")

    # Device: Use id.orig_h as device identifier
    df = df.withColumn(
        "device_addr",
        F.when(
            F.col("`id.orig_h`").isNotNull() & (F.col("`id.orig_h`") != ""),
            F.col("`id.orig_h`")
        ).otherwise(F.lit(None))
    )

    # Service: port:proto
    df = df.withColumn(
        "service_key",
        F.concat_ws(
            ":",
            F.col("`id.resp_p`").cast("string"),
            F.col("proto").cast("string")
        )
    )

    # Subnet: First 3 octets of id.resp_h (like /24)
    df = df.withColumn(
        "subnet_block",
        F.regexp_extract(F.col("`id.resp_h`"), r"^(\d+\.\d+\.\d+)\.\d+$", 1)
    )

    print(f"   ✅ Graph keys derived")

    # ========================================
    # 2. COUNT NODES
    # ========================================

    print("\n[2/5] Counting nodes...")

    # Device nodes
    num_devices = df.select("device_addr").where(
        F.col("device_addr").isNotNull() & (F.col("device_addr") != "")
    ).distinct().count()

    # Service nodes
    num_services = df.select("service_key").where(
        F.col("service_key").isNotNull() & (F.col("service_key") != "")
    ).distinct().count()

    # Subnet nodes
    num_subnets = df.select("subnet_block").where(
        F.col("subnet_block").isNotNull() & (F.col("subnet_block") != "")
    ).distinct().count()

    total_nodes = num_devices + num_services + num_subnets

    stats['node_counts'] = {
        'devices': num_devices,
        'services': num_services,
        'subnets': num_subnets,
        'total': total_nodes
    }

    print(f"   Device nodes:  {num_devices:,}")
    print(f"   Service nodes: {num_services:,}")
    print(f"   Subnet nodes:  {num_subnets:,}")
    print(f"   Total nodes:   {total_nodes:,}")

    # ========================================
    # 3. COUNT EDGES
    # ========================================

    print("\n[3/5] Counting edges...")

    # Device-Service edges
    num_dev_service_edges = df.select("device_addr", "service_key").where(
        F.col("device_addr").isNotNull() &
        (F.col("device_addr") != "") &
        F.col("service_key").isNotNull() &
        (F.col("service_key") != "")
    ).distinct().count()

    # Device-Subnet edges
    num_dev_subnet_edges = df.select("device_addr", "subnet_block").where(
        F.col("device_addr").isNotNull() &
        (F.col("device_addr") != "") &
        F.col("subnet_block").isNotNull() &
        (F.col("subnet_block") != "")
    ).distinct().count()

    total_edges = num_dev_service_edges + num_dev_subnet_edges

    stats['edge_counts'] = {
        'device_service': num_dev_service_edges,
        'device_subnet': num_dev_subnet_edges,
        'total': total_edges
    }

    print(f"   Device-Service edges: {num_dev_service_edges:,}")
    print(f"   Device-Subnet edges:  {num_dev_subnet_edges:,}")
    print(f"   Total edges:          {total_edges:,}")

    # ========================================
    # 4. CONNECTIVITY ANALYSIS
    # ========================================

    print("\n[4/5] Analyzing connectivity...")

    # Average degree per device
    if num_devices > 0:
        avg_services_per_device = num_dev_service_edges / num_devices
        stats['connectivity']['avg_services_per_device'] = round(avg_services_per_device, 2)
        print(f"   Avg services per device: {avg_services_per_device:.2f}")

    # Graph density (edges / possible_edges)
    # For bipartite graph: max edges = num_devices * num_services
    if num_devices > 0 and num_services > 0:
        max_possible_edges = num_devices * num_services
        density = (num_dev_service_edges / max_possible_edges) * 100
        stats['connectivity']['graph_density'] = round(density, 4)
        print(f"   Graph density: {density:.4f}%")

    # Check for isolated devices (devices with no service connections)
    device_with_edges = df.select("device_addr").where(
        F.col("device_addr").isNotNull() &
        (F.col("device_addr") != "") &
        F.col("service_key").isNotNull() &
        (F.col("service_key") != "")
    ).distinct().count()

    isolated_devices = num_devices - device_with_edges
    stats['connectivity']['isolated_devices'] = isolated_devices
    stats['connectivity']['isolated_device_percentage'] = round(
        (isolated_devices / num_devices * 100) if num_devices > 0 else 0, 2
    )

    print(f"   Isolated devices: {isolated_devices:,} ({stats['connectivity']['isolated_device_percentage']}%)")

    # ========================================
    # 5. FEASIBILITY ASSESSMENT
    # ========================================

    print("\n[5/5] Assessing GNN feasibility...")

    # Check minimum requirements for GNN
    if total_nodes < 100:
        stats['feasibility'] = 'NO-GO'
        stats['reason'] = f'Too few nodes: {total_nodes} (need 100+ for GNN)'
    elif num_devices < 50:
        stats['feasibility'] = 'NO-GO'
        stats['reason'] = f'Too few device nodes: {num_devices} (need 50+)'
    elif total_edges < 200:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = f'Few edges: {total_edges} (prefer 1000+ for robust GNN training)'
    elif stats['connectivity']['isolated_device_percentage'] > 50:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = (
            f"{stats['connectivity']['isolated_device_percentage']:.1f}% devices are isolated. "
            f"GNN works best with connected graphs."
        )
    elif avg_services_per_device < 2:
        stats['feasibility'] = 'WARNING'
        stats['reason'] = (
            f'Low connectivity: avg {avg_services_per_device:.1f} services per device. '
            f'Prefer 5+ connections per device.'
        )

    # Recommendations
    stats['recommendations'] = []

    if stats['feasibility'] == 'GO':
        stats['recommendations'].append(
            'Graph structure is suitable for GNN models (GAT, GraphSAGE, HeteroGNN)'
        )
        stats['recommendations'].append(
            'Consider heterogeneous GNN to leverage device-service-subnet relationships'
        )
    elif stats['feasibility'] == 'WARNING':
        stats['recommendations'].append(
            'GNN possible but may not outperform traditional ML significantly'
        )
        stats['recommendations'].append(
            'Consider traditional ML (XGBoost, LightGBM) as baseline'
        )
    else:  # NO-GO
        stats['recommendations'].append(
            'GNN not recommended. Use traditional ML models instead.'
        )

    # GNN-specific recommendations
    if num_devices >= 1000 and total_edges >= 5000:
        stats['recommendations'].append(
            'Strong candidate for Graph Attention Networks (GAT) - can learn edge importance'
        )

    if num_services > 100:
        stats['recommendations'].append(
            'High service diversity - good for learning service-based attack patterns'
        )

    # ========================================
    # SUMMARY
    # ========================================

    print("\n" + "=" * 70)
    print("GRAPH STRUCTURE SUMMARY")
    print("=" * 70)
    print(
        f"Total nodes:  {total_nodes:,} (Device: {num_devices:,}, Service: {num_services:,}, Subnet: {num_subnets:,})")
    print(f"Total edges:  {total_edges:,}")
    print(f"Connectivity: {avg_services_per_device:.2f} services/device")
    print(f"Isolated:     {isolated_devices:,} devices ({stats['connectivity']['isolated_device_percentage']}%)")
    print(f"\n✅ Feasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"   Reason: {stats['reason']}")

    print(f"\n💡 Recommendations:")
    for rec in stats['recommendations']:
        print(f"   - {rec}")

    print("=" * 70)

    return stats


if __name__ == "__main__":
    """Standalone testing"""
    import os
    from config import Config
    from analysis.stage1.utils import load_raw_data, save_json_report

    Config.ensure_output_dirs()
    spark = Config.get_spark_session("Stage1-GraphAnalysis-Test")

    try:
        df = load_raw_data(spark)

        # Graph structure analysis
        graph_stats = analyze_graph_structure(df)
        output_path = os.path.join(Config.STAGE1_FEASIBILITY_DIR, 'graph_feasibility.json')
        save_json_report(graph_stats, output_path)

    finally:
        if not Config.is_databricks():
            spark.stop()
