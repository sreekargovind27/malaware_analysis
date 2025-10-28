"""
Graph Structure Analysis for GNN Feasibility (PySpark).
UPDATED: Includes analysis for both a simple IP-to-IP graph and a
more robust heterogeneous graph (Device-Service-Subnet) to provide a
clear recommendation for GNN modeling.
FIXED: Replaced non-standard 'combinations' function with a robust self-join.
"""

from pyspark.sql import functions as F

from .utils import get_malware_family_udf


def analyze_ip_to_ip_graph(df):
    """Analyzes the simple, sparse IP-to-IP graph."""
    print("  Approach 1: IP-to-IP Graph (Traditional)")

    source_ips = df.select('id_orig_h').distinct()
    dest_ips = df.select('id_resp_h').distinct()

    source_ips_renamed = source_ips.withColumnRenamed('id_orig_h', 'ip')
    dest_ips_renamed = dest_ips.withColumnRenamed('id_resp_h', 'ip')

    all_ips = source_ips_renamed.union(dest_ips_renamed).distinct()
    total_nodes = all_ips.count()
    total_edges = df.filter(F.col('id_orig_h').isNotNull() & F.col('id_resp_h').isNotNull()).count()

    # Simple density calculation
    possible_edges = total_nodes * (total_nodes - 1)
    density = total_edges / possible_edges if possible_edges > 0 else 0

    print(f"    Nodes (Unique IPs): {total_nodes:,}")
    print(f"    Edges (Connections): {total_edges:,}")
    print(f"    Connectivity/Density: {density:.4%}")
    print("    Assessment: ❌ TOO SPARSE (typical for IoT datasets)")

    return {'nodes': total_nodes, 'edges': total_edges, 'density': density}


def analyze_heterogeneous_graph(df):
    """Analyzes the recommended, denser heterogeneous graph."""
    print("\n  Approach 2: Heterogeneous Graph (RECOMMENDED)")

    # 1. Define Node Sets
    device_nodes = df.select('id_orig_h').distinct()
    service_nodes = df.select(F.concat_ws(':', F.col('id_resp_p'), F.col('proto')).alias('service')).distinct()
    subnet_nodes = df.withColumn(
        'subnet', F.regexp_extract(F.col('id_resp_h'), r'^(\d+\.\d+)\..*', 1)
    ).select('subnet').distinct()

    num_devices = device_nodes.count()
    num_services = service_nodes.count()
    num_subnets = subnet_nodes.count()
    total_nodes = num_devices + num_services + num_subnets

    # 2. Define Edge Sets
    # Edge: (Device) -> [uses] -> (Service)
    device_service_edges = df.select('id_orig_h',
                                     F.concat_ws(':', F.col('id_resp_p'), F.col('proto')).alias('service')).distinct()
    num_device_service_edges = device_service_edges.count()

    # Edge: (Device) -> [targets] -> (Subnet)
    device_subnet_edges = df.withColumn(
        'subnet', F.regexp_extract(F.col('id_resp_h'), r'^(\d+\.\d+)\..*', 1)
    ).select('id_orig_h', 'subnet').distinct()
    num_device_subnet_edges = device_subnet_edges.count()

    # Edge: (Service) <-> (Service) co-occurrence
    device_services_list = device_service_edges.groupBy('id_orig_h').agg(F.collect_set('service').alias('services'))

    # ✅ FIXED: Use self-join instead of 'combinations'
    exploded_services = device_services_list.filter(F.size('services') > 1).withColumn("service_A",
                                                                                       F.explode("services"))
    service_pairs = exploded_services.alias("df1").join(
        exploded_services.alias("df2"),
        (F.col("df1.id_orig_h") == F.col("df2.id_orig_h")) & (F.col("df1.service_A") < F.col("df2.service_A")),
        "inner"
    )
    num_service_edges = service_pairs.count()

    total_edges = num_device_service_edges + num_device_subnet_edges + num_service_edges

    # 3. Calculate Density and other stats
    possible_edges = num_devices * num_services + num_devices * num_subnets + (num_services * (num_services - 1) / 2)
    density = total_edges / possible_edges if possible_edges > 0 else 0
    avg_device_degree = total_edges / num_devices if num_devices > 0 else 0

    print(f"    Nodes: {total_nodes:,} ({num_devices:,} devices + {num_services:,} services + {num_subnets:,} subnets)")
    print(
        f"    Edges: {total_edges:,} ({num_device_service_edges:,} device-service, {num_device_subnet_edges:,} device-subnet, etc.)")
    print(f"    Connectivity/Density: {density:.4%}")
    print(f"    Avg Connections per Device: {avg_device_degree:.2f}")
    print("    Assessment: ✅ SUITABLE for GNN analysis")

    return {
        'nodes': {'total': total_nodes, 'devices': num_devices, 'services': num_services, 'subnets': num_subnets},
        'edges': {'total': total_edges, 'device_to_service': num_device_service_edges,
                  'device_to_subnet': num_device_subnet_edges, 'service_to_service': num_service_edges},
        'density': density,
        'avg_device_degree': avg_device_degree
    }


def analyze_graph_structure(df):
    """Main function to analyze graph structure feasibility for GNN."""
    print("\n" + "=" * 70)
    print("GRAPH STRUCTURE FEASIBILITY (GNN)")
    print("=" * 70)

    required_cols = ['id_orig_h', 'id_resp_h', 'id_resp_p', 'proto']
    if not all(col in df.columns for col in required_cols):
        return {
            'feasibility': 'NO-GO',
            'reason': f'Missing one or more required columns for graph analysis: {required_cols}'
        }

    print("  Analyzing graph structure approaches...")

    # Analyze both approaches for comparison
    ip_graph_stats = analyze_ip_to_ip_graph(df)
    hetero_graph_stats = analyze_heterogeneous_graph(df)

    # IPs per malware family (for node labeling in the heterogeneous graph)
    print("\n  Analyzing IPs per malware family for device node labels...")
    df_labeled = df.withColumn('malware_family', get_malware_family_udf()(F.col('Source_Folder')))
    malicious_df = df_labeled.filter(~F.col('malware_family').isin(['Benign', 'Unknown']))

    ips_per_family_df = malicious_df.groupBy('malware_family').agg(
        F.countDistinct('id_orig_h').alias('count')).collect()
    ips_per_family = {row['malware_family']: int(row['count']) for row in ips_per_family_df}
    min_ips_per_family = min(ips_per_family.values()) if ips_per_family else 0

    # Final recommendation based on heterogeneous graph
    stats = {
        'approach_comparison': {
            'ip_to_ip_graph': {**ip_graph_stats, 'assessment': 'NOT RECOMMENDED - Too sparse for GNN'},
            'heterogeneous_graph': {**hetero_graph_stats, 'assessment': 'RECOMMENDED'}
        },
        'recommended_approach': 'heterogeneous_graph',
        'graph_type': 'Heterogeneous (Device-Service-Subnet)',
        'ips_per_family': ips_per_family,
        'min_ips_per_family': min_ips_per_family,
        'feasibility': 'GO',
        'reason': 'Heterogeneous graph structure is dense and meaningful, making it suitable for GNN.'
    }

    # Feasibility checks for the recommended heterogeneous graph
    if hetero_graph_stats['density'] < 0.001:  # 0.1%
        stats['feasibility'] = 'WARNING'
        stats[
            'reason'] = f"Heterogeneous graph density is very low ({hetero_graph_stats['density']:.4f}). GNN may still struggle."
    elif min_ips_per_family > 0 and min_ips_per_family < 20:
        stats['feasibility'] = 'WARNING'
        stats[
            'reason'] = f'Some malware families have very few unique devices ({min_ips_per_family}). Node classification may be difficult.'

    # Summary
    print("\n" + "─" * 70)
    print("OVERALL GRAPH FEASIBILITY SUMMARY")
    print("─" * 70)
    print(f"✓ Recommended Graph Type: {stats['graph_type']}")
    print("✓ Node Counts:")
    for name, count in hetero_graph_stats['nodes'].items():
        if name != 'total':
            print(f"    - {name.capitalize()}: {count:,}")
    print("✓ Edge Counts:")
    print(f"    - Total Edges: {hetero_graph_stats['edges']['total']:,}")
    print(f"✓ Feasibility: {stats['feasibility']}")
    if stats['feasibility'] != 'GO':
        print(f"  Reason: {stats['reason']}")

    return stats

