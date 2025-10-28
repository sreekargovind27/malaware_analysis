"""
Heterogeneous Graph Construction for GNN

Builds a heterogeneous graph with three node types:
- Device nodes (IoT devices - originating IPs)
- Service nodes (destination port:protocol combinations)
- Subnet nodes (destination IP subnets)

And three edge types:
- device → service (device uses service)
- device → subnet (device contacts subnet)
- service ↔ service (co-occurrence)
"""
import os
import pandas as pd
import numpy as np
import torch
import json
from collections import defaultdict
from config import Config


def build_heterogeneous_graph():
    """Build heterogeneous graph from flow data."""
    print("\n" + "=" * 70)
    print("HETEROGENEOUS GRAPH CONSTRUCTION")
    print("=" * 70)

    # Check if PyTorch Geometric is available
    try:
        import torch_geometric
        from torch_geometric.data import HeteroData
    except ImportError:
        print("❌ ERROR: PyTorch Geometric not installed")
        print("   Install with: pip install torch-geometric")
        return

    # Load flow data
    print(f"📂 Loading flow data from: {Config.ENGINEERED_DATA_PATH}")
    df = pd.read_parquet(Config.ENGINEERED_DATA_PATH)
    print(f"✓ Loaded {len(df):,} flows")

    # Load device features
    print(f"📂 Loading device features from: {Config.DEVICE_FEATURES_PATH}")
    device_df = pd.read_parquet(Config.DEVICE_FEATURES_PATH)
    print(f"✓ Loaded {len(device_df):,} devices")

    # Build node mappings
    print("\n🔍 Building node mappings...")

    # Device nodes (originating IPs)
    device_to_idx = {ip: idx for idx, ip in enumerate(device_df['id.orig_h'].unique())}
    idx_to_device = {idx: ip for ip, idx in device_to_idx.items()}

    # Service nodes (port:protocol)
    df['service_id'] = df['id.resp_p'].astype(str) + ':' + df['proto'].astype(str)
    service_to_idx = {svc: idx for idx, svc in enumerate(df['service_id'].unique())}
    idx_to_service = {idx: svc for svc, idx in service_to_idx.items()}

    # Subnet nodes (destination IP /24 subnets)
    df['subnet'] = df['id.resp_h'].str.rsplit('.', n=1).str[0]  # Get first 3 octets
    subnet_to_idx = {subnet: idx for idx, subnet in enumerate(df['subnet'].dropna().unique())}
    idx_to_subnet = {idx: subnet for subnet, idx in subnet_to_idx.items()}

    print(f"✓ Device nodes: {len(device_to_idx):,}")
    print(f"✓ Service nodes: {len(service_to_idx):,}")
    print(f"✓ Subnet nodes: {len(subnet_to_idx):,}")

    # Build edges
    print("\n🔗 Building edges...")

    # Edge type 1: device → service
    device_service_edges = df[['id.orig_h', 'service_id']].dropna()
    device_service_src = [device_to_idx[ip] for ip in device_service_edges['id.orig_h']]
    device_service_dst = [service_to_idx[svc] for svc in device_service_edges['service_id']]

    # Edge type 2: device → subnet
    device_subnet_edges = df[['id.orig_h', 'subnet']].dropna()
    device_subnet_src = [device_to_idx[ip] for ip in device_subnet_edges['id.orig_h']]
    device_subnet_dst = [subnet_to_idx[subnet] for subnet in device_subnet_edges['subnet']]

    # Edge type 3: service ↔ service (co-occurrence within same flow)
    # Build co-occurrence matrix
    print("   Computing service co-occurrence...")
    service_groups = df.groupby('id.orig_h')['service_id'].apply(list).values
    service_cooccur = defaultdict(int)
    for services in service_groups:
        unique_services = list(set(services))
        for i, svc1 in enumerate(unique_services):
            for svc2 in unique_services[i + 1:]:
                key = tuple(sorted([svc1, svc2]))
                service_cooccur[key] += 1

    service_service_src = []
    service_service_dst = []
    for (svc1, svc2), count in service_cooccur.items():
        if count >= 5:  # Threshold: co-occur at least 5 times
            service_service_src.append(service_to_idx[svc1])
            service_service_dst.append(service_to_idx[svc2])
            # Add reverse edge (undirected)
            service_service_src.append(service_to_idx[svc2])
            service_service_dst.append(service_to_idx[svc1])

    print(f"✓ Device→Service edges: {len(device_service_src):,}")
    print(f"✓ Device→Subnet edges: {len(device_subnet_src):,}")
    print(f"✓ Service↔Service edges: {len(service_service_src):,}")

    # Create HeteroData object
    print("\n🏗️  Creating HeteroData object...")
    data = HeteroData()

    # Device node features
    device_features = device_df.drop(columns=['id.orig_h', 'device_label', 'device_malware_family'], errors='ignore')
    data['device'].x = torch.FloatTensor(device_features.values)
    data['device'].num_nodes = len(device_to_idx)

    # Device labels
    label_mapping = {'Benign': 0, 'Malicious': 1}
    data['device'].y = torch.LongTensor([label_mapping.get(lbl, 0) for lbl in device_df['device_label']])

    # Service node features (simple: port number and protocol encoding)
    service_features = []
    for svc in idx_to_service.values():
        port, proto = svc.split(':')
        proto_enc = {'tcp': 1, 'udp': 2, 'icmp': 3}.get(proto.lower(), 0)
        service_features.append([float(port), float(proto_enc)])
    data['service'].x = torch.FloatTensor(service_features)
    data['service'].num_nodes = len(service_to_idx)

    # Subnet node features (simple: encoded subnet)
    subnet_features = [[float(idx)] for idx in range(len(subnet_to_idx))]
    data['subnet'].x = torch.FloatTensor(subnet_features)
    data['subnet'].num_nodes = len(subnet_to_idx)

    # Add edges
    data['device', 'uses', 'service'].edge_index = torch.LongTensor([device_service_src, device_service_dst])
    data['device', 'contacts', 'subnet'].edge_index = torch.LongTensor([device_subnet_src, device_subnet_dst])
    data['service', 'cooccurs', 'service'].edge_index = torch.LongTensor([service_service_src, service_service_dst])

    print("✓ HeteroData object created")

    # Save graph
    print(f"\n💾 Saving heterogeneous graph to: {Config.HETERO_GRAPH_PATH}")
    torch.save(data, Config.HETERO_GRAPH_PATH)

    # Save node mappings
    print("💾 Saving node mappings...")
    pd.DataFrame({'device_ip': list(device_to_idx.keys()), 'node_id': list(device_to_idx.values())}).to_parquet(
        Config.DEVICE_NODES_PATH, index=False
    )
    pd.DataFrame({'service': list(service_to_idx.keys()), 'node_id': list(service_to_idx.values())}).to_parquet(
        Config.SERVICE_NODES_PATH, index=False
    )
    pd.DataFrame({'subnet': list(subnet_to_idx.keys()), 'node_id': list(subnet_to_idx.values())}).to_parquet(
        Config.SUBNET_NODES_PATH, index=False
    )

    # Save graph statistics
    stats = {
        'num_device_nodes': len(device_to_idx),
        'num_service_nodes': len(service_to_idx),
        'num_subnet_nodes': len(subnet_to_idx),
        'num_device_service_edges': len(device_service_src),
        'num_device_subnet_edges': len(device_subnet_src),
        'num_service_service_edges': len(service_service_src),
        'total_nodes': len(device_to_idx) + len(service_to_idx) + len(subnet_to_idx),
        'total_edges': len(device_service_src) + len(device_subnet_src) + len(service_service_src)
    }

    with open(Config.GRAPH_STATS_PATH, 'w') as f:
        json.dump(stats, f, indent=2)

    print("✅ Graph construction complete!")
    print("\n📊 Graph Statistics:")
    for key, value in stats.items():
        print(f"   {key}: {value:,}")

    return data


if __name__ == "__main__":
    build_heterogeneous_graph()