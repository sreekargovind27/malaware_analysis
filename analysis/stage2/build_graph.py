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

import json
import os
import sys
from collections import defaultdict

import pandas as pd
import torch

# Add project root to path to import Config
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import Config


def reconstruct_proto_column(df):
    """Reconstruct proto column from one-hot encoded columns."""
    proto_columns = [col for col in df.columns if col.startswith('proto_')]

    if not proto_columns:
        print("⚠️  WARNING: No proto columns found! Using 'unknown'")
        df['proto'] = 'unknown'
        return df

    # Find which proto column has value 1 for each row
    proto_values = []
    for idx in range(len(df)):
        found_proto = 'unknown'
        for col in proto_columns:
            if df[col].iloc[idx] == 1:
                found_proto = col.replace('proto_', '')
                break
        proto_values.append(found_proto)

    df['proto'] = proto_values
    return df


def build_heterogeneous_graph():
    """Build heterogeneous graph from flow data."""
    print("\n" + "=" * 70)
    print("HETEROGENEOUS GRAPH CONSTRUCTION")
    print("=" * 70)

    try:
        from torch_geometric.data import HeteroData
    except ImportError:
        print("❌ ERROR: PyTorch Geometric not installed. Install with: pip install torch-geometric")
        return

    print(f"📂 Loading flow data from: {Config.ENGINEERED_DATA_PATH}")
    df = pd.read_parquet(Config.ENGINEERED_DATA_PATH)
    print(f"✓ Loaded {len(df):,} flows")

    print(f"📂 Loading device features from: {Config.DEVICE_FEATURES_PATH}")
    device_df = pd.read_parquet(Config.DEVICE_FEATURES_PATH)
    print(f"✓ Loaded {len(device_df):,} devices")

    # Reconstruct proto column from one-hot encoding
    print("\n🔄 Reconstructing proto column from one-hot encoding...")
    df = reconstruct_proto_column(df)
    print(f"✓ Proto values found: {df['proto'].nunique()} unique protocols")

    print("\n🔍 Building node mappings...")
    device_to_idx = {ip: idx for idx, ip in enumerate(device_df['id_orig_h'].unique())}

    df['service_id'] = df['id_resp_p'].astype(str) + ':' + df['proto'].astype(str)
    service_to_idx = {svc: idx for idx, svc in enumerate(df['service_id'].unique())}

    df['subnet'] = df['id_resp_h'].str.rsplit('.', n=1).str[0]
    subnet_to_idx = {subnet: idx for idx, subnet in enumerate(df['subnet'].dropna().unique())}

    print(f"✓ Device nodes: {len(device_to_idx):,}")
    print(f"✓ Service nodes: {len(service_to_idx):,}")
    print(f"✓ Subnet nodes: {len(subnet_to_idx):,}")

    print("\n🔗 Building edges...")
    # Filter df to only include devices that are in device_to_idx to prevent KeyErrors
    df_filtered = df[df['id_orig_h'].isin(device_to_idx)]

    device_service_edges = df_filtered[['id_orig_h', 'service_id']].dropna()
    device_service_src = [device_to_idx[ip] for ip in device_service_edges['id_orig_h']]
    device_service_dst = [service_to_idx[svc] for svc in device_service_edges['service_id']]

    device_subnet_edges = df_filtered[['id_orig_h', 'subnet']].dropna()
    device_subnet_src = [device_to_idx[ip] for ip in device_subnet_edges['id_orig_h']]
    device_subnet_dst = [subnet_to_idx[subnet] for subnet in device_subnet_edges['subnet']]

    print("   Computing service co-occurrence...")
    service_groups = df_filtered.groupby('id_orig_h')['service_id'].apply(list).values
    service_cooccur = defaultdict(int)
    for services in service_groups:
        unique_services = list(set(services))
        # Limit to top 5 services per device to avoid explosion
        unique_services = unique_services[:5]
        for i, svc1 in enumerate(unique_services):
            for svc2 in unique_services[i + 1:]:
                service_cooccur[tuple(sorted([svc1, svc2]))] += 1

    # Only keep edges with at least 2 co-occurrences (stronger connections)
    service_cooccur = {k: v for k, v in service_cooccur.items() if v >= 2}

    service_cooccur_edges = [(service_to_idx[svc1], service_to_idx[svc2])
                             for (svc1, svc2) in service_cooccur.keys()]
    if service_cooccur_edges:
        service_src, service_dst = zip(*service_cooccur_edges)
    else:
        service_src, service_dst = [], []

    print(f"✓ Device → Service edges: {len(device_service_src):,}")
    print(f"✓ Device → Subnet edges: {len(device_subnet_src):,}")
    print(f"✓ Service ↔ Service edges: {len(service_src):,}")

    # Create HeteroData object
    print("\n🏗️  Creating HeteroData object...")
    data = HeteroData()

    # Device node features (use device-level aggregated features)
    print("   Adding device node features...")
    device_feature_cols = [col for col in device_df.columns
                           if col not in ['id_orig_h', 'device_label', 'device_malware_family']]
    device_features = device_df[device_feature_cols].fillna(0).values
    data['device'].x = torch.tensor(device_features, dtype=torch.float)

    # Device labels
    if 'device_label' in device_df.columns:
        labels = (device_df['device_label'] == 'Malicious').astype(int).values
        data['device'].y = torch.tensor(labels, dtype=torch.long)
    print(f"   ✓ Device nodes: {data['device'].x.shape}")

    # Service node features (simple: port number + proto encoding)
    print("   Adding service node features...")
    service_features = []
    for svc in service_to_idx.keys():
        port, proto = svc.split(':')
        # Simple encoding: [port_number, is_tcp, is_udp, is_icmp]
        feat = [
            float(port),
            1.0 if proto == 'tcp' else 0.0,
            1.0 if proto == 'udp' else 0.0,
            1.0 if proto == 'icmp' else 0.0
        ]
        service_features.append(feat)
    data['service'].x = torch.tensor(service_features, dtype=torch.float)
    print(f"   ✓ Service nodes: {data['service'].x.shape}")

    # Subnet node features (simple: subnet encoding)
    print("   Adding subnet node features...")
    subnet_features = []
    for subnet in subnet_to_idx.keys():
        # Handle both IPv4 and IPv6
        if ':' in subnet:
            # IPv6 - use hash encoding
            hash_val = hash(subnet) % 10000
            feat = [float(hash_val), 1.0, 0.0]  # [hash, is_ipv6, is_ipv4]
        else:
            # IPv4 - use last 3 octets as features
            octets = subnet.split('.')
            feat = [float(octets[i]) if i < len(octets) else 0.0 for i in range(min(3, len(octets)))]
            # Pad if less than 3 octets
            while len(feat) < 3:
                feat.append(0.0)
            feat = feat[:3]  # Ensure exactly 3 features
        subnet_features.append(feat)
    data['subnet'].x = torch.tensor(subnet_features, dtype=torch.float)
    print(f"   ✓ Subnet nodes: {data['subnet'].x.shape}")

    # Add edges (with reverse edges for message passing)
    print("   Adding edges...")

    # Device → Service (forward)
    data['device', 'uses', 'service'].edge_index = torch.tensor(
        [device_service_src, device_service_dst], dtype=torch.long
    )
    # Service → Device (reverse - so device nodes receive messages)
    data['service', 'used_by', 'device'].edge_index = torch.tensor(
        [device_service_dst, device_service_src], dtype=torch.long
    )

    # Device → Subnet (forward)
    data['device', 'contacts', 'subnet'].edge_index = torch.tensor(
        [device_subnet_src, device_subnet_dst], dtype=torch.long
    )
    # Subnet → Device (reverse - so device nodes receive messages)
    data['subnet', 'contacted_by', 'device'].edge_index = torch.tensor(
        [device_subnet_dst, device_subnet_src], dtype=torch.long
    )

    # Service ↔ Service (already bidirectional)
    if service_src:
        data['service', 'cooccurs', 'service'].edge_index = torch.tensor(
            [list(service_src) + list(service_dst),
             list(service_dst) + list(service_src)],
            dtype=torch.long
        )

    # Save graph
    os.makedirs(Config.GRAPH_DIR, exist_ok=True)
    graph_path = Config.HETERO_GRAPH_PATH
    torch.save(data, graph_path)
    print(f"\n✅ Graph saved to: {graph_path}")

    # Save node mappings for reference
    mappings = {
        'device_to_idx': device_to_idx,
        'service_to_idx': service_to_idx,
        'subnet_to_idx': subnet_to_idx
    }
    mappings_path = os.path.join(Config.GRAPH_DIR, 'node_mappings.json')
    with open(mappings_path, 'w') as f:
        # Convert to serializable format
        json.dump({
            'device_to_idx': {str(k): int(v) for k, v in device_to_idx.items()},
            'service_to_idx': {str(k): int(v) for k, v in service_to_idx.items()},
            'subnet_to_idx': {str(k): int(v) for k, v in subnet_to_idx.items()}
        }, f, indent=2)
    print(f"✅ Node mappings saved to: {mappings_path}")

    # Print summary
    print("\n📊 Graph Summary:")
    total_nodes = data['device'].num_nodes + data['service'].num_nodes + data['subnet'].num_nodes
    total_edges = (data['device', 'uses', 'service'].edge_index.shape[1] +
                   data['service', 'used_by', 'device'].edge_index.shape[1] +
                   data['device', 'contacts', 'subnet'].edge_index.shape[1] +
                   data['subnet', 'contacted_by', 'device'].edge_index.shape[1])
    if 'service' in data.edge_types:
        if ('service', 'cooccurs', 'service') in data.edge_types:
            total_edges += data['service', 'cooccurs', 'service'].edge_index.shape[1]

    print(f"   Nodes: {total_nodes:,}")
    print(f"   Edges: {total_edges:,}")
    print(f"   Edge types: {len(data.edge_types)}")
    print("=" * 70)


if __name__ == "__main__":
    build_heterogeneous_graph()
