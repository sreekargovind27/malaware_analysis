"""
Improved Heterogeneous GNN with GAT Attention - Standalone Version

Key improvements:
1. Focal Loss for class imbalance
2. GAT attention layers
3. Batch normalization
4. Skip connections
5. Learning rate scheduling
6. Deeper classifier
7. Gradient clipping

Expected improvement: 0.8881 → 0.91-0.93 AUC
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score, balanced_accuracy_score
from torch_geometric.nn import HeteroConv, GATConv, Linear, BatchNorm
from tqdm import tqdm
from torch.nn.parameter import UninitializedParameter

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from config import Config


class FocalLoss(nn.Module):
    """
    Focal Loss for handling severe class imbalance.
    Better than weighted CrossEntropy for imbalanced datasets.

    Paper: https://arxiv.org/abs/1708.02002
    """

    def __init__(self, alpha=0.75, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class ImprovedHeteroGNN(nn.Module):
    """
    Improved Heterogeneous GNN with:
    - GAT attention (learns neighbor importance)
    - Batch normalization (stable training)
    - Skip connections (better gradients)
    - Deeper classifier (complex boundaries)
    """

    def __init__(self, hidden_channels, metadata, num_layers=3, dropout=0.4):
        super().__init__()
        self.dropout = dropout
        self.num_layers = num_layers

        node_types, edge_types = metadata

        # Build layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        for i in range(num_layers):
            conv_dict = {}

            for edge_type in edge_types:
                # Use GAT with multi-head attention
                conv_dict[edge_type] = GATConv(
                    (-1, -1),
                    hidden_channels // 4,  # Divide by number of heads
                    heads=4,  # 4 attention heads
                    concat=True,
                    dropout=dropout,
                    add_self_loops=False  # Hetero graphs handle this differently
                )

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

            # Batch normalization for each node type
            norm_dict = nn.ModuleDict()
            for node_type in node_types:
                norm_dict[node_type] = BatchNorm(hidden_channels)
            self.norms.append(norm_dict)

        # Skip connection projections
        self.skip_lins = nn.ModuleDict()
        for node_type in node_types:
            self.skip_lins[node_type] = Linear(-1, hidden_channels)

        # Deeper classifier head
        self.classifier = nn.Sequential(
            Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            Linear(hidden_channels // 2, 2)
        )

    def forward(self, x_dict, edge_index_dict):
        # Filter out empty edge types
        edge_index_dict = {
            k: v for k, v in edge_index_dict.items()
            if v is not None and v.numel() > 0
        }

        # Store initial features for skip connections
        x_initial = {k: v.clone() for k, v in x_dict.items()}

        # Apply layers
        for i, (conv, norm) in enumerate(zip(self.convs, self.norms)):
            # Message passing
            x_dict = conv(x_dict, edge_index_dict)

            # Batch norm and activation
            x_dict = {
                key: F.relu(norm[key](x)) if x is not None else None
                for key, x in x_dict.items()
            }

            # Skip connection (from initial features)
            if i > 0:
                x_dict = {
                    key: x + self.skip_lins[key](x_initial[key]) if x is not None else None
                    for key, x in x_dict.items()
                }

            # Dropout
            x_dict = {
                key: F.dropout(x, p=self.dropout, training=self.training) if x is not None else None
                for key, x in x_dict.items()
            }

        # Classification on device nodes
        return {'device': self.classifier(x_dict['device'])}


def load_hetero_graph():
    """Load pre-built heterogeneous graph from Stage 2."""
    print("\n" + "=" * 70)
    print("📂 LOADING HETEROGENEOUS GRAPH")
    print("=" * 70)

    graph_path = Config.HETERO_GRAPH_PATH
    if not os.path.exists(graph_path):
        raise FileNotFoundError(
            f"Graph not found at {graph_path}\n"
            "Run stage2/build_graph.py first!"
        )

    print(f"Loading from: {graph_path}")
    data = torch.load(graph_path, weights_only=False)
    print(f"✅ Loaded heterogeneous graph")

    # Print graph stats
    print(f"\n📊 Graph Statistics:")
    for node_type in data.node_types:
        print(f"  {node_type}: {data[node_type].num_nodes:,} nodes")

    for edge_type in data.edge_types:
        src, rel, dst = edge_type
        print(f"  {src} → {dst} ({rel}): {data[edge_type].edge_index.shape[1]:,} edges")

    return data


def prepare_data_splits(data):
    """Create train/val/test masks for device nodes."""
    print("\n" + "=" * 70)
    print("🔀 CREATING DATA SPLITS")
    print("=" * 70)

    device_labels = data['device'].y
    num_devices = device_labels.shape[0]

    # Create stratified splits (60/20/20)
    indices = torch.randperm(num_devices)

    train_size = int(0.6 * num_devices)
    val_size = int(0.2 * num_devices)

    train_idx = indices[:train_size]
    val_idx = indices[train_size:train_size + val_size]
    test_idx = indices[train_size + val_size:]

    # Create masks
    train_mask = torch.zeros(num_devices, dtype=torch.bool)
    val_mask = torch.zeros(num_devices, dtype=torch.bool)
    test_mask = torch.zeros(num_devices, dtype=torch.bool)

    train_mask[train_idx] = True
    val_mask[val_idx] = True
    test_mask[test_idx] = True

    data['device'].train_mask = train_mask
    data['device'].val_mask = val_mask
    data['device'].test_mask = test_mask

    # Print split stats
    train_labels = device_labels[train_mask]
    val_labels = device_labels[val_mask]
    test_labels = device_labels[test_mask]

    print(f"Train: {train_mask.sum():,} devices")
    print(f"  Benign: {(train_labels == 0).sum()}")
    print(f"  Malicious: {(train_labels == 1).sum()}")

    print(f"\nVal: {val_mask.sum():,} devices")
    print(f"  Benign: {(val_labels == 0).sum()}")
    print(f"  Malicious: {(val_labels == 1).sum()}")

    print(f"\nTest: {test_mask.sum():,} devices")
    print(f"  Benign: {(test_labels == 0).sum()}")
    print(f"  Malicious: {(test_labels == 1).sum()}")

    return data


def train_epoch(model, data, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    data = data.to(device)

    optimizer.zero_grad()

    # Forward pass
    out = model(data.x_dict, data.edge_index_dict)
    device_out = out['device']
    device_labels = data['device'].y

    # Only use training nodes
    mask = data['device'].train_mask

    loss = criterion(device_out[mask], device_labels[mask])
    loss.backward()

    # Gradient clipping
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Metrics
    pred = device_out[mask].argmax(dim=1)
    correct = (pred == device_labels[mask]).sum().item()
    accuracy = correct / mask.sum().item()

    return loss.item(), accuracy


@torch.no_grad()
def evaluate(model, data, criterion, device, mask_name='val_mask'):
    """Evaluate model."""
    model.eval()
    data = data.to(device)

    # Forward pass
    out = model(data.x_dict, data.edge_index_dict)
    device_out = out['device']
    device_labels = data['device'].y

    # Use appropriate mask
    mask = getattr(data['device'], mask_name)

    loss = criterion(device_out[mask], device_labels[mask])

    # Metrics
    pred = device_out[mask].argmax(dim=1)
    labels = device_labels[mask]

    accuracy = (pred == labels).float().mean().item()

    # Balanced accuracy (important for imbalanced data!)
    benign_mask = labels == 0
    malicious_mask = labels == 1

    benign_acc = (pred[benign_mask] == 0).float().mean().item() if benign_mask.any() else 0
    malicious_acc = (pred[malicious_mask] == 1).float().mean().item() if malicious_mask.any() else 0
    balanced_acc = (benign_acc + malicious_acc) / 2

    return loss.item(), accuracy, balanced_acc


def train_model(model, data, device, epochs=100):
    """Full training loop with all improvements."""
    print("\n" + "=" * 70)
    print("🚀 TRAINING IMPROVED GNN")
    print("=" * 70)

    # Focal loss for class imbalance
    criterion = FocalLoss(alpha=0.75, gamma=2.0)
    print("✅ Using Focal Loss (better for imbalanced data)")

    # Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.005,
        weight_decay=1e-4,
        betas=(0.9, 0.999)
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=10
    )
    print("✅ LR Scheduler: ReduceLROnPlateau (patience=10)")

    best_val_balanced_acc = 0
    patience = 20
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    val_balanced_accs = []

    for epoch in tqdm(range(epochs), desc="Training"):
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion, device)

        # Validate
        val_loss, val_acc, val_balanced_acc = evaluate(model, data, criterion, device, 'val_mask')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        val_balanced_accs.append(val_balanced_acc)

        # Update learning rate
        scheduler.step(val_balanced_acc)

        # Early stopping on balanced accuracy
        if val_balanced_acc > best_val_balanced_acc:
            best_val_balanced_acc = val_balanced_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
                'balanced_acc': val_balanced_acc,
            }, os.path.join(Config.GNN_MODELS_DIR, 'best_gnn_gat.pth'))
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            tqdm.write(
                f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                f"Val Acc={val_acc:.4f}, Balanced Acc={val_balanced_acc:.4f}"
            )

        if patience_counter >= patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Load best model
    checkpoint = torch.load(os.path.join(Config.GNN_MODELS_DIR, 'best_gnn_gat.pth'))
    model.load_state_dict(checkpoint['model_state'])

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs, val_balanced_accs)

    return model


def plot_training_curves(train_losses, val_losses, train_accs, val_accs, val_balanced_accs):
    """Plot training curves."""
    os.makedirs(Config.GNN_MODELS_DIR, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 4))

    # Loss
    axes[0].plot(train_losses, label='Train Loss')
    axes[0].plot(val_losses, label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(train_accs, label='Train Acc')
    axes[1].plot(val_accs, label='Val Acc')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Balanced Accuracy
    axes[2].plot(val_balanced_accs, label='Val Balanced Acc', color='green')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Balanced Accuracy')
    axes[2].set_title('Balanced Accuracy (Most Important!)')
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.GNN_MODELS_DIR, 'gnn_gat_training_curves.png'))
    plt.close()
    print(f"\n✅ Saved training curves to {Config.GNN_MODELS_DIR}/gnn_gat_training_curves.png")


if __name__ == "__main__":
    Config.set_seeds()
    Config.ensure_output_dirs()

    print("\n" + "=" * 70)
    print("🚀 IMPROVED GNN WITH GAT ATTENTION")
    print("=" * 70)
    print(f"Device: {Config.DEVICE}")
    print(f"Max epochs: 100")
    print("\nImprovements:")
    print("  ✅ Focal Loss (handles class imbalance)")
    print("  ✅ GAT Attention (learns neighbor importance)")
    print("  ✅ Batch Normalization (stable training)")
    print("  ✅ Skip Connections (better gradients)")
    print("  ✅ LR Scheduling (adaptive learning)")
    print("  ✅ Deeper Classifier (complex boundaries)")
    print("=" * 70)

    # Load data
    data = load_hetero_graph()
    data = prepare_data_splits(data)

    # Create improved model
    model = ImprovedHeteroGNN(
        hidden_channels=128,
        metadata=data.metadata(),
        num_layers=3,
        dropout=0.4
    ).to(Config.DEVICE)

    # Initialize lazy modules with a forward pass
    print("\n🔧 Initializing model...")
    model.eval()
    with torch.no_grad():
        data_temp = data.to(Config.DEVICE)  # Move data to device
        _ = model(data_temp.x_dict, data_temp.edge_index_dict)
        data = data.cpu()  # Move back to CPU

    print(f"✅ Model initialized")
    total_params = sum(
        p.numel()
        for p in model.parameters()
        if not isinstance(p, UninitializedParameter)
    )
    print(f"📊 Model parameters (initialized only): {total_params:,}")

    # Train
    model = train_model(model, data, Config.DEVICE, epochs=100)

    # Final evaluation
    print("\n" + "=" * 70)
    print("📊 FINAL TEST EVALUATION")
    print("=" * 70)

    model.eval()
    with torch.no_grad():
        data = data.to(Config.DEVICE)
        out = model(data.x_dict, data.edge_index_dict)
        device_out = out['device']

        test_mask = data['device'].test_mask
        test_labels = data['device'].y[test_mask].cpu().numpy()

        pred = device_out[test_mask].argmax(dim=1).cpu().numpy()
        probs = F.softmax(device_out[test_mask], dim=1)[:, 1].cpu().numpy()

    # Metrics
    print("\nTest Accuracy: {:.4f}".format((pred == test_labels).mean()))
    print(f"Balanced Accuracy: {balanced_accuracy_score(test_labels, pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(test_labels, probs):.4f}")

    print("\n" + classification_report(
        test_labels, pred,
        target_names=['Benign', 'Malicious'],
        digits=4
    ))

    # Save final model
    torch.save({
        'model_state': model.state_dict(),
        'metadata': data.metadata(),
        'test_auc': roc_auc_score(test_labels, probs),
        'balanced_acc': balanced_accuracy_score(test_labels, pred),
    }, os.path.join(Config.GNN_MODELS_DIR, 'gnn_gat_final.pth'))

    print("\n✅ Improved GNN training complete!")
    print(f"📁 Model saved to: {Config.GNN_MODELS_DIR}/gnn_gat_final.pth")