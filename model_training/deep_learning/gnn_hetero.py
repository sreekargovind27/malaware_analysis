"""
Heterogeneous GNN for IoT Malware Detection with Optuna (Version-Safe)

Uses manual HeteroConv instead of to_hetero to avoid version conflicts.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from tqdm import tqdm

from config import Config


class HeteroGNN(nn.Module):
    """Heterogeneous GNN with manual HeteroConv layers."""

    def __init__(self, hidden_channels, metadata, num_layers=2, dropout=0.5):
        super().__init__()
        self.dropout = dropout

        # Extract node types and edge types from metadata
        node_types, edge_types = metadata

        # Build HeteroConv layers
        self.convs = nn.ModuleList()
        for i in range(num_layers):
            conv_dict = {}
            for edge_type in edge_types:
                src_type, _, dst_type = edge_type
                # Use -1 for lazy initialization of input size
                conv_dict[edge_type] = SAGEConv((-1, -1), hidden_channels)

            self.convs.append(HeteroConv(conv_dict, aggr='mean'))

        # Classification head for device nodes only
        self.lin = Linear(-1, 2)  # Binary classification

    def forward(self, x_dict, edge_index_dict):
        # Filter out empty edge types (can happen in mini-batches)
        edge_index_dict = {
            k: v for k, v in edge_index_dict.items()
            if v is not None and v.numel() > 0
        }

        # Apply HeteroConv layers
        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)
            # Apply activation and dropout
            x_dict = {key: F.relu(x) for key, x in x_dict.items() if x is not None}
            x_dict = {key: F.dropout(x, p=self.dropout, training=self.training)
                     for key, x in x_dict.items()}

        # Return only device node predictions
        return {'device': self.lin(x_dict['device'])}


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

    # Get device node labels
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

    # Calculate class weights for imbalanced data
    num_benign = (train_labels == 0).sum().item()
    num_malicious = (train_labels == 1).sum().item()

    weight_benign = len(train_labels) / (2 * num_benign)
    weight_malicious = len(train_labels) / (2 * num_malicious)

    class_weights = torch.tensor([weight_benign, weight_malicious], dtype=torch.float)

    print(f"\n⚖️  Class weights: Benign={weight_benign:.3f}, Malicious={weight_malicious:.3f}")

    return data, class_weights


def full_batch_mode(data):
    """Return data for full-batch training (no mini-batches)."""
    # For graphs this small, full-batch training is fine
    return data


def train_epoch(model, data, optimizer, criterion, device):
    """Train for one epoch (full-batch)."""
    model.train()
    data = data.to(device)

    optimizer.zero_grad()

    # Forward pass on entire graph
    out = model(data.x_dict, data.edge_index_dict)

    # Get predictions for device nodes
    device_out = out['device']
    device_labels = data['device'].y

    # Only use training nodes
    mask = data['device'].train_mask

    loss = criterion(device_out[mask], device_labels[mask])
    loss.backward()

    # Gradient clipping to prevent explosion
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    optimizer.step()

    # Calculate metrics
    pred = device_out[mask].argmax(dim=1)
    correct = (pred == device_labels[mask]).sum().item()
    accuracy = correct / mask.sum().item()

    return loss.item(), accuracy


@torch.no_grad()
def evaluate(model, data, criterion, device, mask_name='val_mask', return_preds=False):
    """Evaluate on validation or test set (full-batch)."""
    model.eval()
    data = data.to(device)

    # Forward pass on entire graph
    out = model(data.x_dict, data.edge_index_dict)
    device_out = out['device']
    device_labels = data['device'].y

    # Use appropriate mask
    mask = getattr(data['device'], mask_name)

    loss = criterion(device_out[mask], device_labels[mask])

    # Calculate metrics
    pred = device_out[mask].argmax(dim=1)
    correct = (pred == device_labels[mask]).sum().item()
    accuracy = correct / mask.sum().item()

    if return_preds:
        probs = F.softmax(device_out[mask], dim=1)[:, 1].cpu().numpy()
        return loss.item(), accuracy, pred.cpu().numpy(), probs, device_labels[mask].cpu().numpy()

    return loss.item(), accuracy


def optuna_objective(trial, data, class_weights, metadata):
    """Optuna objective function (full-batch)."""

    # Sample hyperparameters (simplified for speed)
    hidden_channels = trial.suggest_categorical('hidden_channels', [64, 128])
    num_layers = trial.suggest_int('num_layers', 2, 3)
    dropout = trial.suggest_float('dropout', 0.3, 0.6)
    lr = trial.suggest_float('lr', 1e-3, 1e-2, log=True)
    weight_decay = trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True)

    # Create model
    model = HeteroGNN(
        hidden_channels=hidden_channels,
        metadata=metadata,
        num_layers=num_layers,
        dropout=dropout
    ).to(Config.DEVICE)

    # Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(Config.DEVICE))

    # Train for limited epochs (faster search)
    max_epochs = 10  # Reduced from 20
    best_val_acc = 0
    patience = 3  # Early stop in search
    patience_counter = 0

    for epoch in range(max_epochs):
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion, Config.DEVICE)
        val_loss, val_acc = evaluate(model, data, criterion, Config.DEVICE, mask_name='val_mask')

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

        # Pruning
        trial.report(val_acc, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return best_val_acc


def optimize_hyperparameters(data, class_weights, metadata):
    """Run Optuna hyperparameter optimization."""
    print("\n" + "=" * 70)
    print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (GNN)")
    print("=" * 70)

    if not Config.USE_OPTUNA:
        print("\n⭐ Optuna disabled, using default params")
        return {
            'hidden_channels': 64,
            'num_layers': 2,
            'dropout': 0.5,
            'lr': 0.01,
            'weight_decay': 5e-4
        }

    def objective(trial):
        return optuna_objective(trial, data, class_weights, metadata)

    study = optuna.create_study(
        direction='maximize',
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(
        objective,
        n_trials=Config.OPTUNA_N_TRIALS,
        timeout=Config.OPTUNA_TIMEOUT,
    )

    print(f"\n✅ Best trial: {study.best_trial.number}")
    print(f"   Best val accuracy: {study.best_value:.4f}")
    print(f"   Best params: {study.best_params}")

    return study.best_params


def train_gnn(model, data, optimizer, criterion, device, epochs=100):
    """Full training loop with early stopping (full-batch)."""
    print("\n" + "=" * 70)
    print("🚀 TRAINING GNN (FULL-BATCH MODE)")
    print("=" * 70)

    best_val_acc = 0
    patience = 15
    patience_counter = 0

    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    for epoch in tqdm(range(epochs), desc="Training GNN"):
        # Train
        train_loss, train_acc = train_epoch(model, data, optimizer, criterion, device)

        # Validate
        val_loss, val_acc = evaluate(model, data, criterion, device, mask_name='val_mask')

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Early stopping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model
            torch.save({
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'epoch': epoch,
                'val_acc': val_acc,
            }, os.path.join(Config.GNN_MODELS_DIR, 'best_gnn.pth'))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"\n  Early stopping at epoch {epoch + 1}")
                break

        if epoch % 10 == 0:
            tqdm.write(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
                       f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    # Load best model
    checkpoint = torch.load(os.path.join(Config.GNN_MODELS_DIR, 'best_gnn.pth'))
    model.load_state_dict(checkpoint['model_state'])

    # Plot training curves
    plot_training_curves(train_losses, val_losses, train_accs, val_accs)

    return model


def plot_training_curves(train_losses, val_losses, train_accs, val_accs):
    """Plot training and validation curves."""
    os.makedirs(Config.GNN_MODELS_DIR, exist_ok=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy
    ax2.plot(train_accs, label='Train Acc')
    ax2.plot(val_accs, label='Val Acc')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(Config.GNN_MODELS_DIR, 'gnn_training_curves.png'))
    plt.close()
    print(f"\n✅ Saved training curves to {Config.GNN_MODELS_DIR}/gnn_training_curves.png")


if __name__ == "__main__":
    Config.set_seeds()
    Config.ensure_output_dirs()

    print("\n" + "=" * 70)
    print("🚀 GNN TRAINING CONFIGURATION")
    print("=" * 70)
    print(f"Device: {Config.DEVICE}")
    print(f"Optuna enabled: {Config.USE_OPTUNA}")
    print(f"Max epochs: 100")
    print("=" * 70)

    # Load graph
    data = load_hetero_graph()

    # Prepare splits
    data, class_weights = prepare_data_splits(data)

    # Optuna hyperparameter search
    best_params = optimize_hyperparameters(data, class_weights, data.metadata())

    print("\n🏗️  Building final model with best params:")
    for key, value in best_params.items():
        print(f"   {key}: {value}")

    # Initialize model with best params
    model = HeteroGNN(
        hidden_channels=best_params['hidden_channels'],
        metadata=data.metadata(),
        num_layers=best_params['num_layers'],
        dropout=best_params['dropout']
    ).to(Config.DEVICE)

    # Optimizer and loss with best params
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=best_params['lr'],
        weight_decay=best_params['weight_decay']
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(Config.DEVICE))

    # Train (full-batch mode - no loaders needed)
    model = train_gnn(model, data, optimizer, criterion, Config.DEVICE)

    # Final test evaluation
    print("\n" + "=" * 70)
    print("📊 FINAL TEST EVALUATION")
    print("=" * 70)

    test_loss, test_acc, preds, probs, labels = evaluate(
        model, data, criterion, Config.DEVICE, mask_name='test_mask', return_preds=True
    )

    print(f"\nTest Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")

    # Classification report
    print("\n" + classification_report(
        labels, preds,
        target_names=['Benign', 'Malicious'],
        digits=4
    ))

    # ROC-AUC
    roc_auc = roc_auc_score(labels, probs)
    print(f"ROC-AUC: {roc_auc:.4f}")

    # Save final model with metadata
    torch.save({
        'model_state': model.state_dict(),
        'metadata': data.metadata(),
        'best_params': best_params,
        'test_acc': test_acc,
        'test_auc': roc_auc,
    }, os.path.join(Config.GNN_MODELS_DIR, 'gnn_final.pth'))

    print("\n✅ GNN training complete!")