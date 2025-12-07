"""
Evaluate trained GNN on test set.

Loads the trained heterogeneous GNN and evaluates on the test split,
generating detailed metrics, confusion matrix, and ROC curve.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, roc_curve)
from torch_geometric.loader import NeighborLoader

from config import Config


def load_trained_gnn():
    """Load trained GNN model."""
    print("\n" + "=" * 70)
    print("📂 LOADING TRAINED GNN")
    print("=" * 70)

    # Load graph data
    data = torch.load(Config.HETERO_GRAPH_PATH, weights_only=False)

    # Load model checkpoint
    model_path = os.path.join(Config.GNN_MODELS_DIR, 'gnn_final.pth')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=Config.DEVICE, weights_only=False)

    # Import model class
    from models.deep_learning.gnn_hetero import HeteroGNN

    # Recreate model
    model = HeteroGNN(
        hidden_channels=64,
        metadata=checkpoint['metadata'],
        num_layers=2
    ).to(Config.DEVICE)

    model.load_state_dict(checkpoint['model_state'])
    model.eval()

    print(f"✅ Loaded model from: {model_path}")
    print(f"   Test accuracy from training: {checkpoint.get('test_acc', 'N/A')}")
    print(f"   Test AUC from training: {checkpoint.get('test_auc', 'N/A')}")

    return model, data


def prepare_test_loader(data, batch_size=128):
    """Create test data loader."""

    # Recreate test mask if not present
    if not hasattr(data['device'], 'test_mask'):
        print("\n⚠️  Test mask not found, recreating splits...")
        device_labels = data['device'].y
        num_devices = device_labels.shape[0]

        indices = torch.randperm(num_devices)
        train_size = int(0.6 * num_devices)
        val_size = int(0.2 * num_devices)
        test_idx = indices[train_size + val_size:]

        test_mask = torch.zeros(num_devices, dtype=torch.bool)
        test_mask[test_idx] = True
        data['device'].test_mask = test_mask

    test_loader = NeighborLoader(
        data,
        num_neighbors=[10, 5],
        batch_size=batch_size,
        input_nodes=('device', data['device'].test_mask),
        shuffle=False,
    )

    return test_loader


@torch.no_grad()
def evaluate_detailed(model, loader, device):
    """Detailed evaluation with predictions and probabilities."""
    model.eval()

    all_preds = []
    all_probs = []
    all_labels = []

    for batch in loader:
        batch = batch.to(device)

        # Forward pass
        out = model(batch.x_dict, batch.edge_index_dict)
        device_out = out['device']
        device_labels = batch['device'].y
        mask = batch['device'].test_mask

        # Get predictions
        pred = device_out[mask].argmax(dim=1)
        probs = F.softmax(device_out[mask], dim=1)[:, 1]  # Probability of malicious

        all_preds.extend(pred.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(device_labels[mask].cpu().numpy())

    return np.array(all_preds), np.array(all_probs), np.array(all_labels)


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap='Blues')
    plt.title('GNN Confusion Matrix')
    plt.colorbar()

    classes = ['Benign', 'Malicious']
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved confusion matrix to {save_path}")


def plot_roc_curve(y_true, y_probs, roc_auc, save_path):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, linewidth=2, label=f'GNN (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('GNN ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved ROC curve to {save_path}")


def plot_score_distribution(y_true, y_probs, save_path):
    """Plot score distribution for benign vs malicious."""
    benign_scores = y_probs[y_true == 0]
    malicious_scores = y_probs[y_true == 1]

    plt.figure(figsize=(10, 6))
    plt.hist(benign_scores, bins=50, alpha=0.6, label='Benign', density=True)
    plt.hist(malicious_scores, bins=50, alpha=0.6, label='Malicious', density=True)
    plt.xlabel('Malicious Probability')
    plt.ylabel('Density')
    plt.title('GNN Score Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved score distribution to {save_path}")


if __name__ == "__main__":
    Config.set_seeds()
    Config.ensure_output_dirs()

    print("\n" + "=" * 70)
    print("🔬 GNN EVALUATION")
    print("=" * 70)

    # Load model and data
    model, data = load_trained_gnn()

    # Create test loader
    print("\n📊 Preparing test data...")
    test_loader = prepare_test_loader(data)

    num_test = data['device'].test_mask.sum().item()
    test_labels = data['device'].y[data['device'].test_mask]
    num_benign = (test_labels == 0).sum().item()
    num_malicious = (test_labels == 1).sum().item()

    print(f"Test set size: {num_test:,} devices")
    print(f"  Benign: {num_benign:,}")
    print(f"  Malicious: {num_malicious:,}")

    # Evaluate
    print("\n🔍 Evaluating on test set...")
    preds, probs, labels = evaluate_detailed(model, test_loader, Config.DEVICE)

    # Metrics
    print("\n" + "=" * 70)
    print("📊 CLASSIFICATION REPORT")
    print("=" * 70)

    print(classification_report(
        labels, preds,
        target_names=['Benign', 'Malicious'],
        digits=4
    ))

    # ROC-AUC
    roc_auc = roc_auc_score(labels, probs)
    accuracy = (preds == labels).mean()

    print(f"\n🎯 SUMMARY:")
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   ROC-AUC: {roc_auc:.4f}")

    # Confusion matrix details
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n📊 DETAILED METRICS:")
    print(f"   True Positives: {tp}")
    print(f"   True Negatives: {tn}")
    print(f"   False Positives: {fp}")
    print(f"   False Negatives: {fn}")
    print(f"   Precision: {precision:.4f}")
    print(f"   Recall (Sensitivity): {recall:.4f}")
    print(f"   Specificity: {specificity:.4f}")
    print(f"   F1-Score: {f1:.4f}")

    # Save plots
    results_dir = os.path.join(Config.RESULTS_DIR, 'gnn')
    os.makedirs(results_dir, exist_ok=True)

    print("\n📈 Generating plots...")
    plot_confusion_matrix(
        labels, preds,
        os.path.join(results_dir, 'gnn_confusion_matrix.png')
    )

    plot_roc_curve(
        labels, probs, roc_auc,
        os.path.join(results_dir, 'gnn_roc_curve.png')
    )

    plot_score_distribution(
        labels, probs,
        os.path.join(results_dir, 'gnn_score_distribution.png')
    )

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)