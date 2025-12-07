"""
Evaluate trained Autoencoder models for anomaly detection.
Tests on both benign and malicious data to measure anomaly detection performance.
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    roc_curve
)

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from config import Config
from models.deep_learning.autoencoder_denoising import DenoisingAutoencoderModel
from models.deep_learning.autoencoder import AutoencoderModel


def load_test_data():
    """Load both benign and malicious data for testing."""
    print("\n" + "=" * 70)
    print("📂 LOADING TEST DATA")
    print("=" * 70)

    # Load full dataset
    df = pd.read_parquet(Config.ENGINEERED_DATA_PATH)
    print(f"✓ Loaded {len(df):,} total samples")

    # Load feature list used by autoencoder
    import joblib
    feature_list = joblib.load(Config.AUTOENCODER_FEATURE_LIST_PATH)
    print(f"✓ Using {len(feature_list)} features")

    # Separate benign and malicious
    benign_df = df[df[Config.TARGET_COL] == 'Benign'][feature_list].fillna(0)
    malicious_df = df[df[Config.TARGET_COL] == 'Malicious'][feature_list].fillna(0)

    print(f"\n📊 Test data split:")
    print(f"   Benign samples: {len(benign_df):,}")
    print(f"   Malicious samples: {len(malicious_df):,}")

    return benign_df, malicious_df, feature_list


def compute_reconstruction_errors(model, data, scaler, batch_size=1024):
    """Compute reconstruction errors for given data."""
    device = Config.DEVICE
    model.model.eval()

    # Scale data
    data_scaled = scaler.transform(data.values)
    data_tensor = torch.FloatTensor(data_scaled).to(device)

    reconstruction_errors = []

    with torch.no_grad():
        for i in range(0, len(data_tensor), batch_size):
            batch = data_tensor[i:i + batch_size]
            reconstructed = model.model(batch)

            # Compute MSE per sample
            errors = torch.mean((batch - reconstructed) ** 2, dim=1)
            reconstruction_errors.extend(errors.cpu().numpy())

    return np.array(reconstruction_errors)


def find_optimal_threshold(benign_errors, malicious_errors, percentiles=[65, 70, 75, 80, 85, 90, 95, 99]):
    """Find optimal threshold based on different percentiles."""
    print("\n" + "=" * 70)
    print("🔍 FINDING OPTIMAL THRESHOLD")
    print("=" * 70)

    results = {}

    for percentile in percentiles:
        threshold = np.percentile(benign_errors, percentile)

        # Classify: error > threshold = anomaly (malicious)
        benign_pred = (benign_errors > threshold).astype(int)  # Should be 0 (normal)
        malicious_pred = (malicious_errors > threshold).astype(int)  # Should be 1 (anomaly)

        # Metrics
        benign_correct = (benign_pred == 0).sum()
        malicious_correct = (malicious_pred == 1).sum()

        specificity = benign_correct / len(benign_errors)  # True Negative Rate
        sensitivity = malicious_correct / len(malicious_errors)  # True Positive Rate (Recall)

        false_positives = (benign_pred == 1).sum()
        false_negatives = (malicious_pred == 0).sum()

        accuracy = (benign_correct + malicious_correct) / (len(benign_errors) + len(malicious_errors))

        results[percentile] = {
            'threshold': threshold,
            'accuracy': accuracy,
            'sensitivity': sensitivity,
            'specificity': specificity,
            'false_positives': false_positives,
            'false_negatives': false_negatives
        }

        print(f"\n📊 Percentile {percentile}:")
        print(f"   Threshold: {threshold:.6f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Sensitivity (Recall): {sensitivity:.4f} - Catches {sensitivity * 100:.1f}% of malicious")
        print(f"   Specificity: {specificity:.4f} - Correct on {specificity * 100:.1f}% of benign")
        print(f"   False Positives: {false_positives:,} ({false_positives / len(benign_errors) * 100:.2f}%)")
        print(f"   False Negatives: {false_negatives:,} ({false_negatives / len(malicious_errors) * 100:.2f}%)")

    # Recommend best threshold (balance between sensitivity and specificity)
    best_percentile = max(results.keys(), key=lambda p: results[p]['accuracy'])
    print(f"\n✅ RECOMMENDED: Use {best_percentile}th percentile threshold")
    print(f"   Best accuracy: {results[best_percentile]['accuracy']:.4f}")

    return results, best_percentile


def plot_error_distributions(benign_errors, malicious_errors, threshold, save_path):
    """Plot reconstruction error distributions."""
    plt.figure(figsize=(12, 6))

    # Plot distributions
    plt.subplot(1, 2, 1)
    plt.hist(benign_errors, bins=50, alpha=0.7, label='Benign', color='green', density=True)
    plt.hist(malicious_errors, bins=50, alpha=0.7, label='Malicious', color='red', density=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.6f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Density')
    plt.title('Reconstruction Error Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot log scale
    plt.subplot(1, 2, 2)
    plt.hist(benign_errors, bins=50, alpha=0.7, label='Benign', color='green', density=True, log=True)
    plt.hist(malicious_errors, bins=50, alpha=0.7, label='Malicious', color='red', density=True, log=True)
    plt.axvline(threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold: {threshold:.6f}')
    plt.xlabel('Reconstruction Error')
    plt.ylabel('Log Density')
    plt.title('Reconstruction Error Distribution (Log Scale)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved error distribution plot: {save_path}")


def plot_roc_curve(benign_errors, malicious_errors, save_path):
    """Plot ROC curve."""
    # Create labels and scores
    y_true = np.concatenate([np.zeros(len(benign_errors)), np.ones(len(malicious_errors))])
    y_scores = np.concatenate([benign_errors, malicious_errors])

    # Compute ROC curve
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = roc_auc_score(y_true, y_scores)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve - Anomaly Detection')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Saved ROC curve: {save_path}")

    return roc_auc


def evaluate_model(model_path, model_type='autoencoder'):
    """Evaluate a trained autoencoder model."""
    print("\n" + "=" * 70)
    print(f"🎯 EVALUATING {model_type.upper()}")
    print("=" * 70)
    print(f"Model: {model_path}")

    if model_type == 'autoencoder':
        model = AutoencoderModel.load_from_checkpoint(model_path)  # ✅ Use class method
    else:  # denoising
        model = DenoisingAutoencoderModel.load_from_checkpoint(model_path)  # ✅ Use class method

    # Load test data
    benign_df, malicious_df, feature_list = load_test_data()

    # Compute reconstruction errors
    print("\n⏳ Computing reconstruction errors...")
    benign_errors = compute_reconstruction_errors(model, benign_df, model.scaler)
    malicious_errors = compute_reconstruction_errors(model, malicious_df, model.scaler)
    print(f"✓ Computed errors for {len(benign_errors):,} benign + {len(malicious_errors):,} malicious samples")

    print(f"\n📊 Error statistics:")
    print(f"   Benign - Mean: {benign_errors.mean():.6f}, Std: {benign_errors.std():.6f}")
    print(f"   Malicious - Mean: {malicious_errors.mean():.6f}, Std: {malicious_errors.std():.6f}")

    # Find optimal threshold
    results, best_percentile = find_optimal_threshold(benign_errors, malicious_errors)
    best_threshold = results[best_percentile]['threshold']

    # Plot error distributions
    model_name = model_path.replace('.pth', '')
    plot_error_distributions(
        benign_errors,
        malicious_errors,
        best_threshold,
        os.path.join(Config.AUTOENCODER_RESULTS_DIR, f'{model_name}_error_distribution.png')
    )

    # Plot ROC curve
    roc_auc = plot_roc_curve(
        benign_errors,
        malicious_errors,
        os.path.join(Config.AUTOENCODER_RESULTS_DIR, f'{model_name}_roc_curve.png')
    )

    # Final classification report
    print("\n" + "=" * 70)
    print("📊 FINAL CLASSIFICATION REPORT (Using Best Threshold)")
    print("=" * 70)

    y_true = np.concatenate([np.zeros(len(benign_errors)), np.ones(len(malicious_errors))])
    y_pred = np.concatenate([
        (benign_errors > best_threshold).astype(int),
        (malicious_errors > best_threshold).astype(int)
    ])

    print(classification_report(y_true, y_pred, target_names=['Benign', 'Malicious'], zero_division=0))

    print(f"\n🎯 SUMMARY:")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   Best Threshold: {best_threshold:.6f} ({best_percentile}th percentile)")
    print(f"   Overall Accuracy: {results[best_percentile]['accuracy']:.4f}")
    print(f"   Anomaly Detection Rate: {results[best_percentile]['sensitivity']:.4f}")

    return {
        'roc_auc': roc_auc,
        'best_threshold': best_threshold,
        'best_percentile': best_percentile,
        'results': results
    }


if __name__ == "__main__":
    Config.set_seeds()
    Config.ensure_output_dirs()

    print("=" * 70)
    print("🔬 AUTOENCODER ANOMALY DETECTION EVALUATION")
    print("=" * 70)

    # Evaluate regular autoencoder
    print("\n" + "=" * 70)
    print("1️⃣  REGULAR AUTOENCODER")
    print("=" * 70)
    try:
        ae_results = evaluate_model('autoencoder_final.pth', 'autoencoder')
    except FileNotFoundError:
        print("❌ Regular autoencoder model not found. Train it first!")

    # Evaluate denoising autoencoder
    print("\n" + "=" * 70)
    print("2️⃣  DENOISING AUTOENCODER")
    print("=" * 70)
    try:
        dae_results = evaluate_model('denoising_autoencoder_final.pth', 'denoising')
    except FileNotFoundError:
        print("❌ Denoising autoencoder model not found. Train it first!")

    print("\n" + "=" * 70)
    print("✅ EVALUATION COMPLETE")
    print("=" * 70)
