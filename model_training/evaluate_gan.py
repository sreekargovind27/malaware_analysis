"""
Evaluate WGAN-GP anomaly detector on full mixed test set.

We:
  - Load trained WGAN-GP checkpoint (generator+critic+scaler+feature_list+params)
  - Load the SAME engineered dataset we used for training
  - Compute anomaly scores: score = -critic(x)
      higher score => more suspicious
  - Sweep thresholds at given percentiles (65,70,...)
  - Print accuracy, recall (malicious), specificity (benign), FP/FN counts
  - Generate classification report + ROC-AUC
  - Save score hist and ROC curve
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from tqdm import tqdm

from config import Config
from models.data_loader import load_engineered_data
from models.deep_learning.gan_wgan_gp import WGAN_GP_Model


def prepare_test_data(feature_list, scaler):
    """
    Build the test tensor for scoring.

    Steps:
    - Load full engineered data (already built in stage2)
    - Select ONLY the columns the GAN was trained on, in that exact order
    - Scale using the saved scaler from training
    - Return torch tensor on the same device as model, plus y_true (0/1)
    """
    df = load_engineered_data()

    # labels first (string version for reporting)
    labels_str = df[Config.TARGET_COL].values  # "Benign" / "Malicious"

    # numeric feature matrix using exact training feature order
    X = df[feature_list].fillna(0).values

    # apply same scaler used in training
    X_scaled = scaler.transform(X)
    X_scaled = np.nan_to_num(X_scaled)

    # push to device
    X_tensor = torch.FloatTensor(X_scaled).to(Config.DEVICE)

    # y_true as 0/1
    y_true = np.array([1 if lab == 'Malicious' else 0 for lab in labels_str], dtype=np.int64)

    return X_tensor, y_true, labels_str


def compute_anomaly_scores(model, X_tensor, batch_size=4096):
    """
    For each sample x: score = -critic(x)
    Higher score => more anomalous => more likely malicious.
    """
    model.critic.eval()
    scores = []

    with torch.no_grad():
        for i in tqdm(range(0, X_tensor.size(0), batch_size), desc="Scoring with WGAN-GP critic"):
            batch = X_tensor[i:i + batch_size]
            critic_out = model.critic(batch).view(-1)
            # anomaly score: negative critic score
            batch_scores = (-critic_out).detach().cpu().numpy()
            scores.extend(batch_scores)

    scores = np.array(scores)
    return scores


def evaluate_thresholds(scores, y_true, benign_mask, percentiles_list):
    """
    Similar to evaluate_autoencoder style.

    We treat benign distribution as "normal".
    We take percentile thresholds of benign scores.
    Above threshold => anomaly => predict malicious (1).
    """
    benign_scores = scores[benign_mask]

    for p in percentiles_list:
        thr = np.percentile(benign_scores, p)

        y_pred = (scores > thr).astype(int)  # 1 = malicious (anomaly)

        # confusion pieces
        TP = np.sum((y_pred == 1) & (y_true == 1))
        TN = np.sum((y_pred == 0) & (y_true == 0))
        FP = np.sum((y_pred == 1) & (y_true == 0))
        FN = np.sum((y_pred == 0) & (y_true == 1))

        # metrics
        accuracy = (TP + TN) / len(y_true) if len(y_true) > 0 else 0.0
        recall_mal = TP / (TP + FN) if (TP + FN) > 0 else 0.0  # sensitivity / malicious recall
        specificity_benign = TN / (TN + FP) if (TN + FP) > 0 else 0.0

        print(f"\n📊 Percentile {p}:")
        print(f"   Threshold: {thr:.6f}")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Sensitivity (Recall): {recall_mal:.4f} - Catches {recall_mal * 100:.1f}% of malicious")
        print(f"   Specificity: {specificity_benign:.4f} - Correct on {specificity_benign * 100:.1f}% of benign")
        print(f"   False Positives: {FP} ({(FP / np.sum(y_true == 0)) * 100:.2f}%)")
        print(f"   False Negatives: {FN} ({(FN / np.sum(y_true == 1)) * 100:.2f}%)")

    # pick "best" percentile as the first one in list (like AE code does)
    best_p = percentiles_list[0]
    best_thr = np.percentile(benign_scores, best_p)
    y_pred_best = (scores > best_thr).astype(int)
    return best_thr, y_pred_best


def save_plots(scores, y_true, roc_auc, thr, out_dir):
    os.makedirs(out_dir, exist_ok=True)

    # Score distribution plot
    plt.figure(figsize=(10, 6))
    plt.hist(scores[y_true == 0], bins=100, alpha=0.6, label='Benign', density=True)
    plt.hist(scores[y_true == 1], bins=100, alpha=0.6, label='Malicious', density=True)
    plt.axvline(thr, linestyle='--', linewidth=2, label=f'Threshold {thr:.4f}')
    plt.xlabel('Anomaly Score (-Critic(x))')
    plt.ylabel('Density')
    plt.title('WGAN-GP Critic Score Distribution')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'wgan_gp_score_distribution.png'))
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_true, scores)
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'WGAN-GP (AUC={roc_auc:.4f})')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('WGAN-GP ROC Curve')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'wgan_gp_roc_curve.png'))
    plt.close()


if __name__ == "__main__":

    Config.set_seeds()
    Config.ensure_output_dirs()

    # Check if we're evaluating multiple models (grid search mode)
    EVALUATE_GRID_SEARCH = getattr(Config, 'GAN_TEST_INJECTION_RATES', False)

    if EVALUATE_GRID_SEARCH:
        # GRID SEARCH EVALUATION MODE
        print("\n" + "=" * 70)
        print("🔬 WGAN-GP GRID SEARCH EVALUATION")
        print("=" * 70)

        test_rates = [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
        results = {}

        for rate in test_rates:
            model_file = f'gan_wgan_gp_rate_{int(rate * 100)}.pth'

            print(f"\n{'=' * 70}")
            print(f"📊 Evaluating injection rate: {rate} ({model_file})")
            print("=" * 70)

            try:
                # Load model
                model, _ = WGAN_GP_Model.load_from_checkpoint(model_file)
                scaler = model.scaler
                feature_list = model.feature_list

                # Prepare test data
                X_tensor, y_true, _ = prepare_test_data(feature_list, scaler)

                # Score samples
                scores = compute_anomaly_scores(model, X_tensor, Config.AUTOENCODER_BATCH_SIZE)

                # Get ROC-AUC (main metric)
                roc_auc = roc_auc_score(y_true, scores)

                # Quick threshold eval (just 75th percentile for speed)
                benign_mask = (y_true == 0)
                thr = np.percentile(scores[benign_mask], 75)
                y_pred = (scores > thr).astype(int)
                acc = (y_pred == y_true).mean()

                results[rate] = {
                    'auc': roc_auc,
                    'accuracy': acc,
                    'threshold': thr
                }

                print(f"   ROC-AUC: {roc_auc:.4f}")
                print(f"   Accuracy: {acc:.4f}")

                # Save individual plots
                save_plots(scores, y_true, roc_auc, thr,
                           os.path.join(Config.GAN_RESULTS_DIR, f'rate_{int(rate * 100)}'))

            except FileNotFoundError:
                print(f"   ⚠️  Model file not found: {model_file}")
                continue

        # Print comparison
        print("\n" + "=" * 70)
        print("📊 INJECTION RATE COMPARISON")
        print("=" * 70)
        for rate in sorted(results.keys(), key=lambda r: results[r]['auc'], reverse=True):
            print(f"  {rate:.2f}: AUC={results[rate]['auc']:.4f}, Acc={results[rate]['accuracy']:.4f}")

        best_rate = max(results.keys(), key=lambda r: results[r]['auc'])
        print(f"\n🏆 Best injection rate: {best_rate} (AUC={results[best_rate]['auc']:.4f})")

    else:
        # NORMAL SINGLE MODEL EVALUATION
        print("\n" + "=" * 70)
        print("🔬 WGAN-GP ANOMALY DETECTION EVALUATION")
        print("=" * 70)

        # Load trained WGAN-GP model
        model, saved_threshold = WGAN_GP_Model.load_from_checkpoint('gan_wgan_gp_final.pth')
        scaler = model.scaler
        feature_list = model.feature_list

        if scaler is None:
            raise RuntimeError("Checkpoint did not contain scaler.")
        if feature_list is None:
            raise RuntimeError("Checkpoint did not contain feature_list.")

        # Build test data
        X_tensor, y_true, label_strings = prepare_test_data(feature_list, scaler)

        print("\n📊 Test data split:")
        print(f"   Benign samples: {np.sum(y_true == 0):,}")
        print(f"   Malicious samples: {np.sum(y_true == 1):,}")

        # Score all samples
        scores = compute_anomaly_scores(model, X_tensor, Config.AUTOENCODER_BATCH_SIZE)

        # Threshold sweep
        percentiles_to_try = [65, 70, 75, 80, 85, 90, 95, 99]

        print("\n" + "=" * 70)
        print("🔍 FINDING OPTIMAL THRESHOLD (WGAN-GP)")
        print("=" * 70)

        benign_mask = (y_true == 0)
        best_thr, y_pred_best = evaluate_thresholds(scores, y_true, benign_mask, percentiles_to_try)

        # Final metrics
        print("\n" + "=" * 70)
        print("📊 FINAL CLASSIFICATION REPORT (Using Best Threshold)")
        print("=" * 70)

        report = classification_report(y_true, y_pred_best,
                                       target_names=['Benign', 'Malicious'], digits=4)
        print(report)

        # ROC AUC
        roc_auc = roc_auc_score(y_true, scores)
        print(f"\n🎯 SUMMARY:")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   Best Threshold: {best_thr:.6f}")
        print(f"   Overall Accuracy: {(y_pred_best == y_true).mean():.4f}")
        print(f"   Anomaly Detection Recall: {np.sum((y_pred_best == 1) & (y_true == 1)) / np.sum(y_true == 1):.4f}")

        # Save plots
        save_plots(scores, y_true, roc_auc, best_thr, Config.GAN_RESULTS_DIR)

        print("\n======================================================================")
        print("✅ EVALUATION COMPLETE")
        print("======================================================================")
