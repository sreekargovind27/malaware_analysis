"""
Autoencoder for Anomaly Detection
Trains on benign data only, detects anomalies by reconstruction error.
UPDATED: Enhanced timing display and progress tracking.
"""

import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from data_loader import get_data_for_autoencoder, load_engineered_data, IoTDataset
from torch.utils.data import DataLoader


class Autoencoder(nn.Module):
    """Simple and stable Autoencoder architecture"""

    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, Config.AUTOENCODER_LATENT_DIM)
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(Config.AUTOENCODER_LATENT_DIM, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class AutoencoderModel:
    """Wrapper class for training and inference"""

    def __init__(self, input_dim):
        self.device = Config.DEVICE
        self.model = Autoencoder(input_dim).to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=Config.AUTOENCODER_LR)
        self.threshold = None
        self.scaler = None
        print(f"Autoencoder initialized on {self.device}")
        print(f"  Input dim: {input_dim}")
        print(f"  Latent dim: {Config.AUTOENCODER_LATENT_DIM}")
        print(
            f"  Architecture: {input_dim} → 64 → 32 → 16 → {Config.AUTOENCODER_LATENT_DIM} → 16 → 32 → 64 → {input_dim}")

    def train(self, train_loader, val_loader, scaler):
        """Train autoencoder on benign data"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING AUTOENCODER")
        print("=" * 70)

        overall_start = time.time()

        self.scaler = scaler
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        train_losses, val_losses = [], []

        print(f"\n📊 Training configuration:")
        print(f"   Epochs: {Config.AUTOENCODER_EPOCHS}")
        print(f"   Batch size: {Config.AUTOENCODER_BATCH_SIZE}")
        print(f"   Learning rate: {Config.AUTOENCODER_LR}")
        print(f"   Early stopping patience: {patience}")
        print(f"   Device: {self.device}")

        print(f"\n⏳ Training for up to {Config.AUTOENCODER_EPOCHS} epochs...")
        t_train_start = time.time()

        for epoch in tqdm(range(Config.AUTOENCODER_EPOCHS), desc="Training Autoencoder"):
            # Training
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = self.criterion(reconstructed, batch)

                # Check for NaN
                if torch.isnan(loss):
                    print(f"\n❌ ERROR: NaN loss at epoch {epoch + 1}. Stopping.")
                    return

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    reconstructed = self.model(batch)
                    loss = self.criterion(reconstructed, batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader) if len(val_loader) > 0 else 1
            val_losses.append(val_loss)

            # Check for NaN
            if np.isnan(val_loss):
                print(f"\n❌ ERROR: NaN validation loss at epoch {epoch + 1}.")
                break

            # Print progress every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(
                    f"\n  Epoch [{epoch + 1}/{Config.AUTOENCODER_EPOCHS}] Train: {train_loss:.6f}, Val: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_autoencoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        self.load_model('best_autoencoder.pth')

        t_train = time.time() - t_train_start

        # Set threshold
        print(f"\n⏳ Setting anomaly detection threshold...")
        t_threshold_start = time.time()
        if len(val_loader.dataset) > 0:
            self._set_threshold(val_loader)
        else:
            print("   ⚠️  Empty validation set, cannot set threshold.")
            self.threshold = float('inf')
        t_threshold = time.time() - t_threshold_start
        print(f"✓ Threshold set ({t_threshold:.2f}s)")

        # Plot training history
        print(f"\n⏳ Generating training plot...")
        t_plot_start = time.time()
        self._plot_training_history(train_losses, val_losses)
        t_plot = time.time() - t_plot_start
        print(f"✓ Plot saved ({t_plot:.2f}s)")

        total_time = time.time() - overall_start

        print(f"\n" + "=" * 70)
        print(f"✅ TRAINING COMPLETE")
        print(f"=" * 70)
        print(f"   Best validation loss: {best_val_loss:.6f}")
        print(f"   Anomaly threshold: {self.threshold:.6f}" if self.threshold is not None and self.threshold != float(
            'inf') else "   Anomaly threshold: N/A")
        print(f"   Total epochs trained: {len(train_losses)}")
        print(f"⏱️  Training time: {t_train:.2f}s")
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")
        print("=" * 70)

    def _set_threshold(self, val_loader):
        """Set anomaly threshold based on validation set"""
        self.model.eval()
        errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                reconstructed = self.model(batch)
                error = torch.mean((batch - reconstructed) ** 2, dim=1)
                errors.extend(error.cpu().numpy())

        if errors:
            self.threshold = np.percentile(errors, Config.ANOMALY_THRESHOLD_PERCENTILE)
            print(f"   Using {Config.ANOMALY_THRESHOLD_PERCENTILE}th percentile: {self.threshold:.6f}")
        else:
            self.threshold = float('inf')

    def detect_anomalies(self, data_loader):
        """Detect anomalies in data"""
        print("\n" + "=" * 70)
        print("🔍 DETECTING ANOMALIES")
        print("=" * 70)

        t_start = time.time()

        self.model.eval()
        all_errors, all_predictions = [], []

        if self.threshold is None or self.threshold == float('inf'):
            print("⚠️  WARNING: Threshold not set, cannot detect anomalies.")
            return np.array([]), np.array([])

        print(f"\n⏳ Computing reconstruction errors...")
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing batches"):
                if isinstance(batch, tuple):
                    batch = batch[0]
                batch = batch.to(self.device)

                reconstructed = self.model(batch)
                errors = torch.mean((batch - reconstructed) ** 2, dim=1)

                all_errors.extend(errors.cpu().numpy())
                predictions = (errors > self.threshold).cpu().numpy()
                all_predictions.extend(predictions)

        all_errors = np.array(all_errors)
        all_predictions = np.array(all_predictions)

        t_elapsed = time.time() - t_start

        # Statistics
        anomaly_count = all_predictions.sum()
        anomaly_pct = (anomaly_count / len(all_predictions) * 100) if len(all_predictions) > 0 else 0

        print(f"\n📊 Detection results:")
        print(f"   Total samples: {len(all_predictions):,}")
        print(f"   Anomalies detected: {anomaly_count:,} ({anomaly_pct:.2f}%)")
        print(f"   Mean error: {np.mean(all_errors):.6f}")
        print(f"   Median error: {np.median(all_errors):.6f}")
        print(f"   Max error: {np.max(all_errors):.6f}")
        print(f"   Threshold: {self.threshold:.6f}")
        print(f"⏱️  Detection time: {t_elapsed:.2f}s")
        if t_elapsed > 0:
            print(f"   Throughput: {len(all_predictions) / t_elapsed:.0f} samples/sec")

        return all_predictions, all_errors

    def _plot_training_history(self, train_losses, val_losses):
        """Plot training and validation loss"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss', linewidth=2)
        plt.plot(val_losses, label='Validation Loss', linewidth=2)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Autoencoder Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'autoencoder_training.png'), dpi=300)
        plt.close()

    def save_model(self, filename):
        """Save model checkpoint"""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'threshold': self.threshold,
            'scaler': self.scaler
        }, filepath)

    def load_model(self, filename):
        """Load model checkpoint"""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"❌ Model file {filepath} not found")
            return

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.threshold = checkpoint.get('threshold')
        self.scaler = checkpoint.get('scaler')


if __name__ == "__main__":
    Config.set_seeds()

    print("=" * 70)
    print("🔮 AUTOENCODER FOR ANOMALY DETECTION")
    print("=" * 70)

    overall_start = time.time()

    # Load data
    train_loader, val_loader, scaler, final_feature_list = get_data_for_autoencoder()

    if not final_feature_list or len(train_loader.dataset) == 0:
        print("\n❌ ERROR: Empty training dataset. Cannot train.")
    else:
        input_dim = len(final_feature_list)

        # Train
        autoencoder = AutoencoderModel(input_dim)
        autoencoder.train(train_loader, val_loader, scaler)

        # Evaluate on validation
        if len(val_loader.dataset) > 0:
            predictions, errors = autoencoder.detect_anomalies(val_loader)

            if len(predictions) > 0:
                print(f"\n" + "=" * 70)
                print(f"✅ VALIDATION SUMMARY")
                print(f"=" * 70)
                print(f"   Anomalies: {predictions.sum():,} / {len(predictions):,}")
                print(f"   Detection rate: {predictions.sum() / len(predictions) * 100:.2f}%")
        else:
            print("\n⚠️  Validation set empty, skipping evaluation.")

        # Save final model
        print(f"\n💾 Saving final model...")
        autoencoder.save_model('autoencoder_final.pth')

        total_time = time.time() - overall_start

        print("\n" + "=" * 70)
        print("🎉 AUTOENCODER TRAINING COMPLETE")
        print("=" * 70)
        print(f"⏱️  Total pipeline time: {total_time:.2f}s ({total_time / 60:.1f} min)")
        print("\n✅ Autoencoder trained and saved successfully!")

    # After training, test on malicious data
    print("\n" + "=" * 70)
    print("🧪 TESTING ON MALICIOUS DATA")
    print("=" * 70)

    # Load malicious samples
    df_full = load_engineered_data()
    malicious_df = df_full[df_full[Config.TARGET_COL] == 'Malicious']

    if len(malicious_df) > 0:
        print(f"\n⏳ Preparing {len(malicious_df):,} malicious samples...")

        # Get same features used in training
        features_used = joblib.load(Config.AUTOENCODER_FEATURE_LIST_PATH)
        X_malicious = malicious_df[features_used].fillna(0)

        # Scale using training scaler
        X_malicious_scaled = autoencoder.scaler.transform(X_malicious.values)

        # ===== THIS IS THE ONLY FIX =====
        # Use our custom IoTDataset to ensure the DataLoader yields a tensor directly, not a list
        malicious_dataset = IoTDataset(X_malicious_scaled)
        malicious_loader = DataLoader(malicious_dataset, batch_size=4096, shuffle=False)
        # ================================

        # Detect anomalies
        mal_predictions, mal_errors = autoencoder.detect_anomalies(malicious_loader)

        if len(mal_predictions) > 0:
            print(f"\n📊 Results on Malicious Data:")
            print(f"   Detected as anomalies: {mal_predictions.sum():,} / {len(mal_predictions):,}")
            print(f"   Detection rate: {mal_predictions.sum() / len(mal_predictions) * 100:.2f}%")
            print(f"   Mean error: {np.mean(mal_errors):.6f}")
            print(f"   This should be HIGH (>80%) for good anomaly detection!")
    else:
        print("\n⚠️  No malicious data found")