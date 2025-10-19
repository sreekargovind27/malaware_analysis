"""
Denoising Autoencoder for Anomaly Detection
Variation: Adds noise during training for better robustness
Uses Optuna-optimized hyperparameters for best performance
Contributor: Nidhi Rajani
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


class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder with noise injection and optimized architecture"""

    def __init__(self, input_dim, noise_factor=0.15):
        super(DenoisingAutoencoder, self).__init__()
        self.noise_factor = noise_factor

        # USE OPTUNA-OPTIMIZED ARCHITECTURE
        # Best params: [116, 45, 25, 5]
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 116),
            nn.ReLU(),
            nn.Dropout(0.01),  # Minimal dropout (from Optuna)
            nn.Linear(116, 45),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(45, 25),
            nn.ReLU(),
            nn.Linear(25, 5)  # Tight bottleneck!
        )

        # Decoder (mirror of encoder)
        self.decoder = nn.Sequential(
            nn.Linear(5, 25),
            nn.ReLU(),
            nn.Linear(25, 45),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(45, 116),
            nn.ReLU(),
            nn.Dropout(0.01),
            nn.Linear(116, input_dim)
        )

    def add_noise(self, x):
        """Add Gaussian noise to input"""
        noise = torch.randn_like(x) * self.noise_factor
        return x + noise

    def forward(self, x, add_noise=True):
        if add_noise and self.training:
            x = self.add_noise(x)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class DenoisingAutoencoderModel:
    """Denoising Autoencoder wrapper with optimized hyperparameters"""

    def __init__(self, input_dim, noise_factor=0.15):
        self.device = Config.DEVICE
        self.model = DenoisingAutoencoder(input_dim, noise_factor).to(self.device)
        self.criterion = nn.MSELoss()

        # USE OPTUNA-OPTIMIZED LEARNING RATE!
        optimal_lr = 0.008123  # From Optuna optimization
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optimal_lr)

        self.threshold = None
        self.scaler = None

        print(f"Denoising Autoencoder initialized on {self.device}")
        print(f"  Input dim: {input_dim}")
        print(f"  Latent dim: 5 (optimized)")
        print(f"  Architecture: [116 → 45 → 25 → 5] (Optuna-optimized)")
        print(f"  Learning rate: {optimal_lr} (optimized)")
        print(f"  Noise factor: {noise_factor}")
        print(f"  Dropout: 0.01 (minimal)")

    def train(self, train_loader, val_loader, scaler):
        """Train denoising autoencoder"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING DENOISING AUTOENCODER")
        print("=" * 70)

        self.scaler = scaler
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        train_losses, val_losses = [], []

        print(f"\n📊 Training configuration:")
        print(f"   Epochs: {Config.AUTOENCODER_EPOCHS}")
        print(f"   Batch size: {Config.AUTOENCODER_BATCH_SIZE}")
        print(f"   Noise injection: {self.model.noise_factor}")
        print(f"   Threshold percentile: {Config.ANOMALY_THRESHOLD_PERCENTILE}th")

        print(f"\n⏳ Training for up to {Config.AUTOENCODER_EPOCHS} epochs...")

        for epoch in tqdm(range(Config.AUTOENCODER_EPOCHS), desc="Training Denoising AE"):
            # Training with noise
            self.model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device)
                self.optimizer.zero_grad()

                # Forward pass with noise injection
                reconstructed = self.model(batch, add_noise=True)
                loss = self.criterion(reconstructed, batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)
            train_losses.append(train_loss)

            # Validation without noise
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    reconstructed = self.model(batch, add_noise=False)
                    loss = self.criterion(reconstructed, batch)
                    val_loss += loss.item()

            val_loss /= len(val_loader) if len(val_loader) > 0 else 1
            val_losses.append(val_loss)

            # Print progress
            if (epoch + 1) % 10 == 0:
                print(
                    f"\n  Epoch [{epoch + 1}/{Config.AUTOENCODER_EPOCHS}] Train: {train_loss:.6f}, Val: {val_loss:.6f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model('best_denoising_autoencoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        self.load_model('best_denoising_autoencoder.pth')

        # Set threshold
        print(f"\n⏳ Setting anomaly detection threshold...")
        self._set_threshold(val_loader)
        print(f"   Using {Config.ANOMALY_THRESHOLD_PERCENTILE}th percentile: {self.threshold:.6f}")

        # Plot
        self._plot_training_history(train_losses, val_losses)

        print(f"\n" + "=" * 70)
        print(f"✅ TRAINING COMPLETE - DENOISING AUTOENCODER")
        print(f"=" * 70)
        print(f"   Best validation loss: {best_val_loss:.6f}")
        print(f"   Anomaly threshold: {self.threshold:.6f}")
        print(f"   Total epochs: {len(train_losses)}")
        print("=" * 70)

    def _set_threshold(self, val_loader):
        """Set anomaly threshold"""
        self.model.eval()
        errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                reconstructed = self.model(batch, add_noise=False)
                error = torch.mean((batch - reconstructed) ** 2, dim=1)
                errors.extend(error.cpu().numpy())

        if errors:
            self.threshold = np.percentile(errors, Config.ANOMALY_THRESHOLD_PERCENTILE)
        else:
            self.threshold = float('inf')

    def detect_anomalies(self, data_loader):
        """Detect anomalies"""
        print("\n" + "=" * 70)
        print("🔍 DETECTING ANOMALIES (Denoising Variant)")
        print("=" * 70)

        self.model.eval()
        all_errors, all_predictions = [], []

        print(f"\n⏳ Computing reconstruction errors...")
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing batches"):
                if isinstance(batch, tuple):
                    batch = batch[0]
                batch = batch.to(self.device)
                reconstructed = self.model(batch, add_noise=False)
                errors = torch.mean((batch - reconstructed) ** 2, dim=1)
                all_errors.extend(errors.cpu().numpy())
                predictions = (errors > self.threshold).cpu().numpy()
                all_predictions.extend(predictions)

        all_errors = np.array(all_errors)
        all_predictions = np.array(all_predictions)

        anomaly_count = all_predictions.sum()
        anomaly_pct = (anomaly_count / len(all_predictions) * 100) if len(all_predictions) > 0 else 0

        print(f"\n📊 Detection results:")
        print(f"   Total samples: {len(all_predictions):,}")
        print(f"   Anomalies detected: {anomaly_count:,} ({anomaly_pct:.2f}%)")
        print(f"   Mean error: {np.mean(all_errors):.6f}")
        print(f"   Median error: {np.median(all_errors):.6f}")
        print(f"   Threshold: {self.threshold:.6f}")

        return np.array(all_predictions), np.array(all_errors)

    def _plot_training_history(self, train_losses, val_losses):
        """Plot training history"""
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss (with noise)', linewidth=2, alpha=0.8)
        plt.plot(val_losses, label='Validation Loss (no noise)', linewidth=2, alpha=0.8)
        plt.xlabel('Epoch', fontsize=12)
        plt.ylabel('Loss (MSE)', fontsize=12)
        plt.title('Denoising Autoencoder Training History', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'denoising_autoencoder_training.png'), dpi=300)
        plt.close()
        print(f"   ✓ Training plot saved")

    def save_model(self, filename):
        """Save model"""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        torch.save({
            'model_state': self.model.state_dict(),
            'threshold': self.threshold,
            'scaler': self.scaler
        }, filepath)

    def load_model(self, filename):
        """Load model"""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if os.path.exists(filepath):
            # =================================================================
            # ADD weights_only=False HERE
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
            # =================================================================
            self.model.load_state_dict(checkpoint['model_state'])
            self.threshold = checkpoint.get('threshold')
            self.scaler = checkpoint.get('scaler')


if __name__ == "__main__":
    Config.set_seeds()

    print("=" * 70)
    print("🔮 DENOISING AUTOENCODER VARIATION")
    print("   Contributor: Nidhi Rajani")
    print("=" * 70)

    overall_start = time.time()

    # Load data
    train_loader, val_loader, scaler, final_feature_list = get_data_for_autoencoder()

    if final_feature_list and len(train_loader.dataset) > 0:
        input_dim = len(final_feature_list)

        # Train denoising variant with optimized parameters
        denoising_ae = DenoisingAutoencoderModel(input_dim, noise_factor=0.15)
        denoising_ae.train(train_loader, val_loader, scaler)

        # Save model
        denoising_ae.save_model('denoising_autoencoder_final.pth')

        # Test on validation
        if len(val_loader.dataset) > 0:
            predictions, errors = denoising_ae.detect_anomalies(val_loader)
            print(f"\n✅ Validation: {predictions.sum():,} / {len(predictions):,} flagged")

        # Test on malicious
        print("\n" + "=" * 70)
        print("🧪 TESTING ON MALICIOUS DATA")
        print("=" * 70)

        df_full = load_engineered_data()
        malicious_df = df_full[df_full[Config.TARGET_COL] == 'Malicious']

        if len(malicious_df) > 0:
            print(f"\n⏳ Preparing {len(malicious_df):,} malicious samples...")

            features_used = joblib.load(Config.AUTOENCODER_FEATURE_LIST_PATH)
            X_malicious = malicious_df[features_used].fillna(0)
            X_malicious_scaled = denoising_ae.scaler.transform(X_malicious.values)

            malicious_dataset = IoTDataset(X_malicious_scaled)
            malicious_loader = DataLoader(malicious_dataset, batch_size=4096, shuffle=False)

            predictions, errors = denoising_ae.detect_anomalies(malicious_loader)

            print(f"\n📊 Results on Malicious Data:")
            print(f"   Detected as anomalies: {predictions.sum():,} / {len(predictions):,}")
            print(f"   Detection rate: {predictions.sum() / len(predictions) * 100:.2f}%")
            print(f"   Mean error: {np.mean(errors):.6f}")
            print(f"   This should be HIGH (>90%) for good anomaly detection!")

    total_time = time.time() - overall_start

    print("\n" + "=" * 70)
    print("🎉 DENOISING AUTOENCODER COMPLETE")
    print("=" * 70)
    print(f"⏱️  Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print("\n✅ Denoising Autoencoder variation complete!")