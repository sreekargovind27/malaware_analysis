"""
Denoising Autoencoder for Anomaly Detection
Variation: Adds noise during training for better robustness
Uses Optuna-optimized hyperparameters for best performance
Contributor: Nidhi Rajani
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from data_loader import get_data_for_autoencoder


class DenoisingAutoencoder(nn.Module):
    """Denoising Autoencoder with noise injection and optimized architecture"""

    def __init__(self, input_dim, noise_factor=0.15):
        super(DenoisingAutoencoder, self).__init__()
        self.noise_factor = noise_factor
        self.encoder = nn.Sequential(nn.Linear(input_dim, 116), nn.ReLU(), nn.Dropout(0.01), nn.Linear(116, 45),
                                     nn.ReLU(), nn.Dropout(0.01), nn.Linear(45, 25), nn.ReLU(), nn.Linear(25, 5))
        self.decoder = nn.Sequential(nn.Linear(5, 25), nn.ReLU(), nn.Linear(25, 45), nn.ReLU(), nn.Dropout(0.01),
                                     nn.Linear(45, 116), nn.ReLU(), nn.Dropout(0.01), nn.Linear(116, input_dim))

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor;
        return x + noise

    def forward(self, x, add_noise=True):
        if add_noise and self.training: x = self.add_noise(x)
        encoded = self.encoder(x);
        decoded = self.decoder(encoded);
        return decoded


class DenoisingAutoencoderModel:
    """Denoising Autoencoder wrapper with optimized hyperparameters"""

    def __init__(self, input_dim, noise_factor=0.15):
        self.device = Config.DEVICE
        self.model = DenoisingAutoencoder(input_dim, noise_factor).to(self.device)
        self.criterion = nn.MSELoss()
        optimal_lr = 0.008123
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=optimal_lr)
        self.threshold = None
        self.scaler = None
        print(f"Denoising Autoencoder initialized on {self.device}")
        print(f"  Input dim: {input_dim}, Noise Factor: {noise_factor}, LR: {optimal_lr}")

    def save_model(self, filename):
        """Save model, including scaler."""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        torch.save({'model_state': self.model.state_dict(), 'threshold': self.threshold, 'scaler': self.scaler},
                   filepath)

    def load_model(self, filename):
        """Load model, raising error if scaler is missing."""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Model file {filepath} not found")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.threshold = checkpoint.get('threshold')
        if 'scaler' not in checkpoint:
            raise KeyError("❌ Scaler not found in the checkpoint file. Please retrain the model.")
        self.scaler = checkpoint['scaler']
        print("✅ Denoising model and scaler loaded successfully.")

    def train(self, train_loader, val_loader, scaler):
        print("\n" + "=" * 70);
        print("🚀 TRAINING DENOISING AUTOENCODER");
        print("=" * 70)
        self.scaler = scaler
        best_val_loss = float('inf');
        patience = 10;
        patience_counter = 0;
        train_losses, val_losses = [], []

        # Main training loop with full tqdm progress bar
        progress_bar = tqdm(range(Config.AUTOENCODER_EPOCHS), desc="Training Denoising AE")
        for epoch in progress_bar:
            self.model.train();
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device);
                self.optimizer.zero_grad()
                reconstructed = self.model(batch, add_noise=True)
                loss = self.criterion(reconstructed, batch);
                loss.backward();
                self.optimizer.step()
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.model.eval();
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device);
                    reconstructed = self.model(batch, add_noise=False)
                    loss = self.criterion(reconstructed, batch);
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 1
            val_losses.append(avg_val_loss)

            # Update the progress bar with the latest loss values
            progress_bar.set_postfix(train_loss=f"{avg_train_loss:.6f}", val_loss=f"{avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss;
                patience_counter = 0
                self.save_model('best_denoising_autoencoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience: print(f"\n  Early stopping at epoch {epoch + 1}"); break

        print("\nLoading best model for final steps...")
        self.load_model('best_denoising_autoencoder.pth')
        self._set_threshold(val_loader)
        self._plot_training_history(train_losses, val_losses)

    def _set_threshold(self, val_loader):
        print("Setting anomaly threshold...")
        self.model.eval();
        errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device);
                reconstructed = self.model(batch, add_noise=False)
                error = torch.mean((batch - reconstructed) ** 2, dim=1);
                errors.extend(error.cpu().numpy())
        if errors:
            self.threshold = np.percentile(errors, Config.ANOMALY_THRESHOLD_PERCENTILE)
        else:
            self.threshold = float('inf')
        print(f"✓ Threshold set to: {self.threshold:.6f}")

    def detect_anomalies(self, data_loader):
        self.model.eval();
        all_errors, all_predictions = [], []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Detecting Anomalies"):
                if isinstance(batch, tuple): batch = batch[0]
                batch = batch.to(self.device);
                reconstructed = self.model(batch, add_noise=False)
                errors = torch.mean((batch - reconstructed) ** 2, dim=1);
                all_errors.extend(errors.cpu().numpy())
                predictions = (errors > self.threshold).cpu().numpy();
                all_predictions.extend(predictions)
        return np.array(all_predictions), np.array(all_errors)

    def _plot_training_history(self, train_losses, val_losses):
        print("Plotting training history...")
        plt.figure(figsize=(10, 6));
        plt.plot(train_losses, label='Train Loss');
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch');
        plt.ylabel('Loss (MSE)');
        plt.title('Denoising Autoencoder Training History')
        plt.legend();
        plt.grid(True);
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'denoising_autoencoder_training.png'));
        plt.close()
        print("✓ Plot saved.")


if __name__ == "__main__":
    Config.set_seeds()
    train_loader, val_loader, scaler, final_feature_list = get_data_for_autoencoder()
    if final_feature_list and len(train_loader.dataset) > 0:
        input_dim = len(final_feature_list)
        denoising_ae = DenoisingAutoencoderModel(input_dim, noise_factor=0.15)
        denoising_ae.train(train_loader, val_loader, scaler)
        denoising_ae.save_model('denoising_autoencoder_final.pth')
        print("\n✅ Denoising Autoencoder training complete and final model saved.")
