"""
Denoising Autoencoder for Anomaly Detection
Variation: Adds noise during training for better robustness.
This version includes a full Optuna hyperparameter search.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from data_loader import get_data_for_autoencoder


class DenoisingAutoencoder(nn.Module):
    """Flexible Denoising Autoencoder for Optuna tuning."""

    def __init__(self, input_dim, hidden_layers, latent_dim, dropout_rate, noise_factor=0.15):
        super(DenoisingAutoencoder, self).__init__()
        self.noise_factor = noise_factor

        # Build encoder
        encoder_layers = [];
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim));
            encoder_layers.append(nn.ReLU())
            if dropout_rate > 0: encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror)
        decoder_layers = [];
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim));
            decoder_layers.append(nn.ReLU())
            if dropout_rate > 0: decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def add_noise(self, x):
        noise = torch.randn_like(x) * self.noise_factor;
        return x + noise

    def forward(self, x, add_noise=True):
        if add_noise and self.training: x = self.add_noise(x)
        encoded = self.encoder(x);
        decoded = self.decoder(encoded);
        return decoded


class DenoisingAutoencoderModel:
    """Wrapper for the Denoising Autoencoder with a full training pipeline."""

    def __init__(self, input_dim):
        self.device = Config.DEVICE
        self.input_dim = input_dim
        self.model = None;
        self.optimizer = None;
        self.criterion = nn.MSELoss()
        self.scaler = None;
        self.best_params = None;
        self.threshold = None
        # Default attributes
        self.hidden_layers = [116, 45, 25];
        self.latent_dim = 5
        self.dropout_rate = 0.01;
        self.learning_rate = 0.001;
        self.noise_factor = 0.15
        print(f"Denoising Autoencoder wrapper initialized on {self.device}")

    def build_model(self):
        """Builds or rebuilds the model based on current attributes."""
        print(
            f"\n🏗️  Building Denoising AE with architecture: {self.hidden_layers} -> {self.latent_dim}, Noise: {self.noise_factor}")
        self.model = DenoisingAutoencoder(
            self.input_dim, self.hidden_layers, self.latent_dim,
            self.dropout_rate, self.noise_factor
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_model(self, filename):
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath): raise FileNotFoundError(f"❌ Model file {filepath} not found")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        print("💾 Reading architecture from checkpoint...")
        self.hidden_layers = checkpoint['hidden_layers'];
        self.latent_dim = checkpoint['latent_dim']
        self.dropout_rate = checkpoint['dropout_rate'];
        self.learning_rate = checkpoint['learning_rate']
        self.noise_factor = checkpoint['noise_factor'];
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint.get('threshold');
        self.best_params = checkpoint.get('best_params')

        self.build_model()
        self.model.load_state_dict(checkpoint['model_state'])
        print("✅ Model state loaded successfully into matching architecture.")

    def save_model(self, filename):
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if self.model is None: raise RuntimeError("Model has not been built yet.")
        torch.save({
            'model_state': self.model.state_dict(), 'scaler': self.scaler, 'threshold': self.threshold,
            'hidden_layers': self.hidden_layers, 'latent_dim': self.latent_dim,
            'dropout_rate': self.dropout_rate, 'learning_rate': self.learning_rate,
            'noise_factor': self.noise_factor, 'best_params': self.best_params
        }, filepath)

    def optimize_hyperparameters(self, train_loader, val_loader):
        print("\n" + "=" * 70);
        print("🔍 HYPERPARAMETER OPTIMIZATION (Denoising AE)");
        print("=" * 70)
        if not Config.USE_OPTUNA:
            print("\n⭐️ Optuna disabled, using default parameters.")
            return {'n_layers': 3, 'layer_1': 116, 'layer_2': 45, 'layer_3': 25, 'latent_dim': 5, 'dropout_rate': 0.01,
                    'learning_rate': 0.001, 'noise_factor': 0.15}

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_layers = [trial.suggest_categorical(f'layer_{i + 1}', [32, 64, 128]) for i in range(n_layers)]
            latent_dim = trial.suggest_int('latent_dim', 4, 16)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.4)
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            noise_factor = trial.suggest_float('noise_factor', 0.05, 0.3)

            temp_model = DenoisingAutoencoder(self.input_dim, hidden_layers, latent_dim, dropout_rate, noise_factor).to(
                self.device)
            temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=learning_rate)
            temp_criterion = nn.MSELoss()

            best_val_loss = float('inf')
            for epoch in range(15):  # Short training for each trial
                temp_model.train()
                for batch in train_loader:
                    batch = batch.to(self.device);
                    temp_optimizer.zero_grad();
                    reconstructed = temp_model(batch, add_noise=True);
                    loss = temp_criterion(reconstructed, batch);
                    loss.backward();
                    temp_optimizer.step()

                temp_model.eval();
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device);
                        reconstructed = temp_model(batch, add_noise=False);
                        loss = temp_criterion(reconstructed, batch);
                        val_loss += loss.item()
                val_loss /= len(val_loader) if len(val_loader) > 0 else 1
                if val_loss < best_val_loss: best_val_loss = val_loss
            return best_val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=Config.OPTUNA_N_TRIALS, timeout=Config.OPTUNA_TIMEOUT,
                       show_progress_bar=True)
        self.best_params = study.best_params
        return study.best_params

    def train(self, train_loader, val_loader, scaler, use_optuna=True):
        print("\n" + "=" * 70);
        print("🚀 TRAINING DENOISING AUTOENCODER");
        print("=" * 70)
        self.scaler = scaler

        if use_optuna and Config.USE_OPTUNA:
            best_params = self.optimize_hyperparameters(train_loader, val_loader)
            n_layers = best_params.pop('n_layers')
            # Use .get() as a safeguard in case Optuna prunes before suggesting all layers
            self.hidden_layers = [best_params.get(f'layer_{i + 1}', 64) for i in range(n_layers)]
            self.latent_dim = best_params['latent_dim']
            self.dropout_rate = best_params['dropout_rate']
            self.learning_rate = best_params['learning_rate']
            self.noise_factor = best_params['noise_factor']

        self.build_model()
        best_val_loss = float('inf');
        patience = 10;
        patience_counter = 0
        train_losses, val_losses = [], []

        progress_bar = tqdm(range(Config.AUTOENCODER_EPOCHS), desc="Training Denoising AE")
        for epoch in progress_bar:
            self.model.train();
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device);
                self.optimizer.zero_grad();
                reconstructed = self.model(batch, add_noise=True);
                loss = self.criterion(reconstructed, batch);
                loss.backward();
                self.optimizer.step();
                train_loss += loss.item()
            avg_train_loss = train_loss / len(train_loader)
            train_losses.append(avg_train_loss)

            self.model.eval();
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device);
                    reconstructed = self.model(batch, add_noise=False);
                    loss = self.criterion(reconstructed, batch);
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 1
            val_losses.append(avg_val_loss)

            progress_bar.set_postfix(train_loss=f"{avg_train_loss:.6f}", val_loss=f"{avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss;
                patience_counter = 0;
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
        denoising_ae = DenoisingAutoencoderModel(input_dim)
        denoising_ae.train(train_loader, val_loader, scaler, use_optuna=True)
        denoising_ae.save_model('denoising_autoencoder_final.pth')
        print("\n✅ Denoising Autoencoder training complete and final model saved.")
