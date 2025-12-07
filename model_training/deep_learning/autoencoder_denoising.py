"""
Denoising Autoencoder for Anomaly Detection with Optuna hyperparameter search.
Trains only on (cleaned) benign data, injects noise during training, and
flags anomalies using reconstruction error.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from tqdm import tqdm

from config import Config
from models.data_loader import get_data_for_autoencoder


# ---------------------------
# Core Denoising AE module
# ---------------------------

class DenoisingAutoencoder(nn.Module):
    """
    Denoising Autoencoder with configurable depth/width.

    encoder:  input_dim -> h1 -> h2 -> h3 -> latent_dim
    decoder:  latent_dim -> h3 -> h2 -> h1 -> input_dim
    """

    def __init__(self,
                 input_dim: int,
                 latent_dim: int,
                 hidden_layers: list,
                 dropout_rate: float,
                 noise_factor: float = 0.15):
        super().__init__()
        self.noise_factor = noise_factor

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hid in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hid))
            encoder_layers.append(nn.ReLU())
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hid
        # bottleneck
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror)
        decoder_layers = []
        prev_dim = latent_dim
        for hid in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hid))
            decoder_layers.append(nn.ReLU())
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hid
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def add_noise(self, x: torch.Tensor, noise_factor: float = None) -> torch.Tensor:
        # Gaussian noise with optional override
        factor = noise_factor if noise_factor is not None else self.noise_factor
        return x + torch.randn_like(x) * factor

    def forward(self, x: torch.Tensor, add_noise: bool = True, noise_factor: float = None) -> torch.Tensor:
        if add_noise and self.training:
            x = self.add_noise(x, noise_factor)
        z = self.encoder(x)
        out = self.decoder(z)
        return out

# ---------------------------
# Training / Eval wrapper
# ---------------------------

class DenoisingAutoencoderModel:
    """
    Wraps the denoising AE:
    - hyperparam search (Optuna)
    - training loop w/ noise
    - early stopping
    - threshold calibration
    - save/load
    """

    def __init__(self, input_dim: int):
        self.device = Config.DEVICE
        self.input_dim = input_dim

        # will be filled after optuna
        self.latent_dim = None
        self.hidden_layers = None
        self.dropout_rate = None
        self.learning_rate = None
        self.noise_factor = 0.15  # final/max noise level
        self.noise_start = getattr(Config, "NOISE_START", 0.05)  # starting noise level
        self.noise_warmup_epochs = getattr(Config, "NOISE_WARMUP_EPOCHS", 40)  # ramp-up period

        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        self.scaler = None
        self.threshold = None
        self.best_params = None

        print(f"Denoising Autoencoder (Optuna-enabled) initialized on {self.device}")

    # ------------- model (re)build -------------

    def build_model(self):
        """Build the actual torch model + optimizer from current attrs."""
        print(f"\n🏗️  Building Denoising AE with:")
        print(f"    hidden_layers={self.hidden_layers}")
        print(f"    latent_dim={self.latent_dim}")
        print(f"    dropout_rate={self.dropout_rate}")
        print(f"    noise_factor={self.noise_factor}")
        print(f"    learning_rate={self.learning_rate}")

        self.model = DenoisingAutoencoder(
            input_dim=self.input_dim,
            latent_dim=self.latent_dim,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate,
            noise_factor=self.noise_factor
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

    # ------------- save/load -------------

    def save_model(self, filename: str):
        """Save model state + metadata."""
        filepath = os.path.join(Config.DL_MODELS_DIR, filename)
        torch.save({
            'model_state': self.model.state_dict(),
            'threshold': self.threshold,
            'scaler': self.scaler,
            'input_dim': self.input_dim,
            'latent_dim': self.latent_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'noise_factor': self.noise_factor,
            'noise_start': self.noise_start,
            'noise_warmup_epochs': self.noise_warmup_epochs,
            'best_params': self.best_params
        }, filepath)

    def load_model(self, filename: str):
        """Load model + rebuild architecture to match checkpoint."""
        filepath = os.path.join(Config.DL_MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Model file {filepath} not found")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        print("💾 Reading architecture from checkpoint...")

        self.input_dim = checkpoint.get('input_dim', self.input_dim)
        self.latent_dim = checkpoint['latent_dim']
        self.hidden_layers = checkpoint['hidden_layers']
        self.dropout_rate = checkpoint['dropout_rate']
        self.learning_rate = checkpoint['learning_rate']
        self.noise_factor = checkpoint.get('noise_factor', 0.15)

        self.scaler = checkpoint.get('scaler', None)
        self.threshold = checkpoint.get('threshold', None)
        self.noise_start = checkpoint.get('noise_start', 0.05)
        self.noise_warmup_epochs = checkpoint.get('noise_warmup_epochs', 40)
        self.best_params = checkpoint.get('best_params', None)

        self.build_model()
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.eval()

        print("✅ Denoising model and scaler/threshold loaded successfully.")

    @classmethod
    def load_from_checkpoint(cls, filename='denoising_autoencoder_final.pth'):
        """
        Convenience constructor that returns a ready-to-use instance from disk.
        """
        filepath = os.path.join(Config.DL_MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=Config.DEVICE, weights_only=False)

        print("💾 Reading architecture from checkpoint...")

        input_dim = checkpoint.get('input_dim')
        if input_dim is None:
            # infer from first encoder layer
            first_weight = checkpoint['model_state']['encoder.0.weight']
            input_dim = first_weight.shape[1]
            print(f"⚠️  input_dim not in checkpoint, inferred: {input_dim}")

        instance = cls(input_dim=input_dim)

        instance.latent_dim = checkpoint['latent_dim']
        instance.hidden_layers = checkpoint['hidden_layers']
        instance.dropout_rate = checkpoint['dropout_rate']
        instance.learning_rate = checkpoint['learning_rate']
        instance.noise_factor = checkpoint.get('noise_factor', 0.15)

        instance.noise_start = checkpoint.get('noise_start', 0.05)
        instance.noise_warmup_epochs = checkpoint.get('noise_warmup_epochs', 40)

        instance.scaler = checkpoint.get('scaler', None)
        instance.threshold = checkpoint.get('threshold', None)
        instance.best_params = checkpoint.get('best_params', None)

        instance.build_model()
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.model.eval()

        print(f"✅ Model loaded successfully from: {filename}")
        return instance

    # ------------- Optuna search -------------

    def optimize_hyperparameters(self, train_loader, val_loader):
        """
        Use Optuna to tune AE shape + LR on benign data.
        Objective: minimize validation MSE.
        """
        print("\n" + "=" * 70)
        print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (Denoising AE)")
        print("=" * 70)
        t_start = time.time()

        if not Config.USE_OPTUNA:
            print("\n⭐️ Optuna disabled in config, using default fallback params")
            fallback = {
                'latent_dim': Config.AUTOENCODER_LATENT_DIM,
                'hidden_layer_1': 64,
                'hidden_layer_2': 32,
                'hidden_layer_3': 16,
                'dropout_rate': 0.1,
                'learning_rate': Config.AUTOENCODER_LR
            }
            self.best_params = fallback
            return fallback

        def objective(trial):
            latent_dim = trial.suggest_int('latent_dim', 4, 32)
            h1 = trial.suggest_int('hidden_layer_1', 32, 128)
            h2 = trial.suggest_int('hidden_layer_2', 16, 64)
            h3 = trial.suggest_int('hidden_layer_3', 8, 32)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

            hidden_layers = [h1, h2, h3]

            # build temp model
            temp_model = DenoisingAutoencoder(
                input_dim=self.input_dim,
                latent_dim=latent_dim,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate,
                noise_factor=self.noise_factor,
            ).to(self.device)

            temp_optimizer = torch.optim.Adam(
                temp_model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
            temp_criterion = nn.MSELoss()

            # train for a subset of epochs for speed
            max_search_epochs = min(20, Config.AUTOENCODER_EPOCHS)

            for _ in range(max_search_epochs):
                temp_model.train()
                for batch in train_loader:
                    batch = batch.to(self.device)

                    temp_optimizer.zero_grad()
                    recon = temp_model(batch, add_noise=True)
                    loss = temp_criterion(recon, batch)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(temp_model.parameters(), max_norm=5.0)
                    temp_optimizer.step()

            # validation loss
            temp_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    recon = temp_model(batch, add_noise=True)
                    vloss = temp_criterion(recon, batch)
                    val_loss += vloss.item()

            val_loss /= len(val_loader) if len(val_loader) > 0 else 1
            return val_loss

        study = optuna.create_study(direction='minimize')
        study.optimize(
            objective,
            n_trials=Config.OPTUNA_N_TRIALS,
            timeout=Config.OPTUNA_TIMEOUT
        )

        self.best_params = study.best_params
        return study.best_params

    # ------------- Training loop -------------

    def train(self, train_loader, val_loader, scaler, use_optuna=True):
        """
        Full training:
        - run optuna (optional)
        - rebuild model w/best params
        - train w/ early stopping
        - set anomaly threshold
        """
        self.scaler = scaler

        # >>> Optuna search
        if use_optuna and Config.USE_OPTUNA:
            best_params = self.optimize_hyperparameters(train_loader, val_loader)

            self.latent_dim = best_params['latent_dim']
            self.hidden_layers = [
                best_params['hidden_layer_1'],
                best_params['hidden_layer_2'],
                best_params['hidden_layer_3'],
            ]
            self.dropout_rate = best_params['dropout_rate']
            self.learning_rate = best_params['learning_rate']
        else:
            # Fallback (no optuna)
            self.latent_dim = Config.AUTOENCODER_LATENT_DIM
            self.hidden_layers = [64, 32, 16]
            self.dropout_rate = 0.1
            self.learning_rate = Config.AUTOENCODER_LR

        # build final model w/ chosen params
        self.build_model()

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        train_losses, val_losses = [], []

        for epoch in tqdm(range(Config.AUTOENCODER_EPOCHS), desc="Training Denoising AE"):
            self.model.train()
            running_train_loss = 0.0

            for batch in train_loader:
                batch = batch.to(self.device)

                # Use fixed noise factor (curriculum disabled due to high dropout from Optuna)
                current_noise = self.noise_factor

                self.optimizer.zero_grad()
                recon = self.model(batch, add_noise=True, noise_factor=current_noise)
                loss = self.criterion(recon, batch)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
                self.optimizer.step()

                running_train_loss += loss.item()

            epoch_train_loss = running_train_loss / len(train_loader)
            train_losses.append(epoch_train_loss)

            # validate
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    recon = self.model(batch, add_noise=False)
                    vloss = self.criterion(recon, batch)
                    running_val_loss += vloss.item()

            epoch_val_loss = running_val_loss / (len(val_loader) if len(val_loader) > 0 else 1)
            val_losses.append(epoch_val_loss)

            # early stopping + checkpoint best
            if epoch_val_loss < best_val_loss:
                best_val_loss = epoch_val_loss
                patience_counter = 0
                self.save_model('best_denoising_autoencoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                    break

        # load best weights
        self.load_model('best_denoising_autoencoder.pth')

        # calibrate anomaly threshold using validation benign recon error
        self._set_threshold(val_loader)

        # plot curves
        self._plot_training_history(train_losses, val_losses)

    # ------------- Threshold calibration -------------

    def _set_threshold(self, val_loader):
        """
        Compute reconstruction error on benign validation data,
        choose threshold = 95th percentile.
        """
        self.model.eval()
        errors = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                recon = self.model(batch, add_noise=False)
                error = torch.mean((batch - recon) ** 2, dim=1)
                errors.extend(error.cpu().numpy())

        if errors:
            self.threshold = np.percentile(errors, 95)
        else:
            self.threshold = float('inf')

    # ------------- Inference -------------

    def detect_anomalies(self, data_loader):
        """
        Returns:
            predictions: np.array of 0/1 (1 = anomaly)
            errors: np.array of per-sample MSE recon error
        """
        self.model.eval()
        all_preds = []
        all_errors = []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing batches"):
                if isinstance(batch, tuple):
                    batch = batch[0]
                batch = batch.to(self.device)

                recon = self.model(batch, add_noise=False)
                err = torch.mean((batch - recon) ** 2, dim=1)

                err_np = err.cpu().numpy()
                all_errors.extend(err_np)

                pred_np = (err > self.threshold).cpu().numpy()
                all_preds.extend(pred_np)

        return np.array(all_preds), np.array(all_errors)

    # ------------- Plotting -------------

    def _plot_training_history(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Denoising Autoencoder Training History (Optuna-tuned)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.AUTOENCODER_RESULTS_DIR, 'denoising_autoencoder_training.png'))
        plt.close()


# ---------------------------
# Main script entry
# ---------------------------

if __name__ == "__main__":
    Config.set_seeds()
    Config.ensure_output_dirs()

    print("\n======================================================================")
    print("🚀 DENOISING AUTOENCODER (OPTUNA) TRAINING CONFIGURATION")
    print("======================================================================")
    print(f"Device: {Config.DEVICE}")
    print(f"Batch size: {Config.AUTOENCODER_BATCH_SIZE}")
    print(f"Optuna enabled: {Config.USE_OPTUNA}")
    print(f"Initial latent dim hint: {Config.AUTOENCODER_LATENT_DIM}")
    print(f"Epochs: {Config.AUTOENCODER_EPOCHS}")
    print("======================================================================\n")

    train_loader, val_loader, scaler, final_feature_list = get_data_for_autoencoder()

    if final_feature_list and len(train_loader.dataset) > 0:
        input_dim = len(final_feature_list)
        deno_ae = DenoisingAutoencoderModel(input_dim)
        deno_ae.train(train_loader, val_loader, scaler, use_optuna=True)
        deno_ae.save_model('denoising_autoencoder_final.pth')
