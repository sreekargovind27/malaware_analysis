"""
Autoencoder for Anomaly Detection.
This model trains on benign data and identifies anomalies based on reconstruction error.
It includes hyperparameter tuning using Optuna.

This version has been modified to:
- enforce strictly compressive encoder architectures
- add dropout + weight decay + gradient clipping for stability
- optionally add latent sparsity penalty
- use saner Optuna search space (no giant overcomplete layers)
- choose a conservative threshold based on benign validation errors
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


# =========================
#  Autoencoder Architecture
# =========================

class Autoencoder(nn.Module):
    """
    Constrained compressive Autoencoder for anomaly detection.
    Forces strictly decreasing hidden dimensions in the encoder so it can't just learn identity.
    """

    def __init__(self, input_dim, latent_dim=8, hidden_layers=None, dropout_rate=0.1):
        super(Autoencoder, self).__init__()

        if hidden_layers is None:
            # default strictly compressive progression
            hidden_layers = [32, 16, 8]

        # enforce strictly decreasing for encoder
        clean_hidden = []
        prev = input_dim
        for h in hidden_layers:
            if h < prev:
                clean_hidden.append(h)
                prev = h
        hidden_layers = clean_hidden  # overwrite with strictly decreasing stack

        self.hidden_layers = hidden_layers
        self.latent_dim = latent_dim
        self.dropout_rate = dropout_rate

        # ----- Encoder -----
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim

        # final bottleneck
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ----- Decoder (mirror) -----
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            prev_dim = hidden_dim

        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat

    def encode(self, x):
        return self.encoder(x)


# =========================
#  Autoencoder Wrapper
# =========================

class AutoencoderModel:
    """Wrapper class for the Autoencoder model, handling training and inference."""

    def __init__(self, input_dim):
        self.device = Config.DEVICE
        self.input_dim = input_dim

        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()
        self.scaler = None

        self.best_params = None
        self.threshold = None

        # Default attributes (fallbacks if Optuna is off)
        self.latent_dim = getattr(Config, "AUTOENCODER_LATENT_DIM", 8)
        self.hidden_layers = [32, 16, 8]
        self.dropout_rate = 0.1

        self.learning_rate = getattr(Config, "AUTOENCODER_LR", 1e-3)

        # Regularization knobs
        self.use_latent_sparsity = getattr(Config, "USE_LATENT_SPARSITY", True)
        self.latent_sparsity_lambda = getattr(Config, "LATENT_SPARSITY_LAMBDA", 1e-3)
        self.sparsity_warmup_epochs = getattr(Config, "SPARSITY_WARMUP_EPOCHS", 30)

        print(f"Autoencoder initialized on {self.device}")

    # -------------------------
    # Checkpoint IO
    # -------------------------

    @classmethod
    def load_from_checkpoint(cls, filename='autoencoder_final.pth'):
        """
        Load model from checkpoint by first reading architecture,
        then creating the model with correct dimensions.
        """
        filepath = os.path.join(Config.DL_MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")

        checkpoint = torch.load(filepath, map_location=Config.DEVICE, weights_only=False)

        print("💾 Reading architecture from checkpoint...")
        hidden_layers = checkpoint['hidden_layers']
        latent_dim = checkpoint['latent_dim']
        input_dim = checkpoint.get('input_dim')

        if input_dim is None:
            # infer input_dim from first encoder layer weight
            first_layer_weight = checkpoint['model_state']['encoder.0.weight']
            input_dim = first_layer_weight.shape[1]
            print(f"⚠️  input_dim not in checkpoint, inferred from model: {input_dim}")

        print(f"🏗️  Building model with input_dim: {input_dim}, "
              f"architecture: {hidden_layers}, latent_dim: {latent_dim}")

        instance = cls(input_dim=input_dim)

        # restore attrs
        instance.latent_dim = latent_dim
        instance.hidden_layers = hidden_layers
        instance.dropout_rate = checkpoint.get('dropout_rate', 0.1)
        instance.learning_rate = checkpoint.get('learning_rate', getattr(Config, "AUTOENCODER_LR", 1e-3))
        instance.best_params = checkpoint.get('best_params')
        instance.threshold = checkpoint.get('threshold')

        instance.use_latent_sparsity = checkpoint.get(
            'use_latent_sparsity',
            getattr(Config, "USE_LATENT_SPARSITY", True)
        )
        instance.latent_sparsity_lambda = checkpoint.get(
            'latent_sparsity_lambda',
            getattr(Config, "LATENT_SPARSITY_LAMBDA", 1e-3)
        )

        instance.build_model()
        instance.model.load_state_dict(checkpoint['model_state'])
        instance.model.eval()

        if 'scaler' in checkpoint:
            instance.scaler = checkpoint['scaler']

        print(f"✅ Model loaded successfully from: {filename}")
        return instance

    def build_model(self):
        """Builds or rebuilds the model based on current attributes."""
        print(f"\n🏗️  Building model with architecture: {self.hidden_layers}, latent_dim: {self.latent_dim}")
        self.model = Autoencoder(
            self.input_dim,
            latent_dim=self.latent_dim,
            hidden_layers=self.hidden_layers,
            dropout_rate=self.dropout_rate
        ).to(self.device)

        # Adam with weight decay for regularization
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=1e-4
        )

    def load_model(self, filename):
        """Load model, rebuilding the architecture from the file first."""
        filepath = os.path.join(Config.DL_MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Model file {filepath} not found")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        print("💾 Reading architecture from checkpoint...")
        self.latent_dim = checkpoint['latent_dim']
        self.hidden_layers = checkpoint['hidden_layers']
        self.dropout_rate = checkpoint['dropout_rate']
        self.learning_rate = checkpoint['learning_rate']
        self.scaler = checkpoint['scaler']
        self.threshold = checkpoint.get('threshold')
        self.best_params = checkpoint.get('best_params')

        self.use_latent_sparsity = checkpoint.get(
            'use_latent_sparsity',
            getattr(Config, "USE_LATENT_SPARSITY", True)
        )
        self.latent_sparsity_lambda = checkpoint.get(
            'latent_sparsity_lambda',
            getattr(Config, "LATENT_SPARSITY_LAMBDA", 1e-3)
        )
        self.sparsity_warmup_epochs = checkpoint.get('sparsity_warmup_epochs', 30)

        self.build_model()
        self.model.load_state_dict(checkpoint['model_state'])
        print("✅ Model state loaded successfully into matching architecture.")

    def save_model(self, filename):
        """Save model and its architecture."""
        filepath = os.path.join(Config.DL_MODELS_DIR, filename)
        if self.model is None:
            raise RuntimeError("Model has not been built yet. Cannot save.")
        torch.save({
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'sparsity_warmup_epochs': self.sparsity_warmup_epochs,

            'threshold': self.threshold,
            'scaler': self.scaler,
            'input_dim': self.input_dim,

            'latent_dim': self.latent_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'best_params': self.best_params,

            'use_latent_sparsity': self.use_latent_sparsity,
            'latent_sparsity_lambda': self.latent_sparsity_lambda,
        }, filepath)

    # -------------------------
    # Hyperparameter Optimization
    # -------------------------

    def optimize_hyperparameters(self, train_loader, val_loader, scaler):
        """Use Optuna to find best hyperparameters."""
        print("\n" + "=" * 70)
        print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (Autoencoder)")
        print("=" * 70)
        t_start = time.time()

        if not getattr(Config, "USE_OPTUNA", True):
            print("\n⭐️  Optuna disabled in config, using default parameters")
            return {
                'latent_dim': getattr(Config, "AUTOENCODER_LATENT_DIM", 8),
                'hidden_layer_1': 32,
                'hidden_layer_2': 16,
                'hidden_layer_3': 8,
                'dropout_rate': 0.1,
                'learning_rate': getattr(Config, "AUTOENCODER_LR", 1e-3),
                'batch_size': getattr(Config, "AUTOENCODER_BATCH_SIZE", 4096),
            }

        def objective(trial):
            # strictly decreasing proposal
            h1 = trial.suggest_int('hidden_layer_1', 32, 64)  # below ~50 input_dim
            h2 = trial.suggest_int('hidden_layer_2', 16, min(32, h1))
            h3 = trial.suggest_int('hidden_layer_3', 8, min(16, h2))

            latent_dim = trial.suggest_int('latent_dim', 2, min(8, h3))
            dropout_rate = trial.suggest_float('dropout_rate', 0.05, 0.3)

            # tighter LR range for stability
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 5e-3, log=True)

            hidden_layers = [h1, h2, h3]

            temp_model = Autoencoder(
                self.input_dim,
                latent_dim=latent_dim,
                hidden_layers=hidden_layers,
                dropout_rate=dropout_rate
            ).to(self.device)

            temp_optimizer = torch.optim.Adam(
                temp_model.parameters(),
                lr=learning_rate,
                weight_decay=1e-4
            )
            temp_criterion = nn.MSELoss()

            best_val = float('inf')
            patience_local = 5
            patience_ctr = 0

            for epoch in range(min(20, getattr(Config, "AUTOENCODER_EPOCHS", 100))):
                temp_model.train()
                train_loss_epoch = 0.0
                for batch in train_loader:
                    batch = batch.to(self.device)

                    temp_optimizer.zero_grad()
                    recon = temp_model(batch)
                    loss = temp_criterion(recon, batch)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(temp_model.parameters(), max_norm=5.0)
                    temp_optimizer.step()

                    train_loss_epoch += loss.item()

                # validation
                temp_model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        recon = temp_model(batch)
                        vloss = temp_criterion(recon, batch)
                        val_loss += vloss.item()
                val_loss /= max(len(val_loader), 1)

                if val_loss < best_val:
                    best_val = val_loss
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                    if patience_ctr >= patience_local:
                        break

            return best_val

        study = optuna.create_study(direction='minimize')
        study.optimize(
            objective,
            n_trials=getattr(Config, "OPTUNA_N_TRIALS", 20),
            timeout=getattr(Config, "OPTUNA_TIMEOUT", None)
        )

        self.best_params = study.best_params
        return study.best_params

    # -------------------------
    # Training
    # -------------------------

    def train(self, train_loader, val_loader, scaler, use_optuna=True):
        print("\n" + "=" * 70)
        print("🚀 TRAINING AUTOENCODER")
        print("=" * 70)

        self.scaler = scaler

        # Hyperparam search
        if use_optuna and getattr(Config, "USE_OPTUNA", True):
            best_params = self.optimize_hyperparameters(train_loader, val_loader, scaler)

            self.latent_dim = best_params['latent_dim']
            self.hidden_layers = [
                best_params['hidden_layer_1'],
                best_params['hidden_layer_2'],
                best_params['hidden_layer_3']
            ]
            self.dropout_rate = best_params['dropout_rate']
            self.learning_rate = best_params['learning_rate']

        # Rebuild model+optimizer with chosen params
        self.build_model()

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0

        train_losses, val_losses = [], []

        for epoch in tqdm(range(getattr(Config, "AUTOENCODER_EPOCHS", 100)), desc="Training Autoencoder"):
            self.model.train()
            train_loss = 0.0

            for batch in train_loader:
                batch = batch.to(self.device)

                self.optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = self.criterion(reconstructed, batch)

                # latent sparsity penalty (optional, encourages compact representations)
                if self.use_latent_sparsity:
                    z = self.model.encode(batch)
                    sparsity_loss = torch.mean(torch.abs(z))

                    # Progressive annealing: start at 0, ramp up to full lambda over warmup_epochs
                    if epoch < self.sparsity_warmup_epochs:
                        current_lambda = (epoch / self.sparsity_warmup_epochs) * self.latent_sparsity_lambda
                    else:
                        current_lambda = self.latent_sparsity_lambda

                    loss = loss + current_lambda * sparsity_loss

                loss.backward()

                # gradient clip to stabilize training on MPS
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)

                self.optimizer.step()
                train_loss += loss.item()

            train_losses.append(train_loss / max(len(train_loader), 1))
            # Log sparsity schedule every 10 epochs
            if epoch % 10 == 0 and self.use_latent_sparsity:
                if epoch < self.sparsity_warmup_epochs:
                    current_lambda = (epoch / self.sparsity_warmup_epochs) * self.latent_sparsity_lambda
                else:
                    current_lambda = self.latent_sparsity_lambda
                tqdm.write(f"Epoch {epoch}: Sparsity lambda = {current_lambda:.6f}")

            # Validation
            self.model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(self.device)
                    reconstructed = self.model(batch)
                    vloss = self.criterion(reconstructed, batch)
                    val_loss += vloss.item()
            val_loss /= max(len(val_loader), 1)

            val_losses.append(val_loss)

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

        # reload best weights
        self.load_model('best_autoencoder.pth')

        # set anomaly threshold from benign validation distribution
        self._set_threshold(val_loader)

        # training curves
        self._plot_training_history(train_losses, val_losses)

    # -------------------------
    # Thresholding / Inference
    # -------------------------

    def _set_threshold(self, val_loader):
        """
        Compute a conservative anomaly threshold:
        - collect reconstruction errors on benign validation batches
        - set threshold so that only ~5% of benign validation exceeds it
        (i.e. low false-positive rate on known-good data)
        """
        self.model.eval()
        errors = []

        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device)
                reconstructed = self.model(batch)
                error = torch.mean((batch - reconstructed) ** 2, dim=1)
                errors.extend(error.cpu().numpy())

        if not errors:
            self.threshold = float('inf')
            return

        # 95th percentile of benign val error = only ~5% of benign would alarm
        self.threshold = np.percentile(errors, 95)

    def detect_anomalies(self, data_loader):
        """
        Returns:
        - predictions (1 = anomaly, 0 = normal)
        - raw per-sample reconstruction errors
        """
        self.model.eval()
        all_errors, all_predictions = [], []

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Processing batches"):
                if isinstance(batch, tuple):
                    batch = batch[0]

                batch = batch.to(self.device)
                reconstructed = self.model(batch)
                errors = torch.mean((batch - reconstructed) ** 2, dim=1)

                errs_np = errors.cpu().numpy()
                all_errors.extend(errs_np)

                preds = (errors > self.threshold).cpu().numpy()
                all_predictions.extend(preds)

        return np.array(all_predictions), np.array(all_errors)

    # -------------------------
    # Visualization
    # -------------------------

    def _plot_training_history(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Train Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss (MSE)')
        plt.title('Autoencoder Training History')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        os.makedirs(Config.AUTOENCODER_RESULTS_DIR, exist_ok=True)
        plt.savefig(os.path.join(Config.AUTOENCODER_RESULTS_DIR, 'autoencoder_training.png'))
        plt.close()


# =========================
#  Entry Point
# =========================

if __name__ == "__main__":
    Config.print_mode_info()
    Config.ensure_output_dirs()
    Config.set_seeds()

    print("\n======================================================================")
    print("🚀 AUTOENCODER TRAINING CONFIGURATION")
    print("======================================================================")
    print(f"Device: {Config.DEVICE}")
    print(f"Batch size: {Config.AUTOENCODER_BATCH_SIZE}")
    print(f"Learning rate: {Config.AUTOENCODER_LR}")
    print(f"Optuna enabled: {Config.USE_OPTUNA}")
    print(f"Latent dim: {Config.AUTOENCODER_LATENT_DIM}")
    print(f"Epochs: {Config.AUTOENCODER_EPOCHS}")
    print("======================================================================\n")

    train_loader, val_loader, scaler, final_feature_list = get_data_for_autoencoder()

    if final_feature_list and len(train_loader.dataset) > 0:
        input_dim = len(final_feature_list)

        autoencoder = AutoencoderModel(input_dim)

        # Train with Optuna tuning (can disable via Config.USE_OPTUNA=False)
        autoencoder.train(
            train_loader,
            val_loader,
            scaler,
            use_optuna=getattr(Config, "USE_OPTUNA", True)
        )

        autoencoder.save_model('autoencoder_final.pth')
