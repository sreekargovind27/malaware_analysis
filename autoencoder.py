"""
Autoencoder for Anomaly Detection
Trains on benign data only, detects anomalies by reconstruction error.
UPDATED: Added Optuna hyperparameter tuning + enhanced timing display.
"""

import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_loader import get_data_for_autoencoder, load_engineered_data, IoTDataset


class Autoencoder(nn.Module):
    """Flexible Autoencoder architecture with configurable layers"""

    def __init__(self, input_dim, latent_dim=8, hidden_layers=None, dropout_rate=0.0):
        super(Autoencoder, self).__init__()

        if hidden_layers is None:
            hidden_layers = [64, 32, 16]

        # Build encoder
        encoder_layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            encoder_layers.append(nn.ReLU())
            if dropout_rate > 0:
                encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # Build decoder (mirror of encoder)
        decoder_layers = []
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            decoder_layers.append(nn.ReLU())
            if dropout_rate > 0:
                decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


class AutoencoderModel:
    """Wrapper class for training and inference with Optuna support"""

    def __init__(self, input_dim, latent_dim=None, hidden_layers=None, dropout_rate=None, learning_rate=None):
        self.device = Config.DEVICE
        self.input_dim = input_dim

        # Use provided params or defaults from Config
        self.latent_dim = latent_dim or Config.AUTOENCODER_LATENT_DIM
        self.hidden_layers = hidden_layers or [64, 32, 16]
        self.dropout_rate = dropout_rate or 0.0
        self.learning_rate = learning_rate or Config.AUTOENCODER_LR

        self.model = Autoencoder(
            input_dim,
            self.latent_dim,
            self.hidden_layers,
            self.dropout_rate
        ).to(self.device)

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.threshold = None
        self.scaler = None
        self.best_params = None

        print(f"Autoencoder initialized on {self.device}")
        print(f"  Input dim: {input_dim}")
        print(f"  Latent dim: {self.latent_dim}")
        print(f"  Hidden layers: {self.hidden_layers}")
        print(f"  Dropout rate: {self.dropout_rate}")
        print(f"  Learning rate: {self.learning_rate}")

    def optimize_hyperparameters(self, train_loader, val_loader, scaler):
        """Use Optuna to find best hyperparameters"""
        print("\n" + "=" * 70)
        print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (Autoencoder)")
        print("=" * 70)

        t_start = time.time()

        if not Config.USE_OPTUNA:
            print("\n⭐️  Optuna disabled in config, using default parameters")
            return {
                'latent_dim': Config.AUTOENCODER_LATENT_DIM,
                'hidden_layer_1': 64,
                'hidden_layer_2': 32,
                'hidden_layer_3': 16,
                'dropout_rate': 0.0,
                'learning_rate': Config.AUTOENCODER_LR,
                'batch_size': Config.AUTOENCODER_BATCH_SIZE
            }

        print(f"\n⏳ Running Optuna optimization...")
        print(f"   Trials: {Config.OPTUNA_N_TRIALS}")
        print(f"   Timeout: {Config.OPTUNA_TIMEOUT}s ({Config.OPTUNA_TIMEOUT / 60:.1f} min)")

        def objective(trial):
            # Suggest hyperparameters
            latent_dim = trial.suggest_int('latent_dim', 4, 32)
            hidden_layer_1 = trial.suggest_int('hidden_layer_1', 32, 128)
            hidden_layer_2 = trial.suggest_int('hidden_layer_2', 16, 64)
            hidden_layer_3 = trial.suggest_int('hidden_layer_3', 8, 32)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)

            # Create model with suggested parameters
            hidden_layers = [hidden_layer_1, hidden_layer_2, hidden_layer_3]
            temp_model = Autoencoder(
                self.input_dim,
                latent_dim,
                hidden_layers,
                dropout_rate
            ).to(self.device)

            temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=learning_rate)
            temp_criterion = nn.MSELoss()

            # Train for a few epochs
            num_epochs = min(20, Config.AUTOENCODER_EPOCHS)
            best_val_loss = float('inf')

            for epoch in range(num_epochs):
                # Training
                temp_model.train()
                train_loss = 0
                for batch in train_loader:
                    batch = batch.to(self.device)
                    temp_optimizer.zero_grad()
                    reconstructed = temp_model(batch)
                    loss = temp_criterion(reconstructed, batch)

                    if torch.isnan(loss):
                        return float('inf')

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(temp_model.parameters(), max_norm=1.0)
                    temp_optimizer.step()
                    train_loss += loss.item()

                # Validation
                temp_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device)
                        reconstructed = temp_model(batch)
                        loss = temp_criterion(reconstructed, batch)
                        val_loss += loss.item()

                val_loss /= len(val_loader) if len(val_loader) > 0 else 1

                if val_loss < best_val_loss:
                    best_val_loss = val_loss

                # Early stopping for optimization
                if epoch > 5 and val_loss > best_val_loss * 1.5:
                    break

            return best_val_loss

        sampler = TPESampler(seed=Config.RANDOM_STATE)
        study = optuna.create_study(direction='minimize', sampler=sampler)
        study.optimize(
            objective,
            n_trials=Config.OPTUNA_N_TRIALS,
            timeout=Config.OPTUNA_TIMEOUT,
            show_progress_bar=True
        )

        t_elapsed = time.time() - t_start

        print(f"\n✅ Optimization complete")
        print(f"⏱️  Time: {t_elapsed:.2f}s ({t_elapsed / 60:.1f} min)")
        print(f"\n📊 Best trial:")
        print(f"   Validation Loss: {study.best_value:.6f}")
        print(f"   Parameters:")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")

        self.best_params = study.best_params
        return study.best_params

    def train(self, train_loader, val_loader, scaler, use_optuna=True):
        """Train autoencoder on benign data"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING AUTOENCODER")
        print("=" * 70)

        overall_start = time.time()
        self.scaler = scaler

        # Hyperparameter optimization
        if use_optuna and Config.USE_OPTUNA:
            best_params = self.optimize_hyperparameters(train_loader, val_loader, scaler)

            # Rebuild model with best parameters
            self.latent_dim = best_params['latent_dim']
            self.hidden_layers = [
                best_params['hidden_layer_1'],
                best_params['hidden_layer_2'],
                best_params['hidden_layer_3']
            ]
            self.dropout_rate = best_params['dropout_rate']
            self.learning_rate = best_params['learning_rate']

            print("\n" + "=" * 70)
            print("🎯 TRAINING FINAL MODEL WITH OPTIMIZED PARAMETERS")
            print("=" * 70)

            self.model = Autoencoder(
                self.input_dim,
                self.latent_dim,
                self.hidden_layers,
                self.dropout_rate
            ).to(self.device)

            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        else:
            print("\n⭐️  Using default parameters (no optimization)")

        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        train_losses, val_losses = [], []

        print(f"\n📊 Training configuration:")
        print(f"   Epochs: {Config.AUTOENCODER_EPOCHS}")
        print(f"   Batch size: {Config.AUTOENCODER_BATCH_SIZE}")
        print(f"   Learning rate: {self.learning_rate}")
        print(f"   Latent dim: {self.latent_dim}")
        print(f"   Hidden layers: {self.hidden_layers}")
        print(f"   Dropout: {self.dropout_rate}")
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
            'scaler': self.scaler,
            'latent_dim': self.latent_dim,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'best_params': self.best_params
        }, filepath)

    def load_model(self, filename):
        """Load model checkpoint"""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"❌ Model file {filepath} not found")
            return

        # =================================================================
        # ADD weights_only=False HERE
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        # =================================================================

        self.model.load_state_dict(checkpoint['model_state'])
        if 'optimizer_state' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.threshold = checkpoint.get('threshold')
        self.scaler = checkpoint.get('scaler')
        self.best_params = checkpoint.get('best_params')


if __name__ == "__main__":
    Config.print_mode_info()
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

        # Train with Optuna optimization
        autoencoder = AutoencoderModel(input_dim)
        autoencoder.train(train_loader, val_loader, scaler, use_optuna=True)

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

        # Use our custom IoTDataset to ensure the DataLoader yields a tensor directly
        malicious_dataset = IoTDataset(X_malicious_scaled)
        malicious_loader = DataLoader(malicious_dataset, batch_size=4096, shuffle=False)

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
