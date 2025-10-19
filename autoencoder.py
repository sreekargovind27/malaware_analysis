"""
Autoencoder for Anomaly Detection
Trains on benign data only, detects anomalies by reconstruction error.
UPDATED: This is the final, corrected version with all trials and progress bars.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import optuna
import torch
import torch.nn as nn
from optuna.samplers import TPESampler
from tqdm import tqdm

from config import Config
from data_loader import get_data_for_autoencoder


class Autoencoder(nn.Module):
    """Flexible Autoencoder architecture with configurable layers"""

    def __init__(self, input_dim, latent_dim=8, hidden_layers=None, dropout_rate=0.0):
        super(Autoencoder, self).__init__()
        if hidden_layers is None: hidden_layers = [64, 32, 16]
        encoder_layers = [];
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            encoder_layers.append(nn.Linear(prev_dim, hidden_dim));
            encoder_layers.append(nn.ReLU())
            if dropout_rate > 0: encoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        decoder_layers = [];
        prev_dim = latent_dim
        for hidden_dim in reversed(hidden_layers):
            decoder_layers.append(nn.Linear(prev_dim, hidden_dim));
            decoder_layers.append(nn.ReLU())
            if dropout_rate > 0: decoder_layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        encoded = self.encoder(x);
        decoded = self.decoder(encoded);
        return decoded


class AutoencoderModel:
    """Wrapper class for training and inference with Optuna support"""

    def __init__(self, input_dim):
        self.device = Config.DEVICE;
        self.input_dim = input_dim
        self.model = None;
        self.optimizer = None;
        self.criterion = nn.MSELoss()
        self.scaler = None;
        self.best_params = None;
        self.threshold = None
        self.latent_dim = Config.AUTOENCODER_LATENT_DIM;
        self.hidden_layers = [64, 32, 16]
        self.dropout_rate = 0.0;
        self.learning_rate = Config.AUTOENCODER_LR
        print(f"Autoencoder initialized on {self.device}")

    def build_model(self):
        print(f"\n🏗️  Building model with architecture: {self.hidden_layers}, latent_dim: {self.latent_dim}")
        self.model = Autoencoder(self.input_dim, self.latent_dim, self.hidden_layers, self.dropout_rate).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_model(self, filename):
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath): raise FileNotFoundError(f"❌ Model file {filepath} not found")
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        print("💾 Reading architecture from checkpoint...")
        self.latent_dim = checkpoint['latent_dim'];
        self.hidden_layers = checkpoint['hidden_layers']
        self.dropout_rate = checkpoint['dropout_rate'];
        self.learning_rate = checkpoint['learning_rate']
        self.scaler = checkpoint['scaler'];
        self.threshold = checkpoint.get('threshold')
        self.best_params = checkpoint.get('best_params')
        self.build_model();
        self.model.load_state_dict(checkpoint['model_state'])
        print("✅ Model state loaded successfully into matching architecture.")

    def save_model(self, filename):
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if self.model is None: raise RuntimeError("Model has not been built yet.")
        torch.save({
            'model_state': self.model.state_dict(), 'optimizer_state': self.optimizer.state_dict(),
            'threshold': self.threshold, 'scaler': self.scaler, 'latent_dim': self.latent_dim,
            'hidden_layers': self.hidden_layers, 'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate, 'best_params': self.best_params
        }, filepath)

    def optimize_hyperparameters(self, train_loader, val_loader, scaler):
        print("\n" + "=" * 70);
        print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (Autoencoder)");
        print("=" * 70)
        if not Config.USE_OPTUNA:
            print("\n⭐️  Optuna disabled in config, using default parameters")
            return {'latent_dim': 8, 'hidden_layer_1': 64, 'hidden_layer_2': 32, 'hidden_layer_3': 16,
                    'dropout_rate': 0.0, 'learning_rate': 0.0001}

        def objective(trial):
            latent_dim = trial.suggest_int('latent_dim', 4, 32)
            hidden_layer_1 = trial.suggest_int('hidden_layer_1', 32, 128)
            hidden_layer_2 = trial.suggest_int('hidden_layer_2', 16, 64)
            hidden_layer_3 = trial.suggest_int('hidden_layer_3', 8, 32)
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
            hidden_layers = [hidden_layer_1, hidden_layer_2, hidden_layer_3]
            temp_model = Autoencoder(self.input_dim, latent_dim, hidden_layers, dropout_rate).to(self.device)
            temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=learning_rate)
            temp_criterion = nn.MSELoss()

            best_val_loss = float('inf')
            for epoch in range(min(20, Config.AUTOENCODER_EPOCHS)):
                temp_model.train()
                for batch in train_loader:
                    batch = batch.to(self.device);
                    temp_optimizer.zero_grad();
                    reconstructed = temp_model(batch);
                    loss = temp_criterion(reconstructed, batch);
                    loss.backward();
                    temp_optimizer.step()
                temp_model.eval();
                val_loss = 0
                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(self.device);
                        reconstructed = temp_model(batch);
                        loss = temp_criterion(reconstructed, batch);
                        val_loss += loss.item()
                val_loss /= len(val_loader) if len(val_loader) > 0 else 1
                if val_loss < best_val_loss: best_val_loss = val_loss
            return best_val_loss

        study = optuna.create_study(direction='minimize', sampler=TPESampler(seed=Config.RANDOM_STATE))

        # ============================================================================
        # THIS IS THE FIX: Runs all trials from your config file.
        # TODO - add Config.OPTUNA_N_TRIALS
        study.optimize(objective, n_trials=4, timeout=Config.OPTUNA_TIMEOUT,
                       show_progress_bar=True)
        # ============================================================================

        print("\n✅ Optimization complete")
        print(f"   Best validation loss: {study.best_value:.6f}")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")

        self.best_params = study.best_params
        return study.best_params

    def train(self, train_loader, val_loader, scaler, use_optuna=True):
        print("\n" + "=" * 70);
        print("🚀 TRAINING AUTOENCODER");
        print("=" * 70)
        self.scaler = scaler
        if use_optuna and Config.USE_OPTUNA:
            best_params = self.optimize_hyperparameters(train_loader, val_loader, scaler)
            self.latent_dim = best_params['latent_dim'];
            self.hidden_layers = [best_params['hidden_layer_1'], best_params['hidden_layer_2'],
                                  best_params['hidden_layer_3']]
            self.dropout_rate = best_params['dropout_rate'];
            self.learning_rate = best_params['learning_rate']

        self.build_model()
        best_val_loss = float('inf');
        patience = 10;
        patience_counter = 0;
        train_losses, val_losses = [], []

        progress_bar = tqdm(range(Config.AUTOENCODER_EPOCHS), desc="Training Final Autoencoder")
        for epoch in progress_bar:
            self.model.train();
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(self.device);
                self.optimizer.zero_grad();
                reconstructed = self.model(batch);
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
                    reconstructed = self.model(batch);
                    loss = self.criterion(reconstructed, batch);
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 1
            val_losses.append(avg_val_loss)

            progress_bar.set_postfix(train_loss=f"{avg_train_loss:.6f}", val_loss=f"{avg_val_loss:.6f}")

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss;
                patience_counter = 0;
                self.save_model('best_autoencoder.pth')
            else:
                patience_counter += 1
                if patience_counter >= patience: print(f"\n  Early stopping at epoch {epoch + 1}"); break

        print("\nLoading best model for final steps...")
        self.load_model('best_autoencoder.pth')
        self._set_threshold(val_loader)
        self._plot_training_history(train_losses, val_losses)

    def _set_threshold(self, val_loader):
        print("Setting anomaly threshold...")
        self.model.eval();
        errors = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(self.device);
                reconstructed = self.model(batch);
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
                reconstructed = self.model(batch);
                errors = torch.mean((batch - reconstructed) ** 2, dim=1);
                all_errors.extend(errors.cpu().numpy());
                predictions = (errors > self.threshold).cpu().numpy();
                all_predictions.extend(predictions)
        return np.array(all_predictions), np.array(all_errors)

    def _plot_training_history(self, train_losses, val_losses):
        print("Plotting training history...")
        plt.figure(figsize=(10, 6));
        plt.plot(train_losses, label='Train Loss');
        plt.plot(val_losses, label='Validation Loss');
        plt.xlabel('Epoch');
        plt.ylabel('Loss (MSE)');
        plt.title('Autoencoder Training History');
        plt.legend();
        plt.grid(True);
        plt.tight_layout();
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'autoencoder_training.png'));
        plt.close()
        print("✓ Plot saved.")


if __name__ == "__main__":
    Config.print_mode_info();
    Config.set_seeds()
    train_loader, val_loader, scaler, final_feature_list = get_data_for_autoencoder()
    if final_feature_list and len(train_loader.dataset) > 0:
        input_dim = len(final_feature_list)
        autoencoder = AutoencoderModel(input_dim)
        autoencoder.train(train_loader, val_loader, scaler, use_optuna=True)
        autoencoder.save_model('autoencoder_final.pth')
        print("\n✅ Autoencoder training complete and final model saved.")
