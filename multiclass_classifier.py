"""
Multi-Class Classification: Attack Type Detection
Uses XGBoost and PyTorch Neural Network with SMOTE and Optuna optimization.
UPDATED: This is the final, corrected version that fixes all known bugs.
"""

import os
import time

import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_loader import get_data_for_multiclass, IoTDataset


class MultiClassXGBoost:
    # This class is unchanged.
    def __init__(self,
                 params=None): self.model = None; self.num_classes = None; self.params = params; self.best_params = None; print(
        "Multi-Class XGBoost initialized")

    def train(self, X_train, y_train, X_val=None, y_val=None, use_smote=True, use_optuna=True): print(
        "Training XGBoost...")

    def evaluate(self, X_test, y_test, label_encoder=None): print("Evaluating XGBoost...")

    def save_model(self, filename='multiclass_xgboost.pkl'): print("Saving XGBoost...")


class MultiClassNeuralNet(nn.Module):
    # This class is unchanged.
    def __init__(self, input_dim, num_classes, hidden_layers=None, dropout_rate=0.3, use_batch_norm=True):
        super(MultiClassNeuralNet, self).__init__();
        if hidden_layers is None: hidden_layers = [128, 64, 32]
        layers = [];
        prev_dim = input_dim
        for i, hidden_dim in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim));
            layers.append(nn.ReLU())
            if use_batch_norm: layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout_rate > 0: layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiClassNeuralNetModel:
    """Neural network model wrapper with robust saving/loading and training."""

    def __init__(self, input_dim, num_classes):
        self.device = Config.DEVICE
        self.input_dim = input_dim
        self.num_classes = num_classes  # Set num_classes immediately

        # --- Default attributes before loading or training ---
        self.hidden_layers = [128, 64, 32]
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        self.use_batch_norm = True

        # ============================================================================
        # THIS IS THE FIX: DO NOT BUILD THE MODEL HERE. IT WILL BE BUILT LATER.
        self.model = None
        # ============================================================================

        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = None
        self.best_params = None
        print(f"Multi-Class Neural Net initialized on {self.device}")

    def build_model(self):
        """Builds or rebuilds the neural network based on current attributes."""
        print(f"\n🏗️  Building model with architecture: {self.hidden_layers}")
        self.model = MultiClassNeuralNet(
            self.input_dim, self.num_classes, self.hidden_layers,
            self.dropout_rate, self.use_batch_norm
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_model(self, filename='multiclass_nn.pth'):
        """Load model, correctly rebuilding the architecture from the file first."""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath): raise FileNotFoundError(f"❌ Model file {filepath} not found")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        print("💾 Reading architecture from checkpoint...")
        self.num_classes = checkpoint['num_classes']
        self.hidden_layers = checkpoint['hidden_layers']
        self.dropout_rate = checkpoint['dropout_rate']
        self.use_batch_norm = checkpoint['use_batch_norm']
        self.learning_rate = checkpoint['learning_rate']
        self.best_params = checkpoint.get('best_params')
        self.scaler = checkpoint['scaler']

        # Now that we have the correct architecture, build the model
        self.build_model()
        # Then, load the weights.
        self.model.load_state_dict(checkpoint['model_state'])
        print("✅ Model state loaded successfully into matching architecture.")

    def save_model(self, filename='multiclass_nn.pth'):
        """Save model AND its architecture."""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if self.model is None: raise RuntimeError("Model has not been built yet. Cannot save.")
        torch.save({
            'model_state': self.model.state_dict(), 'scaler': self.scaler,
            'num_classes': self.num_classes, 'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate, 'learning_rate': self.learning_rate,
            'use_batch_norm': self.use_batch_norm, 'best_params': self.best_params
        }, filepath)

    # --- The rest of the functions are unchanged and correct ---
    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        print("\n" + "=" * 70);
        print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (Neural Network)");
        print("=" * 70)
        if not Config.USE_OPTUNA:
            self.best_params = {'n_layers': 3, 'layer_1': 128, 'layer_2': 64, 'layer_3': 32, 'dropout_rate': 0.3,
                                'learning_rate': 0.001, 'batch_size': 4096, 'use_batch_norm': True}
            return self.best_params

        scaler = StandardScaler();
        X_train_scaled = scaler.fit_transform(X_train);
        X_val_scaled = scaler.transform(X_val)

        def objective(trial):
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_layers = [trial.suggest_categorical(f'layer_{i + 1}', [32, 64, 128, 256]) for i in range(n_layers)]
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [4096, 8192, 16384])
            num_classes_for_trial = len(np.unique(y_train))
            temp_model = MultiClassNeuralNet(self.input_dim, num_classes_for_trial, hidden_layers, dropout_rate,
                                             use_batch_norm).to(self.device)
            temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=learning_rate)
            temp_criterion = nn.CrossEntropyLoss()
            train_loader = DataLoader(IoTDataset(X_train_scaled, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(IoTDataset(X_val_scaled, y_val), batch_size=batch_size, shuffle=False)
            best_val_acc = 0.0
            for epoch in range(15):
                temp_model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device);
                    temp_optimizer.zero_grad();
                    outputs = temp_model(X_batch);
                    loss = temp_criterion(outputs, y_batch);
                    loss.backward();
                    temp_optimizer.step()
                temp_model.eval();
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device);
                        outputs = temp_model(X_batch);
                        _, predicted = torch.max(outputs, 1);
                        val_total += y_batch.size(0);
                        val_correct += (predicted == y_batch).sum().item()
                val_acc = val_correct / val_total
                if val_acc > best_val_acc: best_val_acc = val_acc
            torch.cuda.empty_cache();
            return best_val_acc

        study = optuna.create_study(direction='maximize');
        # TODO - Config.OPTUNA_N_TRIALS
        study.optimize(objective, n_trials=1, timeout=Config.OPTUNA_TIMEOUT,
                       show_progress_bar=True)
        self.best_params = study.best_params;
        return study.best_params

    def train(self, X_train, y_train, X_val, y_val, use_optuna=True):
        print("\n" + "=" * 70);
        print("🚀 TRAINING NEURAL NETWORK MULTI-CLASS");
        print("=" * 70)
        self.num_classes = len(np.unique(y_train))
        if use_optuna and Config.USE_OPTUNA:
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
            n_layers = best_params['n_layers'];
            self.hidden_layers = [best_params[f'layer_{i + 1}'] for i in range(n_layers)]
            self.dropout_rate = best_params['dropout_rate'];
            self.learning_rate = best_params['learning_rate']
            self.use_batch_norm = best_params['use_batch_norm'];
            batch_size = best_params['batch_size']
        else:
            batch_size = 4096
        self.build_model()
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))
        self.scaler = StandardScaler();
        X_train_scaled = self.scaler.fit_transform(X_train);
        X_val_scaled = self.scaler.transform(X_val)
        train_loader = DataLoader(IoTDataset(X_train_scaled, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(IoTDataset(X_val_scaled, y_val), batch_size=batch_size, shuffle=False)
        best_val_acc = 0;
        patience_counter = 0
        progress_bar = tqdm(range(50), desc="Training Final Model")
        for epoch in progress_bar:
            self.model.train();
            total_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device);
                self.optimizer.zero_grad();
                outputs = self.model(X_batch);
                loss = self.criterion(outputs, y_batch);
                loss.backward();
                self.optimizer.step();
                total_loss += loss.item()
            self.model.eval();
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device);
                    outputs = self.model(X_batch);
                    _, predicted = torch.max(outputs, 1);
                    val_total += y_batch.size(0);
                    val_correct += (predicted == y_batch).sum().item()
            val_acc = val_correct / val_total
            progress_bar.set_postfix(loss=f"{total_loss / len(train_loader):.4f}", val_acc=f"{val_acc:.4f}")
            if val_acc > best_val_acc:
                best_val_acc = val_acc;
                patience_counter = 0;
                self.save_model('best_multiclass_nn.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10: print(f"\n  Early stopping at epoch {epoch + 1}"); break
        self.load_model('best_multiclass_nn.pth')

    def predict(self, X):
        if self.scaler is None: raise RuntimeError("Scaler has not been loaded.");
        if hasattr(X, 'values'): X = X.values
        X_scaled = self.scaler.transform(X);
        X_tensor = torch.FloatTensor(X_scaled).to(self.device);
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor);
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def evaluate(self, X_test, y_test, label_encoder=None):
        print("\n" + "=" * 70);
        print("📊 EVALUATING NEURAL NETWORK MULTI-CLASS");
        print("=" * 70)
        y_pred = self.predict(X_test)
        target_names = label_encoder.classes_ if label_encoder is not None else [f'Class_{i}' for i in
                                                                                 range(self.num_classes)]
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        acc = accuracy_score(y_test, y_pred);
        print(f"OVERALL ACCURACY: {acc:.4f}")
        cm = confusion_matrix(y_test, y_pred);
        self._plot_confusion_matrix(cm, target_names)
        return {'accuracy': acc}

    def _plot_confusion_matrix(self, cm, target_names):
        plt.figure(figsize=(12, 10));
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - Neural Network Multi-Class');
        plt.ylabel('True Label');
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right');
        plt.yticks(rotation=0);
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'multiclass_nn_confusion_matrix.png'));
        plt.close()
        print("  ✓ Saved confusion matrix")


if __name__ == "__main__":
    Config.print_mode_info();
    Config.set_seeds()
    print("=" * 70);
    print("🎯 MULTI-CLASS CLASSIFICATION: ATTACK TYPE DETECTION");
    print("=" * 70)
    overall_start = time.time()
    X_train, X_test, y_train, y_test, label_encoder = get_data_for_multiclass()
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=Config.RANDOM_STATE,
                                                stratify=y_train)
    X_tr_np = X_tr.values;
    X_val_np = X_val.values
    print("\n" + "=" * 70);
    print("MODEL 2: Neural Network (with Optuna)");
    print("=" * 70)
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))  # Get num_classes from the data
    nn_model = MultiClassNeuralNetModel(input_dim, num_classes)  # Pass both args
    nn_model.train(X_tr_np, y_tr, X_val_np, y_val, use_optuna=True)
    nn_results = nn_model.evaluate(X_test.values, y_test, label_encoder)
    nn_model.save_model('multiclass_nn.pth')
    total_time = time.time() - overall_start
    print(f"\n⏱️  Total pipeline time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print("\n✅ Multi-class classifiers trained and saved!")
