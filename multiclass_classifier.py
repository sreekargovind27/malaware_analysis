"""
Multi-Class Classification: Attack Type Detection
Uses XGBoost and PyTorch Neural Network with SMOTE and Optuna optimization.
UPDATED: Final, cleaned version.
"""

import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
import torch
import torch.nn as nn
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_loader import get_data_for_multiclass, IoTDataset


# ============================================================
# XGBoost Model
# ============================================================
class MultiClassXGBoost:
    """Multi-class classification using XGBoost with Optuna support and NN-like API."""

    def __init__(self, params=None):
        self.model = None
        self.num_classes = None
        self.params = params
        self.best_params = None
        self.label_encoder = None  # optional, for pretty reports
        print("Multi-Class XGBoost initialized")

    def _get_default_params(self):
        # Pull from Config when available; otherwise use safe defaults
        n_estimators = getattr(Config, "MULTICLASS_N_ESTIMATORS", 400)
        learning_rate = getattr(Config, "MULTICLASS_LEARNING_RATE", 0.1)
        max_depth = getattr(Config, "MULTICLASS_MAX_DEPTH", 8)
        n_jobs = getattr(Config, "N_JOBS", -1)
        random_state = getattr(Config, "RANDOM_STATE", 42)
        tree_method = getattr(Config, "XGB_TREE_METHOD", "hist")  # "gpu_hist" if you have GPU

        return {
            "objective": "multi:softmax",
            "num_class": self.num_classes,
            "tree_method": tree_method,
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "n_jobs": n_jobs,
            "random_state": random_state,
            "eval_metric": "mlogloss",
        }

    def apply_smote(self, X_train, y_train):
        # Placeholder: pipeline already undersamples in data_loader
        print("\n" + "=" * 70)
        print("🔄 HANDLING CLASS IMBALANCE (Multi-Class) [XGBoost]")
        print("=" * 70)
        return X_train, y_train

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        print("\n" + "=" * 70)
        print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (XGBoost)")
        print("=" * 70)

        if not getattr(Config, "USE_OPTUNA", False):
            self.best_params = self._get_default_params()
            return self.best_params

        def objective(trial):
            max_depth = trial.suggest_int("max_depth", 4, 12)
            learning_rate = trial.suggest_float("learning_rate", 1e-3, 0.3, log=True)
            n_estimators = trial.suggest_int("n_estimators", 200, 1200, step=100)
            subsample = trial.suggest_float("subsample", 0.6, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.6, 1.0)
            reg_lambda = trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-3, 2.0, log=True)
            tree_method = getattr(Config, "XGB_TREE_METHOD", "hist")

            params = {
                "objective": "multi:softmax",
                "num_class": self.num_classes,
                "tree_method": tree_method,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "n_estimators": n_estimators,
                "subsample": subsample,
                "colsample_bytree": colsample_bytree,
                "reg_lambda": reg_lambda,
                "reg_alpha": reg_alpha,
                "random_state": getattr(Config, "RANDOM_STATE", 42),
                "n_jobs": getattr(Config, "N_JOBS", -1),
                "eval_metric": "mlogloss",
            }

            # NOTE: No early stopping here (old xgboost sklearn API)
            model = xgb.XGBClassifier(**params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
            preds = model.predict(X_val)
            return accuracy_score(y_val, preds)

        study = optuna.create_study(direction="maximize")
        n_trials = getattr(Config, "OPTUNA_N_TRIALS", 1)
        timeout = getattr(Config, "OPTUNA_TIMEOUT", None)
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)

        best = self._get_default_params()
        best.update(study.best_params)
        # Ensure required keys present
        best["objective"] = "multi:softmax"
        best["num_class"] = self.num_classes
        best["tree_method"] = getattr(Config, "XGB_TREE_METHOD", "hist")
        best["n_jobs"] = getattr(Config, "N_JOBS", -1)
        best["random_state"] = getattr(Config, "RANDOM_STATE", 42)
        best["eval_metric"] = "mlogloss"

        self.best_params = best
        return best

    def train(self, X_train, y_train, X_val=None, y_val=None, use_smote=True, use_optuna=True, label_encoder=None):
        print("\n" + "=" * 70)
        print("🚀 TRAINING XGBOOST MULTI-CLASS")
        print("=" * 70)

        self.label_encoder = label_encoder
        self.num_classes = len(np.unique(y_train))

        if use_smote:
            X_train, y_train = self.apply_smote(X_train, y_train)

        if use_optuna and getattr(Config, "USE_OPTUNA", False) and X_val is not None and y_val is not None:
            params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            params = self._get_default_params()
            self.best_params = params

        self.model = xgb.XGBClassifier(**params)

        # NOTE: No early stopping (keep compatibility with older XGBoost)
        eval_set = [(X_val, y_val)] if (X_val is not None and y_val is not None) else None
        self.model.fit(
            X_train, y_train,
            eval_set=eval_set,
            verbose=False,
        )
        print("✅ XGBoost training complete.")

    def predict(self, X):
        if self.model is None:
            raise RuntimeError("XGBoost model not trained.")
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, label_encoder=None):
        print("\n" + "=" * 70)
        print("📊 EVALUATING XGBOOST MULTI-CLASS")
        print("=" * 70)

        y_pred = self.predict(X_test)
        if label_encoder is None:
            label_encoder = self.label_encoder

        target_names = (
            label_encoder.classes_.tolist()
            if label_encoder is not None
            else [f"Class_{i}" for i in range(self.num_classes)]
        )

        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        acc = accuracy_score(y_test, y_pred)
        print(f"OVERALL ACCURACY: {acc:.4f}")

        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, target_names)
        return {"accuracy": acc}

    def _plot_confusion_matrix(self, cm, target_names):
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=target_names, yticklabels=target_names)
        plt.title("Confusion Matrix - XGBoost Multi-Class")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        outpath = os.path.join(Config.RESULTS_DIR, "multiclass_xgb_confusion_matrix.png")
        plt.savefig(outpath)
        plt.close()
        print(f"  ✓ Saved confusion matrix -> {outpath}")

    def save_model(self, filename="multiclass_xgboost.pkl"):
        if self.model is None:
            raise RuntimeError("XGBoost model not trained; cannot save.")
        filepath = os.path.join(Config.MODELS_DIR, filename)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        joblib.dump(
            {"model": self.model, "num_classes": self.num_classes, "best_params": self.best_params},
            filepath
        )
        print(f"💾 Saved XGBoost model to {filepath}")

    def load_model(self, filename="multiclass_xgboost.pkl"):
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Model file {filepath} not found")
        payload = joblib.load(filepath)
        self.model = payload["model"]
        self.num_classes = payload["num_classes"]
        self.best_params = payload.get("best_params")
        print("✅ XGBoost model loaded.")


# ============================================================
# Neural Network Model
# ============================================================
class MultiClassNeuralNet(nn.Module):
    """Flexible neural network for multi-class classification"""

    def __init__(self, input_dim, num_classes, hidden_layers=None, dropout_rate=0.3, use_batch_norm=True):
        super(MultiClassNeuralNet, self).__init__()
        if hidden_layers is None:
            hidden_layers = [128, 64, 32]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class MultiClassNeuralNetModel:
    """Neural network model wrapper with robust saving/loading and training."""

    def __init__(self, input_dim):
        self.device = Config.DEVICE
        self.input_dim = input_dim
        self.num_classes = None  # Will be set during training/loading
        self.hidden_layers = [128, 64, 32]
        self.dropout_rate = 0.3
        self.learning_rate = 0.001
        self.use_batch_norm = True
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()
        self.scaler = None
        self.best_params = None
        print(f"Multi-Class Neural Net initialized on {self.device}")

    def build_model(self):
        """Builds or rebuilds the neural network based on current attributes."""
        if self.num_classes is None:
            raise ValueError("num_classes is not set. Cannot build model.")
        print(f"\n🏗️  Building model with architecture: {self.hidden_layers}")
        self.model = MultiClassNeuralNet(
            self.input_dim, self.num_classes, self.hidden_layers, self.dropout_rate, self.use_batch_norm
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def load_model(self, filename='multiclass_nn.pth'):
        """Load model, correctly rebuilding the architecture from the file first."""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Model file {filepath} not found")

        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        print("💾 Reading architecture from checkpoint...")
        self.num_classes = checkpoint['num_classes']
        self.hidden_layers = checkpoint['hidden_layers']
        self.dropout_rate = checkpoint['dropout_rate']
        self.use_batch_norm = checkpoint['use_batch_norm']
        self.learning_rate = checkpoint['learning_rate']
        self.best_params = checkpoint.get('best_params')
        self.scaler = checkpoint['scaler']

        self.build_model()
        self.model.load_state_dict(checkpoint['model_state'])
        print("✅ Model state loaded successfully into matching architecture.")

    def save_model(self, filename='multiclass_nn.pth'):
        """Save model AND its architecture."""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        os.makedirs(Config.MODELS_DIR, exist_ok=True)
        if self.model is None:
            raise RuntimeError("Model has not been built yet. Cannot save.")
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'num_classes': self.num_classes,
            'hidden_layers': self.hidden_layers,
            'dropout_rate': self.dropout_rate,
            'learning_rate': self.learning_rate,
            'use_batch_norm': self.use_batch_norm,
            'best_params': self.best_params
        }, filepath)

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Use Optuna to find best hyperparameters for Neural Network"""
        print("\n" + "=" * 70)
        print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (Neural Network)")
        print("=" * 70)
        if not Config.USE_OPTUNA:
            self.best_params = {
                'n_layers': 3, 'layer_1': 128, 'layer_2': 64, 'layer_3': 32,
                'dropout_rate': 0.3, 'learning_rate': 0.001, 'batch_size': 4096, 'use_batch_norm': True
            }
            return self.best_params

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        def objective(trial):
            print(f"\n--- Starting Optuna Trial #{trial.number} ---")
            n_layers = trial.suggest_int('n_layers', 2, 4)
            hidden_layers = [trial.suggest_categorical(f'layer_{i + 1}', [32, 64, 128, 256]) for i in range(n_layers)]
            dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
            use_batch_norm = trial.suggest_categorical('use_batch_norm', [True, False])
            learning_rate = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)
            batch_size = trial.suggest_categorical('batch_size', [4096, 8192, 16384])
            print(f"  > Params: Layers={hidden_layers}, Batch={batch_size}, LR={learning_rate:.5f}")

            num_classes_for_trial = len(np.unique(y_train))
            temp_model = MultiClassNeuralNet(
                self.input_dim, num_classes_for_trial, hidden_layers, dropout_rate, use_batch_norm
            ).to(self.device)
            temp_optimizer = torch.optim.Adam(temp_model.parameters(), lr=learning_rate)
            temp_criterion = nn.CrossEntropyLoss()

            train_loader = DataLoader(IoTDataset(X_train_scaled, y_train), batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(IoTDataset(X_val_scaled, y_val), batch_size=batch_size, shuffle=False)

            best_val_acc = 0.0
            for _ in tqdm(range(15), desc=f"  > Trial #{trial.number}", leave=False):  # fewer epochs for trials
                temp_model.train()
                for X_batch, y_batch in train_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    temp_optimizer.zero_grad()
                    outputs = temp_model(X_batch)
                    loss = temp_criterion(outputs, y_batch)
                    loss.backward()
                    temp_optimizer.step()

                temp_model.eval()
                val_correct, val_total = 0, 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                        outputs = temp_model(X_batch)
                        _, predicted = torch.max(outputs, 1)
                        val_total += y_batch.size(0)
                        val_correct += (predicted == y_batch).sum().item()
                val_acc = val_correct / val_total
                if val_acc > best_val_acc:
                    best_val_acc = val_acc

            print(f"  > Trial #{trial.number} finished. Best Val Acc: {best_val_acc:.4f}")
            torch.cuda.empty_cache()
            return best_val_acc

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=1, timeout=Config.OPTUNA_TIMEOUT, show_progress_bar=True)

        self.best_params = study.best_params
        return study.best_params

    def train(self, X_train, y_train, X_val, y_val, use_optuna=True):
        print("\n" + "=" * 70)
        print("🚀 TRAINING NEURAL NETWORK MULTI-CLASS")
        print("=" * 70)
        self.num_classes = len(np.unique(y_train))

        if use_optuna and Config.USE_OPTUNA:
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
            n_layers = best_params['n_layers']
            self.hidden_layers = [best_params[f'layer_{i + 1}'] for i in range(n_layers)]
            self.dropout_rate = best_params['dropout_rate']
            self.learning_rate = best_params['learning_rate']
            self.use_batch_norm = best_params['use_batch_norm']
            batch_size = best_params['batch_size']
        else:
            batch_size = 4096

        self.build_model()
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)

        train_loader = DataLoader(IoTDataset(X_train_scaled, y_train), batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(IoTDataset(X_val_scaled, y_val), batch_size=batch_size, shuffle=False)

        best_val_acc = 0.0
        patience_counter = 0
        progress_bar = tqdm(range(50), desc="Training Final Model")
        for epoch in progress_bar:
            self.model.train()
            total_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            self.model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()
            val_acc = val_correct / val_total

            progress_bar.set_postfix(loss=f"{total_loss / max(1, len(train_loader)):.4f}", val_acc=f"{val_acc:.4f}")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model('best_multiclass_nn.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                    break

        self.load_model('best_multiclass_nn.pth')

    def predict(self, X):
        if self.scaler is None:
            raise RuntimeError("Scaler has not been loaded.")
        if hasattr(X, 'values'):
            X = X.values
        X_scaled = self.scaler.transform(X)
        X_tensor = torch.FloatTensor(X_scaled).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            _, predicted = torch.max(outputs, 1)
        return predicted.cpu().numpy()

    def evaluate(self, X_test, y_test, label_encoder=None):
        print("\n" + "=" * 70)
        print("📊 EVALUATING NEURAL NETWORK MULTI-CLASS")
        print("=" * 70)
        y_pred = self.predict(X_test)
        target_names = label_encoder.classes_ if label_encoder is not None else [f'Class_{i}' for i in
                                                                                 range(self.num_classes)]
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        acc = accuracy_score(y_test, y_pred)
        print(f"OVERALL ACCURACY: {acc:.4f}")
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, target_names)
        return {'accuracy': acc}

    def _plot_confusion_matrix(self, cm, target_names):
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - Neural Network Multi-Class')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        os.makedirs(Config.RESULTS_DIR, exist_ok=True)
        outpath = os.path.join(Config.RESULTS_DIR, 'multiclass_nn_confusion_matrix.png')
        plt.savefig(outpath)
        plt.close()
        print("  ✓ Saved confusion matrix")


# ============================================================
# Main
# ============================================================
if __name__ == "__main__":
    Config.print_mode_info()
    Config.set_seeds()

    print("=" * 70)
    print("🎯 MULTI-CLASS CLASSIFICATION: ATTACK TYPE DETECTION")
    print("=" * 70)

    overall_start = time.time()

    # Load data and split
    X_train, X_test, y_train, y_test, label_encoder = get_data_for_multiclass()
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=Config.RANDOM_STATE, stratify=y_train
    )
    X_tr_np = X_tr.values
    X_val_np = X_val.values

    # --------------------------------------------------------------
    # MODEL 1: XGBoost
    # --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL 1: XGBoost")
    print("=" * 70)

    xgb_model = MultiClassXGBoost()
    xgb_model.train(X_tr_np, y_tr, X_val_np, y_val, use_smote=True, use_optuna=True, label_encoder=label_encoder)
    xgb_results = xgb_model.evaluate(X_test.values, y_test, label_encoder)
    xgb_model.save_model("multiclass_xgboost.pkl")

    # --------------------------------------------------------------
    # MODEL 2: Neural Network (with Optuna)
    # --------------------------------------------------------------
    print("\n" + "=" * 70)
    print("MODEL 2: Neural Network (with Optuna)")
    print("=" * 70)

    input_dim = X_train.shape[1]
    nn_model = MultiClassNeuralNetModel(input_dim)
    nn_model.train(X_tr_np, y_tr, X_val_np, y_val, use_optuna=True)
    nn_results = nn_model.evaluate(X_test.values, y_test, label_encoder)
    nn_model.save_model('multiclass_nn.pth')

    total_time = time.time() - overall_start
    print(f"\n⏱️  Total pipeline time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print("\n✅ Multi-class classifiers trained and saved!")
