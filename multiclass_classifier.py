"""
Multi-Class Classification: Attack Type Detection
Uses XGBoost and PyTorch Neural Network with SMOTE and Optuna optimization.
UPDATED: Added detailed timing, SMOTE for imbalance, and Optuna tuning.
"""

import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
# Optuna imports
import optuna
import seaborn as sns
import torch
import torch.nn as nn
import xgboost as xgb
# SMOTE imports
from imblearn.over_sampling import SMOTE
from optuna.samplers import TPESampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader
from tqdm import tqdm

from config import Config
from data_loader import get_data_for_multiclass, IoTDataset


class MultiClassXGBoost:
    """Multi-class classification using XGBoost"""

    def __init__(self, params=None):
        self.model = None
        self.num_classes = None
        self.params = params
        self.best_params = None
        print("Multi-Class XGBoost initialized")
        print(f"  Using Optuna: {Config.USE_OPTUNA}")
        print(f"  Using SMOTE: {Config.USE_SMOTE}")

    # ===== THIS IS THE ONLY FUNCTION THAT IS UPDATED =====
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE (oversampling only) for class imbalance."""
        print("\n" + "=" * 70)
        print("🔄 HANDLING CLASS IMBALANCE (Multi-Class)")
        print("=" * 70)
        t_start = time.time()
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n📊 Original class distribution:")
        for cls, count in zip(unique, counts):
            print(f"   Class {cls}: {count:,} ({count / len(y_train) * 100:.2f}%)")

        if not Config.USE_SMOTE or len(X_train) > Config.SMOTE_SAMPLE_THRESHOLD:
            if not Config.USE_SMOTE:
                print(f"\n⏭️  SMOTE disabled in config")
            else:
                print(f"\n⏭️  Dataset too large ({len(X_train):,}), using class weights instead")
            t_elapsed = time.time() - t_start
            print(f"⏱️  Time: {t_elapsed:.2f}s")
            return X_train, y_train

        print(f"\n⏳ Applying SMOTE (Oversampling only)...")
        try:
            # Set a target sample count for the minority classes.
            # A good heuristic is a fraction of the second-largest class, or a fixed minimum.
            sorted_counts = sorted(counts, reverse=True)
            # Ensure there are at least two classes to get sorted_counts[1]
            if len(sorted_counts) > 1:
                target_sampling_count = max(500,
                                            int(sorted_counts[1] * 0.2))  # At least 500, or 20% of 2nd largest class
            else:  # Handle case with only one class (though unlikely)
                target_sampling_count = sorted_counts[0]

            sampling_strategy = {}
            for cls, count in zip(unique, counts):
                if count < target_sampling_count:
                    sampling_strategy[cls] = target_sampling_count

            if not sampling_strategy:
                print("   ✓ No classes needed oversampling. Continuing with original data.")
                t_elapsed = time.time() - t_start
                print(f"⏱️  Time: {t_elapsed:.2f}s")
                return X_train, y_train

            print(f"   Targeting {target_sampling_count:,} samples for minority classes.")

            minority_count = min(counts)
            k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=Config.RANDOM_STATE,
                          k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            t_elapsed = time.time() - t_start
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            print(f"\n📊 Resampled class distribution:")
            for cls, count in zip(unique_new, counts_new):
                print(f"   Class {cls}: {count:,} ({count / len(y_resampled) * 100:.2f}%)")
            print(f"\n✅ Resampling complete")
            print(f"   Original: {len(X_train):,} → Resampled: {len(X_resampled):,}")
            print(f"⏱️  Time: {t_elapsed:.2f}s")
            return X_resampled, y_resampled

        except Exception as e:
            print(f"\n❌ SMOTE failed: {e}")
            print(f"   Continuing with original data + class weights")
            t_elapsed = time.time() - t_start
            print(f"⏱️  Time: {t_elapsed:.2f}s")
            return X_train, y_train

    # =================================================================

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        """Use Optuna to find best hyperparameters."""
        print("\n" + "=" * 70)
        print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (XGBoost)")
        print("=" * 70)
        t_start = time.time()
        if not Config.USE_OPTUNA:
            print("\n⏭️  Optuna disabled in config, using default parameters")
            self.best_params = self.params if self.params else self._get_default_params()
            return self.best_params
        print(f"\n⏳ Running Optuna optimization...")
        print(f"   Trials: {Config.OPTUNA_N_TRIALS}")
        print(f"   Timeout: {Config.OPTUNA_TIMEOUT}s ({Config.OPTUNA_TIMEOUT / 60:.1f} min)")

        def objective(trial):
            param = {'objective': 'multi:softmax', 'num_class': self.num_classes, 'tree_method': 'hist',
                     'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                     'max_depth': trial.suggest_int('max_depth', 3, 15),
                     'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
                     'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                     'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
                     'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
                     'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True), 'n_jobs': Config.N_JOBS,
                     'random_state': Config.RANDOM_STATE, 'eval_metric': 'mlogloss'}
            model = xgb.XGBClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
            y_pred = model.predict(X_val)
            return accuracy_score(y_val, y_pred)

        sampler = TPESampler(seed=Config.RANDOM_STATE)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=Config.OPTUNA_N_TRIALS, timeout=Config.OPTUNA_TIMEOUT,
                       show_progress_bar=True)
        t_elapsed = time.time() - t_start
        print(f"\n✅ Optimization complete")
        print(f"⏱️  Time: {t_elapsed:.2f}s ({t_elapsed / 60:.1f} min)")
        print(f"\n📊 Best trial:\n   Accuracy: {study.best_value:.4f}\n   Parameters:")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")
        base_params = {'objective': 'multi:softmax', 'num_class': self.num_classes, 'tree_method': 'hist',
                       'n_jobs': Config.N_JOBS, 'random_state': Config.RANDOM_STATE, 'eval_metric': 'mlogloss'}
        self.best_params = {**base_params, **study.best_params}
        return self.best_params

    def _get_default_params(self):
        """Get default XGBoost parameters"""
        return {'objective': 'multi:softmax', 'num_class': self.num_classes, 'tree_method': 'hist',
                'n_estimators': Config.MULTICLASS_N_ESTIMATORS, 'learning_rate': Config.MULTICLASS_LEARNING_RATE,
                'max_depth': Config.MULTICLASS_MAX_DEPTH, 'n_jobs': Config.N_JOBS, 'random_state': Config.RANDOM_STATE,
                'eval_metric': 'mlogloss'}

    def train(self, X_train, y_train, X_val=None, y_val=None, use_smote=True, use_optuna=True):
        """Train multi-class XGBoost classifier"""
        print("\n" + "=" * 70);
        print("🚀 TRAINING MULTI-CLASS XGBOOST");
        print("=" * 70)
        overall_start = time.time()
        self.num_classes = len(np.unique(np.concatenate((y_train, y_test))))
        print(
            f"\n📊 Training data:\n   Samples: {len(X_train):,}\n   Features: {X_train.shape[1]}\n   Classes: {self.num_classes}")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"     Class {cls}: {count:,} samples")
        if use_smote and Config.USE_SMOTE:
            X_train, y_train = self.apply_smote(X_train, y_train)
        if use_optuna and Config.USE_OPTUNA and X_val is not None:
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            best_params = self._get_default_params()
            print("\n⏭️  Using default parameters (no optimization)")
        print("\n" + "=" * 70);
        print("🎯 TRAINING FINAL MODEL");
        print("=" * 70)
        t_train_start = time.time()
        if len(np.unique(y_train)) > 1:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            sample_weights = class_weights[y_train]
            print(f"\n⚖️  Using class weights for imbalance")
        else:
            sample_weights = None
        print(f"\n⏳ Training XGBoost...");
        self.model = xgb.XGBClassifier(**best_params)
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        t_train = time.time() - t_train_start;
        total_time = time.time() - overall_start
        print(
            f"\n✅ Training complete\n⏱️  Training time: {t_train:.2f}s\n⏱️  Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, label_encoder=None):
        print("\n" + "=" * 70);
        print("📊 EVALUATING MULTI-CLASS XGBOOST");
        print("=" * 70)
        t_eval_start = time.time()
        print(f"\n⏳ Generating predictions...");
        t_pred_start = time.time()
        y_pred = self.predict(X_test);
        t_pred = time.time() - t_pred_start
        print(
            f"✓ Predictions complete ({t_pred:.2f}s)\n   Throughput: {len(X_test) / (t_pred if t_pred > 0 else 1):.0f} samples/sec")
        target_names = label_encoder.classes_ if label_encoder else [f'Class_{i}' for i in range(self.num_classes)]
        print("\n" + "─" * 70);
        print("CLASSIFICATION REPORT");
        print("─" * 70);
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{'─' * 70}\nOVERALL ACCURACY: {acc:.4f}\n{'─' * 70}")
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, target_names)
        t_eval = time.time() - t_eval_start;
        print(f"\n⏱️  Total evaluation time: {t_eval:.2f}s")
        return {'accuracy': acc}

    def _plot_confusion_matrix(self, cm, target_names):
        plt.figure(figsize=(12, 10));
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - XGBoost Multi-Class');
        plt.ylabel('True Label');
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right');
        plt.yticks(rotation=0);
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'multiclass_xgboost_confusion_matrix.png'), dpi=300);
        plt.close();
        print("  ✓ Saved confusion matrix")

    def save_model(self, filename='multiclass_xgboost.pkl'):
        filepath = os.path.join(Config.MODELS_DIR, filename)
        joblib.dump({'model': self.model, 'params': self.best_params if self.best_params else self.params}, filepath)
        print(f"\n💾 Model saved: {filename}")


class MultiClassNeuralNet(nn.Module):
    """Neural network for multi-class classification"""

    def __init__(self, input_dim, num_classes):
        super(MultiClassNeuralNet, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.BatchNorm1d(64), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.BatchNorm1d(32), nn.Dropout(0.2),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.network(x)


class MultiClassNeuralNetModel:
    """Neural network model wrapper"""

    def __init__(self, input_dim, num_classes):
        self.device = Config.DEVICE
        self.model = MultiClassNeuralNet(input_dim, num_classes).to(self.device)
        self.num_classes = num_classes
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.scaler = None
        print(f"Multi-Class Neural Net initialized on {self.device}")
        print(f"  Input dim: {input_dim}, Classes: {num_classes}")

    def train(self, X_train, y_train, X_val, y_val):
        """Train neural network"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING NEURAL NETWORK MULTI-CLASS")
        print("=" * 70)

        overall_start = time.time()

        # Calculate class weights
        print(f"\n⏳ Calculating class weights...")
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        self.criterion = nn.CrossEntropyLoss(weight=torch.FloatTensor(class_weights).to(self.device))
        print(f"✓ Class weights calculated")

        # Scale data
        print(f"\n⏳ Scaling features...")
        t_scale_start = time.time()
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        t_scale = time.time() - t_scale_start
        print(f"✓ Scaling complete ({t_scale:.2f}s)")

        # Create data loaders
        print(f"\n⏳ Creating DataLoaders...")
        train_loader = DataLoader(IoTDataset(X_train_scaled, y_train), batch_size=4096, shuffle=True)
        val_loader = DataLoader(IoTDataset(X_val_scaled, y_val), batch_size=4096, shuffle=False)
        print(f"✓ DataLoaders created")

        # Training loop
        print(f"\n⏳ Training for 50 epochs with early stopping...")
        t_train_start = time.time()

        best_val_acc = 0
        patience_counter = 0

        for epoch in tqdm(range(50), desc="Training"):
            # Train
            self.model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()

            # Validate
            self.model.eval()
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                    outputs = self.model(X_batch)
                    _, predicted = torch.max(outputs, 1)
                    val_total += y_batch.size(0)
                    val_correct += (predicted == y_batch).sum().item()

            val_acc = val_correct / val_total

            # Early stopping
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_model('best_multiclass_nn.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"\n  Early stopping at epoch {epoch + 1}")
                    break

        # Load best model
        self.load_model('best_multiclass_nn.pth')

        t_train = time.time() - t_train_start
        total_time = time.time() - overall_start

        print(f"\n✅ Training complete")
        print(f"   Best validation accuracy: {best_val_acc:.4f}")
        print(f"⏱️  Training time: {t_train:.2f}s")
        print(f"⏱️  Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")

    def predict(self, X):
        """Predict class labels"""
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
        """Evaluate model performance"""
        print("\n" + "=" * 70)
        print("📊 EVALUATING NEURAL NETWORK MULTI-CLASS")
        print("=" * 70)

        t_eval_start = time.time()

        print(f"\n⏳ Generating predictions...")
        t_pred_start = time.time()
        y_pred = self.predict(X_test)
        t_pred = time.time() - t_pred_start
        print(f"✓ Predictions complete ({t_pred:.2f}s)")
        print(f"   Throughput: {len(X_test) / t_pred:.0f} samples/sec")

        target_names = label_encoder.classes_ if label_encoder else [f'Class_{i}' for i in range(self.num_classes)]

        print("\n" + "─" * 70)
        print("CLASSIFICATION REPORT")
        print("─" * 70)
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))

        acc = accuracy_score(y_test, y_pred)
        print(f"\n{'─' * 70}")
        print(f"OVERALL ACCURACY: {acc:.4f}")
        print('─' * 70)

        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, target_names)

        t_eval = time.time() - t_eval_start
        print(f"\n⏱️  Total evaluation time: {t_eval:.2f}s")

        return {'accuracy': acc}

    def _plot_confusion_matrix(self, cm, target_names):
        """Plot confusion matrix"""
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - Neural Network Multi-Class')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'multiclass_nn_confusion_matrix.png'), dpi=300)
        plt.close()
        print("  ✓ Saved confusion matrix")

    def save_model(self, filename='multiclass_nn.pth'):
        """Save model"""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        torch.save({
            'model_state': self.model.state_dict(),
            'scaler': self.scaler,
            'num_classes': self.num_classes
        }, filepath)

    def load_model(self, filename='multiclass_nn.pth'):
        """Load model"""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"❌ Model file {filepath} not found")
            return
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state'])
        self.scaler = checkpoint['scaler']
        self.num_classes = checkpoint['num_classes']


if __name__ == "__main__":
    Config.set_seeds()
    Config.print_mode_info()
    
    print("=" * 70)
    print("🎯 MULTI-CLASS CLASSIFICATION: ATTACK TYPE DETECTION")
    print("=" * 70)

    overall_start = time.time()

    # Load data
    X_train, X_test, y_train, y_test, label_encoder = get_data_for_multiclass()

    # Create validation split
    print("\n⏳ Creating validation split...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=Config.RANDOM_STATE, stratify=y_train
    )
    print(f"✓ Split complete")
    print(f"   Train: {len(X_tr):,}")
    print(f"   Val:   {len(X_val):,}")
    print(f"   Test:  {len(X_test):,}")

    # Convert to numpy for processing
    X_tr_np = X_tr.values
    X_val_np = X_val.values

    # Train XGBoost
    print("\n" + "=" * 70)
    print("MODEL 1: XGBoost")
    print("=" * 70)
    xgb_model = MultiClassXGBoost()
    xgb_model.train(X_train, y_train, X_val, y_val, use_smote=True, use_optuna=True)
    xgb_results = xgb_model.evaluate(X_test, y_test, label_encoder)
    xgb_model.save_model()

    # Train Neural Network
    print("\n" + "=" * 70)
    print("MODEL 2: Neural Network")
    print("=" * 70)
    input_dim = X_train.shape[1]
    num_classes = len(np.unique(y_train))
    nn_model = MultiClassNeuralNetModel(input_dim, num_classes)
    nn_model.train(X_tr_np, y_tr, X_val_np, y_val)
    nn_results = nn_model.evaluate(X_test.values, y_test, label_encoder)
    nn_model.save_model('multiclass_nn.pth')

    # Final summary
    total_time = time.time() - overall_start

    print("\n" + "=" * 70)
    print("🎉 MULTI-CLASS CLASSIFICATION COMPLETE")
    print("=" * 70)
    print(f"\n📊 Model Comparison:")
    print(f"   XGBoost Accuracy:        {xgb_results['accuracy']:.4f}")
    print(f"   Neural Network Accuracy: {nn_results['accuracy']:.4f}")
    print(f"\n⏱️  Total pipeline time: {total_time:.2f}s ({total_time / 60:.1f} min)")
    print("\n✅ Multi-class classifiers trained and saved!")
