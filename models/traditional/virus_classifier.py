"""
This script trains and evaluates a model for classifying malware into specific families.
It utilizes LightGBM with SMOTE for handling class imbalance and Optuna for hyperparameter optimization.
"""

import os
import time

import joblib
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import optuna
import seaborn as sns
from imblearn.over_sampling import SMOTE
from optuna.samplers import TPESampler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from config import Config
from models.data_loader import get_data_for_virus


class VirusClassifier:
    """A LightGBM-based classifier for identifying malware families."""

    def __init__(self, params=None):
        self.model = None
        self.num_classes = None
        self.params = params
        self.best_params = None
        print("Virus/Malware Family Classifier initialized")
        print(f"  Using Optuna: {Config.USE_OPTUNA}")
        print(f"  Using SMOTE: {Config.USE_SMOTE}")

    def apply_smote(self, X_train, y_train):
        """Applies SMOTE to handle class imbalance among malware families."""
        print("\n" + "=" * 70)
        print("🔄 HANDLING CLASS IMBALANCE (Virus Families)")
        print("=" * 70)
        t_start = time.time()
        unique, counts = np.unique(y_train, return_counts=True)
        print(f"\n📊 Original class distribution:")
        for cls, count in zip(unique, counts):
            print(f"   Family {cls}: {count:,} ({count / len(y_train) * 100:.2f}%)")

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
            sorted_counts = sorted(counts, reverse=True)
            if len(sorted_counts) > 1:
                target_sampling_count = max(500, int(sorted_counts[1] * 0.2))
            else:
                target_sampling_count = sorted_counts[0]

            sampling_strategy = {}
            for cls, count in zip(unique, counts):
                if count < target_sampling_count:
                    sampling_strategy[cls] = target_sampling_count

            if not sampling_strategy:
                print("   ✓ No classes needed oversampling. Continuing.")
                t_elapsed = time.time() - t_start
                print(f"⏱️  Time: {t_elapsed:.2f}s")
                return X_train, y_train

            print(f"   Targeting {target_sampling_count:,} samples for minority families.")

            minority_count = min(counts)
            k_neighbors = min(5, minority_count - 1) if minority_count > 1 else 1

            smote = SMOTE(sampling_strategy=sampling_strategy, random_state=Config.RANDOM_STATE,
                          k_neighbors=k_neighbors)
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            t_elapsed = time.time() - t_start
            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            print(f"\n📊 Resampled class distribution:")
            for cls, count in zip(unique_new, counts_new):
                print(f"   Family {cls}: {count:,} ({count / len(y_resampled) * 100:.2f}%)")
            print(f"\n✅ Resampling complete\n   Original: {len(X_train):,} → Resampled: {len(X_resampled):,}")
            print(f"⏱️  Time: {t_elapsed:.2f}s")
            return X_resampled, y_resampled

        except Exception as e:
            print(f"\n❌ SMOTE failed: {e}\n   Continuing with original data + class weights")
            t_elapsed = time.time() - t_start
            print(f"⏱️  Time: {t_elapsed:.2f}s")
            return X_train, y_train

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        print("\n" + "=" * 70)
        print("🔍 HYPERPARAMETER OPTIMIZATION WITH OPTUNA (Virus)")
        print("=" * 70)
        t_start = time.time()
        if not Config.USE_OPTUNA:
            print("\n⏭️  Optuna disabled in config, using default parameters")
            self.best_params = self.params if self.params else self._get_default_params()
            return self.best_params
        print(f"\n⏳ Running Optuna optimization...")

        def objective(trial):
            param = {'objective': 'multiclass', 'num_class': self.num_classes, 'boosting_type': 'gbdt',
                     'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                     'n_estimators': trial.suggest_int('n_estimators', 50, 300),
                     'max_depth': trial.suggest_int('max_depth', 3, 15),
                     'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                     'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'n_jobs': Config.N_JOBS,
                     'random_state': Config.RANDOM_STATE, 'metric': 'multi_logloss', 'verbose': -1}
            model = lgb.LGBMClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
            return accuracy_score(y_val, model.predict(X_val))

        study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=Config.RANDOM_STATE))
        study.optimize(objective, n_trials=Config.OPTUNA_N_TRIALS, timeout=Config.OPTUNA_TIMEOUT,
                       show_progress_bar=True)
        t_elapsed = time.time() - t_start
        print(
            f"\n✅ Optimization complete\n⏱️  Time: {t_elapsed:.2f}s\n\n📊 Best trial:\n   Accuracy: {study.best_value:.4f}\n   Parameters:")
        for key, value in study.best_params.items():
            print(f"      {key}: {value}")
        base_params = {'objective': 'multiclass', 'num_class': self.num_classes, 'boosting_type': 'gbdt',
                       'n_jobs': Config.N_JOBS, 'random_state': Config.RANDOM_STATE, 'metric': 'multi_logloss',
                       'verbose': -1}
        self.best_params = {**base_params, **study.best_params}
        return self.best_params

    def _get_default_params(self):
        return {'objective': 'multiclass', 'num_class': self.num_classes, 'boosting_type': 'gbdt', 'num_leaves': 50,
                'learning_rate': 0.1, 'n_estimators': 200, 'n_jobs': Config.N_JOBS,
                'random_state': Config.RANDOM_STATE,
                'metric': 'multi_logloss', 'verbose': -1}

    def train(self, X_train, y_train, X_val=None, y_val=None, use_smote=True, use_optuna=True):
        print("\n" + "=" * 70)
        print("🚀 TRAINING VIRUS CLASSIFIER")
        print("=" * 70)
        overall_start = time.time()
        self.num_classes = len(np.unique(y_train))
        print(
            f"\n📊 Training data:\n   Samples: {len(X_train):,}\n   Features: {X_train.shape[1]}\n   Malware families: {self.num_classes}")
        unique, counts = np.unique(y_train, return_counts=True)
        for cls, count in zip(unique, counts):
            print(f"     Family {cls}: {count:,} samples")
        if use_smote and Config.USE_SMOTE:
            X_train, y_train = self.apply_smote(X_train, y_train)
        if use_optuna and Config.USE_OPTUNA and X_val is not None:
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            best_params = self._get_default_params()
            print("\n⏭️  Using default parameters (no optimization)")
        print("\n" + "=" * 70)
        print("🎯 TRAINING FINAL MODEL")
        print("=" * 70)
        t_train_start = time.time()
        if len(np.unique(y_train)) > 1:
            class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
            sample_weights = class_weights[y_train]
            print(f"\n⚖️  Using class weights for imbalance")
        else:
            sample_weights = None
        print(f"\n⏳ Training LightGBM...")
        self.model = lgb.LGBMClassifier(**best_params)
        self.model.fit(X_train, y_train, sample_weight=sample_weights)
        t_train = time.time() - t_train_start
        total_time = time.time() - overall_start
        print(
            f"\n✅ Training complete\n⏱️  Training time: {t_train:.2f}s\n⏱️  Total time: {total_time:.2f}s ({total_time / 60:.1f} min)")

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X_test, y_test, label_encoder):
        print("\n" + "=" * 70)
        print("📊 EVALUATING VIRUS CLASSIFIER")
        print("=" * 70)
        t_eval_start = time.time()
        print(f"\n⏳ Generating predictions...")
        t_pred_start = time.time()
        y_pred = self.predict(X_test)
        t_pred = time.time() - t_pred_start
        print(
            f"✓ Predictions complete ({t_pred:.2f}s)\n   Throughput: {len(X_test) / (t_pred if t_pred > 0 else 1):.0f} samples/sec")
        target_names = label_encoder.classes_
        print("\n" + "─" * 70)
        print("CLASSIFICATION REPORT")
        print("─" * 70)
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        acc = accuracy_score(y_test, y_pred)
        print(f"\n{'─' * 70}\nOVERALL ACCURACY: {acc:.4f}\n{'─' * 70}")
        cm = confusion_matrix(y_test, y_pred, labels=np.arange(len(target_names)))
        self._plot_confusion_matrix(cm, target_names)
        self._plot_feature_importance(feature_names=X_test.columns)
        t_eval = time.time() - t_eval_start
        print(f"\n⏱️  Total evaluation time: {t_eval:.2f}s")
        return {'accuracy': acc}

    def _plot_confusion_matrix(self, cm, target_names):
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', xticklabels=target_names, yticklabels=target_names)
        plt.title('Confusion Matrix - Malware Family Classification')
        plt.ylabel('True Family')
        plt.xlabel('Predicted Family')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'virus_confusion_matrix.png'), dpi=300)
        plt.close()
        print("  ✓ Saved confusion matrix")

    def _plot_feature_importance(self, feature_names):
        importance = self.model.feature_importances_
        indices = np.argsort(importance)[::-1][:20]
        plt.figure(figsize=(12, 8))
        sns.barplot(x=importance[indices], y=[feature_names[i] for i in indices], palette='viridis')
        plt.xlabel('Importance')
        plt.ylabel('Features')
        plt.title('Top 20 Feature Importance - Virus Classification')
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'virus_feature_importance.png'), dpi=300)
        plt.close()
        print("  ✓ Saved feature importance plot")

    def analyze_families(self, X_test, y_test, label_encoder):
        print("\n" + "=" * 70)
        print("🔬 ANALYZING PREDICTION CONFIDENCE PER FAMILY")
        print("=" * 70)
        t_start = time.time()
        y_pred = self.predict(X_test)
        y_proba = self.model.predict_proba(X_test)
        families = label_encoder.classes_
        for i, family in enumerate(families):
            mask = y_test == i
            if not np.any(mask):
                continue
            family_pred = y_pred[mask]
            correct = (family_pred == i).sum()
            total = mask.sum()
            accuracy = correct / total if total > 0 else 0
            avg_confidence = y_proba[mask][family_pred == i, i].mean() if correct > 0 else 0
            print(f"\n📊 {family}:\n   Samples: {total:,}\n   Accuracy: {accuracy:.2%} ({correct}/{total})")
            if correct > 0:
                print(f"   Avg Confidence (when correct): {avg_confidence:.4f}")
        print(f"\n⏱️  Analysis time: {time.time() - t_start:.2f}s")

    def save_model(self, filename='virus_classifier.pkl'):
        filepath = os.path.join(Config.MODELS_DIR, filename)
        joblib.dump({'model': self.model, 'params': self.best_params if self.best_params else self.params}, filepath)
        print(f"\n💾 Model saved: {filename}")


if __name__ == "__main__":
    Config.set_seeds()
    Config.print_mode_info()

    print("=" * 70)
    print("🦠 MALWARE FAMILY CLASSIFICATION")
    print("=" * 70)
    overall_start = time.time()
    try:
        X_train, X_test, y_train, y_test, label_encoder = get_data_for_virus()
        if len(X_train) < 2 or len(np.unique(y_train)) < 2:
            print("\n⚠️  Not enough data or class diversity to train virus classifier. Skipping.")
        else:
            print("\n⏳ Creating validation split...")
            try:
                X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                            random_state=Config.RANDOM_STATE, stratify=y_train)
                print(f"✓ Split complete (stratified)")
            except ValueError:
                print("   ⚠️  Could not stratify. Using random split.")
                X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2,
                                                            random_state=Config.RANDOM_STATE)
                print(f"✓ Split complete (random)")
            print(f"   Train: {len(X_tr):,}\n   Val:   {len(X_val):,}\n   Test:  {len(X_test):,}")
            classifier = VirusClassifier()
            classifier.train(X_tr, y_tr, X_val, y_val, use_smote=True, use_optuna=True)
            classifier.evaluate(X_test, y_test, label_encoder)
            classifier.analyze_families(X_test, y_test, label_encoder)
            classifier.save_model()
            total_time = time.time() - overall_start
            print("\n" + "=" * 70)
            print("🎉 VIRUS CLASSIFIER COMPLETE")
            print("=" * 70)
            print(f"⏱️  Total pipeline time: {total_time:.2f}s ({total_time / 60:.1f} min)")
            print("\n✅ Virus classifier trained successfully!")
    except Exception as e:
        print(f"\n❌ Error training virus classifier: {e}")
        print("This can happen if your dataset has too few malware families with enough samples.")
