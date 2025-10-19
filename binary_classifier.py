"""
Binary Classification: Benign vs Malicious
Uses LightGBM with SMOTE for class imbalance and Optuna for hyperparameter tuning.
Also compares against a LightGBM-based logistic regression baseline.
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
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split

from config import Config
from data_loader import get_data_for_binary


class BinaryClassifier:
    """Binary classification using LightGBM with advanced features"""

    def __init__(self, params=None):
        self.model = None
        self.params = params or self._get_default_params()
        self.best_params = None
        print(f"\n" + "=" * 70)
        print("ADVANCED MODEL: Tuned LightGBM")
        print("=" * 70)
        print(f"Binary Classifier (LightGBM) initialized")
        print(f"  Using Optuna: {Config.USE_OPTUNA}")
        print(f"  Using SMOTE: {Config.USE_SMOTE}")

    def _get_default_params(self):
        return {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
                'num_leaves': Config.BINARY_NUM_LEAVES, 'learning_rate': Config.BINARY_LEARNING_RATE,
                'n_estimators': Config.BINARY_N_ESTIMATORS, 'n_jobs': Config.N_JOBS,
                'random_state': Config.RANDOM_STATE, 'verbose': -1}

    # ===== THIS IS THE ONLY FUNCTION THAT HAS BEEN UPDATED =====
    def apply_smote(self, X_train, y_train):
        """Apply SMOTE (oversampling only) for balanced dataset."""
        print("\n" + "=" * 70)
        print("🔄 HANDLING CLASS IMBALANCE")
        print("=" * 70)
        t_start = time.time()
        unique, counts = np.unique(y_train, return_counts=True)
        minority_class_index = np.argmin(counts)
        majority_class_index = np.argmax(counts)

        print(f"\n📊 Original class distribution:")
        print(f"   Benign: {counts[0]:,} ({counts[0] / len(y_train) * 100:.2f}%)")
        print(f"   Malicious: {counts[1]:,} ({counts[1] / len(y_train) * 100:.2f}%)")

        if not Config.USE_SMOTE or len(X_train) > Config.SMOTE_SAMPLE_THRESHOLD:
            if not Config.USE_SMOTE:
                print(f"\n⏭️  SMOTE disabled in config")
            else:
                print(f"\n⏭️  Dataset too large ({len(X_train):,}), using class weights instead")
            print(f"⏱️  Time: {time.time() - t_start:.2f}s")
            return X_train, y_train

        print(f"\n⏳ Applying SMOTE (Oversampling only)...")
        try:
            minority_count = counts[minority_class_index]
            if minority_count <= 1:
                raise ValueError("Not enough samples in minority class for SMOTE.")

            # Oversample the minority class to be equal to the majority class
            smote = SMOTE(sampling_strategy='auto', random_state=Config.RANDOM_STATE,
                          k_neighbors=min(5, minority_count - 1))
            X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

            unique_new, counts_new = np.unique(y_resampled, return_counts=True)
            print(f"\n📊 Resampled class distribution:")
            print(f"   Benign: {counts_new[0]:,} ({counts_new[0] / len(y_resampled) * 100:.2f}%)")
            print(f"   Malicious: {counts_new[1]:,} ({counts_new[1] / len(y_resampled) * 100:.2f}%)")
            print(f"\n✅ Resampling complete\n   Original: {len(X_train):,} → Resampled: {len(X_resampled):,}")
            print(f"⏱️  Time: {time.time() - t_start:.2f}s")
            return X_resampled, y_resampled

        except Exception as e:
            print(f"\n❌ SMOTE failed: {e}\n   Continuing with original data + class weights")
            print(f"⏱️  Time: {time.time() - t_start:.2f}s")
            return X_train, y_train

    # =================================================================

    def optimize_hyperparameters(self, X_train, y_train, X_val, y_val):
        print("\n" + "=" * 70);
        print("🔍 HYPERPARAMETER OPTIMIZATION (OPTUNA)");
        print("=" * 70)
        t_start = time.time()
        if not Config.USE_OPTUNA:
            print("\n⏭️  Optuna disabled in config");
            self.best_params = self.params;
            return self.params
        print(f"\n⏳ Running Optuna for {Config.OPTUNA_N_TRIALS} trials (max {Config.OPTUNA_TIMEOUT / 60:.1f} min)...")

        def objective(trial):
            param = {'objective': 'binary', 'metric': 'binary_logloss', 'boosting_type': 'gbdt',
                     'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                     'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                     'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                     'max_depth': trial.suggest_int('max_depth', 3, 15),
                     'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
                     'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                     'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0), 'n_jobs': Config.N_JOBS,
                     'random_state': Config.RANDOM_STATE, 'verbose': -1}
            if (y_train == 1).sum() > 0: param['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()
            model = lgb.LGBMClassifier(**param)
            model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(10, verbose=False)])
            return roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

        study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=Config.RANDOM_STATE))
        study.optimize(objective, n_trials=Config.OPTUNA_N_TRIALS, timeout=Config.OPTUNA_TIMEOUT,
                       show_progress_bar=True)
        print(
            f"\n✅ Optimization complete in {time.time() - t_start:.2f}s.\n📊 Best trial:\n   ROC-AUC: {study.best_value:.4f}\n   Parameters:")
        for key, value in study.best_params.items(): print(f"      {key}: {value}")
        self.best_params = {**self.params, **study.best_params}
        return self.best_params

    def train(self, X_train, y_train, X_val=None, y_val=None, use_smote=True, use_optuna=True):
        print("\n" + "=" * 70);
        print("🚀 TRAINING ADVANCED LightGBM MODEL");
        print("=" * 70)
        if use_smote: X_train, y_train = self.apply_smote(X_train, y_train)
        if use_optuna and X_val is not None:
            best_params = self.optimize_hyperparameters(X_train, y_train, X_val, y_val)
        else:
            best_params = self.params
            print("\n⏭️  Using default parameters (no optimization)")
        print("\n" + "=" * 70);
        print("🎯 TRAINING FINAL ADVANCED MODEL");
        print("=" * 70)
        t_train_start = time.time()
        if (y_train == 1).sum() > 0:
            scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
            best_params['scale_pos_weight'] = scale_pos_weight
            print(f"\n⚖️  Using class weight: {scale_pos_weight:.2f}")
        print(f"\n⏳ Training LightGBM with final parameters...")
        self.model = lgb.LGBMClassifier(**best_params)
        self.model.fit(X_train, y_train)
        print(f"\n✅ Training complete in {time.time() - t_train_start:.2f}s")

    def predict(self, X):
        return self.model.predict(X)

    def predict_proba(self, X):
        return self.model.predict_proba(X)

    def evaluate(self, X_test, y_test, save_plots=True):
        print("\n" + "=" * 70);
        print("📊 EVALUATING ADVANCED LightGBM MODEL");
        print("=" * 70)
        t_eval_start = time.time()
        y_pred = self.predict(X_test);
        y_proba = self.predict_proba(X_test)[:, 1]
        print(f"✓ Predictions complete ({(time.time() - t_eval_start):.2f}s)")
        print("\n" + "─" * 70);
        print("CLASSIFICATION REPORT");
        print("─" * 70)
        print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious'], zero_division=0))
        cm = confusion_matrix(y_test, y_pred)
        print("─" * 70);
        print("CONFUSION MATRIX");
        print("─" * 70)
        print(
            f"                 Predicted\n                 Benign  Malicious\nActual Benign    {cm[0][0]:6d}  {cm[0][1]:6d}\n       Malicious {cm[1][0]:6d}  {cm[1][1]:6d}")
        tn, fp, fn, tp = cm.ravel()
        accuracy = (tp + tn) / (tp + tn + fp + fn);
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0;
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        roc_auc = roc_auc_score(y_test, y_proba)
        print("\n" + "─" * 70);
        print("KEY METRICS");
        print("─" * 70)
        print(
            f"  Accuracy:  {accuracy:.4f}\n  Recall:    {recall:.4f} (We catch {recall * 100:.1f}% of all Malicious)\n  ROC-AUC:   {roc_auc:.4f}\n\n  False Positives: {fp:,}\n  False Negatives: {fn:,} (Malicious missed - BAD!)")
        if save_plots:
            self._plot_confusion_matrix(cm)
            self._plot_roc_curve(y_test, y_proba, roc_auc)
            self._plot_feature_importance(X_test.columns)
        return {'roc_auc': roc_auc, 'accuracy': accuracy, 'recall': recall, 'false_negatives': fn}

    def _plot_confusion_matrix(self, cm):
        plt.figure(figsize=(8, 6));
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Benign', 'Malicious'],
                    yticklabels=['Benign', 'Malicious'])
        plt.title('Confusion Matrix - Tuned LightGBM');
        plt.ylabel('True Label');
        plt.xlabel('Predicted Label');
        plt.tight_layout()
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'binary_tuned_confusion_matrix.png'), dpi=300);
        plt.close()

    def _plot_roc_curve(self, y_test, y_proba, roc_auc):
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6));
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})');
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate');
        plt.ylabel('True Positive Rate');
        plt.title('ROC Curve - Tuned LightGBM');
        plt.legend();
        plt.grid(True)
        plt.tight_layout();
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'binary_tuned_roc_curve.png'), dpi=300);
        plt.close()

    def _plot_feature_importance(self, feature_names):
        lgb.plot_importance(self.model, max_num_features=20, height=0.8, figsize=(12, 8))
        plt.title('Top 20 Feature Importance - Tuned LightGBM', fontsize=14)
        plt.tight_layout();
        plt.savefig(os.path.join(Config.RESULTS_DIR, 'binary_tuned_feature_importance.png'), dpi=300);
        plt.close()
        print("✓ All plots saved.")

    def save_model(self, filename='binary_classifier_tuned.pkl'):
        filepath = os.path.join(Config.MODELS_DIR, filename)
        joblib.dump({'model': self.model, 'params': self.best_params if self.best_params else self.params}, filepath)
        print(f"\n💾 Model saved: {filename}")


def train_and_evaluate_baseline(X_train, y_train, X_test, y_test):
    """Trains and evaluates the LightGBM Logistic Regression baseline."""
    print("\n" + "=" * 70);
    print("🚀 TRAINING BASELINE: LightGBM (Logistic Config)");
    print("=" * 70)
    t_start = time.time()
    params = {'objective': 'binary', 'boosting_type': 'gbdt', 'num_leaves': 2, 'max_depth': 1, 'n_estimators': 200,
              'learning_rate': 0.1, 'n_jobs': Config.N_JOBS, 'random_state': Config.RANDOM_STATE, 'verbose': -1}
    if (y_train == 1).sum() > 0: params['scale_pos_weight'] = (y_train == 0).sum() / (y_train == 1).sum()

    model = lgb.LGBMClassifier(**params)
    model.fit(X_train, y_train)
    print(f"✓ Baseline training complete ({time.time() - t_start:.2f}s)")

    print("\n" + "─" * 70);
    print("EVALUATING BASELINE");
    print("─" * 70)
    y_pred = model.predict(X_test);
    y_proba = model.predict_proba(X_test)[:, 1]
    print(classification_report(y_test, y_pred, target_names=['Benign', 'Malicious'], zero_division=0))
    roc_auc = roc_auc_score(y_test, y_proba)
    print(f"  ROC-AUC: {roc_auc:.4f}")
    joblib.dump(model, os.path.join(Config.MODELS_DIR, 'binary_classifier_logistic.pkl'))
    print(f"\n💾 Baseline model saved.")
    return {'roc_auc': roc_auc}


if __name__ == "__main__":
    Config.set_seeds()
    Config.print_mode_info()
    
    print("=" * 70);
    print("🎯 BINARY CLASSIFICATION: BENIGN VS MALICIOUS");
    print("=" * 70)
    overall_start = time.time()

    X_train, X_test, y_train, y_test = get_data_for_binary()
    X_tr, X_val, y_tr, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=Config.RANDOM_STATE,
                                                stratify=y_train)

    # --- 1. Train Advanced, Tuned LightGBM Model ---
    lgbm_classifier = BinaryClassifier()
    lgbm_classifier.train(X_tr, y_tr, X_val, y_val, use_smote=True, use_optuna=True)
    lgbm_results = lgbm_classifier.evaluate(X_test, y_test)
    lgbm_classifier.save_model()

    # --- 2. Train LightGBM Logistic Regression Baseline ---
    if Config.RUN_LOGISTIC_REGRESSION:
        print("\n" + "=" * 70);
        print("🚀 TRAINING BASELINE: LightGBM (Logistic Config)");
        print("=" * 70)
        lr_params = {'objective': 'binary', 'boosting_type': 'gbdt', 'num_leaves': 2, 'max_depth': 1,
                     'n_estimators': 200, 'learning_rate': 0.1, 'n_jobs': Config.N_JOBS,
                     'random_state': Config.RANDOM_STATE, 'verbose': -1}
        lr_classifier = BinaryClassifier(params=lr_params)
        lr_classifier.train(X_train, y_train, use_smote=False,
                            use_optuna=False)  # Train baseline on original data without tuning
        lr_results = lr_classifier.evaluate(X_test, y_test, save_plots=False)  # Don't overwrite plots
        lr_classifier.save_model('binary_classifier_logistic.pkl')

    # --- 3. Final Summary ---
    print("\n" + "=" * 70);
    print("🎉 FINAL RESULTS SUMMARY");
    print("=" * 70)
    print(f"⏱️  Total pipeline time: {time.time() - overall_start:.2f}s")
    print(f"\n🎯 Model Comparison (ROC-AUC):")
    if Config.RUN_LOGISTIC_REGRESSION:
        print(f"   LightGBM (Logistic): {lr_results['roc_auc']:.4f} (Baseline)")
    print(f"   LightGBM (Tuned GBDT): {lgbm_results['roc_auc']:.4f} (Advanced)")
    print("\n✅ All models trained and saved successfully!")
