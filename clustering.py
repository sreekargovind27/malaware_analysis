"""
Performs K-Means clustering on malicious network traffic data.
This script includes training, prediction, and visualization of clusters against true labels,
using a stratified sample for efficient plotting.
"""
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

from config import Config
from data_loader import get_data_for_clustering, load_engineered_data


class KMeansClustering:
    def __init__(self, n_clusters=None):
        self.n_clusters = n_clusters or Config.KMEANS_N_CLUSTERS
        self.model = None
        self.scaler = None
        self.labels = None
        self.feature_names_in_ = None
        print(f"K-Means Clustering initialized with k={self.n_clusters}")

    def fit(self, X):
        print("\n" + "=" * 70)
        print("🚀 TRAINING K-MEANS CLUSTERING")
        print("=" * 70)
        print(f"\n📊 Input data:\n   Samples: {len(X):,}\n   Features: {X.shape[1]}")
        print(f"\n⏳ Removing constant columns...")
        self.feature_names_in_ = X.columns[X.std() > 1e-6].tolist()
        X_filtered = X[self.feature_names_in_].fillna(0)
        print(f"\n⏳ Scaling features...")
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_filtered.values)
        print(f"✓ Scaling complete.")
        print(f"\n⏳ Training K-Means...")
        self.model = MiniBatchKMeans(n_clusters=self.n_clusters, batch_size=Config.KMEANS_BATCH_SIZE, n_init=10,
                                     random_state=Config.RANDOM_STATE, verbose=0)
        self.labels = self.model.fit_predict(X_scaled)
        print(f"✓ K-Means training complete.")
        return self.labels

    def predict(self, X):
        if self.scaler is None or self.model is None:
            raise RuntimeError("Model has not been trained yet.")
        X_filtered = X[self.feature_names_in_].fillna(0)
        X_scaled = self.scaler.transform(X_filtered.values)
        return self.model.predict(X_scaled)

    def visualize_clusters_by_label(self, X, true_labels, label_names=None, save_plot=True):
        print("\n" + "=" * 70)
        print("📊 GENERATING CLUSTER VISUALIZATION")
        print("=" * 70)
        X_filtered = X[self.feature_names_in_].fillna(0)
        print(f"\n⏳ Scaling data for PCA...")
        X_scaled = self.scaler.transform(X_filtered.values)
        print(f"⏳ Applying PCA...")
        pca = PCA(n_components=2, random_state=Config.RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
        print(f"✓ PCA complete. Explained variance: {sum(pca.explained_variance_ratio_) * 100:.1f}%")
        print(f"\n⏳ Creating visualization...")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels, cmap='viridis', alpha=0.6, s=20)
        ax1.set_title('K-Means Clusters (Unsupervised)')
        plt.colorbar(scatter1, ax=ax1, label='K-Means Cluster')
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='Set1', alpha=0.6, s=20)
        ax2.set_title('True Malware Labels (Supervised)')
        if label_names is not None and len(label_names) > 0:
            cbar = plt.colorbar(scatter2, ax=ax2, label='Malware Type')
            cbar.set_ticks(np.arange(len(label_names)))
            cbar.set_ticklabels(label_names)
        plt.tight_layout()
        if save_plot:
            filepath = os.path.join(Config.RESULTS_DIR, 'clusters_vs_true_labels.png')
            plt.savefig(filepath, dpi=300)
            plt.close()
            print(f"✓ Visualization saved to {filepath}")
        else:
            plt.show()

    def save_model(self, filename='kmeans_clustering.pkl'):
        filepath = os.path.join(Config.MODELS_DIR, filename)
        joblib.dump({'model': self.model, 'scaler': self.scaler, 'n_clusters': self.n_clusters,
                     'features': self.feature_names_in_}, filepath)
        print(f"\n💾 Model saved: {filename}")

    def load_model(self, filename='kmeans_clustering.pkl'):
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"❌ Model file {filepath} not found")
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.n_clusters = data['n_clusters']
        self.feature_names_in_ = data['features']
        print(f"✓ Model loaded: {filename}")


if __name__ == "__main__":
    Config.set_seeds()
    Config.print_mode_info()
    print("=" * 70)
    print("🔬 K-MEANS CLUSTERING ON MALICIOUS TRAFFIC")
    print("=" * 70)

    df = load_engineered_data()
    malicious_df = df[df[Config.TARGET_COL] == 'Malicious'].copy()
    X_malicious = get_data_for_clustering()
    true_labels_str = malicious_df[Config.FAMILY_TARGET_COL].fillna('Unknown').astype(str)
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(true_labels_str)
    label_names = le.classes_.tolist()

    num_clusters = min(Config.KMEANS_N_CLUSTERS, len(np.unique(true_labels_encoded)))
    if num_clusters < 2:
        print(f"\n❌ ERROR: Not enough data ({len(X_malicious)}) to form at least 2 clusters. Skipping.")
    else:
        kmeans = KMeansClustering(n_clusters=num_clusters)
        labels = kmeans.fit(X_malicious)

        if labels is not None:
            print("\n⏳ Creating a sample for fast visualization...")
            max_plot_samples = 100000

            if len(X_malicious) > max_plot_samples:
                print(
                    f"   Dataset is large ({len(X_malicious):,}), creating stratified sample of size {max_plot_samples:,}...")
                X_malicious_sample, _, true_labels_encoded_sample, _ = train_test_split(
                    X_malicious, true_labels_encoded,
                    train_size=max_plot_samples, stratify=true_labels_encoded,
                    random_state=Config.RANDOM_STATE
                )
            else:
                print(f"   Dataset is small enough ({len(X_malicious):,}), using full dataset for plotting.")
                X_malicious_sample = X_malicious
                true_labels_encoded_sample = true_labels_encoded

            print("⏳ Predicting clusters for the sample to be visualized...")
            sample_cluster_labels = kmeans.predict(X_malicious_sample)
            kmeans.labels = sample_cluster_labels

            kmeans.visualize_clusters_by_label(X_malicious_sample, true_labels_encoded_sample, label_names)
            kmeans.save_model()

            print("\n" + "=" * 70)
            print("🎉 K-MEANS CLUSTERING COMPLETE")
            print("=" * 70)
        else:
            print("\n❌ Clustering failed - fit() returned None")
