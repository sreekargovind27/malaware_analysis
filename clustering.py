"""
Clustering: K-Means on Malicious Traffic
UPDATED: Enhanced timing, better error handling, and progress tracking.
"""
import os
import time

import joblib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score
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
        print(f"K-Means Clustering initialized")
        print(f"  Target clusters: {self.n_clusters}")
        print(f"  Batch size: {Config.KMEANS_BATCH_SIZE}")

    def fit(self, X):
        """Fit K-Means clustering model"""
        print("\n" + "=" * 70)
        print("🚀 TRAINING K-MEANS CLUSTERING")
        print("=" * 70)

        overall_start = time.time()

        print(f"\n📊 Input data:")
        print(f"   Samples: {len(X):,}")
        print(f"   Features: {X.shape[1]}")

        # Step 1: Remove constant columns
        print(f"\n⏳ Removing constant columns...")
        t_filter_start = time.time()
        self.feature_names_in_ = X.columns[X.std() > 1e-6].tolist()

        if len(self.feature_names_in_) < X.shape[1]:
            removed_cols = set(X.columns) - set(self.feature_names_in_)
            print(f"   ⚠️  Removed {len(removed_cols)} constant columns")
            if len(removed_cols) <= 10:
                print(f"      {list(removed_cols)}")

        if not self.feature_names_in_:
            print("❌ ERROR: No features with variance found. Cannot perform clustering.")
            return None

        t_filter = time.time() - t_filter_start
        print(f"✓ Kept {len(self.feature_names_in_)} features ({t_filter:.2f}s)")

        X_filtered = X[self.feature_names_in_]

        # Step 2: Handle any remaining NaN values
        nan_count = X_filtered.isna().sum().sum()
        if nan_count > 0:
            print(f"\n⚠️  WARNING: Found {nan_count:,} NaN values. Filling with 0...")
            X_filtered = X_filtered.fillna(0)
        else:
            print(f"\n✓ No missing values")

        # Step 3: Scale
        print(f"\n⏳ Scaling features...")
        t_scale_start = time.time()
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_filtered.values)
        t_scale = time.time() - t_scale_start
        print(f"✓ Scaling complete ({t_scale:.2f}s)")
        print(f"   Data range: [{X_scaled.min():.2f}, {X_scaled.max():.2f}]")

        # Step 4: Fit k-means
        print(f"\n⏳ Training K-Means (k={self.n_clusters})...")
        t_kmeans_start = time.time()

        self.model = MiniBatchKMeans(
            n_clusters=self.n_clusters,
            batch_size=Config.KMEANS_BATCH_SIZE,
            n_init=10,
            random_state=Config.RANDOM_STATE,
            verbose=0
        )
        self.labels = self.model.fit_predict(X_scaled)

        t_kmeans = time.time() - t_kmeans_start
        print(f"✓ K-Means training complete ({t_kmeans:.2f}s)")

        # Step 5: Print cluster distribution
        unique, counts = np.unique(self.labels, return_counts=True)
        print(f"\n📊 Cluster Distribution:")
        for cluster, count in zip(unique, counts):
            pct = (count / len(X)) * 100
            print(f"   Cluster {cluster}: {count:,} samples ({pct:.1f}%)")

        # Step 6: Calculate metrics
        print(f"\n⏳ Calculating clustering quality metrics...")
        t_metrics_start = time.time()

        if len(unique) > 1:
            # Silhouette score (sample for speed if dataset is large)
            sample_size = min(10000, len(X))
            if len(X) > sample_size:
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X_scaled[indices]
                labels_sample = self.labels[indices]
            else:
                X_sample = X_scaled
                labels_sample = self.labels

            silhouette = silhouette_score(X_sample, labels_sample)
            davies_bouldin = davies_bouldin_score(X_scaled, self.labels)

            t_metrics = time.time() - t_metrics_start

            print(f"✓ Metrics calculated ({t_metrics:.2f}s)")
            print(f"\n📈 Clustering Quality Metrics:")
            print(f"   Silhouette Score: {silhouette:.4f} (higher is better, range: -1 to 1)")
            print(f"   Davies-Bouldin Index: {davies_bouldin:.4f} (lower is better)")
        else:
            print(f"   ⚠️  Only one cluster found, skipping metrics")

        total_time = time.time() - overall_start

        print(f"\n" + "=" * 70)
        print(f"✅ CLUSTERING COMPLETE")
        print(f"=" * 70)
        print(f"⏱️  Total time: {total_time:.2f}s")
        print("=" * 70)

        return self.labels

    def predict(self, X):
        """Predict cluster labels for new data"""
        if self.scaler is None or self.model is None:
            print("❌ ERROR: Model not trained yet")
            return None

        X_filtered = X[self.feature_names_in_]
        X_scaled = self.scaler.transform(X_filtered.values)
        return self.model.predict(X_scaled)

    def visualize_clusters_by_label(self, X, true_labels, label_names=None, save_plot=True):
        """Visualize clusters vs true labels using PCA"""
        print("\n" + "=" * 70)
        print("📊 GENERATING CLUSTER VISUALIZATION")
        print("=" * 70)

        t_start = time.time()

        X_filtered = X[self.feature_names_in_]

        # Handle NaN values
        if X_filtered.isna().any().any():
            print("   ⚠️  Found NaN in visualization data. Filling with 0...")
            X_filtered = X_filtered.fillna(0)

        print(f"\n⏳ Scaling data...")
        X_scaled = self.scaler.transform(X_filtered.values)

        print(f"⏳ Applying PCA (2 components)...")
        t_pca_start = time.time()
        pca = PCA(n_components=2, random_state=Config.RANDOM_STATE)
        X_pca = pca.fit_transform(X_scaled)
        t_pca = time.time() - t_pca_start
        print(f"✓ PCA complete ({t_pca:.2f}s)")
        print(
            f"   Explained variance: PC1={pca.explained_variance_ratio_[0] * 100:.1f}%, PC2={pca.explained_variance_ratio_[1] * 100:.1f}%")
        print(f"   Total: {sum(pca.explained_variance_ratio_) * 100:.1f}%")

        print(f"\n⏳ Creating visualization...")
        t_plot_start = time.time()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

        # Left plot: K-Means clusters
        scatter1 = ax1.scatter(X_pca[:, 0], X_pca[:, 1], c=self.labels, cmap='viridis', alpha=0.6, s=20)
        ax1.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)', fontsize=12)
        ax1.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)', fontsize=12)
        ax1.set_title('K-Means Clusters (Unsupervised)', fontsize=14, fontweight='bold')
        plt.colorbar(scatter1, ax=ax1, label='K-Means Cluster')

        # Right plot: True labels
        scatter2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=true_labels, cmap='Set1', alpha=0.6, s=20)
        ax2.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0] * 100:.1f}% variance)', fontsize=12)
        ax2.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1] * 100:.1f}% variance)', fontsize=12)
        ax2.set_title('True Malware Labels (Supervised)', fontsize=14, fontweight='bold')

        if label_names is not None and len(label_names) > 0:
            cbar = plt.colorbar(scatter2, ax=ax2, label='Malware Type')
            cbar.set_ticks(np.arange(len(label_names)))
            cbar.set_ticklabels(label_names)

        plt.tight_layout()

        if save_plot:
            filepath = os.path.join(Config.RESULTS_DIR, 'clusters_vs_true_labels.png')
            plt.savefig(filepath, dpi=300)
            plt.close()

            t_plot = time.time() - t_plot_start
            print(f"✓ Visualization saved ({t_plot:.2f}s)")
            print(f"   Saved to: {filepath}")
        else:
            plt.show()

        t_total = time.time() - t_start
        print(f"\n⏱️  Total visualization time: {t_total:.2f}s")

    def save_model(self, filename='kmeans_clustering.pkl'):
        """Save clustering model"""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'n_clusters': self.n_clusters,
            'features': self.feature_names_in_,
            'labels': self.labels
        }, filepath)
        print(f"\n💾 Model saved: {filename}")

    def load_model(self, filename='kmeans_clustering.pkl'):
        """Load clustering model"""
        filepath = os.path.join(Config.MODELS_DIR, filename)
        if not os.path.exists(filepath):
            print(f"❌ Model file {filepath} not found")
            return

        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.n_clusters = data['n_clusters']
        self.feature_names_in_ = data['features']
        self.labels = data.get('labels')
        print(f"✓ Model loaded: {filename}")


if __name__ == "__main__":
    Config.set_seeds()
    Config.print_mode_info()

    print("=" * 70)
    print("🔬 K-MEANS CLUSTERING ON MALICIOUS TRAFFIC")
    print("=" * 70)

    overall_start = time.time()

    # Load data
    df = load_engineered_data()

    print(f"\n⏳ Filtering malicious samples...")
    t_filter_start = time.time()
    malicious_df = df[df[Config.TARGET_COL] == 'Malicious'].copy()
    t_filter = time.time() - t_filter_start
    print(f"✓ Filtered ({t_filter:.2f}s)")

    X_malicious = get_data_for_clustering()
    true_labels_str = malicious_df[Config.FAMILY_TARGET_COL].fillna('Unknown').astype(str)

    print(f"\n⏳ Encoding true labels for visualization...")
    le = LabelEncoder()
    true_labels_encoded = le.fit_transform(true_labels_str)
    label_names = le.classes_.tolist()
    print(f"✓ Found {len(label_names)} unique malware types")

    print(f"\n📊 Dataset summary:")
    print(f"   Malicious samples: {len(X_malicious):,}")
    print(f"   Features: {X_malicious.shape[1]}")
    print(f"   Unique malware types: {len(label_names)}")

    # Adjust number of clusters if needed
    num_clusters = min(Config.KMEANS_N_CLUSTERS, len(X_malicious))

    if num_clusters < 2:
        print(f"\n❌ ERROR: Not enough malicious data ({len(X_malicious)}) to form at least 2 clusters. Skipping.")
    else:
        # Initialize and train
        kmeans = KMeansClustering(n_clusters=num_clusters)
        labels = kmeans.fit(X_malicious)

        if labels is not None:
            # Visualize
            kmeans.visualize_clusters_by_label(X_malicious, true_labels_encoded, label_names)

            # Save model
            kmeans.save_model()

            total_time = time.time() - overall_start

            print("\n" + "=" * 70)
            print("🎉 K-MEANS CLUSTERING COMPLETE")
            print("=" * 70)
            print(f"⏱️  Total pipeline time: {total_time:.2f}s ({total_time / 60:.1f} min)")
            print("\n✅ Clustering model trained and saved successfully!")
        else:
            print("\n❌ Clustering failed - fit() returned None")
