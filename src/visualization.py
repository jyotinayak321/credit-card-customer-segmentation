"""
visualization.py
-----------------
Plots for EDA and for viewing the final clusters.
"""

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA

try:
    from .config import OUTPUT_DIR
except ImportError:
    from config import OUTPUT_DIR


def plot_correlation_heatmap(df, save_path=None):
    save_path = save_path or (OUTPUT_DIR / "correlation_heatmap.png")
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.select_dtypes("number").corr(), cmap="coolwarm", annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_two_feature_scatter(df, x, y, hue="Cluster", save_path=None):
    save_path = save_path or (OUTPUT_DIR / "cluster_plot.png")
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df, x=x, y=y, hue=hue, palette="deep")
    plt.title(f"Customer Clusters: {x} vs {y}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_pca_clusters(X_scaled, labels, save_path=None):
    save_path = save_path or (OUTPUT_DIR / "pca_clusters.png")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="tab10")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("Clusters in PCA Space")
    plt.colorbar(scatter, label="Cluster")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()