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
    

def plot_outlier_boxplots(df, features=None, save_path=None):
    """Visual representation of outliers using boxplots - complements
    the numeric IQR report in preprocessing.detect_outliers().

    Each box shows: Q1-Q3 range (the box), whiskers (1.5x IQR), and
    individual dots beyond the whiskers = the same outliers that
    detect_outliers() counts numerically.
    """
    try:
        from .config import OUTPUT_DIR, DEFAULT_FEATURES
    except ImportError:
        from config import OUTPUT_DIR, DEFAULT_FEATURES

    features = features or DEFAULT_FEATURES
    save_path = save_path or (OUTPUT_DIR / "outlier_boxplots.png")

    n_cols = 3
    n_rows = -(-len(features) // n_cols)  # ceiling division
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(features):
        sns.boxplot(y=df[col], ax=axes[i], color="skyblue")
        axes[i].set_title(col, fontsize=10)

    # Hide any unused subplot slots
    for j in range(len(features), len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
def plot_dbscan_pca(X_scaled, dbscan_labels, save_path=None):
    """PCA scatter plot for DBSCAN results - noise points (-1) are
    shown as gray X markers, distinct from real clusters."""
    try:
        from .config import OUTPUT_DIR
    except ImportError:
        from config import OUTPUT_DIR

    save_path = save_path or (OUTPUT_DIR / "dbscan_pca.png")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    noise_mask = dbscan_labels == -1
    cluster_mask = ~noise_mask

    if cluster_mask.any():
        scatter = plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1],
                            c=dbscan_labels[cluster_mask], cmap="tab10", s=20, label="Clusters")
        plt.colorbar(scatter, label="DBSCAN cluster")
    if noise_mask.any():
        plt.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1],
                    c="gray", marker="x", s=30, label="Noise (-1)")

    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title("DBSCAN Clusters in PCA Space (noise marked as x)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_kmeans_elbow(k_values, inertias, silhouettes=None, chosen_k=None, save_path=None):
    """Elbow curve (inertia vs k) for K-Means, with an optional silhouette
    panel. Elbow = the k after which inertia stops dropping sharply."""
    try:
        from .config import OUTPUT_DIR
    except ImportError:
        from config import OUTPUT_DIR

    save_path = save_path or (OUTPUT_DIR / "kmeans_elbow_curve.png")
    n_panels = 2 if silhouettes else 1
    fig, axes = plt.subplots(1, n_panels, figsize=(6.5 * n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    axes[0].plot(list(k_values), inertias, marker="o")
    axes[0].set_xlabel("Number of clusters (k)")
    axes[0].set_ylabel("Inertia (within-cluster sum of squares)")
    axes[0].set_title("K-Means Elbow Method")
    axes[0].grid(True, alpha=0.3)
    if chosen_k is not None and chosen_k in list(k_values):
        axes[0].axvline(chosen_k, color="red", linestyle="--", label=f"chosen k={chosen_k}")
        axes[0].legend()

    if silhouettes:
        ks = list(silhouettes.keys())
        axes[1].plot(ks, list(silhouettes.values()), marker="o", color="green")
        axes[1].set_xlabel("Number of clusters (k)")
        axes[1].set_ylabel("Silhouette score")
        axes[1].set_title("K-Means Silhouette Score by k")
        axes[1].grid(True, alpha=0.3)
        if chosen_k is not None and chosen_k in ks:
            axes[1].axvline(chosen_k, color="red", linestyle="--", label=f"chosen k={chosen_k}")
            axes[1].legend()

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_kmeans_clusters(X_scaled, labels, centers=None, save_path=None):
    """K-Means clusters in 2D PCA space. Centroids (scaled space) are
    projected with the SAME PCA so they land in the right place."""
    try:
        from .config import OUTPUT_DIR
    except ImportError:
        from config import OUTPUT_DIR

    save_path = save_path or (OUTPUT_DIR / "kmeans_clusters.png")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    plt.figure(figsize=(8, 6))
    cmap = plt.get_cmap("tab10")
    for i, cl in enumerate(sorted(set(labels))):
        mask = labels == cl
        plt.scatter(X_pca[mask, 0], X_pca[mask, 1], color=cmap(i), s=20, label=f"Cluster {cl}")
    if centers is not None:
        c_pca = pca.transform(centers)
        plt.scatter(c_pca[:, 0], c_pca[:, 1], c="black", marker="X", s=200,
                    edgecolors="white", label="Centroids")
    plt.legend()
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.title(f"K-Means Clusters in PCA Space (k={len(set(labels))})")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()