"""
clustering.py
--------------
The "engine" of the project: builds the dendrogram, tests k values with
silhouette score, trains the final AgglomerativeClustering model,
profiles the resulting clusters, runs EDA + visualization, and
auto-generates business recommendations - run_pipeline() covers the
FULL guide workflow end-to-end.

Run with defaults:
    python clustering.py

Run with custom cluster count (CLI arg):
    python clustering.py --k 5

Run as a package module:
    python -m src.clustering --k 5
"""

import argparse
import logging

import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

try:
    from .config import DEFAULT_DATA_PATH, OUTPUT_DIR, DEFAULT_FEATURES, DEFAULT_N_CLUSTERS
    from .data_loader import load_data, inspect_data
    from .preprocessing import clean_data, select_features, scale_features, detect_outliers
    from .visualization import plot_correlation_heatmap, plot_two_feature_scatter, plot_pca_clusters
    from .business_rules import generate_business_recommendations, evaluate_cluster_health
except ImportError:
    from config import DEFAULT_DATA_PATH, OUTPUT_DIR, DEFAULT_FEATURES, DEFAULT_N_CLUSTERS
    from data_loader import load_data, inspect_data
    from preprocessing import clean_data, select_features, scale_features, detect_outliers
    from visualization import plot_correlation_heatmap, plot_two_feature_scatter, plot_pca_clusters
    from business_rules import generate_business_recommendations, evaluate_cluster_health

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def build_dendrogram(X_scaled, save_path=None):
    save_path = save_path or (OUTPUT_DIR / "dendrogram.png")
    linked = linkage(X_scaled, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(linked, truncate_mode="level", p=5)
    plt.xlabel("Customers / merged clusters")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    logger.info(f"Dendrogram saved to {save_path}")
    return linked


def find_best_k(X_scaled, k_range=range(2, 8)) -> dict:
    scores = {}
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        labels = model.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def train_agglomerative(X_scaled, n_clusters: int = DEFAULT_N_CLUSTERS):
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    labels = model.fit_predict(X_scaled)
    return model, labels


def compare_with_kmeans(X_scaled, n_clusters: int = DEFAULT_N_CLUSTERS, agglo_labels=None) -> dict:
    """Train K-Means with the same k and compare against Agglomerative
    Clustering. Useful for the common interview question: "Why did you
    pick Agglomerative Clustering over K-Means?"
    """
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)
    kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)

    result = {
        "kmeans_labels": kmeans_labels,
        "kmeans_silhouette": kmeans_silhouette,
    }

    if agglo_labels is not None:
        agglo_silhouette = silhouette_score(X_scaled, agglo_labels)
        result["agglomerative_silhouette"] = agglo_silhouette
        logger.info(
            f"Silhouette comparison — Agglomerative: {agglo_silhouette:.4f} | "
            f"K-Means: {kmeans_silhouette:.4f}"
        )

    return result


def profile_clusters(df: pd.DataFrame, features: list, cluster_col: str = "Cluster") -> pd.DataFrame:
    profile = df.groupby(cluster_col)[features].mean().round(2)
    profile["count"] = df[cluster_col].value_counts().sort_index()
    return profile


def run_pipeline(data_path=None,
                  n_clusters: int = DEFAULT_N_CLUSTERS,
                  output_dir=None,
                  run_kmeans_comparison: bool = True):
    data_path = data_path or DEFAULT_DATA_PATH
    output_dir = output_dir or OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading dataset...")
    df_raw = load_data(data_path)
    inspect_data(df_raw)

    logger.info("Cleaning data...")
    df_clean = clean_data(df_raw)

    plot_correlation_heatmap(df_clean, save_path=output_dir / "correlation_heatmap.png")
    logger.info(f"Correlation heatmap saved to {output_dir / 'correlation_heatmap.png'}")

    outlier_report = detect_outliers(df_clean)
    logger.info("Outlier report (IQR method, values NOT removed automatically):")
    print(outlier_report.to_string(index=False))

    X = select_features(df_clean, DEFAULT_FEATURES)

    logger.info("Scaling features...")
    X_scaled, scaler = scale_features(X)

    build_dendrogram(X_scaled, save_path=output_dir / "dendrogram.png")

    scores = find_best_k(X_scaled)
    logger.info("Silhouette scores by k:")
    for k, s in scores.items():
        logger.info(f"  k={k}: {s:.4f}")

    logger.info(f"Training Agglomerative Clustering with k={n_clusters}...")
    model, labels = train_agglomerative(X_scaled, n_clusters=n_clusters)
    df_clean = df_clean.copy()
    df_clean["Cluster"] = labels

    if run_kmeans_comparison:
        compare_with_kmeans(X_scaled, n_clusters=n_clusters, agglo_labels=labels)

    plot_two_feature_scatter(df_clean, x="PURCHASES", y="CREDIT_LIMIT",
                              save_path=output_dir / "cluster_plot.png")
    plot_pca_clusters(X_scaled, labels, save_path=output_dir / "pca_clusters.png")
    logger.info(f"Cluster plots saved to {output_dir}")

    profile = profile_clusters(df_clean, DEFAULT_FEATURES)

    evaluate_cluster_health(df_clean)

    recommendations = generate_business_recommendations(profile)
    recommendations.to_csv(output_dir / "cluster_profile.csv")
    logger.info("Pipeline complete. Cluster profile + recommendations:")
    print(recommendations[["count", "Label", "Recommended Action"]])

    return df_clean, recommendations, scores


def _parse_args():
    parser = argparse.ArgumentParser(description="Credit card customer segmentation pipeline")
    parser.add_argument("--k", type=int, default=DEFAULT_N_CLUSTERS,
                         help=f"Number of clusters (default: {DEFAULT_N_CLUSTERS})")
    parser.add_argument("--no-kmeans-comparison", action="store_true",
                         help="Skip the K-Means comparison step")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_pipeline(n_clusters=args.k, run_kmeans_comparison=not args.no_kmeans_comparison)