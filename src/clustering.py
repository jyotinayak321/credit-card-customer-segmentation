"""
clustering.py
--------------
The "engine" of the project: builds the dendrogram, tests k values with
silhouette score, trains the final AgglomerativeClustering model, and
profiles the resulting clusters.

Can be run standalone:
    python src/clustering.py
"""

import os
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

from data_loader import load_data, inspect_data
from preprocessing import clean_data, select_features, scale_features, DEFAULT_FEATURES


def build_dendrogram(X_scaled, save_path: str = "../outputs/dendrogram.png"):
    """Build and save the dendrogram using Ward linkage.

    Ward linkage merges clusters in the way that increases within-cluster
    variance the LEAST at each step - it tends to produce compact,
    evenly-sized clusters, which is why it's the most common default.
    """
    linked = linkage(X_scaled, method="ward")

    plt.figure(figsize=(12, 6))
    dendrogram(linked, truncate_mode="level", p=5)
    plt.xlabel("Customers / merged clusters")
    plt.ylabel("Distance")
    plt.title("Hierarchical Clustering Dendrogram")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Dendrogram saved to {save_path}")
    return linked


def find_best_k(X_scaled, k_range=range(2, 8)) -> dict:
    """Try several k values, return silhouette score for each.

    Silhouette score (-1 to 1): how similar a point is to its own cluster
    vs. the nearest other cluster. Closer to 1 = well separated clusters.
    Rule of thumb: pick the k with a HIGH score AND clusters that make
    business sense - don't blindly pick the max.
    """
    scores = {}
    for k in k_range:
        model = AgglomerativeClustering(n_clusters=k, metric="euclidean", linkage="ward")
        labels = model.fit_predict(X_scaled)
        scores[k] = silhouette_score(X_scaled, labels)
    return scores


def train_agglomerative(X_scaled, n_clusters: int = 4):
    """Train the final Agglomerative Clustering model."""
    model = AgglomerativeClustering(n_clusters=n_clusters, metric="euclidean", linkage="ward")
    labels = model.fit_predict(X_scaled)
    return model, labels


def profile_clusters(df: pd.DataFrame, features: list, cluster_col: str = "Cluster") -> pd.DataFrame:
    """Compute the average behaviour of each cluster - this turns
    anonymous cluster numbers (0,1,2,3) into interpretable customer
    profiles (e.g. 'high spenders', 'cash-advance heavy')."""
    profile = df.groupby(cluster_col)[features].mean().round(2)
    profile["count"] = df[cluster_col].value_counts().sort_index()
    return profile


def run_pipeline(data_path: str = "../data/credit_card_customers.csv",
                  n_clusters: int = 4,
                  output_dir: str = "../outputs"):
    """Runs the full workflow end-to-end and saves outputs."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Load + inspect
    df_raw = load_data(data_path)
    inspect_data(df_raw)

    # 2. Clean
    df_clean = clean_data(df_raw)

    # 3. Feature selection
    X = select_features(df_clean, DEFAULT_FEATURES)

    # 4. Scale
    X_scaled, scaler = scale_features(X)

    # 5. Dendrogram
    build_dendrogram(X_scaled, save_path=os.path.join(output_dir, "dendrogram.png"))

    # 6. Try multiple k values
    scores = find_best_k(X_scaled)
    print("\nSilhouette scores by k:")
    for k, s in scores.items():
        print(f"  k={k}: {s:.4f}")

    # 7. Train final model
    model, labels = train_agglomerative(X_scaled, n_clusters=n_clusters)
    df_clean["Cluster"] = labels

    # 8. Profile clusters
    profile = profile_clusters(df_clean, DEFAULT_FEATURES)
    profile.to_csv(os.path.join(output_dir, "cluster_profile.csv"))
    print("\nCLUSTER PROFILE:")
    print(profile)

    return df_clean, profile, scores


if __name__ == "__main__":
    run_pipeline()
