from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from .config import MODELS_DIR
from .data import basic_cleaning, load_data
from .features import add_time_features, select_cluster_features


def train_cluster_models() -> dict[str, dict[str, float | int]]:
    df = load_data()
    df = basic_cleaning(df)
    df = add_time_features(df)

    feature_df = select_cluster_features(df)
    if len(feature_df) < 5:
        raise ValueError("Poucos registros validos para clusterizacao. Use pelo menos 5 linhas.")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(feature_df)

    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans_labels = kmeans.fit_predict(X_scaled)

    dbscan = DBSCAN(eps=0.9, min_samples=5)
    dbscan_labels = dbscan.fit_predict(X_scaled)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": kmeans, "scaler": scaler}, MODELS_DIR / "kmeans.joblib")
    joblib.dump({"model": dbscan, "scaler": scaler}, MODELS_DIR / "dbscan.joblib")

    results = {
        "kmeans": {
            "clusters": int(len(set(kmeans_labels))),
            "silhouette": _safe_silhouette(X_scaled, kmeans_labels),
        },
        "dbscan": {
            "clusters_without_noise": int(len(set(label for label in dbscan_labels if label != -1))),
            "noise_points": int((dbscan_labels == -1).sum()),
            "silhouette": _safe_silhouette(X_scaled, dbscan_labels),
        },
    }

    return results


def _safe_silhouette(X_scaled, labels) -> float:
    unique_labels = set(labels)
    non_noise_labels = {label for label in unique_labels if label != -1}

    if len(non_noise_labels) < 2:
        return float("nan")

    valid_mask = labels != -1
    if valid_mask.sum() < 2:
        return float("nan")

    return float(silhouette_score(X_scaled[valid_mask], labels[valid_mask]))


def main() -> None:
    results = train_cluster_models()
    print("Resultados dos modelos nao supervisionados:")
    for model_name, metrics in results.items():
        formatted = " | ".join(f"{key}={value}" for key, value in metrics.items())
        print(f"- {model_name}: {formatted}")


if __name__ == "__main__":
    main()
