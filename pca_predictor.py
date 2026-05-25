"""
pca_predictor.py
PCA-based prediction system for the Digital Government Index.
Fits a PCA + KMeans pipeline and exposes a predict() function
for the Flask prediction endpoint.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# ── Configuration ──────────────────────────────────────────────────────────────
DATA_PATH    = "indice_gobierno_digital.csv"
FEATURES     = [
    "PUNTAJE ENTIDAD",
    "PROMEDIO GRUPO PAR",
    "MÁXIMO GRUPO PAR",
    "MÍNIMO GRUPO PAR",
    "QUINTIL GRUPO PAR",
    "PERCENTIL GRUPO PAR",
]
N_COMPONENTS = 2
N_CLUSTERS   = 3
RANDOM_STATE = 42

# Cluster labels (ordered low→high digital performance after fitting)
CLUSTER_LABELS = {
    0: "Low Digital Performance",
    1: "Medium Digital Performance",
    2: "High Digital Performance",
}
CLUSTER_RISKS = {
    0: "High",
    1: "Medium",
    2: "Low",
}
CLUSTER_COLORS = {
    0: "#f43f5e",
    1: "#f59e0b",
    2: "#22c55e",
}
CLUSTER_DESCRIPTIONS = {
    0: "Entities in this cluster score significantly below national averages. "
       "Priority targets for digital infrastructure investment.",
    1: "Entities with moderate digital maturity. Improvement programs would "
       "yield measurable gains in government service delivery.",
    2: "High-performing entities with strong digital governance indicators. "
       "Potential reference benchmarks for lower clusters.",
}

# ── Module-level singletons (loaded once) ─────────────────────────────────────
_scaler: StandardScaler = None
_pca:    PCA            = None
_kmeans: KMeans         = None
_cluster_order: dict    = None   # raw_label → ordered_label
_feature_stats: dict    = None   # min/max/mean per feature for UI hints


def _load_and_fit():
    """Load data, fit StandardScaler → PCA → KMeans. Called once."""
    global _scaler, _pca, _kmeans, _cluster_order, _feature_stats

    df = pd.read_csv(DATA_PATH)

    # Decimal-separator fix (comma → dot)
    for col in FEATURES:
        df[col] = pd.to_numeric(
            df[col].astype(str).str.replace(",", "."), errors="coerce"
        )

    data = df[FEATURES].dropna()

    # Feature stats for UI sliders / placeholders
    _feature_stats = {
        col: {
            "min":  round(float(data[col].min()), 2),
            "max":  round(float(data[col].max()), 2),
            "mean": round(float(data[col].mean()), 2),
        }
        for col in FEATURES
    }

    X = data.values

    # Fit pipeline
    _scaler = StandardScaler()
    X_scaled = _scaler.fit_transform(X)

    _pca = PCA(n_components=N_COMPONENTS, random_state=RANDOM_STATE)
    X_pca = _pca.fit_transform(X_scaled)

    _kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        n_init=20,
        random_state=RANDOM_STATE,
    )
    raw_labels = _kmeans.fit_predict(X_pca)

    # Map raw cluster IDs to ordered IDs (by ascending centroid PC1 value)
    centers = _kmeans.cluster_centers_
    order   = np.argsort(centers[:, 0])        # sort by PC1 (performance axis)
    _cluster_order = {int(raw): int(rank) for rank, raw in enumerate(order)}


def get_feature_stats() -> dict:
    """Return min/max/mean per feature (for form hints)."""
    if _scaler is None:
        _load_and_fit()
    return _feature_stats


def predict(values: dict) -> dict:
    """
    Predict cluster for a single entity.

    Parameters
    ----------
    values : dict  {feature_name: float_value}

    Returns
    -------
    dict with:
        cluster          int   (ordered 0-low … 2-high)
        label            str
        risk             str
        color            str
        description      str
        pc1              float
        pc2              float
        pc1_variance     float
        pc2_variance     float
        confidence       float   (0–1, based on distance to centroid)
        cluster_probs    list[float]   probabilities for each cluster
        pca_components   list[dict]    feature loadings
    """
    if _scaler is None:
        _load_and_fit()

    # Build input vector (same order as FEATURES)
    x = np.array([[values[f] for f in FEATURES]], dtype=float)

    x_scaled = _scaler.transform(x)
    x_pca    = _pca.transform(x_scaled)

    raw_label = int(_kmeans.predict(x_pca)[0])
    cluster   = _cluster_order[raw_label]

    # Distances to all centroids → pseudo-probabilities via softmax(-dist)
    dists = np.linalg.norm(
        _kmeans.cluster_centers_ - x_pca, axis=1
    )                                                  # shape (K,)
    # Map raw→ordered distance array
    ordered_dists = np.zeros(N_CLUSTERS)
    for raw, rank in _cluster_order.items():
        ordered_dists[rank] = dists[raw]

    # Softmax over negative distances
    neg = -ordered_dists
    neg -= neg.max()
    exp = np.exp(neg)
    probs = (exp / exp.sum()).tolist()

    confidence = float(probs[cluster])

    # Feature loadings
    loadings = [
        {
            "feature": f,
            "pc1": round(float(_pca.components_[0, i]), 4),
            "pc2": round(float(_pca.components_[1, i]), 4),
        }
        for i, f in enumerate(FEATURES)
    ]

    return {
        "cluster":        cluster,
        "label":          CLUSTER_LABELS[cluster],
        "risk":           CLUSTER_RISKS[cluster],
        "color":          CLUSTER_COLORS[cluster],
        "description":    CLUSTER_DESCRIPTIONS[cluster],
        "pc1":            round(float(x_pca[0, 0]), 4),
        "pc2":            round(float(x_pca[0, 1]), 4),
        "pc1_variance":   round(float(_pca.explained_variance_ratio_[0] * 100), 2),
        "pc2_variance":   round(float(_pca.explained_variance_ratio_[1] * 100), 2),
        "confidence":     round(confidence, 4),
        "cluster_probs":  [round(p, 4) for p in probs],
        "pca_components": loadings,
        "input_values":   {f: values[f] for f in FEATURES},
    }


# Pre-load on import so the first request is fast
_load_and_fit()