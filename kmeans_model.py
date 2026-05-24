"""
kmeans_model.py
───────────────
K-Means clustering model for classifying Colombian public entities
by their Digital Government Index (Índice de Gobierno Digital).

Dataset : indice_gobierno_digital.csv  (77 163 rows, 18 columns)
Features: 7 sub-index scores per entity (I18, I20, I21, I82, I83, I85, POL06)
Output  : 4 clusters ranked by overall digital maturity
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

# ─── Configuration ────────────────────────────────────────────────────────────────
DATA_PATH   = "indice_gobierno_digital.csv"
OUT_DIR     = "static/img/kmeans"
FEATURES    = ["I18", "I20", "I21", "I82", "I83", "I85", "POL06"]
K_FINAL     = 4
RANDOM_STATE = 42
PALETTE     = ["#22c55e", "#38bdf8", "#f59e0b", "#f43f5e"]

CLUSTER_LABELS = {
    0: "Medium-High Digital",
    1: "High Digital",
    2: "Low Digital",
    3: "Medium Digital",
}

os.makedirs(OUT_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
# 1 ▸ DATA LOADING & PREPROCESSING
# ═══════════════════════════════════════════════════════════════════════════════

def load_and_prepare(path: str):
    """Load CSV, fix decimal separators, pivot to one entity per row."""
    df = pd.read_csv(path)

    def to_float(s):
        try:
            return float(str(s).replace(",", "."))
        except Exception:
            return np.nan

    for col in ["PUNTAJE ENTIDAD", "PROMEDIO GRUPO PAR",
                "MÁXIMO GRUPO PAR", "MÍNIMO GRUPO PAR"]:
        df[col + "_f"] = df[col].apply(to_float)

    pivot = df.pivot_table(
        index=["CÓDIGO_SIGEP", "ENTIDAD", "ORDEN", "DEPARTAMENTO", "MUNICIPIO"],
        columns="ID_ÍNDICE",
        values="PUNTAJE ENTIDAD_f",
        aggfunc="mean",
    ).reset_index()

    return pivot


# ═══════════════════════════════════════════════════════════════════════════════
# 2 ▸ DATASET SPLIT (train / validation)
#     K-Means is unsupervised; we simulate a hold-out split by fitting on 80 %
#     of the data and predicting on the remaining 20 % to validate label
#     consistency via Silhouette on the unseen subset.
# ═══════════════════════════════════════════════════════════════════════════════

def split_data(X_scaled: np.ndarray):
    n = len(X_scaled)
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.permutation(n)
    split = int(0.8 * n)
    train_idx = idx[:split]
    val_idx   = idx[split:]
    return train_idx, val_idx


# ═══════════════════════════════════════════════════════════════════════════════
# 3 ▸ HYPERPARAMETER SELECTION — Elbow + Silhouette
# ═══════════════════════════════════════════════════════════════════════════════

def hyperparameter_search(X_scaled: np.ndarray, k_range=range(2, 11)):
    inertias, silhouettes = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lbls = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, lbls))

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor("#0f172a")
    for ax in axes:
        ax.set_facecolor("#1e293b")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#334155")

    axes[0].plot(list(k_range), inertias, "o-", color="#22c55e", lw=2.5, ms=8)
    axes[0].axvline(K_FINAL, color="#f59e0b", ls="--", lw=1.5, label=f"K={K_FINAL}")
    axes[0].set_title("Elbow Method — WCSS vs K", color="white", fontsize=13, pad=12)
    axes[0].set_xlabel("K", color="#94a3b8")
    axes[0].set_ylabel("Inertia (WCSS)", color="#94a3b8")
    axes[0].legend(facecolor="#1e293b", labelcolor="white")
    axes[0].grid(True, color="#334155", alpha=0.5)

    axes[1].plot(list(k_range), silhouettes, "s-", color="#38bdf8", lw=2.5, ms=8)
    axes[1].axvline(K_FINAL, color="#f59e0b", ls="--", lw=1.5, label=f"K={K_FINAL}")
    axes[1].set_title("Silhouette Score vs K", color="white", fontsize=13, pad=12)
    axes[1].set_xlabel("K", color="#94a3b8")
    axes[1].set_ylabel("Silhouette Score", color="#94a3b8")
    axes[1].legend(facecolor="#1e293b", labelcolor="white")
    axes[1].grid(True, color="#334155", alpha=0.5)

    plt.tight_layout(pad=2)
    plt.savefig(f"{OUT_DIR}/elbow_silhouette.png", dpi=130,
                bbox_inches="tight", facecolor="#0f172a")
    plt.close()

    best_sil_k = list(k_range)[int(np.argmax(silhouettes))]
    return inertias, silhouettes, best_sil_k


# ═══════════════════════════════════════════════════════════════════════════════
# 4 ▸ TRAINING
# ═══════════════════════════════════════════════════════════════════════════════

def train_kmeans(X_train: np.ndarray):
    km = KMeans(
        n_clusters=K_FINAL,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=RANDOM_STATE,
    )
    km.fit(X_train)

    # ── Convergence plot ──────────────────────────────────────────────────────
    iters_inertia = []
    for max_it in range(1, 31):
        km_it = KMeans(n_clusters=K_FINAL, random_state=RANDOM_STATE,
                       n_init=1, max_iter=max_it)
        km_it.fit(X_train)
        iters_inertia.append(km_it.inertia_)

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

    ax.plot(range(1, 31), iters_inertia, "o-", color="#a78bfa", lw=2.5, ms=7)
    ax.fill_between(range(1, 31), iters_inertia, alpha=0.15, color="#a78bfa")
    ax.set_title("Training Convergence — Inertia per Iteration", color="white",
                 fontsize=13, pad=12)
    ax.set_xlabel("Iteration", color="#94a3b8")
    ax.set_ylabel("Inertia", color="#94a3b8")
    ax.grid(True, color="#334155", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/convergence.png", dpi=130,
                bbox_inches="tight", facecolor="#0f172a")
    plt.close()

    return km


# ═══════════════════════════════════════════════════════════════════════════════
# 5 ▸ VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate(km, X_train, X_val):
    train_labels = km.predict(X_train)
    val_labels   = km.predict(X_val)

    metrics = {
        "train_silhouette": silhouette_score(X_train, train_labels),
        "val_silhouette":   silhouette_score(X_val,   val_labels),
        "train_db":         davies_bouldin_score(X_train, train_labels),
        "val_db":           davies_bouldin_score(X_val,   val_labels),
        "inertia":          km.inertia_,
        "n_iter":           km.n_iter_,
    }
    return metrics, train_labels


# ═══════════════════════════════════════════════════════════════════════════════
# 6 ▸ VISUALISATIONS (PCA scatter, cluster profiles, heatmap, pie)
# ═══════════════════════════════════════════════════════════════════════════════

def build_visualisations(X_scaled, labels, data_df, features):
    # ── PCA 2D ────────────────────────────────────────────────────────────────
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X2  = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

    for c in range(K_FINAL):
        mask = labels == c
        ax.scatter(X2[mask, 0], X2[mask, 1], c=PALETTE[c],
                   label=f"Cluster {c} — {CLUSTER_LABELS[c]} (n={mask.sum()})",
                   alpha=0.72, s=35, edgecolors="none")

    centers_2d = pca.transform(km_global.cluster_centers_)
    for c in range(K_FINAL):
        ax.scatter(*centers_2d[c], marker="*", s=350, c=PALETTE[c],
                   edgecolors="white", lw=1.5, zorder=5)

    ax.set_title("K-Means Clusters — PCA 2D Projection", color="white",
                 fontsize=14, pad=14)
    ax.set_xlabel(
        f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)",
        color="#94a3b8")
    ax.set_ylabel(
        f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)",
        color="#94a3b8")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax.grid(True, color="#334155", alpha=0.4)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/pca_scatter.png", dpi=130,
                bbox_inches="tight", facecolor="#0f172a")
    plt.close()

    # ── Cluster profiles bar ──────────────────────────────────────────────────
    cluster_df    = data_df.copy()
    cluster_df["cluster"] = labels
    cluster_means = cluster_df.groupby("cluster")[features].mean()

    x     = np.arange(len(features))
    width = 0.2

    fig, ax = plt.subplots(figsize=(13, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")
    ax.tick_params(colors="#94a3b8")
    for sp in ax.spines.values():
        sp.set_edgecolor("#334155")

    for i, (idx, row) in enumerate(cluster_means.iterrows()):
        ax.bar(x + i * width, row.values, width,
               label=f"Cluster {idx} — {CLUSTER_LABELS[idx]}",
               color=PALETTE[i], alpha=0.88)

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(features, color="#94a3b8", rotation=20)
    ax.set_title("Average Score per Sub-Index by Cluster", color="white",
                 fontsize=14, pad=12)
    ax.set_ylabel("Mean Score (0–100)", color="#94a3b8")
    ax.legend(facecolor="#1e293b", labelcolor="white", fontsize=9)
    ax.grid(True, axis="y", color="#334155", alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cluster_profiles.png", dpi=130,
                bbox_inches="tight", facecolor="#0f172a")
    plt.close()

    # ── Heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#1e293b")

    im = ax.imshow(cluster_means.values, cmap="RdYlGn",
                   aspect="auto", vmin=20, vmax=100)
    ax.set_xticks(range(len(features)))
    ax.set_xticklabels(features, color="white", fontsize=11)
    ax.set_yticks(range(K_FINAL))
    ax.set_yticklabels([f"Cluster {c}" for c in range(K_FINAL)],
                       color="white", fontsize=11)
    for i in range(K_FINAL):
        for j in range(len(features)):
            ax.text(j, i, f"{cluster_means.values[i,j]:.1f}",
                    ha="center", va="center",
                    color="black", fontweight="bold", fontsize=10)

    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.ax.tick_params(colors="white")
    ax.set_title("Cluster Mean Scores Heatmap (by Sub-Index)",
                 color="white", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/heatmap.png", dpi=130,
                bbox_inches="tight", facecolor="#0f172a")
    plt.close()

    # ── Pie ───────────────────────────────────────────────────────────────────
    sizes = [(labels == c).sum() for c in range(K_FINAL)]
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor("#0f172a")
    ax.set_facecolor("#0f172a")
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=[f"Cluster {c}\n{CLUSTER_LABELS[c]}\n(n={s})"
                for c, s in enumerate(sizes)],
        colors=PALETTE,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(linewidth=2, edgecolor="#0f172a"),
        pctdistance=0.82,
    )
    for t in texts:
        t.set_color("white")
        t.set_fontsize(9)
    for a in autotexts:
        a.set_color("#0f172a")
        a.set_fontweight("bold")
    ax.set_title("Cluster Size Distribution", color="white",
                 fontsize=13, pad=14)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/cluster_sizes.png", dpi=130,
                bbox_inches="tight", facecolor="#0f172a")
    plt.close()

    return cluster_means


# ═══════════════════════════════════════════════════════════════════════════════
# 7 ▸ PREDICTION EXAMPLES
# ═══════════════════════════════════════════════════════════════════════════════

def generate_predictions(km, scaler, data_df, meta_df, n=8):
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.choice(len(data_df), n, replace=False)
    sample = data_df.iloc[idx]
    pred   = km.predict(scaler.transform(sample))
    rows   = []
    for i, (_, row) in enumerate(sample.iterrows()):
        m = meta_df.iloc[idx[i]]
        c = pred[i]
        rows.append({
            "entity":    m["ENTIDAD"][:55],
            "dept":      m["DEPARTAMENTO"],
            "orden":     m["ORDEN"],
            "cluster":   c,
            "label":     CLUSTER_LABELS[c],
            "pol06":     round(row["POL06"], 1),
            "i82":       round(row["I82"],   1),
            "i18":       round(row["I18"],   1),
        })
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN  (also used by Flask via get_kmeans_results())
# ═══════════════════════════════════════════════════════════════════════════════

km_global   = None   # module-level model kept for visualisations
scaler_global = None

def get_kmeans_results():
    global km_global, scaler_global

    # 1 · Load
    pivot = load_and_prepare(DATA_PATH)
    data  = pivot[FEATURES].dropna()
    meta  = pivot.loc[data.index, ["ENTIDAD", "ORDEN", "DEPARTAMENTO", "MUNICIPIO"]]
    data.reset_index(drop=True, inplace=True)
    meta.reset_index(drop=True, inplace=True)

    # 2 · Scale
    scaler = StandardScaler()
    X      = scaler.fit_transform(data)
    scaler_global = scaler

    # 3 · Hyperparameter search
    _, silhouettes, best_sil_k = hyperparameter_search(X)

    # 4 · Split
    train_idx, val_idx = split_data(X)
    X_train = X[train_idx]
    X_val   = X[val_idx]

    # 5 · Train
    km = train_kmeans(X_train)
    km_global = km

    # 6 · Full-dataset labels (for plots)
    all_labels = km.predict(X)

    # 7 · Validate
    metrics, _ = validate(km, X_train, X_val)

    # 8 · Visualisations
    cluster_means = build_visualisations(X, all_labels, data, FEATURES)

    # 9 · Predictions
    preds = generate_predictions(km, scaler, data, meta)

    # 10 · Comparison table (K=2..6)
    comparison = []
    for k in range(2, 7):
        km_k  = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        lbls  = km_k.fit_predict(X)
        comparison.append({
            "k":          k,
            "inertia":    round(km_k.inertia_, 1),
            "silhouette": round(silhouette_score(X, lbls), 4),
            "db":         round(davies_bouldin_score(X, lbls), 4),
            "selected":   k == K_FINAL,
        })

    return {
        "metrics":       metrics,
        "cluster_means": cluster_means.round(2).to_dict(),
        "predictions":   preds,
        "comparison":    comparison,
        "n_train":       len(train_idx),
        "n_val":         len(val_idx),
        "n_total":       len(data),
        "best_sil_k":    best_sil_k,
        "k_final":       K_FINAL,
        "features":      FEATURES,
        "cluster_labels": CLUSTER_LABELS,
    }


if __name__ == "__main__":
    res = get_kmeans_results()
    print("\n" + "═" * 60)
    print("  K-MEANS MODEL SUMMARY")
    print("═" * 60)
    print(f"  Total entities   : {res['n_total']}")
    print(f"  Training set     : {res['n_train']}")
    print(f"  Validation set   : {res['n_val']}")
    print(f"  Final K          : {res['k_final']}")
    print(f"  Train Silhouette : {res['metrics']['train_silhouette']:.4f}")
    print(f"  Val  Silhouette  : {res['metrics']['val_silhouette']:.4f}")
    print(f"  Train DB Index   : {res['metrics']['train_db']:.4f}")
    print(f"  Val  DB Index    : {res['metrics']['val_db']:.4f}")
    print(f"  Inertia (WCSS)   : {res['metrics']['inertia']:.2f}")
    print(f"  Convergence in   : {res['metrics']['n_iter']} iterations")
    print("\n  Cluster Assignments:")
    for c, lbl in res['cluster_labels'].items():
        print(f"    Cluster {c} → {lbl}")
    print("\n  Sample Predictions:")
    for p in res['predictions']:
        print(f"    [{p['cluster']}] {p['entity'][:45]:<45} POL06={p['pol06']}")
    print("═" * 60)