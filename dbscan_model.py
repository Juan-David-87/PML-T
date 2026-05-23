import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from sklearn.model_selection import train_test_split
 
warnings.filterwarnings("ignore")
 
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "indice_gobierno_digital.csv")
 
# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────
df_raw = pd.read_csv(CSV_PATH, encoding="utf-8", thousands=".")
 
# Score columns stored with comma as decimal separator → convert
SCORE_COLS = [
    "PUNTAJE ENTIDAD",
    "PROMEDIO GRUPO PAR",
    "MÁXIMO GRUPO PAR",
    "MÍNIMO GRUPO PAR",
]
for col in SCORE_COLS:
    df_raw[col] = pd.to_numeric(
        df_raw[col].astype(str).str.replace(",", "."), errors="coerce"
    )
 
# Filter to most recent complete year
df21 = df_raw[df_raw["VIGENCIA"] == 2021].copy()
 
# Pivot: one row per entity × one column per sub-index score
META_COLS = [
    "CÓDIGO_SIGEP", "ENTIDAD", "MUNICIPIO",
    "DEPARTAMENTO", "ORDEN", "NATURALEZA_JURÍDICA",
]
pivot = df21.pivot_table(
    index=META_COLS,
    columns="ÍNDICE",
    values="PUNTAJE ENTIDAD",
    aggfunc="mean",
).reset_index()
pivot.columns.name = None
 
# Keep only sub-indices present in ≥ 50% of entities (removes 2 sparse ones)
ALL_FEATURES = [c for c in pivot.columns if c not in META_COLS]
threshold = len(pivot) * 0.5
FEATURE_COLS = [c for c in ALL_FEATURES if pivot[c].notna().sum() >= threshold]
pivot = pivot.dropna(subset=FEATURE_COLS).reset_index(drop=True)
 
# Enrich with mean percentile / quintile per entity (from raw data)
agg_extra = (
    df21.groupby(META_COLS)
    .agg(mean_pct=("PERCENTIL GRUPO PAR", "mean"),
         mean_quintil=("QUINTIL GRUPO PAR", "mean"))
    .reset_index()
)
pivot = pivot.merge(agg_extra[META_COLS + ["mean_pct", "mean_quintil"]],
                    on=META_COLS, how="left")
 
X_raw = pivot[FEATURE_COLS].values
print(f"[1. Data]  Entities: {len(pivot):,}  |  Features: {len(FEATURE_COLS)}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET SPLIT
# ─────────────────────────────────────────────────────────────────────────────
train_idx, test_idx = train_test_split(
    np.arange(len(pivot)), test_size=0.20, random_state=42
)
X_train_raw = X_raw[train_idx]
X_test_raw  = X_raw[test_idx]
print(f"[2. Split] Train: {len(train_idx):,}  |  Test: {len(test_idx):,}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING — StandardScaler + PCA 2-D
# ─────────────────────────────────────────────────────────────────────────────
scaler       = StandardScaler()
X_train_sc   = scaler.fit_transform(X_train_raw)
X_test_sc    = scaler.transform(X_test_raw)
X_all_sc     = scaler.transform(X_raw)
 
pca_2d       = PCA(n_components=2, random_state=42)
X_train_2d   = pca_2d.fit_transform(X_train_sc)
X_all_2d     = pca_2d.transform(X_all_sc)
pca_variance = [round(float(v) * 100, 2) for v in pca_2d.explained_variance_ratio_]
print(f"[3. PCA]   Explained variance: PC1={pca_variance[0]}%  PC2={pca_variance[1]}%")
 
# ─────────────────────────────────────────────────────────────────────────────
# 4. HYPERPARAMETER CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
# 4a. k-distance graph to estimate elbow ε
K = 5
nbrs = NearestNeighbors(n_neighbors=K).fit(X_train_sc)
distances, _ = nbrs.kneighbors(X_train_sc)
k_dist = np.sort(distances[:, K - 1])[::-1]
diffs2 = np.diff(np.diff(k_dist))
elbow_idx = int(np.argmax(np.abs(diffs2))) + 1
recommended_eps = round(float(k_dist[elbow_idx]), 3)
print(f"[4. Hyper] k-distance elbow ε: {recommended_eps}")
 
# 4b. Grid search — eps × min_samples on training set
EPS_GRID = [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.5]
MS_GRID  = [3, 5, 8, 10, 15]
grid_rows = []
 
for eps in EPS_GRID:
    for ms in MS_GRID:
        db  = DBSCAN(eps=eps, min_samples=ms).fit(X_train_sc)
        lbl = db.labels_
        nc  = len(set(lbl)) - (1 if -1 in lbl else 0)
        noise_pct = round(100 * np.sum(lbl == -1) / len(lbl), 2)
        valid = lbl[lbl != -1]
        sil = (round(silhouette_score(X_train_sc[lbl != -1], valid), 4)
               if nc > 1 and len(valid) > nc else None)
        dbi = (round(davies_bouldin_score(X_train_sc[lbl != -1], valid), 4)
               if nc > 1 else None)
        chi = (round(calinski_harabasz_score(X_train_sc[lbl != -1], valid), 2)
               if nc > 1 else None)
        grid_rows.append(dict(
            eps=eps, min_samples=ms, n_clusters=nc,
            noise_pct=noise_pct, silhouette=sil,
            davies_bouldin=dbi, calinski_harabasz=chi, is_best=False,
        ))
 
# Select best: highest Silhouette with noise < 30%
valid_configs = [r for r in grid_rows if r["silhouette"] is not None
                 and r["noise_pct"] < 30]
best_config = (max(valid_configs, key=lambda r: r["silhouette"])
               if valid_configs else {"eps": 1.0, "min_samples": 5})
BEST_EPS = best_config["eps"]
BEST_MS  = best_config["min_samples"]
 
for r in grid_rows:
    r["is_best"] = (r["eps"] == BEST_EPS and r["min_samples"] == BEST_MS)
print(f"[4. Hyper] Best → eps={BEST_EPS}  min_samples={BEST_MS}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 5. TRAINING — fit DBSCAN on full scaled dataset
# ─────────────────────────────────────────────────────────────────────────────
model      = DBSCAN(eps=BEST_EPS, min_samples=BEST_MS)
all_labels = model.fit_predict(X_all_sc)
 
pivot["cluster"] = all_labels
n_clusters = len(set(all_labels)) - (1 if -1 in all_labels else 0)
noise_n    = int(np.sum(all_labels == -1))
print(f"[5. Train] Clusters found: {n_clusters}  |  Noise points: {noise_n}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 6. VALIDATION — internal metrics + train-set stability check
# ─────────────────────────────────────────────────────────────────────────────
valid_mask = all_labels != -1
if n_clusters > 1 and valid_mask.sum() > n_clusters:
    sil_full = round(silhouette_score(X_all_sc[valid_mask], all_labels[valid_mask]), 4)
    dbi_full = round(davies_bouldin_score(X_all_sc[valid_mask], all_labels[valid_mask]), 4)
    chi_full = round(calinski_harabasz_score(X_all_sc[valid_mask], all_labels[valid_mask]), 2)
else:
    sil_full = dbi_full = chi_full = None
 
# Stability: re-run on train subset only
db_tr      = DBSCAN(eps=BEST_EPS, min_samples=BEST_MS).fit(X_train_sc)
tr_lbl     = db_tr.labels_
n_tr_cl    = len(set(tr_lbl)) - (1 if -1 in tr_lbl else 0)
noise_tr_p = round(100 * np.sum(tr_lbl == -1) / len(tr_lbl), 2)
sil_tr = (round(silhouette_score(X_train_sc[tr_lbl != -1], tr_lbl[tr_lbl != -1]), 4)
          if n_tr_cl > 1 else None)
 
print(f"[6. Valid] Silhouette={sil_full}  DBI={dbi_full}  CHI={chi_full}")
print(f"           Train stability → clusters={n_tr_cl}  noise%={noise_tr_p}  sil={sil_tr}")
 
# ─────────────────────────────────────────────────────────────────────────────
# 7. CLUSTER PROFILING
# ─────────────────────────────────────────────────────────────────────────────
cluster_profiles = []
for cid in sorted(set(all_labels)):
    mask  = all_labels == cid
    grp   = pivot[mask]
    means = {c: round(float(grp[c].mean()), 2) for c in FEATURE_COLS}
    cluster_profiles.append(dict(
        cluster_id=int(cid),
        label="Noise" if cid == -1 else f"Cluster {cid}",
        size=int(mask.sum()),
        means=means,
        top_departments=grp["DEPARTAMENTO"].value_counts().head(5).to_dict(),
        orden_dist=grp["ORDEN"].value_counts().to_dict(),
        mean_pct=round(float(grp["mean_pct"].mean()), 1),
        mean_quintil=round(float(grp["mean_quintil"].mean()), 1),
    ))
 
# ─────────────────────────────────────────────────────────────────────────────
# 8. PREDICTION GENERATION — risk labelling
# ─────────────────────────────────────────────────────────────────────────────
GD_COL = "Gobierno Digital"
non_noise = [p for p in cluster_profiles if p["cluster_id"] != -1]
sorted_cl = sorted(non_noise,
    key=lambda p: p["means"].get(GD_COL, float(np.mean(list(p["means"].values())))))
 
risk_map = {-1: "Indeterminado"}
n_nc = len(sorted_cl)
for rank, cp in enumerate(sorted_cl):
    frac = rank / max(n_nc - 1, 1)
    risk_map[cp["cluster_id"]] = ("Alto" if frac < 0.34
                                  else "Medio" if frac < 0.67
                                  else "Bajo")
 
pivot["risk_level"] = pivot["cluster"].map(risk_map)
 
for cp in cluster_profiles:
    cp["risk"] = risk_map.get(cp["cluster_id"], "-")
 
# ─────────────────────────────────────────────────────────────────────────────
# 9. BUILD & EXPORT JSON
# ─────────────────────────────────────────────────────────────────────────────
step = max(1, len(k_dist) // 200)
kdist_plot = [round(float(v), 4) for v in k_dist[::step]]
 
scatter_data = [
    dict(
        x=round(float(X_all_2d[i, 0]), 4),
        y=round(float(X_all_2d[i, 1]), 4),
        cluster=int(all_labels[i]),
        entidad=str(pivot.loc[i, "ENTIDAD"]),
        municipio=str(pivot.loc[i, "MUNICIPIO"]),
        departamento=str(pivot.loc[i, "DEPARTAMENTO"]),
        risk=str(pivot.loc[i, "risk_level"]),
        gd_score=(round(float(pivot.loc[i, GD_COL]), 1)
                  if pd.notna(pivot.loc[i, GD_COL]) else None),
    )
    for i in range(len(pivot))
]
 
dept_risk = (
    pivot[pivot["cluster"] != -1]
    .groupby("DEPARTAMENTO")["risk_level"]
    .agg(lambda x: x.value_counts().index[0])
    .reset_index()
    .rename(columns={"risk_level": "dominant_risk"})
)
dept_counts = pivot.groupby("DEPARTAMENTO").size().reset_index(name="count")
dept_gd     = (pivot.groupby("DEPARTAMENTO")[GD_COL]
               .mean().round(2).reset_index()
               .rename(columns={GD_COL: "gd_mean"}))
dept_table  = (dept_risk
               .merge(dept_counts, on="DEPARTAMENTO")
               .merge(dept_gd, on="DEPARTAMENTO")
               .sort_values("count", ascending=False))
 
results = dict(
    # Summary
    total_entities=int(len(pivot)),
    train_size=int(len(train_idx)),
    test_size=int(len(test_idx)),
    n_clusters=n_clusters,
    noise_count=noise_n,
    noise_pct=round(100 * noise_n / len(pivot), 2),
    feature_cols=FEATURE_COLS,
    pca_variance=pca_variance,
    # Hyperparameters
    best_eps=BEST_EPS,
    best_min_samples=BEST_MS,
    recommended_eps=recommended_eps,
    grid_results=grid_rows,
    # Validation
    silhouette=sil_full,
    davies_bouldin=dbi_full,
    calinski_harabasz=chi_full,
    silhouette_train=sil_tr,
    n_train_clusters=n_tr_cl,
    noise_train_pct=noise_tr_p,
    # Clustering
    cluster_profiles=cluster_profiles,
    # Visualisation
    scatter_data=scatter_data,
    kdist_plot=kdist_plot,
    # Predictions
    dept_table=dept_table.to_dict(orient="records"),
    risk_distribution=pivot["risk_level"].value_counts().to_dict(),
    overall_gd_mean=round(float(pivot[GD_COL].mean()), 2),
    overall_gd_std=round(float(pivot[GD_COL].std()), 2),
)
 
OUT_PATH = os.path.join(BASE_DIR, "static", "data", "dbscan_results.json")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
 
print("\n=== DBSCAN COMPLETE ===")
print(f"  Entities        : {results['total_entities']:,}")
print(f"  Clusters        : {results['n_clusters']}")
print(f"  Noise           : {results['noise_count']} ({results['noise_pct']}%)")
print(f"  Silhouette      : {results['silhouette']}")
print(f"  Davies-Bouldin  : {results['davies_bouldin']}")
print(f"  Calinski-Harabasz: {results['calinski_harabasz']}")
print(f"  Risk distribution: {results['risk_distribution']}")
print(f"\n  Results → {OUT_PATH}")