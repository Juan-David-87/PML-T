"""
K-Means Clustering Pipeline
Dataset: Índice de Gobierno Digital - Histórico
Target: Group municipalities/departments by digital governance risk level
Output: kmeans_results.json  (consumed by the Flask HTML template)
"""

import json
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score
)
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# ─── 0. LOAD & BASIC CLEANING ────────────────────────────────────────────────
df = pd.read_csv("indice_gobierno_digital.csv")
print(f"Columnas originales detectadas: {df.columns.tolist()}")

# Convertir todo a MAYÚSCULAS y cambiar espacios por guiones bajos primero
df.columns = df.columns.str.strip().str.upper().str.replace(" ", "_")

# Remover todas las tildes de los nombres de las columnas
df.columns = (
    df.columns.str.replace("Á", "A")
              .str.replace("É", "E")
              .str.replace("Í", "I")
              .str.replace("Ó", "O")
              .str.replace("Ú", "U")
              .str.replace("Ñ", "N")
)

# Mapear con precisión quirúrgica
RENAME = {
    "CODIGO_SIGEP":       "codigo_sigep",
    "ENTIDAD":            "entidad",
    "ORDEN":              "orden",
    "SECTOR":             "sector",
    "NATURALEZA_JURIDICA":"naturaleza_juridica",
    "ID_DEPARTAMENTO":    "id_departamento",
    "DEPARTAMENTO":       "departamento",
    "ID_MUNICIPIO":       "id_municipio",
    "MUNICIPIO":          "municipio",
    "VIGENCIA":           "vigencia",
    "ID_INDICE":          "id_indice",
    "INDICE":             "indice",
    "PUNTAJE_ENTIDAD":    "puntaje_entidad",
    "PROMEDIO_GRUPO_PAR": "promedio_grupo_par",
    "MAXIMO_GRUPO_PAR":   "maximo_grupo_par",
    "MINIMO_GRUPO_PAR":   "minimo_grupo_par",
    "QUINTIL_GRUPO_PAR":  "quintil_grupo_par",
    "PERCENTIL_GRUPO_PAR":"percentil_grupo_par",
}
df.rename(columns=RENAME, inplace=True)

# ─── 1. DATASET SPLIT ────────────────────────────────────────────────────────
# Filtrar por el año más reciente con datos completos (2021)
df21 = df[df["vigencia"] == 2021].copy()

# SOLUCIÓN: Convertir 'puntaje_entidad' y 'percentil' de texto a número (comas a puntos)
for col in ["puntaje_entidad", "percentil_grupo_par"]:
    if col in df21.columns:
        if df21[col].dtype == object:
            df21[col] = df21[col].astype(str).str.replace(',', '.')
        df21[col] = pd.to_numeric(df21[col], errors='coerce')

# Verificar si la columna del percentil existe para resguardarla
has_percentil = "percentil_grupo_par" in df21.columns

# Pivotar: Agrupa por municipio y departamento, indexando cada sub-índice en columnas
pivot = (
    df21.pivot_table(
        index=["municipio", "departamento"],
        columns="indice",
        values="puntaje_entidad",
        aggfunc="mean"
    )
    .reset_index()
)

# Mantener solo los sub-índices con presencia en al menos el 50% de los registros
threshold = 0.50 * len(pivot)
valid_cols = [
    c for c in pivot.columns
    if c not in ("municipio", "departamento")
    and pivot[c].notna().sum() >= threshold
]

pivot_clean = pivot[["municipio", "departamento"] + valid_cols].dropna()

feature_cols = valid_cols  # Características para el entrenamiento
X_raw = pivot_clean[feature_cols].values

# Generar puntaje promedio compuesto de Gobierno Digital por municipio
pivot_clean = pivot_clean.copy()
pivot_clean["gd_score"] = pivot_clean[feature_cols].mean(axis=1)

# Re-inyectar el percentil mapeado para el resumen final del JSON
if has_percentil:
    pct_map = df21.groupby(["municipio", "departamento"])["percentil_grupo_par"].mean().to_dict()
    pivot_clean["percentil_grupo_par"] = pivot_clean.set_index(["municipio", "departamento"]).index.map(pct_map)

# Split 80/20 estructurado únicamente para evaluar consistencia matemática
X_train, X_test, idx_train, idx_test = train_test_split(
    X_raw, pivot_clean.index.tolist(),
    test_size=0.20, random_state=42
)

print(f"Total municipios procesados: {len(pivot_clean)}")
print(f"Train: {len(X_train)} | Test: {len(X_test)}")

# ─── 2. PREPROCESSING ────────────────────────────────────────────────────────
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
X_all_sc   = scaler.transform(X_raw)

# PCA → 2D for visualisation
pca = PCA(n_components=2, random_state=42)
X_train_pca = pca.fit_transform(X_train_sc)
X_all_pca   = pca.transform(X_all_sc)
pca_var = [round(v * 100, 1) for v in pca.explained_variance_ratio_]

# ─── 3. HYPERPARAMETER SEARCH — best k via Elbow + Silhouette ────────────────
K_RANGE = range(2, 9)
inertias, sil_scores, grid_results = [], [], []

for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_train_sc)
    inertia = km.inertia_
    sil = round(silhouette_score(X_train_sc, labels), 4) if k > 1 else None
    dbi = round(davies_bouldin_score(X_train_sc, labels), 4)
    chi = round(calinski_harabasz_score(X_train_sc, labels), 2)
    inertias.append(round(inertia, 2))
    if sil: sil_scores.append(sil)
    grid_results.append({
        "k": k,
        "inertia": round(inertia, 2),
        "silhouette": sil,
        "davies_bouldin": dbi,
        "calinski_harabasz": chi,
        "is_best": False,
    })

# Best k = highest silhouette (with preference for 3–4 interpretable clusters)
best_idx = int(np.argmax(sil_scores))
best_k   = list(K_RANGE)[best_idx]
grid_results[best_idx]["is_best"] = True

print(f"Best k: {best_k}  | Silhouette: {sil_scores[best_idx]}")

# ─── 4. TRAINING — final model on full dataset ────────────────────────────────
km_final = KMeans(n_clusters=best_k, random_state=42, n_init=20)
km_final.fit(X_train_sc)

labels_all   = km_final.predict(X_all_sc)
labels_train = km_final.predict(X_train_sc)
labels_test  = km_final.predict(X_test_sc)

pivot_clean = pivot_clean.copy()
pivot_clean["cluster"] = labels_all

# ─── 5. VALIDATION METRICS ────────────────────────────────────────────────────
sil_final  = round(silhouette_score(X_all_sc, labels_all), 4)
dbi_final  = round(davies_bouldin_score(X_all_sc, labels_all), 4)
chi_final  = round(calinski_harabasz_score(X_all_sc, labels_all), 2)

sil_train  = round(silhouette_score(X_train_sc, labels_train), 4)
sil_test   = round(silhouette_score(X_test_sc,  labels_test),  4)

print(f"Silhouette (all): {sil_final} | DBI: {dbi_final} | CHI: {chi_final}")
print(f"Silhouette train: {sil_train} | test: {sil_test}")

# ─── 6. RISK ASSIGNMENT ────────────────────────────────────────────────────────
# Rank clusters by mean GD score → lowest = Alto, highest = Bajo
cluster_means = (
    pivot_clean.groupby("cluster")["gd_score"].mean().sort_values()
)
n = len(cluster_means)
RISK_MAP = {}
LABELS_MAP = {}
risk_names = ["Alto", "Medio-Alto", "Medio-Bajo", "Bajo"]
# Use only as many risk labels as there are clusters
risk_labels_used = risk_names[:n] if n <= 4 else risk_names + [f"Nivel {i}" for i in range(5, n+1)]

for rank, cluster_id in enumerate(cluster_means.index):
    RISK_MAP[int(cluster_id)]   = risk_labels_used[rank]
    LABELS_MAP[int(cluster_id)] = f"Cluster {cluster_id} — {risk_labels_used[rank]}"

pivot_clean["risk"]  = pivot_clean["cluster"].map(RISK_MAP)
pivot_clean["label"] = pivot_clean["cluster"].map(LABELS_MAP)

# ─── 7. BUILD OUTPUT JSON ──────────────────────────────────────────────────────

# 7a. Cluster profiles
cluster_profiles = []
RISK_COLORS = {
    "Alto": "#dc2626", "Medio-Alto": "#f59e0b",
    "Medio-Bajo": "#3b82f6", "Bajo": "#16a34a"
}

for cid in sorted(pivot_clean["cluster"].unique()):
    grp = pivot_clean[pivot_clean["cluster"] == cid]
    risk = RISK_MAP[int(cid)]
    means_dict = {c: round(float(grp[c].mean()), 2) for c in feature_cols}
    means_dict["Gobierno Digital"] = round(float(grp["gd_score"].mean()), 2)
    top_depts = grp["departamento"].value_counts().head(3).to_dict()
    cluster_profiles.append({
        "cluster_id": int(cid),
        "label": LABELS_MAP[int(cid)],
        "risk": risk,
        "color": RISK_COLORS.get(risk, "#6366f1"),
        "size": int(len(grp)),
        "means": means_dict,
        "mean_gd": round(float(grp["gd_score"].mean()), 2),
        "mean_pct": round(float(grp["percentil_grupo_par"].mean()), 1)
            if "percentil_grupo_par" in pivot_clean.columns else None,
        "top_departments": {str(k): int(v) for k, v in top_depts.items()},
    })

# 7b. Scatter data (PCA)
scatter_data = []
for i, row in pivot_clean.iterrows():
    scatter_data.append({
        "x": round(float(X_all_pca[pivot_clean.index.get_loc(i), 0]), 4),
        "y": round(float(X_all_pca[pivot_clean.index.get_loc(i), 1]), 4),
        "municipio":   str(row["municipio"]),
        "departamento":str(row["departamento"]),
        "cluster":     int(row["cluster"]),
        "risk":        str(row["risk"]),
        "gd_score":    round(float(row["gd_score"]), 2),
    })

# 7c. Department summary table
dept_table = (
    pivot_clean.groupby("departamento")
    .agg(
        count=("municipio", "count"),
        gd_mean=("gd_score", "mean"),
        dominant_risk=("risk", lambda x: x.value_counts().idxmax()),
    )
    .reset_index()
    .sort_values("gd_mean")
    .rename(columns={"departamento": "DEPARTAMENTO"})
)
dept_table["gd_mean"] = dept_table["gd_mean"].round(2)
dept_table_list = dept_table.to_dict(orient="records")

# 7d. Risk distribution
risk_dist = pivot_clean["risk"].value_counts().to_dict()

# 7e. Elbow curve data
elbow_data = [{"k": r["k"], "inertia": r["inertia"]} for r in grid_results]

# 7f. Critical municipalities (Alto risk)
critical = pivot_clean[pivot_clean["risk"] == "Alto"].copy()
critical_list = critical[["municipio", "departamento", "gd_score"]].copy()
critical_list["gd_score"] = critical_list["gd_score"].round(2)
critical_list = critical_list.sort_values("gd_score").to_dict(orient="records")

# 7g. Centroids (in original scale) for model summary
centroids_original = scaler.inverse_transform(km_final.cluster_centers_)
centroids_dict = []
for cid in range(best_k):
    c = {feature_cols[j]: round(float(centroids_original[cid, j]), 2)
         for j in range(len(feature_cols))}
    centroids_dict.append({"cluster_id": cid, "centroid": c})

# 7h. Final JSON
result = {
    # Dataset split
    "total_municipalities": int(len(pivot_clean)),
    "train_size":           int(len(X_train)),
    "test_size":            int(len(X_test)),
    "feature_cols":         feature_cols,
    "pca_variance":         pca_var,

    # Hyperparameters
    "best_k":               int(best_k),
    "grid_results":         grid_results,
    "elbow_data":           elbow_data,

    # Validation
    "silhouette":           sil_final,
    "davies_bouldin":       dbi_final,
    "calinski_harabasz":    chi_final,
    "silhouette_train":     sil_train,
    "silhouette_test":      sil_test,

    # Predictions
    "n_clusters":           int(best_k),
    "cluster_profiles":     cluster_profiles,
    "risk_distribution":    risk_dist,
    "scatter_data":         scatter_data,
    "dept_table":           dept_table_list,
    "critical_municipalities": critical_list,
    "centroids":            centroids_dict,
}

with open("kmeans_results.json", "w", encoding="utf-8") as f:
    json.dump(result, f, ensure_ascii=False, indent=2)

print("\n✓  kmeans_results.json saved successfully")
print(f"   Clusters: {best_k} | Municipalities: {len(pivot_clean)}")
print(f"   Risk distribution: {risk_dist}")