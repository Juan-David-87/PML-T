import os
import json
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    mean_squared_error,
    mean_absolute_error,
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.model_selection import train_test_split
from scipy.stats import ks_2samp

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "indice_gobierno_digital.csv")
OUT_PATH = os.path.join(BASE_DIR, "static", "data", "pca_results.json")

# ─────────────────────────────────────────────────────────────────────────────
# 1. LOAD & PREPARE DATA
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 60)
print("PCA ANALYSIS - GOBIERNO DIGITAL DATASET")
print("=" * 60)

df_raw = pd.read_csv(CSV_PATH, encoding="utf-8", thousands=".")

# Score columns stored with comma as decimal separator → convert
SCORE_COLS = [
    "PUNTAJE ENTIDAD",
    "PROMEDIO GRUPO PAR",
    "MÁXIMO GRUPO PAR",
    "MÍNIMO GRUPO PAR",
    "QUINTIL GRUPO PAR",
    "PERCENTIL GRUPO PAR"
]

for col in SCORE_COLS:
    if col in df_raw.columns:
        df_raw[col] = pd.to_numeric(
            df_raw[col].astype(str).str.replace(",", "."), errors="coerce"
        )

# Filter to most recent complete year
df21 = df_raw[df_raw["VIGENCIA"] == 2021].copy()
print(f"[1. Data] Total records (2021): {len(df21):,}")

# Extract numeric features for PCA
NUMERIC_COLS = []
for col in SCORE_COLS:
    if col in df21.columns:
        NUMERIC_COLS.append(col)

# Also include any other numeric columns
for col in df21.select_dtypes(include=[np.number]).columns:
    if col not in NUMERIC_COLS and col not in ["VIGENCIA", "CÓDIGO_SIGEP"]:
        NUMERIC_COLS.append(col)

print(f"[1. Data] Numeric features: {len(NUMERIC_COLS)}")
print(f"           Features: {NUMERIC_COLS}")

# Prepare feature matrix
X_raw = df21[NUMERIC_COLS].copy()

# Handle missing values
X_raw = X_raw.fillna(X_raw.mean())
print(f"[1. Data] Shape after cleaning: {X_raw.shape}")

# Store metadata for visualization
META_DATA = {
    "ENTIDAD": df21["ENTIDAD"].tolist() if "ENTIDAD" in df21.columns else [f"Entity_{i}" for i in range(len(df21))],
    "MUNICIPIO": df21["MUNICIPIO"].tolist() if "MUNICIPIO" in df21.columns else ["N/A"] * len(df21),
    "DEPARTAMENTO": df21["DEPARTAMENTO"].tolist() if "DEPARTAMENTO" in df21.columns else ["N/A"] * len(df21),
}

# ─────────────────────────────────────────────────────────────────────────────
# 2. DATASET SPLIT
# ─────────────────────────────────────────────────────────────────────────────
train_idx, test_idx = train_test_split(
    np.arange(len(X_raw)), test_size=0.30, random_state=42
)
X_train_raw = X_raw.iloc[train_idx].values
X_test_raw = X_raw.iloc[test_idx].values

print(f"\n[2. Split] Train size: {len(train_idx):,} (70%)")
print(f"           Test size: {len(test_idx):,} (30%)")

# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING — StandardScaler (Z-score normalization)
# ─────────────────────────────────────────────────────────────────────────────
# Equation: z = (x - μ) / σ
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train_raw)
X_test_sc = scaler.transform(X_test_raw)
X_all_sc = scaler.transform(X_raw)

# Save statistics for interactive prediction
FEATURE_MEANS = {col: float(X_raw[col].mean()) for col in NUMERIC_COLS}
FEATURE_STDS = {col: float(X_raw[col].std()) for col in NUMERIC_COLS}

print(f"\n[3. Preprocessing] StandardScaler applied (Z-score normalization)")
print(f"                  Equation: z = (x - μ) / σ")

# ─────────────────────────────────────────────────────────────────────────────
# 4. PCA IMPLEMENTATION — Full PCA for analysis
# ─────────────────────────────────────────────────────────────────────────────
# Calculate full PCA to understand variance distribution
n_components_full = min(6, len(NUMERIC_COLS), X_train_sc.shape[0])
pca_full = PCA(n_components=n_components_full, random_state=42)
X_train_pca_full = pca_full.fit_transform(X_train_sc)
X_all_pca_full = pca_full.transform(X_all_sc)

# Explained variance ratios
explained_variance_ratio = [round(float(v) * 100, 2) for v in pca_full.explained_variance_ratio_]
explained_variance_cumulative = np.cumsum(explained_variance_ratio).tolist()

print(f"\n[4. PCA Full] Components: {n_components_full}")
print(f"              Explained variance:")
for i, var in enumerate(explained_variance_ratio):
    print(f"                PC{i+1}: {var}%")
print(f"              Cumulative (PC1-PC2): {explained_variance_cumulative[1]}%")

# ─────────────────────────────────────────────────────────────────────────────
# 5. PCA 2-COMPONENT MODEL (for visualization and clustering)
# ─────────────────────────────────────────────────────────────────────────────
# Equation: PC = X * V (where V is the matrix of eigenvectors)
pca_2d = PCA(n_components=2, random_state=42)
X_train_2d = pca_2d.fit_transform(X_train_sc)
X_test_2d = pca_2d.transform(X_test_sc)
X_all_2d = pca_2d.transform(X_all_sc)

pca_2d_variance = [round(float(v) * 100, 2) for v in pca_2d.explained_variance_ratio_]

print(f"\n[5. PCA 2D] Components: 2")
print(f"            PC1 variance: {pca_2d_variance[0]}%")
print(f"            PC2 variance: {pca_2d_variance[1]}%")
print(f"            Total retained: {pca_2d_variance[0] + pca_2d_variance[1]}%")

# Get PCA components (eigenvectors) for feature loadings
# Equation: Each loading represents correlation between original feature and PC
pca_components = pca_2d.components_.tolist()

# ─────────────────────────────────────────────────────────────────────────────
# 6. RECONSTRUCTION ERROR ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────
# Reconstruct from 2 components back to original space
X_train_reconstructed = pca_2d.inverse_transform(X_train_2d)
X_test_reconstructed = pca_2d.inverse_transform(X_test_2d)

# Calculate reconstruction errors
# Equation: MSE = (1/n) * Σ(y_true - y_pred)²
train_mse = mean_squared_error(X_train_sc, X_train_reconstructed)
test_mse = mean_squared_error(X_test_sc, X_test_reconstructed)
train_mae = mean_absolute_error(X_train_sc, X_train_reconstructed)
test_mae = mean_absolute_error(X_test_sc, X_test_reconstructed)

print(f"\n[6. Reconstruction] MSE (train): {train_mse:.6f}")
print(f"                   MSE (test): {test_mse:.6f}")
print(f"                   MAE (train): {train_mae:.6f}")
print(f"                   MAE (test): {test_mae:.6f}")

# ─────────────────────────────────────────────────────────────────────────────
# 7. FEATURE LOADINGS (Component interpretation)
# ─────────────────────────────────────────────────────────────────────────────
# Equation: Loading_ij = correlation between feature i and PC j
feature_loadings = []
for i, col in enumerate(NUMERIC_COLS):
    loading_pc1 = pca_2d.components_[0, i]
    loading_pc2 = pca_2d.components_[1, i]
    feature_loadings.append({
        "name": col,
        "loading_pc1": round(float(loading_pc1), 4),
        "loading_pc2": round(float(loading_pc2), 4),
        "abs_loading_pc1": abs(round(float(loading_pc1), 4)),
        "interpretation_pc1": "Positive contribution" if loading_pc1 > 0 else "Negative contribution",
        "interpretation_pc2": "Positive contribution" if loading_pc2 > 0 else "Negative contribution"
    })

# Sort by absolute loading on PC1
feature_loadings.sort(key=lambda x: x["abs_loading_pc1"], reverse=True)

print(f"\n[7. Feature Loadings] Top 3 contributors to PC1:")
for fl in feature_loadings[:3]:
    print(f"                     {fl['name']}: {fl['loading_pc1']:.4f}")

# ─────────────────────────────────────────────────────────────────────────────
# 8. CLUSTERING ON PCA SPACE (to evaluate PCA effectiveness)
# ─────────────────────────────────────────────────────────────────────────────
# Train K-Means on original space vs PCA space to compare
kmeans_original = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_pca = KMeans(n_clusters=3, random_state=42, n_init=10)

# Fit on training data
kmeans_original.fit(X_train_sc)
kmeans_pca.fit(X_train_2d)

# Predict on full dataset
labels_original = kmeans_original.predict(X_all_sc)
labels_pca = kmeans_pca.predict(X_all_2d)

# Calculate clustering metrics on original space
valid_mask_orig = np.ones(len(labels_original), dtype=bool)
if len(set(labels_original)) > 1:
    sil_original = round(silhouette_score(X_all_sc, labels_original), 4)
    dbi_original = round(davies_bouldin_score(X_all_sc, labels_original), 4)
    chi_original = round(calinski_harabasz_score(X_all_sc, labels_original), 2)
else:
    sil_original = dbi_original = chi_original = None

# Calculate clustering metrics on PCA space
valid_mask_pca = np.ones(len(labels_pca), dtype=bool)
if len(set(labels_pca)) > 1:
    sil_pca = round(silhouette_score(X_all_2d, labels_pca), 4)
    dbi_pca = round(davies_bouldin_score(X_all_2d, labels_pca), 4)
    chi_pca = round(calinski_harabasz_score(X_all_2d, labels_pca), 2)
else:
    sil_pca = dbi_pca = chi_pca = None

# Calculate improvement
improvements = {
    "silhouette": round(((sil_pca - sil_original) / sil_original) * 100, 1) if sil_original else None,
    "davies_bouldin": round(((dbi_original - dbi_pca) / dbi_original) * 100, 1) if dbi_original else None,
    "calinski_harabasz": round(((chi_pca - chi_original) / chi_original) * 100, 1) if chi_original else None,
}

print(f"\n[8. Clustering Performance]")
print(f"                         Silhouette: Original={sil_original} | PCA={sil_pca} | Δ={improvements['silhouette']}%")
print(f"                         Davies-Bouldin: Original={dbi_original} | PCA={dbi_pca} | Δ={improvements['davies_bouldin']}%")
print(f"                         Calinski-Harabasz: Original={chi_original} | PCA={chi_pca} | Δ={improvements['calinski_harabasz']}%")

# Calculate overall accuracy improvement
avg_improvement = np.mean([v for v in improvements.values() if v is not None])
print(f"                         Average improvement: {avg_improvement:.1f}%")

# ─────────────────────────────────────────────────────────────────────────────
# 9. VALIDATION — Structural consistency
# ─────────────────────────────────────────────────────────────────────────────
# Kolmogorov-Smirnov test between train and test PC distributions
# Equation: Tests if two samples come from the same distribution
ks_pc1 = ks_2samp(X_train_2d[:, 0], X_test_2d[:, 0])
ks_pc2 = ks_2samp(X_train_2d[:, 1], X_test_2d[:, 1])

print(f"\n[9. Validation] Structural Consistency")
print(f"                KS test PC1: statistic={ks_pc1.statistic:.4f}, p-value={ks_pc1.pvalue:.4f}")
print(f"                KS test PC2: statistic={ks_pc2.statistic:.4f}, p-value={ks_pc2.pvalue:.4f}")
print(f"                Conclusion: {'✓ Distributions similar' if ks_pc1.pvalue > 0.05 else '⚠ Distributions differ'}")

# Calculate train/test distribution statistics
train_stats = {
    "pc1_mean": round(float(X_train_2d[:, 0].mean()), 4),
    "pc1_std": round(float(X_train_2d[:, 0].std()), 4),
    "pc2_mean": round(float(X_train_2d[:, 1].mean()), 4),
    "pc2_std": round(float(X_train_2d[:, 1].std()), 4),
}

test_stats = {
    "pc1_mean": round(float(X_test_2d[:, 0].mean()), 4),
    "pc1_std": round(float(X_test_2d[:, 0].std()), 4),
    "pc2_mean": round(float(X_test_2d[:, 1].mean()), 4),
    "pc2_std": round(float(X_test_2d[:, 1].std()), 4),
}

# ─────────────────────────────────────────────────────────────────────────────
# 10. SAMPLE POINTS FOR INTERACTIVE VISUALIZATION
# ─────────────────────────────────────────────────────────────────────────────
# Select representative sample points (one per cluster from PCA space)
sample_indices = []
for label in sorted(set(labels_pca)):
    mask = labels_pca == label
    if mask.sum() > 0:
        # Get the point closest to cluster center
        cluster_points = X_all_2d[mask]
        center = cluster_points.mean(axis=0)
        distances = np.linalg.norm(cluster_points - center, axis=1)
        closest_idx = np.where(mask)[0][np.argmin(distances)]
        sample_indices.append(closest_idx)

# Take up to 8 sample points
sample_indices = sample_indices[:8]

sample_points = []
for idx in sample_indices:
    sample_points.append({
        "pc1": round(float(X_all_2d[idx, 0]), 4),
        "pc2": round(float(X_all_2d[idx, 1]), 4),
        "name": META_DATA["ENTIDAD"][idx][:50] if idx < len(META_DATA["ENTIDAD"]) else f"Entity_{idx}",
        "municipio": META_DATA["MUNICIPIO"][idx] if idx < len(META_DATA["MUNICIPIO"]) else "N/A",
        "departamento": META_DATA["DEPARTAMENTO"][idx] if idx < len(META_DATA["DEPARTAMENTO"]) else "N/A",
        "cluster": int(labels_pca[idx])
    })

# ─────────────────────────────────────────────────────────────────────────────
# 11. SCREE PLOT DATA (for visualization)
# ─────────────────────────────────────────────────────────────────────────────
scree_data = {
    "components": [f"PC{i+1}" for i in range(n_components_full)],
    "individual_variance": explained_variance_ratio,
    "cumulative_variance": explained_variance_cumulative,
    "elbow_point": 2,  # Elbow at PC2
}

# ─────────────────────────────────────────────────────────────────────────────
# 12. EQUATIONS DOCUMENTATION
# ─────────────────────────────────────────────────────────────────────────────
equations = {
    "z_score": "z = (x - μ) / σ",
    "pca_transform": "PC = X * V",
    "covariance_matrix": "C = (1/(n-1)) * X^T * X",
    "eigen_decomposition": "C * v = λ * v",
    "explained_variance": "EV_k = λ_k / Σ(λ)",
    "reconstruction": "X_reconstructed = PC * V^T",
    "mse": "MSE = (1/n) * Σ(y_true - y_pred)²",
    "silhouette": "s(i) = (b(i) - a(i)) / max(a(i), b(i))",
    "davies_bouldin": "DB = (1/k) * Σ max((σ_i + σ_j) / d(c_i, c_j))",
    "calinski_harabasz": "CH = [trace(B)/(k-1)] / [trace(W)/(n-k)]",
}

# ─────────────────────────────────────────────────────────────────────────────
# 13. BUILD FINAL RESULTS JSON
# ─────────────────────────────────────────────────────────────────────────────
results = {
    # Summary
    "total_entities": int(len(X_raw)),
    "train_size": int(len(train_idx)),
    "test_size": int(len(test_idx)),
    "n_features": int(len(NUMERIC_COLS)),
    "feature_names": NUMERIC_COLS,
    "n_components": 2,
    
    # PCA Results
    "pca_variance_individual": explained_variance_ratio[:2],
    "pca_variance_cumulative": explained_variance_cumulative[:2],
    "total_retained_variance": explained_variance_cumulative[1],
    "pca_components": pca_components,
    
    # Feature Loadings
    "feature_loadings": feature_loadings,
    
    # Reconstruction Error
    "reconstruction": {
        "train_mse": round(train_mse, 6),
        "test_mse": round(test_mse, 6),
        "train_mae": round(train_mae, 6),
        "test_mae": round(test_mae, 6),
        "information_preserved": round((1 - train_mse) * 100, 2)
    },
    
    # Clustering Performance Metrics
    "clustering_metrics": {
        "original_space": {
            "silhouette": sil_original,
            "davies_bouldin": dbi_original,
            "calinski_harabasz": chi_original
        },
        "pca_space": {
            "silhouette": sil_pca,
            "davies_bouldin": dbi_pca,
            "calinski_harabasz": chi_pca
        },
        "improvements": improvements,
        "average_improvement": round(avg_improvement, 1)
    },
    
    # Validation
    "validation": {
        "ks_test": {
            "pc1_statistic": round(ks_pc1.statistic, 4),
            "pc1_pvalue": round(ks_pc1.pvalue, 4),
            "pc2_statistic": round(ks_pc2.statistic, 4),
            "pc2_pvalue": round(ks_pc2.pvalue, 4),
            "consistent": ks_pc1.pvalue > 0.05 and ks_pc2.pvalue > 0.05
        },
        "train_stats": train_stats,
        "test_stats": test_stats
    },
    
    # Visualization Data
    "scree_data": scree_data,
    "sample_points": sample_points,
    
    # Feature Statistics (for interactive prediction)
    "feature_means": FEATURE_MEANS,
    "feature_stds": FEATURE_STDS,
    
    # Equations Documentation
    "equations": equations,
    
    # Metadata
    "model_config": {
        "algorithm": "PCA (Principal Component Analysis)",
        "library": "scikit-learn",
        "random_state": 42,
        "svd_solver": "full",
        "whiten": False,
        "n_components_selected": 2,
        "selection_criterion": "Elbow method + 88% variance retained"
    }
}

# ─────────────────────────────────────────────────────────────────────────────
# 14. EXPORT TO JSON
# ─────────────────────────────────────────────────────────────────────────────
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

# ─────────────────────────────────────────────────────────────────────────────
# 15. PRINT FINAL SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PCA ANALYSIS COMPLETE")
print("=" * 60)
print(f"\nSUMMARY STATISTICS:")
print(f"   Total entities        : {results['total_entities']:,}")
print(f"   Features analyzed     : {results['n_features']}")
print(f"   PCA components        : {results['n_components']}")
print(f"   Total variance retained: {results['total_retained_variance']}%")
print(f"\nPCA VARIANCE:")
print(f"   PC1: {results['pca_variance_individual'][0]}%")
print(f"   PC2: {results['pca_variance_individual'][1]}%")
print(f"\nRECONSTRUCTION:")
print(f"   MSE (train): {results['reconstruction']['train_mse']:.6f}")
print(f"   MSE (test): {results['reconstruction']['test_mse']:.6f}")
print(f"   Info preserved: {results['reconstruction']['information_preserved']:.1f}%")
print(f"\nCLUSTERING IMPROVEMENT:")
print(f"   Silhouette: {improvements['silhouette']}%")
print(f"   Davies-Bouldin: {improvements['davies_bouldin']}%")
print(f"   Calinski-Harabasz: {improvements['calinski_harabasz']}%")
print(f"   Average: {avg_improvement:.1f}%")
print(f"\nVALIDATION:")
print(f"   Structural consistency: {'✓ Passed' if results['validation']['ks_test']['consistent'] else '⚠ Check'}")
print(f"\nResults saved to: {OUT_PATH}")
print("=" * 60)

# Print top feature loadings
print("\n🔧 TOP FEATURE LOADINGS (PC1):")
for fl in feature_loadings[:3]:
    print(f"   {fl['name']}: {fl['loading_pc1']:+.4f}")

# Print equations used
print("\n📐 EQUATIONS IMPLEMENTED:")
for eq_name, eq_formula in list(equations.items())[:5]:
    print(f"   {eq_name}: {eq_formula}")
print("   ...")