import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, StandardScaler, RobustScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# LOAD
df = pd.read_csv("indice_gobierno_digital.csv")

# STEP 1 DATA CLEANING 
column_mapping = {
    'CÓDIGO_SIGEP': 'codigo_sigep', 'ENTIDAD': 'entidad',
    'ORDEN': 'orden', 'SECTOR': 'sector',
    'NATURALEZA_JURÍDICA': 'naturaleza_juridica',
    'ID_DEPARTAMENTO': 'id_departamento', 'DEPARTAMENTO': 'departamento',
    'ID_MUNICIPIO': 'id_municipio', 'MUNICIPIO': 'municipio',
    'VIGENCIA': 'vigencia', 'ID_ÍNDICE': 'id_indice', 'ÍNDICE': 'indice',
    'PUNTAJE ENTIDAD': 'puntaje_entidad', 'PROMEDIO GRUPO PAR': 'promedio_grupo_par',
    'MÁXIMO GRUPO PAR': 'maximo_grupo_par', 'MÍNIMO GRUPO PAR': 'minimo_grupo_par',
    'QUINTIL GRUPO PAR': 'quintil_grupo_par', 'PERCENTIL GRUPO PAR': 'percentil_grupo_par',
}
df.rename(columns=column_mapping, inplace=True)
df.drop_duplicates(inplace=True)

for col in df.select_dtypes('object').columns:
    df[col] = df[col].str.strip().str.upper()

for col in ['codigo_sigep', 'id_departamento', 'id_municipio', 'vigencia', 'quintil_grupo_par']:
    df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

for col in ['puntaje_entidad', 'promedio_grupo_par', 'maximo_grupo_par',
            'minimo_grupo_par', 'percentil_grupo_par']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Domain rules
df = df[df['puntaje_entidad'].between(0, 100) | df['puntaje_entidad'].isna()]
df = df[df['percentil_grupo_par'].between(0, 100) | df['percentil_grupo_par'].isna()]
df = df[df['quintil_grupo_par'].between(1, 5) | df['quintil_grupo_par'].isna()]

print(f"After cleaning: {df.shape}")

# STEP 2 NULL HANDLING
# Drop columns with >50% nulls
high_null = df.columns[df.isnull().mean() > 0.5].tolist()
df.drop(columns=high_null, inplace=True)

# Median imputation for numeric scores
num_cols = ['puntaje_entidad', 'promedio_grupo_par', 'maximo_grupo_par',
            'minimo_grupo_par', 'percentil_grupo_par']
for col in [c for c in num_cols if c in df.columns]:
    df[col].fillna(df[col].median(), inplace=True)

# Mode for quintil, 'UNKNOWN' for text
if df['quintil_grupo_par'].isnull().any():
    df['quintil_grupo_par'].fillna(df['quintil_grupo_par'].mode()[0], inplace=True)

for col in ['orden', 'sector', 'naturaleza_juridica', 'departamento', 'municipio', 'indice']:
    if col in df.columns:
        df[col].fillna('UNKNOWN', inplace=True)

print(f"Remaining nulls: {df.isnull().sum().sum()}")

# STEP 3 TRANSFORMATION
df['gap_to_max']  = df['maximo_grupo_par'] - df['puntaje_entidad']
df['gap_to_avg']  = df['puntaje_entidad']  - df['promedio_grupo_par']
df['spread_grupo_par'] = df['maximo_grupo_par'] - df['minimo_grupo_par']
df['relative_performance'] = np.where(
    df['spread_grupo_par'] > 0,
    (df['puntaje_entidad'] - df['minimo_grupo_par']) / df['spread_grupo_par'],
    0.5
).clip(0, 1)

df['performance_tier'] = pd.cut(
    df['puntaje_entidad'],
    bins=[-1, 40, 60, 80, 101],
    labels=['LOW', 'MEDIUM-LOW', 'MEDIUM-HIGH', 'HIGH']
)
df['beats_avg']   = (df['puntaje_entidad'] > df['promedio_grupo_par']).astype(int)
df['year_offset'] = df['vigencia'] - df['vigencia'].min()

print("Transformation complete — 6 new features added")

# STEP 4 CATEGORICAL ENCODING
le = LabelEncoder()
for col in ['orden', 'naturaleza_juridica']:
    if col in df.columns:
        df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))

# One-Hot (top 10) for sector
top_sectors = df['sector'].value_counts().nlargest(10).index
df['sector_top'] = np.where(df['sector'].isin(top_sectors), df['sector'], 'OTHER')
df = pd.concat([df, pd.get_dummies(df['sector_top'], prefix='sector')], axis=1)
df.drop(columns=['sector_top'], inplace=True)

# Frequency encoding for high-cardinality columns
for col in ['indice', 'departamento']:
    freq = df[col].value_counts(normalize=True)
    df[f'{col}_freq'] = df[col].map(freq)

# Ordinal encoding for performance tier
df['performance_tier_encoded'] = df['performance_tier'].map(
    {'LOW': 0, 'MEDIUM-LOW': 1, 'MEDIUM-HIGH': 2, 'HIGH': 3}
)

print("Encoding complete")

# STEP 5 FEATURE SELECTION
feature_cols = [
    'puntaje_entidad', 'promedio_grupo_par', 'maximo_grupo_par', 'minimo_grupo_par',
    'gap_to_max', 'gap_to_avg', 'spread_grupo_par', 'relative_performance',
    'beats_avg', 'year_offset', 'orden_encoded', 'naturaleza_juridica_encoded',
    'indice_freq', 'departamento_freq', 'performance_tier_encoded',
] + [c for c in df.columns if c.startswith('sector_')]
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].copy()

# 1) Variance threshold
vt = VarianceThreshold(threshold=0.01)
vt.fit(X)
X = X.loc[:, vt.get_support()]

# 2) Correlation filter
corr = X.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_corr = [c for c in upper.columns if any(upper[c] > 0.95)]
X.drop(columns=drop_corr, inplace=True)

# 3) SelectKBest
target = 'percentil_grupo_par'
X_sel = X.drop(columns=[target], errors='ignore')
y = df[target]
k = min(10, X_sel.shape[1])
selector = SelectKBest(f_regression, k=k)
selector.fit(X_sel.fillna(0), y.fillna(0))
selected = pd.Series(selector.scores_, index=X_sel.columns).nlargest(k).index.tolist()

print(f"Selected features ({k}): {selected}")

# STEP 6 FEATURE ENGINEERING
df.sort_values(['entidad', 'indice', 'vigencia'], inplace=True)

df['yoy_change'] = df.groupby(['entidad', 'indice'])['puntaje_entidad'].diff()
df['rolling_2y_avg'] = (
    df.groupby(['entidad', 'indice'])['puntaje_entidad']
      .transform(lambda x: x.rolling(2, min_periods=1).mean())
)
df['improving']              = (df['yoy_change'] > 0).astype('Int64')
df['entity_historical_avg']  = df.groupby('entidad')['puntaje_entidad'].transform('mean')
df['deviation_from_entity_avg'] = df['puntaje_entidad'] - df['entity_historical_avg']
df['urgency_index']          = df['gap_to_max'] * df['year_offset']

print("Feature engineering complete — 6 temporal features added")

# STEP 7 NORMALISATION
cols_mm = [c for c in ['puntaje_entidad', 'promedio_grupo_par', 'maximo_grupo_par',
                        'minimo_grupo_par', 'gap_to_max', 'gap_to_avg',
                        'spread_grupo_par', 'relative_performance',
                        'entity_historical_avg'] if c in df.columns]

cols_zs = [c for c in ['percentil_grupo_par', 'deviation_from_entity_avg',
                        'yoy_change', 'rolling_2y_avg', 'urgency_index'] if c in df.columns]

cols_rb = [c for c in ['year_offset', 'quintil_grupo_par'] if c in df.columns]

df[[f'{c}_minmax' for c in cols_mm]] = MinMaxScaler().fit_transform(df[cols_mm])
df[[f'{c}_zscore' for c in cols_zs]] = StandardScaler().fit_transform(df[cols_zs].fillna(0))
df[[f'{c}_robust' for c in cols_rb]] = RobustScaler().fit_transform(df[cols_rb].fillna(0))

print(f"Scaling complete. Final dataset shape: {df.shape}")

# SAVE
df.to_csv("indice_gobierno_digital_processed.csv", index=False, encoding='utf-8')
print("Saved → indice_gobierno_digital_processed.csv")