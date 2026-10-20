#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 22 21:20:55 2026

@author: breyner
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import missingno as msno

from sklearn.model_selection import RepeatedStratifiedKFold

# =========================
# 1. Cargar datos
# =========================

df = pd.read_csv("datos_modelo_limpio.csv")

target = "y"

numeric_features = [
    "edad", "pas", "pad", "fc",
    "bil_total", "bil_directa",
    "alt", "ast", "fosfatasa",
    "ggt", "lipasa", "pcr",
    "leucocitos", "neutrofilos",
    "diam_via_biliar"
]

categorical_features = [
    "sexo",
    "dolor_abdominal",
    "ictericia",
    "fiebre",
    "prueba_imagen",
    "colelitiasis",
    "coledocolitiasis",
    "colecistitis"
]

X = df[numeric_features + categorical_features].copy()
y = df[target].copy()

# Outcome binario: clase positiva = No_CDL
y_bin = (y == "No_CDL").astype(int)

print("X:", X.shape)
print("y:", y_bin.shape)
print(y.value_counts())
print(y.value_counts(normalize=True) * 100)

# =========================
# Etiquetas en inglés para las figuras
# (solo se usan al graficar; los nombres originales
#  de las columnas no se modifican)
# =========================

labels_en = {
    "edad": "Age",
    "pas": "Systolic BP",
    "pad": "Diastolic BP",
    "fc": "Heart rate",
    "bil_total": "Total bilirubin",
    "bil_directa": "Direct bilirubin",
    "alt": "ALT",
    "ast": "AST",
    "fosfatasa": "Alkaline phosphatase",
    "ggt": "GGT",
    "lipasa": "Lipase",
    "pcr": "CRP",
    "leucocitos": "Leukocytes",
    "neutrofilos": "Neutrophils",
    "diam_via_biliar": "Bile duct diameter",
    "sexo": "Sex",
    "dolor_abdominal": "Abdominal pain",
    "ictericia": "Jaundice",
    "fiebre": "Fever",
    "prueba_imagen": "Imaging test",
    "colelitiasis": "Cholelithiasis",
    "coledocolitiasis": "Suspected CDL",
    "colecistitis": "Cholecystitis",
}

X_plot = X.rename(columns=labels_en)

# =========================
# 2. Missingness básico
# =========================

missing_summary = pd.DataFrame({
    "variable": X.columns,
    "n_missing": X.isna().sum().values,
    "pct_missing": (
        X.isna().mean().values * 100
    ).round(2)
})

missing_summary = (
    missing_summary
    .sort_values(
        by="pct_missing",
        ascending=False
    )
    .reset_index(drop=True)
)

print(missing_summary)

# Solo variables con missing
missing_vars = missing_summary[
    missing_summary["n_missing"] > 0
]

print("\nVariables con missing:")
print(missing_vars)

# =========================
# 3. Visualización missingness (figuras en inglés)
# =========================

DPI = 300

# --- Figura 3: matriz del patrón de missingness ---
# msno crea su propia figura: se usa figsize dentro de la función
# (un plt.figure() previo solo dejaba una figura vacía).
ax = msno.matrix(X_plot, figsize=(14, 7), fontsize=11, sparkline=True)
ax.set_title("Missingness pattern", fontsize=14, pad=20)
fig = ax.get_figure()
fig.savefig("missing_heatmap.png", dpi=DPI, bbox_inches="tight")
plt.show()

# --- Figura 4: correlación entre patrones de missingness ---
ax = msno.heatmap(X_plot, figsize=(12, 10), fontsize=11)
ax.set_title("Correlation between missingness patterns", fontsize=14)
fig = ax.get_figure()
fig.savefig("missing_correlation.png", dpi=DPI, bbox_inches="tight")
plt.show()

# --- Figura 2: porcentaje de missingness por variable ---
missing_plot = missing_summary[missing_summary["pct_missing"] > 0].copy()
missing_plot["label"] = missing_plot["variable"].map(labels_en)

fig, ax = plt.subplots(figsize=(10, 6))

ax.barh(
    missing_plot["label"],
    missing_plot["pct_missing"]
)

ax.set_xlabel("Missing values (%)")
ax.set_ylabel("Variable")
ax.set_title("Percentage of missing values per variable")
ax.invert_yaxis()

fig.tight_layout()
fig.savefig("missing_porcentaje.png", dpi=DPI, bbox_inches="tight")
plt.show()

from pyampute.exploration.mcar_statistical_tests import MCARTest

# =========================
# 4. Little MCAR Test
# =========================

mcar_test = MCARTest(method="little")

result = mcar_test(X)

print(result)

from scipy.stats import chi2_contingency, fisher_exact

# =========================
# Missingness vs outcome
# con selección automática
# =========================

missing_outcome_results = []

for col in X.columns:

    missing_indicator = X[col].isna()

    contingency = pd.crosstab(
        missing_indicator,
        y_bin
    )

    # Saltar variables sin missing
    if contingency.shape[0] < 2:
        continue

    # Chi expected frequencies
    chi2, p_chi, dof, expected = chi2_contingency(contingency)

    # Elegir prueba
    if (expected < 5).any():
        _, p_value = fisher_exact(contingency)
        test_used = "Fisher"
    else:
        p_value = p_chi
        test_used = "Chi-square"

    pct_positive_missing = (
        y_bin[missing_indicator].mean() * 100
    )

    pct_positive_not_missing = (
        y_bin[~missing_indicator].mean() * 100
    )

    missing_outcome_results.append({
        "variable": col,
        "pct_missing": round(
            X[col].isna().mean() * 100,
            2
        ),
        "n_missing": missing_indicator.sum(),
        "pct_No_CDL_missing": round(
            pct_positive_missing,
            2
        ),
        "pct_No_CDL_not_missing": round(
            pct_positive_not_missing,
            2
        ),
        "test": test_used,
        "p_value": round(p_value, 5)
    })

missing_outcome_df = pd.DataFrame(
    missing_outcome_results
)

missing_outcome_df = (
    missing_outcome_df
    .sort_values("p_value")
    .reset_index(drop=True)
)

print(missing_outcome_df)

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import roc_auc_score

# =========================
# 6. Predictibilidad del missingness
# =========================

missing_predictability_results = []

cv_missing = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

for target_missing_var in missing_vars["variable"]:

    # Outcome del modelo: si falta esa variable
    m = X[target_missing_var].isna().astype(int)

    # Saltar si hay muy pocos casos missing
    if m.sum() < 10:
        continue

    # Predictores: todas las demás variables
    predictors = [
        col for col in X.columns
        if col != target_missing_var
    ]

    X_miss = X[predictors].copy()

    num_pred = [
        col for col in numeric_features
        if col in predictors
    ]

    cat_pred = [
        col for col in categorical_features
        if col in predictors
    ]

    preprocess_miss = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_pred),

        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), cat_pred)
    ])

    model_miss = Pipeline([
        ("preprocess", preprocess_miss),
        ("model", LogisticRegression(
            max_iter=5000,
            class_weight="balanced",
            random_state=42
        ))
    ])

    auc_scores = cross_val_score(
        model_miss,
        X_miss,
        m,
        cv=cv_missing,
        scoring="roc_auc"
    )

    missing_predictability_results.append({
        "variable": target_missing_var,
        "n_missing": int(m.sum()),
        "pct_missing": round(m.mean() * 100, 2),
        "missing_auc_mean": round(auc_scores.mean(), 3),
        "missing_auc_sd": round(auc_scores.std(), 3)
    })

missing_predictability_df = pd.DataFrame(
    missing_predictability_results
).sort_values(
    "missing_auc_mean",
    ascending=False
)

print(missing_predictability_df)

