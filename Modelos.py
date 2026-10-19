#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 14:57:23 2026

@author: breyner
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from xgboost import XGBClassifier
from catboost import CatBoostClassifier

import torch
import torch.nn as nn
import torch.optim as optim


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

# Clase positiva = No_CDL
y_bin = (y == "No_CDL").astype(int)

print("X:", X.shape)
print("y:", y_bin.shape)
print(y.value_counts())
print(y.value_counts(normalize=True) * 100)


##########exploración inical de los datos

numeric_features = [
    "edad", "pas", "pad", "fc",
    "bil_total", "bil_directa",
    "alt", "ast", "fosfatasa",
    "ggt", "lipasa", "pcr",
    "leucocitos", "neutrofilos",
    "diam_via_biliar"
]

a=df[numeric_features].describe().T

summary_numeric = pd.DataFrame({
    "n": df[numeric_features].notna().sum(),
    "missing_n": df[numeric_features].isna().sum(),
    "mean": df[numeric_features].mean(),
    "sd": df[numeric_features].std(),
    "median": df[numeric_features].median(),
    "q1": df[numeric_features].quantile(0.25),
    "q3": df[numeric_features].quantile(0.75),
    "min": df[numeric_features].min(),
    "max": df[numeric_features].max()
})

a=summary_numeric

categorical_features = [
    "sexo",
    "dolor_abdominal",
    "ictericia",
    "fiebre",
    "prueba_imagen",
    "colelitiasis",
    "coledocolitiasis", # SOSPECHA de coledocolitiasis en la imagen inicial,
                        # disponible al ingreso. NO es el diagnóstico final:
                        # el outcome "y" se define por la prueba de referencia.
    "colecistitis"
]

for col in categorical_features + ["y"]:
    print("\n", col)
    print(df[col].value_counts(dropna=False))
    print((df[col].value_counts(dropna=False, normalize=True) * 100).round(2))

# Numéricas por outcome
summary_by_y = (
    df
    .groupby("y")[numeric_features]
    .agg(["count", "mean", "std", "median", "min", "max"])
)

b=summary_by_y

rows = []

for col in numeric_features:
    for group, data in df.groupby("y"):
        x = data[col].dropna()
        rows.append({
            "variable": col,
            "group": group,
            "n": x.shape[0],
            "missing_n": data[col].isna().sum(),
            "median": x.median(),
            "q1": x.quantile(0.25),
            "q3": x.quantile(0.75),
            "mean": x.mean(),
            "sd": x.std()
        })

numeric_by_y = pd.DataFrame(rows)

for col in categorical_features:
    print("\n====================")
    print(col)
    print(pd.crosstab(df[col], df["y"], dropna=False))
    print(pd.crosstab(df[col], df["y"], normalize="columns", dropna=False).round(3) * 100)

# =========================
# 2. Nested CV
# =========================

outer_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=10,
    random_state=42
)

inner_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=123
)

print("Outer evaluations:", outer_cv.get_n_splits(X, y_bin))
print("Inner evaluations:", inner_cv.get_n_splits(X, y_bin))


# =========================
# 3. Missing Indicators
# =========================

class MissingIndicatorAdder(BaseEstimator, TransformerMixin):
    def __init__(self, variables):
        self.variables = variables

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        for var in self.variables:
            X_copy[var + "_missing"] = X_copy[var].isna().astype(int)
        return X_copy


missing_indicator_features = numeric_features + categorical_features

missing_adder = MissingIndicatorAdder(
    variables=missing_indicator_features
)


# =========================
# 4. Preprocesadores
# =========================

numeric_transformer_simple = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer_simple = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_simple = ColumnTransformer([
    ("num", numeric_transformer_simple, numeric_features),
    ("cat", categorical_transformer_simple, categorical_features)
])

preprocess_simple_mi = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_simple, numeric_features),
        ("cat", categorical_transformer_simple, categorical_features)
    ],
    remainder="passthrough"
)


numeric_transformer_knn = Pipeline([
    ("scaler", StandardScaler()),
    ("imputer", KNNImputer(n_neighbors=5))
])

categorical_transformer_knn = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_knn = ColumnTransformer([
    ("num", numeric_transformer_knn, numeric_features),
    ("cat", categorical_transformer_knn, categorical_features)
])

preprocess_knn_mi = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_knn, numeric_features),
        ("cat", categorical_transformer_knn, categorical_features)
    ],
    remainder="passthrough"
)


numeric_transformer_mice = Pipeline([
    ("imputer", IterativeImputer(
        random_state=42,
        max_iter=20
    )),
    ("scaler", StandardScaler())
])

categorical_transformer_mice = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocess_mice = ColumnTransformer([
    ("num", numeric_transformer_mice, numeric_features),
    ("cat", categorical_transformer_mice, categorical_features)
])

preprocess_mice_mi = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer_mice, numeric_features),
        ("cat", categorical_transformer_mice, categorical_features)
    ],
    remainder="passthrough"
)

print("Preprocesadores listos.")


# =========================
# 5.1 NAM
# =========================

class NAMModule(nn.Module):
    def __init__(self, n_features, hidden=32):
        super().__init__()
        self.feature_nets = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )
            for _ in range(n_features)
        ])
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        effects = [
            net(x[:, j:j + 1])
            for j, net in enumerate(self.feature_nets)
        ]
        return torch.sum(torch.cat(effects, dim=1), dim=1) + self.bias


class NAMClassifier(ClassifierMixin,BaseEstimator):
    _estimator_type = "classifier"
    def __init__(
        self,
        hidden=32,
        lr=0.001,
        epochs=100,
        batch_size=64,
        weight_decay=1e-4,
        patience=15,
        random_state=42
    ):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.patience = patience
        self.random_state = random_state

    def fit(self, X, y):
        torch.manual_seed(self.random_state)

        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        self.model_ = NAMModule(
            n_features=X.shape[1],
            hidden=self.hidden
        )

        pos_weight = torch.tensor(
            [(y == 0).sum() / max((y == 1).sum(), 1)],
            dtype=torch.float32
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )

        optimizer = optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        best_loss = np.inf
        patience_counter = 0

        for epoch in range(self.epochs):
            self.model_.train()
            perm = torch.randperm(X_tensor.shape[0])
            epoch_loss = 0.0

            for i in range(0, X_tensor.shape[0], self.batch_size):
                idx = perm[i:i + self.batch_size]
                xb = X_tensor[idx]
                yb = y_tensor[idx]

                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= self.patience:
                break

        self.classes_ = np.array([0, 1])

        return self

    def predict_proba(self, X):
        if hasattr(X, "toarray"):
            X = X.toarray()

        X = np.asarray(X, dtype=np.float32)
        X_tensor = torch.tensor(X, dtype=torch.float32)

        self.model_.eval()
        with torch.no_grad():
            logits = self.model_(X_tensor)
            proba_1 = torch.sigmoid(logits).numpy()

        return np.column_stack([
            1 - proba_1,
            proba_1
        ])

    def predict(self, X):
        return (
            self.predict_proba(X)[:, 1] >= 0.5
        ).astype(int)

# =========================
# 5.2 Catboost
# =========================
class CatBoostSklearnWrapper(ClassifierMixin, BaseEstimator):

    _estimator_type = "classifier"

    def __init__(
        self,
        iterations=300,
        learning_rate=0.05,
        depth=6,
        l2_leaf_reg=3,
        random_seed=42
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

    def fit(self, X, y):

        if hasattr(X, "toarray"):
            X = X.toarray()

        self.model_ = CatBoostClassifier(
            iterations=self.iterations,
            learning_rate=self.learning_rate,
            depth=self.depth,
            l2_leaf_reg=self.l2_leaf_reg,
            loss_function="Logloss",
            eval_metric="AUC",
            verbose=False,
            random_seed=self.random_seed,
            auto_class_weights="Balanced"
        )

        self.model_.fit(X, y)

        self.classes_ = np.array([0, 1])

        return self

    def predict_proba(self, X):

        if hasattr(X, "toarray"):
            X = X.toarray()

        return self.model_.predict_proba(X)

    def predict(self, X):

        if hasattr(X, "toarray"):
            X = X.toarray()

        return self.model_.predict(X).astype(int)

# =========================
# 6. Modelos base
# =========================

models = {
    "LogisticRegression": LogisticRegression(
        max_iter=5000,
        class_weight="balanced",
        random_state=42
    ),

    "RandomForest": RandomForestClassifier(
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "SVM": SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42
    ),

    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1
    ),

    "CatBoost": CatBoostSklearnWrapper(
    random_seed=42
    ),

    "NAM": NAMClassifier(
        random_state=42
    )
}


# =========================
# 7. Espacios de búsqueda
# =========================

param_grids = {
    "LogisticRegression": {
        "model__C": [0.01, 0.1, 1, 10],
        "model__penalty": ["l2"],
        "model__solver": ["lbfgs"]
    },

    "RandomForest": {
        "model__n_estimators": [300, 500],
        "model__max_depth": [None, 5, 10],
        "model__max_features": ["sqrt", "log2"],
        "model__min_samples_leaf": [1, 2, 5]
    },

    "SVM": {
        "model__C": [0.1, 1, 10],
        "model__gamma": ["scale", 0.01, 0.1]
    },

    "XGBoost": {
        "model__n_estimators": [200, 300],
        "model__learning_rate": [0.03, 0.05],
        "model__max_depth": [3, 4],
        "model__subsample": [0.8, 1.0],
        "model__colsample_bytree": [0.8, 1.0],
        "model__min_child_weight": [1, 3]
    },

    "CatBoost": {
        "model__iterations": [200, 300],
        "model__learning_rate": [0.03, 0.05],
        "model__depth": [4, 6],
        "model__l2_leaf_reg": [3, 5, 7]
    },

    "NAM": {
        "model__hidden": [16, 32, 64],
        "model__lr": [0.001, 0.0005],
        "model__batch_size": [32, 64],
        "model__weight_decay": [1e-4, 1e-3],
        "model__patience": [10, 15]
    }
}

print("Modelos y espacios de búsqueda listos.")

# =========================
# 8. Constructor de pipelines
# =========================

preprocessors = {
    ("Simple", False): preprocess_simple,
    ("Simple", True): preprocess_simple_mi,

    ("KNN", False): preprocess_knn,
    ("KNN", True): preprocess_knn_mi,

    ("MICE", False): preprocess_mice,
    ("MICE", True): preprocess_mice_mi
}


def build_pipeline(model_name, filling, use_mi):
    """
    Construye automáticamente el pipeline completo
    según:
    - modelo
    - método de imputación
    - uso de Missing Indicators
    """

    model = models[model_name]

    preprocess = preprocessors[
        (filling, use_mi)
    ]

    # -------------------------
    # Pipeline con MI
    # -------------------------

    if use_mi:

        pipe = Pipeline([
            ("missing_adder", missing_adder),

            ("preprocess", preprocess),

            ("model", model)
        ])

    # -------------------------
    # Pipeline sin MI
    # -------------------------

    else:

        pipe = Pipeline([
            ("preprocess", preprocess),

            ("model", model)
        ])

    return pipe


print("Constructor de pipelines listo.")

# =========================
# 9. Métricas
# =========================

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    brier_score_loss
)


def compute_metrics(y_true, y_pred, y_prob):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    specificity = tn / (tn + fp)

    metrics = {

        # Discriminación
        "roc_auc": roc_auc_score(
            y_true,
            y_prob
        ),

        "pr_auc": average_precision_score(
            y_true,
            y_prob
        ),

        # Clasificación
        "accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "balanced_accuracy": balanced_accuracy_score(
            y_true,
            y_pred
        ),

        "precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "recall": recall_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "specificity": specificity,

        "f1": f1_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "mcc": matthews_corrcoef(
            y_true,
            y_pred
        ),

        # Calibración
        "brier": brier_score_loss(
            y_true,
            y_prob
        )
    }

    return metrics


print("Funciones de métricas listas.")


# =========================
# 10. Nested CV y evaluación
# =========================

from sklearn.model_selection import RandomizedSearchCV
import time

results = []
best_params_results = []

fillings = ["Simple", "KNN", "MICE"]
mi_options = [False, True]

selected_models = [
    "LogisticRegression",
    "RandomForest",
    "CatBoost",
    "SVM",
    "XGBoost"
]

scoring_primary = "average_precision"

inner_cv = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=123
)

outer_splits = list(
    enumerate(
        outer_cv.split(X, y_bin),
        start=1
    )
)

for filling in fillings:

    for use_mi in mi_options:

        for model_name in selected_models:

            print(
                f"\nEjecutando: {model_name} | "
                f"{filling} | MI={use_mi}"
            )

            for split_id, (train_idx, test_idx) in outer_splits:

                start_time = time.time()

                X_train_outer = X.iloc[train_idx].copy()
                X_test_outer = X.iloc[test_idx].copy()

                y_train_outer = y_bin.iloc[train_idx].copy()
                y_test_outer = y_bin.iloc[test_idx].copy()

                pipe = build_pipeline(
                    model_name=model_name,
                    filling=filling,
                    use_mi=use_mi
                )

                n_possible = np.prod([
                    len(v)
                    for v in param_grids[model_name].values()
                ])

                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=param_grids[model_name],
                    n_iter=min(10, n_possible),
                    scoring=scoring_primary,
                    cv=inner_cv,
                    n_jobs=-1,
                    random_state=42,
                    refit=True,
                    verbose=0
                )

                search.fit(
                    X_train_outer,
                    y_train_outer
                )

                best_model = search.best_estimator_

                y_prob = best_model.predict_proba(
                    X_test_outer
                )[:, 1]

                y_pred = best_model.predict(
                    X_test_outer
                )

                metric_values = compute_metrics(
                    y_true=y_test_outer,
                    y_pred=y_pred,
                    y_prob=y_prob
                )

                elapsed = time.time() - start_time

                row = {
                    "model": model_name,
                    "filling": filling,
                    "use_mi": use_mi,
                    "split_id": split_id,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "elapsed_seconds": round(elapsed, 2)
                }

                row.update(metric_values)
                results.append(row)

                best_params_results.append({
                    "model": model_name,
                    "filling": filling,
                    "use_mi": use_mi,
                    "split_id": split_id,
                    "best_score_inner_pr_auc": search.best_score_,
                    "best_params": search.best_params_
                })

                print(
                    f"Split {split_id:02d} | "
                    f"PR-AUC={metric_values['pr_auc']:.3f} | "
                    f"ROC-AUC={metric_values['roc_auc']:.3f} | "
                    f"MCC={metric_values['mcc']:.3f} | "
                    f"{elapsed:.1f}s"
                )

                pd.DataFrame(results).to_csv(
                    "nested_cv_results_mac_partial.csv",
                    index=False
                )

                pd.DataFrame(best_params_results).to_csv(
                    "nested_cv_best_params_mac_partial.csv",
                    index=False
                )

print("\nNested CV finalizado.")
# =========================
# 11. Guardado final
# =========================

results_df = pd.DataFrame(results)
best_params_df = pd.DataFrame(best_params_results)

results_df.to_csv(
    "nested_cv_results_mac.csv",
    index=False
)

best_params_df.to_csv(
    "nested_cv_best_params_mac.csv",
    index=False
)

print("Resultados finales guardados.")
print(results_df.shape)
print(best_params_df.shape)

# =========================
# 12. Nested CV - NAM 
# Se hizo por aparte porque necesita un codigo diferente
# =========================

from sklearn.model_selection import RandomizedSearchCV
import time

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

results_nam = []
best_params_results_nam = []

fillings = ["Simple", "KNN", "MICE"]
mi_options = [False, True]

selected_models = [
    "NAM"
]

scoring_primary = "average_precision"

inner_cv_nam = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=123
)

outer_splits = list(
    enumerate(
        outer_cv.split(X, y_bin),
        start=1
    )
)

for filling in fillings:

    for use_mi in mi_options:

        for model_name in selected_models:

            print(
                f"\nEjecutando: {model_name} | "
                f"{filling} | MI={use_mi}"
            )

            for split_id, (train_idx, test_idx) in outer_splits:

                start_time = time.time()

                X_train_outer = X.iloc[train_idx].copy()
                X_test_outer = X.iloc[test_idx].copy()

                y_train_outer = y_bin.iloc[train_idx].copy()
                y_test_outer = y_bin.iloc[test_idx].copy()

                pipe = build_pipeline(
                    model_name=model_name,
                    filling=filling,
                    use_mi=use_mi
                )

                n_possible = np.prod([
                    len(v)
                    for v in param_grids[model_name].values()
                ])

                search = RandomizedSearchCV(
                    estimator=pipe,
                    param_distributions=param_grids[model_name],
                    n_iter=min(3, n_possible),
                    scoring=scoring_primary,
                    cv=inner_cv_nam,
                    n_jobs=4,
                    random_state=42,
                    refit=True,
                    verbose=0
                )

                search.fit(
                    X_train_outer,
                    y_train_outer
                )

                best_model = search.best_estimator_

                y_prob = best_model.predict_proba(
                    X_test_outer
                )[:, 1]

                y_pred = best_model.predict(
                    X_test_outer
                )

                metric_values = compute_metrics(
                    y_true=y_test_outer,
                    y_pred=y_pred,
                    y_prob=y_prob
                )

                elapsed = time.time() - start_time

                row = {
                    "model": model_name,
                    "filling": filling,
                    "use_mi": use_mi,
                    "split_id": split_id,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "elapsed_seconds": round(elapsed, 2)
                }

                row.update(metric_values)
                results_nam.append(row)

                best_params_results_nam.append({
                    "model": model_name,
                    "filling": filling,
                    "use_mi": use_mi,
                    "split_id": split_id,
                    "best_score_inner_pr_auc": search.best_score_,
                    "best_params": search.best_params_
                })

                print(
                    f"Split {split_id:02d} | "
                    f"PR-AUC={metric_values['pr_auc']:.3f} | "
                    f"ROC-AUC={metric_values['roc_auc']:.3f} | "
                    f"MCC={metric_values['mcc']:.3f} | "
                    f"{elapsed:.1f}s"
                )

                pd.DataFrame(results_nam).to_csv(
                    "nested_cv_results_nam_partial.csv",
                    index=False
                )

                pd.DataFrame(best_params_results_nam).to_csv(
                    "nested_cv_best_params_nam_partial.csv",
                    index=False
                )

print("\nNested CV NAM finalizado.")

# =========================
# 13. Guardado final NAM
# =========================

results_nam_df = pd.DataFrame(results_nam)
best_params_nam_df = pd.DataFrame(best_params_results_nam)

results_nam_df.to_csv(
    "nested_cv_results_nam.csv",
    index=False
)

best_params_nam_df.to_csv(
    "nested_cv_best_params_nam.csv",
    index=False
)

print("Resultados finales NAM guardados.")
print(results_nam_df.shape)
print(best_params_nam_df.shape)

# =========================
# 14. Unir resultados
# =========================

import pandas as pd

results_files = [
    "nested_cv_results_mac.csv",
    "nested_cv_results_nam.csv"
]

best_params_files = [
    "nested_cv_best_params_mac.csv",
    "nested_cv_best_params_nam.csv"
]

results_df = pd.concat(
    [pd.read_csv(file) for file in results_files],
    ignore_index=True
)

best_params_df = pd.concat(
    [pd.read_csv(file) for file in best_params_files],
    ignore_index=True
)

results_df.to_csv(
    "nested_cv_results_complete.csv",
    index=False
)

best_params_df.to_csv(
    "nested_cv_best_params_complete.csv",
    index=False
)

print("Resultados unidos.")
print("results_df:", results_df.shape)
print("best_params_df:", best_params_df.shape)

# =========================
# 15. Verificación de integridad
# =========================

expected_models = [
    "LogisticRegression",
    "RandomForest",
    "SVM",
    "XGBoost",
    "CatBoost",
    "NAM"
]

expected_fillings = [
    "Simple",
    "KNN",
    "MICE"
]

expected_mi = [
    False,
    True
]

summary_check = (
    results_df
    .groupby(["model", "filling", "use_mi"])
    .agg(
        n_splits=("split_id", "nunique"),
        n_rows=("split_id", "count")
    )
    .reset_index()
    .sort_values(["model", "filling", "use_mi"])
)

print(summary_check)

duplicates = results_df.duplicated(
    subset=["model", "filling", "use_mi", "split_id"]
).sum()

print("Duplicados:", duplicates)

missing_combinations = summary_check[
    summary_check["n_splits"] != 50
]

print("Combinaciones incompletas:")
print(missing_combinations)

# =========================
# 16. Resumen descriptivo
# =========================

summary_metrics = (
    results_df
    .groupby(["model", "filling", "use_mi"])
    .agg(
        pr_auc_mean=("pr_auc", "mean"),
        pr_auc_std=("pr_auc", "std"),

        roc_auc_mean=("roc_auc", "mean"),
        roc_auc_std=("roc_auc", "std"),

        mcc_mean=("mcc", "mean"),
        mcc_std=("mcc", "std"),

        brier_mean=("brier", "mean"),
        brier_std=("brier", "std")
    )
    .reset_index()
)

summary_metrics = summary_metrics.sort_values(
    by="pr_auc_mean",
    ascending=False
)

summary_metrics

# =========================
# 17. Resumen de resultados
# =========================
import pandas as pd
import numpy as np

metrics = [
    "pr_auc", "roc_auc", "mcc", "balanced_accuracy",
    "accuracy", "precision", "recall", "specificity", "brier"
]

# 1. Tabla maestra: 36 pipelines
pipeline_summary = (
    results_df
    .groupby(["model", "filling", "use_mi"])[metrics]
    .agg(["mean", "std", "median", "min", "max"])
)

pipeline_summary.columns = [
    f"{metric}_{stat}" for metric, stat in pipeline_summary.columns
]

pipeline_summary = (
    pipeline_summary
    .reset_index()
    .sort_values("pr_auc_mean", ascending=False)
)

print("\n=== TABLA MAESTRA 36 PIPELINES ===")
print(pipeline_summary.to_string(index=False))


# 2. Tabla compacta media ± DE
pipeline_pretty = pipeline_summary[["model", "filling", "use_mi"]].copy()

for m in metrics:
    pipeline_pretty[m] = (
        pipeline_summary[f"{m}_mean"].round(3).astype(str)
        + " ± " +
        pipeline_summary[f"{m}_std"].round(3).astype(str)
    )

print("\n=== TABLA COMPACTA MEDIA ± DE ===")
print(pipeline_pretty.to_string(index=False))


# 3. Ranking general por PR-AUC
print("\n=== RANKING GENERAL POR PR-AUC ===")
print(
    pipeline_summary[
        ["model", "filling", "use_mi",
         "pr_auc_mean", "pr_auc_std",
         "roc_auc_mean", "mcc_mean",
         "balanced_accuracy_mean", "brier_mean"]
    ].to_string(index=False)
)


# 4. Top 5 por métrica
ranking_metrics = {
    "pr_auc": False,
    "roc_auc": False,
    "mcc": False,
    "balanced_accuracy": False,
    "brier": True
}

for metric, ascending in ranking_metrics.items():
    print(f"\n=== TOP 5 {metric.upper()} ===")
    print(
        pipeline_summary
        .sort_values(f"{metric}_mean", ascending=ascending)
        [["model", "filling", "use_mi",
          f"{metric}_mean", f"{metric}_std"]]
        .head(5)
        .to_string(index=False)
    )


# 5. Resumen por algoritmo
model_summary = (
    results_df
    .groupby("model")[metrics]
    .agg(["mean", "std"])
)

model_summary.columns = [
    f"{metric}_{stat}" for metric, stat in model_summary.columns
]

model_summary = (
    model_summary
    .reset_index()
    .sort_values("pr_auc_mean", ascending=False)
)

print("\n=== RESUMEN POR ALGORITMO ===")
print(model_summary.to_string(index=False))


# 6. Resumen por imputación
filling_summary = (
    results_df
    .groupby("filling")[metrics]
    .agg(["mean", "std"])
)

filling_summary.columns = [
    f"{metric}_{stat}" for metric, stat in filling_summary.columns
]

filling_summary = (
    filling_summary
    .reset_index()
    .sort_values("pr_auc_mean", ascending=False)
)

print("\n=== RESUMEN POR IMPUTACIÓN ===")
print(filling_summary.to_string(index=False))


# 7. Resumen global con/sin MI
mi_summary = (
    results_df
    .groupby("use_mi")[metrics]
    .agg(["mean", "std"])
)

mi_summary.columns = [
    f"{metric}_{stat}" for metric, stat in mi_summary.columns
]

mi_summary = mi_summary.reset_index()

print("\n=== RESUMEN GLOBAL CON/SIN MI ===")
print(mi_summary.to_string(index=False))


# 8. Estabilidad: más y menos variables en PR-AUC
print("\n=== PIPELINES MÁS ESTABLES EN PR-AUC ===")
print(
    pipeline_summary
    .sort_values("pr_auc_std", ascending=True)
    [["model", "filling", "use_mi", "pr_auc_mean", "pr_auc_std"]]
    .head(10)
    .to_string(index=False)
)

print("\n=== PIPELINES MÁS VARIABLES EN PR-AUC ===")
print(
    pipeline_summary
    .sort_values("pr_auc_std", ascending=False)
    [["model", "filling", "use_mi", "pr_auc_mean", "pr_auc_std"]]
    .head(10)
    .to_string(index=False)
)


# 9. Distribución del mejor pipeline según PR-AUC
best_row = pipeline_summary.iloc[0]
best_model = best_row["model"]
best_filling = best_row["filling"]
best_mi = best_row["use_mi"]

best_pipeline_data = results_df[
    (results_df["model"] == best_model) &
    (results_df["filling"] == best_filling) &
    (results_df["use_mi"] == best_mi)
]

best_dist = best_pipeline_data[metrics].describe().T

print("\n=== MEJOR PIPELINE SEGÚN PR-AUC ===")
print(best_row[["model", "filling", "use_mi", "pr_auc_mean", "pr_auc_std"]])

print("\n=== DISTRIBUCIÓN DE MÉTRICAS DEL MEJOR PIPELINE ===")
print(best_dist.to_string())


# 10. Ganadores por split según PR-AUC
split_winners = (
    results_df
    .sort_values(["split_id", "pr_auc"], ascending=[True, False])
    .groupby("split_id")
    .first()
    .reset_index()
)

print("\n=== GANADORES POR SPLIT SEGÚN PR-AUC: MODELO ===")
print(split_winners["model"].value_counts().to_string())

print("\n=== GANADORES POR SPLIT SEGÚN PR-AUC: PIPELINE COMPLETO ===")
print(
    split_winners
    .groupby(["model", "filling", "use_mi"])
    .size()
    .reset_index(name="n_wins")
    .sort_values("n_wins", ascending=False)
    .to_string(index=False)
)


# 11. Guardar tablas
pipeline_summary.to_csv("pipeline_summary_36.csv", index=False)
pipeline_pretty.to_csv("pipeline_summary_pretty.csv", index=False)
model_summary.to_csv("model_summary.csv", index=False)
filling_summary.to_csv("filling_summary.csv", index=False)
mi_summary.to_csv("mi_summary.csv", index=False)
split_winners.to_csv("split_winners_pr_auc.csv", index=False)




# ============================================================
# 18. IMPACTO DE MISSING INDICATORS
#
# Comparación pareada:
#   Δ = métrica(con MI) - métrica(sin MI)
#
# Para Brier:
#   Δ = Brier(sin MI) - Brier(con MI)
#   Así, Δ > 0 siempre significa mejora con Missing Indicators.
#
# Análisis principal:
#   Bayesian correlated t-test con ROPE = 0.005
#
# Sensibilidad 1:
#   Bayesian correlated t-test sin ROPE, equivalente a ROPE = 0
#
# Sensibilidad 2:
#   Wilcoxon signed-rank test con corrección Bonferroni
#   dentro de cada familia inferencial.
#   Familia inferencial = métrica.
#   Corrección: m = 18 pruebas por métrica.
# ============================================================

import numpy as np
import pandas as pd
from scipy.stats import t, wilcoxon

# =========================
# 1. Cargar resultados
# =========================

results_df = pd.read_csv("nested_cv_results_complete.csv")
results_df["use_mi"] = results_df["use_mi"].astype(bool)

metrics_to_test = [
    "pr_auc",
    "roc_auc",
    "mcc",
    "balanced_accuracy",
    "brier"
]

model_order = [
    "LogisticRegression",
    "RandomForest",
    "SVM",
    "XGBoost",
    "CatBoost",
    "NAM"
]

filling_order = [
    "Simple",
    "KNN",
    "MICE"
]

# 5-fold outer:
# test/train = 20/80 = 1/4
k_outer = 5
rho = 1 / (k_outer - 1)

ROPE_MAIN = 0.005
ROPE_SENS = 0.0

N_TESTS_PER_FAMILY = len(model_order) * len(filling_order)

print("rho:", rho)
print("ROPE principal:", ROPE_MAIN)
print("ROPE sensibilidad:", ROPE_SENS)
print("Corrección Bonferroni por familia:", N_TESTS_PER_FAMILY)


# =========================
# 2. Funciones auxiliares
# =========================

def get_ordered_pair(results_df, model_name, filling, metric):
    df_no_mi = results_df[
        (results_df["model"] == model_name) &
        (results_df["filling"] == filling) &
        (results_df["use_mi"] == False)
    ].sort_values("split_id")

    df_mi = results_df[
        (results_df["model"] == model_name) &
        (results_df["filling"] == filling) &
        (results_df["use_mi"] == True)
    ].sort_values("split_id")

    assert len(df_no_mi) == 50, (
        model_name, filling, metric, "No MI", len(df_no_mi)
    )

    assert len(df_mi) == 50, (
        model_name, filling, metric, "MI", len(df_mi)
    )

    assert np.all(
        df_no_mi["split_id"].values == df_mi["split_id"].values
    ), (model_name, filling, metric)

    values_no_mi = df_no_mi[metric].values
    values_mi = df_mi[metric].values

    return values_no_mi, values_mi


def compute_deltas(values_no_mi, values_mi, metric):
    if metric == "brier":
        # En Brier, menor es mejor.
        # Por eso se invierte la diferencia.
        return values_no_mi - values_mi
    else:
        # En las demás métricas, mayor es mejor.
        return values_mi - values_no_mi


def bayesian_correlated_ttest_rope(deltas, rho, rope):
    deltas = np.asarray(deltas, dtype=float)

    n = len(deltas)
    mean_delta = np.mean(deltas)
    std_delta = np.std(deltas, ddof=1)
    df = n - 1

    corrected_sd = np.sqrt(
        ((1 / n) + rho) * (std_delta ** 2)
    )

    posterior = t(
        df=df,
        loc=mean_delta,
        scale=corrected_sd
    )

    prob_worse = posterior.cdf(-rope)

    prob_equivalent = (
        posterior.cdf(rope)
        -
        posterior.cdf(-rope)
    )

    prob_better = 1 - posterior.cdf(rope)

    credible_low = posterior.ppf(0.025)
    credible_high = posterior.ppf(0.975)

    effect_size = (
        mean_delta / std_delta
        if std_delta > 0
        else np.nan
    )

    return {
        "n_splits": n,
        "delta_mean": mean_delta,
        "delta_std": std_delta,
        "credible_low": credible_low,
        "credible_high": credible_high,
        "posterior_prob_mi_worse": prob_worse,
        "posterior_prob_equivalent": prob_equivalent,
        "posterior_prob_mi_better": prob_better,
        "effect_size": effect_size,
        "rope": rope,
        "rho": rho
    }


def wilcoxon_signed_rank_corrected(deltas, n_tests_family):
    deltas = np.asarray(deltas, dtype=float)

    # Wilcoxon contrasta si la mediana de las diferencias es 0.
    # alternative="greater" porque Δ > 0 significa mejora con MI.
    try:
        stat, p_raw = wilcoxon(
            deltas,
            alternative="greater",
            zero_method="wilcox",
            correction=False,
            mode="auto"
        )
    except ValueError:
        stat, p_raw = np.nan, np.nan

    p_bonferroni = (
        min(p_raw * n_tests_family, 1.0)
        if not np.isnan(p_raw)
        else np.nan
    )

    return {
        "wilcoxon_stat": stat,
        "p_wilcoxon_raw": p_raw,
        "p_wilcoxon_bonferroni": p_bonferroni,
        "n_tests_family": n_tests_family
    }


# =========================
# 3. Ejecutar análisis
# =========================

rows_main_rope = []
rows_sens_rope0 = []
rows_wilcoxon = []

for metric in metrics_to_test:

    for model_name in model_order:

        for filling in filling_order:

            values_no_mi, values_mi = get_ordered_pair(
                results_df=results_df,
                model_name=model_name,
                filling=filling,
                metric=metric
            )

            deltas = compute_deltas(
                values_no_mi=values_no_mi,
                values_mi=values_mi,
                metric=metric
            )

            base_row = {
                "metric": metric,
                "model": model_name,
                "filling": filling,
                "mean_no_mi": np.mean(values_no_mi),
                "std_no_mi": np.std(values_no_mi, ddof=1),
                "mean_mi": np.mean(values_mi),
                "std_mi": np.std(values_mi, ddof=1)
            }

            # Principal: ROPE = 0.005
            main_stats = bayesian_correlated_ttest_rope(
                deltas=deltas,
                rho=rho,
                rope=ROPE_MAIN
            )

            row_main = base_row.copy()
            row_main.update(main_stats)
            rows_main_rope.append(row_main)

            # Sensibilidad: ROPE = 0
            sens_stats = bayesian_correlated_ttest_rope(
                deltas=deltas,
                rho=rho,
                rope=ROPE_SENS
            )

            row_sens = base_row.copy()
            row_sens.update(sens_stats)
            rows_sens_rope0.append(row_sens)

            # Wilcoxon + Bonferroni
            wilcoxon_stats = wilcoxon_signed_rank_corrected(
                deltas=deltas,
                n_tests_family=N_TESTS_PER_FAMILY
            )

            row_wilcoxon = base_row.copy()
            row_wilcoxon.update({
                "delta_mean": np.mean(deltas),
                "delta_std": np.std(deltas, ddof=1),
                "delta_median": np.median(deltas),
                "delta_q1": np.quantile(deltas, 0.25),
                "delta_q3": np.quantile(deltas, 0.75)
            })
            row_wilcoxon.update(wilcoxon_stats)
            rows_wilcoxon.append(row_wilcoxon)


bayes_rope0005_all = pd.DataFrame(rows_main_rope)
bayes_rope0_all = pd.DataFrame(rows_sens_rope0)
wilcoxon_all = pd.DataFrame(rows_wilcoxon)


# =========================
# 4. Ordenar tablas
# =========================

for df in [bayes_rope0005_all, bayes_rope0_all, wilcoxon_all]:
    df["model"] = pd.Categorical(
        df["model"],
        categories=model_order,
        ordered=True
    )

    df["filling"] = pd.Categorical(
        df["filling"],
        categories=filling_order,
        ordered=True
    )

    df["metric"] = pd.Categorical(
        df["metric"],
        categories=metrics_to_test,
        ordered=True
    )

bayes_rope0005_all = (
    bayes_rope0005_all
    .sort_values(["metric", "model", "filling"])
    .reset_index(drop=True)
)

bayes_rope0_all = (
    bayes_rope0_all
    .sort_values(["metric", "model", "filling"])
    .reset_index(drop=True)
)

wilcoxon_all = (
    wilcoxon_all
    .sort_values(["metric", "model", "filling"])
    .reset_index(drop=True)
)


# =========================
# 5. Tablas específicas para PR-AUC
# =========================

bayes_rope0005_pr_auc = bayes_rope0005_all[
    bayes_rope0005_all["metric"] == "pr_auc"
].copy()

bayes_rope0_pr_auc = bayes_rope0_all[
    bayes_rope0_all["metric"] == "pr_auc"
].copy()

wilcoxon_pr_auc = wilcoxon_all[
    wilcoxon_all["metric"] == "pr_auc"
].copy()


# =========================
# 6. Tablas redondeadas para revisar
# =========================

def round_df(df, digits=4):
    out = df.copy()
    num_cols = out.select_dtypes(include=[np.number]).columns
    out[num_cols] = out[num_cols].round(digits)
    return out


bayes_rope0005_pr_auc_print = round_df(
    bayes_rope0005_pr_auc[
        [
            "model",
            "filling",
            "mean_no_mi",
            "mean_mi",
            "delta_mean",
            "credible_low",
            "credible_high",
            "posterior_prob_mi_worse",
            "posterior_prob_equivalent",
            "posterior_prob_mi_better",
            "effect_size"
        ]
    ]
)

bayes_rope0_pr_auc_print = round_df(
    bayes_rope0_pr_auc[
        [
            "model",
            "filling",
            "delta_mean",
            "credible_low",
            "credible_high",
            "posterior_prob_mi_worse",
            "posterior_prob_equivalent",
            "posterior_prob_mi_better"
        ]
    ]
)

wilcoxon_pr_auc_print = round_df(
    wilcoxon_pr_auc[
        [
            "model",
            "filling",
            "delta_mean",
            "delta_median",
            "delta_q1",
            "delta_q3",
            "wilcoxon_stat",
            "p_wilcoxon_raw",
            "p_wilcoxon_bonferroni"
        ]
    ]
)


# =========================
# 7. Guardar resultados
# =========================

bayes_rope0005_all.to_csv(
    "bayesian_correlated_rope0005_all_metrics.csv",
    index=False
)

bayes_rope0_all.to_csv(
    "bayesian_correlated_rope0_all_metrics.csv",
    index=False
)

wilcoxon_all.to_csv(
    "wilcoxon_signed_rank_bonferroni_all_metrics.csv",
    index=False
)

bayes_rope0005_pr_auc.to_csv(
    "bayesian_correlated_rope0005_pr_auc.csv",
    index=False
)

bayes_rope0_pr_auc.to_csv(
    "bayesian_correlated_rope0_pr_auc.csv",
    index=False
)

wilcoxon_pr_auc.to_csv(
    "wilcoxon_signed_rank_bonferroni_pr_auc.csv",
    index=False
)


# =========================
# 8. Imprimir salidas 
# =========================

print("\n" + "=" * 100)
print("ANÁLISIS PRINCIPAL: BAYESIAN CORRELATED T-TEST CON ROPE = 0.005 | PR-AUC")
print("=" * 100)
print(bayes_rope0005_pr_auc_print.to_string(index=False))

print("\n" + "=" * 100)
print("SENSIBILIDAD BAYESIANA: ROPE = 0 | PR-AUC")
print("=" * 100)
print(bayes_rope0_pr_auc_print.to_string(index=False))

print("\n" + "=" * 100)
print("SENSIBILIDAD FRECUENTISTA: WILCOXON + BONFERRONI | PR-AUC")
print("=" * 100)
print(wilcoxon_pr_auc_print.to_string(index=False))

print("\nObjetos creados:")
print("bayes_rope0005_all")
print("bayes_rope0_all")
print("wilcoxon_all")
print("bayes_rope0005_pr_auc")
print("bayes_rope0_pr_auc")
print("wilcoxon_pr_auc")


# ============================================================
# RESUMEN PARA MÉTRICAS SECUNDARIAS
# ROC-AUC, MCC, Balanced Accuracy y Brier
# ============================================================

secondary_metrics = [
    "roc_auc",
    "mcc",
    "balanced_accuracy",
    "brier"
]

# Tabla resumen por métrica
summary_secondary = []

for metric in secondary_metrics:
    bayes_metric = bayes_rope0005_all[
        bayes_rope0005_all["metric"] == metric
    ].copy()

    bayes0_metric = bayes_rope0_all[
        bayes_rope0_all["metric"] == metric
    ].copy()

    wilcox_metric = wilcoxon_all[
        wilcoxon_all["metric"] == metric
    ].copy()

    summary_secondary.append({
        "metric": metric,
        "n_comparisons": len(bayes_metric),

        # Bayesian ROPE = 0.005
        "mean_delta": bayes_metric["delta_mean"].mean(),
        "min_delta": bayes_metric["delta_mean"].min(),
        "max_delta": bayes_metric["delta_mean"].max(),

        "mean_p_better_rope": bayes_metric["posterior_prob_mi_better"].mean(),
        "n_p_better_rope_ge_090": (bayes_metric["posterior_prob_mi_better"] >= 0.90).sum(),
        "n_p_better_rope_ge_080": (bayes_metric["posterior_prob_mi_better"] >= 0.80).sum(),
        "n_p_worse_rope_ge_050": (bayes_metric["posterior_prob_mi_worse"] >= 0.50).sum(),

        # Bayesian ROPE = 0
        "mean_p_better_rope0": bayes0_metric["posterior_prob_mi_better"].mean(),
        "n_p_better_rope0_ge_090": (bayes0_metric["posterior_prob_mi_better"] >= 0.90).sum(),

        # Wilcoxon
        "n_wilcoxon_bonf_significant": (wilcox_metric["p_wilcoxon_bonferroni"] < 0.05).sum()
    })

summary_secondary = pd.DataFrame(summary_secondary)

print("\n" + "=" * 100)
print("RESUMEN GLOBAL DE MÉTRICAS SECUNDARIAS")
print("=" * 100)
print(summary_secondary.round(4).to_string(index=False))


# ============================================================
# TABLAS DETALLADAS POR MÉTRICA SECUNDARIA
# ============================================================

for metric in secondary_metrics:
    print("\n" + "=" * 100)
    print(f"BAYES ROPE = 0.005 | {metric.upper()}")
    print("=" * 100)

    tmp = bayes_rope0005_all[
        bayes_rope0005_all["metric"] == metric
    ][
        [
            "model",
            "filling",
            "mean_no_mi",
            "mean_mi",
            "delta_mean",
            "credible_low",
            "credible_high",
            "posterior_prob_mi_worse",
            "posterior_prob_equivalent",
            "posterior_prob_mi_better"
        ]
    ].copy()

    print(tmp.round(4).to_string(index=False))


    print("\n" + "-" * 100)
    print(f"WILCOXON + BONFERRONI | {metric.upper()}")
    print("-" * 100)

    tmp_w = wilcoxon_all[
        wilcoxon_all["metric"] == metric
    ][
        [
            "model",
            "filling",
            "delta_mean",
            "delta_median",
            "p_wilcoxon_bonferroni"
        ]
    ].copy()

    print(tmp_w.round(4).to_string(index=False))
    
    
    
# ============================================================
# 18. MODELO DE REFERENCIA: MISSING INDICATORS ONLY
#
# Objetivo:
# Evaluar cuánto rendimiento predictivo puede obtenerse
# utilizando exclusivamente el patrón de ausencia de datos,
# sin utilizar ninguna variable clínica original.
#
# Se mantienen los 6 modelos utilizados en el análisis principal.
# No se utiliza ningún método de imputación, ya que los MI son
# variables binarias completas y no presentan valores faltantes.
#
# Cada modelo se evalúa mediante los mismos 50 splits externos
# utilizados en el análisis principal.
#
# Total:
#   6 modelos x 50 outer splits = 300 evaluaciones
#
# El resumen final se calcula de forma análoga a la Tabla 5:
# se consideran conjuntamente todas las evaluaciones MI-only.
# ============================================================

import pandas as pd
import numpy as np
import time


# =========================
# 1. Crear Missing Indicators
# =========================

# Los indicadores representan únicamente si cada variable
# original presenta un valor ausente.
#
# No se incluyen las variables clínicas originales.

mi_features = [
    var + "_missing"
    for var in numeric_features + categorical_features
]

X_mi_only = pd.DataFrame(index=X.index)

for var in numeric_features + categorical_features:
    X_mi_only[var + "_missing"] = X[var].isna().astype(int)

y_mi_only = y_bin.copy()


print("\n" + "=" * 100)
print("MODELO DE REFERENCIA: MISSING INDICATORS ONLY")
print("=" * 100)

print("X_mi_only:", X_mi_only.shape)
print("Número de Missing Indicators:", len(mi_features))
print("Variables utilizadas:")
print(mi_features)

print("\nComprobación de valores faltantes en los MI:")
print(X_mi_only.isna().sum().sum())


# =========================
# 2. Modelos
# =========================

# Se utilizan los mismos seis modelos del análisis principal.
# Los espacios de búsqueda de hiperparámetros también se mantienen
# iguales a los definidos anteriormente.

selected_models_mi_only = [
    "LogisticRegression",
    "RandomForest",
    "SVM",
    "XGBoost",
    "CatBoost",
    "NAM"
]


# =========================
# 3. Preprocesamiento
# =========================

# Los Missing Indicators ya son variables binarias completas.
# Por tanto, no se realiza imputación.
#
# Para los modelos que se benefician del escalado se utiliza
# StandardScaler. Para los modelos basados en árboles el escalado
# no es necesario, pero se mantiene una estructura común de
# preprocesamiento para los modelos que lo requieren.

preprocess_mi_only = ColumnTransformer(
    transformers=[
        (
            "mi",
            StandardScaler(),
            mi_features
        )
    ],
    remainder="drop"
)


# =========================
# 4. Construcción de pipeline
# =========================

def build_mi_only_pipeline(model_name):

    model = models[model_name]

    pipe = Pipeline([
        ("preprocess", preprocess_mi_only),
        ("model", model)
    ])

    return pipe


print("\nPipeline MI-only listo.")


# =========================
# 5. Nested CV MI-only
# =========================

results_mi_only = []
best_params_results_mi_only = []

# Se utilizan exactamente los mismos splits externos
# definidos para el análisis principal.

outer_splits_mi_only = list(
    enumerate(
        outer_cv.split(X_mi_only, y_mi_only),
        start=1
    )
)


# Se mantiene el mismo esquema de validación interna.

inner_cv_mi_only = RepeatedStratifiedKFold(
    n_splits=5,
    n_repeats=5,
    random_state=123
)


for model_name in selected_models_mi_only:

    print(
        f"\nEjecutando MI-only: {model_name}"
    )

    for split_id, (train_idx, test_idx) in outer_splits_mi_only:

        start_time = time.time()

        X_train_outer = X_mi_only.iloc[train_idx].copy()
        X_test_outer = X_mi_only.iloc[test_idx].copy()

        y_train_outer = y_mi_only.iloc[train_idx].copy()
        y_test_outer = y_mi_only.iloc[test_idx].copy()


        # -------------------------
        # Pipeline
        # -------------------------

        pipe = build_mi_only_pipeline(
            model_name=model_name
        )


        # -------------------------
        # Espacio de búsqueda
        # -------------------------

        n_possible = np.prod([
            len(v)
            for v in param_grids[model_name].values()
        ])


        # NAM requiere menos configuraciones debido
        # al coste computacional.

        if model_name == "NAM":
            n_iter_mi_only = min(3, n_possible)
            n_jobs_mi_only = 4
        else:
            n_iter_mi_only = min(10, n_possible)
            n_jobs_mi_only = -1


        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grids[model_name],
            n_iter=n_iter_mi_only,
            scoring="average_precision",
            cv=inner_cv_mi_only,
            n_jobs=n_jobs_mi_only,
            random_state=42,
            refit=True,
            verbose=0
        )


        # -------------------------
        # Ajuste en el outer train
        # -------------------------

        search.fit(
            X_train_outer,
            y_train_outer
        )


        best_model = search.best_estimator_


        # -------------------------
        # Predicción en outer test
        # -------------------------

        y_prob = best_model.predict_proba(
            X_test_outer
        )[:, 1]

        y_pred = best_model.predict(
            X_test_outer
        )


        # -------------------------
        # Métricas
        # -------------------------

        metric_values = compute_metrics(
            y_true=y_test_outer,
            y_pred=y_pred,
            y_prob=y_prob
        )


        elapsed = time.time() - start_time


        # -------------------------
        # Guardar resultados
        # -------------------------

        row = {
            "model": model_name,
            "analysis": "MI-only",
            "split_id": split_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "elapsed_seconds": round(elapsed, 2)
        }

        row.update(metric_values)

        results_mi_only.append(row)


        # -------------------------
        # Guardar mejores parámetros
        # -------------------------

        best_params_results_mi_only.append({
            "model": model_name,
            "analysis": "MI-only",
            "split_id": split_id,
            "best_score_inner_pr_auc": search.best_score_,
            "best_params": search.best_params_
        })


        print(
            f"Split {split_id:02d} | "
            f"PR-AUC={metric_values['pr_auc']:.3f} | "
            f"ROC-AUC={metric_values['roc_auc']:.3f} | "
            f"MCC={metric_values['mcc']:.3f} | "
            f"{elapsed:.1f}s"
        )


        # -------------------------
        # Guardado parcial
        # -------------------------

        pd.DataFrame(results_mi_only).to_csv(
            "nested_cv_results_mi_only_partial.csv",
            index=False
        )

        pd.DataFrame(
            best_params_results_mi_only
        ).to_csv(
            "nested_cv_best_params_mi_only_partial.csv",
            index=False
        )


print("\nNested CV MI-only finalizado.")


# =========================
# 6. Guardado final
# =========================

results_mi_only_df = pd.DataFrame(
    results_mi_only
)

best_params_mi_only_df = pd.DataFrame(
    best_params_results_mi_only
)


results_mi_only_df.to_csv(
    "nested_cv_results_mi_only.csv",
    index=False
)

best_params_mi_only_df.to_csv(
    "nested_cv_best_params_mi_only.csv",
    index=False
)


print("\nResultados MI-only guardados.")
print(
    "results_mi_only_df:",
    results_mi_only_df.shape
)

print(
    "best_params_mi_only_df:",
    best_params_mi_only_df.shape
)


# =========================
# 7. Verificación
# =========================

print("\n" + "=" * 100)
print("VERIFICACIÓN DE INTEGRIDAD MI-ONLY")
print("=" * 100)


summary_check_mi_only = (
    results_mi_only_df
    .groupby("model")
    .agg(
        n_splits=("split_id", "nunique"),
        n_rows=("split_id", "count")
    )
    .reset_index()
)


print(summary_check_mi_only)


duplicates_mi_only = results_mi_only_df.duplicated(
    subset=["model", "split_id"]
).sum()


print(
    "Duplicados:",
    duplicates_mi_only
)


expected_n_models_mi_only = 6
expected_n_splits_mi_only = 50
expected_total_mi_only = (
    expected_n_models_mi_only *
    expected_n_splits_mi_only
)


print(
    "Evaluaciones esperadas:",
    expected_total_mi_only
)

print(
    "Evaluaciones obtenidas:",
    len(results_mi_only_df)
)


# =========================
# 8. Resumen MI-only
# =========================

metrics_mi_only = [
    "pr_auc",
    "roc_auc",
    "mcc",
    "balanced_accuracy",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "brier"
]


# Se calcula directamente sobre las 300 evaluaciones,
# de la misma forma que el resumen global de la Tabla 5.

mi_only_summary = (
    results_mi_only_df[metrics_mi_only]
    .agg(["mean", "std"])
    .T
    .reset_index()
)


mi_only_summary.columns = [
    "metric",
    "mean",
    "std"
]


print("\n" + "=" * 100)
print("RESUMEN GLOBAL MI-ONLY")
print("=" * 100)

print(
    mi_only_summary.round(4).to_string(
        index=False
    )
)


# =========================
# 9. Resumen MI-only en formato media ± DE
# =========================

mi_only_summary_pretty = mi_only_summary.copy()

mi_only_summary_pretty["mean_sd"] = (
    mi_only_summary_pretty["mean"].round(3).astype(str)
    + " ± " +
    mi_only_summary_pretty["std"].round(3).astype(str)
)


print("\n" + "=" * 100)
print("RESUMEN MI-ONLY: MEDIA ± DE")
print("=" * 100)

print(
    mi_only_summary_pretty[
        ["metric", "mean_sd"]
    ].to_string(index=False)
)


# =========================
# 10. Resumen por modelo
# =========================

mi_only_model_summary = (
    results_mi_only_df
    .groupby("model")[metrics_mi_only]
    .agg(["mean", "std"])
)


mi_only_model_summary.columns = [
    f"{metric}_{stat}"
    for metric, stat
    in mi_only_model_summary.columns
]


mi_only_model_summary = (
    mi_only_model_summary
    .reset_index()
    .sort_values(
        "pr_auc_mean",
        ascending=False
    )
)


print("\n" + "=" * 100)
print("RESUMEN MI-ONLY POR MODELO")
print("=" * 100)

print(
    mi_only_model_summary.round(4).to_string(
        index=False
    )
)
# ============================================================
# 18.2 MODELO DE REFERENCIA: MISSING INDICATORS ONLY-restringido 
#
# Objetivo:
# Evaluar si la señal predictiva se mantenía al excluir indicadores 
# potencialmente más vinculados a las prácticas de registro y al proceso 
# de evaluación clínica del hospital, y no únicamente a la información 
# clínica disponible.
# ============================================================

# ============================================================
# 1. Matriz con solo variables de imagen y laboratorio
# ============================================================

# Indicadores de ausencia de variables analíticas
mi_laboratorio = [
    "bil_total_missing",
    "bil_directa_missing",
    "alt_missing",
    "ast_missing",
    "fosfatasa_missing",
    "ggt_missing",
    "lipasa_missing",
    "pcr_missing",
    "leucocitos_missing",
    "neutrofilos_missing"
]

# Indicadores de ausencia relacionados con imagen
mi_imagen = [
    "diam_via_biliar_missing",
    "prueba_imagen_missing",
    "colelitiasis_missing",
    "colecistitis_missing"
]

# Conjunto de indicadores para el segundo MI-only
mi_features_restricted = mi_laboratorio + mi_imagen

# Comprobar que todos existen
faltantes = [
    col for col in mi_features_restricted
    if col not in X_mi_only.columns
]

print("Número de indicadores:", len(mi_features_restricted))
print("Indicadores no encontrados:", faltantes)

X_mi_only_restricted = X_mi_only[mi_features_restricted].copy()

print("Dimensiones:", X_mi_only_restricted.shape)
print("Valores faltantes:", X_mi_only_restricted.isna().sum().sum())
print("Distribución de indicadores:")
print(X_mi_only_restricted.sum().sort_values(ascending=False))


# ------------------------------------------------------------
# 2. Verificar que reutilizamos los splits originales
# ------------------------------------------------------------

assert len(outer_splits_mi_only) == 50, (
    "No se han encontrado los 50 splits originales."
)

assert X_mi_only_restricted.shape == (761, 14)

print("Splits externos:", len(outer_splits_mi_only))
print("Indicadores:", len(mi_features_restricted))


# ------------------------------------------------------------
# 3. Preprocesamiento específico
# ------------------------------------------------------------

preprocess_mi_only_restricted = ColumnTransformer(
    transformers=[
        (
            "mi",
            StandardScaler(),
            mi_features_restricted
        )
    ],
    remainder="drop"
)


# ------------------------------------------------------------
# 4. Pipeline específico
# ------------------------------------------------------------

def build_mi_only_restricted_pipeline(model_name):

    model = models[model_name]

    pipe = Pipeline([
        ("preprocess", preprocess_mi_only_restricted),
        ("model", model)
    ])

    return pipe


# ------------------------------------------------------------
# 5. Inicializar resultados independientes
# ------------------------------------------------------------

results_mi_only_restricted = []
best_params_results_mi_only_restricted = []


# ------------------------------------------------------------
# 6. Validación cruzada anidada
# ------------------------------------------------------------

for model_name in selected_models_mi_only:

    print(f"\nEjecutando MI-only restringido: {model_name}")

    for split_id, (train_idx, test_idx) in outer_splits_mi_only:

        start_time = time.time()

        X_train_outer = (
            X_mi_only_restricted.iloc[train_idx].copy()
        )
        X_test_outer = (
            X_mi_only_restricted.iloc[test_idx].copy()
        )

        y_train_outer = y_mi_only.iloc[train_idx].copy()
        y_test_outer = y_mi_only.iloc[test_idx].copy()

        # Pipeline
        pipe = build_mi_only_restricted_pipeline(
            model_name=model_name
        )

        # Espacio de búsqueda original
        n_possible = np.prod([
            len(v)
            for v in param_grids[model_name].values()
        ])

        if model_name == "NAM":
            n_iter_restricted = min(3, n_possible)
            n_jobs_restricted = 4
        else:
            n_iter_restricted = min(10, n_possible)
            n_jobs_restricted = -1

        # Optimización con el mismo esquema de validación interna
        search = RandomizedSearchCV(
            estimator=pipe,
            param_distributions=param_grids[model_name],
            n_iter=n_iter_restricted,
            scoring="average_precision",
            cv=inner_cv_mi_only,
            n_jobs=n_jobs_restricted,
            random_state=42,
            refit=True,
            verbose=0
        )

        # Ajuste
        search.fit(
            X_train_outer,
            y_train_outer
        )

        best_model = search.best_estimator_

        # Predicciones
        y_prob = best_model.predict_proba(
            X_test_outer
        )[:, 1]

        y_pred = best_model.predict(
            X_test_outer
        )

        # Métricas
        metric_values = compute_metrics(
            y_true=y_test_outer,
            y_pred=y_pred,
            y_prob=y_prob
        )

        elapsed = time.time() - start_time

        # Guardar métricas
        row = {
            "model": model_name,
            "analysis": "MI-only-restricted",
            "split_id": split_id,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "n_features": len(mi_features_restricted),
            "elapsed_seconds": round(elapsed, 2)
        }

        row.update(metric_values)

        results_mi_only_restricted.append(row)

        # Guardar mejores parámetros
        best_params_results_mi_only_restricted.append({
            "model": model_name,
            "analysis": "MI-only-restricted",
            "split_id": split_id,
            "best_score_inner_pr_auc": search.best_score_,
            "best_params": search.best_params_
        })

        # Guardado parcial independiente
        pd.DataFrame(
            results_mi_only_restricted
        ).to_csv(
            "nested_cv_results_mi_only_restricted_partial.csv",
            index=False
        )

        pd.DataFrame(
            best_params_results_mi_only_restricted
        ).to_csv(
            "nested_cv_best_params_mi_only_restricted_partial.csv",
            index=False
        )

        print(
            f"Split {split_id:02d} | "
            f"PR-AUC={metric_values['pr_auc']:.3f} | "
            f"ROC-AUC={metric_values['roc_auc']:.3f} | "
            f"MCC={metric_values['mcc']:.3f} | "
            f"{elapsed:.1f}s"
        )


# ------------------------------------------------------------
# 6. Guardado final
# ------------------------------------------------------------

results_mi_only_restricted_df = pd.DataFrame(
    results_mi_only_restricted
)

best_params_mi_only_restricted_df = pd.DataFrame(
    best_params_results_mi_only_restricted
)

results_mi_only_restricted_df.to_csv(
    "nested_cv_results_mi_only_restricted.csv",
    index=False
)

best_params_mi_only_restricted_df.to_csv(
    "nested_cv_best_params_mi_only_restricted.csv",
    index=False
)

print("\nAnálisis MI-only restringido finalizado.")
print("Resultados:", results_mi_only_restricted_df.shape)
print("Parámetros:", best_params_mi_only_restricted_df.shape)


# Cargar resultados originales y restringidos
results_mi_only_df = pd.read_csv("nested_cv_results_mi_only.csv")
results_mi_only_restricted_df = pd.read_csv(
    "nested_cv_results_mi_only_restricted.csv"
)

# Métricas 
metrics_compare = [
    "pr_auc",
    "roc_auc",
    "mcc",
    "balanced_accuracy",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "brier"
]

# Media de cada métrica por modelo
summary_original = (
    results_mi_only_df
    .groupby("model")[metrics_compare]
    .mean()
)

summary_restricted = (
    results_mi_only_restricted_df
    .groupby("model")[metrics_compare]
    .mean()
)

# Comparación global entre ambos análisis
comparison = pd.DataFrame({
    "Original": results_mi_only_df[metrics_compare].mean(),
    "Restringido": results_mi_only_restricted_df[metrics_compare].mean()
})

comparison["Diferencia"] = (
    comparison["Restringido"] - comparison["Original"]
)

display(comparison.round(4))

# Comparación por modelo
print("Resultados por modelo: MI-only original")
display(summary_original.round(4))

print("Resultados por modelo: MI-only restringido")
display(summary_restricted.round(4))

# Métricas de la Tabla 5
metrics = [
    "pr_auc",
    "roc_auc",
    "mcc",
    "balanced_accuracy",
    "accuracy",
    "precision",
    "recall",
    "specificity",
    "brier"
]

# Calcular media y desviación estándar del MI-only restringido
summary_restricted = results_mi_only_restricted_df[metrics].agg(["mean", "std"]).T

# Crear la nueva columna con el formato media ± desviación estándar
summary_restricted["MI-only restringido"] = (
    summary_restricted["mean"].map(lambda x: f"{x:.3f}")
    + r" $\pm$ "
    + summary_restricted["std"].map(lambda x: f"{x:.3f}")
)

# Mostrar los resultados para incorporarlos a la tabla
display(summary_restricted[["MI-only restringido"]])