# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import optuna
import json
import math
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, make_scorer

# Funções auxiliares
def compute_nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def compute_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# Carregamento e preparação dos dados
df = pd.read_excel("DataFrame_unificado_one_hot.xlsx")
X = df.drop("Resposta(S/cm)", axis=1)
y = df["Resposta(S/cm)"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Treinamento dos modelos base
modelos = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "MLP Regressor": MLPRegressor(random_state=42)
}

metricas_gerais = {}
for nome, modelo in modelos.items():
    modelo.fit(X_train_scaled, y_train)
    y_pred = modelo.predict(X_test_scaled)
    metricas_gerais[nome] = {
        "RMSE": math.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": compute_mape(y_test, y_pred),
        "NSE": compute_nse(y_test, y_pred)
    }

with open("metricas_modelos_base.json", "w") as f:
    json.dump(metricas_gerais, f, indent=4)

# Otimização com Optuna
def nse_score(y_true, y_pred):
    return compute_nse(y_true, y_pred)

nse_scorer = make_scorer(nse_score, greater_is_better=True)

def objective(trial):
    n_estimators = trial.suggest_int("n_estimators", 1, 10000)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 200)
    max_features = trial.suggest_float("max_features", 0.01, 1.0, step=0.01)
    min_samples_leaf = trial.suggest_int("min_samples_leaf", 1, 100)
    criterion = trial.suggest_categorical(
        "criterion",
        ["squared_error", "absolute_error", "friedman_mse", "poisson"]
    )
    min_weight_fraction_leaf = trial.suggest_float(
        "min_weight_fraction_leaf", 0.0, 0.1, step=0.01
    )
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    ccp_alpha = trial.suggest_float("ccp_alpha", 0.0, 0.3, step=0.001)
    max_samples = (
        trial.suggest_float("max_samples", 0.1, 1.0, step=0.1)
        if bootstrap else None
    )

    # Espaço contínuo de max_depth: None + [2, 3, …, 70]
    max_depth_values = [None] + list(range(2, 71))
    max_depth = trial.suggest_categorical("max_depth", max_depth_values)

    # Espaço contínuo de max_leaf_nodes: None + [2, 3, …, 300]
    max_leaf_nodes_values = [None] + list(range(2, 301))
    max_leaf_nodes = trial.suggest_categorical("max_leaf_nodes", max_leaf_nodes_values)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        criterion=criterion,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        min_weight_fraction_leaf=min_weight_fraction_leaf,
        max_features=max_features,
        max_leaf_nodes=max_leaf_nodes,
        bootstrap=bootstrap,
        ccp_alpha=ccp_alpha,
        max_samples=max_samples,
        random_state=42,
        n_jobs=-1
    )
    scores = cross_val_score(
        model, X_train, y_train,
        scoring=nse_scorer, cv=5, n_jobs=-1
    )
    return scores.mean()

study = optuna.create_study(direction="maximize")
# Trial inicial com seus valores recomendados
study.enqueue_trial({
    "n_estimators": 31,
    "min_samples_split": 2,
    "max_features": 0.63,
    "min_samples_leaf": 1,
    "criterion": "absolute_error",
    "max_depth": 45,
    "min_weight_fraction_leaf": 0.02,
    "max_leaf_nodes": 10,
    "bootstrap": False,
    "ccp_alpha": 0.0,
    "max_samples": None
})
study.optimize(objective, n_trials=50000)

# Salvar estudo Optuna
joblib.dump(study, "optuna_study.pkl")
with open("optuna_best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)

# Avaliação do melhor modelo
best_params = study.best_params
final_max_samples = best_params["max_samples"] if best_params["bootstrap"] else None
best_model = RandomForestRegressor(
    n_estimators=best_params["n_estimators"],
    criterion=best_params["criterion"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    min_weight_fraction_leaf=best_params["min_weight_fraction_leaf"],
    max_features=best_params["max_features"],
    max_leaf_nodes=best_params["max_leaf_nodes"],
    bootstrap=best_params["bootstrap"],
    ccp_alpha=best_params["ccp_alpha"],
    max_samples=final_max_samples,
    random_state=42,
    n_jobs=-1
)
best_model.fit(X_train, y_train)
y_pred = best_model.predict(X_test)
final_metrics = {
    "RMSE": math.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE": mean_absolute_error(y_test, y_pred),
    "MAPE": compute_mape(y_test, y_pred),
    "NSE": compute_nse(y_test, y_pred)
}

with open("best_model_metrics.json", "w") as f:
    json.dump(final_metrics, f, indent=4)

# Salvar o modelo treinado final
joblib.dump(best_model, "best_random_forest_model.joblib")
