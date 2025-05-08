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
from sklearn.metrics import mean_squared_error, mean_absolute_error, make_scorer

n_trials = 50000

# --------------------------
# Funções auxiliares
# --------------------------
def compute_nse(y_true, y_pred):
    return 1 - np.sum((y_true - y_pred) ** 2) / np.sum((y_true - np.mean(y_true)) ** 2)

def compute_mape(y_true, y_pred):
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100

# --------------------------
# Carregamento e preparação dos dados
# --------------------------
df = pd.read_excel("DataFrame_unificado_one_hot.xlsx")
X = df.drop("Resposta(S/cm)", axis=1)
y = df["Resposta(S/cm)"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------
# Treinamento dos modelos base
# --------------------------
base_models = {
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "MLP Regressor": MLPRegressor(random_state=42)
}

base_metrics = {}
for name, model in base_models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    base_metrics[name] = {
        "RMSE": math.sqrt(mean_squared_error(y_test, y_pred)),
        "MAE": mean_absolute_error(y_test, y_pred),
        "MAPE": compute_mape(y_test, y_pred),
        "NSE": compute_nse(y_test, y_pred)
    }

with open("metricas_modelos_base_mlp.json", "w") as f:
    json.dump(base_metrics, f, indent=4)

# --------------------------
# Definindo o scorer NSE para Optuna
# --------------------------
nse_scorer = make_scorer(compute_nse, greater_is_better=True)

# --------------------------
# Função objetivo para MLPRegressor
# --------------------------
def objective(trial):
    # Arquitetura
    n_layers = trial.suggest_int("n_layers", 1, 5)
    neurons = trial.suggest_int("neurons", 1, 16)
    hidden_layer_sizes = tuple([neurons] * n_layers)

    # Hiperparâmetros de otimização e regularização
    activation = trial.suggest_categorical("activation", ["identity", "logistic", "tanh", "relu"])
    solver = trial.suggest_categorical("solver", ["lbfgs", "sgd", "adam"])
    alpha = trial.suggest_float("alpha", 1e-6, 1e-2, log=True)

    # Batch size variando de 13 até n amostras + opção "auto"
    n_samples = X_train.shape[0]
    batch_values = ["auto"] + list(range(13, n_samples + 1))
    batch_size = trial.suggest_categorical("batch_size", batch_values)

    # Learning rate
    learning_rate = trial.suggest_categorical("learning_rate", ["constant", "invscaling", "adaptive"])
    learning_rate_init = trial.suggest_float("learning_rate_init", 1e-5, 0.1, log=True)

    # Parâmetros específicos para SGD com power_t restrito
    if solver == "sgd" and learning_rate == "invscaling":
        power_t = trial.suggest_float("power_t", 1e-2, 2.0, log=True)
    else:
        power_t = 0.5

    # Iterações e tolerância
    max_iter = 20000
    tol = trial.suggest_float("tol", 1e-6, 1e-2, log=True)

    # Momentum para SGD
    if solver == "sgd":
        momentum = trial.suggest_float("momentum", 0.0, 1.0, step=0.05)
        nesterovs_momentum = trial.suggest_categorical("nesterovs_momentum", [True, False])
    else:
        momentum = 0.9
        nesterovs_momentum = True

    # Early stopping
    early_stopping = trial.suggest_categorical("early_stopping", [False, True])
    validation_fraction = 0.1
    n_iter_no_change = trial.suggest_int("n_iter_no_change", 1, 1000)

    # max_fun para L-BFGS
    if solver == "lbfgs":
        max_fun = trial.suggest_int("max_fun", 150, 1500000)
    else:
        max_fun = 15000

    model = MLPRegressor(
        hidden_layer_sizes=hidden_layer_sizes,
        activation=activation,
        solver=solver,
        alpha=alpha,
        batch_size=batch_size,
        learning_rate=learning_rate,
        learning_rate_init=learning_rate_init,
        power_t=power_t,
        max_iter=max_iter,
        shuffle=True,
        tol=tol,
        verbose=False,
        warm_start=trial.suggest_categorical("warm_start", [False, True]),
        momentum=momentum,
        nesterovs_momentum=nesterovs_momentum,
        early_stopping=early_stopping,
        validation_fraction=validation_fraction,
        n_iter_no_change=n_iter_no_change,
        max_fun=max_fun,
        random_state=42
    )

    # Validação cruzada com tratamento de falhas
    try:
        scores = cross_val_score(
            model, X_train_scaled, y_train,
            scoring=nse_scorer, cv=5, n_jobs=-1,
            error_score=-np.inf
        )
        return np.mean(scores)
    except OverflowError:
        return -np.inf

# --------------------------
# Configuração do estudo Optuna
# --------------------------
study = optuna.create_study(direction="maximize")
study.enqueue_trial({
    "n_layers": 1,
    "neurons": 7,
    "activation": "relu",
    "solver": "adam",
    "alpha": 1e-4,
    "batch_size": "auto",
    "learning_rate": "constant",
    "learning_rate_init": 1e-3,
    "power_t": 0.5,
    "tol": 1e-4,
    "warm_start": False,
    "momentum": 0.9,
    "nesterovs_momentum": True,
    "early_stopping": False,
    "n_iter_no_change": 10,
    "max_fun": 15000
})
study.optimize(objective, n_trials=n_trials)

# --------------------------
# Salvar estudo e parâmetros ótimos
# --------------------------
joblib.dump(study, "optuna_mlp_study.pkl")
with open("optuna_mlp_best_params.json", "w") as f:
    json.dump(study.best_params, f, indent=4)

# --------------------------
# Treinar e avaliar o melhor modelo
# --------------------------
best_params = study.best_params
hidden_layer_sizes = tuple([best_params["neurons"]] * best_params["n_layers"])

mlp_best = MLPRegressor(
    hidden_layer_sizes=hidden_layer_sizes,
    activation=best_params["activation"],
    solver=best_params["solver"],
    alpha=best_params["alpha"],
    batch_size=best_params["batch_size"],
    learning_rate=best_params["learning_rate"],
    learning_rate_init=best_params["learning_rate_init"],
    power_t=best_params.get("power_t", 0.5),
    max_iter=20000,
    shuffle=True,
    tol=best_params["tol"],
    warm_start=best_params["warm_start"],
    momentum=best_params.get("momentum", 0.9),
    nesterovs_momentum=best_params.get("nesterovs_momentum", True),
    early_stopping=best_params["early_stopping"],
    validation_fraction=0.1,
    n_iter_no_change=best_params["n_iter_no_change"],
    max_fun=best_params.get("max_fun", 15000),
    random_state=42
)
mlp_best.fit(X_train_scaled, y_train)
y_pred = mlp_best.predict(X_test_scaled)

# --------------------------
# Métricas do melhor modelo
# --------------------------
best_metrics = {
    "RMSE": math.sqrt(mean_squared_error(y_test, y_pred)),
    "MAE": mean_absolute_error(y_test, y_pred),
    "MAPE": compute_mape(y_test, y_pred),
    "NSE": compute_nse(y_test, y_pred)
}

with open("mlp_best_model_metrics.json", "w") as f:
    json.dump(best_metrics, f, indent=4)

# --------------------------
# Salvar o melhor modelo MLP
# --------------------------
joblib.dump(mlp_best, "best_mlp_model.joblib")

print("Base metrics, study, best params, and models saved successfully.")

