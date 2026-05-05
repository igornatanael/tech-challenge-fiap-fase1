"""
Definicao dos modelos, pipelines sklearn e funcoes de treinamento.

Modelos implementados:
- Regressao Logistica  (baseline linear, familia estudada na Aula 1)
- Random Forest        (ensemble de arvores,  Aula 4)
- SVM com kernel RBF   (metodo de kernel,     Aula 2)

Cada modelo e encapsulado num sklearn Pipeline que inclui o StandardScaler,
garantindo que:
1. A escala seja sempre ajustada no treino e aplicada no teste sem leakage.
2. O pipeline inteiro possa ser usado diretamente para predicao em producao
   (pipeline.predict(X_novo) ja aplica scaler + modelo automaticamente).
3. O GridSearchCV funcione corretamente sobre todos os hiperparametros.

Uso rapido:
    from src.models.train import build_pipelines, run_cross_validation, run_grid_search

    pipelines = build_pipelines()
    cv_results = run_cross_validation(pipelines, X_train, y_train)
    best_pipelines = run_grid_search(pipelines, X_train, y_train)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

# Semente global de reproducibilidade
RANDOM_STATE = 42

# Numero de folds para validacao cruzada
CV_FOLDS = 5

# Metricas avaliadas na validacao cruzada
CV_SCORING = {
    "f1_macro": "f1_macro",
    "recall_macro": "recall_macro",
    "accuracy": "accuracy",
}


# ---------------------------------------------------------------------------
# Definicao dos pipelines
# ---------------------------------------------------------------------------


def build_pipelines() -> dict[str, Pipeline]:
    """
    Retorna um dicionario com os 3 pipelines prontos para treino.

    Cada pipeline contem:
        StandardScaler -> Modelo

    O scaler e incluido mesmo no Random Forest (que nao precisa de escala)
    para manter a API uniforme e permitir uso direto em producao.

    Hiperparametros iniciais sao razoaveis para um primeiro treino;
    o GridSearch refina os valores criticos de cada modelo.
    """
    pipelines = {
        # --- Baseline linear ---
        # Solver lbfgs + multinomial: melhor para 3 classes com dataset pequeno.
        # C=1.0 e o default do sklearn; GridSearch explorara outros valores.
        # class_weight='balanced' compensa o desbalanceamento moderado das classes.
        # max_iter=1000 evita warning de convergencia com dados padronizados.
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                solver="lbfgs",
                class_weight="balanced",
                max_iter=1000,
                random_state=RANDOM_STATE,
            )),
        ]),

        # --- Ensemble de arvores ---
        # n_estimators=200: numero razoavel de arvores para dataset pequeno.
        # class_weight='balanced_subsample': versao do balanced adaptada ao
        # bootstrap do RF — mais robusta que 'balanced' simples em ensembles.
        # n_jobs=-1: usa todos os nucleos disponiveis.
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestClassifier(
                n_estimators=200,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        # --- Metodo de kernel ---
        # kernel='rbf': captura fronteiras nao-lineares — adequado dado que
        # mid risk se sobrep->e a low risk nos boxplots da EDA.
        # probability=True: necessario para calcular ROC-AUC e predict_proba.
        # class_weight='balanced': compensa desbalanceamento.
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("model", SVC(
                kernel="rbf",
                class_weight="balanced",
                probability=True,
                random_state=RANDOM_STATE,
            )),
        ]),
    }
    return pipelines


# ---------------------------------------------------------------------------
# Validacao cruzada
# ---------------------------------------------------------------------------


def run_cross_validation(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = CV_FOLDS,
    scoring: dict[str, str] = CV_SCORING,
) -> pd.DataFrame:
    """
    Executa validacao cruzada estratificada (k=5) para todos os pipelines.

    Estratificada: preserva a proporcao das 3 classes em cada fold,
    importante dado o desbalanceamento moderado.

    Retorna um DataFrame comparativo com media e desvio padrao de cada metrica.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    rows = []
    for nome, pipeline in pipelines.items():
        print(f"[CV] Avaliando {nome}...")
        resultado = cross_validate(
            pipeline, X_train, y_train,
            cv=cv,
            scoring=scoring,
            return_train_score=False,
            n_jobs=-1,
        )
        row: dict[str, Any] = {"modelo": nome}
        for metrica in scoring:
            scores = resultado[f"test_{metrica}"]
            row[f"{metrica}_media"] = scores.mean().round(4)
            row[f"{metrica}_std"] = scores.std().round(4)
        rows.append(row)

    df_cv = pd.DataFrame(rows).set_index("modelo")
    return df_cv


# ---------------------------------------------------------------------------
# Busca de hiperparametros (GridSearchCV)
# ---------------------------------------------------------------------------

# Grades de hiperparametros para cada modelo.
# Mantidas enxutas dado o tamanho reduzido do dataset (~360 amostras de treino):
# grades grandes com dataset pequeno geram variancia alta nas estimativas de CV.
PARAM_GRIDS: dict[str, dict] = {
    "logistic_regression": {
        # C: inverso da regularizacao L2 (maior C = menos regularizacao)
        "model__C": [0.01, 0.1, 1.0, 10.0, 100.0],
    },
    "random_forest": {
        # n_estimators: mais arvores = mais estavel, mas diminishing returns
        "model__n_estimators": [100, 200, 300],
        # max_depth: None deixa crescer ate pureza; valores menores regularizam
        "model__max_depth": [None, 10, 20],
        # min_samples_leaf: folhas maiores = menos overfitting
        "model__min_samples_leaf": [1, 2, 4],
    },
    "svm": {
        # C: penalidade por violacao da margem (maior = margem mais estreita)
        "model__C": [0.1, 1.0, 10.0, 100.0],
        # gamma: largura do kernel RBF ('scale' = 1/(n_features * var(X)))
        "model__gamma": ["scale", "auto", 0.01, 0.1],
    },
}


def run_grid_search(
    pipelines: dict[str, Pipeline],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = CV_FOLDS,
    scoring: str = "f1_macro",
    param_grids: dict[str, dict] = PARAM_GRIDS,
) -> dict[str, GridSearchCV]:
    """
    Executa GridSearchCV para cada pipeline usando F1-macro como criterio.

    Retorna um dicionario com os objetos GridSearchCV ja ajustados,
    que contem o melhor pipeline em `.best_estimator_`.

    Parametros
    ----------
    scoring : str
        Metrica usada para selecionar o melhor conjunto de hiperparametros.
        Default: 'f1_macro' — consistente com a metrica principal do projeto.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)

    grid_searches: dict[str, GridSearchCV] = {}
    for nome, pipeline in pipelines.items():
        grid = param_grids.get(nome, {})
        if not grid:
            print(f"[GridSearch] {nome}: sem grade definida, pulando.")
            continue

        n_combinacoes = 1
        for valores in grid.values():
            n_combinacoes *= len(valores)
        print(f"[GridSearch] {nome}: {n_combinacoes} combinacoes x {n_splits} folds...")

        gs = GridSearchCV(
            estimator=pipeline,
            param_grid=grid,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            refit=True,    # re-treina com os melhores params no X_train completo
            verbose=0,
        )
        gs.fit(X_train, y_train)

        print(f"  Melhores params : {gs.best_params_}")
        print(f"  Melhor F1-macro : {gs.best_score_:.4f}")
        grid_searches[nome] = gs

    return grid_searches


# ---------------------------------------------------------------------------
# Sumario dos melhores hiperparametros
# ---------------------------------------------------------------------------


def summarize_best_params(grid_searches: dict[str, GridSearchCV]) -> pd.DataFrame:
    """
    Retorna um DataFrame com os melhores hiperparametros e score de cada modelo.
    """
    rows = []
    for nome, gs in grid_searches.items():
        row = {"modelo": nome, "melhor_f1_macro_cv": round(gs.best_score_, 4)}
        row.update(gs.best_params_)
        rows.append(row)
    return pd.DataFrame(rows).set_index("modelo")


# ---------------------------------------------------------------------------
# Execucao direta para verificacao de imports
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pipelines = build_pipelines()
    for nome, pipe in pipelines.items():
        print(f"{nome}: {[step[0] for step in pipe.steps]}")
