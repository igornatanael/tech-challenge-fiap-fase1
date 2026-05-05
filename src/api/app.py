"""
API Flask — Sistema de Classificacao de Risco Gestacional

Ao iniciar, o servidor treina o modelo Random Forest com os melhores
hiperparametros encontrados no GridSearch (notebook 02) usando o pipeline
completo de preprocessamento. O modelo fica em memoria e responde predicoes
sem necessidade de banco de dados ou armazenamento externo.

Endpoints:
    GET  /health      — status do servidor e do modelo
    POST /predict     — predicao de risco a partir dos sinais vitais

Uso:
    python -m src.api.app
    # ou
    flask --app src.api.app run
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from flask import Flask, jsonify, request
from sklearn.ensemble import RandomForestClassifier

from src.pipelines.preprocessing import (
    RISK_DECODING,
    build_preprocessed_splits,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Estado global — modelo treinado em memoria ao iniciar o servidor
# ---------------------------------------------------------------------------

_modelo: RandomForestClassifier | None = None
_scaler = None
_feature_columns: list[str] = []
_modelo_pronto: bool = False
_erro_inicializacao: str = ""

MELHORES_PARAMS = {
    "n_estimators": 200,
    "max_depth": 20,
    "min_samples_leaf": 1,
    "class_weight": "balanced_subsample",
    "random_state": 42,
    "n_jobs": -1,
}


def _treinar_modelo() -> None:
    """Executa o pipeline completo e treina o Random Forest em memoria."""
    global _modelo, _scaler, _feature_columns, _modelo_pronto, _erro_inicializacao

    print("[api] Iniciando treinamento do modelo em memoria...")
    try:
        X_train, X_test, y_train, y_test, scaler = build_preprocessed_splits(
            add_pulse_pressure=True,
            add_age_features=True,
        )

        _scaler = scaler
        _feature_columns = X_train.columns.tolist()

        modelo = RandomForestClassifier(**MELHORES_PARAMS)
        modelo.fit(X_train, y_train)

        _modelo = modelo
        _modelo_pronto = True
        print(f"[api] Modelo pronto. Features: {_feature_columns}")
        print(f"[api] Acuracia no treino: {modelo.score(X_train, y_train):.3f}")

    except Exception as e:
        _erro_inicializacao = str(e)
        print(f"[api] ERRO ao treinar modelo: {e}")


def _preparar_entrada(dados: dict) -> pd.DataFrame:
    """
    Converte os dados do formulario em DataFrame pronto para predicao.

    Aplica as mesmas transformacoes do pipeline de preprocessamento:
    - Converte BodyTemp de Celsius para a escala esperada pelo scaler
      (o scaler foi fitado em dados ja convertidos para Celsius)
    - Cria features derivadas: pulse_pressure, age_advanced, age_group
    - Aplica StandardScaler
    - Garante a ordem correta das colunas
    """
    age = float(dados["Age"])
    sbp = float(dados["SystolicBP"])
    dbp = float(dados["DiastolicBP"])
    bs = float(dados["BS"])
    body_temp = float(dados["BodyTemp"])   # esperado em Celsius
    hr = float(dados["HeartRate"])

    row = {
        "Age": age,
        "SystolicBP": sbp,
        "DiastolicBP": dbp,
        "BS": bs,
        "BodyTemp": body_temp,
        "HeartRate": hr,
        "pulse_pressure": sbp - dbp,
        "age_advanced": int(age >= 35),
        "age_group": int(pd.cut([age], bins=[0, 20, 35, np.inf], labels=[0, 1, 2], right=False)[0]),
    }

    df = pd.DataFrame([row])[_feature_columns]
    df_scaled = pd.DataFrame(
        _scaler.transform(df),
        columns=_feature_columns,
    )
    return df_scaled


def _validar_entrada(dados: dict) -> list[str]:
    """Retorna lista de erros de validacao (vazia se tudo ok)."""
    campos_obrigatorios = ["Age", "SystolicBP", "DiastolicBP", "BS", "BodyTemp", "HeartRate"]
    erros = []

    for campo in campos_obrigatorios:
        if campo not in dados:
            erros.append(f"Campo obrigatorio ausente: '{campo}'")
            continue
        try:
            float(dados[campo])
        except (ValueError, TypeError):
            erros.append(f"'{campo}' deve ser um numero. Recebido: {dados[campo]!r}")

    if erros:
        return erros

    faixas = {
        "Age": (10, 60, "anos"),
        "SystolicBP": (70, 200, "mmHg"),
        "DiastolicBP": (40, 130, "mmHg"),
        "BS": (3.0, 25.0, "mmol/L"),
        "BodyTemp": (30.0, 45.0, "graus Celsius"),
        "HeartRate": (30, 200, "bpm"),
    }
    for campo, (minimo, maximo, unidade) in faixas.items():
        valor = float(dados[campo])
        if not (minimo <= valor <= maximo):
            erros.append(
                f"'{campo}' fora da faixa plausivel: {valor} "
                f"(esperado entre {minimo} e {maximo} {unidade})"
            )

    return erros


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.route("/health", methods=["GET"])
def health():
    """Retorna o status do servidor e se o modelo esta pronto."""
    if _modelo_pronto:
        return jsonify({
            "status": "ok",
            "modelo": "Random Forest",
            "features": _feature_columns,
            "n_classes": 3,
            "classes": list(RISK_DECODING.values()),
        })
    return jsonify({
        "status": "erro",
        "mensagem": _erro_inicializacao or "Modelo nao inicializado.",
    }), 503


@app.route("/predict", methods=["POST"])
def predict():
    """
    Recebe os sinais vitais da paciente e retorna o nivel de risco gestacional.

    Body (JSON ou form-data):
        Age         : idade em anos (inteiro)
        SystolicBP  : pressao arterial sistolica em mmHg
        DiastolicBP : pressao arterial diastolica em mmHg
        BS          : glicemia em mmol/L
        BodyTemp    : temperatura corporal em graus Celsius
        HeartRate   : frequencia cardiaca em bpm

    Retorna:
        {
            "risco": "high risk",
            "nivel": 2,
            "probabilidades": {
                "low risk": 0.05,
                "mid risk": 0.12,
                "high risk": 0.83
            }
        }
    """
    if not _modelo_pronto:
        return jsonify({"erro": "Modelo nao esta pronto. Tente novamente em instantes."}), 503

    # Aceita JSON ou form-data
    dados = request.get_json(silent=True) or request.form.to_dict()
    if not dados:
        return jsonify({"erro": "Nenhum dado recebido. Envie JSON ou form-data."}), 400

    erros = _validar_entrada(dados)
    if erros:
        return jsonify({"erro": "Dados invalidos.", "detalhes": erros}), 422

    try:
        X = _preparar_entrada(dados)
        nivel_pred = int(_modelo.predict(X)[0])
        probas = _modelo.predict_proba(X)[0]

        return jsonify({
            "risco": RISK_DECODING[nivel_pred],
            "nivel": nivel_pred,
            "probabilidades": {
                RISK_DECODING[i]: round(float(p), 4)
                for i, p in enumerate(probas)
            },
        })

    except Exception as e:
        return jsonify({"erro": f"Erro interno ao processar predicao: {e}"}), 500


# ---------------------------------------------------------------------------
# Inicializacao
# ---------------------------------------------------------------------------

# Treina o modelo ao importar o modulo (funciona com flask run e python -m)
_treinar_modelo()


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
