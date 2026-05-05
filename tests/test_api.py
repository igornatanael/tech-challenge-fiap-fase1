"""
Testes da API de Classificacao de Risco Gestacional.

Execucao:
    pytest tests/test_api.py -v

Pre-requisito: nenhum (o cliente de teste do Flask nao precisa da API no ar).

Os testes cobrem:
- Endpoint /health
- Predicoes com perfis de pacientes representativos das 3 classes
- Validacao de campos invalidos e ausentes
- Resposta a payloads malformados
"""

from __future__ import annotations

import pytest

# Importa app e forca o treinamento antes dos testes
from src.api.app import app, _modelo_pronto


@pytest.fixture(scope="module")
def client():
    """Cliente de teste Flask — nao precisa da API no ar."""
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


# ---------------------------------------------------------------------------
# Perfis de pacientes para os testes
# Baseados nos padroes identificados na EDA e nas metricas do modelo
# ---------------------------------------------------------------------------

PERFIS = {
    # Paciente jovem, sinais vitais normais — esperado: low risk
    "baixo_risco_tipico": {
        "Age": 22,
        "SystolicBP": 110,
        "DiastolicBP": 70,
        "BS": 7.0,
        "BodyTemp": 36.6,
        "HeartRate": 72,
    },
    # Paciente jovem com glicemia levemente elevada — provavelmente low/mid
    "baixo_risco_glicemia_leve": {
        "Age": 25,
        "SystolicBP": 115,
        "DiastolicBP": 75,
        "BS": 8.0,
        "BodyTemp": 36.8,
        "HeartRate": 76,
    },
    # Paciente com hipertensao e hiperglicemia — esperado: high risk
    "alto_risco_tipico": {
        "Age": 40,
        "SystolicBP": 140,
        "DiastolicBP": 100,
        "BS": 15.0,
        "BodyTemp": 37.0,
        "HeartRate": 80,
    },
    # Paciente mais velha com pressao e glicemia muito elevadas — esperado: high risk
    "alto_risco_grave": {
        "Age": 48,
        "SystolicBP": 160,
        "DiastolicBP": 110,
        "BS": 19.0,
        "BodyTemp": 37.5,
        "HeartRate": 90,
    },
    # Perfil classico de high risk do dataset (medianas da classe)
    "alto_risco_mediana": {
        "Age": 35,
        "SystolicBP": 130,
        "DiastolicBP": 90,
        "BS": 11.0,
        "BodyTemp": 37.0,
        "HeartRate": 77,
    },
    # Paciente adolescente, sinais normais — esperado: low risk
    "baixo_risco_adolescente": {
        "Age": 17,
        "SystolicBP": 100,
        "DiastolicBP": 65,
        "BS": 6.5,
        "BodyTemp": 36.5,
        "HeartRate": 80,
    },
    # Perfil intermediario — resultado incerto (mid risk e dificil de classificar)
    "perfil_intermediario": {
        "Age": 30,
        "SystolicBP": 120,
        "DiastolicBP": 80,
        "BS": 9.0,
        "BodyTemp": 36.8,
        "HeartRate": 74,
    },
    # Valores no limite das faixas validas — testa robustez
    "limite_inferior": {
        "Age": 10,
        "SystolicBP": 70,
        "DiastolicBP": 40,
        "BS": 3.0,
        "BodyTemp": 30.0,
        "HeartRate": 30,
    },
    "limite_superior": {
        "Age": 54,
        "SystolicBP": 180,
        "DiastolicBP": 120,
        "BS": 24.0,
        "BodyTemp": 42.0,
        "HeartRate": 190,
    },
}


# ---------------------------------------------------------------------------
# Testes de /health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_retorna_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_modelo_pronto(self, client):
        dados = resp = client.get("/health").get_json()
        assert dados["status"] == "ok"

    def test_health_contem_features(self, client):
        dados = client.get("/health").get_json()
        assert "features" in dados
        assert len(dados["features"]) > 0

    def test_health_contem_classes(self, client):
        dados = client.get("/health").get_json()
        assert "classes" in dados
        classes = dados["classes"]
        assert "low risk" in classes
        assert "mid risk" in classes
        assert "high risk" in classes


# ---------------------------------------------------------------------------
# Testes de /predict — estrutura da resposta
# ---------------------------------------------------------------------------


class TestPredictEstrutura:
    def test_retorna_200(self, client):
        resp = client.post("/predict", json=PERFIS["baixo_risco_tipico"])
        assert resp.status_code == 200

    def test_resposta_contem_risco(self, client):
        resp = client.post("/predict", json=PERFIS["baixo_risco_tipico"]).get_json()
        assert "risco" in resp

    def test_resposta_contem_nivel(self, client):
        resp = client.post("/predict", json=PERFIS["baixo_risco_tipico"]).get_json()
        assert "nivel" in resp
        assert resp["nivel"] in (0, 1, 2)

    def test_resposta_contem_probabilidades(self, client):
        resp = client.post("/predict", json=PERFIS["baixo_risco_tipico"]).get_json()
        assert "probabilidades" in resp
        probas = resp["probabilidades"]
        assert "low risk" in probas
        assert "mid risk" in probas
        assert "high risk" in probas

    def test_probabilidades_somam_1(self, client):
        resp = client.post("/predict", json=PERFIS["baixo_risco_tipico"]).get_json()
        total = sum(resp["probabilidades"].values())
        assert abs(total - 1.0) < 0.01

    def test_risco_consistente_com_nivel(self, client):
        mapa = {0: "low risk", 1: "mid risk", 2: "high risk"}
        resp = client.post("/predict", json=PERFIS["alto_risco_tipico"]).get_json()
        assert resp["risco"] == mapa[resp["nivel"]]


# ---------------------------------------------------------------------------
# Testes de /predict — perfis clinicos
# ---------------------------------------------------------------------------


class TestPredictPerfis:
    def test_alto_risco_tipico(self, client):
        """Paciente com hipertensao grave e hiperglicemia deve ser high risk."""
        resp = client.post("/predict", json=PERFIS["alto_risco_tipico"]).get_json()
        assert resp["risco"] == "high risk", (
            f"Esperado high risk, obtido {resp['risco']}. "
            f"Probabilidades: {resp['probabilidades']}"
        )

    def test_alto_risco_grave(self, client):
        """Paciente com sinais muito elevados deve ser high risk."""
        resp = client.post("/predict", json=PERFIS["alto_risco_grave"]).get_json()
        assert resp["risco"] == "high risk"

    def test_alto_risco_mediana(self, client):
        """Perfil com medianas da classe high risk deve ser classificado corretamente."""
        resp = client.post("/predict", json=PERFIS["alto_risco_mediana"]).get_json()
        assert resp["risco"] == "high risk"

    def test_baixo_risco_tipico(self, client):
        """Paciente jovem com sinais normais deve ser low risk."""
        resp = client.post("/predict", json=PERFIS["baixo_risco_tipico"]).get_json()
        assert resp["risco"] == "low risk", (
            f"Esperado low risk, obtido {resp['risco']}. "
            f"Probabilidades: {resp['probabilidades']}"
        )

    def test_baixo_risco_adolescente(self, client):
        """Paciente adolescente com sinais saudaveis deve ser low risk."""
        resp = client.post("/predict", json=PERFIS["baixo_risco_adolescente"]).get_json()
        assert resp["risco"] == "low risk"

    def test_alta_probabilidade_high_risk_para_perfil_grave(self, client):
        """Perfil grave deve ter probabilidade > 0.7 para high risk."""
        resp = client.post("/predict", json=PERFIS["alto_risco_grave"]).get_json()
        prob_high = resp["probabilidades"]["high risk"]
        assert prob_high > 0.70, f"Probabilidade high risk muito baixa: {prob_high}"

    def test_todos_perfis_retornam_200(self, client):
        """Todos os perfis validos devem retornar 200."""
        for nome, perfil in PERFIS.items():
            resp = client.post("/predict", json=perfil)
            assert resp.status_code == 200, (
                f"Perfil '{nome}' retornou status {resp.status_code}"
            )

    def test_todos_perfis_imprime_resumo(self, client, capsys):
        """Imprime tabela com predicoes de todos os perfis (util para inspecao manual)."""
        print("\n--- Predicoes por perfil ---")
        print(f"{'Perfil':<35} {'Risco':<12} {'P(low)':<8} {'P(mid)':<8} {'P(high)':<8}")
        print("-" * 75)
        for nome, perfil in PERFIS.items():
            resp = client.post("/predict", json=perfil).get_json()
            p = resp["probabilidades"]
            print(
                f"{nome:<35} {resp['risco']:<12} "
                f"{p['low risk']:<8.3f} {p['mid risk']:<8.3f} {p['high risk']:<8.3f}"
            )


# ---------------------------------------------------------------------------
# Testes de validacao de entrada
# ---------------------------------------------------------------------------


class TestValidacao:
    def test_campo_ausente_retorna_422(self, client):
        perfil_incompleto = {k: v for k, v in PERFIS["baixo_risco_tipico"].items() if k != "BS"}
        resp = client.post("/predict", json=perfil_incompleto)
        assert resp.status_code == 422

    def test_campo_ausente_retorna_detalhes(self, client):
        perfil_incompleto = {k: v for k, v in PERFIS["baixo_risco_tipico"].items() if k != "HeartRate"}
        dados = client.post("/predict", json=perfil_incompleto).get_json()
        assert "detalhes" in dados
        assert any("HeartRate" in d for d in dados["detalhes"])

    def test_valor_nao_numerico_retorna_422(self, client):
        perfil = {**PERFIS["baixo_risco_tipico"], "Age": "trinta"}
        resp = client.post("/predict", json=perfil)
        assert resp.status_code == 422

    def test_body_vazio_retorna_400(self, client):
        resp = client.post("/predict", data="", content_type="application/json")
        assert resp.status_code == 400

    def test_aceita_valores_como_string(self, client):
        """Campos numericos enviados como string devem ser aceitos."""
        perfil_str = {k: str(v) for k, v in PERFIS["baixo_risco_tipico"].items()}
        resp = client.post("/predict", json=perfil_str)
        assert resp.status_code == 200

    def test_aceita_form_data(self, client):
        """A API deve aceitar form-data alem de JSON."""
        resp = client.post(
            "/predict",
            data=PERFIS["baixo_risco_tipico"],
            content_type="application/x-www-form-urlencoded",
        )
        assert resp.status_code == 200
