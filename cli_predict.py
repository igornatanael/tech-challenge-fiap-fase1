"""
CLI interativa — Classificacao de Risco Gestacional

Apresenta um formulario em linha de comando, coleta os sinais vitais da
paciente e consulta a API Flask para obter a predicao de risco.

Uso:
    python cli_predict.py
    python cli_predict.py --url http://localhost:5000   # URL customizada

Pre-requisito: API rodando (python -m src.api.app)
"""

from __future__ import annotations

import argparse
import sys

import requests

DEFAULT_URL = "http://localhost:5000"

# Cores ANSI para terminal
VERDE = "\033[92m"
AMARELO = "\033[93m"
VERMELHO = "\033[91m"
NEGRITO = "\033[1m"
RESET = "\033[0m"
CIANO = "\033[96m"

CORES_RISCO = {
    "low risk": VERDE,
    "mid risk": AMARELO,
    "high risk": VERMELHO,
}

LABELS_RISCO = {
    "low risk": "BAIXO RISCO",
    "mid risk": "MEDIO RISCO",
    "high risk": "ALTO RISCO",
}

CAMPOS = [
    {
        "chave": "Age",
        "label": "Idade da paciente",
        "unidade": "anos",
        "tipo": int,
        "min": 10,
        "max": 60,
        "exemplo": "ex: 28",
    },
    {
        "chave": "SystolicBP",
        "label": "Pressao arterial sistolica",
        "unidade": "mmHg",
        "tipo": int,
        "min": 70,
        "max": 200,
        "exemplo": "ex: 120",
    },
    {
        "chave": "DiastolicBP",
        "label": "Pressao arterial diastolica",
        "unidade": "mmHg",
        "tipo": int,
        "min": 40,
        "max": 130,
        "exemplo": "ex: 80",
    },
    {
        "chave": "BS",
        "label": "Glicemia (Blood Sugar)",
        "unidade": "mmol/L",
        "tipo": float,
        "min": 3.0,
        "max": 25.0,
        "exemplo": "ex: 7.5",
    },
    {
        "chave": "BodyTemp",
        "label": "Temperatura corporal",
        "unidade": "graus Celsius",
        "tipo": float,
        "min": 30.0,
        "max": 45.0,
        "exemplo": "ex: 36.6",
    },
    {
        "chave": "HeartRate",
        "label": "Frequencia cardiaca",
        "unidade": "bpm",
        "tipo": int,
        "min": 30,
        "max": 200,
        "exemplo": "ex: 72",
    },
]


def _coletar_campo(campo: dict) -> float | int:
    """Solicita e valida um campo do formulario com retentativas."""
    while True:
        prompt = (
            f"  {NEGRITO}{campo['label']}{RESET} "
            f"({campo['unidade']}, {campo['exemplo']}): "
        )
        entrada = input(prompt).strip()

        if not entrada:
            print(f"  {AMARELO}Campo obrigatorio. Tente novamente.{RESET}")
            continue

        try:
            valor = campo["tipo"](entrada)
        except ValueError:
            print(f"  {AMARELO}Valor invalido. Digite um numero.{RESET}")
            continue

        if not (campo["min"] <= valor <= campo["max"]):
            print(
                f"  {AMARELO}Valor fora da faixa esperada "
                f"({campo['min']} – {campo['max']} {campo['unidade']}). "
                f"Confirma? (s/n): {RESET}",
                end="",
            )
            confirmacao = input().strip().lower()
            if confirmacao != "s":
                continue

        return valor


def _verificar_api(url: str) -> bool:
    """Verifica se a API esta no ar antes de iniciar o formulario."""
    try:
        resp = requests.get(f"{url}/health", timeout=5)
        if resp.status_code == 200:
            return True
        print(f"{VERMELHO}API retornou status {resp.status_code}.{RESET}")
        return False
    except requests.exceptions.ConnectionError:
        print(
            f"{VERMELHO}Nao foi possivel conectar a API em {url}.\n"
            f"Certifique-se de que o servidor esta rodando:{RESET}\n"
            f"  python -m src.api.app"
        )
        return False


def _exibir_resultado(resultado: dict) -> None:
    """Exibe o resultado da predicao formatado."""
    risco = resultado["risco"]
    cor = CORES_RISCO.get(risco, RESET)
    label = LABELS_RISCO.get(risco, risco.upper())

    print()
    print("=" * 50)
    print(f"  RESULTADO DA TRIAGEM")
    print("=" * 50)
    print(f"  Nivel de risco: {cor}{NEGRITO}{label}{RESET}")
    print()
    print(f"  {CIANO}Probabilidades:{RESET}")
    for classe, prob in resultado["probabilidades"].items():
        barra = "█" * int(prob * 30)
        print(f"    {classe:<12} {prob*100:5.1f}%  {barra}")
    print("=" * 50)

    if risco == "high risk":
        print(
            f"\n  {VERMELHO}{NEGRITO}ATENCAO:{RESET} Paciente com indicadores de alto risco.\n"
            f"  Encaminhar para avaliacao medica imediata."
        )
    elif risco == "mid risk":
        print(
            f"\n  {AMARELO}Paciente com indicadores de medio risco.\n"
            f"  Recomenda-se acompanhamento clinico proximo.{RESET}"
        )
    else:
        print(
            f"\n  {VERDE}Paciente com indicadores de baixo risco.\n"
            f"  Manter acompanhamento pre-natal de rotina.{RESET}"
        )
    print()


def main(url: str = DEFAULT_URL) -> None:
    print()
    print(f"{NEGRITO}{'=' * 50}{RESET}")
    print(f"{NEGRITO}  TRIAGEM DE RISCO GESTACIONAL{RESET}")
    print(f"{NEGRITO}{'=' * 50}{RESET}")
    print(f"  Sistema de suporte a decisao clinica")
    print(f"  API: {url}")
    print(f"{NEGRITO}{'=' * 50}{RESET}")
    print()

    if not _verificar_api(url):
        sys.exit(1)

    print(f"  {CIANO}Preencha os dados clinicos da paciente:{RESET}\n")

    dados: dict[str, float | int] = {}
    for campo in CAMPOS:
        dados[campo["chave"]] = _coletar_campo(campo)

    print(f"\n  {CIANO}Consultando o modelo...{RESET}")

    try:
        resp = requests.post(f"{url}/predict", json=dados, timeout=10)
        if resp.status_code == 200:
            _exibir_resultado(resp.json())
        else:
            erro = resp.json()
            print(f"\n{VERMELHO}Erro na predicao:{RESET}")
            print(f"  {erro.get('erro', 'Erro desconhecido')}")
            if "detalhes" in erro:
                for d in erro["detalhes"]:
                    print(f"  - {d}")
            sys.exit(1)

    except requests.exceptions.RequestException as e:
        print(f"\n{VERMELHO}Falha na comunicacao com a API: {e}{RESET}")
        sys.exit(1)

    print(
        f"  {CIANO}Este sistema e uma ferramenta de triagem.\n"
        f"  O profissional de saude tem sempre a palavra final.{RESET}\n"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI de triagem de risco gestacional"
    )
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"URL base da API (padrao: {DEFAULT_URL})",
    )
    args = parser.parse_args()
    main(url=args.url)
