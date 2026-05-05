"""
Pipeline de carregamento do dataset Maternal Health Risk.

Tenta carregar o dataset em ordem:
1. CSV local em data/raw/maternal_health_risk.csv
2. Pacote ucimlrepo (fetch direto da UCI, requer rede)
3. Falha com instrucoes claras
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

# Caminho padrao do dataset bruto (resolvido a partir da raiz do projeto)
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RAW_PATH = PROJECT_ROOT / "data" / "raw" / "maternal_health_risk.csv"

# Esquema esperado do dataset
EXPECTED_COLUMNS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
    "RiskLevel",
]

TARGET_COLUMN = "RiskLevel"


def load_from_csv(path: Path = DEFAULT_RAW_PATH) -> pd.DataFrame:
    """Carrega o dataset a partir de um CSV local."""
    if not path.exists():
        raise FileNotFoundError(
            f"Arquivo nao encontrado em {path}. "
            "Baixe o dataset Maternal Health Risk da UCI ou Kaggle "
            "e coloque-o em data/raw/maternal_health_risk.csv"
        )
    # encoding utf-8-sig remove BOM se houver
    df = pd.read_csv(path, encoding='utf-8-sig')
    return df


def load_from_uci() -> pd.DataFrame:
    """Carrega o dataset diretamente da UCI via pacote ucimlrepo."""
    try:
        from ucimlrepo import fetch_ucirepo  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "Pacote 'ucimlrepo' nao instalado. "
            "Rode: pip install ucimlrepo"
        ) from exc

    dataset = fetch_ucirepo(id=863)
    features = dataset.data.features
    target = dataset.data.targets

    df = pd.concat([features, target], axis=1)
    return df


def load_dataset(path: Optional[Path] = None) -> pd.DataFrame:
    """
    Funcao principal de carregamento.
    Tenta CSV local primeiro, cai para UCI se falhar.
    """
    csv_path = path or DEFAULT_RAW_PATH

    if csv_path.exists():
        df = load_from_csv(csv_path)
    else:
        try:
            df = load_from_uci()
            # Cacheia localmente para proximas execucoes
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(csv_path, index=False)
        except Exception as exc:
            raise RuntimeError(
                f"Nao foi possivel carregar o dataset. "
                f"CSV nao existe em {csv_path} e fallback UCI falhou: {exc}"
            ) from exc

    validate_schema(df)
    return df


def validate_schema(df: pd.DataFrame) -> None:
    """Valida que o dataframe tem as colunas esperadas."""
    missing = set(EXPECTED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes no dataset: {missing}. "
            f"Esperadas: {EXPECTED_COLUMNS}"
        )


if __name__ == "__main__":
    df = load_dataset()
    print(f"Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")
    print(df.head())
    print("\nDistribuicao do alvo:")
    print(df[TARGET_COLUMN].value_counts())
