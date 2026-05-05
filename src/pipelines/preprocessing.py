"""
Pipeline de pre-processamento do dataset Maternal Health Risk.

Etapas executadas (nesta ordem):
1. Remocao de outliers fisiologicamente invalidos (HeartRate = 7 bpm)
2. Remocao de linhas duplicadas
3. Criacao de features derivadas opcionais:
   - pulse_pressure = SystolicBP - DiastolicBP
   - age_advanced  = 1 se Age >= 35, caso contrario 0  (flag AMA: Advanced Maternal Age)
   - age_group     = categoria ordinal <20 / 20-34 / >=35 (codificada como 0/1/2)
4. Encoding ordinal do alvo: low risk=0, mid risk=1, high risk=2
5. Split estratificado treino/teste (80/20, random_state fixo)
6. Padronizacao das features numericas com StandardScaler (fit apenas no treino)

Features derivadas opcionais:
- add_pulse_pressure: cria pulse_pressure; util para reduzir colinearidade entre
  SystolicBP e DiastolicBP na Regressao Logistica. Nao remove as originais por
  padrao, pois modelos de arvore se beneficiam delas individualmente.
- add_age_features: cria age_advanced e age_group. Os grupos seguem o limiar
  clinico padrao em obstetricia (Advanced Maternal Age = >= 35 anos).
  Mantidas junto com Age continua, que permanece no dataset. Os modelos de arvore
  ignorarao redundancias; para Regressao Logistica, age_advanced captura o efeito
  de degrau que a relacao linear nao conseguiria representar bem.

Uso rapido:
    from src.pipelines.preprocessing import build_preprocessed_splits

    X_train, X_test, y_train, y_test = build_preprocessed_splits()
    # Com features derivadas:
    X_train, X_test, y_train, y_test = build_preprocessed_splits(
        add_pulse_pressure=True,
        add_age_features=True,
    )
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.pipelines.data_loading import load_dataset, TARGET_COLUMN

# Mapeamento ordinal do alvo — preserva a ordem clinica do risco
RISK_ENCODING = {
    "low risk": 0,
    "mid risk": 1,
    "high risk": 2,
}
RISK_DECODING = {v: k for k, v in RISK_ENCODING.items()}

# Colunas base de features (sem o alvo)
BASE_FEATURE_COLUMNS = [
    "Age",
    "SystolicBP",
    "DiastolicBP",
    "BS",
    "BodyTemp",
    "HeartRate",
]

# Hiperparametros fixos de reproducibilidade
RANDOM_STATE = 42
TEST_SIZE = 0.20


# ---------------------------------------------------------------------------
# Etapas individuais do pipeline
# ---------------------------------------------------------------------------


def remove_invalid_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas com valores fisiologicamente impossíveis.

    Critério atual: HeartRate < 30 bpm (impossível em humano vivo).
    Os 2 registros com HeartRate = 7 identificados na EDA caem aqui.
    """
    antes = len(df)
    df = df[df["HeartRate"] >= 30].copy()
    removidos = antes - len(df)
    if removidos > 0:
        print(f"[outliers] Removidas {removidos} linha(s) com HeartRate < 30 bpm.")
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove linhas completamente duplicadas.

    Deduplicar ANTES do split evita data leakage — sem isso, a mesma
    observacao poderia aparecer em treino e teste, inflando artificialmente
    as metricas.
    """
    antes = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removidos = antes - len(df)
    print(f"[dedup] Removidas {removidos} duplicatas. Linhas restantes: {len(df)}")
    return df


def _add_pulse_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria feature derivada pulse_pressure = SystolicBP - DiastolicBP.

    Reduz a colinearidade entre as duas pressoes (r=0.79 na EDA) sem descartar
    informacao. Util especialmente para a Regressao Logistica.

    Prefixo _ indica funcao auxiliar interna — use build_preprocessed_splits
    com add_pulse_pressure=True para ativa-la.
    """
    df = df.copy()
    df["pulse_pressure"] = df["SystolicBP"] - df["DiastolicBP"]
    print("[features] pulse_pressure criado.")
    return df


def _add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cria duas features derivadas de idade baseadas em limiares clinicos:

    - age_advanced (binario): 1 se Age >= 35 (Advanced Maternal Age — AMA),
      0 caso contrario. Captura o efeito de degrau documentado na literatura
      obstetrica a partir dos 35 anos.

    - age_group (ordinal): 0 = adolescente (<20), 1 = adulta (20-34),
      2 = idade materna avancada (>=35). Codificada como inteiro ordinal
      — preserva a ordem clinica de risco crescente.

    Ambas sao mantidas junto com Age continua. Modelos de arvore ignoram
    redundancias; a Regressao Logistica se beneficia de age_advanced para
    capturar a nao-linearidade.

    Prefixo _ indica funcao auxiliar interna — use build_preprocessed_splits
    com add_age_features=True para ativa-la.
    """
    df = df.copy()

    # Binario AMA
    df["age_advanced"] = (df["Age"] >= 35).astype(int)

    # Grupo ordinal com limiares clinicos standard em obstetricia
    bins = [0, 20, 35, np.inf]
    labels = [0, 1, 2]           # adolescente / adulta / AMA
    df["age_group"] = pd.cut(
        df["Age"],
        bins=bins,
        labels=labels,
        right=False,             # intervalo fechado a esquerda: [0,20), [20,35), [35,inf)
    ).astype(int)

    print("[features] age_advanced e age_group criados.")
    return df


def _convert_body_temp_to_celsius(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte BodyTemp de Fahrenheit para Celsius: (F - 32) * 5/9.

    O dataset original registra a temperatura em Fahrenheit. Celsius e mais
    intuitivo no contexto clinico brasileiro e facilita interpretacao dos
    valores SHAP e feature importance.

    A coluna BodyTemp e substituida in-place (sem criar coluna nova),
    pois nao ha razao para manter os dois.
    """
    df = df.copy()
    df["BodyTemp"] = (df["BodyTemp"] - 32) * 5 / 9
    print("[features] BodyTemp convertida de Fahrenheit para Celsius.")
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Codifica a coluna alvo como inteiro ordinal: low=0, mid=1, high=2.

    Preserva a ordem clinica do risco, o que e coerente com o problema
    (erro entre low e high e mais grave que entre low e mid).
    """
    df = df.copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].map(RISK_ENCODING)
    if df[TARGET_COLUMN].isna().any():
        valores_desconhecidos = df[df[TARGET_COLUMN].isna()][TARGET_COLUMN].unique()
        raise ValueError(
            f"Valores de RiskLevel nao mapeados: {valores_desconhecidos}. "
            f"Esperados: {list(RISK_ENCODING.keys())}"
        )
    return df


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separa features e alvo."""
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split estratificado treino/teste.

    Estratificacao preserva a proporcao das classes em ambos os conjuntos,
    importante dado o desbalanceamento moderado (low 40% / mid 33% / high 27%).
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    print(
        f"[split] Treino: {len(X_train)} linhas | Teste: {len(X_test)} linhas "
        f"({test_size*100:.0f}% teste)"
    )
    return X_train, X_test, y_train, y_test


def fit_scaler(X_train: pd.DataFrame) -> StandardScaler:
    """
    Ajusta o StandardScaler apenas nos dados de treino.

    NUNCA ajustar no conjunto completo (data leakage): o scaler deve
    'ver' somente o treino e ser aplicado ao teste sem reajuste.
    """
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler


def apply_scaler(
    scaler: StandardScaler,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Aplica o scaler ja ajustado a treino e teste, preservando nomes de colunas.
    """
    cols = X_train.columns.tolist()
    X_train_scaled = pd.DataFrame(
        scaler.transform(X_train), columns=cols, index=X_train.index
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=cols, index=X_test.index
    )
    return X_train_scaled, X_test_scaled


# ---------------------------------------------------------------------------
# Funcao principal: orquestra todas as etapas
# ---------------------------------------------------------------------------


def build_preprocessed_splits(
    path: Optional[Path] = None,
    add_pulse_pressure: bool = False,
    add_age_features: bool = False,
    convert_temp: bool = True,
    scale: bool = True,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, StandardScaler]:
    """
    Executa o pipeline completo e retorna os splits prontos para modelagem.

    Parametros
    ----------
    path : Path, opcional
        Caminho alternativo para o CSV bruto. Se None, usa o padrao.
    add_pulse_pressure : bool
        Se True, cria pulse_pressure = SystolicBP - DiastolicBP.
    convert_temp : bool
        Se True (padrao), converte BodyTemp de Fahrenheit para Celsius.
    add_age_features : bool
        Se True, cria age_advanced (binario) e age_group (ordinal 0/1/2).
        Os grupos seguem limiares clinicos: <20, 20-34, >=35.
    scale : bool
        Se True (padrao), aplica StandardScaler. Desative para modelos
        que nao precisam de escala (ex.: Random Forest isolado).
    test_size : float
        Fracao do dataset para o conjunto de teste (padrao: 0.20).
    random_state : int
        Semente para reproducibilidade (padrao: 42).

    Retorno
    -------
    X_train, X_test, y_train, y_test : DataFrames / Series
    scaler : StandardScaler ajustado no treino (use para transformar dados novos)
    """
    print("=" * 55)
    print("PIPELINE DE PRE-PROCESSAMENTO")
    print("=" * 55)

    # 1. Carregamento
    df = load_dataset(path)
    print(f"[load] Dataset carregado: {df.shape[0]} linhas x {df.shape[1]} colunas")

    # 2. Limpeza
    df = remove_invalid_outliers(df)
    df = remove_duplicates(df)

    # 3. Features derivadas opcionais e conversoes
    if convert_temp:
        df = _convert_body_temp_to_celsius(df)
    if add_pulse_pressure:
        df = _add_pulse_pressure(df)
    if add_age_features:
        df = _add_age_features(df)

    # 4. Encoding do alvo
    df = encode_target(df)

    # 5. Separar features e alvo
    X, y = split_features_target(df)

    # 6. Split estratificado
    X_train, X_test, y_train, y_test = stratified_split(X, y, test_size, random_state)

    # 7. Padronizacao
    scaler = fit_scaler(X_train)
    if scale:
        X_train, X_test = apply_scaler(scaler, X_train, X_test)
        print("[scaler] StandardScaler aplicado (fit no treino, transform em ambos).")
    else:
        print("[scaler] Padronizacao desativada (scale=False).")

    # Relatorio final
    print("-" * 55)
    print(f"Features finais ({len(X_train.columns)}): {X_train.columns.tolist()}")
    print("Distribuicao do alvo — treino:")
    for cls, nome in RISK_DECODING.items():
        n = (y_train == cls).sum()
        print(f"  {nome} ({cls}): {n} ({n/len(y_train)*100:.1f}%)")
    print("Distribuicao do alvo — teste:")
    for cls, nome in RISK_DECODING.items():
        n = (y_test == cls).sum()
        print(f"  {nome} ({cls}): {n} ({n/len(y_test)*100:.1f}%)")
    print("=" * 55)

    return X_train, X_test, y_train, y_test, scaler


# ---------------------------------------------------------------------------
# Execucao direta para verificacao rapida
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("\n--- Pipeline base (sem features derivadas) ---")
    X_tr, X_te, y_tr, y_te, sc = build_preprocessed_splits()

    print("\n--- Pipeline com pulse_pressure e age_features ---")
    X_tr2, X_te2, y_tr2, y_te2, sc2 = build_preprocessed_splits(
        add_pulse_pressure=True,
        add_age_features=True,
    )

    print("\nShapes:")
    print(f"  X_train base:     {X_tr.shape}")
    print(f"  X_train derivado: {X_tr2.shape}")
    print(f"  X_test base:      {X_te.shape}")
    print(f"  X_test derivado:  {X_te2.shape}")
