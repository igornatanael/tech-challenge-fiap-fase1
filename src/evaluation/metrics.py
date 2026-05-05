"""
Funcoes de avaliacao de modelos de classificacao.

Metricas implementadas:
- Classification report completo (precision, recall, F1 por classe + macro/weighted)
- Matriz de confusao (absoluta e normalizada)
- F1-macro e recall por classe (metricas principais do projeto)
- ROC-AUC multiclasse (estrategia One-vs-Rest, media macro)
- Tabela comparativa entre multiplos modelos

Metrica principal do projeto: F1-score macro.
Metrica de seguranca clinica: recall da classe 'high risk' (2).

Justificativa: o custo de um falso negativo em alto risco gestacional
(paciente de risco alto classificada como baixo) e clinicamente inaceitavel.
O sistema deve errar para o lado da seguranca — prefere-se sobrediagnosticar
a subdiagnosticar. Acuracia simples seria enganosa dado o desbalanceamento.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

# Mapeamento ordinal do alvo (deve estar sincronizado com preprocessing.py)
RISK_DECODING = {0: "low risk", 1: "mid risk", 2: "high risk"}
CLASS_NAMES = ["low risk", "mid risk", "high risk"]
N_CLASSES = 3


# ---------------------------------------------------------------------------
# Metricas individuais
# ---------------------------------------------------------------------------


def compute_classification_report(
    y_true: pd.Series,
    y_pred: np.ndarray,
    output_dict: bool = False,
):
    """
    Exibe o classification report do sklearn.

    Inclui precision, recall e F1 por classe, mais media macro e weighted.
    Nomes das classes sao traduzidos via RISK_DECODING para legibilidade.
    """
    target_names = [RISK_DECODING[i] for i in range(N_CLASSES)]
    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        output_dict=output_dict,
    )
    if not output_dict:
        print(report)
    return report


def compute_roc_auc(
    y_true: pd.Series,
    y_prob: np.ndarray,
) -> float:
    """
    Calcula ROC-AUC multiclasse com estrategia One-vs-Rest (OvR), media macro.

    OvR trata cada classe como binaria (classe X vs todas as outras),
    adequado para desbalanceamento moderado.

    Parametros
    ----------
    y_prob : array de shape (n_amostras, n_classes)
        Probabilidades preditas por pipeline.predict_proba().
    """
    y_bin = label_binarize(y_true, classes=list(range(N_CLASSES)))
    auc = roc_auc_score(y_bin, y_prob, multi_class="ovr", average="macro")
    return round(float(auc), 4)


def compute_metrics_summary(
    y_true: pd.Series,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> dict:
    """
    Retorna um dicionario com as metricas consolidadas de um modelo.

    Inclui F1-macro, recall por classe, acuracia e ROC-AUC (se y_prob fornecido).
    """
    report = compute_classification_report(y_true, y_pred, output_dict=True)

    metricas = {
        "accuracy":           round(report["accuracy"], 4),
        "f1_macro":           round(report["macro avg"]["f1-score"], 4),
        "recall_low_risk":    round(report[RISK_DECODING[0]]["recall"], 4),
        "recall_mid_risk":    round(report[RISK_DECODING[1]]["recall"], 4),
        "recall_high_risk":   round(report[RISK_DECODING[2]]["recall"], 4),
        "precision_macro":    round(report["macro avg"]["precision"], 4),
    }

    if y_prob is not None:
        metricas["roc_auc_macro_ovr"] = compute_roc_auc(y_true, y_prob)

    return metricas


# ---------------------------------------------------------------------------
# Avaliacao completa de um modelo
# ---------------------------------------------------------------------------


def evaluate_model(
    nome: str,
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    plot_confusion: bool = True,
    save_path: Optional[str] = None,
) -> dict:
    """
    Avalia um pipeline treinado no conjunto de teste.

    Imprime o classification report, plota a matriz de confusao e
    retorna um dicionario com todas as metricas.

    Parametros
    ----------
    nome : str
        Nome do modelo (para titulo do grafico).
    pipeline : Pipeline sklearn ja treinado.
    X_test : features do conjunto de teste.
    y_test : alvo do conjunto de teste.
    plot_confusion : bool
        Se True, exibe a matriz de confusao normalizada.
    save_path : str, opcional
        Se fornecido, salva o grafico neste caminho.

    Retorno
    -------
    dict com as metricas (pode ser acumulado numa tabela comparativa).
    """
    y_pred = pipeline.predict(X_test)
    y_prob = None
    if hasattr(pipeline, "predict_proba"):
        y_prob = pipeline.predict_proba(X_test)

    print(f"\n{'='*55}")
    print(f"AVALIACAO — {nome.upper()}")
    print(f"{'='*55}")
    compute_classification_report(y_true=y_test, y_pred=y_pred)

    metricas = compute_metrics_summary(y_test, y_pred, y_prob)

    if y_prob is not None:
        print(f"ROC-AUC macro OvR: {metricas['roc_auc_macro_ovr']:.4f}")

    if plot_confusion:
        _plot_confusion_matrix(nome, y_test, y_pred, save_path=save_path)

    return metricas


def _plot_confusion_matrix(
    nome: str,
    y_true: pd.Series,
    y_pred: np.ndarray,
    save_path: Optional[str] = None,
) -> None:
    """Plota matriz de confusao normalizada por linha (recall por classe)."""
    cm = confusion_matrix(y_true, y_pred, normalize="true")
    cm_abs = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm, annot=False, fmt=".0%", cmap="Blues",
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        linewidths=0.5, ax=ax,
    )
    # Anotacoes: proporcao (normalizada) + contagem absoluta
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(
                j + 0.5, i + 0.5,
                f"{cm[i, j]:.0%}\n({cm_abs[i, j]})",
                ha="center", va="center",
                fontsize=9,
                color="white" if cm[i, j] > 0.6 else "black",
            )

    ax.set_title(f"Matriz de Confusao — {nome}\n(normalizada por linha = recall por classe)")
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ---------------------------------------------------------------------------
# Tabela comparativa entre modelos
# ---------------------------------------------------------------------------


def build_comparison_table(
    resultados: dict[str, dict],
    highlight_col: str = "f1_macro",
) -> pd.DataFrame:
    """
    Constroi uma tabela comparativa de metricas entre todos os modelos.

    Parametros
    ----------
    resultados : dict {nome_modelo: dict_de_metricas}
        Dicionario acumulado por evaluate_model().
    highlight_col : str
        Coluna usada para ordenar a tabela (decrescente).

    Retorno
    -------
    DataFrame ordenado pela metrica principal, pronto para exibicao no notebook.
    """
    df = pd.DataFrame(resultados).T
    df = df.sort_values(highlight_col, ascending=False)
    return df


# ---------------------------------------------------------------------------
# Feature importance (Random Forest)
# ---------------------------------------------------------------------------


def plot_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
    top_n: int = 10,
    save_path: Optional[str] = None,
) -> pd.Series:
    """
    Plota e retorna a importancia das features de um modelo de arvore.

    Funciona com qualquer estimador que tenha `feature_importances_`
    (Random Forest, Decision Tree, Gradient Boosting).

    Parametros
    ----------
    pipeline : Pipeline sklearn treinado.
    feature_names : nomes das colunas de X_train.
    top_n : numero de features a exibir.
    """
    modelo = pipeline.named_steps["model"]
    if not hasattr(modelo, "feature_importances_"):
        raise ValueError(
            f"O modelo '{type(modelo).__name__}' nao possui feature_importances_. "
            "Use Random Forest ou outro modelo de arvore."
        )

    importancias = pd.Series(
        modelo.feature_importances_,
        index=feature_names,
    ).sort_values(ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(9, 5))
    importancias.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(f"Feature Importance — Random Forest (top {top_n})")
    ax.set_xlabel("Importancia (reducao media de impureza de Gini)")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    return importancias
