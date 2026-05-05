"""
Funcoes de explicabilidade e visualizacao avancada de avaliacao.

Conteudo:
- Curvas ROC por classe (estrategia One-vs-Rest, multiclasse)
- Analise SHAP: calculo de valores e plots (summary, waterfall, bar)

Sobre SHAP:
    SHAP (SHapley Additive exPlanations) atribui a cada feature uma contribuicao
    marginal para a predicao de uma instancia especifica, baseado na teoria dos
    jogos cooperativos. E o padrao atual de explicabilidade em ML.

Estrategia por modelo:
    - Random Forest  -> TreeExplainer  (exato, rapido para modelos de arvore)
    - Regressao Log. -> LinearExplainer (exato, rapido para modelos lineares)
    - SVM            -> KernelExplainer (aproximado, usa amostra de background)

O scaler do Pipeline e aplicado antes de passar ao explainer, pois o SHAP
precisa dos dados na mesma escala que o modelo 'viu' durante o treino.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc

# SHAP e importado dentro das funcoes para nao quebrar o modulo se nao instalado
# pip install shap

CLASS_NAMES = ["low risk", "mid risk", "high risk"]
N_CLASSES = 3
RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# Curvas ROC — multiclasse One-vs-Rest
# ---------------------------------------------------------------------------


def plot_roc_curves(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    nome_modelo: str = "Modelo",
    save_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Plota curvas ROC individuais por classe (estrategia One-vs-Rest).

    Para cada classe k, trata o problema como binario:
    - Positivo: classe k
    - Negativo: todas as outras classes

    Parametros
    ----------
    pipeline : Pipeline sklearn treinado (com predict_proba).
    X_test   : features do conjunto de teste.
    y_test   : alvo codificado (0/1/2).
    nome_modelo : string para o titulo do grafico.
    save_path : caminho para salvar a figura (opcional).

    Retorno
    -------
    DataFrame com AUC por classe e AUC macro.
    """
    y_prob = pipeline.predict_proba(X_test)
    y_bin = label_binarize(y_test, classes=list(range(N_CLASSES)))

    fig, ax = plt.subplots(figsize=(8, 6))

    aucs = {}
    cores = ["#4C72B0", "#DD8452", "#55A868"]

    for k, (classe, cor) in enumerate(zip(CLASS_NAMES, cores)):
        fpr, tpr, _ = roc_curve(y_bin[:, k], y_prob[:, k])
        auc_k = auc(fpr, tpr)
        aucs[classe] = round(auc_k, 4)
        ax.plot(fpr, tpr, label=f"{classe} (AUC = {auc_k:.3f})", color=cor, lw=2)

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Aleatório (AUC = 0.500)")
    ax.set_xlabel("Taxa de Falso Positivo (1 - Especificidade)")
    ax.set_ylabel("Taxa de Verdadeiro Positivo (Recall / Sensibilidade)")
    ax.set_title(f"Curvas ROC por classe — {nome_modelo}\n(estrategia One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()

    auc_macro = round(np.mean(list(aucs.values())), 4)
    aucs["macro"] = auc_macro
    return pd.Series(aucs, name=nome_modelo)


# ---------------------------------------------------------------------------
# SHAP — utilitarios internos
# ---------------------------------------------------------------------------


def _transform_X(pipeline: Pipeline, X: pd.DataFrame) -> pd.DataFrame:
    """
    Aplica apenas o scaler do pipeline (sem o modelo) para obter X na escala
    que o modelo recebeu durante o treino.

    Assume que o scaler e o primeiro passo do pipeline (nome: 'scaler').
    """
    scaler = pipeline.named_steps["scaler"]
    X_scaled = pd.DataFrame(
        scaler.transform(X),
        columns=X.columns,
        index=X.index,
    )
    return X_scaled


def _get_model(pipeline: Pipeline):
    """Extrai o estimador final do pipeline (nome: 'model')."""
    return pipeline.named_steps["model"]


# ---------------------------------------------------------------------------
# SHAP — calculo de valores
# ---------------------------------------------------------------------------


def compute_shap_values(
    pipeline: Pipeline,
    X_background: pd.DataFrame,
    X_explain: pd.DataFrame,
    tipo_modelo: str,
    n_background: int = 100,
) -> tuple:
    """
    Calcula valores SHAP para um pipeline treinado.

    Parametros
    ----------
    pipeline : Pipeline sklearn treinado.
    X_background : dados de treino (usados como referencia/background).
    X_explain : dados para os quais os valores SHAP serao calculados.
    tipo_modelo : 'random_forest', 'logistic_regression' ou 'svm'.
    n_background : numero de amostras de background para KernelExplainer (SVM).

    Retorno
    -------
    (explainer, shap_values)
    - shap_values: array de shape (n_amostras, n_features, n_classes) para
      TreeExplainer, ou lista de arrays para os demais.
    """
    import shap

    modelo = _get_model(pipeline)
    X_bg_scaled = _transform_X(pipeline, X_background)
    X_ex_scaled = _transform_X(pipeline, X_explain)

    if tipo_modelo == "random_forest":
        explainer = shap.TreeExplainer(modelo)
        shap_values = explainer.shap_values(X_ex_scaled, check_additivity=False)

    elif tipo_modelo == "logistic_regression":
        explainer = shap.LinearExplainer(modelo, X_bg_scaled)
        shap_values = explainer.shap_values(X_ex_scaled)

    elif tipo_modelo == "svm":
        # KernelExplainer e custoso — usa subset do background
        bg_sample = shap.sample(X_bg_scaled, min(n_background, len(X_bg_scaled)))
        explainer = shap.KernelExplainer(
            modelo.predict_proba, bg_sample, link="logit"
        )
        shap_values = explainer.shap_values(
            X_ex_scaled, nsamples=200, silent=True
        )
    else:
        raise ValueError(
            f"tipo_modelo '{tipo_modelo}' nao reconhecido. "
            "Use 'random_forest', 'logistic_regression' ou 'svm'."
        )

    return explainer, shap_values


# ---------------------------------------------------------------------------
# SHAP — visualizacoes
# ---------------------------------------------------------------------------


def plot_shap_summary(
    shap_values,
    X_explain: pd.DataFrame,
    tipo_modelo: str,
    classe_idx: int = 2,
    max_display: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Plota o summary plot (beeswarm) do SHAP para uma classe especifica.

    O beeswarm mostra:
    - Eixo Y: features ordenadas por importancia media (|SHAP|)
    - Eixo X: valor SHAP de cada amostra (+ = empurra para a classe, - = afasta)
    - Cor: valor original da feature (vermelho = alto, azul = baixo)

    Parametros
    ----------
    shap_values : saida de compute_shap_values.
    X_explain : DataFrame na escala original (para os labels das features).
    tipo_modelo : nome do modelo (para titulo do grafico).
    classe_idx : indice da classe a visualizar (0=low, 1=mid, 2=high).
                 Default=2 (high risk) — classe de maior relevancia clinica.
    max_display : numero maximo de features a exibir.
    """
    import shap

    classe_nome = ["low risk", "mid risk", "high risk"][classe_idx]

    # TreeExplainer retorna lista [array_classe0, array_classe1, array_classe2]
    # LinearExplainer idem; KernelExplainer idem
    if isinstance(shap_values, list):
        sv = shap_values[classe_idx]
    else:
        # formato (n_amostras, n_features, n_classes)
        sv = shap_values[:, :, classe_idx]

    plt.figure()
    shap.summary_plot(
        sv,
        X_explain,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(
        f"SHAP Summary Plot — {tipo_modelo}\nClasse: {classe_nome} (impacto de cada feature)"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_bar(
    shap_values,
    X_explain: pd.DataFrame,
    tipo_modelo: str,
    classe_idx: int = 2,
    max_display: int = 10,
    save_path: Optional[str] = None,
) -> None:
    """
    Plota o bar plot de importancia global SHAP (media de |SHAP|) por feature.

    Mais legivel que o beeswarm para uma visao de importancia global.
    """
    import shap

    classe_nome = ["low risk", "mid risk", "high risk"][classe_idx]

    if isinstance(shap_values, list):
        sv = shap_values[classe_idx]
    else:
        sv = shap_values[:, :, classe_idx]

    importancia_media = pd.Series(
        np.abs(sv).mean(axis=0),
        index=X_explain.columns,
    ).sort_values(ascending=True).tail(max_display)

    fig, ax = plt.subplots(figsize=(8, 5))
    importancia_media.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_title(
        f"SHAP — Importancia Global (|SHAP| medio)\n"
        f"{tipo_modelo} | Classe: {classe_nome}"
    )
    ax.set_xlabel("Valor SHAP medio absoluto")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_shap_waterfall_single(
    explainer,
    shap_values,
    X_explain: pd.DataFrame,
    amostra_idx: int,
    tipo_modelo: str,
    classe_idx: int = 2,
    save_path: Optional[str] = None,
) -> None:
    """
    Plota o waterfall plot para uma unica predicao.

    O waterfall mostra como cada feature empurrou a predicao para cima ou
    para baixo a partir do valor base (media das predicoes no background).

    E a forma mais intuitiva de explicar UMA decisao individual do modelo
    para um medico ou paciente.

    Parametros
    ----------
    amostra_idx : indice da amostra em X_explain a detalhar.
    classe_idx  : classe a visualizar (default=2: high risk).
    """
    import shap

    classe_nome = ["low risk", "mid risk", "high risk"][classe_idx]

    if isinstance(shap_values, list):
        sv = shap_values[classe_idx]
    else:
        sv = shap_values[:, :, classe_idx]

    # Recupera o valor base (expected_value)
    if isinstance(explainer.expected_value, (list, np.ndarray)):
        base_value = explainer.expected_value[classe_idx]
    else:
        base_value = explainer.expected_value

    shap_exp = shap.Explanation(
        values=sv[amostra_idx],
        base_values=base_value,
        data=X_explain.iloc[amostra_idx].values,
        feature_names=X_explain.columns.tolist(),
    )

    plt.figure()
    shap.plots.waterfall(shap_exp, show=False)
    plt.title(
        f"SHAP Waterfall — {tipo_modelo} | Amostra #{amostra_idx}\n"
        f"Classe analisada: {classe_nome}"
    )
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
