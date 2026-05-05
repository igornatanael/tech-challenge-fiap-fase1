# Relatório Técnico — Tech Challenge FIAP Fase 1
## Sistema de Classificação de Risco Gestacional

**Aluno:** Igor Natanael  
**Curso:** Pós-graduação IA para Devs — FIAP  
**Dataset:** Maternal Health Risk Data Set (UCI, ID 863)  
**Repositório:** *(inserir link do GitHub)*

---

## 1. Problema e Justificativa

Uma rede de hospitais especializada em saúde da mulher necessita de um sistema inteligente de suporte ao diagnóstico para triagem precoce de risco gestacional. O problema foi modelado como **classificação multiclasse supervisionada**: dadas medições clínicas básicas de uma paciente grávida, o modelo prediz o nível de risco gestacional — baixo, médio ou alto.

A escolha do dataset *Maternal Health Risk* foi motivada por três fatores: alinhamento direto com o tema de saúde materna, variáveis clinicamente significativas (sinais vitais coletáveis em qualquer posto de saúde) e disponibilidade pública documentada na UCI.

O modelo proposto atua exclusivamente como **suporte à decisão clínica**. O profissional de saúde tem sempre a palavra final — o sistema visa reduzir o tempo de triagem e chamar atenção para casos de alto risco, não substituir o julgamento médico.

---

## 2. Análise Exploratória de Dados (EDA)

### 2.1 Visão geral

O dataset original contém 1.014 registros × 7 colunas (6 features numéricas + alvo categórico). Não há valores ausentes. Todos os atributos são contínuos ou discretos — sem variáveis categóricas nas features.

### 2.2 Qualidade dos dados

**Outliers fisiológicos:** 2 registros com `HeartRate < 30 bpm` — valor impossível em humano vivo, provável erro de digitação. Removidos.

**Outlier de idade:** o dataset apresenta um pico concentrado em exatamente 60 anos que não aparece documentado como clinicamente válido em nenhum artigo que utiliza esta base. Gestações acima dos 54 anos são fisiologicamente muito improváveis sem reprodução assistida avançada — contexto não documentado no dataset. O corte em **Age ≤ 54** corresponde ao p75 da menopausa natural em estudos populacionais (mediana: 51,25 anos; p75 = 54 anos — *Maturitas*, 1995). 84 registros removidos.

**Medições repetidas:** o dataset possui 562 linhas (55,4%) com ao menos uma cópia idêntica em todos os campos. Este dataset é amplamente referenciado em artigos publicados na IEEE, PLOS ONE, Nature Scientific Reports e PMC — nenhum deles cita as repetições como problema de qualidade. Optamos por uma abordagem conservadora: **mantemos até 3 ocorrências por grupo idêntico**, removendo apenas grupos com 4 ou mais repetições, que têm maior probabilidade de serem artefatos de logging do sistema IoT de coleta. 140 linhas removidas. Dataset final após todas as etapas de limpeza: **~790 registros**.

### 2.3 Balanceamento do alvo

Após limpeza, a distribuição das classes é:

| Classe | Contagem | % |
|---|---|---|
| low risk | ~186 (treino) | ~51,7% |
| mid risk | ~85 (treino) | ~23,6% |
| high risk | ~89 (treino) | ~24,7% |

Desbalanceamento moderado — não justifica SMOTE, mas justifica `class_weight='balanced'` nos modelos.

### 2.4 Correlações com o alvo

Codificando o alvo como ordinal (low=0, mid=1, high=2):

| Feature | Pearson r |
|---|---|
| BS (glicemia) | 0,57 |
| SystolicBP | 0,40 |
| DiastolicBP | 0,35 |
| Age | 0,27 |
| HeartRate | 0,19 |
| BodyTemp | 0,16 |

Glicemia é a feature individualmente mais preditiva. Nenhuma feature tem correlação muito alta, indicando que o problema exige captura de interações — favorece modelos não-lineares.

### 2.5 Multicolinearidade

`SystolicBP` e `DiastolicBP` apresentam correlação de Pearson r = 0,79 — alta, esperada clinicamente. Isso afeta modelos lineares (Regressão Logística) mas não árvores.

### 2.6 Padrões clínicos

Os boxplots por classe confirmam padrões clinicamente coerentes: pacientes de alto risco apresentam medianas mais elevadas de idade (35 vs. 22-25 anos nas demais classes), glicemia (11,0 vs. 7,0-7,5 mmol/L) e pressão arterial. Isso valida a qualidade do sinal no dataset.

A classe `mid risk` é frequentemente indistinguível de `low risk` nas medianas — antecipa-se dificuldade de separação entre essas duas classes.

---

## 3. Pré-processamento

O pipeline de pré-processamento foi implementado em `src/pipelines/preprocessing.py` e executa as seguintes etapas em ordem:

**1. Remoção de outliers:** linhas com `HeartRate < 30 bpm` (2 registros).

**2. Remoção de registros com Age > 54:** p75 da menopausa natural — pico suspeito em 60 anos não documentado na literatura que usa esta base.

**3. Remoção conservadora de medições repetidas:** mantém até 3 ocorrências por grupo idêntico, remove a partir da 4ª. Aplicado antes do split para evitar *data leakage*.

**4. Conversão de temperatura:** `BodyTemp` convertida de Fahrenheit para Celsius (`(F - 32) × 5/9`), mais intuitivo no contexto clínico.

**5. Features derivadas opcionais:**
- `pulse_pressure = SystolicBP - DiastolicBP`: reduz a colinearidade entre as duas pressões sem descartar informação. Útil para a Regressão Logística.
- `age_advanced` (binário): `1` se `Age >= 35` (Advanced Maternal Age — AMA), limiar clínico padrão em obstetrícia.
- `age_group` (ordinal): `0` para `<20 anos`, `1` para `20–34 anos`, `2` para `≥35 anos`. Preserva a ordem clínica de risco crescente.

As features derivadas de idade foram mantidas junto com `Age` contínua. Modelos de árvore ignoram redundâncias; para a Regressão Logística, `age_advanced` captura o efeito de degrau que a relação linear não representaria bem.

**6. Encoding ordinal do alvo:** `low risk=0`, `mid risk=1`, `high risk=2`.

**7. Split estratificado 80/20:** `random_state=42`, preservando a proporção das classes. Dataset final após limpeza: ~790 registros (~632 treino / ~158 teste).

**8. StandardScaler:** ajustado exclusivamente no treino, aplicado em treino e teste. Necessário para Regressão Logística e SVM; incluído no Random Forest para uniformidade da API.

---

## 4. Modelagem

### 4.1 Escolha dos modelos

Três modelos de famílias algorítmicas distintas foram selecionados, todos cobertos pelo currículo do curso:

**Regressão Logística** (Aula 1): baseline linear interpretável. Usa `solver='lbfgs'` com `class_weight='balanced'`. Beneficia-se de `pulse_pressure` para mitigar a multicolinearidade. Limitado por fronteiras de decisão lineares.

**Random Forest** (Aula 4): ensemble de árvores de decisão via *bagging*. Robusto ao overfitting, lida naturalmente com multicolinearidade, fornece *feature importance* nativo. `class_weight='balanced_subsample'` — versão do balanceamento adaptada ao *bootstrap* do ensemble.

**SVM com kernel RBF** (Aula 2): método de kernel que mapeia os dados para espaço de alta dimensão, capturando fronteiras não-lineares. Especialmente eficaz em datasets de tamanho reduzido como este. `probability=True` habilita `predict_proba` para cálculo de ROC-AUC e SHAP.

### 4.2 Pipelines sklearn

Cada modelo foi encapsulado em um `sklearn.Pipeline`:

```
Pipeline: StandardScaler → Modelo
```

O pipeline garante que a padronização seja re-executada corretamente em cada fold da validação cruzada e no GridSearch, eliminando *data leakage* de escala.

### 4.3 Validação cruzada

`StratifiedKFold` com k=5, `shuffle=True`, `random_state=42`. A estratificação preserva a proporção das 3 classes em cada fold.

### 4.4 Ajuste de hiperparâmetros

`GridSearchCV` com k=5 estratificado, otimizando F1-macro:

| Modelo | Hiperparâmetros pesquisados |
|---|---|
| Regressão Logística | C ∈ {0.01, 0.1, 1, 10, 100} |
| Random Forest | n_estimators ∈ {100, 200, 300} × max_depth ∈ {None, 10, 20} × min_samples_leaf ∈ {1, 2, 4} |
| SVM | C ∈ {0.1, 1, 10, 100} × gamma ∈ {'scale', 'auto', 0.01, 0.1} |

---

## 5. Resultados

### 5.1 Melhores hiperparâmetros (GridSearchCV, k=5, F1-macro)

| Modelo | Hiperparâmetros selecionados | F1-macro CV |
|---|---|---|
| Regressão Logística | C = 0.1 | 0.594 |
| Random Forest | n_estimators=200, max_depth=20, min_samples_leaf=1 | 0.786 |
| SVM | C=100, gamma=scale | 0.716 |

Com o dataset ampliado (~790 registros após a revisão dos critérios de limpeza), os resultados de validação cruzada melhoraram significativamente — o Random Forest ganhou +14 pontos percentuais de F1-macro em relação à versão anterior com 451 registros, confirmando que os dados removidos anteriormente (duplicatas excessivas e outliers de idade) introduziam ruído real no treinamento.

### 5.2 Desempenho no conjunto de teste

| Modelo | Accuracy | F1-macro | Recall high risk | Precision high risk | ROC-AUC macro |
|---|---|---|---|---|---|
| Regressão Logística | 0.59 | 0.54 | 0.65 | 0.68 | 0.760 |
| SVM | 0.69 | 0.59 | 0.83 | 0.76 | 0.767 |
| **Random Forest** | **0.70** | **0.62** | **0.91** | **0.75** | **0.784** |

**Classification report completo — Random Forest (modelo selecionado):**

|  | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| low risk | 0.76 | 0.83 | 0.80 | 47 |
| mid risk | 0.33 | 0.19 | 0.24 | 21 |
| high risk | 0.75 | 0.91 | 0.82 | 23 |
| **macro avg** | **0.62** | **0.64** | **0.62** | **91** |

### 5.3 Análise de falsos positivos para alto risco

Para o Random Forest, das 23 pacientes de alto risco reais no conjunto de teste:

- **21 corretamente identificadas** como alto risco (verdadeiros positivos)
- **2 não detectadas** — classificadas como baixo ou médio risco (falsos negativos)
- **7 pacientes de baixo/médio risco** foram erroneamente alertadas como alto risco (falsos positivos)

O modelo gerou 28 alertas de alto risco no total: 21 corretos (75% de precisão) e 7 desnecessários. Em termos práticos, isso significa que para cada 4 alertas emitidos, 3 correspondem a pacientes genuinamente em risco. O custo dos 7 encaminhamentos desnecessários — uma avaliação adicional — é clinicamente aceitável em comparação com os 2 casos graves que eventualmente não são detectados.

### 5.4 AUC por classe (One-vs-Rest)

| Modelo | AUC low risk | AUC mid risk | AUC high risk | AUC macro |
|---|---|---|---|---|
| Regressão Logística | 0.826 | 0.583 | 0.871 | 0.760 |
| SVM | 0.810 | 0.537 | 0.955 | 0.767 |
| Random Forest | 0.830 | 0.563 | **0.958** | **0.784** |

### 5.5 Feature importance (Random Forest)

*(Ver gráfico: `reports/figures/09_feature_importance_rf.png`)*

A análise de feature importance do Random Forest e os valores SHAP convergem para o mesmo padrão identificado na EDA: `BS` (glicemia) é a variável de maior poder preditivo, especialmente para a classe `high risk`. As features de pressão arterial contribuem em conjunto, com `pulse_pressure` capturando parte do sinal combinado das duas. A feature `age_advanced` (binário AMA, ≥35 anos) apresentou contribuição relevante para identificação de alto risco, validando a decisão de incluí-la como feature derivada.

### 5.6 Análise SHAP

*(Ver gráfico: `reports/figures/11_shap_summary_rf_high.png`)*

O summary plot SHAP para a classe `high risk` confirma: valores elevados de `BS` (vermelho) empurram fortemente a predição para alto risco, enquanto valores baixos (azul) afastam. `SystolicBP` e `Age` seguem o mesmo padrão direcional. O waterfall de predição individual demonstra como o modelo combina esses sinais para uma decisão específica — ferramenta diretamente útil para explicar uma triagem ao profissional de saúde.

*(Ver gráfico: `reports/figures/13_shap_waterfall_rf.png`)*

---

## 6. Discussão Crítica

### 6.1 O modelo é adequado para triagem de alto risco gestacional

Os resultados demonstram que o Random Forest é eficaz para o objetivo clínico central: **identificar pacientes de alto risco gestacional**. Com recall de 0.91 e AUC de 0.958 para a classe `high risk`, o modelo detecta 9 em cada 10 pacientes de alto risco com base apenas em sinais vitais básicos coletáveis em qualquer posto de saúde.

Esse desempenho é significativo no contexto de aplicação: o dataset foi construído a partir de dados de clínicas comunitárias rurais de Bangladesh, onde recursos diagnósticos avançados frequentemente não estão disponíveis. Um sistema capaz de sinalizar alto risco com precisão de 75% e recall de 91% — usando apenas pressão arterial, glicemia, temperatura, frequência cardíaca e idade — representa uma ferramenta de triagem de valor real nesse cenário.

### 6.2 A limitação principal: a classe mid risk

O desempenho para `mid risk` é fraco em todos os modelos (F1 de 0.19 a 0.24, AUC de 0.54 a 0.58 — próximo ao aleatório). Essa limitação não é um problema de modelagem — é uma limitação dos dados. A análise exploratória já antecipava essa dificuldade: as medianas de `mid risk` e `low risk` são praticamente idênticas em quase todas as features, e em `BS` a mediana de `mid risk` (7.0 mmol/L) é até menor que a de `low risk` (7.5 mmol/L).

A hipótese mais provável é que o rótulo `mid risk` foi atribuído com base em critérios clínicos que não estão representados nas 6 variáveis disponíveis — como histórico obstétrico, paridade ou resultado de exames complementares. Sem essas informações, nenhum modelo conseguirá separar `mid risk` de `low risk` de forma confiável.

**Implicação prática:** o sistema deve ser comunicado claramente como uma ferramenta de triagem de `high risk`, não como um classificador de três níveis. A distinção entre `low` e `mid risk` requer avaliação clínica presencial.

### 6.3 Por que F1-macro e recall de high risk são as métricas corretas

A acurácia simples é enganosa neste problema: um modelo que classificasse todas as pacientes como `low risk` ainda acertaria ~52% dos casos do conjunto de teste. O **F1-macro** penaliza modelos que ignoram classes, calculando a média não-ponderada do F1 de cada classe.

O **recall de `high risk`** é a métrica de segurança clínica por excelência. Um falso negativo nessa classe significa uma paciente de alto risco gestacional enviada para casa sem acompanhamento — o erro com maior potencial de consequências graves. O sistema deve preferir sobrediagnosticar (7 encaminhamentos desnecessários no conjunto de teste) a subdiagnosticar (2 casos graves não detectados).

### 6.4 Limitações adicionais

**Tamanho da base:** ~790 registros após limpeza. Os resultados da validação cruzada (k=5) são estimativas mais confiáveis que o resultado único no conjunto de teste (~158 amostras).

**Origem dos dados:** dataset de Bangladesh, contexto rural específico. Padrões aprendidos podem não generalizar para outras populações com perfis demográficos, nutricionais e de acesso à saúde distintos.

**Qualidade do ground truth:** o protocolo exato de rotulagem do `RiskLevel` não é público. Incerteza sobre a consistência da anotação afeta diretamente o teto de desempenho possível, especialmente para `mid risk`.

### 6.5 Uso clínico responsável

O modelo atua como **sistema de triagem**, não como substituto ao julgamento clínico. O profissional de saúde tem sempre a palavra final.

Fluxo de uso proposto:
1. Sinais vitais básicos coletados na consulta de pré-natal.
2. Sistema classifica o risco e exibe alerta com as features que mais contribuíram (via SHAP) — ex.: *"Glicemia elevada e idade ≥ 35 anos são os principais fatores de risco para esta paciente"*.
3. Casos sinalizados como `high risk` são priorizados para avaliação médica ou encaminhamento especializado.
4. O profissional decide o encaminhamento com base no alerta e na avaliação clínica presencial.

Para uso em produção: validação externa em dataset independente, análise de fairness por faixa etária e origem geográfica, calibração do limiar de decisão para `high risk` e monitoramento contínuo de data drift.

---

## 7. Conclusão

O projeto construiu e avaliou um sistema de classificação de risco gestacional baseado em Machine Learning, aplicando o ciclo completo: EDA orientada ao domínio clínico, pipeline de pré-processamento robusto com prevenção explícita de *data leakage*, modelagem comparativa com três algoritmos de famílias distintas e explicabilidade via SHAP.

O **Random Forest** foi o modelo com melhor desempenho geral, com F1-macro de 0.62, acurácia de 70% e, mais importante, **recall de 0.91 e AUC de 0.958 para a classe `high risk`**. Esses números demonstram que o modelo é adequado para seu propósito principal: sinalizar pacientes de alto risco gestacional com base em sinais vitais básicos, priorizando casos críticos para avaliação médica.

A principal limitação identificada — a dificuldade de separar `mid risk` de `low risk` — não é um problema de modelagem, mas uma limitação intrínseca das features disponíveis. Essa restrição deve ser comunicada claramente na interface do sistema, posicionando-o como ferramenta de triagem de alto risco, não como classificador completo de três níveis.

O sistema proposto tem potencial de valor real em contextos de triagem com recursos limitados, reduzindo o tempo de identificação de casos críticos e apoiando a tomada de decisão clínica — sempre com o profissional de saúde como responsável final pelo diagnóstico e conduta.

---

*Relatório gerado como parte do Tech Challenge — Fase 1 / IA para Devs / FIAP.*
