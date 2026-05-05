# Tech Challenge FIAP — Fase 1
## Predição de Risco em Saúde Materna

Projeto de conclusão da Fase 1 da pós-graduação **IA para Devs** da FIAP. Solução baseada em Machine Learning para classificação do nível de risco gestacional (baixo, médio, alto) a partir de sinais vitais e dados clínicos básicos, como suporte à triagem precoce em saúde materna.

---

## Sumário
- [Contexto](#contexto)
- [Dataset](#dataset)
- [Estrutura do Projeto](#estrutura-do-projeto)
- [Como Executar](#como-executar)
- [Modelos](#modelos)
- [Resultados](#resultados)
- [Autor](#autor)

---

## Contexto

Uma rede de hospitais especializada em saúde da mulher precisa de um sistema inteligente de suporte ao diagnóstico capaz de identificar precocemente situações de risco gestacional. Esta entrega corresponde à Fase 1 do Tech Challenge: a base de Machine Learning sobre dados estruturados.

O sistema classifica pacientes em três níveis de risco — baixo, médio e alto — usando sinais vitais coletados em consultas de pré-natal. O modelo atua como **suporte à decisão**: o profissional de saúde tem sempre a palavra final.

## Dataset

**Maternal Health Risk Data Set** — UCI Machine Learning Repository  
Link: https://archive.ics.uci.edu/dataset/863/maternal+health+risk

- **Origem:** dados coletados em hospitais e clínicas comunitárias rurais de Bangladesh via dispositivos IoT
- **Volume original:** 1.014 registros × 7 colunas (6 features + alvo)
- **Após limpeza:** 451 registros únicos (remoção de 2 outliers e 561 duplicatas)

| Atributo | Descrição | Unidade |
|---|---|---|
| `Age` | Idade da paciente | anos |
| `SystolicBP` | Pressão arterial sistólica | mmHg |
| `DiastolicBP` | Pressão arterial diastólica | mmHg |
| `BS` | Glicemia | mmol/L |
| `BodyTemp` | Temperatura corporal | °F → convertida para °C no pipeline |
| `HeartRate` | Frequência cardíaca | bpm |
| `RiskLevel` | **Alvo** — nível de risco gestacional | low / mid / high risk |

## Estrutura do Projeto

```
.
├── data/
│   └── raw/
│       └── maternal_health_risk.csv    # dataset original (imutável)
├── notebooks/
│   ├── 01_eda.ipynb                    # análise exploratória completa
│   ├── 02_modeling.ipynb               # pré-processamento, CV, GridSearch, avaliação
│   └── 03_evaluation.ipynb             # curvas ROC, SHAP, discussão crítica
├── src/
│   ├── pipelines/
│   │   ├── data_loading.py             # carga e validação do dataset
│   │   └── preprocessing.py            # pipeline de pré-processamento
│   ├── models/
│   │   └── train.py                    # pipelines sklearn, CV, GridSearch
│   ├── evaluation/
│   │   ├── metrics.py                  # métricas, matrizes de confusão
│   │   └── explainability.py           # curvas ROC, SHAP
│   └── utils/
├── reports/
│   ├── relatorio_tecnico.md            # relatório técnico completo
│   └── figures/                        # gráficos gerados pelos notebooks
├── requirements.txt
├── Dockerfile
└── README.md
```

## Como Executar

### Opção 1 — Ambiente virtual local

```bash
# 1. Criar e ativar o virtualenv
python3 -m venv .venv
source .venv/bin/activate          # Linux/Mac
# .venv\Scripts\activate           # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Registrar o kernel no JupyterLab
python -m ipykernel install --user --name=tech-challenge --display-name "Tech Challenge (venv)"

# 4. Abrir o JupyterLab
jupyter lab
```

Selecione o kernel **Tech Challenge (venv)** nos notebooks e execute na ordem:
1. `notebooks/01_eda.ipynb`
2. `notebooks/02_modeling.ipynb`
3. `notebooks/03_evaluation.ipynb`

### Opção 2 — Docker

```bash
docker build -t tech-challenge-fiap .
docker run -p 8888:8888 -v $(pwd):/app tech-challenge-fiap
```

Acesse `http://localhost:8888` e abra os notebooks em `notebooks/`.

### Verificação rápida (sem notebook)

```bash
python -m src.pipelines.data_loading     # testa carga do CSV
python -m src.pipelines.preprocessing    # testa pipeline completo
python -m src.models.train               # testa imports dos pipelines
```

## Modelos

Três algoritmos de famílias distintas, todos encapsulados em `sklearn.Pipeline` (StandardScaler + modelo):

| Modelo | Família | Papel |
|---|---|---|
| Regressão Logística | Linear | Baseline interpretável |
| Random Forest | Ensemble de árvores | Modelo principal — feature importance nativo |
| SVM (kernel RBF) | Método de kernel | Forte em datasets pequenos, captura não-linearidades |

Hiperparâmetros otimizados via `GridSearchCV` com validação cruzada estratificada k=5, métrica de otimização: **F1-macro**.

## Resultados

*(Preencher com os valores reais após execução do notebook 02.)*

**Métricas no conjunto de teste:**

| Modelo | F1-macro | Recall high risk | Accuracy | ROC-AUC |
|---|---|---|---|---|
| Regressão Logística | — | — | — | — |
| Random Forest | — | — | — | — |
| SVM | — | — | — | — |

**Métrica principal:** F1-score macro — penaliza modelos que ignoram classes minoritárias.  
**Métrica de segurança clínica:** recall da classe `high risk` — falsos negativos nessa classe representam o erro mais grave no contexto obstétrico.

## Autor

**Igor Natanael** — Pós-graduação IA para Devs / FIAP
