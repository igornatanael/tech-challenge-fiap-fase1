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
- **Após limpeza:** ~790 registros (remoção de outliers fisiológicos, registros com idade > 54 anos e ocorrências excessivas de medições repetidas)

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
docker run -p 8888:8888 -v "$(pwd):/app" tech-challenge-fiap
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

**Métricas no conjunto de teste:**

| Modelo | F1-macro | Recall high risk | Accuracy | ROC-AUC macro |
|---|---|---|---|---|
| Regressão Logística | 0.68 | 0.79 | 0.68 | 0.827 |
| SVM | 0.76 | 0.87 | 0.76 | 0.871 |
| **Random Forest** | **0.90** | **0.95** | **0.89** | **0.980** |

> Relatório técnico completo com análise detalhada: [reports/relatorio_tecnico.md](reports/relatorio_tecnico.md)

**Métrica principal:** F1-score macro — penaliza modelos que ignoram classes minoritárias.  
**Métrica de segurança clínica:** recall da classe `high risk` — falsos negativos nessa classe representam o erro mais grave no contexto obstétrico.

## API de Predição

O projeto inclui uma API Flask que treina o modelo em memória ao iniciar e responde predições de risco gestacional.

### Iniciar o servidor

**Local:**
```bash
python -m src.api.app
# Servidor disponível em http://localhost:5000
```

**Docker:**
```bash
docker build -t tech-challenge-api -f Dockerfile.api .
docker run -p 5000:5000 tech-challenge-api
```

### Endpoints

| Método | Rota | Descrição |
|---|---|---|
| GET | `/health` | Status do servidor e modelo |
| POST | `/predict` | Predição de risco a partir dos sinais vitais |

### Exemplo de requisição

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Age": 35,
    "SystolicBP": 130,
    "DiastolicBP": 90,
    "BS": 11.0,
    "BodyTemp": 37.0,
    "HeartRate": 77
  }'
```

```json
{
  "risco": "high risk",
  "nivel": 2,
  "probabilidades": {
    "low risk": 0.03,
    "mid risk": 0.08,
    "high risk": 0.89
  }
}
```

> `BodyTemp` deve ser informado em **graus Celsius**.

## CLI Interativa

Com a API no ar, rode o formulário em linha de comando:

```bash
python cli_predict.py
# URL customizada:
python cli_predict.py --url http://localhost:5000
```

A CLI guia o usuário campo a campo, valida os valores e exibe o resultado com indicação visual de risco.

## Testes

```bash
# Instalar dependências (inclui pytest)
pip install -r requirements.txt

# Rodar todos os testes
pytest tests/test_api.py -v

# Ver tabela com predições de todos os perfis de teste
pytest tests/test_api.py -v -s
```

Os testes cobrem: endpoint `/health`, predições para perfis clínicos representativos das 3 classes de risco, validação de campos inválidos/ausentes e aceite de form-data além de JSON. Não é necessário ter a API no ar — o cliente de teste do Flask é usado internamente.

## Autor

**Igor Natanael** — Pós-graduação IA para Devs / FIAP
