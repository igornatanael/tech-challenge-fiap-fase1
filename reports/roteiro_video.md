# Roteiro — Video de Demonstracao
## Tech Challenge FIAP Fase 1 — Sistema de Classificacao de Risco Gestacional

**Duracao total sugerida:** 12 a 15 minutos  
**Formato:** gravacao de tela + naracao  
**Ferramenta sugerida:** OBS Studio ou Quicktime (Mac)

---

## BLOCO 1 — Abertura e contexto (1-2 min)

**O que mostrar na tela:** slide ou README.md aberto no navegador

**Naracao sugerida:**
> "Ola, meu nome e Igor Natanael e este e o Tech Challenge da Fase 1 da pos-graduacao
> IA para Devs da FIAP. O objetivo deste projeto e construir um sistema de Machine
> Learning capaz de classificar o nivel de risco gestacional de pacientes gravidas
> em tres niveis: baixo, medio e alto risco.
>
> Utilizamos o dataset Maternal Health Risk, da UCI, coletado em clinicas rurais de
> Bangladesh. Apos limpeza, trabalhamos com 451 registros unicos e 6 variaveis
> clinicas basicas — pressao arterial, glicemia, temperatura, frequencia cardiaca
> e idade. Vou mostrar todo o fluxo: da analise exploratoria ate a explicabilidade
> do modelo."

---

## BLOCO 2 — EDA (2-3 min)

**O que mostrar na tela:** `notebooks/01_eda.ipynb` — percorrer as secoes principais

**Pontos a destacar:**
- Mostrar o dataset (`df.head()`, `df.describe()`)
- Destacar os 55% de duplicatas e a decisao de remover antes do split
- Mostrar os boxplots por classe (secao 6.2) — evidenciar que `high risk` tem
  glicemia e pressao visivelmente mais altas
- Mostrar a matriz de correlacao (secao 7) — BS com r=0.57, multicolinearidade
  entre SystolicBP e DiastolicBP
- Mencionar que `mid risk` e dificil de separar de `low risk` nas medianas

**Naracao sugerida:**
> "Na analise exploratoria, dois achados orientaram todo o pre-processamento:
> 55% do dataset era duplicatas — removemos antes do split para evitar data leakage.
> E a glicemia e a variavel mais correlacionada com o risco, com r de 0.57.
> Tambem identificamos que mid risk se confunde com low risk nas medianas —
> spoiler: isso vai aparecer nos resultados do modelo."

---

## BLOCO 3 — Pre-processamento (1-2 min)

**O que mostrar na tela:** terminal rodando `python -m src.pipelines.preprocessing`

**Pontos a destacar:**
- Mostrar o output do pipeline rodando: outliers removidos, duplicatas, features derivadas
- Mencionar o sklearn Pipeline e por que ele evita data leakage
- Mencionar rapidamente as features derivadas: pulse_pressure, age_advanced

**Naracao sugerida:**
> "O pipeline de pre-processamento foi implementado em src/pipelines/preprocessing.py.
> Ele executa em sequencia: remove os 2 outliers de frequencia cardiaca impossivel,
> as duplicatas, converte a temperatura para Celsius e cria features derivadas.
> Tudo encapsulado em sklearn Pipelines — o StandardScaler so ve o treino,
> garantindo que nao haja vazamento de informacao para o teste."

---

## BLOCO 4 — Modelagem e resultados (4-5 min)

**O que mostrar na tela:** `notebooks/02_modeling.ipynb` — executar ou mostrar ja executado

**Pontos a destacar:**

1. Mostrar os 3 pipelines (secao 3) e explicar brevemente cada modelo
2. Mostrar os resultados da CV k=5 (secao 4) — grafico de barras com F1-macro
3. Mostrar os melhores hiperparametros do GridSearch (secao 5)
4. Mostrar o classification report do Random Forest (secao 6) — destacar recall 0.91 em high risk
5. Mostrar a tabela comparativa (secao 7) — RF como melhor modelo
6. Mostrar o grafico de feature importance (secao 8)

**Naracao sugerida:**
> "Treinamos tres modelos de familias distintas: Regressao Logistica como baseline linear,
> Random Forest como ensemble de arvores, e SVM com kernel RBF para capturar
> fronteiras nao-lineares. Todos encapsulados em sklearn Pipelines com GridSearchCV.
>
> O Random Forest foi o melhor: F1-macro de 0.62 e, mais importante para o contexto
> clinico, recall de 0.91 para alto risco. Isso significa que o modelo identificou
> 21 de 23 pacientes de alto risco no conjunto de teste, com apenas 7 falsos positivos.
>
> Como esperado pela EDA, mid risk foi a classe mais dificil — recall de apenas 0.19.
> Isso nao e falha do modelo: as features disponíveis simplesmente nao diferenciam
> mid de low risk. Vamos ver isso confirmado no SHAP."

---

## BLOCO 5 — Explicabilidade SHAP (2-3 min)

**O que mostrar na tela:** `notebooks/03_evaluation.ipynb` — secao 4

**Pontos a destacar:**
1. Mostrar as curvas ROC (secao 3) — destacar AUC de 0.958 para high risk no RF
2. Mostrar o SHAP summary plot (secao 4.1) — glicemia dominando
3. Mostrar o waterfall de predicao individual (secao 4.2) — explicar como funciona

**Naracao sugerida:**
> "Para explicabilidade, usamos SHAP — que atribui a cada feature sua contribuicao
> marginal para cada predicao individual.
>
> No summary plot para high risk, fica claro: glicemia alta em vermelho empurra
> fortemente para alto risco. Pressao arterial e idade seguem o mesmo padrao.
>
> O waterfall mostra uma predicao individual — e exatamente assim que um medico
> veria o sistema: 'para esta paciente, a glicemia elevada e a idade acima de 35
> foram os principais fatores que levaram ao alerta de alto risco'.
>
> As curvas ROC confirmam: AUC de 0.958 para high risk no Random Forest.
> O modelo separa muito bem os casos criticos das demais classes."

---

## BLOCO 6 — Discussao critica e fechamento (1-2 min)

**O que mostrar na tela:** `reports/relatorio_tecnico.md` ou secao 5 do notebook 03

**Pontos a destacar:**
- O modelo e uma ferramenta de triagem, nao substitui o medico
- Limitacoes honestas: 451 amostras, mid risk irresoluvel com estas features
- Potencial real: util em contextos rurais com recursos limitados

**Naracao sugerida:**
> "Para concluir: este sistema nao substitui o profissional de saude — ele
> apoia a triagem. Com base em sinais vitais coletaveis em qualquer posto de saude,
> o modelo sinaliza 9 em cada 10 pacientes de alto risco corretamente.
>
> A limitacao principal e a classe mid risk, que as features disponiveis nao conseguem
> separar de low risk — isso deve ser comunicado claramente na interface do sistema.
>
> O codigo completo esta no repositorio com README, Dockerfile e todos os notebooks
> executados. Obrigado."

---

## Checklist antes de gravar

- [ ] Ambiente virtual ativado e todos os notebooks executados com outputs visiveis
- [ ] Resolucao de tela adequada (recomendado: 1920x1080)
- [ ] Microfone testado
- [ ] Notificacoes do sistema desativadas
- [ ] Abas do navegador/terminal organizadas na ordem do roteiro
