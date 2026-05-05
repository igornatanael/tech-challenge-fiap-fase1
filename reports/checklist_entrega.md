# Checklist de Entrega — Tech Challenge Fase 1

## Codigo e estrutura
- [ ] Repositorio Git criado e publico (ou link compartilhavel)
- [ ] Todos os arquivos commitados (exceto .venv, __pycache__, data/raw se grande)
- [ ] README.md completo com instrucoes de execucao
- [ ] requirements.txt atualizado
- [ ] Dockerfile funcional
- [ ] .gitignore cobrindo .venv, __pycache__, *.pyc, .DS_Store

## Dados
- [ ] Dataset em data/raw/maternal_health_risk.csv (ou link de download no README)

## Notebooks executados e com outputs salvos
- [ ] 01_eda.ipynb — executado, graficos visiveis, secao 9 preenchida
- [ ] 02_modeling.ipynb — executado, resultados da CV e teste visiveis, secao 9 preenchida
- [ ] 03_evaluation.ipynb — executado, curvas ROC e SHAP visiveis, secao 5 preenchida

## Graficos gerados em reports/figures/
- [ ] 01_target_distribution.png
- [ ] 02_feature_distributions.png
- [ ] 03_boxplots_by_risk.png
- [ ] 04_correlation_matrix.png
- [ ] 05_pairplot.png
- [ ] 06_cv_f1_macro.png
- [ ] 07_confusion_logistic_regression.png
- [ ] 07_confusion_random_forest.png
- [ ] 07_confusion_svm.png
- [ ] 08_metrics_heatmap.png
- [ ] 09_feature_importance_rf.png
- [ ] 10_roc_logistic_regression.png
- [ ] 10_roc_random_forest.png
- [ ] 10_roc_svm.png
- [ ] 11_shap_summary_rf_high.png
- [ ] 12_shap_bar_rf_high.png
- [ ] 13_shap_waterfall_rf.png
- [ ] 14_shap_bar_lr_high.png

## Relatorio tecnico
- [ ] reports/relatorio_tecnico.md preenchido com resultados reais
- [ ] Secoes 5 (resultados) e 7 (conclusao) preenchidas apos execucao dos notebooks
- [ ] Graficos referenciados inseridos (ou link para figures/)

## Video de demonstracao
- [ ] Gravado (ate 15 minutos)
- [ ] Publicado no YouTube (nao listado) ou Vimeo
- [ ] Link funcionando e acessivel

## PDF final de entrega (o que enviar na plataforma FIAP)
- [ ] Link do repositorio Git
- [ ] Link do video
- [ ] Resumo dos resultados principais (tabela comparativa dos 3 modelos)
- [ ] PDF gerado e revisado

## Revisao final
- [ ] Executar todos os notebooks do zero (Restart Kernel + Run All) e confirmar que nao ha erros
- [ ] Conferir que nenhum resultado no relatorio contradiz os outputs dos notebooks
- [ ] Confirmar autoria correta em todos os notebooks e no relatorio
