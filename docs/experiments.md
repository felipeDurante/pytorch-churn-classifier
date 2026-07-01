# Experimentos e MLflow

## Objetivo

Documentar como os experimentos são executados, registrados e comparados no projeto.

## Ferramenta de tracking

O projeto utiliza MLflow para rastrear parâmetros, métricas e artefatos dos treinamentos.

## Como iniciar o tracking local

O servidor local é iniciado pelo script `scripts/start_mlflow.py`, que sobe um backend SQLite em `mlflow.db` e armazena artefatos em `mlartifacts/`.

## O que é registrado

### Parâmetros

* `learning_rate`
* `batch_size`
* `epochs`
* `optimizer`
* parâmetros de treino e avaliação do pipeline

### Métricas

* `accuracy`
* `precision`
* `recall`
* `f1`
* `roc_auc`
* `pr_auc`
* `cost_saved`

### Artefatos

* modelo serializado
* classification report
* confusion matrix
* model card por fold
* gráficos de comparação

## Estratégia experimental

Os modelos são comparados com validação cruzada estratificada e a seleção final considera desempenho estatístico e custo de negócio.

## Resultados consolidados

A comparação atual foi salva em `models/model_comparison_results.csv`, com destaque para:

* Logistic Regression como baseline forte
* MLP como melhor desempenho agregado em ROC-AUC, PR-AUC e Cost Saved

## Boas práticas adotadas

* uso de logging estruturado
* rastreabilidade dos artefatos por execução
* persistência dos modelos treinados
* comparação entre baseline simples e modelo neural

## Relação com a documentação

Este documento complementa a seção de MLflow do README e a avaliação consolidada em `docs/model_evaluation.md`.