# Arquitetura da Solução

## Objetivo

Documentar a arquitetura usada no Tech Challenge, preservando a estrutura modular atual do projeto e o fluxo end-to-end de treinamento, registro e inferência.

## Visão Geral

A solução está organizada em camadas desacopladas:

* ingestão e preparação dos dados
* engenharia de features e transformação
* treinamento e avaliação dos modelos
* registro de experimentos com MLflow
* inferência em tempo real via FastAPI

O código principal segue a separação já existente entre `api/`, `src/`, `scripts/`, `models/`, `docs/` e `tests/`.

## Fluxo Funcional

1. Os dados brutos são carregados da base em `data/raw/`.
2. O pipeline de pré-processamento transforma os dados para treino e inferência.
3. Os modelos são treinados com validação cruzada estratificada.
4. Os experimentos, métricas e artefatos são registrados no MLflow.
5. O melhor pipeline é serializado em `models/baseline_pipeline.joblib`.
6. A API FastAPI carrega o pipeline e expõe o endpoint `/predict`.

## Componentes Principais

### Treinamento

O treinamento está concentrado em `src/projeto_ml/train.py` e utiliza:

* `StratifiedKFold`
* `DummyClassifier`
* `LogisticRegression`
* `MLPClassifier` em PyTorch

### Pré-processamento

O pré-processamento é montado como pipeline em `src/projeto_ml/features.py`, evitando lógica de transformação duplicada entre treino e inferência.

### Inferência

A inferência em tempo real é fornecida por `api/app.py`, que carrega o artefato salvo e responde requisições síncronas.

### Persistência de artefatos

Os artefatos principais são:

* `models/baseline_pipeline.joblib`
* `models/model_comparison_results.csv`
* artefatos versionados pelo MLflow em `mlartifacts/`

## Decisões de Deploy

O projeto adota arquitetura real-time porque o caso de uso exige resposta sob demanda para um registro por vez ou pequenos lotes.

Batch scoring continua viável para uso futuro, mas não é o fluxo principal documentado neste desafio.

## Reprodutibilidade

O projeto centraliza seed e comportamento determinístico no pacote de código para reduzir variação entre execuções.

## Relação com o Tech Challenge

Este documento complementa a visão geral do README e serve como base para a justificativa de arquitetura exigida no desafio.