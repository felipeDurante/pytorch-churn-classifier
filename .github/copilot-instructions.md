# Tech Challenge - Diretrizes do Projeto

## Objetivo
Construir uma solução profissional end-to-end para previsão de churn utilizando boas práticas de Engenharia de Machine Learning.

# Arquitetura

Implementar uma arquitetura modular e desacoplada.

Separar responsabilidades em:

- api/
- config/
- data/
- features/
- models/
- pipelines/
- services/
- utils/

Evitar lógica de negócio dentro da API.

A API deve apenas receber requisições e delegar processamento aos services.

---

# Convenções de código

- Utilizar Python >= 3.12.
- Utilizar type hints em todas as funções.
- Utilizar dataclasses ou Pydantic para schemas.
- Funções devem possuir responsabilidade única.
- Evitar funções maiores que 50 linhas.
- Evitar duplicação de código.
- Utilizar docstrings no padrão Google.
- Não utilizar variáveis globais.

---

# Logging

Não utilizar print().

Sempre utilizar logging estruturado:

- logging.getLogger(__name__)
- logger.info()
- logger.warning()
- logger.error()

Logs devem conter contexto suficiente para depuração.

---

# Configurações

Não hardcodar valores.

Centralizar configurações em:

src/config/settings.py

Variáveis configuráveis:

- random_seed
- batch_size
- learning_rate
- epochs
- test_size
- mlflow_tracking_uri
- model_path

Utilizar variáveis de ambiente sempre que possível.

---

# Reprodutibilidade

Sempre fixar seeds para:

- random
- numpy
- torch

Habilitar comportamento determinístico do PyTorch.

Resultados devem ser reproduzíveis.

---

# Pipeline de dados

Utilizar Pipeline do Scikit-Learn.

Separar:

- ingestão
- limpeza
- feature engineering
- transformação
- treinamento

Evitar processamento diretamente no notebook.

Notebooks são apenas para exploração.

---

# Feature Engineering

Salvar transformadores treinados.

Não recalcular transformações em produção.

Persistir encoders e scalers.

---

# Modelos

Implementar:

1. Baseline com Regressão Logística
2. Baseline com Random Forest
3. Rede Neural MLP em PyTorch

Comparar resultados entre os modelos.

Não assumir que a rede neural será a melhor solução.

---

# Validação

Utilizar Stratified K-Fold Cross Validation.

Avaliar:

- Accuracy
- Precision
- Recall
- F1-score
- ROC AUC

Registrar média e desvio padrão.

Evitar avaliar apenas em um split train/test.

---

# MLflow

Registrar em todos os experimentos:

Parâmetros:

- learning_rate
- batch_size
- epochs
- optimizer

Métricas:

- accuracy
- precision
- recall
- f1
- roc_auc

Artefatos:

- confusion_matrix.png
- classification_report.txt
- model_card.md

Salvar o modelo treinado no MLflow.

Nenhum treinamento deve ocorrer sem tracking.

---

# PyTorch

Utilizar:

- Dataset
- DataLoader
- nn.Module

Implementar:

- Early stopping
- Checkpoint do melhor modelo

Separar:

- treinamento
- validação
- inferência

Evitar código procedural em um único arquivo.

---

# FastAPI

Separar:

api/
    routers/
    schemas/
    dependencies/

Criar endpoints:

GET /health

POST /predict

GET /model-info

Utilizar Pydantic para request e response.

Adicionar tratamento de exceções.

Retornar mensagens claras.

---

# Persistência

Salvar:

- modelo
- scaler
- encoder

Utilizar joblib ou MLflow.

Nunca retreinar durante inferência.

---

# Testes

Utilizar pytest.

Implementar:

test_smoke.py

- Verifica importação dos módulos.

test_schema.py

- Verifica schemas da API.

test_api.py

- Testa endpoint /health.
- Testa endpoint /predict.

test_model.py

- Verifica saída do modelo.

test_pipeline.py

- Verifica pipeline de pré-processamento.

Cobertura mínima de 80%.

---

# Ruff

Executar antes dos commits:

ruff check .
ruff format .

Nenhum warning é aceitável.

---

# Git

Commits pequenos, significativos e com mensagens claras.

Exemplos:

feat: create preprocessing pipeline

feat: implement mlp model

feat: add fastapi prediction endpoint

test: add api tests

docs: add model card

refactor: separate training service

Evitar commits genéricos:

update
fixes
ajustes

---

# Documentação

Manter:

README.md

docs/model_card.md

docs/architecture.md

docs/experiments.md

---

# Model Card

Documentar:

- Objetivo do modelo.
- Fonte dos dados.
- Features utilizadas.
- Métricas.
- Limitações.
- Possíveis vieses.
- Cenários em que o modelo não deve ser utilizado.

---

# Ao gerar código

Sempre:

- produzir código de qualidade de produção;
- adicionar type hints;
- adicionar logs;
- criar funções pequenas;
- seguir princípios SOLID;
- criar testes quando necessário;
- evitar soluções simplificadas;
- evitar código de notebook;
- priorizar legibilidade, manutenção e reprodutibilidade.