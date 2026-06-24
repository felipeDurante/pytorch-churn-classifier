# Tech Challenge — Predição de Churn com Machine Learning

## Objetivo

Este projeto tem como objetivo prever a probabilidade de cancelamento (Churn) de clientes de uma operadora de telecomunicações utilizando técnicas de Machine Learning.

A identificação antecipada de clientes com risco de evasão permite ações preventivas de retenção, reduzindo perdas financeiras e aumentando a fidelização dos clientes.

---

## Problema de Negócio

A perda de clientes impacta diretamente a receita da empresa. O objetivo deste projeto é construir um modelo preditivo capaz de identificar clientes com maior probabilidade de cancelamento, permitindo que a empresa direcione campanhas e ações de retenção de forma mais eficiente.

---

## Dataset

**Telco Customer Churn**

Características do dataset:

* Problema: Classificação Binária
* Variável alvo: Churn
* Domínio: Telecomunicações
* Objetivo: Prever cancelamento de clientes

---

## Machine Learning Canvas

O planejamento inicial do projeto está documentado em:

`docs/Machine Learning Canvas (v1.2).docx`

---

## Arquitetura da Solução

Fluxo da solução:

Dados Brutos
→ EDA
→ Pré-processamento
→ Engenharia de Features
→ Treinamento
→ Avaliação
→ MLflow
→ Modelo Final

---

## Modelos Avaliados

### Baselines

* Dummy Classifier
* Logistic Regression

### Redes Neurais

* Multi Layer Perceptron (MLP) utilizando PyTorch

---

## Estratégia de Validação

* Stratified K-Fold Cross Validation
* 5 Folds
* Seed fixa para reprodutibilidade

---

## Métricas Avaliadas

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* PR-AUC
* Cost Saved

---

## Estrutura do Projeto

```text
projeto-tech-chalenge/
├── data/
│   ├── raw/
│   └── processed/
├── docs/
├── models/
├── notebooks/
├── src/
│   └── projeto_ml/
├── tests/
├── README.md
└── pyproject.toml
```

---

## Setup

### Clone do projeto

```bash
git clone <URL_DO_REPOSITORIO>
cd projeto-tech-chalenge
```

### Ambiente virtual

Windows:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Instalação

```bash
pip install -U pip
pip install -e ".[dev]"
```

---

## Execução

### Treinamento

```bash
python -m projeto_ml.train
```

### Predição

```bash
python -m projeto_ml.predict
```

### Testes

```bash
pytest
```

---

## MLflow

O projeto utiliza MLflow para:

* Registro de experimentos
* Versionamento de modelos
* Armazenamento de métricas
* Gerenciamento de artefatos

### Iniciar servidor local

```bash
mlflow server ^
--backend-store-uri sqlite:///mlflow.db ^
--default-artifact-root ./mlartifacts ^
--host 127.0.0.1 ^
--port 5000
```

Interface:

```text
http://127.0.0.1:5000
```

---

## Notebooks

### EDA

`notebooks/01_eda_telco_churn.ipynb`

---

## Resultados

Os resultados completos dos experimentos estão documentados em:

`docs/model_evaluation.md`

---

## Tecnologias Utilizadas

* Python
* Scikit-Learn
* PyTorch
* Pandas
* NumPy
* Matplotlib
* Seaborn
* MLflow
* Pytest

---

## Próximos Passos

* Hyperparameter Tuning
* Deploy via API
* Containerização com Docker
* Monitoramento de Modelos
* Automação do Pipeline

---

## Autor

Projeto desenvolvido para o Tech Challenge da Pós-Graduação em Machine Learning.
