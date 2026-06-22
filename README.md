# Projeto Tech Challenge — Machine Learning

Estrutura-base para desenvolvimento de um projeto de Machine Learning em Python, com separação clara entre código-fonte, dados, modelos, experimentos, testes e documentação.

## Machine Learning Canvas

O planejamento inicial do projeto está documentado no [Machine Learning Canvas (v1.2)](docs/Machine%20Learning%20Canvas%20%28v1.2%29.docx).

## Estrutura do projeto

```text
projeto-tech-chalenge/
├── data/
│   ├── raw/                 # Dados originais, sem alterações
│   └── processed/           # Dados tratados e prontos para modelagem
├── docs/                    # Documentação complementar
├── models/                  # Modelos treinados e artefatos
├── notebooks/               # Análises exploratórias e experimentos
├── src/
│   └── projeto_ml/
│       ├── data.py          # Carregamento e preparação de dados
│       ├── features.py      # Engenharia de atributos
│       ├── train.py         # Treinamento do modelo
│       └── predict.py       # Geração de previsões
├── tests/                   # Testes automatizados
├── .gitignore
├── pyproject.toml
└── README.md
```

## Pré-requisitos

- Python 3.11 ou superior
- Git

## Setup

### 1. Clone o repositório

```bash
git clone <URL_DO_REPOSITORIO>
cd projeto-tech-chalenge
```

### 2. Crie e ative um ambiente virtual

No Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

No Linux ou macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Instale o projeto e as dependências

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

## Dados

Coloque os arquivos originais em `data/raw/`. Dados tratados devem ser gravados em `data/processed/`.

Os arquivos de dados e modelos treinados não são versionados pelo Git, evitando o armazenamento acidental de arquivos grandes ou informações sensíveis. Apenas os diretórios vazios são mantidos no repositório.

## Execução

O fluxo inicial está dividido em módulos independentes:

```bash
python -m projeto_ml.data
python -m projeto_ml.features
python -m projeto_ml.train
python -m projeto_ml.predict
```

Os módulos são pontos de entrada mínimos e deverão ser adaptados quando a fonte de dados, o problema de negócio, as métricas e o algoritmo forem definidos.

## Notebooks

Inicie o Jupyter Lab com:

```bash
jupyter lab
```

Use notebooks para exploração e experimentação. Código reutilizável e definitivo deve ser movido para `src/projeto_ml/`.

### Análise exploratória

O EDA inicial do dataset Telco Customer Churn está disponível em
[`notebooks/01_eda_telco_churn.ipynb`](notebooks/01_eda_telco_churn.ipynb).

## Testes

Execute a suíte:

```bash
pytest
```

Com relatório de cobertura:

```bash
pytest --cov=projeto_ml --cov-report=term-missing
```

## Qualidade de código

Verifique o código:

```bash
ruff check .
```

Formate o código:

```bash
ruff format .
```

## Fluxo recomendado

1. Armazenar os dados originais em `data/raw/`.
2. Explorar e validar os dados em `notebooks/`.
3. Implementar preparação e engenharia de atributos em `src/projeto_ml/`.
4. Treinar e avaliar modelos.
5. Salvar os artefatos aprovados em `models/`.
6. Criar testes para as regras de transformação e inferência.
7. Documentar decisões, métricas e limitações em `docs/`.

## Sobre o projeto

Este repositório é o ponto de partida para o Tech Challenge. A descrição do problema, a origem dos dados, a variável-alvo, as métricas de avaliação e os resultados serão detalhados conforme o desenvolvimento avançar.
