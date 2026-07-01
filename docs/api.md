# Documentação da API

## Objetivo

Disponibilizar o modelo treinado para inferência em tempo real com FastAPI.

## Execução

O serviço é implementado em `api/app.py` e carrega o pipeline salvo em `models/baseline_pipeline.joblib` no startup da aplicação.

## Swagger

FastAPI expõe documentação automática nos caminhos padrão:

* `/docs` para Swagger UI
* `/redoc` para ReDoc

## Endpoints

### GET /health

Retorna o estado da API e indica se o modelo foi carregado.

Exemplo de resposta:

```json
{
  "status": "ok",
  "model_loaded": true,
  "model_path": "models/baseline_pipeline.joblib"
}
```

### POST /predict

Recebe registros em `records` e retorna previsão e probabilidade para cada item.

Exemplo de request:

```json
{
  "records": [
    {
      "CustomerID": "0001",
      "Count": 1,
      "Country": "United States",
      "State": "California",
      "City": "Los Angeles",
      "Latitude": 33.964131,
      "Longitude": -118.272783,
      "Tenure Months": 12,
      "Monthly Charges": 70.5,
      "Total Charges": 846.0,
      "Churn Value": 0,
      "Churn Score": 50,
      "CLTV": 1000
    }
  ]
}
```

Exemplo de resposta:

```json
[
  {
    "prediction": 0,
    "probability": 0.1234
  }
]
```

## Validação de entrada

O endpoint normaliza colunas numéricas e rejeita valores inválidos com `HTTPException` 400.

## Tratamento de erros

Casos cobertos atualmente:

* falha ao converter colunas numéricas
* payload inválido ao converter a lista de registros em `DataFrame`
* ausência do modelo no startup da aplicação

## Observações

O contrato foi desenhado para manter a API simples: a aplicação só recebe requisições, carrega o pipeline salvo e delega a predição ao módulo de inferência.