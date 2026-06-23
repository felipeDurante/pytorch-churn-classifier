# Fluxo de execução do projeto

Este documento descreve o fluxo completo para ativar o ambiente, instalar dependências, treinar o modelo, subir o servidor e validar a API.

## 1) Ativar o ambiente virtual
No PowerShell, dentro da pasta do projeto:

```powershell
cd C:\felipe\fiap\pytorch-churn-classifier
.\.venv\Scripts\Activate.ps1
```

Se houver restrição de política de execução:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

Verifique se o ambiente está ativo pelo prompt:

```powershell
(.venv) PS C:\felipe\fiap\pytorch-churn-classifier>
```

## 2) Instalar dependências
Depois de ativar o ambiente, instale todas as dependências do projeto:

```powershell
python -m pip install -e .
```

Isso usa o `pyproject.toml` para instalar o projeto e seus pacotes necessários.

## 3) Treinar o modelo
Sempre treine antes de subir o servidor, porque a API carrega o modelo salvo em `models/baseline_pipeline.joblib`.

```powershell
python src\projeto_ml\train.py --data "data/raw/Telco_customer_churn(Telco_Churn).csv" --out models/baseline_pipeline.joblib
```

Se quiser usar métricas diferentes ou parâmetros personalizados:

```powershell
python src\projeto_ml\train.py --data "data/raw/Telco_customer_churn(Telco_Churn).csv" --out models/baseline_pipeline.joblib --metric pr_auc --cost-per-churn 120 --intervention-success 0.25 --models dummy logistic
```

## 4) Subir o servidor FastAPI
Com o modelo treinado e salvo, inicie o servidor:

```powershell
.\.venv\Scripts\uvicorn.exe api.app:app --reload --host 127.0.0.1 --port 8000
```

A API ficará disponível em:

- `http://127.0.0.1:8000`
- documentação automática: `http://127.0.0.1:8000/docs`

## 5) Testar o endpoint `/predict`
Use a documentação Swagger em `/docs` ou faça uma requisição `POST` com um body JSON válido.

Exemplo de payload:

```json
{
  "records": [
    {
      "CustomerID": "0001",
      "Count": 1,
      "Country": "United States",
      "State": "California",
      "City": "Los Angeles",
      "Zip Code": "90003",
      "Lat Long": "33.964131, -118.272783",
      "Latitude": 33.964131,
      "Longitude": -118.272783,
      "Gender": "Male",
      "Senior Citizen": "No",
      "Partner": "No",
      "Dependents": "No",
      "Tenure Months": 12,
      "Phone Service": "Yes",
      "Multiple Lines": "No",
      "Internet Service": "DSL",
      "Online Security": "Yes",
      "Online Backup": "No",
      "Device Protection": "No",
      "Tech Support": "No",
      "Streaming TV": "No",
      "Streaming Movies": "No",
      "Contract": "Month-to-month",
      "Paperless Billing": "Yes",
      "Payment Method": "Mailed check",
      "Monthly Charges": 70.5,
      "Total Charges": 846.0,
      "Churn Label": "No",
      "Churn Value": 0,
      "Churn Score": 50,
      "CLTV": 1000,
      "Churn Reason": "No reason"
    }
  ]
}
```

## 6) Verificar o MLflow UI
Se quiser visualizar os experimentos e métricas:

```powershell
python -m mlflow ui --host 127.0.0.1 --port 5000
```

Depois abra:

- `http://127.0.0.1:5000`

### Observação
O MLflow só mostra runs se o treinamento tiver sido executado com `mlflow` disponível no ambiente virtual. Se não houver runs, instale o pacote com:

```powershell
python -m pip install mlflow
```

## 7) Fluxo ideal resumido
1. Ativar `.venv`
2. Instalar dependências (`python -m pip install -e .`)
3. Treinar o modelo
4. Subir o servidor FastAPI
5. Testar `/predict`
6. (Opcional) Ver MLflow UI
