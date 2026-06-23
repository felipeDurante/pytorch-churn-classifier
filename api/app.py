"""FastAPI app para inferência do modelo salvo."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from projeto_ml.predict import load_model, predict

MODEL_PATH = Path("models/baseline_pipeline.joblib")

app = FastAPI(title="Telco Churn Inference API")


class PredictRequest(BaseModel):
    records: list[dict] = Field(
        ...,
        example=[
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
                "Churn Reason": "No reason",
            }
        ],
    )


class PredictResponse(BaseModel):
    prediction: int
    probability: float


def _normalize_numerics(df: pd.DataFrame) -> pd.DataFrame:
    numeric_columns = [
        "Count",
        "Latitude",
        "Longitude",
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        "Churn Value",
        "Churn Score",
        "CLTV",
    ]

    for column in numeric_columns:
        if column not in df.columns:
            continue

        values = df[column].astype(str).str.replace(",", ".", regex=False)
        try:
            df[column] = pd.to_numeric(values, errors="raise")
        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Coluna '{column}' deve conter valores numéricos válidos. "
                    f"Erro ao converter: {exc}"
                ),
            )

    return df


@app.on_event("startup")
def load():
    if not MODEL_PATH.exists():
        raise RuntimeError("Modelo não encontrado, treine o modelo antes de subir a API")
    app.state.model = load_model(MODEL_PATH)


@app.post("/predict", response_model=list[PredictResponse])
def infer(req: PredictRequest):
    try:
        df = pd.DataFrame(req.records)
    except Exception as exc:  # pragma: no cover - input validation
        raise HTTPException(status_code=400, detail=str(exc))

    df = _normalize_numerics(df)
    preds = predict(app.state.model, df)
    results = []
    for _, r in preds.iterrows():
        pred_val = int(r.prediction)
        prob_val = float(r.probability)
        results.append(PredictResponse(prediction=pred_val, probability=prob_val))

    return results
