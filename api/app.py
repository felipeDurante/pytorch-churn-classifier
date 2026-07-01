"""FastAPI app para inferência do modelo salvo."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request, Response
from pydantic import BaseModel, Field

from projeto_ml.predict import load_model, predict

MODEL_PATH = Path("models/baseline_pipeline.joblib")

app = FastAPI(title="Telco Churn Inference API")
LOGGER = logging.getLogger(__name__)


@app.middleware("http")
async def latency_middleware(request: Request, call_next) -> Response:
    start_time = time.perf_counter()
    response = await call_next(request)
    duration_ms = (time.perf_counter() - start_time) * 1000
    LOGGER.info(
        "request_completed",
        extra={
            "method": request.method,
            "path": request.url.path,
            "status_code": response.status_code,
            "duration_ms": round(duration_ms, 2),
        },
    )
    response.headers["X-Process-Time-ms"] = f"{duration_ms:.2f}"
    return response


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


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str


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
    LOGGER.info("model_loaded", extra={"model_path": str(MODEL_PATH)})


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=hasattr(app.state, "model"),
        model_path=str(MODEL_PATH),
    )


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

    LOGGER.info("prediction_completed", extra={"records": len(results)})

    return results
