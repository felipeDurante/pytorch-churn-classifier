"""FastAPI app para inferência do modelo salvo."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from projeto_ml.predict import load_model, predict

MODEL_PATH = Path("models/baseline_pipeline.joblib")

app = FastAPI(title="Telco Churn Inference API")


class PredictRequest(BaseModel):
    records: list[dict]


class PredictResponse(BaseModel):
    prediction: int
    probability: float


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

    preds = predict(app.state.model, df)
    results = []
    for _, r in preds.iterrows():
        pred_val = int(r.prediction)
        prob_val = float(r.probability)
        results.append(PredictResponse(prediction=pred_val, probability=prob_val))

    return results
