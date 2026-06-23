"""Inferência: carregar pipeline salvo e predizer."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd


def load_model(path: str | Path) -> Any:
    """Carrega pipeline salvo com joblib."""
    return joblib.load(path)


def predict(model: Any, X: pd.DataFrame) -> pd.DataFrame:
    """Retorna previsões e probabilidades para `X`.

    Retorna um DataFrame com colunas `prediction` e `probability`.
    """
    proba = model.predict_proba(X)[:, 1]
    pred = model.predict(X)
    return pd.DataFrame({"prediction": pred.astype(int), "probability": proba})


def main() -> None:  # pragma: no cover - quick demo
    from projeto_ml.data import load_telco_churn

    model = load_model("models/baseline_pipeline.joblib")
    df = load_telco_churn("data/raw/Telco_customer_churn(Telco_Churn).csv")
    X = df.drop(columns=["Churn Value"])  # type: ignore[list-index]
    out = predict(model, X.head())
    print(out)


if __name__ == "__main__":
    main()
"""Carregamento do modelo e geração de previsões."""


def main() -> None:
    """Executa a etapa de inferência."""
    print("Etapa de inferência ainda não implementada.")


if __name__ == "__main__":
    main()

