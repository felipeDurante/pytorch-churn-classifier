import pandas as pd
from pathlib import Path

from projeto_ml.train import train


def test_train_returns_metrics_and_persists_model(tmp_path, monkeypatch):
    csv_path = tmp_path / "mini_churn.csv"
    df = pd.DataFrame(
        {
            "CustomerID": [1, 2, 3, 4, 5, 6, 7, 8],
            "Tenure Months": [1, 2, 3, 4, 5, 6, 7, 8],
            "Monthly Charges": [20.0, 30.0, 40.0, 50.0, 20.0, 30.0, 40.0, 50.0],
            "Total Charges": [20.0, 60.0, 120.0, 200.0, 20.0, 60.0, 120.0, 200.0],
            "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month", "One year", "Two year", "Month-to-month", "One year"],
            "Churn Label": ["No", "No", "Yes", "Yes", "No", "No", "Yes", "Yes"],
            "Churn Value": [0, 0, 1, 1, 0, 0, 1, 1],
        }
    )
    df.to_csv(csv_path, sep=";", index=False)

    monkeypatch.setattr("projeto_ml.train._maybe_import_mlflow", lambda: None)
    model_path = tmp_path / "model.joblib"

    result = train(
        csv_path,
        model_out=model_path,
        n_splits=2,
        metric="roc_auc",
        cost_per_churn=50.0,
        intervention_success=0.5,
        models=["dummy"],
    )

    assert Path(result["model_path"]).exists()
    assert model_path.exists()
    metrics = result["metrics"]
    assert "dummy_f1" in metrics
    assert "dummy_roc_auc" in metrics
    assert "dummy_pr_auc" in metrics
    assert "dummy_cost_saved" in metrics
    assert metrics["dummy_cost_saved"] >= 0.0
