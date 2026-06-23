"""Treinamento e avaliação do modelo.

Este script implementa um pipeline mínimo de treinamento com:
- Carregamento dos dados via `data.load_telco_churn`
- Pré-processamento via `features.build_preprocessor`
- Validação cruzada estratificada (StratifiedKFold)
- Baseline `LogisticRegression` (substituível por PyTorch MLP)
- Logging estruturado e integração com MLflow quando disponível
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from projeto_ml import config, features
from projeto_ml import data as data_mod

LOGGER = logging.getLogger(__name__)


def _maybe_import_mlflow():
    try:
        import mlflow

        return mlflow
    except Exception:
        return None


def train(
    dataset_path: str | Path,
    model_out: str | Path = Path("models/baseline_pipeline.joblib"),
    n_splits: int = 5,
) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    config.set_seed()

    mlflow = _maybe_import_mlflow()
    if mlflow is not None:
        mlflow.start_run()

    df = data_mod.load_telco_churn(dataset_path)
    target = data_mod.TARGET_COLUMN
    X_df = df.drop(columns=[target])
    y = df[target].astype(int).to_numpy()

    pipeline = features.build_preprocessor(X_df)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    metrics: dict[str, list[float]] = {"f1": [], "roc_auc": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y), start=1):
        LOGGER.info("start_fold", extra={"fold": fold})
        X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_trans = pipeline.fit_transform(X_train)
        X_val_trans = pipeline.transform(X_val)

        clf = LogisticRegression(max_iter=1000, random_state=config.SEED)
        clf.fit(X_train_trans, y_train)

        y_pred = clf.predict(X_val_trans)
        y_proba = clf.predict_proba(X_val_trans)[:, 1]

        f1 = f1_score(y_val, y_pred)
        roc = roc_auc_score(y_val, y_proba)

        metrics["f1"].append(float(f1))
        metrics["roc_auc"].append(float(roc))

        LOGGER.info(
            "fold_metrics",
            extra={"fold": fold, "f1": f1, "roc_auc": roc},
        )

    # Fit final pipeline on full data
    LOGGER.info("fitting_final_model")
    X_full_trans = pipeline.fit_transform(X_df)
    final_clf = LogisticRegression(max_iter=1000, random_state=config.SEED)
    final_clf.fit(X_full_trans, y)

    full_pipeline = Pipeline([("preprocessor", pipeline), ("classifier", final_clf)])

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(full_pipeline, out_path)

    if mlflow is not None:
        mlflow.log_param("model_type", "logistic_regression")
        metrics_to_log = {
            "f1_mean": float(np.mean(metrics["f1"])),
            "roc_auc_mean": float(np.mean(metrics["roc_auc"])),
        }
        mlflow.log_metrics(metrics_to_log)
        mlflow.log_artifact(str(out_path))
        mlflow.end_run()

    return {
        "model_path": str(out_path),
        "metrics": {k: float(np.mean(v)) for k, v in metrics.items()},
    }


def main() -> None:  # pragma: no cover - integration entrypoint
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Telco_customer_churn(Telco_Churn).csv")
    parser.add_argument("--out", default="models/baseline_pipeline.joblib")
    args = parser.parse_args()

    result = train(args.data, args.out)
    LOGGER.info("training_complete", extra=result)


if __name__ == "__main__":
    main()

