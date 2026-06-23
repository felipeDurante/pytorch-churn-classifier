"""Treinamento e avaliação do modelo.

Este script implementa um pipeline mínimo de treinamento com:
- Carregamento dos dados via `data.load_telco_churn`
- Pré-processamento via `features.build_preprocessor`
- Validação cruzada estratificada (StratifiedKFold)
- Baselines `DummyClassifier` e `LogisticRegression`
- Logging estruturado e integração com MLflow quando disponível
- Cálculo de métrica de negócio (custo de churn evitado)
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
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
    metric: str = "roc_auc",
    cost_per_churn: float = 100.0,
    intervention_success: float = 0.3,
    models: list[str] | None = None,
) -> dict[str, Any]:
    logging.basicConfig(level=logging.INFO)
    config.set_seed()

    mlflow = _maybe_import_mlflow()
    if mlflow is not None:
        mlflow.start_run()

    df = data_mod.load_telco_churn(dataset_path, drop_leakage=True)
    target = data_mod.TARGET_COLUMN
    X_df = df.drop(columns=[target])
    y = df[target].astype(int).to_numpy()

    # compute dataset digest/version and log as tag
    try:
        dataset_bytes = Path(dataset_path).read_bytes()
        dataset_digest = hashlib.sha256(dataset_bytes).hexdigest()
    except Exception:
        dataset_digest = "unknown"

    pipeline = features.build_preprocessor(X_df)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=config.SEED)
    if models is None:
        models = ["dummy", "logistic"]

    metrics: dict[str, list[float]] = {}

    def _score_fold(y_true, y_pred, y_proba) -> dict[str, float]:
        out: dict[str, float] = {}
        out["f1"] = float(f1_score(y_true, y_pred))
        try:
            out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            out["roc_auc"] = float("nan")
        try:
            out["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            out["pr_auc"] = float("nan")
        # business metric: custo economizado assumindo apenas intervenções sobre TP
        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        out["cost_saved"] = float(tp * cost_per_churn * intervention_success)
        return out

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_df, y), start=1):
        LOGGER.info("start_fold", extra={"fold": fold})
        X_train, X_val = X_df.iloc[train_idx], X_df.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        X_train_trans = pipeline.fit_transform(X_train)
        X_val_trans = pipeline.transform(X_val)

        # Train and evaluate requested models for this fold
        for model_name in models:
            if model_name == "dummy":
                clf = DummyClassifier(strategy="most_frequent", random_state=config.SEED)
            elif model_name == "logistic":
                clf = LogisticRegression(max_iter=1000, random_state=config.SEED)
            else:
                raise ValueError(f"Unknown model: {model_name}")

            clf.fit(X_train_trans, y_train)
            y_pred = clf.predict(X_val_trans)
            # prefer predict_proba when available
            if hasattr(clf, "predict_proba"):
                y_proba = clf.predict_proba(X_val_trans)[:, 1]
            else:
                # fallback to decision_function or constant
                try:
                    y_proba = clf.decision_function(X_val_trans)
                except Exception:
                    y_proba = y_pred

            fold_scores = _score_fold(y_val, y_pred, y_proba)
            # accumulate under model-specific keys
            for k, v in fold_scores.items():
                metrics_key = f"{model_name}_{k}"
                metrics.setdefault(metrics_key, []).append(float(v))

            LOGGER.info(
                "fold_metrics",
                extra={"fold": fold, "model": model_name, **fold_scores},
            )

    # Fit final logistic model on full data and save
    LOGGER.info("fitting_final_model")
    X_full_trans = pipeline.fit_transform(X_df)
    final_clf = LogisticRegression(max_iter=1000, random_state=config.SEED)
    final_clf.fit(X_full_trans, y)

    full_pipeline = Pipeline([("preprocessor", pipeline), ("classifier", final_clf)])

    out_path = Path(model_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    import joblib

    joblib.dump(full_pipeline, out_path)

    # Log results to MLflow if available
    if mlflow is not None:
        mlflow.log_param("model_type", "logistic_regression")
        mlflow.set_tag("dataset_digest", dataset_digest)
        mlflow.log_param("metric", metric)
        mlflow.log_param("cost_per_churn", float(cost_per_churn))
        mlflow.log_param("intervention_success", float(intervention_success))

        # aggregate metrics per model
        metrics_to_log: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                metrics_to_log[f"{k}_mean"] = float(np.nanmean(v))
            except Exception:
                metrics_to_log[f"{k}_mean"] = float("nan")

        mlflow.log_metrics(metrics_to_log)
        mlflow.log_artifact(str(out_path))
        mlflow.end_run()

    return {
        "model_path": str(out_path),
        "metrics": {k: float(np.nanmean(v)) for k, v in metrics.items()},
    }


def main() -> None:  # pragma: no cover - integration entrypoint
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Telco_customer_churn(Telco_Churn).csv")
    parser.add_argument("--out", default="models/baseline_pipeline.joblib")
    parser.add_argument("--metric", default="roc_auc", choices=["roc_auc", "pr_auc", "f1"])
    parser.add_argument("--cost-per-churn", default=100.0, type=float)
    parser.add_argument("--intervention-success", default=0.3, type=float)
    parser.add_argument("--models", nargs="+", default=["dummy", "logistic"], help="models to train: dummy, logistic")
    args = parser.parse_args()

    result = train(
        args.data,
        args.out,
        metric=args.metric,
        cost_per_churn=args.cost_per_churn,
        intervention_success=args.intervention_success,
        models=args.models,
    )
    LOGGER.info("training_complete", extra=result)


if __name__ == "__main__":
    main()

