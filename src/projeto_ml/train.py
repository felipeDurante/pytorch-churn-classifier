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
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from torch.utils.data import DataLoader

from projeto_ml import config, features
from projeto_ml import data as data_mod
from projeto_ml import eval as eval_mod
from projeto_ml.models import ChurnDataset, EarlyStopping, MLPClassifier
from datetime import datetime

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
        try:

            mlflow.set_tracking_uri(
                "http://127.0.0.1:5000"
            )

            mlflow.set_experiment(
                "telco_churn"
            )
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            mlflow.start_run(
                run_name=run_name
            )

        except Exception as e:

            LOGGER.warning(
                f"MLflow indisponível: {e}"
            )

            mlflow = None

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

    # def _score_fold(y_true, y_pred, y_proba) -> dict[str, float]:
    #     out: dict[str, float] = {}
    #     out["f1"] = float(f1_score(y_true, y_pred))
    #     try:
    #         out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    #     except Exception:
    #         out["roc_auc"] = float("nan")
    #     try:
    #         out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    #     except Exception:
    #         out["pr_auc"] = float("nan")
    #     # business metric: custo economizado assumindo apenas intervenções sobre TP
    #     tp = int(((y_true == 1) & (y_pred == 1)).sum())
    #     out["cost_saved"] = float(tp * cost_per_churn * intervention_success)
    #     return out

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
                clf.fit(X_train_trans, y_train)
                y_pred = clf.predict(X_val_trans)
                if hasattr(clf, "predict_proba"):
                    y_proba = clf.predict_proba(X_val_trans)[:, 1]
                else:
                    try:
                        y_proba = clf.decision_function(X_val_trans)
                    except Exception:
                        y_proba = y_pred
                # fold_scores = _score_fold(y_val, y_pred, y_proba)
                fold_scores = eval_mod.compute_metrics(
                    y_val,
                    y_pred,
                    y_proba,
                )
                cost_analysis = eval_mod.compute_cost_analysis(
                    y_val,
                    y_pred,
                    cost_per_churn=cost_per_churn,
                    intervention_success=intervention_success,
                )
                fold_scores.update(cost_analysis)
                
                for k, v in fold_scores.items():
                    metrics_key = f"{model_name}_{k}"
                    metrics.setdefault(metrics_key, []).append(float(v))
                # LOGGER.info(
                #     "fold_metrics",
                #     extra={"fold": fold, "model": model_name, **fold_scores},
                # )

            elif model_name == "logistic":
                clf = LogisticRegression(max_iter=1000, random_state=config.SEED)
                clf.fit(X_train_trans, y_train)
                y_pred = clf.predict(X_val_trans)
                if hasattr(clf, "predict_proba"):
                    y_proba = clf.predict_proba(X_val_trans)[:, 1]
                else:
                    try:
                        y_proba = clf.decision_function(X_val_trans)
                    except Exception:
                        y_proba = y_pred
                fold_scores = eval_mod.compute_metrics(
                    y_val,
                    y_pred,
                    y_proba,
                )
                cost_analysis = eval_mod.compute_cost_analysis(
                    y_val,
                    y_pred,
                    cost_per_churn=cost_per_churn,
                    intervention_success=intervention_success,
                )

                fold_scores.update(cost_analysis)

                for k, v in fold_scores.items():
                    metrics_key = f"{model_name}_{k}"
                    metrics.setdefault(metrics_key, []).append(float(v))
                # LOGGER.info(
                #     "fold_metrics",
                #     extra={"fold": fold, "model": model_name, **fold_scores},
                # )

            elif model_name == "mlp":
                result = train_mlp_fold(
                    X_train_trans,
                    y_train,
                    X_val_trans,
                    y_val,
                    epochs=50,
                    batch_size=32,
                    learning_rate=0.001,
                    hidden_layers=[128, 64, 32],
                    dropout_rate=0.3,
                    patience=10,
                )
                y_pred = result["y_pred"]
                y_proba = result["y_proba"]
                fold_scores = result["metrics"]
                cost_analysis = eval_mod.compute_cost_analysis(
                    y_val,
                    y_pred,
                    cost_per_churn=cost_per_churn,
                    intervention_success=intervention_success,
                )

                fold_scores.update(cost_analysis)
                
                for k, v in fold_scores.items():
                    metrics_key = f"{model_name}_{k}"
                    metrics.setdefault(metrics_key, []).append(float(v))
                # LOGGER.info(
                #     "fold_metrics",
                #     extra={"fold": fold, "model": model_name, **fold_scores},
                # )

            else:
                raise ValueError(f"Unknown model: {model_name}")

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
   
    metrics_mean = {
        k: float(np.nanmean(v))
        for k, v in metrics.items()
    }
    LOGGER.info("=== Mean Metrics ===")

    for metric_name, metric_value in sorted(metrics_mean.items()):
        LOGGER.info(
            f"{metric_name}: {metric_value:.6f}"
        )

    metrics_std = {
        k: float(np.nanstd(v))
        for k, v in metrics.items()
    }
    LOGGER.info("=== Std Metrics ===")

    for metric_name, metric_value in sorted(metrics_std.items()):
        LOGGER.info(
            f"{metric_name}: {metric_value:.6f}"
        )

    return {
        "model_path": str(out_path),
        "metrics": metrics,
        "metrics_mean": metrics_mean,
        "metrics_std": metrics_std
    }
    # return {
    #     "model_path": str(out_path),
    #     "metrics": {k: float(np.nanmean(v)) for k, v in metrics.items()},
    # }


def train_mlp_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 0.001,
    hidden_layers: list[int] | None = None,
    dropout_rate: float = 0.3,
    patience: int = 10,
) -> dict[str, Any]:
    """Treina MLP em um fold e retorna métricas e modelo.

    Args:
        X_train: Features de treino.
        y_train: Labels de treino.
        X_val: Features de validação.
        y_val: Labels de validação.
        epochs: Número de epochs.
        batch_size: Tamanho do batch.
        learning_rate: Taxa de aprendizado.
        hidden_layers: Camadas ocultas.
        dropout_rate: Taxa de dropout.
        patience: Patience para early stopping.

    Returns:
        Dicionário com métricas e modelo treinado.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Using device: {device}")

    if hidden_layers is None:
        hidden_layers = [128, 64, 32]

    # Criar datasets e dataloaders
    train_dataset = ChurnDataset(X_train, y_train)
    val_dataset = ChurnDataset(X_val, y_val)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=0
    )
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Inicializar modelo
    input_size = X_train.shape[1]
    model = MLPClassifier(
        input_size=input_size,
        hidden_layers=hidden_layers,
        dropout_rate=dropout_rate,
    ).to(device)

    # Loss e otimizador
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Early stopping
    early_stop = EarlyStopping(
        patience=patience, model_path=Path("models/best_mlp.pt")
    )

    # Treinar
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device).float().unsqueeze(
                1
            )
            optimizer.zero_grad()
            y_pred = model(X_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        # Validate
        model.eval()
        val_loss = 0.0
        y_val_preds = []
        y_val_probas = []

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device).float().unsqueeze(
                    1
                )
                y_pred = model(X_batch)
                loss = criterion(y_pred, y_batch)
                val_loss += loss.item()
                y_val_preds.append(model.predict(X_batch.cpu()))
                y_val_probas.append(model.predict_proba(X_batch.cpu()))

        val_loss /= len(val_loader)
        y_val_preds = np.concatenate(y_val_preds)
        y_val_probas = np.concatenate(y_val_probas)

        LOGGER.info(
            f"Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f}, val_loss: {val_loss:.6f}"
        )

        # Early stopping
        if early_stop(val_loss, model):
            LOGGER.info(f"Early stopping at epoch {epoch+1}")
            break

    # Carrega melhor modelo
    early_stop.load_best_model(model)
    model.eval()

    # Compute final metrics
    with torch.no_grad():
        y_pred = []
        y_proba = []
        for X_batch, _ in val_loader:
            X_batch = X_batch.to(device)
            y_pred.append(model.predict(X_batch.cpu()))
            y_proba.append(model.predict_proba(X_batch.cpu()))

    y_pred = np.concatenate(y_pred)
    y_proba = np.concatenate(y_proba)

    metrics = eval_mod.compute_metrics(y_val, y_pred, y_proba)

    return {
        "model": model,
        "metrics": metrics,
        "y_val": y_val,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def main() -> None:  # pragma: no cover - integration entrypoint
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/raw/Telco_customer_churn(Telco_Churn).csv")
    parser.add_argument("--out", default="models/baseline_pipeline.joblib")
    parser.add_argument("--metric", default="roc_auc", choices=["roc_auc", "pr_auc", "f1"])
    parser.add_argument("--cost-per-churn", default=100.0, type=float)
    parser.add_argument("--intervention-success", default=0.3, type=float)
    parser.add_argument(
        "--models",
        nargs="+",
        default=["dummy", "logistic"],
        help="models to train: dummy, logistic, mlp",
    )
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

