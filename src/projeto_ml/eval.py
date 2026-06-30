"""Avaliação e comparação de modelos de classificação de churn.

Este módulo implementa:
- `compute_metrics`: Calcula múltiplas métricas de classificação
- `compare_models`: Compara modelos e gera tabela comparativa
- `compute_cost_analysis`: Análise de custo/benefício com trade-off FP vs FN
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

LOGGER = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray | None = None,
) -> dict[str, float]:
    """Calcula múltiplas métricas de classificação.

    Args:
        y_true: Labels verdadeiros (0 ou 1).
        y_pred: Predições (0 ou 1).
        y_proba: Probabilidades da classe positiva (opcional).

    Returns:
        Dicionário com métricas calculadas.
    """
    metrics: dict[str, float] = {}

    # Métricas básicas
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
    metrics["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
    metrics["f1"] = float(f1_score(y_true, y_pred, zero_division=0))

    # Métricas probabilísticas
    if y_proba is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_proba))
        except Exception:
            metrics["roc_auc"] = float("nan")

        try:
            metrics["pr_auc"] = float(average_precision_score(y_true, y_proba))
        except Exception:
            metrics["pr_auc"] = float("nan")
    else:
        metrics["roc_auc"] = float("nan")
        metrics["pr_auc"] = float("nan")

    return metrics


def compute_cost_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    cost_per_churn: float = 100.0,
    intervention_cost: float = 30.0,
    intervention_success: float = 0.3,
) -> dict[str, float]:
    """Análise de custo/benefício com trade-off FP vs FN.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Predições.
        cost_per_churn: Custo de perder um cliente (churn).
        intervention_cost: Custo de intervenção por cliente.
        intervention_success: Taxa de sucesso da intervenção.

    Returns:
        Dicionário com análise de custo.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Custo de falsos negativos (clientes que deixaram e não foram detectados)
    cost_fn = fn * cost_per_churn

    # Custo de falsos positivos (clientes que não deixariam mas recebem intervenção)
    cost_fp = fp * intervention_cost

    # Benefício de verdadeiros positivos (clientes retidos)
    benefit_tp = tp * cost_per_churn * intervention_success - tp * intervention_cost

    # Custo total
    total_cost = cost_fn + cost_fp - benefit_tp
    baseline_cost = (tp + fn) * cost_per_churn
    cost_saved = baseline_cost - total_cost

    return {
        "cost_fn": float(cost_fn),
        "cost_fp": float(cost_fp),
        "benefit_tp": float(benefit_tp),
        "cost_saved": float(max(0.0, cost_saved)),
        "total_cost": float(total_cost),
        "roi": float((benefit_tp / (tp * intervention_cost)) if tp > 0 else 0.0),
    }


def compare_models(
    results: dict[str, Any],
    cost_per_churn: float = 100.0,
    intervention_cost: float = 30.0,
    intervention_success: float = 0.3,
) -> pd.DataFrame:
    """Compara múltiplos modelos e retorna tabela comparativa.

    Args:
        results: Dicionário com resultados de cada modelo.
        cost_per_churn: Custo por cliente perdido.
        intervention_cost: Custo de intervenção.
        intervention_success: Taxa de sucesso de intervenção.

    Returns:
        DataFrame com comparação de modelos.
    """
    comparison_rows = []

    for model_name, model_data in results.items():
        y_true = model_data["y_true"]
        y_pred = model_data["y_pred"]
        y_proba = model_data.get("y_proba")

        metrics = compute_metrics(y_true, y_pred, y_proba)
        cost_analysis = compute_cost_analysis(
            y_true,
            y_pred,
            cost_per_churn=cost_per_churn,
            intervention_cost=intervention_cost,
            intervention_success=intervention_success,
        )

        row = {"Model": model_name, **metrics, **cost_analysis}
        comparison_rows.append(row)

    df = pd.DataFrame(comparison_rows)

    # Ordenar por F1 score (descendente)
    df = df.sort_values("f1", ascending=False).reset_index(drop=True)

    return df


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> str:
    """Gera relatório de classificação em texto.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Predições.
        model_name: Nome do modelo (para logging).

    Returns:
        String formatada com relatório.
    """
    report = classification_report(
        y_true,
        y_pred,
        target_names=["No Churn", "Churn"],
        digits=4,
    )
    return f"Classification Report - {model_name}:\n{report}"


def generate_confusion_matrix_text(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Gera matriz de confusão em texto.

    Args:
        y_true: Labels verdadeiros.
        y_pred: Predições.

    Returns:
        String formatada com matriz de confusão.
    """
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    text = (
        "Confusion Matrix:\n"
        f"TN (True Negatives):  {tn}\n"
        f"FP (False Positives): {fp}\n"
        f"FN (False Negatives): {fn}\n"
        f"TP (True Positives):  {tp}\n"
    )
    return text
