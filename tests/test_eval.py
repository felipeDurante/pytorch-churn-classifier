"""Testes para avaliação e comparação de modelos."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from projeto_ml.eval import (
    compare_models,
    compute_cost_analysis,
    compute_metrics,
    generate_classification_report,
    generate_confusion_matrix_text,
)


def test_compute_metrics_basic():
    """Testa cálculo de métricas básicas."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])
    y_proba = np.array([0.1, 0.2, 0.9, 0.4, 0.1, 0.8])

    metrics = compute_metrics(y_true, y_pred, y_proba)

    assert "accuracy" in metrics
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics

    # Verificar valores básicos
    assert 0 <= metrics["accuracy"] <= 1
    assert 0 <= metrics["precision"] <= 1
    assert 0 <= metrics["recall"] <= 1
    assert 0 <= metrics["f1"] <= 1


def test_compute_metrics_no_proba():
    """Testa cálculo de métricas sem probabilidades."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    metrics = compute_metrics(y_true, y_pred, y_proba=None)

    assert "accuracy" in metrics
    assert "f1" in metrics
    assert np.isnan(metrics["roc_auc"])
    assert np.isnan(metrics["pr_auc"])


def test_compute_cost_analysis():
    """Testa análise de custo."""
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_pred = np.array([0, 0, 1, 0, 0, 1])

    cost_analysis = compute_cost_analysis(
        y_true,
        y_pred,
        cost_per_churn=100.0,
        intervention_cost=30.0,
        intervention_success=0.3,
    )

    assert "cost_fn" in cost_analysis
    assert "cost_fp" in cost_analysis
    assert "benefit_tp" in cost_analysis
    assert "cost_saved" in cost_analysis
    assert "total_cost" in cost_analysis
    assert "roi" in cost_analysis

    # Verificar que custos/benefícios são não-negativos
    assert cost_analysis["cost_fn"] >= 0
    assert cost_analysis["cost_fp"] >= 0
    assert cost_analysis["cost_saved"] >= 0


def test_compare_models():
    """Testa comparação de modelos."""
    results = {
        "model_a": {
            "y_true": np.array([0, 0, 1, 1, 0, 1]),
            "y_pred": np.array([0, 0, 1, 0, 0, 1]),
            "y_proba": np.array([0.1, 0.2, 0.9, 0.4, 0.1, 0.8]),
        },
        "model_b": {
            "y_true": np.array([0, 0, 1, 1, 0, 1]),
            "y_pred": np.array([0, 0, 1, 1, 0, 1]),
            "y_proba": np.array([0.1, 0.2, 0.9, 0.8, 0.1, 0.8]),
        },
    }

    df = compare_models(results)

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 2
    assert "Model" in df.columns
    assert "f1" in df.columns
    assert "accuracy" in df.columns


def test_generate_classification_report():
    """Testa geração de relatório de classificação."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])

    report = generate_classification_report(y_true, y_pred, model_name="TestModel")

    assert isinstance(report, str)
    assert "TestModel" in report
    assert "Churn" in report


def test_generate_confusion_matrix_text():
    """Testa geração de matriz de confusão em texto."""
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 0])

    cm_text = generate_confusion_matrix_text(y_true, y_pred)

    assert isinstance(cm_text, str)
    assert "TN" in cm_text
    assert "FP" in cm_text
    assert "FN" in cm_text
    assert "TP" in cm_text
