"""Script para treinar MLP e comparar com baselines.

Este script treina:
1. DummyClassifier (baseline)
2. LogisticRegression (baseline)
3. MLPClassifier (neural network)

E gera uma tabela comparativa com todas as métricas.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from projeto_ml import train

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Treina todos os modelos e gera comparação."""
    dataset_path = "data/raw/Telco_customer_churn(Telco_Churn).csv"
    n_splits = 5

    LOGGER.info("Treinando todos os modelos...")
    result = train.train(
        dataset_path=dataset_path,
        model_out=Path("models/comparison_pipeline.joblib"),
        n_splits=n_splits,
        metric="roc_auc",
        cost_per_churn=100.0,
        intervention_success=0.3,
        models=["dummy", "logistic", "mlp"],
    )

    # Exibir resultados
    metrics_mean = result["metrics_mean"]
    metrics_std = result["metrics_std"]
    LOGGER.info("Métricas por modelo:")
    for key, value in sorted(metrics_mean.items()):
        LOGGER.info(f"  {key}: {value:.6f}")

    # Salvar resultado
    df_results = pd.DataFrame(
        [
            {
                "Metric": metric_name,
                "Mean": mean_value,
                "Std": metrics_std.get(metric_name, 0.0),
            }
            for metric_name, mean_value in metrics_mean.items()
        ]
    )

    df_results.head()

    output_path = Path("models/comparison_results.csv")
    df_results.to_csv(output_path, index=False)
    LOGGER.info(f"Resultados salvos em {output_path}")

    LOGGER.info("Treinamento concluído!")


if __name__ == "__main__":
    main()
