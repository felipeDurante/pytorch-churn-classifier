"""Carregamento e validação dos dados de churn."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

LOGGER = logging.getLogger(__name__)

TARGET_COLUMN = "Churn Value"
REQUIRED_COLUMNS = {
    "CustomerID",
    "Tenure Months",
    "Monthly Charges",
    "Total Charges",
    "Contract",
    "Churn Label",
    TARGET_COLUMN,
}


def load_telco_churn(path: str | Path) -> pd.DataFrame:
    """Carrega o dataset Telco Churn e normaliza os tipos conhecidos."""
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    dataframe = pd.read_csv(dataset_path, sep=";", decimal=",")
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas obrigatórias ausentes: {columns}")

    dataframe["Total Charges"] = pd.to_numeric(
        dataframe["Total Charges"].str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )

    LOGGER.info(
        "dataset_loaded",
        extra={
            "dataset_path": str(dataset_path),
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
        },
    )
    return dataframe


def main() -> None:
    """Carrega o dataset padrão para uma verificação rápida."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    load_telco_churn(Path("data/raw/Telco_customer_churn(Telco_Churn).csv"))


if __name__ == "__main__":
    main()
