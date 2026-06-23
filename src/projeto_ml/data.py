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


def load_telco_churn(
    path: str | Path,
    *,
    as_category: bool = True,
    category_threshold: int = 50,
    drop_leakage: bool = False,
) -> pd.DataFrame:
    """Carrega o dataset Telco Churn e normaliza os tipos conhecidos.

    Parameters
    - path: caminho para o CSV raw
    - as_category: se True, converte colunas textuais de baixa cardinalidade para `category`
    - category_threshold: limite superior de cardinalidade para conversão automática
    - drop_leakage: se True, remove colunas de vazamento/identificadores padrão
    """
    dataset_path = Path(path)
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_path}")

    dataframe = pd.read_csv(dataset_path, sep=";", decimal=",")
    missing_columns = REQUIRED_COLUMNS.difference(dataframe.columns)
    if missing_columns:
        columns = ", ".join(sorted(missing_columns))
        raise ValueError(f"Colunas obrigatórias ausentes: {columns}")

    # Normaliza Total Charges: algumas células têm espaços ou vírgula decimal
    dataframe["Total Charges"] = pd.to_numeric(
        dataframe["Total Charges"].astype(str).str.replace(",", ".", regex=False).str.strip(),
        errors="coerce",
    )

    # Opcional: remover colunas que vazam informação ou são identificadores
    if drop_leakage:
        to_drop = [
            "Churn Label",
            "Churn Score",
            "Churn Reason",
            "CustomerID",
            "Country",
            "State",
            "City",
            "Zip Code",
        ]
        present = [c for c in to_drop if c in dataframe.columns]
        if present:
            dataframe = dataframe.drop(columns=present)

    # Converter colunas textuais de baixa cardinalidade para category facilita memória e codificação
    if as_category:
        text_cols = dataframe.select_dtypes(include=["object", "category", "string"]).columns.tolist()
        # Exclui o target se presente
        text_cols = [c for c in text_cols if c != TARGET_COLUMN]
        for c in text_cols:
            try:
                nunique = int(dataframe[c].nunique(dropna=False))
            except Exception:
                nunique = category_threshold + 1
            if nunique <= category_threshold:
                dataframe[c] = dataframe[c].astype("category")

    LOGGER.info(
        "dataset_loaded",
        extra={
            "dataset_path": str(dataset_path),
            "rows": len(dataframe),
            "columns": len(dataframe.columns),
        },
    )
    return dataframe


def save_processed(
    raw_path: str | Path,
    out_path: str | Path = Path("data/processed/telco_processed.csv"),
    *,
    as_category: bool = True,
    category_threshold: int = 50,
    drop_leakage: bool = True,
) -> Path:
    """Carrega, aplica limpeza leve e salva arquivo processado em CSV.

    Retorna o caminho para o arquivo salvo.
    """
    df = load_telco_churn(
        raw_path, as_category=as_category, category_threshold=category_threshold, drop_leakage=drop_leakage
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Salva mantendo separador padrão (vírgula) e sem índices
    df.to_csv(out, index=False)
    LOGGER.info("processed_saved", extra={"out_path": str(out), "rows": len(df)})
    return out


def main() -> None:
    """Carrega o dataset padrão para uma verificação rápida."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    load_telco_churn(Path("data/raw/Telco_customer_churn(Telco_Churn).csv"))


if __name__ == "__main__":
    main()
