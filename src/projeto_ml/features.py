"""Engenharia e seleção de atributos.

Este módulo expõe utilitários para construir um pipeline de pré-processamento
compatível com `scikit-learn` que pode ser salvo e reutilizado durante a
inferência.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def _get_column_types(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Infer numeric and categorical columns from a dataframe."""
    numeric = df.select_dtypes(include=["number"]).columns.tolist()
    # Pandas can represent text columns as 'object' or the newer 'string' dtype;
    # include both plus 'category' to capture categorical encoded columns.
    categorical = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame, categorical_max_cardinality: int = 50) -> Pipeline:
    """Build a preprocessing Pipeline from example dataframe.

    - Imputa valores numéricos com média
    - Imputa valores categóricos com o valor mais frequente
    - One-hot encode para categóricas (ignora categorias desconhecidas)
    - Escala features numéricas com `StandardScaler`

    Retorna um `Pipeline` que recebe um DataFrame e retorna um array numpy
    pronto para consumo por modelos.
    """
    numeric_cols, categorical_cols = _get_column_types(df)

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler()),
    ])

    # `sparse` was replaced by `sparse_output` in newer scikit-learn releases.
    onehot_kwargs = {
        "handle_unknown": "ignore",
        "sparse_output": False,
    }
    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(**onehot_kwargs)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
        sparse_threshold=0,
    )

    pipeline = Pipeline([("preprocessor", preprocessor)])
    return pipeline


def fit_transform_pipeline(df: pd.DataFrame) -> Tuple[Pipeline, np.ndarray]:
    """Builds and fits a preprocessing pipeline on `df`, returning pipeline and transformed array.

    The returned array is a NumPy ndarray ready for model consumption.
    """
    pipeline = build_preprocessor(df)
    X = pipeline.fit_transform(df)
    return pipeline, X


def save_pipeline(pipeline: Pipeline, path: str) -> None:
    """Save pipeline to disk using joblib."""
    import joblib

    joblib.dump(pipeline, path)


def load_pipeline(path: str) -> Pipeline:
    """Load pipeline saved with `save_pipeline`."""
    import joblib

    return joblib.load(path)


def main() -> None:  # pragma: no cover - simple demo
    import pandas as pd

    demo = pd.DataFrame({"a": [1, 2, None], "b": ["x", "y", "x"]})
    pipe, X = fit_transform_pipeline(demo)
    print("Transformed shape:", X.shape)


if __name__ == "__main__":
    main()

