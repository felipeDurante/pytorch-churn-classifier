"""Testes para modelos de redes neurais em PyTorch."""
from __future__ import annotations

import numpy as np
import pytest
import torch

from projeto_ml.models import ChurnDataset, EarlyStopping, MLPClassifier


def test_churn_dataset_creation():
    """Testa criação do dataset."""
    X = np.random.randn(100, 20).astype(np.float32)
    y = np.random.randint(0, 2, 100)

    dataset = ChurnDataset(X, y)

    assert len(dataset) == 100
    x_sample, y_sample = dataset[0]
    assert isinstance(x_sample, torch.Tensor)
    assert isinstance(y_sample, torch.Tensor)
    assert x_sample.shape == (20,)
    assert y_sample.shape == ()


def test_mlp_classifier_architecture():
    """Testa arquitetura do MLP."""
    input_size = 20
    hidden_layers = [128, 64, 32]
    model = MLPClassifier(
        input_size=input_size,
        hidden_layers=hidden_layers,
        dropout_rate=0.3,
        use_batch_norm=True,
    )

    # Verificar número de camadas
    assert len(model.layers) == 3
    assert len(model.batch_norms) == 3
    assert len(model.dropouts) == 3


def test_mlp_classifier_forward():
    """Testa forward pass do MLP."""
    model = MLPClassifier(input_size=20, hidden_layers=[64, 32])
    X = torch.randn(10, 20)

    output = model(X)

    assert output.shape == (10, 1)
    assert torch.all(output >= 0) and torch.all(output <= 1)  # Sigmoid output


def test_mlp_classifier_predict():
    """Testa predição do MLP."""
    model = MLPClassifier(input_size=20, hidden_layers=[64, 32])
    X = torch.randn(10, 20)

    predictions = model.predict(X, threshold=0.5)

    assert predictions.shape == (10,)
    assert np.all((predictions == 0) | (predictions == 1))


def test_mlp_classifier_predict_proba():
    """Testa probabilidades do MLP."""
    model = MLPClassifier(input_size=20, hidden_layers=[64, 32])
    X = torch.randn(10, 20)

    probabilities = model.predict_proba(X)

    assert probabilities.shape == (10,)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


def test_early_stopping_initialization():
    """Testa inicialização do early stopping."""
    es = EarlyStopping(patience=5, min_delta=0.001, model_path="test_model.pt")

    assert es.patience == 5
    assert es.min_delta == 0.001
    assert es.best_loss == float("inf")
    assert es.patience_counter == 0


def test_early_stopping_improvement():
    """Testa early stopping com melhora."""
    model = MLPClassifier(input_size=20)
    es = EarlyStopping(patience=2, model_path="test_model.pt")

    # Primeira melhora
    should_stop = es(val_loss=0.5, model=model)
    assert not should_stop
    assert es.best_loss == 0.5

    # Segunda melhora (menor loss)
    should_stop = es(val_loss=0.3, model=model)
    assert not should_stop
    assert es.best_loss == 0.3


def test_early_stopping_no_improvement():
    """Testa early stopping sem melhora."""
    model = MLPClassifier(input_size=20)
    es = EarlyStopping(patience=2, model_path="test_model.pt")

    es(val_loss=0.5, model=model)

    # Sem melhora (patience = 2)
    should_stop = es(val_loss=0.5, model=model)
    assert not should_stop
    assert es.patience_counter == 1

    should_stop = es(val_loss=0.5, model=model)
    assert should_stop
    assert es.patience_counter == 2
