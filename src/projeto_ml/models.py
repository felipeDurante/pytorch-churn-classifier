"""Modelos de redes neurais em PyTorch para classificação de churn.

Este módulo implementa:
- `ChurnDataset`: Dataset customizado para PyTorch
- `MLPClassifier`: Rede neural MLP com early stopping
- `EarlyStopping`: Callback para interrupção antecipada do treinamento
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

LOGGER = logging.getLogger(__name__)


class ChurnDataset(Dataset):
    """Dataset customizado para dados de churn em PyTorch.

    Args:
        X: Array numpy (n_samples, n_features).
        y: Array numpy (n_samples,) com labels 0/1.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        """Inicializa o dataset.

        Args:
            X: Features como array numpy.
            y: Labels como array numpy (0 ou 1).
        """
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self) -> int:
        """Retorna o tamanho do dataset."""
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Retorna um exemplo (X, y).

        Args:
            idx: Índice do exemplo.

        Returns:
            Tupla (features, label).
        """
        return self.X[idx], self.y[idx]


class MLPClassifier(nn.Module):
    """Rede neural MLP para classificação binária de churn.

    Arquitetura:
    - Input layer: n_features
    - Hidden layers: configurable (default: [128, 64, 32])
    - Activation: ReLU em hidden layers, Sigmoid no output
    - Dropout: optional (default: 0.3)
    - BatchNorm: optional (default: True)

    Args:
        input_size: Número de features de entrada.
        hidden_layers: Lista com tamanho de cada camada oculta.
        dropout_rate: Taxa de dropout (0 = sem dropout).
        use_batch_norm: Se True, adiciona BatchNorm1d.
    """

    def __init__(
        self,
        input_size: int,
        hidden_layers: list[int] | None = None,
        dropout_rate: float = 0.3,
        use_batch_norm: bool = True,
    ) -> None:
        """Inicializa a arquitetura da rede.

        Args:
            input_size: Número de features.
            hidden_layers: Tamanho de cada camada oculta.
            dropout_rate: Taxa de dropout.
            use_batch_norm: Se usar batch normalization.
        """
        super().__init__()

        if hidden_layers is None:
            hidden_layers = [128, 64, 32]

        self.use_batch_norm = use_batch_norm
        self.dropout_rate = dropout_rate
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()

        # Construir camadas ocultas
        prev_size = input_size
        for hidden_size in hidden_layers:
            self.layers.append(nn.Linear(prev_size, hidden_size))
            if use_batch_norm:
                self.batch_norms.append(nn.BatchNorm1d(hidden_size))
            if dropout_rate > 0:
                self.dropouts.append(nn.Dropout(dropout_rate))
            prev_size = hidden_size

        # Camada de saída (sigmoid para probabilidades)
        self.output_layer = nn.Linear(prev_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass da rede.

        Args:
            x: Tensor de entrada (batch_size, input_size).

        Returns:
            Tensor de saída (batch_size, 1) com probabilidades [0, 1].
        """
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.use_batch_norm and i < len(self.batch_norms):
                x = self.batch_norms[i](x)
            x = torch.relu(x)
            if self.dropout_rate > 0 and i < len(self.dropouts):
                x = self.dropouts[i](x)

        x = self.output_layer(x)
        return torch.sigmoid(x)

    def predict(self, X: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
        """Predição em batch (sem gradiente).

        Args:
            X: Tensor de features.
            threshold: Limiar para classificação binária.

        Returns:
            Array numpy com predições (0 ou 1).
        """
        with torch.no_grad():
            proba = self.forward(X)
            predictions = (proba >= threshold).long().numpy()
        return predictions.flatten()

    def predict_proba(self, X: torch.Tensor) -> np.ndarray:
        """Retorna probabilidades de classe positiva.

        Args:
            X: Tensor de features.

        Returns:
            Array numpy com probabilidades [0, 1].
        """
        with torch.no_grad():
            proba = self.forward(X)
        return proba.numpy().flatten()


class EarlyStopping:
    """Monitora loss de validação e interrompe treinamento se não melhorar.

    Args:
        patience: Número de epochs sem melhora antes de interromper.
        min_delta: Melhora mínima para resetar patience.
        model_path: Caminho onde salvar o melhor modelo.
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        model_path: str | Path = "best_model.pt",
    ) -> None:
        """Inicializa o callback de early stopping.

        Args:
            patience: Epochs sem melhora.
            min_delta: Melhora mínima.
            model_path: Onde salvar o melhor modelo.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.model_path = Path(model_path)
        self.best_loss = float("inf")
        self.patience_counter = 0
        self.stopped_epoch = 0

    def __call__(
        self,
        val_loss: float,
        model: nn.Module,
    ) -> bool:
        """Verifica se deve parar o treinamento.

        Args:
            val_loss: Loss de validação atual.
            model: Modelo PyTorch para salvar.

        Returns:
            True se deve parar, False caso contrário.
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
            self._save_checkpoint(model)
            LOGGER.info(f"EarlyStopping: val_loss improved to {val_loss:.6f}")
            return False

        self.patience_counter += 1
        if self.patience_counter >= self.patience:
            self.stopped_epoch = self.patience_counter
            LOGGER.info(
                f"EarlyStopping: stopping after {self.patience_counter} epochs "
                f"without improvement"
            )
            return True

        return False

    def _save_checkpoint(self, model: nn.Module) -> None:
        """Salva o melhor modelo encontrado.

        Args:
            model: Modelo PyTorch a salvar.
        """
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), self.model_path)
        LOGGER.info(f"EarlyStopping: saved best model to {self.model_path}")

    def load_best_model(self, model: nn.Module) -> None:
        """Carrega o melhor modelo salvo.

        Args:
            model: Modelo PyTorch para carregar state_dict.
        """
        if self.model_path.exists():
            model.load_state_dict(torch.load(self.model_path, weights_only=True))
            LOGGER.info(f"EarlyStopping: loaded best model from {self.model_path}")
        else:
            LOGGER.warning(f"EarlyStopping: checkpoint not found at {self.model_path}")
