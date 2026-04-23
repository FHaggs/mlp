from __future__ import annotations

from pathlib import Path

import numpy as np


def _one_hot(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    y = np.zeros((labels.shape[0], num_classes), dtype=np.float32)
    y[np.arange(labels.shape[0]), labels.astype(int)] = 1.0
    return y


def load_mnist_binary(
    path: str = "mnist.npz",
    positive_digit: int = 0,
    train_limit: int | None = 12000,
    val_limit: int | None = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from .npz and convert to binary classification.

    The target is 1 for `positive_digit` and 0 for all other digits.
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. Coloque o mnist.npz na raiz do projeto."
        )

    data = np.load(data_path)

    X_train = data["x_train"]
    y_train = data["y_train"]
    X_val = data["x_test"]
    y_val = data["y_test"]

    # Flatten (28x28 -> 784) and normalize pixel values to [0, 1].
    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_val = X_val.reshape(X_val.shape[0], -1).astype(np.float32) / 255.0

    # Binary target with shape (n_samples, 1).
    y_train = (y_train == positive_digit).astype(np.float32).reshape(-1, 1)
    y_val = (y_val == positive_digit).astype(np.float32).reshape(-1, 1)

    if train_limit is not None:
        X_train = X_train[:train_limit]
        y_train = y_train[:train_limit]
    if val_limit is not None:
        X_val = X_val[:val_limit]
        y_val = y_val[:val_limit]

    return X_train, y_train, X_val, y_val


def load_mnist_multiclass(
    path: str = "mnist.npz",
    train_limit: int | None = 12000,
    val_limit: int | None = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load MNIST from .npz and return one-hot targets for 10 classes."""
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. Coloque o mnist.npz na raiz do projeto."
        )

    data = np.load(data_path)

    X_train = data["x_train"]
    y_train = data["y_train"]
    X_val = data["x_test"]
    y_val = data["y_test"]

    X_train = X_train.reshape(X_train.shape[0], -1).astype(np.float32) / 255.0
    X_val = X_val.reshape(X_val.shape[0], -1).astype(np.float32) / 255.0

    if train_limit is not None:
        X_train = X_train[:train_limit]
        y_train = y_train[:train_limit]
    if val_limit is not None:
        X_val = X_val[:val_limit]
        y_val = y_val[:val_limit]

    y_train_one_hot = _one_hot(y_train, num_classes=10)
    y_val_one_hot = _one_hot(y_val, num_classes=10)

    return X_train, y_train_one_hot, X_val, y_val_one_hot


def load_boston_regression(
    path: str = "Boston House Prices.csv",
    val_split: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load Boston Housing CSV for a regression task.

    Returns train/validation arrays with:
    - X standardized using train split statistics
    - y with shape (n_samples, 1)
    """
    data_path = Path(path)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {path}. Coloque o CSV na raiz do projeto."
        )

    data = np.genfromtxt(data_path, delimiter=",", skip_header=1, dtype=np.float32)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError("CSV inválido para Boston Housing: esperado pelo menos 2 colunas.")

    X = data[:, :-1]
    y = data[:, -1].reshape(-1, 1)

    n_samples = X.shape[0]
    val_size = max(1, int(n_samples * val_split))
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_train = X[train_idx]
    y_train = y[train_idx]
    X_val = X[val_idx]
    y_val = y[val_idx]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std = np.where(std < 1e-9, 1.0, std)

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std

    return X_train.astype(np.float32), y_train.astype(np.float32), X_val.astype(np.float32), y_val.astype(np.float32)
