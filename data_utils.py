from __future__ import annotations

from pathlib import Path

import numpy as np


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
