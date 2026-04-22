import numpy as np


class MSE:
    """Mean Squared Error — regressão."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return float(np.mean((y_true - y_pred) ** 2))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        return 2 * (y_pred - y_true) / y_true.size

    def __repr__(self) -> str:
        return "mse"


class BinaryCrossEntropy:
    """Binary Cross-Entropy — classificação binária."""

    _eps = 1e-9

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self._eps, 1 - self._eps)
        return float(-np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        y_pred = np.clip(y_pred, self._eps, 1 - self._eps)
        return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.size)

    def __repr__(self) -> str:
        return "binary_crossentropy"


class CategoricalCrossEntropy:
    """Categorical Cross-Entropy — classificação multiclasse com one-hot."""

    _eps = 1e-9

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_pred = np.clip(y_pred, self._eps, 1 - self._eps)
        sample_losses = -np.sum(y_true * np.log(y_pred), axis=1)
        return float(np.mean(sample_losses))

    def derivative(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        # dL/dZ for softmax + cross-entropy combination.
        return (y_pred - y_true) / y_true.shape[0]

    def __repr__(self) -> str:
        return "categorical_crossentropy"
