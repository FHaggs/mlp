import numpy as np


class Accuracy:
    """Acurácia para classificação binária (threshold=0.5)."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        predictions = (y_pred >= 0.5).astype(int)
        return float(np.mean(predictions == y_true))

    def __repr__(self) -> str:
        return "accuracy"


class R2:
    """Coeficiente de determinação R² para regressão."""

    def __call__(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return float(1 - ss_res / (ss_tot + 1e-9))

    def __repr__(self) -> str:
        return "r2"
