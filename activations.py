import numpy as np


class Sigmoid:
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

    def derivative(self, z: np.ndarray) -> np.ndarray:
        s = self(z)
        return s * (1 - s)

    def __repr__(self) -> str:
        return "sigmoid"


class ReLU:
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.maximum(0, z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return (z > 0).astype(float)

    def __repr__(self) -> str:
        return "relu"


class Tanh:
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return np.tanh(z)

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return 1 - np.tanh(z) ** 2

    def __repr__(self) -> str:
        return "tanh"


class Linear:
    def __call__(self, z: np.ndarray) -> np.ndarray:
        return z

    def derivative(self, z: np.ndarray) -> np.ndarray:
        return np.ones_like(z)

    def __repr__(self) -> str:
        return "linear"
