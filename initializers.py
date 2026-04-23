import numpy as np
from typing import Callable, Optional, Union


InitializerLike = Union[str, Callable[[int, int], np.ndarray], None]


class He:
    """Inicialização He (normal) para camadas com ReLU."""

    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        scale = np.sqrt(2.0 / input_size)
        return np.random.randn(output_size, input_size) * scale

    def __repr__(self) -> str:
        return "he"


class Xavier:
    """Inicialização Xavier (normal)."""

    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        scale = np.sqrt(1.0 / input_size)
        return np.random.randn(output_size, input_size) * scale

    def __repr__(self) -> str:
        return "xavier"


class Normal:
    """Inicialização normal genérica."""

    def __init__(self, mean: float = 0.0, std: float = 0.01) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        return np.random.randn(output_size, input_size) * self.std + self.mean

    def __repr__(self) -> str:
        return f"normal(mean={self.mean}, std={self.std})"


class Uniform:
    """Inicialização uniforme genérica."""

    def __init__(self, low: float = -0.05, high: float = 0.05) -> None:
        self.low = low
        self.high = high

    def __call__(self, input_size: int, output_size: int) -> np.ndarray:
        return np.random.uniform(self.low, self.high, size=(output_size, input_size))

    def __repr__(self) -> str:
        return f"uniform(low={self.low}, high={self.high})"


def get_initializer(
    initializer: InitializerLike = None,
    activation=None,
) -> Callable[[int, int], np.ndarray]:
    """Resolve o inicializador a partir de objeto, string ou padrão automático."""
    if initializer is None:
        return He() if repr(activation) == "relu" else Xavier()

    if callable(initializer):
        return initializer

    if isinstance(initializer, str):
        name = initializer.strip().lower()
        if name == "he":
            return He()
        if name == "xavier":
            return Xavier()
        if name == "normal":
            return Normal()
        if name == "uniform":
            return Uniform()

    raise ValueError(
        "initializer deve ser callable, uma das strings "
        "{'he', 'xavier', 'normal', 'uniform'} ou None"
    )