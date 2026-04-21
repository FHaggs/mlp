import numpy as np

class He:
    """
    Inicialização de He (Kaiming).
    Recomendada para redes que usam ReLU ou Leaky ReLU.
    Mantém a variância das ativações constante em camadas profundas.
    """
    def __call__(self, fan_in: int, fan_out: int) -> np.ndarray:
        # A fórmula matemática do He: desvio padrão de sqrt(2 / fan_in)
        std = np.sqrt(2.0 / fan_in)
        return np.random.randn(fan_out, fan_in) * std

class Xavier:
    """
    Inicialização de Xavier.
    Recomendada para Sigmoid e Tanh.
    """
    def __call__(self, fan_in: int, fan_out: int) -> np.ndarray:
        std = np.sqrt(2.0 / (fan_in + fan_out))
        return np.random.randn(fan_out, fan_in) * std

class Normal:
    def __init__(self, mean: float = 0.0, std: float = 0.01) -> None:
        self.mean = mean
        self.std = std

    def __call__(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.normal(self.mean, self.std, size=(fan_out, fan_in))

class Uniform:
    def __init__(self, limit: float = 0.01) -> None:
        self.limit = limit

    def __call__(self, fan_in: int, fan_out: int) -> np.ndarray:
        return np.random.uniform(-self.limit, self.limit, size=(fan_out, fan_in))