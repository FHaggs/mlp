import numpy as np
from activations import Linear


class Dense:
    """Camada totalmente conectada.

    Parâmetros
    ----------
    input_size  : dimensão de entrada
    output_size : dimensão de saída
    activation  : objeto de ativação (deve ter __call__ e .derivative)
                  padrão = Linear (sem ativação)
    """

    def __init__(self, input_size: int, output_size: int, activation=None) -> None:
        self.activation = activation if activation is not None else Linear()

        # Inicialização He para ReLU, Xavier para as demais
        # TODO: Permitir escolher entre outras estratégias de inicialização
        # Recebe essas coisas por parametro
        
        scale = np.sqrt(2 / input_size) if repr(self.activation) == "relu" \
            else np.sqrt(1 / input_size)
        self.W = np.random.randn(output_size, input_size) * scale
        self.b = np.zeros((1, output_size))

        # Preenchidos no forward — usados no backward e no monitor
        self._cache: dict = {}
        self.grads: dict = {}

    # ------------------------------------------------------------------
    # Forward: A_prev → Z → A
    # ------------------------------------------------------------------
    def forward(self, A_prev: np.ndarray) -> np.ndarray:
        Z = A_prev @ self.W.T + self.b          # (n_samples, output_size)
        A = self.activation(Z)
        self._cache = {"A_prev": A_prev, "Z": Z, "A": A}
        return A

    # ------------------------------------------------------------------
    # Backward: recebe dL/dA, devolve dL/dA_prev e armazena dW, db
    # ------------------------------------------------------------------
    def backward(self, dA: np.ndarray) -> np.ndarray:
        Z = self._cache["Z"]
        A_prev = self._cache["A_prev"]

        dZ = dA * self.activation.derivative(Z)     # hadamard

        n = A_prev.shape[0]
        dW = (A_prev.T @ dZ).T / n                  # (output_size, input_size)
        db = np.mean(dZ, axis=0, keepdims=True)      # (1, output_size)
        dA_prev = dZ @ self.W                        # (n_samples, input_size)

        self.grads = {"dW": dW, "db": db, "dZ": dZ}
        return dA_prev

    def __repr__(self) -> str:
        in_sz, out_sz = self.W.shape[1], self.W.shape[0]
        return f"Dense({in_sz} → {out_sz}, activation={self.activation})"
