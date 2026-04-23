import numpy as np
from activations import Linear
from initializers import get_initializer


class Dense:
    """Camada totalmente conectada.

    Parâmetros
    ----------
    input_size  : dimensão de entrada
    output_size : dimensão de saída
    activation  : objeto de ativação (deve ter __call__ e .derivative)
                  padrão = Linear (sem ativação)
    initializer : estratégia de inicialização
                  pode ser string ('he', 'xavier', 'normal', 'uniform')
                  ou um callable(input_size, output_size)
                  padrão = automático (He para ReLU, Xavier para as demais)
    """

    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation=None,
        initializer=None,
    ) -> None:
        self.activation = activation if activation is not None else Linear()

        # Inicialização configurável: He, Xavier, Normal, Uniform ou callable.
        self.initializer = get_initializer(initializer, activation=self.activation)
        self.W: np.ndarray = self.initializer(input_size, output_size)
        self.b: np.ndarray = np.zeros((1, output_size))

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

        # S(i) = dJ/dA(i) * dA(i)/dZ(i)
        # Na última camada, dA vem da loss: S(l) = dJ/dA(l) * dA(l)/dZ(l)
        # Nas camadas anteriores, dA já carrega S(i+1) * T(i+1), então:
        #   S(i) = S(i+1) * T(i+1) * a'(i)
        dZ = dA * self.activation.derivative(Z)     # S(i)

        n = A_prev.shape[0]
        dW = (A_prev.T @ dZ).T / n                  # dJ/dT(i) = S(i) * a(i-1)
        db = np.mean(dZ, axis=0, keepdims=True)      # dJ/db(i) = S(i)
        dA_prev = dZ @ self.W                        # S(i) * T(i) — vira o dA da camada anterior

        self.grads = {"dW": dW, "db": db, "dZ": dZ}
        return dA_prev

    def __repr__(self) -> str:
        in_sz, out_sz = self.W.shape[1], self.W.shape[0]
        return f"Dense({in_sz} → {out_sz}, activation={self.activation})"
