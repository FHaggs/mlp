from __future__ import annotations

from typing import Union

import numpy as np


OptimizerLike = Union[str, "Optimizer", None]


class Optimizer:
    """Interface simples para otimizadores."""

    def update(self, layers, lr: float) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic Gradient Descent (sem momentum)."""

    def update(self, layers, lr: float) -> None:
        for layer in layers:
            # Passo direto no sentido oposto do gradiente.
            layer.W -= lr * layer.grads["dW"]
            layer.b -= lr * layer.grads["db"]

    def __repr__(self) -> str:
        return "sgd"


class Adam(Optimizer):
    """Adam optimizer."""

    def __init__(
        self,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self._state: dict[int, dict[str, np.ndarray]] = {}

    def _get_state(self, layer) -> dict[str, np.ndarray]:
        key = id(layer)
        if key not in self._state:
            # m = 1º momento (média dos gradientes), v = 2º momento (média dos gradientes ao quadrado).
            self._state[key] = {
                "mW": np.zeros_like(layer.W),
                "vW": np.zeros_like(layer.W),
                "mb": np.zeros_like(layer.b),
                "vb": np.zeros_like(layer.b),
            }
        return self._state[key]

    def update(self, layers, lr: float) -> None:
        # t é o passo global, usado na correção de viés de m e v.
        self.t += 1

        for layer in layers:
            dW = layer.grads["dW"]
            db = layer.grads["db"]
            state = self._get_state(layer)

            # Médias móveis exponenciais dos gradientes (m) e do quadrado deles (v).
            state["mW"] = self.beta1 * state["mW"] + (1.0 - self.beta1) * dW
            state["mb"] = self.beta1 * state["mb"] + (1.0 - self.beta1) * db
            state["vW"] = self.beta2 * state["vW"] + (1.0 - self.beta2) * (dW ** 2)
            state["vb"] = self.beta2 * state["vb"] + (1.0 - self.beta2) * (db ** 2)

            # Correção de viés dos momentos no começo do treino.
            mW_hat = state["mW"] / (1.0 - self.beta1 ** self.t)
            mb_hat = state["mb"] / (1.0 - self.beta1 ** self.t)
            vW_hat = state["vW"] / (1.0 - self.beta2 ** self.t)
            vb_hat = state["vb"] / (1.0 - self.beta2 ** self.t)

            # Atualização adaptativa: normaliza o passo por sqrt(v) para cada parâmetro.
            layer.W -= lr * mW_hat / (np.sqrt(vW_hat) + self.eps)
            layer.b -= lr * mb_hat / (np.sqrt(vb_hat) + self.eps)

    def __repr__(self) -> str:
        return f"adam(beta1={self.beta1}, beta2={self.beta2}, eps={self.eps})"


def get_optimizer(optimizer: OptimizerLike = None) -> Optimizer:
    """Resolve otimizador por objeto, string ou padrão (SGD)."""
    if optimizer is None:
        return SGD()

    if isinstance(optimizer, Optimizer):
        return optimizer

    if isinstance(optimizer, str):
        name = optimizer.strip().lower()
        if name == "sgd":
            return SGD()
        if name == "adam":
            return Adam()

    raise ValueError("optimizer deve ser objeto Optimizer, 'sgd', 'adam' ou None")