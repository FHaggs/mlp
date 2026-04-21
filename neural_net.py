"""NeuralNet — rede MLP genérica com backpropagation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from optimizers import SGD, Adam

import numpy as np

if TYPE_CHECKING:
    from layer import Dense
    from monitor import TrainingMonitor


class NeuralNet:
    """Rede neural totalmente conectada (MLP).

    Parâmetros
    ----------
    loss   : objeto de loss com __call__(y_true, y_pred) e .derivative(...)
    metric : objeto de métrica opcional com __call__(y_true, y_pred)

    Exemplo
    -------
    net = NeuralNet(loss=MSE(), metric=R2())
    net.add_layer(Dense(2, 8, ReLU()))
    net.add_layer(Dense(8, 1, Linear()))
    net.fit(X_train, y_train, epochs=200, lr=0.01,
            val_data=(X_val, y_val), monitor=monitor)
    """

    def __init__(self, loss, metric=None, optimizer=None) -> None:
        self.loss_fn = loss
        self.metric_fn = metric
        self.layers: list["Dense"] = []
        self.optimizer = optimizer if optimizer is not None else SGD(lr=0.01)

    # ------------------------------------------------------------------
    # Construção
    # ------------------------------------------------------------------
    def add_layer(self, layer: "Dense") -> None:
        self.layers.append(layer)

    # ------------------------------------------------------------------
    # Forward / Backward
    # ------------------------------------------------------------------
    def forward(self, X: np.ndarray) -> np.ndarray:
        output = X
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, y_true: np.ndarray, y_pred: np.ndarray) -> None:
        dA = self.loss_fn.derivative(y_true, y_pred)
        for layer in reversed(self.layers):
            dA = layer.backward(dA)

    def _update_weights(self) -> None:
        self.optimizer.update(self.layers)

    # ------------------------------------------------------------------
    # Treinamento
    # ------------------------------------------------------------------
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 100,
        val_data: tuple[np.ndarray, np.ndarray] | None = None,
        monitor: "TrainingMonitor | None" = None,
        verbose: bool = True,
        verbose_every: int = 10,
    ) -> None:
        for epoch in range(1, epochs + 1):
            y_pred_train = self.forward(X)
            train_loss = self.loss_fn(y, y_pred_train)
            self.backward(y, y_pred_train)
            
            # Agora não passamos mais o lr aqui, o otimizador já tem seus parâmetros
            self._update_weights()

            # --- métricas ---
            train_metric = self.metric_fn(y, y_pred_train) if self.metric_fn else None

            val_loss, val_metric = None, None
            if val_data is not None:
                X_val, y_val = val_data
                y_pred_val = self.forward(X_val)
                val_loss = self.loss_fn(y_val, y_pred_val)
                val_metric = self.metric_fn(y_val, y_pred_val) if self.metric_fn else None

            # --- monitor ---
            if monitor is not None:
                monitor.record(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    train_metric=train_metric,
                    val_metric=val_metric,
                    layers=self.layers,
                )

            # --- log ---
            if verbose and epoch % verbose_every == 0:
                msg = f"Época {epoch:>5} | loss: {train_loss:.6f}"
                if val_loss is not None:
                    msg += f"  val_loss: {val_loss:.6f}"
                if train_metric is not None:
                    msg += f"  {self.metric_fn}: {train_metric:.4f}"
                if val_metric is not None:
                    msg += f"  val_{self.metric_fn}: {val_metric:.4f}"
                print(msg)

    # ------------------------------------------------------------------
    # Inferência
    # ------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.forward(X)

    def __repr__(self) -> str:
        lines = ["NeuralNet("]
        for i, layer in enumerate(self.layers):
            lines.append(f"  [{i}] {layer}")
        lines.append(f"  loss={self.loss_fn}, metric={self.metric_fn}")
        lines.append(")")
        return "\n".join(lines)