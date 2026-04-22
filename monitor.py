"""TrainingMonitor — coleta e visualiza estatísticas de treinamento.

Uso típico
----------
monitor = TrainingMonitor(snapshot_every=10)
net.fit(..., monitor=monitor)
monitor.plot_loss()
monitor.plot_metric()
monitor.plot_activation_histograms()
monitor.plot_gradient_histograms()
"""


from dataclasses import dataclass
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from layer import Dense


@dataclass
class _Snapshot:
    """Dados de uma época com snapshot de ativações/gradientes."""
    epoch: int
    activations: dict[str, np.ndarray]   # nome_camada → valores achatados
    gradients: dict[str, np.ndarray]     # nome_camada → dW achatado


class TrainingMonitor:
    """Coleta métricas por época e oferece métodos de visualização.

    Parâmetros
    ----------
    snapshot_every : a cada quantas épocas capturar ativações e gradientes.
                     Use 0 para nunca capturar (desativa histogramas).
    """

    def __init__(self, snapshot_every: int = 10) -> None:
        self.snapshot_every = snapshot_every

        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.train_metrics: list[float] = []
        self.val_metrics: list[float] = []
        self.epochs: list[int] = []

        self._snapshots: list[_Snapshot] = []

    # ------------------------------------------------------------------
    # Interface chamada pelo NeuralNet.fit()
    # ------------------------------------------------------------------
    def record(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float | None,
        train_metric: float | None,
        val_metric: float | None,
        layers: list["Dense"],
    ) -> None:
        self.epochs.append(epoch)
        self.train_losses.append(train_loss)
        if val_loss is not None:
            self.val_losses.append(val_loss)
        if train_metric is not None:
            self.train_metrics.append(train_metric)
        if val_metric is not None:
            self.val_metrics.append(val_metric)

        if self.snapshot_every > 0 and epoch % self.snapshot_every == 0:
            self._snapshots.append(self._capture(epoch, layers))

    # ------------------------------------------------------------------
    # Plots públicos
    # ------------------------------------------------------------------
    def plot_loss(self) -> None:
        """Custo vs Época (treino e validação)."""
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.epochs, self.train_losses, label="Treino")
        if self.val_losses:
            ax.plot(self.epochs, self.val_losses, label="Validação")
        ax.set_xlabel("Época")
        ax.set_ylabel("Loss")
        ax.set_title("Função de Custo vs Época")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_metric(self, metric_name: str = "Métrica") -> None:
        """Métrica de avaliação vs Época (treino e validação)."""
        if not self.train_metrics:
            print("Nenhuma métrica registrada.")
            return
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(self.epochs, self.train_metrics, label="Treino")
        if self.val_metrics:
            ax.plot(self.epochs, self.val_metrics, label="Validação")
        ax.set_xlabel("Época")
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} vs Época")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_activation_histograms(self, epochs_to_show: int = 4) -> None:
        """Histogramas das ativações por camada em snapshots selecionados."""
        self._plot_histograms(
            key="activations",
            title_prefix="Ativações",
            epochs_to_show=epochs_to_show,
        )

    def plot_gradient_histograms(self, epochs_to_show: int = 4) -> None:
        """Histogramas dos gradientes (dW) por camada em snapshots selecionados."""
        self._plot_histograms(
            key="gradients",
            title_prefix="Gradientes (dW)",
            epochs_to_show=epochs_to_show,
        )

    # ------------------------------------------------------------------
    # Helpers internos
    # ------------------------------------------------------------------
    def _capture(self, epoch: int, layers: list["Dense"]) -> _Snapshot:
        activations: dict[str, np.ndarray] = {}
        gradients: dict[str, np.ndarray] = {}
        for i, layer in enumerate(layers):
            name = f"Camada {i + 1}"
            if layer._cache.get("A") is not None:
                activations[name] = layer._cache["A"].ravel()
            if layer.grads.get("dW") is not None:
                gradients[name] = layer.grads["dW"].ravel()
        return _Snapshot(epoch=epoch, activations=activations, gradients=gradients)

    def _plot_histograms(
        self,
        key: str,
        title_prefix: str,
        epochs_to_show: int,
    ) -> None:
        if not self._snapshots:
            print("Nenhum snapshot disponível. Verifique snapshot_every.")
            return

        # Seleciona snapshots distribuídos uniformemente
        total = len(self._snapshots)
        indices = np.linspace(0, total - 1, min(epochs_to_show, total), dtype=int)
        selected = [self._snapshots[i] for i in indices]

        layer_names = list(getattr(selected[0], key).keys())
        n_layers = len(layer_names)
        if n_layers == 0:
            print(f"Nenhum dado de {title_prefix} para plotar.")
            return

        n_cols = len(selected)
        fig, axes = plt.subplots(n_layers, n_cols, figsize=(4 * n_cols, 3 * n_layers))

        # Garante array 2D mesmo com 1 linha ou 1 coluna
        if n_layers == 1:
            axes = axes[np.newaxis, :]
        if n_cols == 1:
            axes = axes[:, np.newaxis]

        for col, snap in enumerate(selected):
            data_dict: dict[str, np.ndarray] = getattr(snap, key)
            for row, lname in enumerate(layer_names):
                ax = axes[row, col]
                values = data_dict.get(lname)
                if values is not None and values.size > 0:
                    ax.hist(values, bins=30, color="steelblue", edgecolor="none")
                ax.set_title(f"{lname}\nÉpoca {snap.epoch}", fontsize=8)
                ax.set_xlabel(title_prefix, fontsize=7)
                ax.tick_params(labelsize=7)

        fig.suptitle(f"Histogramas — {title_prefix}", fontsize=11)
        plt.tight_layout()
        plt.show()
