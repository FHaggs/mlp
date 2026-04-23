"""Demos de MLP: classificação (MNIST) e regressão (Boston Housing)."""

from __future__ import annotations

import argparse

from matplotlib import pyplot as plt

from activations import Linear, ReLU, Softmax
from data_utils import load_boston_regression, load_mnist_multiclass
from layer import Dense
from losses import CategoricalCrossEntropy, MSE
from metrics import MulticlassAccuracy, R2
from monitor import TrainingMonitor
from neural_net import NeuralNet


def run_mnist_classification() -> None:
    X_train, y_train, X_val, y_val = load_mnist_multiclass(
        path="mnist.npz",
        train_limit=12000,
        val_limit=2000,
    )

    net = NeuralNet(loss=CategoricalCrossEntropy(), metric=MulticlassAccuracy(), optimizer="adam")
    net.add_layer(Dense(784, 128, ReLU()))
    net.add_layer(Dense(128, 64, ReLU()))
    net.add_layer(Dense(64, 10, Softmax()))

    print("Modo: classificação (MNIST)")
    print(net)

    monitor = TrainingMonitor(snapshot_every=2)
    net.fit(
        X_train,
        y_train,
        epochs=30,
        lr=0.05,
        batch_size=128,
        shuffle=True,
        val_data=(X_val, y_val),
        monitor=monitor,
        verbose=True,
        verbose_every=1,
    )

    sample_idx = 0
    pred = net.predict(X_val[sample_idx:sample_idx + 1])
    pred_class = int(pred.argmax(axis=1)[0])
    true_class = int(y_val[sample_idx].argmax())

    plt.figure(figsize=(4, 4))
    plt.imshow(X_val[sample_idx].reshape(28, 28), cmap="gray")
    plt.title(f"MNIST - True: {true_class}, Pred: {pred_class}")
    plt.axis("off")
    plt.show()

    monitor.plot_loss()
    monitor.plot_metric(metric_name="Accuracy")
    monitor.plot_activation_histograms(epochs_to_show=4)
    monitor.plot_gradient_histograms(epochs_to_show=4)


def run_boston_regression() -> None:
    X_train, y_train, X_val, y_val = load_boston_regression(path="Boston House Prices.csv")

    net = NeuralNet(loss=MSE(), metric=R2(), optimizer="adam")
    net.add_layer(Dense(X_train.shape[1], 32, ReLU()))
    net.add_layer(Dense(32, 16, ReLU()))
    net.add_layer(Dense(16, 1, Linear()))

    print("Modo: regressão (Boston Housing)")
    print(net)

    monitor = TrainingMonitor(snapshot_every=5)
    net.fit(
        X_train,
        y_train,
        epochs=250,
        lr=0.01,
        batch_size=32,
        shuffle=True,
        val_data=(X_val, y_val),
        monitor=monitor,
        verbose=True,
        verbose_every=10,
    )

    pred_val = net.predict(X_val)

    plt.figure(figsize=(6, 5))
    plt.scatter(y_val.ravel(), pred_val.ravel(), alpha=0.65, edgecolors="none")
    min_v = min(float(y_val.min()), float(pred_val.min()))
    max_v = max(float(y_val.max()), float(pred_val.max()))
    plt.plot([min_v, max_v], [min_v, max_v], "r--", linewidth=1.5)
    plt.xlabel("MEDV real")
    plt.ylabel("MEDV previsto")
    plt.title("Boston Housing - Real vs Previsto")
    plt.tight_layout()
    plt.show()

    monitor.plot_loss()
    monitor.plot_metric(metric_name="R2")
    monitor.plot_activation_histograms(epochs_to_show=4)
    monitor.plot_gradient_histograms(epochs_to_show=4)


def main() -> None:
    parser = argparse.ArgumentParser(description="MLP demo: classificação (MNIST) e regressão (Boston)")
    parser.add_argument(
        "--mode",
        choices=["mnist", "boston"],
        default="mnist",
        help="Escolha o modo de teste: 'mnist' para classificação, 'boston' para regressão.",
    )
    args = parser.parse_args()

    if args.mode == "mnist":
        run_mnist_classification()
    else:
        run_boston_regression()


if __name__ == "__main__":
    main()

