"""Demo: regressão sintética com MLP."""

import numpy as np

from activations import ReLU, Linear
from layer import Dense
from losses import MSE
from metrics import R2
from monitor import TrainingMonitor
from neural_net import NeuralNet

# ------------------------------------------------------------------
# Dados sintéticos: y = 0.5*x1 + 2*x2 + ruído
# ------------------------------------------------------------------
np.random.seed(42)
N = 200
X = np.random.randn(N, 2)
y = (0.5 * X[:, 0:1] + 2.0 * X[:, 1:2]) + 0.1 * np.random.randn(N, 1)

split = int(0.8 * N)
X_train, X_val = X[:split], X[split:]
y_train, y_val = y[:split], y[split:]

# ------------------------------------------------------------------
# Construção da rede
# ------------------------------------------------------------------
net = NeuralNet(loss=MSE(), metric=R2())
net.add_layer(Dense(2, 16, ReLU()))
net.add_layer(Dense(16, 8, ReLU()))
net.add_layer(Dense(8, 1, Linear()))

print(net)

# ------------------------------------------------------------------
# Treinamento com monitor
# ------------------------------------------------------------------
monitor = TrainingMonitor(snapshot_every=20)

net.fit(
    X_train, y_train,
    epochs=200,
    lr=0.01,
    val_data=(X_val, y_val),
    monitor=monitor,
    verbose=True,
    verbose_every=20,
)

# ------------------------------------------------------------------
# Visualizações
# ------------------------------------------------------------------
monitor.plot_loss()
monitor.plot_metric(metric_name="R²")
monitor.plot_activation_histograms(epochs_to_show=4)
monitor.plot_gradient_histograms(epochs_to_show=4)

