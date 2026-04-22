"""Demo: classificação binária com MNIST (.npz)."""

from activations import ReLU, Sigmoid
from data_utils import load_mnist_binary
from layer import Dense
from losses import BinaryCrossEntropy
from metrics import Accuracy
from monitor import TrainingMonitor
from neural_net import NeuralNet

# ------------------------------------------------------------------
# Carregamento do MNIST
# ------------------------------------------------------------------
X_train, y_train, X_val, y_val = load_mnist_binary(
    path="mnist.npz",
    positive_digit=0,
    train_limit=12000,
    val_limit=2000,
)

# ------------------------------------------------------------------
# Construção da rede
# ------------------------------------------------------------------
net = NeuralNet(loss=BinaryCrossEntropy(), metric=Accuracy())
net.add_layer(Dense(784, 128, ReLU()))
net.add_layer(Dense(128, 64, ReLU()))
net.add_layer(Dense(64, 1, Sigmoid()))

print(net)

# ------------------------------------------------------------------
# Treinamento com monitor
# ------------------------------------------------------------------
monitor = TrainingMonitor(snapshot_every=2)

net.fit(
    X_train, y_train,
    epochs=12,
    lr=0.05,
    batch_size=128,
    shuffle=True,
    val_data=(X_val, y_val),
    monitor=monitor,
    verbose=True,
    verbose_every=1,
)

# ------------------------------------------------------------------
# Visualizações
# ------------------------------------------------------------------
monitor.plot_loss()
monitor.plot_metric(metric_name="Accuracy")
monitor.plot_activation_histograms(epochs_to_show=4)
monitor.plot_gradient_histograms(epochs_to_show=4)

