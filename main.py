"""Demo: classificação multiclasse (0-9) com MNIST (.npz)."""

from matplotlib import pyplot as plt

from activations import ReLU, Softmax
from data_utils import load_mnist_multiclass
from layer import Dense
from losses import CategoricalCrossEntropy
from metrics import MulticlassAccuracy
from monitor import TrainingMonitor
from neural_net import NeuralNet

# ------------------------------------------------------------------
# Carregamento do MNIST
# ------------------------------------------------------------------
X_train, y_train, X_val, y_val = load_mnist_multiclass(
    path="mnist.npz",
    train_limit=12000,
    val_limit=2000,
)

# ------------------------------------------------------------------
# Construção da rede
# ------------------------------------------------------------------
net = NeuralNet(loss=CategoricalCrossEntropy(), metric=MulticlassAccuracy(), optimizer="adam")
net.add_layer(Dense(784, 128, ReLU()))
net.add_layer(Dense(128, 64, ReLU()))
net.add_layer(Dense(64, 10, Softmax()))

print(net)

# ------------------------------------------------------------------
# Treinamento com monitor
# ------------------------------------------------------------------
monitor = TrainingMonitor(snapshot_every=2)

net.fit(
    X_train, y_train,
    epochs=30,
    lr=0.05,
    batch_size=128,
    shuffle=True,
    val_data=(X_val, y_val),
    monitor=monitor,
    verbose=True,
    verbose_every=1,
)

# Plot value from validation set and see the prediction
sample_idx = 0
pred = net.predict(X_val[sample_idx:sample_idx + 1])
pred_class = int(pred.argmax(axis=1)[0])
true_class = int(y_val[sample_idx].argmax())
plt.imshow(X_val[sample_idx].reshape(28, 28), cmap="gray")
plt.title(f"True: {true_class}, Pred: {pred_class}")
plt.axis("off")
plt.show()


# ------------------------------------------------------------------
# Visualizações
# ------------------------------------------------------------------
monitor.plot_loss()
monitor.plot_metric(metric_name="Accuracy")
monitor.plot_activation_histograms(epochs_to_show=4)
monitor.plot_gradient_histograms(epochs_to_show=4)

