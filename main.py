import numpy as np

# Teste manual sem hiperparametros

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def loss(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

X = np.array([[0.5, 0.1], 
              [0.2, 0.6]])

y = np.array([[0.6, 0.8]])

# Layer 3x2
layer_1 = np.random.rand(3, 2)
layer_2 = np.random.rand(1, 3)

print("Layer 1 weights:\n", layer_1)
print("Layer 2 weights:\n", layer_2)

Z1 = X @ layer_1.T # @ é o operador de multiplicação de matrizes
A1 = sigmoid(Z1)
Z2 = A1 @ layer_2.T
A2 = sigmoid(Z2)

print("Output:\n", A2)

