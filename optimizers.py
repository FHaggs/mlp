import numpy as np

class SGD:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr

    def update(self, layers) -> None:
        for layer in layers:
            layer.W -= self.lr * layer.grads["dW"]
            layer.b -= self.lr * layer.grads["db"]

class Adam:
    def __init__(
        self, 
        lr: float = 0.001, 
        beta1: float = 0.9, 
        beta2: float = 0.999, 
        eps: float = 1e-8
    ) -> None:
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0 # Contador de passos para correção de viés
        
        # Dicionários para guardar os momentos m e v de cada camada
        self.m_W, self.v_W = {}, {}
        self.m_b, self.v_b = {}, {}

    def update(self, layers) -> None:
        self.t += 1
        for i, layer in enumerate(layers):
            # Inicializa os momentos se for a primeira vez
            if i not in self.m_W:
                self.m_W[i] = np.zeros_like(layer.W)
                self.v_W[i] = np.zeros_like(layer.W)
                self.m_b[i] = np.zeros_like(layer.b)
                self.v_b[i] = np.zeros_like(layer.b)

            # --- Atualização dos Pesos (W) ---
            dW = layer.grads["dW"]
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dW
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * (dW**2)
            
            # Correção de viés (importante no início do treino)
            m_hat = self.m_W[i] / (1 - self.beta1**self.t)
            v_hat = self.v_W[i] / (1 - self.beta2**self.t)
            
            layer.W -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # --- Atualização dos Vieses (b) ---
            db = layer.grads["db"]
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * (db**2)
            
            m_hat_b = self.m_b[i] / (1 - self.beta1**self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2**self.t)
            
            layer.b -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)