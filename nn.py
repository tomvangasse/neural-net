"""A tiny neural-net library. Everything here was derived in stages 1-3."""
import numpy as np

class Dense:
    def __init__(self, n_in, n_out, rng):
        self.W = rng.normal(scale=np.sqrt(2 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)
    def forward(self, X):
        self.X = X
        return X @ self.W + self.b
    def backward(self, dZ):
        self.dW = self.X.T @ dZ
        self.db = dZ.sum(axis=0)
        return dZ @ self.W.T
    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class ReLU:
    """max(0, x). Cheap, and its gradient is 1 or 0 — no vanishing for positive inputs."""
    def forward(self, X):
        self.mask = X > 0
        return X * self.mask
    def backward(self, dA):
        return dA * self.mask      # gradient flows only where the unit was active
    def step(self, lr):
        pass

class Tanh:
    def forward(self, X):
        self.out = np.tanh(X); return self.out
    def backward(self, dA):
        return dA * (1 - self.out ** 2)
    def step(self, lr):
        pass

class Net:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, X):
        for L in self.layers: X = L.forward(X)
        return X
    def backward(self, dZ):
        for L in reversed(self.layers): dZ = L.backward(dZ)
    def step(self, lr):
        for L in self.layers: L.step(lr)

def softmax(z):
    z = z - z.max(axis=1, keepdims=True)      # subtract max for numerical stability
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def softmax_cross_entropy(logits, y):
    """Returns (loss, dLoss/dlogits). y is integer class labels."""
    p = softmax(logits)
    n = len(y)
    loss = -np.log(p[np.arange(n), y] + 1e-9).mean()
    dZ = p.copy()
    dZ[np.arange(n), y] -= 1                  # again: (prediction - truth)
    return loss, dZ / n
