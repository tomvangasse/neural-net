import numpy as np


class Layer:
    def __init__(self):
        self.params = {} # learnable numbers, e.g. {"W": ..., "b": ...}
        self.grads = {} # their gradients, same keys

    def forward(self, X): raise NotImplementedError
    def backward(self, dout): raise NotImplementedError


class Dense(Layer):
    def __init__(self, n_in, n_out, rng=None):
        super().__init__()
        if rng is None:
            rng = np.random.default_rng()
        self.params["W"] = rng.normal(0.0, np.sqrt(2.0 / n_in), size=(n_in, n_out))
        self.params["b"] = np.zeros(n_out)
        self.X = None

    def forward(self, X):
        self.X = X
        return X @ self.params["W"] + self.params["b"]

    def backward(self, dout):
        self.grads["W"] = self.X.T @ dout           # Each weight's gradient is the input it was multiplied by, times the error on the output it fed into
        self.grads["b"] = dout.sum(axis=0)          # The bias was added to every example, so its gradient is the total error across all of them
        return dout @ self.params["W"].T            # Pass the error backwards through the weights, to the layer below

