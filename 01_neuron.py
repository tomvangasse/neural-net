"""Stage 1: a neuron, a layer, and why the nonlinearity matters."""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# --- one neuron ------------------------------------------------------
x = np.array([0.5, -1.2, 3.0])   # 3 input features
w = np.array([0.8,  0.1, -0.4])  # one weight per input feature
b = 0.2

z = w @ x + b                    # pre-activation: a dot product + bias
a = sigmoid(z)                   # activation: squashed into (0, 1)
print(f"one neuron:  z = {z:.3f}   a = {a:.3f}")

# --- a layer of 4 neurons -------------------------------------------
# Each neuron has its own row of weights -> W is (n_out, n_in).
rng = np.random.default_rng(0)
W = rng.normal(size=(4, 3))
B = np.zeros(4)

A = sigmoid(W @ x + B)           # 4 neurons, all computed in one matmul
print("layer of 4:", np.round(A, 3))

# --- a batch of inputs ----------------------------------------------
# Real training pushes many examples through at once. Convention:
# X is (n_samples, n_features), so the layer is X @ W.T + B.
X = rng.normal(size=(5, 3))      # 5 examples
A = sigmoid(X @ W.T + B)         # (5, 3) @ (3, 4) -> (5, 4)
print("batch of 5 ->", A.shape)

# --- why we need the nonlinearity ------------------------------------
# Stack two LINEAR layers and you get... one linear layer.
W1, W2 = rng.normal(size=(4, 3)), rng.normal(size=(2, 4))
two_layers = (X @ W1.T) @ W2.T
equivalent = X @ (W2 @ W1).T     # a single matrix does the same job
print("linear stack collapses:", np.allclose(two_layers, equivalent))
