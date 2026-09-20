"""Stage 3: a real multi-layer net with backprop. Layers as objects."""
import numpy as np

# NOTE: convention change from stage 1 — W is now (n_in, n_out) so forward
# is the tidy `X @ W + b`. Shapes are the #1 source of bugs; pick one and stick to it.

class Dense:
    def __init__(self, n_in, n_out, rng):
        # He init: scale by sqrt(2/n_in) so signal variance doesn't explode/vanish with depth.
        self.W = rng.normal(scale=np.sqrt(2 / n_in), size=(n_in, n_out))
        self.b = np.zeros(n_out)

    def forward(self, X):
        self.X = X                      # cached — backward needs it
        return X @ self.W + self.b

    def backward(self, dZ):             # dZ = dLoss/d(output of this layer)
        self.dW = self.X.T @ dZ         # how much each weight contributed to the error
        self.db = dZ.sum(axis=0)
        return dZ @ self.W.T            # error signal handed to the layer below

    def step(self, lr):
        self.W -= lr * self.dW
        self.b -= lr * self.db

class Tanh:
    def forward(self, X):
        self.out = np.tanh(X)
        return self.out
    def backward(self, dA):
        return dA * (1 - self.out ** 2)  # local derivative, multiplied in
    def step(self, lr):
        pass

class Net:
    def __init__(self, layers):
        self.layers = layers
    def forward(self, X):
        for L in self.layers:
            X = L.forward(X)
        return X
    def backward(self, dZ):
        for L in reversed(self.layers):   # <- this reversal IS backpropagation
            dZ = L.backward(dZ)
    def step(self, lr):
        for L in self.layers:
            L.step(lr)

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# XOR: not linearly separable. One neuron cannot do this. Ever.
X = np.array([[0., 0.], [0., 1.], [1., 0.], [1., 1.]])
y = np.array([[0.], [1.], [1.], [0.]])

rng = np.random.default_rng(3)
net = Net([Dense(2, 8, rng), Tanh(), Dense(8, 1, rng)])

for step in range(2001):
    logits = net.forward(X)
    p = sigmoid(logits)
    loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))

    net.backward((p - y) / len(y))   # seed the chain with dL/dz from stage 2
    net.step(lr=0.5)

    if step % 500 == 0:
        print(f"step {step:4d}  loss {loss:.4f}")

print("\npredictions:", np.round(sigmoid(net.forward(X)).ravel(), 3), " target:", y.ravel())
