"""Stage 2: train ONE neuron with gradient descent, gradients derived by hand."""
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

rng = np.random.default_rng(1)

# Toy problem: is the point above the line y = x?  (linearly separable)
X = rng.normal(size=(200, 2))
y = (X[:, 1] > X[:, 0]).astype(float)

w = np.zeros(2)
b = 0.0
lr = 0.5

def forward(X, w, b):
    return sigmoid(X @ w + b)                      # predicted probability

def loss(p, y):                                     # binary cross-entropy
    eps = 1e-9
    return -np.mean(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps))

# The chain rule, written out:
#   L depends on p, p depends on z, z depends on w.
#   dL/dp = (p - y) / (p(1-p))      <- derivative of cross-entropy
#   dp/dz = p(1 - p)                <- derivative of sigmoid
#   => dL/dz = p - y                <- the messy parts cancel. Remember this one.
#   dz/dw = x   =>  dL/dw = mean over batch of (p - y) * x
for step in range(301):
    p = forward(X, w, b)
    dz = (p - y) / len(y)          # (n,)  prediction error, averaged
    dw = X.T @ dz                  # (2,)  push each weight by error * its input
    db = dz.sum()

    w -= lr * dw                   # step downhill
    b -= lr * db

    if step % 100 == 0:
        acc = ((p > 0.5) == y).mean()
        print(f"step {step:3d}  loss {loss(p, y):.4f}  acc {acc:.3f}  w {np.round(w,2)}")

# Sanity check: compare the hand-derived gradient to a numerical one.
def num_grad(w, b, i, h=1e-6):
    wp, wm = w.copy(), w.copy()
    wp[i] += h; wm[i] -= h
    return (loss(forward(X, wp, b), y) - loss(forward(X, wm, b), y)) / (2 * h)

p = forward(X, w, b)
dz = (p - y) / len(y)
analytic = X.T @ dz
numeric  = np.array([num_grad(w, b, i) for i in range(2)])
print("\nanalytic grad:", np.round(analytic, 8))
print("numeric  grad:", np.round(numeric, 8))
