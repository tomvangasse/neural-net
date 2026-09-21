import numpy as np


def numerical_gradient(f, x, h=1e-5):
    """Estimate d f() / d x by nudging each element of x up and down."""
    grad = np.zeros_like(x)
    for idx in np.ndindex(*x.shape):
        original = x[idx]
        x[idx] = original + h
        f_plus = f()
        x[idx] = original - h
        f_minus = f()
        x[idx] = original
        grad[idx] = (f_plus - f_minus) / (2 * h)
    return grad


def relative_error(a, b):
    """below 1e-6 is a pass, above 1e-3 is a bug"""
    largest = 0
    for idx in np.ndindex(*a.shape):
        current = np.abs(a[idx] - b[idx]) / np.maximum(np.abs(a[idx]) + np.abs(b[idx]), 1e-8)
        largest = np.maximum(largest, current)

    return float(largest)


def check_layer_gradients(layer, X, rng):
    """Compare a layer's backward() with numerical gradients.

    Returns {name: relative_error} for the input "X" and each parameter.
    """
    R = rng.normal(size=layer.forward(X).shape)

    def loss():
        return float(np.sum(layer.forward(X) * R))

    targets = {"X": X, **layer.params}
    numeric = {name: numerical_gradient(loss, arr) for name, arr in targets.items()}

    layer.forward(X)
    dX = layer.backward(R)
    analytic = {"X": dX, **layer.grads}

    return {name: relative_error(analytic[name], numeric[name]) for name in targets}