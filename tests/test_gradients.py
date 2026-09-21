import numpy as np
import pytest

from nn.gradcheck import check_layer_gradients, numerical_gradient, relative_error
from nn.layers import Dense

TOL = 1e-6


def test_numerical_gradient_matches_known_derivative():
    x = np.array([[1.0, -2.0], [3.0, 0.5]])
    g = numerical_gradient(lambda: float(np.sum(x ** 2)), x)
    assert relative_error(g, 2 * x) < TOL


def test_forward_matches_known_number():
    layer = Dense(3, 2)
    layer.params["W"] = np.array([[1.0, 0.0],[0.0, 1.0],[1.0, 1.0]])
    layer.params["b"] = np.array([0.5, -0.5])
    X = np.array([[1,2,3]])
    out = layer.forward(X)
    assert out.shape == (1,2)
    np.testing.assert_allclose(out, [[4.5, 4.5]])


@pytest.mark.parametrize("batch,n_in,n_out", [(1, 3, 2), (4, 5, 3), (7, 2, 6)])
def test_dense_backward_gradients(batch, n_in, n_out):
    rng = np.random.default_rng(42)
    layer = Dense(n_in, n_out, rng=rng)
    X = rng.normal(size=(batch, n_in))
    errors = check_layer_gradients(layer, X, rng)
    for name, err in errors.items():    # check each error is below a safe margin
        assert err < TOL, f"Gradient for {name} is incorrect: relative error is {err}"