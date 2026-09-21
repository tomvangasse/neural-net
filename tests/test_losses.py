import numpy as np

from nn.gradcheck import numerical_gradient, relative_error
from nn.losses import softmax, softmax_cross_entropy

TOL = 1e-6


def test_softmax_rows_add_to_one():
    p = softmax(np.array([[1000.0, 1001.0, 999.0]]))
    assert np.allclose(p.sum(axis=1), 1)


def test_loss_with_equal_scores():
    logits = np.zeros((4, 10))
    y = np.array([2,5,1,7])
    loss, _ = softmax_cross_entropy(logits, y)
    np.testing.assert_allclose(loss, -np.log(0.1))


def test_loss_gradient():
    rng = np.random.default_rng(42)
    logits = rng.normal(size=(5, 4))
    y = np.array([0, 3, 1, 2, 0])
    _, dZ = softmax_cross_entropy(logits, y)
    numeric = numerical_gradient(lambda: softmax_cross_entropy(logits, y)[0], logits)
    assert relative_error(dZ, numeric) < TOL