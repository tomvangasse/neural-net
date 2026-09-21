import numpy as np

from nn.model import Model
from nn.gradcheck import numerical_gradient, relative_error
from nn.losses import softmax_cross_entropy
from nn.layers import Dense, ReLU

TOL = 1e-6


def test_model_forward_shape():
    rng = np.random.default_rng(42)
    model = Model([Dense(4, 8, rng=rng), ReLU(), Dense(8, 3, rng=rng)])
    X = rng.normal(size=(5, 4))
    out = model.forward(X)
    assert out.shape == (5,3)

def test_model_gradients():
    # tests the model using a fixed seed random data set (the data is the same every time)
    # compares the analytical gradients (stored in the layers) to the numerical ones from gradcheck


    # setup the model and data
    rng = np.random.default_rng(0)
    model = Model([Dense(4, 8, rng=rng), ReLU(), Dense(8, 3, rng=rng)])
    X = rng.normal(size=(6, 4))
    y = rng.integers(0, 3, size=6)

    # analytical route: backpropeggate using the model
    logits = model.forward(X)
    loss, dZ = softmax_cross_entropy(logits, y)
    model.backward(dZ)

    for param, grad in model.params_and_grads():

        # numerical route - independent of backpropeggation
        numeric = numerical_gradient(
            lambda: softmax_cross_entropy(model.forward(X), y)[0], param
        )

        # if any dont match, the test fails
        assert relative_error(grad, numeric) < TOL, "a gradient in the model is wrong"