import numpy as np
from nn.model import Model
from nn.layers import Dense, ReLU
from nn.optimisers import SGD
from nn.losses import softmax_cross_entropy


def test_sgd_updates_parameters():
    param = np.array([1.0, 2.0])   
    grad = np.array([0.5, -1.0])

    SGD(lr=0.1).step([(param, grad)])   # runs one step of the optimiser on one set of parameters and gradients
    np.testing.assert_allclose(param, [0.95, 2.1])  # each value should have gone down by (lr * grad)

def test_sgd_reduces_loss():
    # tests the Model reduces loss on a small dataset 
    rng = np.random.default_rng(0)
    model = Model([Dense(4, 8, rng=rng), ReLU(), Dense(8, 3, rng=rng)])

    X = rng.normal(size=(20, 4))    # small dataset
    y = rng.integers(0, 3, size=20)
    opt = SGD(lr=0.1)   # creates the optimiser

    first_loss, _ = softmax_cross_entropy(model.forward(X), y)  # gets the loss of the first cycle
    for _ in range(100): 
        loss, dZ = softmax_cross_entropy(model.forward(X), y)
        model.backward(dZ)
        opt.step(model.params_and_grads())

    assert loss < first_loss    # checks the final loss is less than the first loss