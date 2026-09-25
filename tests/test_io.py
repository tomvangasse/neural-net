import numpy as np

from nn.io import save_model, load_model
from nn.layers import Dense, ReLU
from nn.model import Model


def test_save_and_load_roundtrip(tmp_path): # tmp_path: pytest's temporary path for testing
    original = Model([Dense(4, 8, rng=np.random.default_rng(0)), ReLU(), Dense(8, 3, rng=np.random.default_rng(0))])
        # original model with a set of random numbers

    X = np.random.default_rng(1).normal(size=(5, 4)) # random data to put througt the model
    before = original.forward(X)    # storing the original model output

    path = tmp_path / "model.npz"   # makes a file inside the temporary folder
    save_model(original, path)

    fresh = Model([Dense(4, 8, rng=np.random.default_rng(999)), ReLU(), Dense(8, 3, rng=np.random.default_rng(999))])
    garbage_output = fresh.forward(X)   # garbage output of the initialised fresh model
    load_model(fresh, path)
    after_load = fresh.forward(X)

    assert not np.allclose(before, garbage_output)  # the garbage output shouldn't match the original
    np.testing.assert_allclose(after_load, before)  
        # the output of the fresh model after loading the original parameters should match the original output