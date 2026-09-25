import numpy as np
from sklearn.datasets import fetch_openml

from nn.layers import Dense, ReLU
from nn.losses import softmax_cross_entropy
from nn.model import Model
from nn.optimisers import SGD

rng = np.random.default_rng(0)

X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False, parser='liac-arff')
                    # fetches the mnist dataset and stores it as X
                    # the 'liac-arff' parser avoids the pandas dependency fetch_openml would otherwise ask for

X = X / 255.0       # scales down the images to [0, 1]
y = y.astype(int)   # converts the text that is given to integers

perm = rng.permutation(len(X))  # gets a random permutation of X
X, y = X[perm], y[perm]         # permutates X and y by the same amount so still they match

split = int(0.8 * len(X))                   # generates a split for the randomised dataset
X_train, y_train = X[:split], y[:split]     # 80% for training
X_test, y_test = X[split:], y[split:]       # 20% for testing the model

model = Model([         # creates the model
    Dense(784, 128, rng=rng),   # 784 is the amount of data in one image, and 128 is the arbitrary size of the hidden layer
    ReLU(),
    Dense(128, 64, rng=rng),    # the next layers input must match the last ones output, so 128. 64 is arbitrary
    ReLU(),
    Dense(64, 10, rng=rng),     # 10 is the number of outputs we need, to represent each digit (0,..,9)
])
optimizer = SGD(lr=0.2)         # creates the optimiser with a learning rate of 0.2

def accuracy(X, y):
    logits = model.forward(X)               # pushes a dataset through the model to get scores
    predictions = np.argmax(logits, axis=1) # looks along each row of the logits array to find the highest scores
    return (predictions == y).mean()        # tests each prediction against each correct digit, and gets the average score

batch_size = 64     # number of images that are proccessed at a time
epochs = 20         # how many times the whole training set gets processed

for epoch in range(epochs):
    order = rng.permutation(len(X_train))       # a random shuffle of X_train, generated per epoch
                                                # this is to stop the model from simply learning where each digit is in the sequence
    for start in range(0, len(X_train), batch_size):    # 'start' marks where each batch begins (0, 64, 128,...)
        batch_idx = order[start:(start + batch_size)]   # the current batch is from the 'start' marker, up to the 'start' marker + the batch size
        X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]   # pulls out the 64 images as well as their matching labels

        logits = model.forward(X_batch)     # pushes the batch through the model
        loss, dZ = softmax_cross_entropy(logits, y_batch)   # calculates the loss of the forward pass
        model.backward(dZ)                  # sends dZ backward through each layer, filling in each layer's own W and b gradients
        optimizer.step(model.params_and_grads())    # collects the p and g pairs and changes each parameter

    if epoch % 4 == 0:
        print(f"epoch {epoch:2d}  loss {loss:.4f}  train acc {accuracy(X_train, y_train):.3f}  test acc {accuracy(X_test, y_test):.3f}")

print(f"\nfinal test accuracy: {accuracy(X_test, y_test):.3f}")