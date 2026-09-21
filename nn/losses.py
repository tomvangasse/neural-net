# loss functions, they return the loss and their gradients
# logits are the raw score values before they get turned into probabilities
import numpy as np


def softmax(z): 
    # turns raw scores into probabilities 
    # each number should be positive and sum to 1
    z = z - z.max(axis=1, keepdims=True) # subtract each rows maximum before exponenting to avoid overflows
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)

def softmax_cross_entropy(logits, y):
    # Return (loss, dlogits) for raw scores `logits` and integer labels `y`
    n = len(y)
    p = softmax(logits)
    loss = -np.log(p[np.arange(n), y] + 1e-9).mean()
    dZ = p.copy()
    dZ[np.arange(n), y] -= 1
    return loss, dZ / n


