"""Stage 4: classify 8x8 handwritten digits with minibatch SGD."""
import numpy as np
from sklearn.datasets import load_digits
from nn import Dense, ReLU, Net, softmax_cross_entropy, softmax

rng = np.random.default_rng(0)
d = load_digits()
X, y = d.data / 16.0, d.target          # scale pixels to [0,1] — inputs on a sane scale matter

perm = rng.permutation(len(X))          # shuffle, then hold out a test set we never train on
X, y = X[perm], y[perm]
split = int(0.8 * len(X))
Xtr, ytr, Xte, yte = X[:split], y[:split], X[split:], y[split:]
print(f"train {Xtr.shape}  test {Xte.shape}  classes {len(set(y))}")

net = Net([Dense(64, 64, rng), ReLU(),
           Dense(64, 32, rng), ReLU(),
           Dense(32, 10, rng)])         # 10 logits, no activation — softmax lives in the loss

def accuracy(Xs, ys):
    return (softmax(net.forward(Xs)).argmax(1) == ys).mean()

batch, lr = 32, 0.2
for epoch in range(31):
    order = rng.permutation(len(Xtr))   # reshuffle each epoch
    for i in range(0, len(Xtr), batch):
        idx = order[i:i + batch]        # a minibatch: noisy gradient, but ~50x more steps
        loss, dZ = softmax_cross_entropy(net.forward(Xtr[idx]), ytr[idx])
        net.backward(dZ)
        net.step(lr)
    if epoch % 10 == 0:
        print(f"epoch {epoch:2d}  batch loss {loss:.4f}  train acc {accuracy(Xtr,ytr):.3f}  TEST acc {accuracy(Xte,yte):.3f}")

print(f"\nfinal test accuracy: {accuracy(Xte, yte):.3f}")

# Look at what it gets wrong.
pred = softmax(net.forward(Xte)).argmax(1)
wrong = np.where(pred != yte)[0]
print(f"{len(wrong)} errors out of {len(yte)}")
for i in wrong[:3]:
    print(f"  true {yte[i]} predicted {pred[i]}")
    print("\n".join("".join(" .:-=+*#%@"[int(v * 9 / 16)] for v in row) for row in Xte[i].reshape(8, 8) * 16))
