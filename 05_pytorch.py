"""Stage 5: the identical network in PyTorch.  pip install torch to run."""
import torch, torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

d = load_digits()
Xtr, Xte, ytr, yte = train_test_split(d.data / 16.0, d.target, test_size=0.2, random_state=0)
Xtr, Xte = torch.tensor(Xtr, dtype=torch.float32), torch.tensor(Xte, dtype=torch.float32)
ytr, yte = torch.tensor(ytr), torch.tensor(yte)

model = nn.Sequential(          # <- our Net([...]), same shape
    nn.Linear(64, 64), nn.ReLU(),
    nn.Linear(64, 32), nn.ReLU(),
    nn.Linear(32, 10))

loss_fn = nn.CrossEntropyLoss()             # <- our softmax_cross_entropy
opt = torch.optim.SGD(model.parameters(), lr=0.2)   # <- our .step(lr)

for epoch in range(31):
    for i in torch.randperm(len(Xtr)).split(32):
        loss = loss_fn(model(Xtr[i]), ytr[i])
        opt.zero_grad()      # torch ACCUMULATES grads; forgetting this is the classic bug
        loss.backward()      # <- our net.backward(), built by autograd from the forward pass
        opt.step()
    if epoch % 10 == 0:
        with torch.no_grad():
            acc = (model(Xte).argmax(1) == yte).float().mean()
        print(f"epoch {epoch:2d}  loss {loss.item():.4f}  TEST acc {acc:.3f}")
