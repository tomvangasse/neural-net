# SGD, or Stochastic Gradient Descent, 
# means computing the gradient on a random sample of the data instead of the whole set

class SGD:
    def __init__(self, lr):
        self.lr = lr        # learning rate

    def step(self, params_and_grads):
        # updates every parameter in the params_and_grads pairs array
        for param, grad in params_and_grads:
            param -= self.lr * grad     # updates the params_and_grads array in place

            