# A neural network from scratch using NumPy

## How the model works
### Training
1) An input X, in our case a grid of 28x28 pixels
   
2) X is pushed forward through the layers, Dense, ReLU, Dense, to get the raw scores
   Dense holds the learnable numbers: W (weights) and b (bias), it computes X @ W + b
   ReLU allows for non-linear correlations, without it, the two Dense layers would collapse into one large linear function
  
3) the raw scores are used to calculate the loss, a number that represents how wrong the models predictions are on a batch
   This is done using the softmax function, which turns the raw scores into probabilities that sum to 1
   Loss is calculated according to how little probability it gave the correct answer. this is cross-entropy

4) next is back propagation
   The network is a chain of functions, so the loss depends on early layers before later ones
   The chain rule can let you calculate each layers gradient using the one that came after it
   This was tested with slow numerical gradient calculations in unit tests
   
5) the optimiser uses the gradients for each parameter in each layer, and changes each value in place based on a learning rate
   param -= lr * grad

   
