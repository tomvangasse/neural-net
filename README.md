# A digit-recognising neural network built from scratch using NumPy
Using no ML frameworks.

I used AI tools to help with teaching me the basics, and to assist with checking and generating code. 

## Why I built this
I built this to teach myself how simple neural networks actually work under all the abstraction of modern ML frameworks. I also took this project to learn how to make effective use of AI coding assistants.

## Results
Final test accuracy is 97.8%, with a training time of 11 seconds on my machine.
The model is trained on 56,000 MNIST images, with 14,000 being used to test the model after training.

## How the model works
### Training
1) An input X, in our case a grid of 28x28 pixels.
   
2) X is pushed forward through the layers, Dense, ReLU, Dense, to get the raw scores.
   Dense holds the learnable numbers: W (weights) and b (bias), it computes `X @ W + b`.
   ReLU allows for non-linear correlations, without it, the two Dense layers would collapse into one large linear function.
  
3) the raw scores are used to calculate the loss, a number that represents how wrong the models predictions are on a batch.
   This is done using the softmax function, which turns the raw scores into probabilities that sum to 1.
   Loss is calculated according to how little probability it gave the correct answer. this is cross-entropy.

4) Next is back propagation
   The network is a chain of functions, so the loss depends on early layers through later ones.
   The chain rule can let you calculate each layers gradient using the one that came after it.
   This was tested with slow numerical gradient calculations in unit tests.
   
5) The optimiser uses the gradients for each parameter in each layer, and changes each value in place based on a learning rate.
   `param -= lr * grad`

### Testing
The model is tested automatically after training, on the rest of the dataset (14,000 images).

### Parameter storage
The trained parameters are saved to `mnist_model.npz` at the end of `train_mnist.py`.
`load_model` reads the file back into a model, so a trained set of parameters can be used without having to retrain the model.

## How I tested it
I used pytest, it allowed me to set up testing for whole system, every file starting with test_\*.py can have test_*() functions that pytest reads and runs automatically when you run the tests.

For example `test_model_gradients()`, it tests the gradients given by backpropegation are correct over the whole model. It checks the gradients with two separate methods and asserts that they are the same.

Use this command to run the tests:
```bash
pytest -v
```

## Running it
Requires Python 3.9+

1. Clone the repository
   ```bash
   git clone https://github.com/tomvangasse/neural-net.git
   cd neural-net
   ```

2. Create a virtual environment and install the dependencies
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate        # on Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Run the tests
   ```bash
   pytest -v
   ```

4. Train the network
   ```bash
   python train_mnist.py
   ```

The trained weights are saved to `mnist_model.npz`, which the web demo uses.

## Web demo
The web demo and deployment code were generated with AI coding tools, as my focus in this project was the neural network instead of the website.

```bash
python export_weights.py
python -m http.server 8000 --directory web
```

Then open http://localhost:8000. Press `Ctrl+C` in the terminal to stop the server.
