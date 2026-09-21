class Model:
    def __init__(self, layers):
        self.layers = layers


    def forward(self, X):
        for layer in self.layers:   # pushes X through each layer in the model
            X = layer.forward(X)

        return X                    # and returns X

    
    def backward(self, dout):
        for layer in reversed(self.layers): # backpropeggation through all the layers in reverse order
            dout = layer.backward(dout)     # no return needed, as the gradients are stored in each layers 'grads' dictionary


    def params_and_grads(self):     # builds a list of pairs of (param_array, grad_array)
        pairs = []            
        for layer in self.layers:       
            for name in layer.params:
                pairs.append((layer.params[name], layer.grads[name]))
        return pairs