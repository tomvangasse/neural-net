import numpy as np


def save_model(model, path):
    arrays = {}
    for i, layer in enumerate(model.layers):    # looping through each layer in the model, saving the position: i, and the layer itself: layer
        for name, value in layer.params.items():
            arrays[f"{i}_{name}"] = value       # gives a different name for parameters in different layers
    np.savez(path, **arrays)    # unpacks the dictionary, and saves the data to a file


def load_model(model, path):
    data = np.load(path)    # gets the parameter data
    for i, layer in enumerate(model.layers):
        for name in layer.params:   
            layer.params[name] = data[f"{i}_{name}"]    # changes each parameter in the model to the new data
