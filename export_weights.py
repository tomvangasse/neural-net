"""One-time conversion: the trained .npz weights -> plain JSON for the website.

The network runs inside the visitor's browser (web/model.js), and JavaScript
can't read NumPy's .npz format. This script does the one thing that needs
NumPy (reading the array file) and writes the same numbers out as nested
lists in JSON, which the browser can load with a single fetch().

Weights are rounded to 5 decimal places. The largest weight is under 0.5, so
that keeps ~4-5 significant figures -- far more precision than the network's
predictions are sensitive to -- while roughly halving the file size.
"""
import json

import numpy as np

data = np.load("mnist_model.npz")

weights = {key: np.round(data[key], 5).tolist() for key in data.files}

with open("web/weights.json", "w") as f:
    json.dump(weights, f, separators=(",", ":"))   # no spaces after , and : -- smaller file

print("wrote web/weights.json")
