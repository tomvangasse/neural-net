// The trained network's forward pass, re-implemented in JavaScript so it runs
// in the visitor's browser. Same maths as nn/layers.py (Dense, ReLU) and the
// softmax in nn/losses.py -- but forward only, since the browser never trains.
//
// Weights are stored in Float32Arrays: flat, typed arrays that are much faster
// than nested JS arrays (contiguous memory, no per-element objects). A W matrix
// of shape (n_in, n_out) is stored row by row, so W[i][j] lives at i * n_out + j.

function toDense(W, b) {
  const nIn = W.length;
  const nOut = W[0].length;
  const flat = new Float32Array(nIn * nOut);
  for (let i = 0; i < nIn; i++) flat.set(W[i], i * nOut);
  return { W: flat, b: Float32Array.from(b), nIn, nOut };
}

// out[j] = sum_i(x[i] * W[i][j]) + b[j]   -- i.e. x @ W + b for a single example
function dense(x, layer) {
  const { W, b, nIn, nOut } = layer;
  const out = Float32Array.from(b);
  for (let i = 0; i < nIn; i++) {
    const xi = x[i];
    if (xi === 0) continue;          // most pixels are black (0), so this skips most of the work
    const row = i * nOut;
    for (let j = 0; j < nOut; j++) out[j] += xi * W[row + j];
  }
  return out;
}

function relu(x) {
  return x.map(v => (v > 0 ? v : 0));
}

function softmax(z) {
  const m = Math.max(...z);           // subtract the max first so exp() can't overflow
  const exps = Array.from(z, v => Math.exp(v - m));
  const total = exps.reduce((a, e) => a + e, 0);
  return exps.map(e => e / total);
}

export async function loadModel(url) {
  const w = await (await fetch(url)).json();
  // Keys match nn/io.py's save_model: "<layer index>_<param>". Layers 1 and 3 are ReLUs (no params).
  const layers = [toDense(w["0_W"], w["0_b"]), toDense(w["2_W"], w["2_b"]), toDense(w["4_W"], w["4_b"])];

  // pixels: 784 numbers in [0, 1]. Returns the probability of each digit 0-9.
  return function predict(pixels) {
    let x = dense(pixels, layers[0]);
    x = relu(x);
    x = dense(x, layers[1]);
    x = relu(x);
    const logits = dense(x, layers[2]);
    return softmax(logits);
  };
}
