/**
 * mlp.js – MLP forward pass in plain JS.
 *
 * Exported:
 *   loadMLP(mlpJson)          – parse and store the model
 *   mlpPredict(features)      – Float32Array(163) → {probs, classes, topLabel, topProb}
 *   getMlpClasses()           – string[]
 *   getMlpMeanStd()           – {mean, std} as Float32Array
 */

let _layers  = null;   // [{W: Float32Array(out*in), b: Float32Array(out), inDim, outDim}]
let _mean    = null;
let _std     = null;
let _classes = null;

/**
 * Load (or reload) the MLP from the JSON object returned by /api/mlp/active.
 * Safe to call multiple times – replaces the current model atomically.
 */
export function loadMLP(mlpJson) {
  const layers = mlpJson.layers.map(l => {
    const outDim = l.W.length;
    const inDim  = l.W[0].length;
    // flatten row-major: W[row][col] → W_flat[row*inDim + col]
    const W = new Float32Array(outDim * inDim);
    for (let r = 0; r < outDim; r++) {
      for (let c = 0; c < inDim; c++) {
        W[r * inDim + c] = l.W[r][c];
      }
    }
    const b = new Float32Array(l.b);
    return { W, b, inDim, outDim };
  });

  _mean    = new Float32Array(mlpJson.mean);
  _std     = new Float32Array(mlpJson.std);
  _classes = mlpJson.classes.slice();
  _layers  = layers;
}

export function getMlpClasses() { return _classes ? _classes.slice() : []; }
export function getMlpMeanStd()  { return { mean: _mean, std: _std }; }
export function isMlpLoaded()    { return _layers !== null; }

/**
 * Run a forward pass.
 * @param {Float32Array} features – length 163
 * @returns {{ probs: Float32Array, classes: string[], topLabel: string, topProb: number }}
 */
export function mlpPredict(features) {
  if (!_layers) throw new Error("MLP not loaded");

  // standardise
  const n = features.length;
  let x = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    x[i] = (features[i] - _mean[i]) / _std[i];
  }

  // linear layers
  for (let li = 0; li < _layers.length; li++) {
    const { W, b, inDim, outDim } = _layers[li];
    const y = new Float32Array(outDim);
    for (let r = 0; r < outDim; r++) {
      let s = b[r];
      const off = r * inDim;
      for (let c = 0; c < inDim; c++) {
        s += W[off + c] * x[c];
      }
      y[r] = s;
    }
    // ReLU on all layers except the last
    if (li < _layers.length - 1) {
      for (let r = 0; r < outDim; r++) {
        if (y[r] < 0) y[r] = 0;
      }
    }
    x = y;
  }

  // softmax
  let maxV = x[0];
  for (let i = 1; i < x.length; i++) if (x[i] > maxV) maxV = x[i];
  let sum = 0;
  const probs = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    probs[i] = Math.exp(x[i] - maxV);
    sum += probs[i];
  }
  for (let i = 0; i < probs.length; i++) probs[i] /= sum;

  // top label
  let topIdx = 0;
  for (let i = 1; i < probs.length; i++) if (probs[i] > probs[topIdx]) topIdx = i;

  return {
    probs,
    classes:  _classes,
    topLabel: _classes[topIdx],
    topProb:  probs[topIdx],
  };
}
