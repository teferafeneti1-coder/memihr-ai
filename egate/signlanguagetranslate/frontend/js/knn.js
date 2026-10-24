/**
 * knn.js – distance-weighted kNN classifier with LOO accuracy and auto threshold.
 *
 * Exported:
 *   KNN class
 *     train(baseSamples, customSamples, mean, std)
 *     predict(features)  → {label, confidence, nearestDist, threshold, accepted, isCustom}
 *     getAccuracy()      → number (0-1) | null
 *     getThreshold()     → number | null
 */

const BASE_SAMPLE_WEIGHT = 0.7;   // base samples are down-weighted vs custom
const LOO_EXCLUDE_WINDOW = 15;    // exclude neighbours within ±N positions (same gesture)
const THRESHOLD_PERCENTILE = 0.95;
const THRESHOLD_MULTIPLIER = 1.5;
const THRESHOLD_FALLBACK   = 3.0;
const K = 5;

export class KNN {
  constructor() {
    this._items     = [];   // {label, feat: Float32Array, isBase, posInClass}
    this._accuracy  = null;
    this._threshold = null;
    this._classes   = new Set();
    this._baseLabelSet = new Set();
  }

  /**
   * Rebuild the index.
   * @param {Object}  baseSamples   – {label: [[163], ...]}
   * @param {Object}  customSamples – {label: [[163], ...]}
   * @param {Float32Array} mean
   * @param {Float32Array} std
   */
  train(baseSamples, customSamples, mean, std) {
    this._items = [];
    this._baseLabelSet = new Set(Object.keys(baseSamples));

    // add base samples
    for (const [label, rows] of Object.entries(baseSamples)) {
      rows.forEach((row, pos) => {
        this._items.push({
          label,
          feat: _standardise(new Float32Array(row), mean, std),
          isBase: true,
          posInClass: pos,
        });
      });
    }

    // add custom samples
    for (const [label, rows] of Object.entries(customSamples)) {
      rows.forEach((row, pos) => {
        this._items.push({
          label,
          feat: _standardise(new Float32Array(row), mean, std),
          isBase: false,
          posInClass: pos,
        });
      });
    }

    this._classes = new Set(this._items.map(it => it.label));

    if (this._items.length === 0) {
      this._accuracy  = null;
      this._threshold = null;
      return;
    }

    // group items by label for LOO
    const byLabel = {};
    for (let i = 0; i < this._items.length; i++) {
      const lbl = this._items[i].label;
      if (!byLabel[lbl]) byLabel[lbl] = [];
      byLabel[lbl].push(i);
    }

    // ── LOO accuracy ──────────────────────────────────────────────────────────
    let correct = 0, total = 0;
    // ── per-sample nearest same-word distances (for threshold) ────────────────
    const sameWordDists = [];

    for (let qi = 0; qi < this._items.length; qi++) {
      const q = this._items[qi];
      // build exclusion set: same label within ±LOO_EXCLUDE_WINDOW
      const excl = new Set();
      for (const idx of byLabel[q.label]) {
        if (Math.abs(this._items[idx].posInClass - q.posInClass) <= LOO_EXCLUDE_WINDOW) {
          excl.add(idx);
        }
      }
      excl.add(qi);  // always exclude self

      // compute distances to all non-excluded items
      const dists = [];
      for (let ci = 0; ci < this._items.length; ci++) {
        if (excl.has(ci)) continue;
        const d = _euclidean(q.feat, this._items[ci].feat);
        dists.push({ d, label: this._items[ci].label, isBase: this._items[ci].isBase });
      }
      if (dists.length === 0) continue;

      dists.sort((a, b) => a.d - b.d);

      // nearest same-word distance (for threshold)
      const sameNearest = dists.find(x => x.label === q.label);
      if (sameNearest) sameWordDists.push(sameNearest.d);

      // LOO vote
      const k = Math.min(K, dists.length);
      const votes = {};
      for (let ki = 0; ki < k; ki++) {
        const { d, label, isBase } = dists[ki];
        const w = (isBase ? BASE_SAMPLE_WEIGHT : 1.0) / (d + 1e-9);
        votes[label] = (votes[label] || 0) + w;
      }
      const pred = Object.entries(votes).sort((a, b) => b[1] - a[1])[0][0];
      if (pred === q.label) correct++;
      total++;
    }

    this._accuracy = total > 0 ? correct / total : null;

    // ── rejection threshold ───────────────────────────────────────────────────
    if (sameWordDists.length >= 5) {
      sameWordDists.sort((a, b) => a - b);
      const p95idx = Math.floor(sameWordDists.length * THRESHOLD_PERCENTILE);
      this._threshold = sameWordDists[Math.min(p95idx, sameWordDists.length - 1)] * THRESHOLD_MULTIPLIER;
    } else {
      this._threshold = THRESHOLD_FALLBACK;
    }
  }

  /**
   * Classify a standardised feature vector (will standardise internally).
   * @param {Float32Array} rawFeatures – unstandardised 163-float vector
   * @param {Float32Array} mean
   * @param {Float32Array} std
   */
  predict(rawFeatures, mean, std) {
    if (this._items.length === 0) {
      return { label: null, confidence: 0, nearestDist: Infinity,
               threshold: this._threshold, accepted: false, isCustom: false };
    }

    const feat = _standardise(new Float32Array(rawFeatures), mean, std);
    const dists = this._items.map(it => ({
      d: _euclidean(feat, it.feat),
      label: it.label,
      isBase: it.isBase,
    }));
    dists.sort((a, b) => a.d - b.d);

    const k = Math.min(K, dists.length);
    const votes = {};
    for (let ki = 0; ki < k; ki++) {
      const { d, label, isBase } = dists[ki];
      const w = (isBase ? BASE_SAMPLE_WEIGHT : 1.0) / (d + 1e-9);
      votes[label] = (votes[label] || 0) + w;
    }

    const sorted = Object.entries(votes).sort((a, b) => b[1] - a[1]);
    const [topLabel, topVote] = sorted[0];
    const totalVote = sorted.reduce((s, [, v]) => s + v, 0);
    const confidence = topVote / totalVote;
    const nearestDist = dists[0].d;

    // "Neutral" is a reject class
    const isNeutral = topLabel.toLowerCase() === 'neutral';
    const accepted  = !isNeutral && nearestDist <= (this._threshold ?? THRESHOLD_FALLBACK);
    const isCustom  = !this._baseLabelSet.has(topLabel);

    return {
      label:       isNeutral ? null : topLabel,
      confidence,
      nearestDist,
      threshold:   this._threshold ?? THRESHOLD_FALLBACK,
      accepted:    isNeutral ? false : accepted,
      isCustom,
    };
  }

  getAccuracy()  { return this._accuracy; }
  getThreshold() { return this._threshold; }
  size()         { return this._items.length; }
}

// ── helpers ───────────────────────────────────────────────────────────────────
function _standardise(feat, mean, std) {
  const out = new Float32Array(feat.length);
  for (let i = 0; i < feat.length; i++) {
    out[i] = (feat[i] - mean[i]) / std[i];
  }
  return out;
}

function _euclidean(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    s += d * d;
  }
  return Math.sqrt(s);
}
