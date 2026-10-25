/**
 * fusion.js – combine MLP and kNN signals into a final label.
 *
 * Constants are at the top for easy tuning.
 * Exported:
 *   Fusion class
 *     update(mlpResult, knnResult, hasHandOrPose)  → fusionResult
 *     getSmoothed()  → string | null  (the smoothed final label)
 *     lastScores()   → Map<label, score>
 */

// ── tuning constants ──────────────────────────────────────────────────────────
const W_KNN           = 0.5;    // weight for kNN confidence
const W_MLP           = 0.5;    // weight for MLP probability
const CUSTOM_MIN_CONF = 0.7;    // min kNN confidence for pure-custom fast path
const FUSION_MIN_SCORE = 0.45;  // min fused score to emit a label (else "—")
const SMOOTH_WINDOW   = 7;      // number of frames for temporal smoothing
const SMOOTH_MIN_WINS = 4;      // label must win this many frames to be shown

export class Fusion {
  constructor() {
    this._history = [];   // last SMOOTH_WINDOW frame labels (string|null)
    this._scores  = new Map();
  }

  /**
   * @param {{ probs: Float32Array, classes: string[], topLabel, topProb }} mlpResult  or null
   * @param {{ label, confidence, accepted, isCustom }} knnResult  or null
   * @param {boolean} hasHandOrPose  – whether pose or hand landmark is available
   * @returns {{ label: string|null, score: number, scores: Map, rawLabel: string|null }}
   */
  update(mlpResult, knnResult, hasHandOrPose) {
    this._scores = new Map();
    let rawLabel = null;

    // ── fast path: accepted custom word with high confidence ──────────────────
    if (
      knnResult &&
      knnResult.accepted &&
      knnResult.isCustom &&
      knnResult.confidence >= CUSTOM_MIN_CONF
    ) {
      rawLabel = knnResult.label;
      this._scores.set(rawLabel, knnResult.confidence);
    } else {
      // ── soft vote ─────────────────────────────────────────────────────────
      const labelSet = new Set();
      if (mlpResult && hasHandOrPose)  mlpResult.classes.forEach(c => labelSet.add(c));
      if (knnResult && knnResult.label) labelSet.add(knnResult.label);

      for (const lbl of labelSet) {
        let score = 0;
        // MLP contribution
        if (mlpResult && hasHandOrPose) {
          const idx = mlpResult.classes.indexOf(lbl);
          if (idx >= 0) score += W_MLP * mlpResult.probs[idx];
        }
        // kNN contribution
        if (knnResult && knnResult.accepted && knnResult.label === lbl) {
          score += W_KNN * knnResult.confidence;
        }
        this._scores.set(lbl, score);
      }

      // pick best
      let bestLabel = null, bestScore = 0;
      for (const [lbl, sc] of this._scores) {
        if (sc > bestScore) { bestScore = sc; bestLabel = lbl; }
      }
      rawLabel = bestScore >= FUSION_MIN_SCORE ? bestLabel : null;
    }

    // ── temporal smoothing ────────────────────────────────────────────────────
    this._history.push(rawLabel);
    if (this._history.length > SMOOTH_WINDOW) this._history.shift();

    const wins = {};
    for (const l of this._history) {
      if (l) wins[l] = (wins[l] || 0) + 1;
    }
    let smoothed = null;
    for (const [lbl, cnt] of Object.entries(wins)) {
      if (cnt >= SMOOTH_MIN_WINS) { smoothed = lbl; break; }
    }
    // pick the one with most wins if multiple reach threshold
    if (Object.keys(wins).length > 1) {
      const top = Object.entries(wins).sort((a,b)=>b[1]-a[1])[0];
      if (top[1] >= SMOOTH_MIN_WINS) smoothed = top[0];
      else smoothed = null;
    }

    return {
      label:    smoothed,
      rawLabel,
      score:    rawLabel ? (this._scores.get(rawLabel) ?? 0) : 0,
      scores:   this._scores,
    };
  }

  getSmoothed() {
    const wins = {};
    for (const l of this._history) if (l) wins[l] = (wins[l]||0)+1;
    const top = Object.entries(wins).sort((a,b)=>b[1]-a[1])[0];
    return top && top[1] >= SMOOTH_MIN_WINS ? top[0] : null;
  }

  lastScores() { return this._scores; }

  reset() { this._history = []; this._scores = new Map(); }
}
