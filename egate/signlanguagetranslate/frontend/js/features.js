/**
 * features.js – extract the 163-float feature vector from MediaPipe landmarks.
 * Mirrors the Python make_features() reference implementation exactly.
 *
 * Exported:
 *   initFeatures(mlpJson)          – store mean/std, validate feature_dim
 *   extractFeatures(handResults, poseResults, videoWidth, videoHeight)
 *                                  – returns Float32Array(163)
 */

export const FEATURE_DIM = 163;
const POSE_IDX = [0,11,12,13,14,15,16,17,18,19,20,21,22];

let _mean = null;
let _std  = null;

/** Call once after loading the MLP JSON. Throws if feature_dim !== 163. */
export function initFeatures(mlpJson) {
  if (mlpJson.feature_dim !== FEATURE_DIM) {
    throw new Error(`MLP feature_dim is ${mlpJson.feature_dim}, expected ${FEATURE_DIM}`);
  }
  _mean = new Float32Array(mlpJson.mean);
  _std  = new Float32Array(mlpJson.std);
}

export function getMean() { return _mean; }
export function getStd()  { return _std;  }

/**
 * Build the 163-float feature vector.
 *
 * @param {Object|null} handResults  – MediaPipe HandLandmarkerResult
 * @param {Object|null} poseResults  – MediaPipe PoseLandmarkerResult
 * @param {number} W  – video width  (pixels)
 * @param {number} H  – video height (pixels)
 * @returns {Float32Array} length 163
 */
export function extractFeatures(handResults, poseResults, W, H) {
  const vec = new Float32Array(FEATURE_DIM);

  // ── convert hand landmarks to pixel space (21 × 3 per hand) ───────────────
  const handsPx = [];   // each: Float32Array(63) in pixel units, shape [21][3]
  if (handResults && handResults.landmarks) {
    for (const lms of handResults.landmarks) {
      const h = new Float32Array(21 * 3);
      for (let j = 0; j < 21; j++) {
        h[j*3+0] = lms[j].x * W;
        h[j*3+1] = lms[j].y * H;
        h[j*3+2] = lms[j].z * W;
      }
      handsPx.push(h);
    }
  }

  // sort by wrist x ascending, keep max 2
  handsPx.sort((a, b) => a[0] - b[0]);
  const hs = handsPx.slice(0, 2);

  // ── hand blocks (indices 0-125) ────────────────────────────────────────────
  const scales = [];
  for (let i = 0; i < hs.length; i++) {
    const h = hs[i];
    const wx = h[0], wy = h[1];
    // palm size: distance wrist(0) → landmark9
    const dx = h[9*3+0] - wx;
    const dy = h[9*3+1] - wy;
    const sc = Math.sqrt(dx*dx + dy*dy) + 1e-6;
    scales.push(sc);
    const base = i * 63;
    for (let j = 0; j < 21; j++) {
      vec[base + j*3+0] = (h[j*3+0] - wx) / sc;
      vec[base + j*3+1] = (h[j*3+1] - wy) / sc;
      vec[base + j*3+2] = (h[j*3+2] - h[2]) / sc;   // z relative to wrist z
    }
  }

  // ── inter-hand offset (indices 126-128) ────────────────────────────────────
  if (hs.length === 2) {
    const maxSc = Math.max(scales[0], scales[1]);
    vec[126] = (hs[1][0] - hs[0][0]) / maxSc;
    vec[127] = (hs[1][1] - hs[0][1]) / maxSc;
    vec[128] = (hs[1][2] - hs[0][2]) / maxSc;
  }

  // ── pose block ─────────────────────────────────────────────────────────────
  if (poseResults && poseResults.landmarks && poseResults.landmarks.length > 0) {
    const plms = poseResults.landmarks[0];
    // convert to pixel (x,y) only
    const posePx = new Float32Array(33 * 2);
    for (let j = 0; j < 33; j++) {
      posePx[j*2+0] = plms[j].x * W;
      posePx[j*2+1] = plms[j].y * H;
    }

    const noseX = posePx[0], noseY = posePx[1];
    const lsX = posePx[11*2], lsY = posePx[11*2+1];
    const rsX = posePx[12*2], rsY = posePx[12*2+1];
    const dsx = lsX - rsX, dsy = lsY - rsY;
    const sw = Math.sqrt(dsx*dsx + dsy*dsy);

    if (sw > 10) {
      const midX = (lsX + rsX) / 2;
      const midY = (lsY + rsY) / 2;

      // body block (137-162): pose[POSE_IDX] minus mid, divided by sw
      let outIdx = 137;
      for (const pi of POSE_IDX) {
        vec[outIdx++] = (posePx[pi*2+0] - midX) / sw;
        vec[outIdx++] = (posePx[pi*2+1] - midY) / sw;
      }

      // hand-to-body offsets (129-136)
      for (let i = 0; i < hs.length; i++) {
        const wx = hs[i][0], wy = hs[i][1];
        const b  = 129 + i * 4;
        vec[b+0] = (wx - noseX) / sw;
        vec[b+1] = (wy - noseY) / sw;
        vec[b+2] = (wx - midX)  / sw;
        vec[b+3] = (wy - midY)  / sw;
      }
    }
  }

  return vec;
}

/**
 * Convenience: same as extractFeatures but accepts raw arrays (for selftest).
 * hands_px: array of [[x,y,z]*21], pose_px: [[x,y]*33] or null
 */
export function extractFeaturesRaw(handsPxRaw, posePxRaw) {
  const vec = new Float32Array(FEATURE_DIM);

  // Convert raw arrays to Float32Arrays
  let hs = handsPxRaw.map(h => {
    const a = new Float32Array(21 * 3);
    for (let j = 0; j < 21; j++) {
      a[j*3+0] = h[j][0];
      a[j*3+1] = h[j][1];
      a[j*3+2] = h[j][2];
    }
    return a;
  });
  hs.sort((a, b) => a[0] - b[0]);
  hs = hs.slice(0, 2);

  const scales = [];
  for (let i = 0; i < hs.length; i++) {
    const h = hs[i];
    const wx = h[0], wy = h[1];
    const dx = h[9*3+0] - wx, dy = h[9*3+1] - wy;
    const sc = Math.sqrt(dx*dx + dy*dy) + 1e-6;
    scales.push(sc);
    const base = i * 63;
    for (let j = 0; j < 21; j++) {
      vec[base + j*3+0] = (h[j*3+0] - wx) / sc;
      vec[base + j*3+1] = (h[j*3+1] - wy) / sc;
      vec[base + j*3+2] = (h[j*3+2] - h[2]) / sc;
    }
  }

  if (hs.length === 2) {
    const maxSc = Math.max(scales[0], scales[1]);
    vec[126] = (hs[1][0] - hs[0][0]) / maxSc;
    vec[127] = (hs[1][1] - hs[0][1]) / maxSc;
    vec[128] = (hs[1][2] - hs[0][2]) / maxSc;
  }

  if (posePxRaw) {
    const posePx = new Float32Array(33 * 2);
    for (let j = 0; j < 33; j++) {
      posePx[j*2+0] = posePxRaw[j][0];
      posePx[j*2+1] = posePxRaw[j][1];
    }
    const noseX = posePx[0], noseY = posePx[1];
    const lsX = posePx[11*2], lsY = posePx[11*2+1];
    const rsX = posePx[12*2], rsY = posePx[12*2+1];
    const dsx = lsX - rsX, dsy = lsY - rsY;
    const sw = Math.sqrt(dsx*dsx + dsy*dsy);
    if (sw > 10) {
      const midX = (lsX + rsX) / 2;
      const midY = (lsY + rsY) / 2;
      let outIdx = 137;
      for (const pi of POSE_IDX) {
        vec[outIdx++] = (posePx[pi*2+0] - midX) / sw;
        vec[outIdx++] = (posePx[pi*2+1] - midY) / sw;
      }
      for (let i = 0; i < hs.length; i++) {
        const wx = hs[i][0], wy = hs[i][1];
        const b  = 129 + i * 4;
        vec[b+0] = (wx - noseX) / sw;
        vec[b+1] = (wy - noseY) / sw;
        vec[b+2] = (wx - midX)  / sw;
        vec[b+3] = (wy - midY)  / sw;
      }
    }
  }

  return vec;
}
