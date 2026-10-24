/**
 * camera.js – webcam access, frame loop, and canvas overlay drawing.
 *
 * Exported:
 *   startCamera(videoEl, onError)  → promise, resolves when stream is live
 *   stopCamera()
 *   isCameraOn()
 *   drawOverlay(ctx, handResults, poseResults, fusionResult, W, H)
 */

export let videoWidth  = 0;
export let videoHeight = 0;

let _stream = null;

/** Start the camera stream into <video> element. */
export async function startCamera(videoEl, onError) {
  try {
    _stream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' },
      audio: false,
    });
    videoEl.srcObject = _stream;
    await new Promise((res, rej) => {
      videoEl.onloadedmetadata = () => {
        videoWidth  = videoEl.videoWidth;
        videoHeight = videoEl.videoHeight;
        res();
      };
      videoEl.onerror = rej;
    });
    videoEl.play();
  } catch (e) {
    let msg = 'Camera error: ' + e.message;
    if (e.name === 'NotAllowedError')  msg = 'Camera access denied. Please allow camera in browser settings.';
    if (e.name === 'NotFoundError')    msg = 'No camera found. Please connect a camera.';
    if (e.name === 'NotReadableError') msg = 'Camera is in use by another app.';
    onError?.(msg);
    throw e;
  }
}

export function stopCamera() {
  if (_stream) {
    _stream.getTracks().forEach(t => t.stop());
    _stream = null;
  }
}

export function isCameraOn() { return _stream !== null; }

// ── overlay drawing ───────────────────────────────────────────────────────────
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [0,9],[9,10],[10,11],[11,12],
  [0,13],[13,14],[14,15],[15,16],
  [0,17],[17,18],[18,19],[19,20],
  [5,9],[9,13],[13,17],
];

/**
 * Draw hand/pose dots, bounding box, and label onto the overlay canvas.
 * The canvas is CSS-mirrored; we un-mirror text by saving/restoring.
 *
 * @param {CanvasRenderingContext2D} ctx
 * @param {Object|null} handResults
 * @param {Object|null} poseResults
 * @param {{ label, score, rawLabel }} fusionResult
 * @param {number} W  canvas width
 * @param {number} H  canvas height
 */
export function drawOverlay(ctx, handResults, poseResults, fusionResult, W, H) {
  ctx.clearRect(0, 0, W, H);

  // ── pose skeleton (upper body, dim) ──────────────────────────────────────
  if (poseResults?.landmarks?.[0]) {
    const lms = poseResults.landmarks[0];
    ctx.strokeStyle = 'rgba(100,200,255,0.5)';
    ctx.lineWidth   = 1.5;
    const POSE_UPPER = [[11,12],[11,13],[13,15],[12,14],[14,16],[11,23],[12,24]];
    for (const [a,b] of POSE_UPPER) {
      if (!lms[a] || !lms[b]) continue;
      ctx.beginPath();
      ctx.moveTo(lms[a].x * W, lms[a].y * H);
      ctx.lineTo(lms[b].x * W, lms[b].y * H);
      ctx.stroke();
    }
    ctx.fillStyle = 'rgba(100,200,255,0.7)';
    for (let i = 0; i < Math.min(25, lms.length); i++) {
      ctx.beginPath();
      ctx.arc(lms[i].x * W, lms[i].y * H, 3, 0, Math.PI * 2);
      ctx.fill();
    }
  }

  // ── hand landmarks ────────────────────────────────────────────────────────
  if (handResults?.landmarks) {
    for (const lms of handResults.landmarks) {
      // connections
      ctx.strokeStyle = 'rgba(0,230,120,0.8)';
      ctx.lineWidth   = 2;
      for (const [a, b] of HAND_CONNECTIONS) {
        ctx.beginPath();
        ctx.moveTo(lms[a].x * W, lms[a].y * H);
        ctx.lineTo(lms[b].x * W, lms[b].y * H);
        ctx.stroke();
      }
      // dots
      ctx.fillStyle = '#00e678';
      for (const lm of lms) {
        ctx.beginPath();
        ctx.arc(lm.x * W, lm.y * H, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // ── bounding box from hand union ──────────────────────────────────────────
  const handBoxes = _handBoundingBoxes(handResults, W, H);
  if (handBoxes.length > 0) {
    const union = _unionBox(handBoxes);
    const pad   = 0.15;
    const bx = Math.max(0, union.x1 - union.w * pad);
    const by = Math.max(0, union.y1 - union.h * pad);
    const bw = Math.min(W - bx, union.w * (1 + 2 * pad));
    const bh = Math.min(H - by, union.h * (1 + 2 * pad));
    ctx.strokeStyle = fusionResult?.label ? '#facc15' : 'rgba(255,255,255,0.4)';
    ctx.lineWidth   = 2;
    ctx.strokeRect(bx, by, bw, bh);
  }

  // ── label text (un-mirror) ────────────────────────────────────────────────
  if (fusionResult?.label) {
    const label = fusionResult.label;
    const conf  = fusionResult.score != null ? ` ${(fusionResult.score*100).toFixed(0)}%` : '';
    const txt   = label + conf;

    // un-mirror: reflect around canvas centre
    ctx.save();
    ctx.transform(-1, 0, 0, 1, W, 0);

    const box = handBoxes.length > 0 ? _unionBox(handBoxes) : null;
    const tx  = box ? Math.max(8, W - (box.x1 + box.w / 2) - 60) : W / 2 - 60;
    const ty  = box ? Math.max(32, box.y1 - 14) : 40;

    ctx.font         = 'bold 22px sans-serif';
    ctx.fillStyle    = '#facc15';
    ctx.strokeStyle  = '#000';
    ctx.lineWidth    = 4;
    ctx.strokeText(txt, tx, ty);
    ctx.fillText(txt, tx, ty);
    ctx.restore();
  }
}

function _handBoundingBoxes(handResults, W, H) {
  if (!handResults?.landmarks) return [];
  return handResults.landmarks.map(lms => {
    let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
    for (const lm of lms) {
      x1 = Math.min(x1, lm.x * W);
      y1 = Math.min(y1, lm.y * H);
      x2 = Math.max(x2, lm.x * W);
      y2 = Math.max(y2, lm.y * H);
    }
    return { x1, y1, x2, y2, w: x2 - x1, h: y2 - y1 };
  });
}

function _unionBox(boxes) {
  let x1 = Infinity, y1 = Infinity, x2 = -Infinity, y2 = -Infinity;
  for (const b of boxes) {
    x1 = Math.min(x1, b.x1); y1 = Math.min(y1, b.y1);
    x2 = Math.max(x2, b.x2); y2 = Math.max(y2, b.y2);
  }
  return { x1, y1, x2, y2, w: x2 - x1, h: y2 - y1 };
}
