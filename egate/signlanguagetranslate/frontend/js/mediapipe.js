/**
 * mediapipe.js – load and run HandLandmarker + PoseLandmarker from MediaPipe Tasks Vision.
 *
 * The package layout at jsDelivr for @mediapipe/tasks-vision@0.10.14:
 *   vision_bundle.mjs                  ← ES module to import
 *   wasm/vision_wasm_internal.js/.wasm ← WASM runtime (path given to FilesetResolver)
 *
 * Exported:
 *   initMediaPipe(onStatus)  – downloads models, returns promise
 *   runHands(video, ts)      – returns HandLandmarkerResult
 *   runPose(video, ts)       – returns PoseLandmarkerResult
 *   isReady()
 */

const MEDIAPIPE_VERSION = '0.10.14';
const CDN_BASE = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;

// vision_bundle.mjs is at the package root (NOT inside wasm/)
const BUNDLE_URL = `${CDN_BASE}/vision_bundle.mjs`;
// FilesetResolver needs the directory that contains vision_wasm_internal.js/.wasm
const WASM_PATH  = `${CDN_BASE}/wasm`;

const HAND_MODEL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';
const POSE_MODEL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task';

let _handLandmarker = null;
let _poseLandmarker = null;
let _ready = false;

export async function initMediaPipe(onStatus) {
  onStatus?.('Loading MediaPipe Tasks Vision…');

  // Import the ES module bundle from the package root
  const { HandLandmarker, PoseLandmarker, FilesetResolver } =
    await import(BUNDLE_URL);

  onStatus?.('Initialising WASM runtime…');
  // Pass the wasm/ subdirectory so the loader finds vision_wasm_internal.js
  const vision = await FilesetResolver.forVisionTasks(WASM_PATH);

  onStatus?.('Loading hand landmark model…');
  _handLandmarker = await HandLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: HAND_MODEL,
      delegate: 'GPU',
    },
    runningMode:                'VIDEO',
    numHands:                   2,
    minHandDetectionConfidence: 0.5,
    minHandPresenceConfidence:  0.5,
    minTrackingConfidence:      0.5,
  }).catch(async () => {
    onStatus?.('GPU unavailable, falling back to CPU…');
    return HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL, delegate: 'CPU' },
      runningMode: 'VIDEO',
      numHands:    2,
      minHandDetectionConfidence: 0.5,
      minHandPresenceConfidence:  0.5,
      minTrackingConfidence:      0.5,
    });
  });

  onStatus?.('Loading pose landmark model…');
  _poseLandmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath: POSE_MODEL,
      delegate: 'GPU',
    },
    runningMode:                 'VIDEO',
    numPoses:                    1,
    minPoseDetectionConfidence:  0.5,
    minPosePresenceConfidence:   0.5,
    minTrackingConfidence:       0.5,
  }).catch(async () => {
    return PoseLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: POSE_MODEL, delegate: 'CPU' },
      runningMode: 'VIDEO',
      numPoses:    1,
      minPoseDetectionConfidence: 0.5,
      minPosePresenceConfidence:  0.5,
      minTrackingConfidence:      0.5,
    });
  });

  _ready = true;
  onStatus?.('MediaPipe ready');
}

export function isReady() { return _ready; }

/** @param {HTMLVideoElement} video @param {number} timestampMs */
export function runHands(video, timestampMs) {
  if (!_handLandmarker) return null;
  return _handLandmarker.detectForVideo(video, timestampMs);
}

/** @param {HTMLVideoElement} video @param {number} timestampMs */
export function runPose(video, timestampMs) {
  if (!_poseLandmarker) return null;
  return _poseLandmarker.detectForVideo(video, timestampMs);
}
