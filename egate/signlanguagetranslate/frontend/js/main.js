/**
 * main.js – application entry point. Wires all modules together.
 */

import { initMediaPipe, runHands, runPose, isReady } from './mediapipe.js';
import { startCamera, stopCamera, isCameraOn, drawOverlay,
         videoWidth, videoHeight } from './camera.js';
import { initFeatures, extractFeatures, FEATURE_DIM } from './features.js';
import { loadMLP, mlpPredict, getMlpClasses, getMlpMeanStd, isMlpLoaded } from './mlp.js';
import { KNN } from './knn.js';
import { Fusion } from './fusion.js';
import {
  fetchHealth, fetchMlpActive, fetchBaseSamples, fetchCustomSamples,
  postCustomSamples, clearCustomSamples, deleteCustomGesture,
  startMlpTraining, fetchMlpStatus, cancelMlpTraining, resetMlp,
} from './api.js';
import {
  initUI, setStatus, setBusy, setRecording, setExampleCount,
  setDetectedWord, setHandPoseIndicators, updateDebug, renderGestures,
  setStartStopEnabled, showBanner, getRetrainMlpChecked,
  showTrainProgress, hideTrainProgress,
} from './ui.js';

// ── constants ─────────────────────────────────────────────────────────────────
const TARGET_FPS         = 15;
const POSE_EVERY_N       = 2;
const RECORD_REQUIRES_HAND = false;
const RECORD_MAX_PER_GESTURE = 400;
const RECORD_MAX_FPS     = 12;
const FLUSH_INTERVAL_MS  = 1000;

// ── state ─────────────────────────────────────────────────────────────────────
const knn    = new KNN();
const fusion = new Fusion();

let baseSamples   = {};
let customSamples = {};  // label → count (from backend)
let gestureMap    = {};  // label → count (local, includes unsaved buffer)
let selectedGesture = null;
let mlpJson       = null;

let recording     = false;
let recBuffer     = [];   // Float32Array[]
let recCount      = 0;    // samples saved to backend for selected gesture
let lastFlushTime = 0;

let frameCount    = 0;
let poseFrame     = 0;
let lastHandResult = null;
let lastPoseResult = null;
let lastFusionResult = { label: null, rawLabel: null, score: 0, scores: new Map() };

let fpsHistory = [];
let lastFrameTs = 0;

let trainPollInterval = null;

// DOM elements
const video   = document.getElementById('video');
const overlay = document.getElementById('overlay');
const ctx     = overlay.getContext('2d');

// ── boot ──────────────────────────────────────────────────────────────────────
async function boot() {
  initUI({
    onAddGesture:    addGesture,
    onSelectGesture: selectGesture,
    onDeleteGesture: deleteGesture,
    onTrain:         onTrain,
    onClear:         onClear,
    onStart:         onStart,
    onStop:          onStop,
    onRevertMlp:     onRevertMlp,
    onCancelTrain:   onCancelTrain,
  });

  setStartStopEnabled(false, false);
  setBusy(true);
  setStatus('Loading backend data…');

  // load MLP and base samples (don't fail hard if backend is down)
  try {
    [mlpJson] = await Promise.all([fetchMlpActive()]);
    loadMLP(mlpJson);
    initFeatures(mlpJson);
    setStatus('MLP loaded');
  } catch (e) {
    setStatus('Backend unreachable – base recognition unavailable', true);
    showBanner('Backend is down. Start the server with ./run.sh');
  }

  try {
    baseSamples = await fetchBaseSamples();
  } catch (e) {
    setStatus('Could not load base samples – kNN disabled', true);
  }

  try {
    const saved = await fetchCustomSamples();
    for (const [lbl, arr] of Object.entries(saved)) {
      gestureMap[lbl] = arr.length;
    }
    customSamples = Object.fromEntries(
      Object.entries(saved).map(([k, v]) => [k, v.length])
    );
  } catch (_) {}

  renderGestures(gestureMap, selectedGesture);
  _rebuildKnn();

  // start camera
  setStatus('Starting camera…');
  try {
    await startCamera(video, msg => { setStatus(msg, true); showBanner(msg); });
    syncOverlaySize();
  } catch (e) {
    setBusy(false);
    return;
  }

  // load mediapipe
  try {
    await initMediaPipe(msg => setStatus(msg));
  } catch (e) {
    setStatus('MediaPipe failed to load: ' + e.message, true);
    showBanner('MediaPipe load failed. Check internet connection.');
    setBusy(false);
    return;
  }

  setBusy(false);
  setStatus('Ready');
  setStartStopEnabled(true, false);

  // start frame loop
  requestAnimationFrame(frameLoop);
}

// ── frame loop ────────────────────────────────────────────────────────────────
function frameLoop(nowMs) {
  requestAnimationFrame(frameLoop);

  if (!isReady() || !isCameraOn() || video.readyState < 2) return;

  // throttle to TARGET_FPS
  if (nowMs - lastFrameTs < 1000 / TARGET_FPS) return;
  const dt = nowMs - lastFrameTs;
  lastFrameTs = nowMs;

  // fps
  fpsHistory.push(1000 / dt);
  if (fpsHistory.length > 30) fpsHistory.shift();
  const fps = fpsHistory.reduce((a, b) => a + b, 0) / fpsHistory.length;

  syncOverlaySize();
  const W = overlay.width, H = overlay.height;

  // run mediapipe on raw (unmirrored) video
  lastHandResult = runHands(video, nowMs);
  if (frameCount % POSE_EVERY_N === 0) {
    lastPoseResult = runPose(video, nowMs);
  }
  frameCount++;

  const hasHand = !!(lastHandResult?.landmarks?.length);
  const hasPose = !!(lastPoseResult?.landmarks?.length);
  setHandPoseIndicators(hasHand, hasPose);

  // feature extraction
  const features = extractFeatures(lastHandResult, lastPoseResult, W, H);

  // MLP
  let mlpResult = null;
  if (isMlpLoaded() && (hasHand || hasPose)) {
    try { mlpResult = mlpPredict(features); } catch (_) {}
  }

  // kNN
  const { mean, std } = getMlpMeanStd();
  let knnResult = null;
  if (mean && knn.size() > 0) {
    knnResult = knn.predict(features, mean, std);
  }

  // fusion
  lastFusionResult = fusion.update(mlpResult, knnResult, hasHand || hasPose);

  setDetectedWord(lastFusionResult.label, lastFusionResult.score);
  updateDebug(mlpResult, knnResult, lastFusionResult, fps);

  // draw overlay
  drawOverlay(ctx, lastHandResult, lastPoseResult, lastFusionResult, W, H);

  // recording
  if (recording && selectedGesture) {
    const canRecord = RECORD_REQUIRES_HAND ? hasHand : (hasHand || hasPose);
    if (canRecord) {
      const recNow = performance.now();
      if (recNow - lastFlushTime > 1000 / RECORD_MAX_FPS) {
        if (recCount + recBuffer.length < RECORD_MAX_PER_GESTURE) {
          recBuffer.push(Array.from(features));
          setExampleCount(recCount + recBuffer.length);
        }
        lastFlushTime = recNow;
      }
    }
    // periodic flush
    if (performance.now() - lastFlushTime > FLUSH_INTERVAL_MS && recBuffer.length > 0) {
      flushBuffer();
    }
  }
}

// ── recording helpers ─────────────────────────────────────────────────────────
async function flushBuffer() {
  if (!recBuffer.length || !selectedGesture) return;
  const toSend = recBuffer.splice(0);
  try {
    const res = await postCustomSamples(selectedGesture, toSend);
    recCount = res.count;
    gestureMap[selectedGesture] = recCount;
    renderGestures(gestureMap, selectedGesture);
  } catch (e) {
    setStatus('Save error: ' + e.message, true);
    // put samples back
    recBuffer.unshift(...toSend);
  }
}

// ── gesture management ────────────────────────────────────────────────────────
function addGesture(label) {
  const key = label.trim();
  if (!key) return;
  // case-insensitive uniqueness check
  const existing = Object.keys(gestureMap).find(k => k.toLowerCase() === key.toLowerCase());
  if (existing) {
    selectGesture(existing);
    return;
  }
  gestureMap[key] = 0;
  renderGestures(gestureMap, key);
  selectGesture(key);
}

function selectGesture(label) {
  selectedGesture = label;
  recCount = gestureMap[label] ?? 0;
  renderGestures(gestureMap, label);
  setExampleCount(recCount);
  setStartStopEnabled(isCameraOn() && isReady(), recording);
}

async function deleteGesture(label) {
  try {
    // may not exist on backend if never saved
    await deleteCustomGesture(label).catch(() => {});
    delete gestureMap[label];
    if (selectedGesture === label) {
      selectedGesture = null;
      setExampleCount(0);
    }
    renderGestures(gestureMap, selectedGesture);
    _rebuildKnn();
    setStatus(`Deleted "${label}"`);
  } catch (e) {
    setStatus('Delete failed: ' + e.message, true);
  }
}

async function onClear(label) {
  try {
    await clearCustomSamples(label);
    gestureMap[label] = 0;
    if (selectedGesture === label) { recCount = 0; setExampleCount(0); }
    renderGestures(gestureMap, selectedGesture);
    _rebuildKnn();
    setStatus(`Cleared examples for "${label}"`);
  } catch (e) {
    setStatus('Clear failed: ' + e.message, true);
  }
}

// ── start / stop recording ────────────────────────────────────────────────────
function onStart() {
  if (!selectedGesture) return;
  recording = true;
  recBuffer = [];
  lastFlushTime = performance.now();
  setRecording(true);
  setBusy(true);
  setStatus(`Recording "${selectedGesture}"…`);
}

async function onStop() {
  recording = false;
  setRecording(false);
  setBusy(false);
  if (recBuffer.length > 0) await flushBuffer();
  setStatus('Stopped. Building kNN…');
  await _rebuildKnnFromBackend();
  setStatus(`Done. kNN accuracy: ${knn.getAccuracy() != null ? (knn.getAccuracy()*100).toFixed(1)+'%' : 'n/a'}`);
}

// ── train ─────────────────────────────────────────────────────────────────────
async function onTrain() {
  // Always rebuild kNN immediately
  await _rebuildKnnFromBackend();
  setStatus(`kNN rebuilt. Accuracy: ${knn.getAccuracy() != null ? (knn.getAccuracy()*100).toFixed(1)+'%' : 'n/a'}`);

  // optionally retrain MLP
  if (!getRetrainMlpChecked()) return;

  setBusy(true);
  try {
    await startMlpTraining(60);
    setStatus('MLP training started…');
    showTrainProgress('starting', 0, 60, 'Launching trainer…');
    _startTrainPoll();
  } catch (e) {
    setBusy(false);
    setStatus('Train failed: ' + e.message, true);
    showBanner(e.message);
  }
}

function _startTrainPoll() {
  if (trainPollInterval) clearInterval(trainPollInterval);
  trainPollInterval = setInterval(async () => {
    try {
      const s = await fetchMlpStatus();
      if (s.state === 'idle') return;
      showTrainProgress(s.state, s.epoch ?? 0, s.epochs ?? 60, s.message ?? s.state);
      if (s.state === 'done') {
        clearInterval(trainPollInterval);
        trainPollInterval = null;
        setBusy(false);
        hideTrainProgress();
        // reload MLP
        try {
          const newMlp = await fetchMlpActive();
          loadMLP(newMlp);
          initFeatures(newMlp);
          mlpJson = newMlp;
          await _rebuildKnnFromBackend();
          const acc = s.val_acc != null ? ` (val acc ${(s.val_acc*100).toFixed(1)}%)` : '';
          setStatus(`MLP updated${acc}. kNN rebuilt.`);
          showBanner(`MLP updated${acc}`);
        } catch (e2) {
          setStatus('MLP reload failed: ' + e2.message, true);
        }
      } else if (s.state === 'error' || s.state === 'cancelled') {
        clearInterval(trainPollInterval);
        trainPollInterval = null;
        setBusy(false);
        setStatus(s.message ?? s.state, s.state === 'error');
        hideTrainProgress();
      }
    } catch (_) {}
  }, 500);
}

async function onCancelTrain() {
  try {
    await cancelMlpTraining();
    clearInterval(trainPollInterval);
    trainPollInterval = null;
    setBusy(false);
    hideTrainProgress();
    setStatus('Training cancelled');
  } catch (e) {
    setStatus('Cancel failed: ' + e.message, true);
  }
}

async function onRevertMlp() {
  try {
    await resetMlp();
    const base = await fetchMlpActive();
    loadMLP(base);
    initFeatures(base);
    mlpJson = base;
    _rebuildKnn();
    setStatus('Reverted to base MLP');
    showBanner('Reverted to base MLP');
  } catch (e) {
    setStatus('Revert failed: ' + e.message, true);
  }
}

// ── kNN rebuild ───────────────────────────────────────────────────────────────
async function _rebuildKnnFromBackend() {
  try {
    const saved = await fetchCustomSamples();
    // update gestureMap counts
    for (const [lbl, arr] of Object.entries(saved)) {
      gestureMap[lbl] = arr.length;
    }
    // add any local gesture entries not on backend
    for (const lbl of Object.keys(gestureMap)) {
      if (!saved[lbl]) saved[lbl] = [];
    }
    renderGestures(gestureMap, selectedGesture);
    const { mean, std } = getMlpMeanStd();
    if (mean) knn.train(baseSamples, saved, mean, std);
  } catch (e) {
    setStatus('kNN rebuild failed: ' + e.message, true);
  }
}

function _rebuildKnn() {
  const { mean, std } = getMlpMeanStd();
  if (!mean) return;
  // use empty custom (no backend call)
  const customRaw = {};
  for (const [lbl, cnt] of Object.entries(gestureMap)) {
    customRaw[lbl] = customRaw[lbl] ?? [];
  }
  knn.train(baseSamples, {}, mean, std);
}

// ── overlay size sync ─────────────────────────────────────────────────────────
function syncOverlaySize() {
  const rect = video.getBoundingClientRect();
  if (overlay.width !== video.videoWidth  && video.videoWidth)  overlay.width  = video.videoWidth;
  if (overlay.height !== video.videoHeight && video.videoHeight) overlay.height = video.videoHeight;
}

// ── start ─────────────────────────────────────────────────────────────────────
boot().catch(e => {
  setStatus('Fatal: ' + e.message, true);
  console.error(e);
});
