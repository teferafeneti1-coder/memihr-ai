/**
 * ui.js – all DOM manipulation: gesture chips, status, debug panel, controls.
 *
 * Exported:
 *   initUI(callbacks)
 *   setStatus(msg, isError)
 *   setBusy(bool)
 *   setRecording(bool)
 *   setExampleCount(n)
 *   setDetectedWord(label, score)
 *   setHandPoseIndicators(hasHand, hasPose)
 *   updateDebug(mlpResult, knnResult, fusionResult, fps)
 *   renderGestures(gestureMap, selectedLabel)
 *   setStartStopEnabled(startOk, stopOk)
 *   showBanner(msg, durationMs)
 *   getRetrainMlpChecked()
 *   showTrainProgress(state, epoch, epochs, message)
 *   hideTrainProgress()
 */

let _callbacks = {};
let _gestures  = {};        // label → count
let _selected  = null;

export function initUI(callbacks) {
  _callbacks = callbacks;

  // Add Word button / Enter key
  const addBtn   = document.getElementById('btn-add-word');
  const nameInp  = document.getElementById('gesture-name-input');
  const plusBtn  = document.getElementById('btn-plus');

  plusBtn.addEventListener('click', () => nameInp.focus());

  function doAdd() {
    const raw = nameInp.value.trim();
    if (!raw) return;
    _callbacks.onAddGesture?.(raw);
    nameInp.value = '';
  }
  addBtn.addEventListener('click', doAdd);
  nameInp.addEventListener('keydown', e => { if (e.key === 'Enter') doAdd(); });

  // Train / Clear
  document.getElementById('btn-train').addEventListener('click', () => _callbacks.onTrain?.());
  document.getElementById('btn-clear').addEventListener('click', () => {
    if (!_selected) return;
    if (confirm(`Clear all examples for "${_selected}"? The gesture stays.`)) {
      _callbacks.onClear?.(_selected);
    }
  });

  // Start / Stop
  document.getElementById('btn-start').addEventListener('click', () => _callbacks.onStart?.());
  document.getElementById('btn-stop').addEventListener('click',  () => _callbacks.onStop?.());

  // Debug toggle
  document.getElementById('btn-debug').addEventListener('click', () => {
    const dbg = document.getElementById('debug-panel');
    dbg.hidden = !dbg.hidden;
  });

  // Phase 2: revert
  document.getElementById('btn-revert-mlp')?.addEventListener('click', () => {
    if (confirm('Revert to base MLP? This removes the retrained model.')) {
      _callbacks.onRevertMlp?.();
    }
  });

  // Phase 2: cancel training
  document.getElementById('btn-cancel-train')?.addEventListener('click', () => {
    _callbacks.onCancelTrain?.();
  });
}

export function setStatus(msg, isError = false) {
  const el = document.getElementById('status-line');
  el.textContent = msg;
  el.style.color = isError ? '#f87171' : '#94a3b8';
}

export function setBusy(busy) {
  document.getElementById('spinner').hidden = !busy;
}

export function setRecording(rec) {
  document.getElementById('btn-start').disabled = rec;
  document.getElementById('btn-stop').disabled  = !rec;
  document.getElementById('recording-indicator').hidden = !rec;
}

export function setExampleCount(n) {
  document.getElementById('example-count').textContent = `${n} examples`;
}

export function setDetectedWord(label, score) {
  const el = document.getElementById('detected-word');
  el.textContent = label || '—';
  el.style.color = label ? '#facc15' : '#475569';
}

export function setHandPoseIndicators(hasHand, hasPose) {
  document.getElementById('ind-hand').textContent = hasHand ? 'hand ✓' : 'hand ✗';
  document.getElementById('ind-hand').style.color = hasHand ? '#4ade80' : '#f87171';
  document.getElementById('ind-pose').textContent = hasPose ? 'pose ✓' : 'pose ✗';
  document.getElementById('ind-pose').style.color = hasPose ? '#4ade80' : '#f87171';
}

export function updateDebug(mlpResult, knnResult, fusionResult, fps) {
  const dbg = document.getElementById('debug-panel');
  if (dbg.hidden) return;

  // MLP top-3
  if (mlpResult) {
    const pairs = Array.from(mlpResult.probs)
      .map((p, i) => [mlpResult.classes[i], p])
      .sort((a, b) => b[1] - a[1])
      .slice(0, 3)
      .map(([c, p]) => `${c}:${(p*100).toFixed(1)}%`)
      .join('  ');
    document.getElementById('dbg-mlp').textContent = 'MLP: ' + pairs;
  } else {
    document.getElementById('dbg-mlp').textContent = 'MLP: —';
  }

  // kNN
  if (knnResult) {
    const acc = knnResult.accepted ? '✓' : '✗';
    document.getElementById('dbg-knn').textContent =
      `kNN: ${knnResult.label ?? '—'} conf=${(knnResult.confidence*100).toFixed(1)}% ` +
      `dist=${knnResult.nearestDist?.toFixed(2)} thr=${knnResult.threshold?.toFixed(2)} ${acc}`;
  } else {
    document.getElementById('dbg-knn').textContent = 'kNN: —';
  }

  // fusion scores
  if (fusionResult?.scores) {
    const top3 = [...fusionResult.scores.entries()]
      .sort((a,b)=>b[1]-a[1]).slice(0,3)
      .map(([l,s])=>`${l}:${(s*100).toFixed(1)}%`).join('  ');
    document.getElementById('dbg-fusion').textContent =
      `Fusion: ${fusionResult.rawLabel ?? '—'} → ${fusionResult.label ?? '—'}  [${top3}]`;
  }

  // fps
  document.getElementById('dbg-fps').textContent = `FPS: ${fps?.toFixed(1) ?? '—'}`;
}

/**
 * Re-render the gesture chip list.
 * @param {Object} gestureMap  – { label: count }
 * @param {string|null} selectedLabel
 */
export function renderGestures(gestureMap, selectedLabel) {
  _gestures  = gestureMap;
  _selected  = selectedLabel;
  const container = document.getElementById('gesture-chips');
  container.innerHTML = '';
  for (const [label, count] of Object.entries(gestureMap)) {
    const chip = document.createElement('div');
    chip.className = 'gesture-chip' + (label === selectedLabel ? ' selected' : '');

    const nameSpan = document.createElement('span');
    nameSpan.className = 'chip-name';
    nameSpan.textContent = label;   // textContent, safe

    const countSpan = document.createElement('span');
    countSpan.className = 'chip-count';
    countSpan.textContent = count;

    const delBtn = document.createElement('button');
    delBtn.className  = 'chip-del';
    delBtn.textContent = '×';
    delBtn.setAttribute('aria-label', `Delete ${label}`);
    delBtn.addEventListener('click', e => {
      e.stopPropagation();
      if (confirm(`Delete gesture "${label}" and all its examples?`)) {
        _callbacks.onDeleteGesture?.(label);
      }
    });

    chip.appendChild(nameSpan);
    chip.appendChild(countSpan);
    chip.appendChild(delBtn);
    chip.addEventListener('click', () => _callbacks.onSelectGesture?.(label));
    container.appendChild(chip);
  }

  // update count for selected
  if (selectedLabel != null && gestureMap[selectedLabel] != null) {
    setExampleCount(gestureMap[selectedLabel]);
  }
}

export function setStartStopEnabled(startOk, stopOk) {
  document.getElementById('btn-start').disabled = !startOk;
  document.getElementById('btn-stop').disabled  = !stopOk;
}

export function showBanner(msg, durationMs = 4000) {
  const el = document.getElementById('banner');
  el.textContent = msg;
  el.hidden = false;
  clearTimeout(el._timer);
  el._timer = setTimeout(() => { el.hidden = true; }, durationMs);
}

export function getRetrainMlpChecked() {
  return document.getElementById('chk-retrain-mlp')?.checked ?? true;
}

/** Phase 2: show training progress bar */
export function showTrainProgress(state, epoch, epochs, message) {
  const bar = document.getElementById('train-progress-wrap');
  if (!bar) return;
  bar.hidden = false;
  const pct = epochs > 0 ? Math.round((epoch / epochs) * 100) : 0;
  document.getElementById('train-progress-bar').style.width = pct + '%';
  document.getElementById('train-progress-label').textContent =
    `${state === 'done' ? '✓ ' : ''}${message ?? state} (${epoch}/${epochs})`;
  document.getElementById('btn-cancel-train').hidden = (state === 'done' || state === 'cancelled' || state === 'error');
}

export function hideTrainProgress() {
  const bar = document.getElementById('train-progress-wrap');
  if (bar) bar.hidden = true;
}
