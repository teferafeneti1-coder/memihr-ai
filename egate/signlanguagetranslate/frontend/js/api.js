/**
 * api.js – thin wrappers around the backend REST API.
 * All functions return promises; network errors propagate as thrown Errors.
 */

const BASE = '';   // same origin

export async function fetchHealth() {
  const r = await fetch(`${BASE}/api/health`);
  if (!r.ok) throw new Error(`Health check failed: ${r.status}`);
  return r.json();
}

export async function fetchMlpActive() {
  const r = await fetch(`${BASE}/api/mlp/active`);
  if (!r.ok) throw new Error(`Could not load MLP: ${r.status}`);
  return r.json();
}

export async function fetchBaseSamples() {
  const r = await fetch(`${BASE}/api/base_samples`);
  if (!r.ok) throw new Error(`Could not load base samples: ${r.status}`);
  return r.json();
}

export async function fetchCustomSamples() {
  const r = await fetch(`${BASE}/api/custom`);
  if (!r.ok) throw new Error(`Could not load custom samples: ${r.status}`);
  return r.json();
}

/**
 * POST samples for a label.
 * @param {string}   label
 * @param {number[][]} samples  – array of 163-float arrays
 */
export async function postCustomSamples(label, samples) {
  const r = await fetch(`${BASE}/api/custom/${encodeURIComponent(label)}`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ samples }),
  });
  if (!r.ok) {
    const msg = await r.text().catch(() => r.status);
    throw new Error(`Failed to save samples for "${label}": ${msg}`);
  }
  return r.json();
}

/** Clear samples of a label (keep the gesture entry). */
export async function clearCustomSamples(label) {
  const r = await fetch(`${BASE}/api/custom/${encodeURIComponent(label)}`, {
    method:  'PUT',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ samples: [] }),
  });
  if (!r.ok) throw new Error(`Failed to clear "${label}": ${r.status}`);
  return r.json();
}

/** Delete one gesture entirely. */
export async function deleteCustomGesture(label) {
  const r = await fetch(`${BASE}/api/custom/${encodeURIComponent(label)}`, {
    method: 'DELETE',
  });
  if (!r.ok) throw new Error(`Failed to delete "${label}": ${r.status}`);
  return r.json();
}

// ── Phase 2 ───────────────────────────────────────────────────────────────────

export async function startMlpTraining(epochs = 60) {
  const r = await fetch(`${BASE}/api/mlp/train`, {
    method:  'POST',
    headers: { 'Content-Type': 'application/json' },
    body:    JSON.stringify({ epochs }),
  });
  if (!r.ok) {
    const msg = await r.text().catch(() => r.status);
    throw new Error(`Training start failed: ${msg}`);
  }
  return r.json();
}

export async function fetchMlpStatus() {
  const r = await fetch(`${BASE}/api/mlp/status`);
  if (!r.ok) throw new Error(`Status check failed: ${r.status}`);
  return r.json();
}

export async function cancelMlpTraining() {
  const r = await fetch(`${BASE}/api/mlp/cancel`, { method: 'POST' });
  if (!r.ok) throw new Error(`Cancel failed: ${r.status}`);
  return r.json();
}

export async function resetMlp() {
  const r = await fetch(`${BASE}/api/mlp/reset`, { method: 'POST' });
  if (!r.ok) throw new Error(`Reset failed: ${r.status}`);
  return r.json();
}
