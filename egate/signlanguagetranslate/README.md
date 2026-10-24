# ASL Landmark Recognizer

Real-time ASL word recognition using MediaPipe hand + pose landmarks, a pre-trained MLP, and a live kNN classifier. No YOLO, no server-side image processing.

## Quick start

```bash
cd /home/feneti/egate/signlanguagetranslate
./run.sh
```

Open **http://localhost:8000** in Chrome or Edge. The browser will ask for camera permission — allow it.

> **Internet required on first load** – MediaPipe downloads its hand and pose model files (~25 MB total) from Google Storage on the first visit. After that, the browser caches them.

## Requirements

- Python venv at `/home/feneti/ai-engineering-camp/myenv` (fastapi, uvicorn, numpy, torch)
- The `asl_landmark_package/` folder (shipped with this repo, read-only)
- A webcam

## How to add a custom gesture

1. Type the gesture name in the **New Gesture Title** box and click **Add Word ▶** (or press Enter).
2. Select the chip that appears.
3. Click **Start** and move your hand slowly through a range of angles and distances. Aim for at least 60 examples (watch the counter).
4. Click **Stop**. The kNN rebuilds instantly.
5. Sign the word — it should appear in the Detected Word readout within a second.

After a page reload the examples are still there (they are stored on the backend).

## Training controls

| Button | What it does |
|--------|--------------|
| **Train** | Rebuilds the kNN immediately; optionally retrains the MLP in the background (see Phase 2 below). |
| **Clear** | Deletes the examples of the selected gesture (keeps the gesture entry). |
| **×** on chip | Deletes the gesture and all its examples permanently. |
| **Revert to base MLP** | Removes the retrained model and goes back to the original 18-word MLP. |

## Parity self-test

Run the Python script to generate the test vectors:

```bash
source /home/feneti/ai-engineering-camp/myenv/bin/activate
python scripts/make_parity_case.py
```

Then open **http://localhost:8000/selftest.html** — it runs the JS feature extractor and MLP against the Python reference values and prints PASS or FAIL with the maximum absolute difference.

## Tuning the fusion constants

Edit `frontend/js/fusion.js` — the constants are at the top:

```js
const W_KNN            = 0.5;    // weight for kNN signal
const W_MLP            = 0.5;    // weight for MLP signal
const CUSTOM_MIN_CONF  = 0.7;    // min kNN confidence for pure-custom fast path
const FUSION_MIN_SCORE = 0.45;   // min fused score to emit a label
const SMOOTH_WINDOW    = 7;      // temporal smoothing window (frames)
const SMOOTH_MIN_WINS  = 4;      // how many of the last N frames must agree
```

Use the **Debug** panel to see per-frame MLP, kNN, and fused scores and adjust accordingly.

## Phase 2 – live MLP retraining

When you click **Train** with *"Also retrain MLP (.pt)"* checked, the app:

1. Rebuilds the kNN instantly.
2. Launches a background Python subprocess (`backend/mlp_trainer.py`) that trains a new MLP on the base samples + your custom samples.
3. Shows a progress bar. Training takes a few seconds on CPU for 60 epochs.
4. When done, the new model is saved to `backend/models/mlp_versions/<timestamp>/` as both `mlp.json` and `mlp.pt`, and activated automatically — no page reload needed.

The validation accuracy (on the last 15% of your recordings) is shown in the status bar.

To go back to the original model click **Revert to base MLP** (with confirm).

## Known limitations

- The base model was trained on ~12 photos per word from a few signers. Recording the same words yourself (use the same names) is the most effective accuracy boost.
- The recognizer works on single frames, so words defined by motion (thank-you, good, help) are not reliably separable.
- MediaPipe needs your hand and upper body to be visible. Fist-type signs may not produce hand landmarks; the body-pose block still helps for those.
- On CPU the frame rate is capped at ~15 fps. If recognition feels sluggish, make sure no other apps are using the CPU heavily.
