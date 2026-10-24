# Kiro prompt: real-time ASL recognition (YOLOv8 + landmark MLP + live custom words)

Put this file in the workspace root (`/home/feneti/egate/signlanguagetranslate/`), open that folder as the Kiro workspace, and send:

> Read `kiro_prompt_asl_realtime_yolo_landmarks.md` in this workspace and implement it completely. Do Phase 1 first and verify it, then Phase 2. If an app from an earlier attempt already exists here, read it first and upgrade it in place. Do not stop until every file is fully implemented.

---

## Environment (read first)

- Workspace root: `/home/feneti/egate/signlanguagetranslate/`.
- **Use this existing Python virtual environment: `/home/feneti/ai-engineering-camp/myenv`.** Do not create another one. Activate it with `source /home/feneti/ai-engineering-camp/myenv/bin/activate`. First run `python -c "import ultralytics, fastapi, uvicorn, cv2, numpy, torch"` inside it and `pip install` only what is missing (`fastapi`, `uvicorn[standard]`, `opencv-python-headless`, `numpy`, `ultralytics`). Do not upgrade packages that already work.
- The machine is **Linux, CPU only**. Everything must run in real time on CPU.
- Trained assets already in the workspace, folder `asl_landmark_package/` (read-only, never modify or delete):
  - `asl_landmark_mlp.json`: a 163-input MLP, keys `classes`, `mean`, `std`, `activation` ("relu"), `feature_dim` (163), `feature_spec`, `layers` (list of `{W: [out][in], b: [out]}`; three Linear layers: 163 to 128 to 64 to 18).
  - `asl_landmark_mlp.pt`: the same network as a PyTorch checkpoint (`state_dict`, `classes`, `mean`, `std`, `feature_dim`, `feature_spec`, `arch`).
  - `asl_landmark_samples.json`: `{ "<word>": [[163 floats], ...] }`, the original feature vectors of the 18 base words.
  - `classes.json`: the 18 base word names.
- **YOLOv8n weights `best.pt`** (trained on the same 18 words) must be at `backend/models/best.pt`. If it is not there, search this workspace for `best.pt` and copy it. If it cannot be found, the app must still start with YOLO disabled: `/api/health` reports `yolo_loaded: false` with a clear message, and the UI shows it. Do not fail.

## Goal

A local web app (FastAPI backend + vanilla HTML/CSS/JS frontend, no build step, `http://localhost:8000`) that recognizes ASL words from the webcam in real time by combining three signals, and lets the user teach new words live. Write complete, working code with no placeholders, no TODOs, no stubs.

## Architecture

1. **Browser, MediaPipe** (`@mediapipe/tasks-vision` from the jsDelivr CDN, pin a version that exists on npm such as 0.10.14, verify): `HandLandmarker` (`numHands: 2`) and `PoseLandmarker` (lite). Models:
   - `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task`
   - `https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task`
   - Use `runningMode: "VIDEO"`, GPU delegate with CPU fallback. The first load needs internet (say so in the README). Run hands every processed frame and pose every 2nd processed frame (reuse the last pose in between; constant `POSE_EVERY_N`). Process at most 15 frames per second (constant `TARGET_FPS`).
2. **Base MLP (in the browser)**: load `asl_landmark_mlp.json` through the backend and run the forward pass in plain JS: `x = (features - mean) / std`, then Linear+ReLU, Linear+ReLU, Linear, then softmax.
3. **kNN (in the browser)** over the base samples from `asl_landmark_samples.json` plus the user's custom samples. It gives instant learning of new words.
4. **YOLOv8 (backend)**: the browser sends downscaled JPEG frames to `POST /api/detect` and gets boxes, labels and confidences for the 18 base words.
5. **Fusion** combines the three (rules below).

**Orientation rules** (very important): feed MediaPipe and YOLO the **raw, unmirrored** video frame, exactly as the camera delivers it (the training photos are not mirrored). Only the on-screen preview and overlay are mirrored with CSS `transform: scaleX(-1)`. Draw overlay text so it is not mirrored.

## Feature spec (must match the training notebook exactly)

163 floats per frame. Landmarks are converted to **pixel space**: hands `(x * videoWidth, y * videoHeight, z * videoWidth)`, pose `(x * videoWidth, y * videoHeight)`.

- Up to 2 hands, sorted by wrist x ascending. For each hand slot `i`: 21 landmarks x (x, y, z) minus the wrist, divided by the palm size (2D distance from landmark 0 to landmark 9, plus 1e-6). Flatten in row-major order (`x0, y0, z0, x1, ...`). That fills indices 0 to 125. A missing hand leaves zeros.
- Indices 126 to 128: `(wrist2 - wrist1) / max(palm sizes)` (x, y, z), only when 2 hands are present.
- Pose landmarks used: nose (0), shoulders (11 and 12), and `POSE_IDX = [0,11,12,13,14,15,16,17,18,19,20,21,22]`. `sw` = shoulder width (2D distance between landmarks 11 and 12). Only if pose is available **and `sw > 10`**: `mid` = shoulder midpoint.
  - Indices 137 to 162: `((pose[POSE_IDX] - mid) / sw)` flattened row-major (x, y per landmark).
  - Indices 129 to 136: for each hand slot `i`, 4 values `[(wx - nose.x)/sw, (wy - nose.y)/sw, (wx - mid.x)/sw, (wy - mid.y)/sw]` where `(wx, wy)` is that hand's wrist.
- Without a detected hand, the hand blocks stay zero but the body block (137 to 162) is still filled. Without pose, only the hand blocks are filled.

**Reference implementation (Python, the source of truth):**

```python
import numpy as np

FEATURE_DIM = 163
POSE_IDX = [0, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]

def make_features(hands_px, pose_px):
    """hands_px: list of (21,3) float arrays in pixel units; pose_px: (33,2) pixel array or None."""
    hs = sorted(hands_px, key=lambda h: h[0, 0])[:2]
    vec = np.zeros(FEATURE_DIM, dtype=np.float32)
    scales = []
    for i, h in enumerate(hs):
        sc = float(np.linalg.norm(h[9, :2] - h[0, :2])) + 1e-6
        vec[i * 63:(i + 1) * 63] = ((h - h[0]) / sc).reshape(-1)
        scales.append(sc)
    if len(hs) == 2:
        vec[126:129] = (hs[1][0] - hs[0][0]) / max(scales)
    if pose_px is not None:
        nose, ls, rs = pose_px[0], pose_px[11], pose_px[12]
        sw = float(np.linalg.norm(ls - rs))
        if sw > 10:
            mid = (ls + rs) / 2
            vec[137:163] = ((pose_px[POSE_IDX] - mid) / sw).reshape(-1)
            for i, h in enumerate(hs):
                w = h[0, :2]
                vec[129 + i * 4:129 + i * 4 + 4] = [(w[0] - nose[0]) / sw, (w[1] - nose[1]) / sw,
                                                     (w[0] - mid[0]) / sw, (w[1] - mid[1]) / sw]
    return vec
```

Implement the same in `frontend/js/features.js` (Float32Array). Read `feature_dim` from the MLP JSON and refuse to run if it is not 163.

**Parity test (required):** create `scripts/make_parity_case.py` that builds a few random but realistic cases (1 hand, 2 hands, hands with no pose, pose with no hands) with the reference function above, computes the expected feature vectors and the expected MLP probabilities (a NumPy forward pass over `asl_landmark_mlp.json`), and writes `frontend/tests/parity_case.json`. Create `frontend/selftest.html` that loads that JSON, runs the JS `extractFeatures` and the JS MLP forward pass, and prints PASS or FAIL with the maximum absolute difference (PASS if below 1e-4 for features and 1e-4 for probabilities). This guards against silent differences between Python and JS.

## Project layout

```
signlanguagetranslate/
  asl_landmark_package/          # existing, read-only
  backend/
    app.py
    mlp_trainer.py               # Phase 2 worker
    requirements.txt
    models/                      # best.pt, mlp_versions/, active_mlp.json (created at runtime)
    data/                        # custom_gestures.json (created at runtime)
  frontend/
    index.html  style.css  selftest.html
    js/main.js camera.js mediapipe.js features.js mlp.js knn.js fusion.js api.js ui.js
    tests/parity_case.json
  scripts/make_parity_case.py
  run.sh                         # activates the venv and starts uvicorn on port 8000
  README.md
```

## Backend (`backend/app.py`, FastAPI)

- Paths are resolved relative to the project root (parent of `backend/`).
- **YOLO**: load `backend/models/best.pt` with `ultralytics.YOLO` (`cpu`). Warm up with one dummy inference. Guard model calls with a `threading.Lock`.
- `GET /api/health` returns `{yolo_loaded, yolo_message, device, yolo_classes, mlp_loaded, mlp_classes, feature_dim, mlp_source}`.
- `POST /api/detect`: raw JPEG bytes in the body (`image/jpeg`), optional query `conf` (default 0.25). Decode with `cv2.imdecode`, convert BGR to GRAY to BGR (the model was trained on grayscale), then `model.predict(img, imgsz=640, conf=conf, max_det=2, verbose=False)`. Return `{detections: [{label, confidence, box: [x1,y1,x2,y2]}], inference_ms}`, boxes normalized 0..1, sorted by confidence. Labels come from `model.names`. HTTP 503 with a message if YOLO is not loaded.
- `GET /api/mlp/active` returns the active MLP JSON: `backend/models/active_mlp.json` points to a version folder if it exists and is valid, otherwise use `asl_landmark_package/asl_landmark_mlp.json`.
- `GET /api/base_samples` returns `asl_landmark_samples.json`.
- **Custom gestures** in `backend/data/custom_gestures.json`, shaped `{ "<label>": [[163 floats], ...] }`, written atomically (temp file + `os.replace`) under a lock:
  - `GET /api/custom` returns everything.
  - `POST /api/custom/{label}` with `{"samples": [[...], ...]}` appends. Validate the label (trimmed, 1 to 40 characters) and that each sample has exactly `feature_dim` finite floats. Return the new count.
  - `PUT /api/custom/{label}` with `{"samples": []}` clears the samples of one gesture but keeps it. `DELETE /api/custom/{label}` removes one gesture. `DELETE /api/custom` removes all.
  - **Migration:** if the file contains samples whose length is not 163 (from an older 129-number design), move the file to `custom_gestures.old_format.bak.json`, start with an empty store, and report it in `/api/health` as a message the UI shows once.
- Serve `frontend/` as static files at `/` (mounted after the API routes, `html=True`). Allow `python app.py` too. `run.sh` must be executable and do: activate the venv, `cd backend`, `uvicorn app:app --port 8000`.

## Frontend behavior

**Layout and controls** (match my mockup): a left column (about 360 px) and a large webcam area on the right.
- A black rounded card with soft shadow titled **Add Gesture**, a large white **+** (focuses the input), a text input with placeholder "New Gesture Title", and a white **Add Word ▶** button (Enter also works). Adding creates a gesture (case-insensitive unique name), selects it, and shows it as a chip with its example count and a small **×** (delete with confirm).
- Below: a green **Train** button and a red **Clear** button. **Clear** deletes the *custom* examples of the selected gesture (with confirm) and keeps the gesture.
- Below that: `N examples` (selected gesture, live while recording) and a green circular spinner, visible only while something is busy (loading MediaPipe or models, recording, training).
- Two large white cards at the bottom: **Start** (begin recording examples for the selected gesture; disabled if no gesture is selected or the camera is off) and **Stop** (stop, flush samples to the backend, then auto-train the kNN).
- A status line, a **Debug** toggle, and a checkbox "Use YOLOv8" (default on, remembered in `localStorage`) so the user can turn YOLO off if the CPU is too slow.

**Main area**: live video with an overlay (hand and pose dots, a bounding box, and the final label with confidence), a large "Detected word" readout, and in Debug mode one line each for YOLO, kNN, MLP, the fused scores, and the current fps. Start the camera automatically with `getUserMedia({video: {width: 1280, height: 720}, audio: false})` and show friendly errors (denied, no camera, backend down, model missing, MediaPipe load failure).

**YOLO loop**: at most 6 requests per second, **never queue**: send the next frame only after the previous response arrives. Draw the raw frame into a hidden canvas 640 px wide, `toBlob('image/jpeg', 0.7)`, POST to `/api/detect`. If the backend fails, show a banner and retry every 2 seconds.

**Recording**: capture at most 12 samples per second, only when pose is detected (constant `RECORD_REQUIRES_HAND = false`; set it to true to also require a hand), cap 400 per gesture, buffer and POST to `/api/custom/{label}` every second and on Stop. Update the visible count immediately. Show live indicators "hand ✓/✗" and "pose ✓/✗" so the user can see what is being detected.

**kNN (`knn.js`)**: the index holds the base samples (label = word, flagged `base`) plus custom samples (in recording order). `train()` rebuilds the index and computes (a) leave-one-out accuracy that excludes same-gesture samples within ±15 positions of the tested one, and (b) a global rejection threshold = 1.5 x the 95th percentile of each sample's nearest same-word distance, computed with the same exclusion (fall back to a constant if there are too few samples). Standardize features with the MLP's `mean` and `std` before measuring distance. `predict()` uses `k = min(5, n)` with a distance-weighted vote and returns `{label, confidence, nearestDist, threshold, accepted, isCustom}`. A gesture titled **Neutral** (case-insensitive) is a reject class: if it wins, the result is "no gesture". Base samples get a smaller weight than custom ones (constant `BASE_SAMPLE_WEIGHT = 0.7`) so a user's own recordings win ties.

## Fusion (`fusion.js`, all constants at the top)

Run per processed frame, using the newest results (YOLO result must be < 700 ms old):
1. If the kNN result is accepted, is a custom word (not one of the 18 base words), and has confidence >= `CUSTOM_MIN_CONF = 0.7`, the final label is that custom word.
2. Otherwise compute a soft vote per label: `score = W_KNN * knnConf(if accepted) + W_MLP * mlpProb + W_YOLO * yoloConf(if that label was detected)` with `W_KNN = 0.35`, `W_MLP = 0.40`, `W_YOLO = 0.25`. The MLP runs only if pose or a hand is available. The best label wins if its score >= `FUSION_MIN_SCORE = 0.40`, otherwise the result is "—".
3. Box for the final label: the YOLO box if YOLO detected that label; otherwise the padded (15%) union bounds of the detected hands; otherwise no box (still show the label).
4. Smooth over the last 7 frames: show a label only if it wins at least 4 of them.
All three signals must be visible in Debug so I can tune the constants.

## Phase 2: live MLP retraining that saves a new `.pt`

(Implement after Phase 1 works.) Because the machine is CPU-only, retraining the small MLP is the fast way to learn new words into a real `.pt` file (seconds, not minutes). YOLOv8 is **not** fine-tuned.

- `POST /api/mlp/train` (body `{epochs?}`, default 60) starts a background job in a **subprocess** (`backend/mlp_trainer.py`, launched with `sys.executable`; it uses `torch` from the venv). Return 409 if a job is running, 400 if there are no custom samples.
- Training data = base samples (`asl_landmark_samples.json`) + custom samples. Classes = the 18 base words in their original order, then the custom words in creation order. A custom word with the same name as a base word simply adds samples to that base class.
- Because only feature vectors are stored, augment at feature level: per class, create copies until each class has about 600 rows (at most 80 copies per sample), adding Gaussian noise (sigma 0.03 to the hand blocks and to indices 126 to 128, sigma 0.04 to indices 129 to 162), and with probability 0.1 zero both hand blocks and indices 126 to 136 (simulating missed hand detections). Cap custom samples at 300 per class by even subsampling. Hold out the last 15% of each custom word's recording (contiguous) as validation and report its accuracy.
- Same architecture as the base MLP with the output size changed: `Linear(163,128), ReLU, Dropout(0.2), Linear(128,64), ReLU, Dropout(0.2), Linear(64,C)`. AdamW (lr 2e-3, weight decay 1e-3), cosine schedule, label smoothing 0.05, batch 256, standardization computed from the training rows (`std` floor 0.05).
- Progress: the worker writes `progress.json` (`state, epoch, epochs, message`). `GET /api/mlp/status` returns it (`{state: "idle"}` if none).
- Finish: save into `backend/models/mlp_versions/<YYYYMMDD_HHMMSS>/` a `.pt` (same checkpoint keys as `asl_landmark_mlp.pt`, with the new `classes`, `arch`, `mean`, `std`) and a `.json` (same format as `asl_landmark_mlp.json`, weights rounded to 5 decimals), plus `result.json` (validation accuracy). Write `backend/models/active_mlp.json` pointing to that folder.
- `POST /api/mlp/reset` restores the original package MLP by removing `active_mlp.json`.
- Frontend: **Train** first rebuilds the kNN instantly, then (if "Also retrain MLP (.pt)" is checked, on by default, remembered) starts the MLP job and shows a progress bar and Cancel. When the job finishes, refetch `/api/mlp/active` and swap the JS MLP in place without reloading; show `MLP updated: <folder> (validation acc 0.xx)`. Add a "Revert to base MLP" button (with confirm).
- The fusion code must use the MLP's own `classes` list so new words appear automatically. When the MLP includes custom words, rule 1 of the fusion no longer needs a special case for them, but keep it working.

## Non-functional requirements

- Modular ES modules (`<script type="module">`), clean code, short comments, no frameworks, no build tools.
- Every failure is visible in the UI (never silent). Use `textContent` (not `innerHTML`) for user-provided names and `encodeURIComponent` in URLs.
- `README.md`: install and run (`./run.sh`), where `best.pt` goes, how to add a gesture (type the name, Add Word, Start, move slowly through angles and distances, Stop), how to tune the fusion constants, the parity self-test, Phase 2, and the limitations below.

## Known limitations (put in the README)

- The base model data is about 12 photos per word from a few signers. Recording the same words yourself in the app (same names) is the most effective accuracy boost.
- All recognizers use single frames, so words defined by motion (thank-you, good, help) are not reliably separable.
- MediaPipe needs the hand and upper body to be visible. Fist-type signs may not produce hand landmarks; the body-pose block still helps for those.
- On CPU the frame rate is limited; use the "Use YOLOv8" checkbox to trade accuracy for speed.

## Acceptance checklist (verify before finishing)

1. `./run.sh` starts the server using the venv `/home/feneti/ai-engineering-camp/myenv`; `http://localhost:8000` shows the UI and asks for the camera.
2. `/api/health` shows `mlp_loaded: true`, `feature_dim: 163`, and `yolo_loaded: true` when `best.pt` is present (a clear message when it is not; the app still works).
3. `python scripts/make_parity_case.py`, then opening `http://localhost:8000/selftest.html` prints PASS for features and MLP probabilities.
4. Signing a base word shows it within about a second; Debug shows the YOLO, kNN and MLP results.
5. Add "hello", Start, move the hand around for about 10 seconds, Stop: at least 60 examples, the kNN reports a leave-one-out accuracy, and "hello" is recognized afterwards. After a page reload it is still there.
6. A hand pose matching nothing does not produce a label ("—").
7. **Clear** empties only the custom examples of the selected gesture, **×** deletes a gesture, and the backend JSON matches.
8. Phase 2: Train starts the MLP job, the progress bar advances, a new folder with `.pt` and `.json` appears under `backend/models/mlp_versions/`, the new word is recognized without a reload, and **Revert to base MLP** restores the original.
9. The UI never freezes while YOLO runs and never queues requests.
