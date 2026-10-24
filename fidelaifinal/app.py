"""
Fidel CNN — Model Testing App  (with learn-from-mistakes)
===========================================================
Serves a browser UI (canvas drawing) for testing the fine-tuned Fidel CNN
model in two modes:
  - Free draw:    draw any character, see top-5 predictions, flag wrong ones
  - Guided test:  cycles through all 287 characters, auto-buffers every miss

Learn-from-mistakes:
  - Corrections are stored in  corrections/  as PNG files + a JSONL manifest
  - Once ≥ MIN_CORRECTIONS_BEFORE_TRAIN are buffered, /api/train does a small
    fine-tune: only last conv block + classifier, 3 epochs, LR=5e-5, with
    light augmentation of each sample.
  - A one-shot backup (.bak) is saved before the first training round.

Setup:
  1. Put your exported model at:  models/fidel_cnn_inference1.pt
  2. Put the exported labels at:  models/chars.json
  3. pip install -r requirements.txt
  4. python3 app.py
  5. Open http://localhost:5000
"""

import base64
import io
import json
import os
import random
import shutil
import threading
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageFilter

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH   = os.environ.get("FIDEL_MODEL_PATH", "models/fidel_cnn_inference1.pt")
CHARS_PATH   = os.environ.get("FIDEL_CHARS_PATH", "models/chars.json")
CORR_DIR     = "corrections"
MANIFEST     = os.path.join(CORR_DIR, "manifest.jsonl")
IMG_SIZE     = 25
NORM_MEAN    = 0.5
NORM_STD     = 0.5

TRAIN_EPOCHS             = 3
TRAIN_LR                 = 5e-5
MIN_CORRECTIONS_BEFORE_TRAIN = 5
AUGMENT_COPIES           = 4   # extra augmented copies per correction sample

app = Flask(__name__)

# ── Model definition (must match training exactly) ─────────────────────────
class FidelCNN(nn.Module):
    def __init__(self, num_classes=287):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),    # 0-2
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),   # 3-5
            nn.MaxPool2d(2),                                                   # 6
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),   # 7-9
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),   # 10-12
            nn.MaxPool2d(2),                                                   # 13
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), # 14-16
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),# 17-19
            nn.MaxPool2d(2),                                                   # 20
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── Load model + labels once at startup ────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model checkpoint not found at '{MODEL_PATH}'. "
        f"Set FIDEL_MODEL_PATH or copy your exported .pt file there."
    )
if not os.path.exists(CHARS_PATH):
    raise FileNotFoundError(
        f"Character labels not found at '{CHARS_PATH}'. "
        f"Export CHARS_FULL from your notebook to chars.json first."
    )

ckpt = torch.load(MODEL_PATH, map_location=device)
NUM_CLASSES = ckpt.get("num_classes", 287)

with open(CHARS_PATH, "r", encoding="utf-8") as f:
    CHARS_FULL = json.load(f)

if len(CHARS_FULL) != NUM_CLASSES:
    raise ValueError(
        f"chars.json has {len(CHARS_FULL)} entries but the model expects "
        f"{NUM_CLASSES} classes."
    )

model = FidelCNN(NUM_CLASSES).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()

# Map character → class index (for converting correction labels to targets)
CHAR_TO_IDX = {ch: i for i, ch in enumerate(CHARS_FULL)}

# Lock so the train thread can't run while a prediction is happening and vice versa
model_lock = threading.Lock()

print(f"Loaded model from {MODEL_PATH}  ({NUM_CLASSES} classes)  device={device}")
print(f"Checkpoint reported val_acc: {ckpt.get('val_acc', 'unknown')}")

# ── Corrections dir setup ──────────────────────────────────────────────────
os.makedirs(CORR_DIR, exist_ok=True)

def _load_manifest():
    """Return list of {img_path, label} dicts from disk."""
    entries = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return entries

# In-memory buffer (synced to disk immediately on each /api/correct call)
correction_buffer = _load_manifest()
print(f"Loaded {len(correction_buffer)} existing corrections from disk.")


# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess_canvas_image(data_url: str) -> torch.Tensor:
    """
    data_url: 'data:image/png;base64,....' from an HTML canvas (black stroke,
    white background).  Returns a (1,1,H,W) float tensor, normalised.
    """
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")

    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("L")

    arr = 255 - np.array(img)   # invert: white stroke on black
    img = Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()


def _pil_from_data_url(data_url: str) -> Image.Image:
    """Return a greyscale (L-mode, inverted, 25×25) PIL image from a data URL."""
    header, encoded = data_url.split(",", 1)
    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGBA")
    bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("L")
    arr = 255 - np.array(img)
    return Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)


def _tensor_from_pil(img: Image.Image) -> torch.Tensor:
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()


# ── Augmentation ───────────────────────────────────────────────────────────
def _augment(img: Image.Image) -> list:
    """
    Return AUGMENT_COPIES mildly-augmented versions of a 25×25 PIL image.
    Kept very light so we don't distort Fidel characters beyond recognition.
    """
    copies = []
    for _ in range(AUGMENT_COPIES):
        aug = img.copy()
        # Tiny random rotation (±10 degrees)
        angle = random.uniform(-10, 10)
        aug = aug.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
        # Random translate by ±1px
        dx, dy = random.randint(-1, 1), random.randint(-1, 1)
        aug = aug.transform(
            aug.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0
        )
        # Occasional mild blur (softens the stroke slightly)
        if random.random() < 0.4:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=0.5))
        copies.append(aug)
    return copies


def predict(data_url: str, top_k: int = 5):
    tensor = preprocess_canvas_image(data_url).to(device)
    with model_lock:
        with torch.no_grad():
            logits = model(tensor)
            probs  = torch.softmax(logits, dim=1)[0]
            top_probs, top_idx = probs.topk(top_k)
    return [
        {"char": CHARS_FULL[idx.item()], "class_id": idx.item(), "prob": prob.item()}
        for prob, idx in zip(top_probs, top_idx)
    ]


# ── Routes ──────────────────────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chars")
def api_chars():
    return jsonify({"chars": CHARS_FULL, "num_classes": NUM_CLASSES})


@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload    = request.get_json(force=True)
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "no image provided"}), 400
    try:
        results = predict(image_data, top_k=5)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"predictions": results})


@app.route("/api/correct", methods=["POST"])
def api_correct():
    """
    Buffer a (image, correct_label) correction pair.
    Body JSON: { image: <data_url>, correct_char: "<char>" }
    """
    payload      = request.get_json(force=True)
    image_data   = payload.get("image")
    correct_char = payload.get("correct_char", "").strip()

    if not image_data:
        return jsonify({"error": "no image provided"}), 400
    if correct_char not in CHAR_TO_IDX:
        return jsonify({"error": f"unknown character: {correct_char!r}"}), 400

    # Save PNG to corrections/
    fname     = f"corr_{int(time.time() * 1000)}_{random.randint(0, 9999):04d}.png"
    img_path  = os.path.join(CORR_DIR, fname)
    pil_img   = _pil_from_data_url(image_data)
    pil_img.save(img_path)

    entry = {"img_path": img_path, "label": correct_char}
    correction_buffer.append(entry)

    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return jsonify({
        "buffered": len(correction_buffer),
        "ready_to_train": len(correction_buffer) >= MIN_CORRECTIONS_BEFORE_TRAIN,
    })


@app.route("/api/correct/discard_last", methods=["POST"])
def api_correct_discard_last():
    """Remove the most recently buffered correction (user said it was actually right)."""
    if not correction_buffer:
        return jsonify({"discarded": False, "buffered": 0})

    removed = correction_buffer.pop()

    # Remove the PNG file if it exists
    try:
        os.remove(removed["img_path"])
    except OSError:
        pass

    # Rewrite manifest without the last entry
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for entry in correction_buffer:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return jsonify({"discarded": True, "buffered": len(correction_buffer)})


@app.route("/api/corrections/count")
def api_corrections_count():
    return jsonify({
        "count": len(correction_buffer),
        "min_needed": MIN_CORRECTIONS_BEFORE_TRAIN,
        "ready": len(correction_buffer) >= MIN_CORRECTIONS_BEFORE_TRAIN,
    })


@app.route("/api/train", methods=["POST"])
def api_train():
    """
    Fine-tune the model on the buffered corrections.
    Only the last conv block (features[14:]) + classifier are updated.
    Early conv layers stay frozen.
    """
    global model, correction_buffer

    if len(correction_buffer) < MIN_CORRECTIONS_BEFORE_TRAIN:
        return jsonify({
            "error": f"Need at least {MIN_CORRECTIONS_BEFORE_TRAIN} corrections "
                     f"(have {len(correction_buffer)})."
        }), 400

    # ── Backup checkpoint (once) ──────────────────────────────────────────
    bak_path = MODEL_PATH + ".before_corrections.bak"
    if not os.path.exists(bak_path):
        shutil.copy2(MODEL_PATH, bak_path)
        print(f"Backup saved to {bak_path}")

    # ── Build dataset (originals + augmented copies) ──────────────────────
    images  = []   # torch tensors (1,1,H,W)
    targets = []   # int class indices

    for entry in correction_buffer:
        try:
            pil = Image.open(entry["img_path"]).convert("L")
        except FileNotFoundError:
            continue
        label_idx = CHAR_TO_IDX[entry["label"]]
        images.append(_tensor_from_pil(pil))
        targets.append(label_idx)
        for aug_pil in _augment(pil):
            images.append(_tensor_from_pil(aug_pil))
            targets.append(label_idx)

    if not images:
        return jsonify({"error": "No valid correction images found on disk."}), 400

    X = torch.cat(images, dim=0).to(device)          # (N,1,H,W)
    Y = torch.tensor(targets, dtype=torch.long).to(device)

    # ── Freeze all layers, then unfreeze last conv block + classifier ──────
    with model_lock:
        model.train()
        for param in model.parameters():
            param.requires_grad = False
        # Unfreeze features[14:] = last conv block (2×Conv+BN+ReLU + MaxPool)
        for layer in list(model.features.children())[14:]:
            for param in layer.parameters():
                param.requires_grad = True
        # Unfreeze classifier head entirely
        for param in model.classifier.parameters():
            param.requires_grad = True

        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=TRAIN_LR,
        )
        criterion = nn.CrossEntropyLoss()

        losses = []
        for epoch in range(TRAIN_EPOCHS):
            # Shuffle
            perm  = torch.randperm(X.size(0))
            X, Y  = X[perm], Y[perm]
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, Y)
            loss.backward()
            optimizer.step()
            losses.append(round(loss.item(), 4))
            print(f"  Correction epoch {epoch+1}/{TRAIN_EPOCHS}  loss={loss.item():.4f}")

        model.eval()

        # Re-enable all grads for future normal forward passes
        for param in model.parameters():
            param.requires_grad = True

    # ── Save updated checkpoint ───────────────────────────────────────────
    save_ckpt = {
        "model_state": model.state_dict(),
        "num_classes":  NUM_CLASSES,
        "val_acc":      ckpt.get("val_acc", "unknown"),
        "correction_rounds": ckpt.get("correction_rounds", 0) + 1,
        "corrections_used":  len(correction_buffer),
    }
    torch.save(save_ckpt, MODEL_PATH)
    print(f"Checkpoint updated → {MODEL_PATH}  (corrections used: {len(correction_buffer)})")

    n_trained = len(correction_buffer)
    # Clear buffer (but keep files on disk for reference)
    correction_buffer.clear()
    # Truncate manifest so next round starts fresh
    with open(MANIFEST, "w", encoding="utf-8") as f:
        pass

    return jsonify({
        "trained_on": n_trained,
        "augmented_total": len(images),
        "epochs": TRAIN_EPOCHS,
        "losses": losses,
        "backup": bak_path,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
