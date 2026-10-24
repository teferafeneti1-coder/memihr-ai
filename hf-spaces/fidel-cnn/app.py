"""
app.py — Fidel CNN Space wrapper for Hugging Face
Identical to the original app but reads PORT from environment (HF uses 7860).
Model and chars are bundled in the models/ folder.
"""
import base64, io, json, os, random, shutil, threading, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from flask import Flask, jsonify, render_template, request
from PIL import Image, ImageFilter

# ── Config ─────────────────────────────────────────────────────────────────
MODEL_PATH = os.environ.get("FIDEL_MODEL_PATH", "models/fidel_cnn_inference1.pt")
CHARS_PATH = os.environ.get("FIDEL_CHARS_PATH", "models/chars.json")
CORR_DIR   = "corrections"
MANIFEST   = os.path.join(CORR_DIR, "manifest.jsonl")
IMG_SIZE   = 25
NORM_MEAN  = 0.5
NORM_STD   = 0.5
TRAIN_EPOCHS = 3
TRAIN_LR   = 5e-5
MIN_CORRECTIONS_BEFORE_TRAIN = 5
AUGMENT_COPIES = 4

# HF Spaces uses port 7860
PORT = int(os.environ.get("PORT", 7860))

app = Flask(__name__)

# ── Model definition ───────────────────────────────────────────────────────
class FidelCNN(nn.Module):
    def __init__(self, num_classes=287):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

# ── Load model ─────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt   = torch.load(MODEL_PATH, map_location=device)
NUM_CLASSES = ckpt.get("num_classes", 287)

with open(CHARS_PATH, encoding="utf-8") as f:
    CHARS_FULL = json.load(f)

model = FidelCNN(NUM_CLASSES).to(device)
model.load_state_dict(ckpt["model_state"])
model.eval()
CHAR_TO_IDX = {ch: i for i, ch in enumerate(CHARS_FULL)}
model_lock  = threading.Lock()
print(f"✅ Model loaded — {NUM_CLASSES} classes, device={device}")

os.makedirs(CORR_DIR, exist_ok=True)

def _load_manifest():
    entries = []
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try: entries.append(json.loads(line))
                    except: pass
    return entries

correction_buffer = _load_manifest()

# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess_canvas_image(data_url):
    header, encoded = data_url.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("L")
    arr = 255 - np.array(img)
    img = Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = (np.array(img).astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()

def _pil_from_data_url(data_url):
    header, encoded = data_url.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("L")
    arr = 255 - np.array(img)
    return Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

def _tensor_from_pil(img):
    arr = (np.array(img).astype(np.float32) / 255.0 - NORM_MEAN) / NORM_STD
    return torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()

def _augment(img):
    copies = []
    for _ in range(AUGMENT_COPIES):
        aug = img.copy().rotate(random.uniform(-10, 10), resample=Image.BILINEAR, fillcolor=0)
        dx, dy = random.randint(-1, 1), random.randint(-1, 1)
        aug = aug.transform(aug.size, Image.AFFINE, (1, 0, dx, 0, 1, dy), fillcolor=0)
        if random.random() < 0.4:
            aug = aug.filter(ImageFilter.GaussianBlur(radius=0.5))
        copies.append(aug)
    return copies

def predict(data_url, top_k=5):
    tensor = preprocess_canvas_image(data_url).to(device)
    with model_lock:
        with torch.no_grad():
            probs = torch.softmax(model(tensor), dim=1)[0]
            top_probs, top_idx = probs.topk(top_k)
    return [{"char": CHARS_FULL[i.item()], "class_id": i.item(), "prob": p.item()}
            for p, i in zip(top_probs, top_idx)]

# ── Routes (same as original) ──────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/chars")
def api_chars():
    return jsonify({"chars": CHARS_FULL, "num_classes": NUM_CLASSES})

@app.route("/api/predict", methods=["POST"])
def api_predict():
    payload = request.get_json(force=True)
    image_data = payload.get("image")
    if not image_data:
        return jsonify({"error": "no image provided"}), 400
    try:
        return jsonify({"predictions": predict(image_data)})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/correct", methods=["POST"])
def api_correct():
    payload      = request.get_json(force=True)
    image_data   = payload.get("image")
    correct_char = payload.get("correct_char", "").strip()
    if not image_data:
        return jsonify({"error": "no image provided"}), 400
    if correct_char not in CHAR_TO_IDX:
        return jsonify({"error": f"unknown character: {correct_char!r}"}), 400
    fname    = f"corr_{int(time.time()*1000)}_{random.randint(0,9999):04d}.png"
    img_path = os.path.join(CORR_DIR, fname)
    _pil_from_data_url(image_data).save(img_path)
    entry = {"img_path": img_path, "label": correct_char}
    correction_buffer.append(entry)
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return jsonify({"buffered": len(correction_buffer),
                    "ready_to_train": len(correction_buffer) >= MIN_CORRECTIONS_BEFORE_TRAIN})

@app.route("/api/correct/discard_last", methods=["POST"])
def api_correct_discard_last():
    if not correction_buffer:
        return jsonify({"discarded": False, "buffered": 0})
    removed = correction_buffer.pop()
    try: os.remove(removed["img_path"])
    except OSError: pass
    with open(MANIFEST, "w", encoding="utf-8") as f:
        for e in correction_buffer:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
    return jsonify({"discarded": True, "buffered": len(correction_buffer)})

@app.route("/api/corrections/count")
def api_corrections_count():
    return jsonify({"count": len(correction_buffer),
                    "min_needed": MIN_CORRECTIONS_BEFORE_TRAIN,
                    "ready": len(correction_buffer) >= MIN_CORRECTIONS_BEFORE_TRAIN})

@app.route("/api/train", methods=["POST"])
def api_train():
    global model, correction_buffer
    if len(correction_buffer) < MIN_CORRECTIONS_BEFORE_TRAIN:
        return jsonify({"error": f"Need {MIN_CORRECTIONS_BEFORE_TRAIN} corrections (have {len(correction_buffer)})."}), 400
    bak_path = MODEL_PATH + ".before_corrections.bak"
    if not os.path.exists(bak_path):
        shutil.copy2(MODEL_PATH, bak_path)
    images, targets = [], []
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
        return jsonify({"error": "No valid correction images found."}), 400
    X = torch.cat(images).to(device)
    Y = torch.tensor(targets, dtype=torch.long).to(device)
    with model_lock:
        model.train()
        for p in model.parameters(): p.requires_grad = False
        for layer in list(model.features.children())[14:]:
            for p in layer.parameters(): p.requires_grad = True
        for p in model.classifier.parameters(): p.requires_grad = True
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=TRAIN_LR)
        criterion = nn.CrossEntropyLoss()
        losses = []
        for epoch in range(TRAIN_EPOCHS):
            perm = torch.randperm(X.size(0))
            X, Y = X[perm], Y[perm]
            optimizer.zero_grad()
            loss = criterion(model(X), Y)
            loss.backward()
            optimizer.step()
            losses.append(round(loss.item(), 4))
        model.eval()
        for p in model.parameters(): p.requires_grad = True
    torch.save({"model_state": model.state_dict(), "num_classes": NUM_CLASSES,
                "val_acc": ckpt.get("val_acc", "unknown"),
                "correction_rounds": ckpt.get("correction_rounds", 0) + 1,
                "corrections_used": len(correction_buffer)}, MODEL_PATH)
    n_trained = len(correction_buffer)
    correction_buffer.clear()
    open(MANIFEST, "w").close()
    return jsonify({"trained_on": n_trained, "augmented_total": len(images),
                    "epochs": TRAIN_EPOCHS, "losses": losses, "backup": bak_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=False)
