"""
Render deploy entry point for Fidel CNN.
Uses ONNX Runtime instead of PyTorch for inference.
ONNXRuntime ~50 MB RAM vs PyTorch ~400 MB — fits on Render free tier.
"""
import os, sys, base64, io, json
import numpy as np
from PIL import Image
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "fidelaifinal")
)
os.chdir(PROJECT_ROOT)

ONNX_PATH  = os.path.join(PROJECT_ROOT, "models", "fidel_cnn.onnx")
CHARS_PATH = os.environ.get("FIDEL_CHARS_PATH",
              os.path.join(PROJECT_ROOT, "models", "chars.json"))
CORR_DIR   = os.path.join(PROJECT_ROOT, "corrections")
MANIFEST   = os.path.join(CORR_DIR, "manifest.jsonl")
IMG_SIZE   = 25
NORM_MEAN  = 0.5
NORM_STD   = 0.5

os.makedirs(CORR_DIR, exist_ok=True)

# ── Load ONNX model ────────────────────────────────────────────────────────
print(f"[fidel] loading ONNX model: {ONNX_PATH}", flush=True)
session = ort.InferenceSession(ONNX_PATH,
           providers=["CPUExecutionProvider"])
print("[fidel] ONNX model ready ✅", flush=True)

with open(CHARS_PATH, encoding="utf-8") as f:
    CHARS_FULL = json.load(f)

NUM_CLASSES = len(CHARS_FULL)
CHAR_TO_IDX = {ch: i for i, ch in enumerate(CHARS_FULL)}
print(f"[fidel] {NUM_CLASSES} classes loaded", flush=True)

# ── Preprocessing ──────────────────────────────────────────────────────────
def preprocess(data_url: str) -> np.ndarray:
    header, encoded = data_url.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255, 255, 255, 255))
    img = Image.alpha_composite(bg, img).convert("L")
    arr = 255 - np.array(img)   # invert
    img = Image.fromarray(arr).resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
    arr = np.array(img).astype(np.float32) / 255.0
    arr = (arr - NORM_MEAN) / NORM_STD
    return arr.reshape(1, 1, IMG_SIZE, IMG_SIZE).astype(np.float32)

def predict(data_url: str, top_k: int = 5):
    tensor = preprocess(data_url)
    logits = session.run(None, {"input": tensor})[0][0]
    # softmax
    e = np.exp(logits - logits.max())
    probs = e / e.sum()
    top_idx = np.argsort(probs)[::-1][:top_k]
    return [
        {"char": CHARS_FULL[i], "class_id": int(i), "prob": float(probs[i])}
        for i in top_idx
    ]

# ── Flask app ──────────────────────────────────────────────────────────────
app = Flask(__name__, template_folder=os.path.join(PROJECT_ROOT, "templates"),
            static_folder=os.path.join(PROJECT_ROOT, "static"))

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
        return jsonify({"predictions": results})
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": str(e)}), 400

@app.route("/api/correct", methods=["POST"])
def api_correct():
    """Buffer a correction (save PNG to disk)."""
    import time, random as rnd
    payload      = request.get_json(force=True)
    image_data   = payload.get("image")
    correct_char = payload.get("correct_char", "").strip()
    if not image_data:
        return jsonify({"error": "no image provided"}), 400
    if correct_char not in CHAR_TO_IDX:
        return jsonify({"error": f"unknown character: {correct_char!r}"}), 400
    fname    = f"corr_{int(time.time()*1000)}_{rnd.randint(0,9999):04d}.png"
    img_path = os.path.join(CORR_DIR, fname)
    # save PNG
    header, encoded = image_data.split(",", 1)
    img = Image.open(io.BytesIO(base64.b64decode(encoded))).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (255,255,255,255))
    Image.alpha_composite(bg, img).convert("L").save(img_path)
    entry = {"img_path": img_path, "label": correct_char}
    with open(MANIFEST, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    # count buffered
    count = sum(1 for _ in open(MANIFEST)) if os.path.exists(MANIFEST) else 0
    return jsonify({"buffered": count, "ready_to_train": count >= 5})

@app.route("/api/correct/discard_last", methods=["POST"])
def api_correct_discard_last():
    if not os.path.exists(MANIFEST):
        return jsonify({"discarded": False, "buffered": 0})
    lines = open(MANIFEST).readlines()
    if not lines:
        return jsonify({"discarded": False, "buffered": 0})
    try:
        entry = json.loads(lines[-1])
        try: os.remove(entry["img_path"])
        except OSError: pass
    except Exception: pass
    with open(MANIFEST, "w") as f:
        f.writelines(lines[:-1])
    return jsonify({"discarded": True, "buffered": len(lines)-1})

@app.route("/api/corrections/count")
def api_corrections_count():
    count = sum(1 for _ in open(MANIFEST)) if os.path.exists(MANIFEST) else 0
    return jsonify({"count": count, "min_needed": 5, "ready": count >= 5})

@app.route("/api/train", methods=["POST"])
def api_train():
    # Training not available on Render free tier (no PyTorch)
    return jsonify({"error": "Training not available on this deployment. Run locally for fine-tuning."}), 503

@app.route("/api/debug")
def api_debug():
    return jsonify({
        "cwd": os.getcwd(),
        "onnx_exists": os.path.exists(ONNX_PATH),
        "chars_count": NUM_CLASSES,
        "ort_version": ort.__version__,
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
