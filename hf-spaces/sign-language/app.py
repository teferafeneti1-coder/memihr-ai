"""
app.py — Sign Language Space wrapper for Hugging Face
FastAPI backend for ASL real-time recognition.
HF Spaces uses port 7860 instead of 8000.
"""

import json, os, sys, math, threading, subprocess
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

# ── Paths ───────────────────────────────────────────────────────────────────
ROOT          = Path(__file__).resolve().parent
PKG           = ROOT / "asl_landmark_package"
MODELS_DIR    = ROOT / "models"
DATA_DIR      = ROOT / "data"
FRONTEND      = ROOT / "frontend"
ACTIVE_MLP    = MODELS_DIR / "active_mlp.json"
CUSTOM_FILE   = DATA_DIR / "custom_gestures.json"
PROGRESS_FILE = MODELS_DIR / "progress.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
(MODELS_DIR / "mlp_versions").mkdir(exist_ok=True)

FEATURE_DIM = 163
PORT = int(os.environ.get("PORT", 7860))

app = FastAPI(title="ASL Sign Language Translator")

# ── Custom gestures ─────────────────────────────────────────────────────────
_custom_lock = threading.Lock()

def _load_custom() -> dict:
    if not CUSTOM_FILE.exists():
        return {}
    try:
        return json.loads(CUSTOM_FILE.read_text())
    except Exception:
        return {}

def _save_custom(data: dict):
    tmp = CUSTOM_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, CUSTOM_FILE)

def _validate_samples(samples):
    if not isinstance(samples, list):
        raise ValueError("samples must be a list")
    for i, s in enumerate(samples):
        if not isinstance(s, list) or len(s) != FEATURE_DIM:
            raise ValueError(f"Sample {i} must have {FEATURE_DIM} floats")
        if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in s):
            raise ValueError(f"Sample {i} contains non-finite values")

def _get_active_mlp_path() -> Path:
    if ACTIVE_MLP.exists():
        try:
            ref    = json.loads(ACTIVE_MLP.read_text())
            folder = MODELS_DIR / "mlp_versions" / ref["folder"]
            p      = folder / ref.get("mlp_json", "mlp.json")
            if p.exists():
                return p
        except Exception:
            pass
    return PKG / "asl_landmark_mlp.json"

# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    mlp_path = _get_active_mlp_path()
    try:
        d = json.loads(mlp_path.read_text())
        return {"mlp_loaded": True, "mlp_classes": d.get("classes", []),
                "feature_dim": d.get("feature_dim"), "mlp_source": str(mlp_path), "message": ""}
    except Exception as e:
        return {"mlp_loaded": False, "mlp_classes": [], "feature_dim": None,
                "mlp_source": str(mlp_path), "message": str(e)}

@app.get("/api/mlp/active")
async def mlp_active():
    mlp_path = _get_active_mlp_path()
    try:
        return JSONResponse(content=json.loads(mlp_path.read_text()))
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/api/base_samples")
async def base_samples():
    p = PKG / "asl_landmark_samples.json"
    try:
        return JSONResponse(content=json.loads(p.read_text()))
    except Exception as e:
        raise HTTPException(500, detail=str(e))

@app.get("/api/custom")
async def custom_get():
    with _custom_lock:
        return _load_custom()

@app.post("/api/custom/{label}")
async def custom_add(label: str, body: dict = Body(...)):
    label = label.strip()
    if not label or len(label) > 40:
        raise HTTPException(400, detail="Label must be 1–40 characters")
    samples = body.get("samples", [])
    try:
        _validate_samples(samples)
    except ValueError as e:
        raise HTTPException(400, detail=str(e))
    with _custom_lock:
        data = _load_custom()
        if label not in data:
            data[label] = []
        data[label].extend(samples)
        _save_custom(data)
        return {"label": label, "count": len(data[label])}

@app.put("/api/custom/{label}")
async def custom_clear_samples(label: str, body: dict = Body(...)):
    label = label.strip()
    with _custom_lock:
        data = _load_custom()
        if label not in data:
            raise HTTPException(404, detail="Gesture not found")
        data[label] = body.get("samples", [])
        _save_custom(data)
        return {"label": label, "count": len(data[label])}

@app.delete("/api/custom/{label}")
async def custom_delete_one(label: str):
    label = label.strip()
    with _custom_lock:
        data = _load_custom()
        if label not in data:
            raise HTTPException(404, detail="Gesture not found")
        del data[label]
        _save_custom(data)
        return {"deleted": label}

@app.delete("/api/custom")
async def custom_delete_all():
    with _custom_lock:
        _save_custom({})
        return {"deleted": "all"}

_train_proc: Optional[subprocess.Popen] = None
_train_lock = threading.Lock()

@app.post("/api/mlp/train")
async def mlp_train(body: dict = Body(default={})):
    global _train_proc
    epochs = int(body.get("epochs", 60))
    with _custom_lock:
        data = _load_custom()
    if not data:
        raise HTTPException(400, detail="No custom samples — nothing to train on")
    with _train_lock:
        if _train_proc is not None and _train_proc.poll() is None:
            raise HTTPException(409, detail="Training already in progress")
        PROGRESS_FILE.write_text(json.dumps({
            "state": "starting", "epoch": 0, "epochs": epochs, "message": "Launching…"
        }))
        trainer = ROOT / "mlp_trainer.py"
        _train_proc = subprocess.Popen(
            [sys.executable, str(trainer), "--epochs", str(epochs), "--root", str(ROOT)],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    return {"status": "started", "epochs": epochs}

@app.get("/api/mlp/status")
async def mlp_status():
    if not PROGRESS_FILE.exists():
        return {"state": "idle"}
    try:
        return json.loads(PROGRESS_FILE.read_text())
    except Exception:
        return {"state": "idle"}

@app.post("/api/mlp/cancel")
async def mlp_cancel():
    global _train_proc
    with _train_lock:
        if _train_proc is not None and _train_proc.poll() is None:
            _train_proc.terminate()
            PROGRESS_FILE.write_text(json.dumps({
                "state": "cancelled", "epoch": 0, "epochs": 0, "message": "Cancelled"
            }))
            return {"status": "cancelled"}
    return {"status": "not_running"}

@app.post("/api/mlp/reset")
async def mlp_reset():
    if ACTIVE_MLP.exists():
        ACTIVE_MLP.unlink()
    return {"status": "reset"}

# Static frontend — must be last
app.mount("/", StaticFiles(directory=str(FRONTEND), html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=PORT, reload=False)
