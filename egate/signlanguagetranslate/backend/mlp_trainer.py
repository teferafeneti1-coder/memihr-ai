"""
mlp_trainer.py – background subprocess for MLP retraining.
Launched by app.py:
    python mlp_trainer.py --epochs 60 --root /path/to/signlanguagetranslate
"""
import argparse, json, math, os, sys, time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=60)
parser.add_argument("--root",   type=str, required=True)
args = parser.parse_args()

ROOT       = Path(args.root)
PKG        = ROOT / "asl_landmark_package"
MODELS_DIR = ROOT / "backend" / "models"
DATA_DIR   = ROOT / "backend" / "data"
PROGRESS   = MODELS_DIR / "progress.json"
ACTIVE_MLP = MODELS_DIR / "active_mlp.json"

FEATURE_DIM     = 163
TARGET_PER_CLASS = 600
MAX_COPIES      = 80
MAX_CUSTOM_SAMP = 300
VAL_FRAC        = 0.15
BATCH           = 256

def write_progress(state, epoch, epochs, message, **extra):
    d = {"state": state, "epoch": epoch, "epochs": epochs, "message": message}
    d.update(extra)
    tmp = PROGRESS.with_suffix(".tmp")
    tmp.write_text(json.dumps(d))
    os.replace(tmp, PROGRESS)

write_progress("loading", 0, args.epochs, "Loading data…")

# ── load base samples ─────────────────────────────────────────────────────────
try:
    base_raw = json.loads((PKG / "asl_landmark_samples.json").read_text())
except Exception as e:
    write_progress("error", 0, args.epochs, f"Could not load base samples: {e}")
    sys.exit(1)

base_classes = json.loads((PKG / "classes.json").read_text())

# ── load custom samples ───────────────────────────────────────────────────────
custom_file = DATA_DIR / "custom_gestures.json"
custom_raw  = {}
if custom_file.exists():
    try:
        custom_raw = json.loads(custom_file.read_text())
    except Exception:
        custom_raw = {}

# ── determine classes (base order, then new custom) ───────────────────────────
base_set = set(base_classes)
custom_new = [k for k in custom_raw if k not in base_set]
all_classes = base_classes + custom_new
class_to_idx = {c: i for i, c in enumerate(all_classes)}
C = len(all_classes)

# ── gather raw arrays per class ───────────────────────────────────────────────
class_rows = {c: [] for c in all_classes}
for c, rows in base_raw.items():
    if c in class_to_idx:
        class_rows[c].extend(rows)
for c, rows in custom_raw.items():
    if c in class_to_idx:
        class_rows[c].extend(rows)

# ── split validation (last 15% of custom words) ───────────────────────────────
val_rows = []   # (features, label_idx)
train_class_rows = {c: list(v) for c, v in class_rows.items()}
for c in custom_new:
    rows = class_rows.get(c, [])
    if len(rows) >= 4:
        n_val = max(1, int(len(rows) * VAL_FRAC))
        val_part   = rows[-n_val:]
        train_part = rows[:-n_val]
        train_class_rows[c] = train_part
        for r in val_part:
            val_rows.append((r, class_to_idx[c]))

# ── augment training rows ─────────────────────────────────────────────────────
rng = np.random.default_rng(0)

def augment(row_f, n_extra):
    out = []
    for _ in range(n_extra):
        x = row_f.copy()
        # Gaussian noise on hand blocks
        x[:126]   += rng.normal(0, 0.03, 126).astype(np.float32)
        x[126:129] += rng.normal(0, 0.03, 3).astype(np.float32)
        x[129:163] += rng.normal(0, 0.04, 34).astype(np.float32)
        # simulate missed hand
        if rng.random() < 0.1:
            x[:137] = 0
        out.append(x)
    return out

all_train_X, all_train_Y = [], []

for c, rows in train_class_rows.items():
    if not rows:
        continue
    idx = class_to_idx[c]
    # subsample custom
    if c not in base_set and len(rows) > MAX_CUSTOM_SAMP:
        step = len(rows) / MAX_CUSTOM_SAMP
        rows = [rows[int(i * step)] for i in range(MAX_CUSTOM_SAMP)]
    arrs = [np.array(r, np.float32) for r in rows]
    # augment
    n_have = len(arrs)
    n_need = max(0, TARGET_PER_CLASS - n_have)
    copies = []
    for i in range(n_need):
        base_i = arrs[i % n_have]
        n_copy = min(MAX_COPIES, n_need - len(copies))
        copies.extend(augment(base_i, 1))
        if len(copies) >= n_need:
            break
    all_arrs = arrs + copies[:n_need]
    all_train_X.extend(all_arrs)
    all_train_Y.extend([idx] * len(all_arrs))

write_progress("training", 0, args.epochs, "Computing normalisation…")

X_train = np.stack(all_train_X).astype(np.float32)
Y_train = np.array(all_train_Y, dtype=np.int64)

# ── normalisation ──────────────────────────────────────────────────────────────
mean = X_train.mean(0)
std  = X_train.std(0)
std  = np.maximum(std, 0.05)

X_train_n = (X_train - mean) / std

# ── validation set ────────────────────────────────────────────────────────────
if val_rows:
    X_val = np.stack([np.array(r, np.float32) for r, _ in val_rows])
    Y_val = np.array([y for _, y in val_rows], dtype=np.int64)
    X_val_n = (X_val - mean) / std
else:
    X_val_n = None
    Y_val   = None

# ── model ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, n_classes),
        )
    def forward(self, x): return self.net(x)

model = MLP(FEATURE_DIM, C)
optim = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-3)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs)
crit  = nn.CrossEntropyLoss(label_smoothing=0.05)

Xt = torch.from_numpy(X_train_n)
Yt = torch.from_numpy(Y_train)
ds = TensorDataset(Xt, Yt)
dl = DataLoader(ds, batch_size=BATCH, shuffle=True, drop_last=False)

# ── training loop ──────────────────────────────────────────────────────────────
write_progress("training", 0, args.epochs, "Training…")

for epoch in range(1, args.epochs + 1):
    model.train()
    total_loss = 0
    for xb, yb in dl:
        optim.zero_grad()
        loss = crit(model(xb), yb)
        loss.backward()
        optim.step()
        total_loss += loss.item()
    sched.step()
    if epoch % 5 == 0 or epoch == args.epochs:
        write_progress("training", epoch, args.epochs,
                       f"Epoch {epoch}/{args.epochs}  loss={total_loss/len(dl):.4f}")

# ── validation accuracy ────────────────────────────────────────────────────────
val_acc = None
if X_val_n is not None:
    model.eval()
    with torch.no_grad():
        preds = model(torch.from_numpy(X_val_n)).argmax(1).numpy()
    val_acc = float((preds == Y_val).mean())

write_progress("saving", args.epochs, args.epochs, "Saving model…")

# ── save ──────────────────────────────────────────────────────────────────────
ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
outdir = MODELS_DIR / "mlp_versions" / ts
outdir.mkdir(parents=True, exist_ok=True)

# extract weights for JSON
def to_list(t): return t.detach().cpu().tolist()

layers_json = []
sd = model.state_dict()
# net.0 → linear0  net.3 → linear1  net.6 → linear2
for prefix in ["net.0", "net.3", "net.6"]:
    W = sd[f"{prefix}.weight"]
    b = sd[f"{prefix}.bias"]
    # round to 5 decimals for compact JSON
    layers_json.append({
        "W": [[round(v, 5) for v in row] for row in to_list(W)],
        "b": [round(v, 5) for v in to_list(b)],
    })

mlp_json = {
    "classes":     all_classes,
    "mean":        mean.tolist(),
    "std":         std.tolist(),
    "activation":  "relu",
    "feature_dim": FEATURE_DIM,
    "feature_spec": "retrained",
    "arch":        f"Linear(163,128) ReLU Dropout(0.2) Linear(128,64) ReLU Dropout(0.2) Linear(64,{C})",
    "layers":      layers_json,
}

json_path = outdir / "mlp.json"
json_path.write_text(json.dumps(mlp_json))

# PyTorch checkpoint
pt_path = outdir / "mlp.pt"
torch.save({
    "state_dict":  model.state_dict(),
    "classes":     all_classes,
    "mean":        mean.tolist(),
    "std":         std.tolist(),
    "feature_dim": FEATURE_DIM,
    "arch":        mlp_json["arch"],
    "feature_spec": "retrained",
}, str(pt_path))

# result
result = {
    "val_acc":  val_acc,
    "classes":  all_classes,
    "n_train":  len(X_train),
    "epochs":   args.epochs,
    "folder":   ts,
}
(outdir / "result.json").write_text(json.dumps(result))

# active pointer
ACTIVE_MLP.write_text(json.dumps({"folder": ts, "mlp_json": "mlp.json"}))

acc_str = f"{val_acc:.4f}" if val_acc is not None else "n/a"
write_progress("done", args.epochs, args.epochs,
               f"Done – val_acc={acc_str}  folder={ts}",
               val_acc=val_acc, folder=ts)
sys.exit(0)
