#!/usr/bin/env bash
# push_to_hf.sh
# Uploads all 3 Spaces + the Adulis model to Hugging Face.
# Run this once: bash push_to_hf.sh
#
# Prerequisites:
#   pip install huggingface_hub
#   huggingface-cli login   (enter your HF token when prompted)

set -euo pipefail

HF_USER="fenetii"
PYTHON="/home/feneti/ai-engineering/myenv/bin/python"
HF_CLI="hf"
BASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project source paths
FIDEL_SRC="/home/feneti/egate/egate/fidelaifinal"
ADULIS_SRC="/home/feneti/egate/egate/oromifa-amharic & amharic-oromifa/adulis"
ADULIS_MODEL_SRC="/home/feneti/egate/egate/oromifa-amharic & amharic-oromifa/archive"
SL_SRC="/home/feneti/egate/egate/egate/signlanguagetranslate"

echo ""
echo "╔══════════════════════════════════════════════╗"
echo "║   memihr ai — Hugging Face Upload Script     ║"
echo "║   User: ${HF_USER}                           ║"
echo "╚══════════════════════════════════════════════╝"
echo ""

# ── Step 0: Check login ────────────────────────────────────────────────────
echo "▶ Step 0: Checking HF login…"
"$HF_CLI" auth whoami || {
    echo ""
    echo "  Not logged in. Run: hf auth login"
    echo "  Then paste your token from https://huggingface.co/settings/tokens"
    exit 1
}
echo ""

# ── Step 1: Upload Adulis NLLB model to Model Hub ─────────────────────────
echo "▶ Step 1: Uploading Adulis NLLB model → ${HF_USER}/adulis-nllb"
echo "  (This is 2.4 GB — may take 10–30 minutes depending on connection)"
echo ""

"$PYTHON" - <<PYMODEL
from huggingface_hub import HfApi
api = HfApi()

repo_id = "${HF_USER}/adulis-nllb"

# Create repo if it doesn't exist
try:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)
    print(f"  Repo ready: {repo_id}")
except Exception as e:
    print(f"  Repo create: {e}")

# Upload the whole archive folder
print("  Uploading model files…")
api.upload_folder(
    folder_path="${ADULIS_MODEL_SRC}",
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload fine-tuned Adulis NLLB model",
    ignore_patterns=["optimizer.pt", "rng_state_*.pth", "scaler.pt",
                     "scheduler.pt", "training_args.bin"],
)
print("  ✅ Model uploaded!")
PYMODEL

echo ""

# ── Step 2: Upload Fidel CNN Space ─────────────────────────────────────────
echo "▶ Step 2: Uploading Fidel CNN Space → spaces/${HF_USER}/fidel-cnn"
echo ""

"$PYTHON" - <<PYFIDEL
from huggingface_hub import HfApi
import shutil, os

api     = HfApi()
repo_id = "${HF_USER}/fidel-cnn"
space_dir = "${BASE}/fidel-cnn"

# Create Space
try:
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"  Space ready: {repo_id}")
except Exception as e:
    print(f"  Space create: {e}")

# Copy templates + static from original project
import shutil, pathlib
src = pathlib.Path("${FIDEL_SRC}")
dst = pathlib.Path(space_dir)

# templates
(dst / "templates").mkdir(exist_ok=True)
for f in (src / "templates").glob("*"):
    shutil.copy2(f, dst / "templates" / f.name)
    print(f"  Copied template: {f.name}")

# static
if (src / "static").exists():
    shutil.copytree(src / "static", dst / "static", dirs_exist_ok=True)
    print("  Copied static/")

# models (the .pt and chars.json — small files)
(dst / "models").mkdir(exist_ok=True)
for f in (src / "models").glob("*.pt"):
    shutil.copy2(f, dst / "models" / f.name)
    print(f"  Copied model: {f.name}")
for f in (src / "models").glob("*.json"):
    shutil.copy2(f, dst / "models" / f.name)
    print(f"  Copied labels: {f.name}")

# Upload
print("  Uploading to HF Spaces…")
api.upload_folder(
    folder_path=space_dir,
    repo_id=repo_id,
    repo_type="space",
    commit_message="Upload Fidel CNN Space",
    ignore_patterns=["__pycache__", "*.pyc", "corrections/", "*.bak"],
)
print("  ✅ Fidel CNN Space uploaded!")
print(f"  🌐 https://huggingface.co/spaces/${HF_USER}/fidel-cnn")
PYFIDEL

echo ""

# ── Step 3: Upload Adulis Space ────────────────────────────────────────────
echo "▶ Step 3: Uploading Adulis Space → spaces/${HF_USER}/adulis"
echo ""

"$PYTHON" - <<PYADULIS
from huggingface_hub import HfApi
import shutil, pathlib

api     = HfApi()
repo_id = "${HF_USER}/adulis"
space_dir = pathlib.Path("${BASE}/adulis")
src       = pathlib.Path("${ADULIS_SRC}")

try:
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"  Space ready: {repo_id}")
except Exception as e:
    print(f"  Space create: {e}")

# Copy templates + static + data from original project
for folder in ["templates", "static", "data"]:
    if (src / folder).exists():
        shutil.copytree(src / folder, space_dir / folder, dirs_exist_ok=True)
        print(f"  Copied {folder}/")

print("  Uploading to HF Spaces…")
api.upload_folder(
    folder_path=str(space_dir),
    repo_id=repo_id,
    repo_type="space",
    commit_message="Upload Adulis Space",
    ignore_patterns=["__pycache__", "*.pyc"],
)
print("  ✅ Adulis Space uploaded!")
print(f"  🌐 https://huggingface.co/spaces/${HF_USER}/adulis")
PYADULIS

echo ""

# ── Step 4: Upload Sign Language Space ────────────────────────────────────
echo "▶ Step 4: Uploading Sign Language Space → spaces/${HF_USER}/sign-language"
echo ""

"$PYTHON" - <<PYSL
from huggingface_hub import HfApi
import shutil, pathlib

api     = HfApi()
repo_id = "${HF_USER}/sign-language"
space_dir = pathlib.Path("${BASE}/sign-language")
sl_root   = pathlib.Path("${SL_SRC}")

try:
    api.create_repo(repo_id=repo_id, repo_type="space", space_sdk="docker", exist_ok=True)
    print(f"  Space ready: {repo_id}")
except Exception as e:
    print(f"  Space create: {e}")

# Copy frontend, asl_landmark_package, mlp_trainer
for folder in ["frontend", "asl_landmark_package"]:
    if (sl_root / folder).exists():
        shutil.copytree(sl_root / folder, space_dir / folder, dirs_exist_ok=True)
        print(f"  Copied {folder}/")

# mlp_trainer.py
trainer = sl_root / "backend" / "mlp_trainer.py"
if trainer.exists():
    shutil.copy2(trainer, space_dir / "mlp_trainer.py")
    print("  Copied mlp_trainer.py")

print("  Uploading to HF Spaces…")
api.upload_folder(
    folder_path=str(space_dir),
    repo_id=repo_id,
    repo_type="space",
    commit_message="Upload Sign Language Space",
    ignore_patterns=["__pycache__", "*.pyc", "data/", "models/mlp_versions/"],
)
print("  ✅ Sign Language Space uploaded!")
print(f"  🌐 https://huggingface.co/spaces/${HF_USER}/sign-language")
PYSL

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║  ✅ ALL DONE!                                                ║"
echo "║                                                              ║"
echo "║  Your live links:                                            ║"
echo "║  ✍️  https://huggingface.co/spaces/${HF_USER}/fidel-cnn      ║"
echo "║  🌍  https://huggingface.co/spaces/${HF_USER}/adulis         ║"
echo "║  🤟  https://huggingface.co/spaces/${HF_USER}/sign-language  ║"
echo "║                                                              ║"
echo "║  Model: https://huggingface.co/${HF_USER}/adulis-nllb        ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo ""
echo "NOTE: HF Spaces build takes 3–5 minutes after upload."
echo "      Adulis first load takes ~3 min (downloading 2.4 GB model)."
