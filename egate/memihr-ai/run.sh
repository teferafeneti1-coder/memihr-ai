#!/usr/bin/env bash
# run.sh – start the memihr ai dashboard (and optionally all projects).
#
# Usage:
#   ./run.sh                  # start dashboard; autostart all projects
#   ./run.sh --no-autostart   # start dashboard only; don't wake projects

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MYENV="/home/feneti/ai-engineering/myenv"
PYTHON="$MYENV/bin/python"
ACTIVATE="$MYENV/bin/activate"

# ── Activate the shared virtual environment ───────────────────────────────────
# shellcheck source=/dev/null
source "$ACTIVATE"

# ── Ensure dashboard dependencies are installed ───────────────────────────────
echo "Checking memihr ai dependencies…"
"$PYTHON" - <<'PYCHECK'
import importlib.util, sys
missing = []
for pkg in ("fastapi", "uvicorn", "psutil"):
    if importlib.util.find_spec(pkg) is None:
        missing.append(pkg)
if missing:
    print(f"  Installing missing packages: {missing}", flush=True)
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install"] + missing)
else:
    print("  All memihr ai packages present.", flush=True)
PYCHECK

# ── Handle --no-autostart flag ────────────────────────────────────────────────
AUTOSTART_ENV=""
for arg in "$@"; do
    if [[ "$arg" == "--no-autostart" ]]; then
        AUTOSTART_ENV="1"
        echo "  --no-autostart: projects will NOT be woken automatically."
    fi
done

# Write a temporary settings override if needed
if [[ -n "$AUTOSTART_ENV" ]]; then
    mkdir -p "$SCRIPT_DIR/data"
    "$PYTHON" - <<PYNO
import json, pathlib
p = pathlib.Path("$SCRIPT_DIR/data/settings.json")
s = json.loads(p.read_text()) if p.exists() else {}
s["autostart_on_launch"] = False
p.write_text(json.dumps(s, indent=2))
PYNO
fi

# ── Report project-package availability (informational) ──────────────────────
echo ""
echo "Checking project packages (flask, torch, transformers, fastapi, uvicorn, sentencepiece)…"
"$PYTHON" - <<'PYPKG'
import importlib.util
pkgs = ["flask", "torch", "transformers", "fastapi", "uvicorn", "sentencepiece"]
for pkg in pkgs:
    spec = importlib.util.find_spec(pkg)
    status = "OK" if spec else "MISSING"
    print(f"  {pkg:20s} {status}")
PYPKG

echo ""
echo "Starting memihr ai on http://127.0.0.1:9000 …"
echo "Press Ctrl+C to stop (all projects will be stopped automatically)."
echo ""

cd "$SCRIPT_DIR"
exec "$PYTHON" -m uvicorn app:app \
    --host 127.0.0.1 \
    --port 9000 \
    --log-level info
