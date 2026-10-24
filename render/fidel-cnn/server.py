"""
Render deploy entry point for Fidel CNN.
Named server.py to avoid circular import with fidelaifinal/app.py.
"""
import os, sys, traceback

# Point to the original project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "fidelaifinal")
)
sys.path.insert(0, PROJECT_ROOT)
os.chdir(PROJECT_ROOT)

# Import the Flask app
from app import app  # noqa: E402

# Patch the predict route to log the real error
from flask import request, jsonify
import app as fidel_app

_orig_predict = fidel_app.predict

def _patched_predict(data_url, top_k=5):
    try:
        return _orig_predict(data_url, top_k)
    except Exception as e:
        print(f"[PREDICT ERROR] {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        raise

fidel_app.predict = _patched_predict

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
