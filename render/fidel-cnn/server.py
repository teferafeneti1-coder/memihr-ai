"""
Render deploy entry point for Fidel CNN.
Named server.py to avoid circular import with fidelaifinal/app.py.
"""
import os, sys

# Point to the original project root
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "fidelaifinal")
)
sys.path.insert(0, PROJECT_ROOT)

# Change working directory so relative paths (models/, corrections/) resolve
os.chdir(PROJECT_ROOT)

# Import the Flask app from the original project
from app import app  # noqa: E402

# Quick sanity check
import importlib.util
for pkg in ["flask", "torch", "PIL", "numpy"]:
    spec = importlib.util.find_spec(pkg)
    print(f"  {pkg}: {'OK' if spec else 'MISSING'}", flush=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
