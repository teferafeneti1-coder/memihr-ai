"""
Render deploy wrapper for Fidel CNN.
Adds sys.path so imports resolve, then runs the original app.
PORT is provided by Render via environment variable.
"""
import os, sys

# Point to the original project
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "fidelaifinal")
sys.path.insert(0, os.path.abspath(PROJECT_ROOT))
os.chdir(os.path.abspath(PROJECT_ROOT))

# Import the original Flask app
from app import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=False)
