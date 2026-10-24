"""
Render deploy wrapper for Sign Language Translator.
Adjusts paths so the FastAPI app finds frontend and model files,
then starts uvicorn on Render's assigned PORT.
"""
import os, sys

# Point to the original backend
BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", 
                       "egate", "signlanguagetranslate", "backend")
sys.path.insert(0, os.path.abspath(BACKEND))
os.chdir(os.path.abspath(BACKEND))

# Import the original FastAPI app
from app import app

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
