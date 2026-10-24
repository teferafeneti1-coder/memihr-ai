"""
Render deploy entry point for Sign Language Translator.
Named server.py to avoid circular import with the original app.py.
"""
import os, sys

BACKEND = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..",
                 "egate", "signlanguagetranslate", "backend")
)
sys.path.insert(0, BACKEND)
os.chdir(BACKEND)

from app import app  # noqa: E402

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
