# dashboard/launchers/run_shim.py
# Overrides Flask/uvicorn host, port, debug and reloader at runtime — no project file is touched.
# Usage (run with the project's Python, cwd = the project's working directory):
#   python run_shim.py <entry.py> <port> <cookie_name>

import os
import sys
import runpy

entry  = os.path.abspath(sys.argv[1])
port   = int(sys.argv[2])
cookie = sys.argv[3]

# Make imports behave exactly as if we ran `python entry.py` from the project root
sys.path[0] = os.path.dirname(entry)
sys.argv    = [entry]

os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["GRADIO_SERVER_PORT"]      = str(port)
os.environ["GRADIO_SERVER_NAME"]      = "127.0.0.1"

# ── Flask shim ──────────────────────────────────────────────────────────────
try:
    import flask
    _orig_flask_run = flask.Flask.run

    def _flask_run(self, *args, **kwargs):
        # Set a per-project session cookie name so cookies never collide
        self.config["SESSION_COOKIE_NAME"] = cookie
        # Drop the keys we're overriding; pass everything else through
        rest = {k: v for k, v in kwargs.items()
                if k not in ("host", "port", "debug", "load_dotenv", "use_reloader")}
        return _orig_flask_run(
            self,
            host="127.0.0.1",
            port=port,
            debug=False,
            load_dotenv=False,
            **rest,
        )

    flask.Flask.run = _flask_run
except ImportError:
    pass

# ── uvicorn shim ────────────────────────────────────────────────────────────
try:
    import uvicorn
    _orig_uvicorn_run = uvicorn.run

    def _uvicorn_run(app, **kwargs):
        kwargs.update(host="127.0.0.1", port=port, reload=False)
        kwargs.pop("workers", None)
        return _orig_uvicorn_run(app, **kwargs)

    uvicorn.run = _uvicorn_run
except ImportError:
    pass

# ── Run the project entry file as __main__ ──────────────────────────────────
runpy.run_path(entry, run_name="__main__")
