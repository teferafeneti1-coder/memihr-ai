# memihr ai

One page that runs and displays all five AI projects together.

---

## How to run

```bash
cd /home/feneti/egate/egate/egate/memihr-ai
./run.sh
```

Then open **http://127.0.0.1:9000** in your browser.

To start the dashboard without waking projects automatically:

```bash
./run.sh --no-autostart
```

---

## How autostart works

When `autostart_on_launch` is `true` (the default), the dashboard calls **Wake all** automatically
as soon as it starts. Every project whose `autostart` field is `true` in `projects.json` will be
started in order.

You can toggle this with `--no-autostart` on the command line, or by editing `data/settings.json`.

---

## How stagger works

Projects do not all start simultaneously. The dashboard starts them one at a time in `order` sequence:

1. Start project N.
2. Wait until its port accepts connections (`running`), or it crashes, or the timeout is reached
   (300 s for `heavy_start` projects, 60 s for others).
3. Then start project N+1.

A crash of one project never blocks the rest.

---

## Memory guard

Before starting each project the dashboard reads `psutil.virtual_memory().available`. If available
RAM is less than the project's `ram_gb` estimate plus 1 GB headroom, it waits up to 90 seconds for
memory to free up. If memory is still too low after waiting, the project is marked
**skipped: low memory** with a clear message in the UI, and a **Start anyway** button is shown.
The dashboard never kills other processes.

---

## Launch modes

### `native`
The project is run exactly as `python app.py` from its `cwd`. Used for Sign Language Translator
(port 8000, unique port, no Flask clash).

### `shim`
The project is run through `dashboard/launchers/run_shim.py`, which:
- Overrides Flask's `host` → `127.0.0.1`, `port` → the assigned port, `debug` → `False`,
  `use_reloader` → `False`.
- Sets `SESSION_COOKIE_NAME` to `session_<project-id>` so Flask session cookies from different
  apps never collide on the same browser.
- Does **not** touch any project file — everything happens in memory at import time via
  monkey-patching.

Benefits: no double model load (Adulis gets single load instead of two), no exposed Werkzeug
debugger (Fidel), no cookie collisions between Flask apps.

---

## How to add a project

1. Edit `dashboard/projects.json` — add a new JSON object following the same schema.
   - `id`: unique slug (lowercase, hyphens).
   - `root` and `cwd` must be under `/home/feneti/`.
   - `launch`: `"shim"` for Flask apps, `"native"` for anything else.
   - `port`: pick a free port.
2. Save the file. The dashboard picks up the change on the next API call (or restart).
3. Click **Start** on the new project card.

---

## Limits and known constraints

- **RAM**: running Adulis and the Amharic-English assistant together needs ~9 GB RAM. On this
  machine the memory guard may skip the second heavy project. Use **Start anyway** to proceed.
- **iframes**: some apps set `X-Frame-Options: SAMEORIGIN` or a restrictive
  `Content-Security-Policy`, which prevents them from loading in an iframe. If the App tab shows
  a blank frame, use **Open in new tab**.
- **Camera app**: the Sign Language Translator iframe is unloaded (`src → about:blank`) when you
  switch away from it, releasing the webcam and CPU. It reloads when you select it again.
- **Cookie domain**: all shimmed Flask apps are served on `127.0.0.1` with unique cookie names,
  so sessions are isolated even though they share the same origin.
- **Amharic-English model path**: after the project moved to
  `/home/feneti/egate/egate/Amharic-English-finetuned/`, the app may fail if its `archive/` path
  was hard-coded to the old location `/home/feneti/Amharic-English-finetuned/archive`. Check the
  Logs tab on startup — if a `FileNotFoundError` appears for that path, the fix must be made
  inside the project's own `app.py` (outside this dashboard's scope). See `PROJECTS_REPORT.md`
  for details.
- **Adulis MODEL_PATH**: the `MODEL_PATH` env var is set in `projects.json` to point the model
  at its new location after the move. No project file was edited.
- **Projects are never modified**: the shim works purely at the Python runtime level. No project
  file is edited, copied, moved or deleted.

---

## File layout

```
memihr-ai/
  app.py                  FastAPI backend (port 9000)
  process_manager.py      Process lifecycle, log capture, wake-all logic
  projects.json           Project registry (human-editable)
  requirements.txt        memihr ai Python dependencies
  run.sh                  Entry point
  PROJECTS_REPORT.md      Project summary and partial-copy report
  README.md               This file
  launchers/
    run_shim.py           Flask/uvicorn port+debug override shim
  scripts/
    manifest.py           Snapshot and verify project files
  static/
    index.html            Single-page memihr ai UI
    style.css             Styles
    js/
      api.js              REST + SSE client
      ui.js               DOM helpers
      app.js              Main application logic
  data/
    manifest_snapshot.json  File snapshot (created by manifest.py snapshot)
    settings.json           Persisted settings (created on first run)
```
