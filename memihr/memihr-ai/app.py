"""
app.py – FastAPI backend for the egate dashboard.
Listens on 127.0.0.1:9000.

Routes:
  GET  /api/projects
  GET  /api/projects/{id}
  POST /api/projects/{id}/start
  POST /api/projects/{id}/stop
  POST /api/projects/{id}/restart
  PUT  /api/projects/{id}
  GET  /api/projects/{id}/logs   (Server-Sent Events)
  POST /api/wake-all
  POST /api/sleep-all
  GET  /api/settings
  PUT  /api/settings
  GET  /api/memory
  GET  /api/wake-progress

Static files served from ./static/ at /
"""

import asyncio
import importlib.util
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from process_manager import (
    PROJECTS_JSON,
    ST_RUNNING,
    ST_SKIPPED,
    manager,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger("dashboard")

DASHBOARD_DIR = Path(__file__).parent.resolve()


# ── Lifespan ───────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load projects and settings on startup
    manager.reload_projects()
    settings = manager.load_settings()
    logger.info("Dashboard started.  Projects loaded: %d", len(manager._projects))

    # Check that dashboard-only packages are present
    for pkg in ["fastapi", "uvicorn", "psutil"]:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            logger.error("Missing dashboard package: %s", pkg)

    # Auto-start all autostart projects if setting is on
    if settings.get("autostart_on_launch", True):
        logger.info("autostart_on_launch=true: waking all projects…")
        asyncio.get_event_loop().create_task(manager.wake_all())

    yield

    # Shutdown: stop all children
    logger.info("Dashboard shutting down — stopping all projects…")
    await manager.shutdown()


app = FastAPI(title="memihr ai", lifespan=lifespan)


# ── API routes ─────────────────────────────────────────────────────────────────

@app.get("/api/projects")
async def get_projects():
    return manager.all_project_info()


@app.get("/api/projects/{pid}")
async def get_project(pid: str):
    info = manager.project_info(pid)
    if not info:
        raise HTTPException(404, f"Project not found: {pid}")
    return info


@app.post("/api/projects/{pid}/start")
async def start_project(pid: str):
    if not manager.get_project(pid):
        raise HTTPException(404, f"Project not found: {pid}")
    result = await manager.start(pid)
    return result


@app.post("/api/projects/{pid}/stop")
async def stop_project(pid: str):
    if not manager.get_project(pid):
        raise HTTPException(404, f"Project not found: {pid}")
    result = await manager.stop(pid)
    return result


@app.post("/api/projects/{pid}/restart")
async def restart_project(pid: str):
    if not manager.get_project(pid):
        raise HTTPException(404, f"Project not found: {pid}")
    result = await manager.restart(pid)
    return result


@app.put("/api/projects/{pid}")
async def update_project(pid: str, request: Request):
    """
    Update a project's editable fields in projects.json.
    Validates that root is under /home/feneti/ and cwd is inside root.
    """
    body = await request.json()
    projects = manager._projects

    proj = None
    for p in projects:
        if p["id"] == pid:
            proj = p
            break
    if proj is None:
        raise HTTPException(404, f"Project not found: {pid}")

    # Editable fields (never allow changing id or dangerous fields)
    editable = [
        "name", "description", "port", "entry", "args", "autostart",
        "ram_gb", "heavy_start", "unload_on_hide", "needs_confirmation",
        "env", "url_path", "order",
    ]
    for key in editable:
        if key in body:
            proj[key] = body[key]

    # If root/cwd provided, validate them
    new_root = body.get("root", proj.get("root", ""))
    new_cwd  = body.get("cwd", proj.get("cwd", ""))

    if new_root:
        if not new_root.startswith("/home/feneti/"):
            raise HTTPException(400, "root must be under /home/feneti/")
        if not os.path.isdir(new_root):
            raise HTTPException(400, f"root directory not found: {new_root}")
        proj["root"] = new_root
    if new_cwd:
        # Normalise before comparison so paths with spaces don't cause false mismatches
        if not os.path.abspath(new_cwd).startswith(os.path.abspath(proj["root"])):
            raise HTTPException(400, "cwd must be inside root")
        proj["cwd"] = new_cwd

    # Write back to projects.json
    with open(PROJECTS_JSON, "w", encoding="utf-8") as fh:
        json.dump(projects, fh, indent=2)

    manager.reload_projects()
    return manager.project_info(pid)


@app.get("/api/projects/{pid}/logs")
async def stream_logs(pid: str, request: Request):
    """Server-Sent Events: stream log lines for a project."""
    if not manager.get_project(pid):
        raise HTTPException(404, f"Project not found: {pid}")

    async def event_generator():
        async for line in manager.log_stream(pid):
            if await request.is_disconnected():
                break
            # Empty string = heartbeat (keep-alive)
            if line == "":
                yield "data: \n\n"
            else:
                # Escape: replace newlines so SSE stays well-formed
                safe = line.replace("\n", " ").replace("\r", "")
                yield f"data: {safe}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/wake-all")
async def wake_all():
    asyncio.get_event_loop().create_task(manager.wake_all())
    return {"ok": True, "message": "Waking all projects…"}


@app.post("/api/sleep-all")
async def sleep_all():
    # Await directly so the response confirms all projects are actually stopped.
    await manager.sleep_all()
    return {"ok": True, "message": "All projects stopped"}


@app.get("/api/wake-progress")
async def wake_progress():
    return {"progress": manager.wake_progress}


@app.get("/api/settings")
async def get_settings():
    return manager.load_settings()


@app.put("/api/settings")
async def put_settings(request: Request):
    body = await request.json()
    result = manager.save_settings(body)
    return result


@app.get("/api/memory")
async def get_memory():
    vm = psutil.virtual_memory()
    return {
        "total_gb":     round(vm.total     / 1024**3, 2),
        "available_gb": round(vm.available / 1024**3, 2),
        "used_gb":      round(vm.used      / 1024**3, 2),
        "percent":      vm.percent,
    }


@app.get("/api/packages/{pid}")
async def check_packages(pid: str):
    missing = manager.check_packages(pid)
    return {"pid": pid, "missing": missing}


# ── Static files ───────────────────────────────────────────────────────────────
# Serve the frontend SPA from ./static/
app.mount("/", StaticFiles(directory=str(DASHBOARD_DIR / "static"), html=True), name="static")


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=9000,
        reload=False,
        log_level="info",
    )
