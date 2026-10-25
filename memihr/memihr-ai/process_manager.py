"""
process_manager.py
Manages lifecycle of all project sub-processes for the egate dashboard.

Responsibilities:
  - Start / stop / restart individual projects
  - Wake-all (staggered, memory-guarded)
  - Sleep-all
  - Log capture (last 2000 lines, streamed via async generator)
  - Port-in-use detection
  - SIGTERM → wait 8 s → SIGKILL
  - Stop all children on shutdown
"""

import asyncio
import importlib.util
import json
import logging
import os
import signal
import socket
import subprocess
import time
from collections import deque
from pathlib import Path
from typing import AsyncIterator, Deque, Dict, Optional

import psutil

logger = logging.getLogger("process_manager")

# ── Constants ──────────────────────────────────────────────────────────────────
DASHBOARD_DIR  = Path(__file__).parent.resolve()
SHIM_PATH      = DASHBOARD_DIR / "launchers" / "run_shim.py"
PROJECTS_JSON  = DASHBOARD_DIR / "projects.json"
MAX_LOG_LINES  = 2000
STAGGER_TIMEOUT_HEAVY  = 300   # seconds
STAGGER_TIMEOUT_LIGHT  = 60
MEMORY_WAIT_SECONDS    = 90
MEMORY_HEADROOM_GB     = 1.0
STOP_WAIT_SECONDS      = 8

# Status values
ST_STOPPED  = "stopped"
ST_STARTING = "starting"
ST_RUNNING  = "running"
ST_CRASHED  = "crashed"
ST_SKIPPED  = "skipped"


# ── Helpers ────────────────────────────────────────────────────────────────────

def _load_projects() -> list[dict]:
    with open(PROJECTS_JSON, encoding="utf-8") as fh:
        return json.load(fh)


def _port_in_use(port: int) -> Optional[str]:
    """Return None if port is free, else a description of what is using it."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.5):
            pass
    except (ConnectionRefusedError, OSError):
        return None

    # Port is in use — try to identify the process
    try:
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == port and conn.pid:
                try:
                    p = psutil.Process(conn.pid)
                    return f"PID {conn.pid} ({p.name()})"
                except psutil.NoSuchProcess:
                    return f"PID {conn.pid}"
    except Exception:
        pass
    return "unknown process"


def _port_accepts(port: int) -> bool:
    """True if TCP connection to 127.0.0.1:<port> succeeds."""
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=1.0):
            return True
    except OSError:
        return False


def _available_memory_gb() -> float:
    return psutil.virtual_memory().available / (1024 ** 3)


def _check_packages(project: dict) -> list[str]:
    """
    Check that key packages are available in the project's Python environment.
    Uses importlib.util.find_spec (no heavy imports).
    Returns list of missing package names.
    """
    pkgs = ["flask", "torch", "transformers", "fastapi", "uvicorn", "sentencepiece"]
    python = project.get("python", "python3")
    missing = []
    for pkg in pkgs:
        result = subprocess.run(
            [python, "-c",
             f"import importlib.util, sys; "
             f"sys.exit(0 if importlib.util.find_spec('{pkg}') else 1)"],
            capture_output=True,
        )
        if result.returncode != 0:
            missing.append(pkg)
    return missing


# ── ProjectState ──────────────────────────────────────────────────────────────

class ProjectState:
    def __init__(self, project: dict):
        self.project   = project
        self.status    = ST_STOPPED
        self.proc: Optional[subprocess.Popen] = None
        self.logs: Deque[str] = deque(maxlen=MAX_LOG_LINES)
        self.last_log  = ""
        self.exit_code: Optional[int] = None
        self.skip_reason: Optional[str] = None
        # For streaming: subscribers waiting for new log lines
        self._log_queues: list[asyncio.Queue] = []
        self._log_reader_task: Optional[asyncio.Task] = None
        self._poll_task: Optional[asyncio.Task] = None

    @property
    def pid(self) -> Optional[int]:
        return self.proc.pid if self.proc else None

    def _push_log(self, line: str) -> None:
        self.logs.append(line)
        self.last_log = line
        for q in self._log_queues:
            try:
                q.put_nowait(line)
            except asyncio.QueueFull:
                pass

    async def _read_logs(self) -> None:
        """Background task: read stdout+stderr from the child process."""
        if not self.proc or not self.proc.stdout:
            return
        loop = asyncio.get_event_loop()
        try:
            while True:
                line = await loop.run_in_executor(None, self.proc.stdout.readline)
                if not line:
                    break
                decoded = line.decode("utf-8", errors="replace").rstrip("\n\r")
                self._push_log(decoded)
        except Exception as exc:
            self._push_log(f"[log reader error: {exc}]")

    async def _poll_exit(self) -> None:
        """Background task: wait for the process to exit, then update status."""
        if not self.proc:
            return
        loop = asyncio.get_event_loop()
        try:
            rc = await loop.run_in_executor(None, self.proc.wait)
            self.exit_code = rc
            if self.status not in (ST_STOPPED, ST_SKIPPED):
                self.status = ST_CRASHED
                self._push_log(f"[process exited with code {rc}]")
        except Exception as exc:
            self._push_log(f"[poll error: {exc}]")

    def start_background_tasks(self) -> None:
        loop = asyncio.get_event_loop()
        self._log_reader_task = loop.create_task(self._read_logs())
        self._poll_task       = loop.create_task(self._poll_exit())

    def get_log_stream(self) -> "LogStream":
        return LogStream(self)

    def cpu_percent(self) -> float:
        try:
            if self.proc and self.proc.pid:
                p = psutil.Process(self.proc.pid)
                return p.cpu_percent(interval=None)
        except Exception:
            pass
        return 0.0

    def memory_mb(self) -> float:
        try:
            if self.proc and self.proc.pid:
                p = psutil.Process(self.proc.pid)
                mem = p.memory_info().rss
                # include child processes
                for child in p.children(recursive=True):
                    try:
                        mem += child.memory_info().rss
                    except Exception:
                        pass
                return mem / (1024 * 1024)
        except Exception:
            pass
        return 0.0


class LogStream:
    """Async iterator that yields lines from a ProjectState's log buffer."""
    def __init__(self, state: ProjectState):
        self._state = state
        self._q: asyncio.Queue = asyncio.Queue(maxsize=500)
        state._log_queues.append(self._q)
        # Replay existing lines first
        self._buffer = list(state.logs)

    def __aiter__(self) -> "LogStream":
        return self

    async def __anext__(self) -> str:
        if self._buffer:
            return self._buffer.pop(0)
        while True:
            try:
                return await asyncio.wait_for(self._q.get(), timeout=15.0)
            except asyncio.TimeoutError:
                return ""   # heartbeat

    def close(self) -> None:
        try:
            self._state._log_queues.remove(self._q)
        except ValueError:
            pass


# ── ProcessManager ────────────────────────────────────────────────────────────

class ProcessManager:
    def __init__(self):
        self._states: Dict[str, ProjectState] = {}
        self._projects: list[dict] = []
        self._wake_task: Optional[asyncio.Task] = None
        self._wake_progress: str = ""

    def reload_projects(self) -> list[dict]:
        self._projects = _load_projects()
        # Ensure a state object exists for every project (keep existing ones)
        for p in self._projects:
            if p["id"] not in self._states:
                self._states[p["id"]] = ProjectState(p)
            else:
                # Update project metadata without disturbing runtime state
                self._states[p["id"]].project = p
        return self._projects

    def get_project(self, pid: str) -> Optional[dict]:
        for p in self._projects:
            if p["id"] == pid:
                return p
        return None

    def get_state(self, pid: str) -> Optional[ProjectState]:
        return self._states.get(pid)

    def project_info(self, pid: str) -> dict:
        """Return a serializable status dict for one project."""
        proj  = self.get_project(pid)
        state = self.get_state(pid)
        if not proj or not state:
            return {}

        # If we don't think the project is running but the port is alive,
        # reconcile — this happens when the dashboard restarts while projects
        # are already up, or when start() sets status before the port check
        # completes on a slow machine.
        if state.status != ST_RUNNING and state.status != ST_SKIPPED:
            if _port_accepts(proj["port"]):
                state.status = ST_RUNNING
                state._push_log("[dashboard] port alive — marking running")

        return {
            "id":          proj["id"],
            "name":        proj["name"],
            "description": proj["description"],
            "port":        proj["port"],
            "url_path":    proj["url_path"],
            "needs_camera":      proj.get("needs_camera", False),
            "unload_on_hide":    proj.get("unload_on_hide", False),
            "needs_confirmation":proj.get("needs_confirmation", False),
            "heavy_start": proj.get("heavy_start", False),
            "ram_gb":      proj.get("ram_gb", 0),
            "order":       proj.get("order", 99),
            "autostart":   proj.get("autostart", False),
            "status":      state.status,
            "pid":         state.pid,
            "exit_code":   state.exit_code,
            "last_log":    state.last_log,
            "skip_reason": state.skip_reason,
            "cpu_percent": state.cpu_percent(),
            "memory_mb":   state.memory_mb(),
        }

    def all_project_info(self) -> list[dict]:
        return [self.project_info(p["id"]) for p in self._projects]

    # ── Start ──────────────────────────────────────────────────────────────────

    async def start(self, pid: str) -> dict:
        """Start a project. Returns {"ok": bool, "message": str}."""
        proj  = self.get_project(pid)
        state = self.get_state(pid)
        if not proj or not state:
            return {"ok": False, "message": f"Unknown project: {pid}"}

        if state.status in (ST_STARTING, ST_RUNNING):
            return {"ok": False, "message": f"{proj['name']} is already {state.status}"}

        # Check root exists
        if not os.path.isdir(proj["root"]):
            msg = f"Project root not found: {proj['root']}"
            state.status = ST_CRASHED
            state._push_log(f"[ERROR] {msg}")
            return {"ok": False, "message": msg}

        # Check port in use
        owner = _port_in_use(proj["port"])
        if owner:
            msg = f"Port {proj['port']} already in use by {owner}"
            state.status = ST_CRASHED
            state._push_log(f"[ERROR] {msg}")
            return {"ok": False, "message": msg}

        # Build command
        python = proj["python"]
        entry  = proj["entry"]
        launch = proj.get("launch", "native")
        port   = proj["port"]

        if launch == "shim":
            cmd = [
                python,
                str(SHIM_PATH),
                entry,
                str(port),
                f"session_{pid}",
            ]
        else:  # native
            cmd = [python, entry] + proj.get("args", [])

        cwd = proj["cwd"]
        env = {
            **os.environ,
            **proj.get("env", {}),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
        }

        state.status     = ST_STARTING
        state.exit_code  = None
        state.skip_reason = None
        state._push_log(f"[dashboard] Starting: {' '.join(cmd)}")
        state._push_log(f"[dashboard] cwd={cwd}")

        try:
            proc = subprocess.Popen(
                cmd,
                cwd=cwd,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except Exception as exc:
            state.status = ST_CRASHED
            state._push_log(f"[ERROR] Failed to start: {exc}")
            return {"ok": False, "message": str(exc)}

        state.proc = proc
        state.start_background_tasks()

        # Wait for port to accept connections (up to timeout)
        timeout = STAGGER_TIMEOUT_HEAVY if proj.get("heavy_start") else STAGGER_TIMEOUT_LIGHT
        deadline = time.time() + timeout
        while time.time() < deadline:
            await asyncio.sleep(1)
            if proc.poll() is not None:
                state.status = ST_CRASHED
                state._push_log(f"[dashboard] Process exited early (code {proc.returncode})")
                return {"ok": False, "message": f"Process exited with code {proc.returncode}"}
            if _port_accepts(port):
                state.status = ST_RUNNING
                state._push_log(f"[dashboard] Running on port {port}")
                return {"ok": True, "message": f"Running on port {port}"}

        # Timed out waiting — still mark running if process is alive
        if proc.poll() is None:
            state.status = ST_RUNNING
            state._push_log(f"[dashboard] Start timeout reached but process is alive; marking running")
            return {"ok": True, "message": "Start timeout reached; process alive"}

        state.status = ST_CRASHED
        return {"ok": False, "message": "Start timed out and process exited"}

    # ── Stop ───────────────────────────────────────────────────────────────────

    async def stop(self, pid: str) -> dict:
        state = self.get_state(pid)
        proj  = self.get_project(pid)
        if not state:
            return {"ok": False, "message": f"Unknown project: {pid}"}

        if state.status in (ST_STOPPED, ST_SKIPPED) and state.proc is None:
            state.status = ST_STOPPED
            return {"ok": True, "message": "Already stopped"}

        if state.proc is None:
            state.status = ST_STOPPED
            return {"ok": True, "message": "No process to stop"}

        proc = state.proc
        state._push_log("[dashboard] Stopping…")
        name = proj["name"] if proj else pid

        def _kill_tree(sig: int) -> None:
            """Send sig to the whole process tree (parent + all descendants)."""
            try:
                parent = psutil.Process(proc.pid)
                children = parent.children(recursive=True)
                # Kill children first, then parent
                for child in children:
                    try:
                        child.send_signal(sig)
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                parent.send_signal(sig)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
            # Also try the process group in case psutil missed anything
            try:
                os.killpg(os.getpgid(proc.pid), sig)
            except (ProcessLookupError, OSError):
                pass

        _kill_tree(signal.SIGTERM)

        # Wait up to STOP_WAIT_SECONDS for clean exit
        loop = asyncio.get_event_loop()
        try:
            await asyncio.wait_for(
                loop.run_in_executor(None, proc.wait),
                timeout=STOP_WAIT_SECONDS,
            )
        except asyncio.TimeoutError:
            state._push_log("[dashboard] SIGTERM timeout, sending SIGKILL…")
            _kill_tree(signal.SIGKILL)
            try:
                await asyncio.wait_for(
                    loop.run_in_executor(None, proc.wait),
                    timeout=5.0,
                )
            except asyncio.TimeoutError:
                pass

        state.proc   = None
        state.status = ST_STOPPED
        state._push_log("[dashboard] Stopped")
        return {"ok": True, "message": f"{name} stopped"}

    async def restart(self, pid: str) -> dict:
        await self.stop(pid)
        return await self.start(pid)

    # ── Wake all ───────────────────────────────────────────────────────────────

    @property
    def wake_progress(self) -> str:
        return self._wake_progress

    async def wake_all(self) -> None:
        """Start all autostart projects in order, staggered."""
        projects_ordered = sorted(self._projects, key=lambda p: p.get("order", 99))
        total    = sum(1 for p in projects_ordered if p.get("autostart", False))
        started  = 0

        for proj in projects_ordered:
            if not proj.get("autostart", False):
                continue

            started += 1
            pid   = proj["id"]
            name  = proj["name"]
            state = self.get_state(pid)

            self._wake_progress = f"Waking {started} of {total}: {name}"

            # Skip if already running
            if state and state.status == ST_RUNNING:
                continue

            # Memory guard
            ram_needed = proj.get("ram_gb", 0) + MEMORY_HEADROOM_GB
            waited = 0
            while _available_memory_gb() < ram_needed and waited < MEMORY_WAIT_SECONDS:
                state._push_log(
                    f"[dashboard] Low memory — waiting for {ram_needed:.1f} GB free "
                    f"(have {_available_memory_gb():.1f} GB)…"
                )
                await asyncio.sleep(5)
                waited += 5

            if _available_memory_gb() < ram_needed:
                msg = (
                    f"Skipped: not enough memory "
                    f"(need {ram_needed:.1f} GB, have {_available_memory_gb():.1f} GB available)"
                )
                state.status      = ST_SKIPPED
                state.skip_reason = msg
                state._push_log(f"[dashboard] {msg}")
                continue

            # Start the project (this waits until running/crashed/timeout)
            self._wake_progress = f"Waking {started} of {total}: {name} (starting…)"
            result = await self.start(pid)
            if not result["ok"]:
                # A crash of one project never blocks the others — just continue
                pass

        self._wake_progress = f"Done waking {total} projects"

    async def sleep_all(self) -> None:
        """Stop all running projects in parallel."""
        targets = [
            proj["id"] for proj in self._projects
            if self.get_state(proj["id"]) and
               self.get_state(proj["id"]).status in (ST_RUNNING, ST_STARTING, ST_CRASHED)
        ]
        if targets:
            await asyncio.gather(*[self.stop(pid) for pid in targets])

    # ── Shutdown ───────────────────────────────────────────────────────────────

    async def shutdown(self) -> None:
        """Stop all children gracefully; called when dashboard exits."""
        await self.sleep_all()

    # ── Package check ──────────────────────────────────────────────────────────

    def check_packages(self, pid: str) -> list[str]:
        proj = self.get_project(pid)
        if not proj:
            return []
        return _check_packages(proj)

    # ── Log streaming ──────────────────────────────────────────────────────────

    def get_log_lines(self, pid: str) -> list[str]:
        state = self.get_state(pid)
        if not state:
            return []
        return list(state.logs)

    async def log_stream(self, pid: str) -> AsyncIterator[str]:
        state = self.get_state(pid)
        if not state:
            return
        stream = state.get_log_stream()
        try:
            async for line in stream:
                yield line
        finally:
            stream.close()

    # ── Settings ───────────────────────────────────────────────────────────────
    # Settings are stored in memory; persisted to data/settings.json

    _SETTINGS_FILE = DASHBOARD_DIR / "data" / "settings.json"
    _DEFAULT_SETTINGS = {
        "autostart_on_launch": True,
        "stagger": True,
    }

    def load_settings(self) -> dict:
        if self._SETTINGS_FILE.exists():
            try:
                with open(self._SETTINGS_FILE, encoding="utf-8") as fh:
                    s = json.load(fh)
                    return {**self._DEFAULT_SETTINGS, **s}
            except Exception:
                pass
        return dict(self._DEFAULT_SETTINGS)

    def save_settings(self, settings: dict) -> dict:
        merged = {**self._DEFAULT_SETTINGS, **settings}
        self._SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(self._SETTINGS_FILE, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2)
        return merged


# Singleton
manager = ProcessManager()
