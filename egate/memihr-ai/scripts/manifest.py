#!/usr/bin/env python3
"""
manifest.py  –  snapshot and verify project files.

Usage:
    python manifest.py snapshot   # record relative path + size + mtime for each project
    python manifest.py verify     # compare current state to snapshot; report differences
"""

import json
import os
import sys
import time

MANIFEST_FILE = os.path.join(os.path.dirname(__file__), "../data/manifest_snapshot.json")

# Projects as defined in the registry (root paths only; no inspection of project internals)
PROJECTS = [
    {"id": "fidelaifinal",              "root": "/home/feneti/egate/egate/fidelaifinal"},
    {"id": "signlanguagetranslate",     "root": "/home/feneti/egate/egate/egate/signlanguagetranslate"},
    {"id": "english-amharic-scratch",   "root": "/home/feneti/egate/egate/English-Amharic-Scratch"},
    {"id": "adulis",                    "root": "/home/feneti/egate/egate/oromifa-amharic & amharic-oromifa/adulis"},
    {"id": "amharic-english-finetuned", "root": "/home/feneti/egate/egate/Amharic-English-finetuned"},
]

MAX_FILES_PER_PROJECT = 100_000
MAX_FILE_SIZE_BYTES   = 500 * 1024  # skip files over 500 KB

# Folder names to skip entirely
SKIP_DIRS = {
    "__pycache__", ".git", "node_modules",
    "archive", "model", "models", "data", "corrections", "checkpoints",
}

# File patterns to skip
SKIP_SUFFIXES = (".pyc", ".pyo")


def should_skip_dir(name: str) -> bool:
    if name in SKIP_DIRS:
        return True
    # skip virtual-environment directories (contain pyvenv.cfg)
    return False


def walk_project(root: str) -> dict:
    """Return {relative_path: {"size": int, "mtime": float}} for every eligible file."""
    files: dict = {}
    count = 0
    warned = False

    for dirpath, dirnames, filenames in os.walk(root):
        # Prune skipped directories in-place so os.walk doesn't descend into them
        dirnames[:] = [
            d for d in dirnames
            if d not in SKIP_DIRS and not os.path.isfile(os.path.join(dirpath, d, "pyvenv.cfg"))
        ]

        for fname in filenames:
            if fname.endswith(SKIP_SUFFIXES):
                continue

            full = os.path.join(dirpath, fname)
            rel  = os.path.relpath(full, root)

            try:
                st = os.stat(full)
            except OSError:
                continue

            if st.st_size > MAX_FILE_SIZE_BYTES:
                continue  # skip large files (model weights, datasets, images)

            files[rel] = {"size": st.st_size, "mtime": round(st.st_mtime, 2)}
            count += 1

            if count >= MAX_FILES_PER_PROJECT:
                if not warned:
                    print(f"  WARNING: reached {MAX_FILES_PER_PROJECT} file limit for {root}", file=sys.stderr)
                    warned = True
                break  # stop adding files for this project

    return files


def snapshot() -> None:
    manifest: dict = {}
    for proj in PROJECTS:
        pid  = proj["id"]
        root = proj["root"]
        if not os.path.isdir(root):
            print(f"  SKIP {pid}: directory not found: {root}")
            manifest[pid] = {"root": root, "missing": True, "files": {}}
            continue
        print(f"  Snapshotting {pid} …", end=" ", flush=True)
        t0 = time.time()
        files = walk_project(root)
        elapsed = time.time() - t0
        print(f"{len(files)} files  ({elapsed:.1f}s)")
        manifest[pid] = {"root": root, "missing": False, "files": files}

    os.makedirs(os.path.dirname(MANIFEST_FILE), exist_ok=True)
    with open(MANIFEST_FILE, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nSnapshot saved → {MANIFEST_FILE}")


def verify() -> None:
    if not os.path.isfile(MANIFEST_FILE):
        print("No snapshot found. Run `manifest.py snapshot` first.", file=sys.stderr)
        sys.exit(1)

    with open(MANIFEST_FILE, encoding="utf-8") as fh:
        old: dict = json.load(fh)

    any_diff = False
    for proj in PROJECTS:
        pid  = proj["id"]
        root = proj["root"]
        old_entry = old.get(pid, {})
        old_files: dict = old_entry.get("files", {})

        if not os.path.isdir(root):
            print(f"[{pid}] DIRECTORY MISSING: {root}")
            any_diff = True
            continue

        print(f"  Verifying {pid} …", end=" ", flush=True)
        new_files = walk_project(root)
        print(f"{len(new_files)} files")

        added   = sorted(set(new_files) - set(old_files))
        removed = sorted(set(old_files) - set(new_files))
        changed = sorted(
            p for p in (set(new_files) & set(old_files))
            if new_files[p]["size"] != old_files[p]["size"]
            or abs(new_files[p]["mtime"] - old_files[p]["mtime"]) > 1.0
        )

        if not (added or removed or changed):
            print(f"    OK – no differences")
        else:
            any_diff = True
            for p in added:
                print(f"    ADDED    {p}")
            for p in removed:
                print(f"    REMOVED  {p}")
            for p in changed:
                o, n = old_files[p], new_files[p]
                print(f"    CHANGED  {p}  (size {o['size']}→{n['size']}, mtime {o['mtime']}→{n['mtime']})")

    if not any_diff:
        print("\nVerify complete: no differences found.")
    else:
        print("\nVerify complete: differences listed above.")
        print("NOTE: differences inside runtime-output folders (corrections/, models/, data/) are expected and should not be reverted.")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in ("snapshot", "verify"):
        print("Usage: manifest.py snapshot | verify", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]
    print(f"=== manifest.py {cmd} ===")
    if cmd == "snapshot":
        snapshot()
    else:
        verify()


if __name__ == "__main__":
    main()
