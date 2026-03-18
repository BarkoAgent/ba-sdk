# ba_ws_sdk/file_system.py
"""
Reusable file management module for Barko agents.
Handles file storage, CRUD operations, and attachment management.
Any agent using the SDK automatically gets file management capabilities.
"""

import re
import os
import json
import logging
import base64
import asyncio
import threading
from datetime import datetime, timezone
from pathlib import Path

# ─── Configuration ───────────────────────────────────────────────────────────

_ATTACHMENTS_DIR: Path = Path(
    os.getenv("AGENT_ATTACHMENTS_DIR", "./attachments")
).resolve()

MAX_READ_BYTES = 1 * 1024 * 1024   # 1MB max per read
DEFAULT_READ_BYTES = 64 * 1024      # 64KB default

# Binary file extensions that cannot be read as text
_BINARY_EXTENSIONS = {
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".zip", ".tar", ".gz", ".rar", ".7z",
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".webp", ".ico", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".exe", ".dll", ".so", ".bin", ".dat",
}

_attachments_cache = None  # list of {name, size_bytes, modified_iso} or None

# ─── Download tracking ──────────────────────────────────────────────────────
# Shared state that any agent (Playwright, Selenium, etc.) can use.
# Agent-specific code calls on_download_started/complete/failed;
# the LLM calls wait_for_download.

_download_lock = threading.Lock()
# run_id -> list of download dicts:
#   {"file_name": str, "status": "pending"|"complete"|"failed",
#    "file_path": str|None, "error": str|None, "size_bytes": int|None,
#    "event": asyncio.Event|None}
_pending_downloads: dict = {}


# ─── Public helpers ──────────────────────────────────────────────────────────

def get_attachments_dir() -> Path:
    """Return the resolved attachments directory path."""
    return _ATTACHMENTS_DIR


def init(attachments_dir: str = None) -> None:
    """
    Optionally override the attachments directory at runtime.
    Call before any file operations if you need a custom path.
    """
    global _ATTACHMENTS_DIR, _attachments_cache
    if attachments_dir is not None:
        _ATTACHMENTS_DIR = Path(attachments_dir).resolve()
        _attachments_cache = None
        logging.info(f"[FileSystem] Attachments dir set to: {_ATTACHMENTS_DIR}")


def sanitize_filename(name: str) -> str:
    """
    Strip dangerous characters and path components from a filename.
    Returns a safe filename or raises ValueError if result is empty.
    """
    name = name.replace("..", "").replace("/", "").replace("\\", "")
    name = re.sub(r"[^a-zA-Z0-9\-_. ]", "", name)
    name = re.sub(r"\s+", " ", name).strip()
    if len(name) > 255:
        name = name[:255]
    name = name.lstrip(".")
    if not name:
        raise ValueError("Filename is empty after sanitization")
    return name


# ─── Internal helpers ────────────────────────────────────────────────────────

def _scan_attachments() -> list:
    """Scan attachments dir and return file metadata."""
    if not _ATTACHMENTS_DIR.is_dir():
        return []
    files = []
    for entry in sorted(_ATTACHMENTS_DIR.iterdir()):
        if entry.is_file() and not entry.name.startswith(".") and not entry.name.startswith("~"):
            stat = entry.stat()
            files.append({
                "name": entry.name,
                "size_bytes": stat.st_size,
                "modified_iso": datetime.fromtimestamp(
                    stat.st_mtime, tz=timezone.utc
                ).isoformat(),
            })
    return files


def _invalidate_cache():
    """Remove cached metadata so next list call rescans."""
    global _attachments_cache
    _attachments_cache = None


def _get_attachments_metadata() -> list:
    """Get cached metadata or scan disk."""
    global _attachments_cache
    if _attachments_cache is None:
        _attachments_cache = _scan_attachments()
    return _attachments_cache


def _migrate_attachments_flat():
    """One-time migration: move files from ./attachments/{subdir}/ to ./attachments/ flat."""
    if not _ATTACHMENTS_DIR.is_dir():
        return
    for entry in list(_ATTACHMENTS_DIR.iterdir()):
        if entry.is_dir() and not entry.name.startswith("."):
            for f in entry.iterdir():
                if f.is_file():
                    dest = _ATTACHMENTS_DIR / f.name
                    if not dest.exists():
                        f.rename(dest)
                        logging.info(f"[Migration] Moved {f} -> {dest}")
                    else:
                        logging.warning(
                            f"[Migration] Skipped {f} (already exists at {dest})"
                        )
            try:
                entry.rmdir()
            except OSError:
                pass


# ─── Save (called by ws_core binary envelope handler) ───────────────────────

def save_uploaded_file(file_name: str, file_bytes: bytes) -> str:
    """
    Save raw file bytes to attachments dir.
    Called by the WS binary envelope handler (not an agent function).
    """
    safe_name = sanitize_filename(file_name)
    _ATTACHMENTS_DIR.mkdir(parents=True, exist_ok=True)
    dest = _ATTACHMENTS_DIR / safe_name
    dest.write_bytes(file_bytes)
    _invalidate_cache()
    logging.info(f"[FileSystem] Saved {len(file_bytes)} bytes -> {dest}")
    return safe_name


# ─── Download lifecycle (called by agent code, NOT by the LLM) ──────────────

def on_download_started(run_id: str, file_name: str) -> dict:
    """
    Called by the agent when the browser fires a download event.
    Returns the tracking entry so the agent can reference it if needed.
    """
    entry = {
        "file_name": file_name,
        "status": "pending",
        "file_path": None,
        "error": None,
        "size_bytes": None,
        "event": None,  # set lazily by wait_for_download
    }
    with _download_lock:
        _pending_downloads.setdefault(run_id, []).append(entry)
    logging.info(f"[Download] Started: {file_name} for run_id={run_id}")
    return entry


def _signal_download_event(run_id: str, file_name: str) -> None:
    """Set the asyncio.Event for a download entry if a waiter attached one."""
    with _download_lock:
        for entry in _pending_downloads.get(run_id, []):
            if entry["file_name"] == file_name and entry["event"] is not None:
                entry["event"].set()
                return


def on_download_complete(run_id: str, file_name: str, file_path: str) -> None:
    """Called by the agent when a download finishes successfully."""
    with _download_lock:
        for entry in _pending_downloads.get(run_id, []):
            if entry["file_name"] == file_name and entry["status"] == "pending":
                entry["status"] = "complete"
                entry["file_path"] = file_path
                try:
                    entry["size_bytes"] = os.path.getsize(file_path)
                except OSError:
                    entry["size_bytes"] = None
                break
    _invalidate_cache()
    _signal_download_event(run_id, file_name)
    logging.info(f"[Download] Complete: {file_name} for run_id={run_id}")


def on_download_failed(run_id: str, file_name: str, error: str) -> None:
    """Called by the agent when a download fails."""
    with _download_lock:
        for entry in _pending_downloads.get(run_id, []):
            if entry["file_name"] == file_name and entry["status"] == "pending":
                entry["status"] = "failed"
                entry["error"] = error
                break
    _signal_download_event(run_id, file_name)
    logging.error(f"[Download] Failed: {file_name} for run_id={run_id}: {error}")


def clear_downloads(run_id: str) -> None:
    """Remove all download tracking entries for a run_id (call on stop_driver)."""
    with _download_lock:
        _pending_downloads.pop(run_id, None)


# ─── Agent-callable functions ────────────────────────────────────────────────
# These are auto-registered into FUNCTION_MAP by ws_core so any agent gets them.

async def list_agent_files(_run_test_id='1') -> str:
    """
    Returns a JSON array of files uploaded to the agent.
    Each entry has: name, size_bytes, modified_iso.
    """
    files = _get_attachments_metadata()
    return json.dumps(files)


async def delete_agent_file(file_name: str, _run_test_id='1') -> str:
    """
    Deletes an uploaded file by name.
    """
    try:
        safe_name = sanitize_filename(file_name)
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)})

    target = _ATTACHMENTS_DIR / safe_name
    if not target.is_file():
        return json.dumps({"status": "error", "error": "file not found"})

    target.unlink()
    _invalidate_cache()
    return json.dumps({"status": "success", "deleted": safe_name})


async def read_agent_file(
    file_name: str,
    offset: str = '0',
    length: str = '',
    as_text: str = 'true',
    _run_test_id='1',
) -> str:
    """
    Reads an uploaded file by name. Supports partial reads via offset/length.
    Defaults to first 64KB if length is not specified. Max single read is 1MB.

    Args:
        file_name: name of the file to read
        offset: byte offset to start reading from (default 0)
        length: number of bytes to read (default 64KB, max 1MB)
        as_text: 'true' to decode as UTF-8, 'false' to return base64
    """
    try:
        safe_name = sanitize_filename(file_name)
    except ValueError as e:
        return json.dumps({"status": "error", "error": str(e)})

    target = _ATTACHMENTS_DIR / safe_name
    if not target.is_file():
        return json.dumps({"status": "error", "error": "file not found"})

    # Check for binary file types that cannot be meaningfully read as text
    file_ext = Path(safe_name).suffix.lower()
    if file_ext in _BINARY_EXTENSIONS:
        total_size = target.stat().st_size
        return json.dumps({
            "status": "error",
            "error": (
                f"Reading '{safe_name}' is not supported. "
                f"The agent currently does not support reading {file_ext} files. "
                f"Binary files like this can still be uploaded to web forms using the 'upload_file_to_form' function."
            ),
            "file_name": safe_name,
            "file_extension": file_ext,
            "total_size": total_size,
        })

    total_size = target.stat().st_size
    try:
        byte_offset = int(offset)
    except (TypeError, ValueError):
        byte_offset = 0
    try:
        byte_length = int(length) if length else DEFAULT_READ_BYTES
    except (TypeError, ValueError):
        byte_length = DEFAULT_READ_BYTES
    byte_length = min(byte_length, MAX_READ_BYTES)
    byte_offset = max(0, min(byte_offset, total_size))

    with open(target, "rb") as f:
        f.seek(byte_offset)
        data = f.read(byte_length)

    truncated = (byte_offset + len(data)) < total_size

    if as_text.lower() == 'true':
        try:
            content = data.decode("utf-8")
        except UnicodeDecodeError:
            return json.dumps({
                "status": "error",
                "error": (
                    f"Reading '{safe_name}' is not supported. "
                    f"The file does not appear to be a valid text file. "
                    f"The agent currently does not support reading binary file contents. "
                    f"Binary files can still be uploaded to web forms using the 'upload_file_to_form' function."
                ),
                "file_name": safe_name,
                "total_size": total_size,
            })
    else:
        content = base64.b64encode(data).decode("ascii")

    return json.dumps({
        "content": content,
        "total_size": total_size,
        "offset": byte_offset,
        "length": len(data),
        "truncated": truncated,
    })


# ─── Lightweight change detection ────────────────────────────────────────────

async def get_file_fingerprint(_run_test_id='1') -> str:
    """
    Returns a lightweight fingerprint of the attachments directory.
    Used for cheap change detection without transferring full file metadata.
    Returns JSON: {"count": N, "total_bytes": M}
    """
    if not _ATTACHMENTS_DIR.is_dir():
        return json.dumps({"count": 0, "total_bytes": 0})
    count = 0
    total_bytes = 0
    for entry in os.scandir(str(_ATTACHMENTS_DIR)):
        if entry.is_file() and not entry.name.startswith(".") and not entry.name.startswith("~"):
            count += 1
            total_bytes += entry.stat().st_size
    return json.dumps({"count": count, "total_bytes": total_bytes})


# ─── Function registry (used by ws_core to inject into FUNCTION_MAP) ────────

def get_agent_functions() -> dict:
    """
    Return a dict of file management functions to merge into FUNCTION_MAP.
    These appear as regular agent functions callable via WebSocket.
    """
    return {
        "list_agent_files": list_agent_files,
        "delete_agent_file": delete_agent_file,
        "read_agent_file": read_agent_file,
        "get_file_fingerprint": get_file_fingerprint,
    }


# ─── Run migration on import ────────────────────────────────────────────────
_migrate_attachments_flat()
