"""
Startup version checker for BA agents.

On each agent start, checks:
  1. Whether the installed ba_sdk is behind the latest on GitHub → pip upgrade
  2. Whether the agent's own git repo is behind its remote → git pull

If anything was updated, restarts the process via os.execv so the new code
takes effect immediately.  An env-var guard (_BA_UPDATED=1) prevents
restart loops in case something goes wrong.
"""

import logging
import os
import subprocess
import sys
import urllib.request

_SDK_PYPROJECT_URL = (
    "https://raw.githubusercontent.com/BarkoAgent/ba-sdk/main/pyproject.toml"
)
_SDK_PACKAGE_NAME = "ba_sdk"
_UPDATED_ENV_VAR = "_BA_UPDATED"


# ─── helpers ─────────────────────────────────────────────────────────────────

def _parse_version(v: str) -> tuple:
    try:
        return tuple(int(x) for x in v.strip().split("."))
    except Exception:
        return (0,)


def _get_installed_sdk_version() -> str | None:
    try:
        from importlib.metadata import version as pkg_version, PackageNotFoundError
        return pkg_version(_SDK_PACKAGE_NAME)
    except Exception:
        return None


def _get_remote_sdk_version() -> str | None:
    try:
        req = urllib.request.Request(
            _SDK_PYPROJECT_URL,
            headers={"User-Agent": "ba-sdk-version-checker/1.0"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            content = resp.read().decode()
        for line in content.splitlines():
            line = line.strip()
            if line.startswith("version"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception as e:
        logging.warning(f"[VersionChecker] Could not fetch SDK version from GitHub: {e}")
    return None


def _get_agent_git_root() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            cwd=os.getcwd(),
            timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


# ─── checks ──────────────────────────────────────────────────────────────────

def _check_sdk() -> bool:
    """Return True if the SDK was upgraded (needs restart)."""
    installed = _get_installed_sdk_version()
    remote = _get_remote_sdk_version()

    if not installed:
        logging.warning("[VersionChecker] Could not determine installed SDK version.")
        return False
    if not remote:
        return False

    if _parse_version(remote) <= _parse_version(installed):
        logging.info(f"[VersionChecker] SDK up to date ({installed}).")
        return False

    answer = input(
        f"[VersionChecker] SDK update available: {installed} → {remote}. Update? [y/N]: "
    ).strip().lower()
    if answer != "y":
        logging.info("[VersionChecker] SDK update skipped.")
        return False

    try:
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install", "--quiet", "--upgrade",
                "git+https://github.com/BarkoAgent/ba-sdk.git",
            ],
            capture_output=True,
            text=True,
            timeout=180,
        )
        if result.returncode == 0:
            logging.info("[VersionChecker] SDK upgraded successfully.")
            return True
        logging.error(f"[VersionChecker] SDK upgrade failed:\n{result.stderr}")
    except Exception as e:
        logging.error(f"[VersionChecker] SDK upgrade error: {e}")
    return False


def _check_agent(agent_dir: str) -> bool:
    """Return True if the agent repo was updated (needs restart)."""
    try:
        fetch = subprocess.run(
            ["git", "fetch", "--quiet"],
            capture_output=True,
            text=True,
            cwd=agent_dir,
            timeout=30,
        )
        if fetch.returncode != 0:
            logging.warning(f"[VersionChecker] git fetch failed: {fetch.stderr.strip()}")
            return False

        status = subprocess.run(
            ["git", "rev-list", "HEAD..@{u}", "--count"],
            capture_output=True,
            text=True,
            cwd=agent_dir,
            timeout=10,
        )
        if status.returncode != 0:
            # No upstream tracking branch configured — skip silently
            return False

        count = status.stdout.strip()
        if count == "0":
            logging.info("[VersionChecker] Agent repo up to date.")
            return False

        answer = input(
            f"[VersionChecker] Agent is {count} commit(s) behind remote. Update? [y/N]: "
        ).strip().lower()
        if answer != "y":
            logging.info("[VersionChecker] Agent update skipped.")
            return False

        pull = subprocess.run(
            ["git", "pull", "--ff-only", "--quiet"],
            capture_output=True,
            text=True,
            cwd=agent_dir,
            timeout=60,
        )
        if pull.returncode == 0:
            logging.info("[VersionChecker] Agent repo updated successfully.")
            return True
        logging.error(f"[VersionChecker] git pull failed:\n{pull.stderr.strip()}")
    except Exception as e:
        logging.error(f"[VersionChecker] Agent update error: {e}")
    return False


# ─── public entry point ───────────────────────────────────────────────────────

def run_version_checks() -> None:
    """
    Run SDK and agent version checks at startup.

    Call once before the WebSocket event loop begins.  If any update is
    applied the process restarts automatically via os.execv.
    """
    # Guard: skip if we already restarted once this session to prevent loops.
    if os.environ.get(_UPDATED_ENV_VAR) == "1":
        logging.info("[VersionChecker] Post-update restart detected — skipping checks.")
        return

    sdk_updated = _check_sdk()

    agent_dir = _get_agent_git_root()
    agent_updated = _check_agent(agent_dir) if agent_dir else False

    if sdk_updated or agent_updated:
        logging.info("[VersionChecker] Restarting agent to apply updates...")
        os.environ[_UPDATED_ENV_VAR] = "1"
        os.execv(sys.executable, [sys.executable] + sys.argv)
