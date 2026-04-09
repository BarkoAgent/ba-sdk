# baweb_ws_sdk/ws_core.py

import asyncio
import inspect
import json
import logging
import re
import time
import websockets
import os
import struct
import hashlib
from . import streaming
from typing import Any, Optional

# ─── Step-output variable resolution ─────────────────────────────────────────
# Syntax: {{step_output:N:REGEX}}
# N is the 0-based index of the producer step inside the current bulk command
# list; REGEX is applied to that step's raw output string. The first capture
# group is used, or the full match if there are no groups.
_STEP_OUTPUT_VAR_RE = re.compile(r'\{\{step_output:(\d+):(.+?)\}\}')


def _resolve_step_output_var(value: str, step_outputs: list) -> str:
    """Replace all {{step_output:N:REGEX}} tokens in *value* using *step_outputs*."""
    if not isinstance(value, str) or "{{step_output:" not in value:
        return value

    def _replace(m):
        idx = int(m.group(1))
        pattern = m.group(2)
        if idx >= len(step_outputs):
            return m.group(0)  # not yet available — keep placeholder
        try:
            match = re.search(pattern, step_outputs[idx], re.DOTALL)
            if match:
                return match.group(1) if match.lastindex else match.group(0)
        except re.error:
            pass
        return m.group(0)

    return _STEP_OUTPUT_VAR_RE.sub(_replace, value)


def _resolve_step_output_vars_in_command(args: list, kwargs: dict, step_outputs: list):
    """Return (resolved_args, resolved_kwargs) with all {{step_output:…}} replaced."""
    resolved_args = [_resolve_step_output_var(a, step_outputs) if isinstance(a, str) else a for a in args]
    resolved_kwargs = {k: (_resolve_step_output_var(v, step_outputs) if isinstance(v, str) else v) for k, v in kwargs.items()}
    return resolved_args, resolved_kwargs


def _output_to_str(value) -> str:
    """Coerce an agent return value to a plain string for step-output tracking."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value)

try:
    from . import file_system
except ImportError:
    file_system = None          # SDK deployed without file_system.py — degrade gracefully
    logging.warning("[SDK] file_system module not found — file management features disabled")


_AGENT_MODULE = None



def _is_sensitive_field(value: str) -> bool:
    lower = value.lower()
    return any(token in lower for token in ("password", "passwd", "pwd", "token", "secret", "api_key", "apikey"))


def _preview_value(value, max_len: int = 60) -> str:
    text = str(value)
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def _format_locator(kwargs: dict) -> str:
    locator_type = kwargs.get("locator_type")
    locator = kwargs.get("locator")
    if locator_type and locator:
        return f"{locator_type}:{locator}"
    if locator:
        return str(locator)
    return ""


def _extract_element_hint(args: list, kwargs: dict, raw_result: Any) -> Optional[dict]:
    """
    Build a function-agnostic hint that can be used by streaming to locate and
    highlight an element in the screenshot.
    """
    hint = {}
    known_types = {
        "id", "name", "xpath", "css", "css selector", "css_selector",
        "class name", "class_name", "tag name", "tag_name",
        "link text", "partial link text", "partial_link_text",
        "text", "role",
    }

    def _norm_type(value):
        if not isinstance(value, str):
            return None
        t = value.strip().lower()
        return t if t in known_types else None

    def _looks_like_locator(value: str) -> bool:
        text = value.strip()
        if not text or "://" in text:
            return False
        if text.startswith(("xpath=", "css=", "text=", "role=", "//", "(//", "#", ".", "[")):
            return True
        if text.startswith(("/", "./")):
            return True
        if " " in text and not text.startswith(("//", "(//", "text=")):
            return False
        return True

    def _pick_from_mapping(mapping: dict):
        locator_type = _norm_type(mapping.get("locator_type") or mapping.get("by") or mapping.get("type"))
        locator = (
            mapping.get("locator")
            or mapping.get("selector")
            or mapping.get("value")
            or mapping.get("query")
            or mapping.get("xpath")
            or mapping.get("css_selector")
            or mapping.get("css")
        )
        if locator is None and mapping.get("id") is not None:
            locator_type = locator_type or "id"
            locator = mapping.get("id")
        if locator is None and mapping.get("name") is not None:
            locator_type = locator_type or "name"
            locator = mapping.get("name")
        if locator is None and mapping.get("text") is not None:
            locator_type = locator_type or "text"
            locator = mapping.get("text")
        if isinstance(locator, str) and locator.strip():
            return locator_type, locator.strip()
        return locator_type, None

    locator_type, locator = _pick_from_mapping(kwargs)

    nested_locator = kwargs.get("locator")
    if not locator and isinstance(nested_locator, dict):
        nested_type, nested_value = _pick_from_mapping(nested_locator)
        locator_type = locator_type or nested_type
        locator = locator or nested_value

    if not locator:
        for arg in args:
            if isinstance(arg, dict):
                nested_type, nested_value = _pick_from_mapping(arg)
                locator_type = locator_type or nested_type
                locator = locator or nested_value
                if locator:
                    break

    # Common positional pattern: (locator_type, locator, ...)
    if not locator and len(args) >= 2:
        if isinstance(args[0], str) and isinstance(args[1], str):
            norm_type = _norm_type(args[0])
            if norm_type:
                locator_type = locator_type or norm_type
                locator = args[1].strip()

    if not locator:
        for arg in args:
            if isinstance(arg, str) and _looks_like_locator(arg):
                locator = arg.strip()
                break

    if locator_type and locator:
        hint["locator_type"] = locator_type
        hint["locator"] = locator
    elif locator:
        hint["locator"] = locator

    # If an action already returns a bbox-like payload, forward it directly.
    if isinstance(raw_result, dict):
        box = raw_result.get("bounding_box")
        if isinstance(box, dict):
            hint["bounding_box"] = box
        elif all(k in raw_result for k in ("x", "y", "width", "height")):
            hint["bounding_box"] = {
                "x": raw_result.get("x"),
                "y": raw_result.get("y"),
                "width": raw_result.get("width"),
                "height": raw_result.get("height"),
            }

    return hint or None


def _format_step_output(func_name: str, args: list, kwargs: dict, raw_result) -> str:
    locator_repr = _format_locator(kwargs)

    if func_name == "create_driver":
        return "create_driver"
    if func_name == "stop_driver":
        return "stop_driver"
    if func_name == "navigate_to_url":
        url = kwargs.get("url")
        if url is None and args:
            url = args[0]
        return f"navigate_to_url url={_preview_value(url)}" if url is not None else "navigate_to_url"
    if func_name == "click":
        return f"click locator={locator_repr}" if locator_repr else "click"
    if func_name == "scroll_to_element":
        return f"scroll_to_element locator={locator_repr}" if locator_repr else "scroll_to_element"
    if func_name == "send_keys":
        value = kwargs.get("value")
        if value is None and len(args) >= 3:
            value = args[2]
        sensitive_hint = " ".join(str(x) for x in (kwargs.get("locator"), kwargs.get("name"), kwargs.get("key")) if x)
        if _is_sensitive_field(sensitive_hint):
            value_repr = "<redacted>"
        else:
            value_repr = _preview_value(value) if value is not None else ""
        if locator_repr and value_repr:
            return f"send_keys locator={locator_repr} value={value_repr}"
        if locator_repr:
            return f"send_keys locator={locator_repr}"
        return "send_keys"
    if func_name == "get_page_html":
        text = raw_result if isinstance(raw_result, str) else str(raw_result)
        return f"get_page_html captured chars={len(text)}"
    if func_name == "exists":
        return f"exists locator={locator_repr}" if locator_repr else "exists"
    if func_name == "does_not_exist":
        return f"does_not_exist locator={locator_repr}" if locator_repr else "does_not_exist"
    if func_name == "exists_with_text":
        text = kwargs.get("text")
        if text is None and args:
            text = args[0]
        return f"exists_with_text text={_preview_value(text)}" if text is not None else "exists_with_text"
    if func_name == "wait_for_download":
        try:
            parsed = json.loads(raw_result) if isinstance(raw_result, str) else raw_result
            if isinstance(parsed, dict) and parsed.get("status") == "success":
                return f"wait_for_download file={parsed.get('file_name')} size={parsed.get('size_bytes')}"
        except (json.JSONDecodeError, TypeError):
            pass
        return f"wait_for_download result={_preview_value(raw_result)}"

    raw_text = raw_result if isinstance(raw_result, str) else str(raw_result)
    if raw_text and raw_text.lower() not in ("success", "ok"):
        return f"{func_name} result={_preview_value(raw_text)}"
    return func_name

# You will need to pass these in from your main app:
# import agent_func
# import streaming

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s'
    )

def get_semaphore():
    CONCURRENCY_LIMIT = int(os.getenv("CONCURRENCY_LIMIT", "4"))
    return asyncio.Semaphore(CONCURRENCY_LIMIT)

def build_function_map(agent_func):
    func_map = {
        name: obj
        for name, obj in inspect.getmembers(agent_func, inspect.isfunction)
        if not name.startswith("_")
    }
    # Merge SDK-provided file management functions (agent can override if needed)
    if file_system is not None:
        fs_funcs = file_system.get_agent_functions()
        for name, func in fs_funcs.items():
            if name not in func_map:
                func_map[name] = func
    return func_map

def build_system_functions():
    return {
        "_start_frame_recording": streaming._start_frame_recording,
        "_stop_frame_recording": streaming._stop_frame_recording,
        "_get_recorded_frames": streaming._get_recorded_frames,
        "_ack_recorded_frames": streaming._ack_recorded_frames,
        "_clear_frame_recording": streaming._clear_frame_recording,
    }

def _make_envelope(header: dict, payload_bytes: bytes) -> bytes:
    header_json = json.dumps(header, separators=(",", ":" )).encode("utf-8")
    header_len = len(header_json)
    return struct.pack(">I", header_len) + header_json + payload_bytes


def _parse_envelope(data: bytes) -> tuple:
    """Parse binary envelope: [4B header_len][header JSON][payload]."""
    if len(data) < 4:
        raise ValueError("Envelope too small")
    header_len = struct.unpack(">I", data[:4])[0]
    if len(data) < 4 + header_len:
        raise ValueError("Incomplete envelope header")
    header = json.loads(data[4:4 + header_len].decode("utf-8"))
    payload = data[4 + header_len:]
    return header, payload


# ─── Chunked file upload state ──────────────────────────────────────────────
_pending_uploads: dict = {}  # msg_id -> {file_name, total_size, temp_path, hasher, bytes_received, file_handle}
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))  # 100MB default


def _cleanup_pending_uploads():
    """Close file handles and remove temp files for all pending uploads."""
    for msg_id, upload in list(_pending_uploads.items()):
        try:
            upload["file_handle"].close()
        except Exception:
            pass
        try:
            os.unlink(upload["temp_path"])
        except OSError:
            pass
        logging.info(f"[FileUpload] Cleaned up stale upload {msg_id} ({upload['file_name']})")
    _pending_uploads.clear()


async def handle_file_upload_envelope(header: dict, payload: bytes, ws) -> None:
    """
    Process a file upload binary envelope (START / CHUNK / END).
    Sends JSON text response on completion or error.
    """
    if file_system is None:
        msg_id = header.get("id", "")
        await ws.send(json.dumps({
            "id": msg_id,
            "status": "error",
            "error": "File management not available — SDK file_system module is missing",
        }))
        return

    msg_type = header.get("type", "")
    msg_id = header.get("id", "")

    if msg_type == "file_upload_start":
        file_name = header.get("file_name", "")
        total_size = header.get("total_size", 0)

        try:
            safe_name = file_system.sanitize_filename(file_name)
        except ValueError as e:
            await ws.send(json.dumps({"id": msg_id, "status": "error", "error": f"Invalid filename: {e}"}))
            return

        # Validate upload size
        try:
            total_size = int(total_size)
        except (TypeError, ValueError):
            total_size = 0
        if total_size > MAX_UPLOAD_BYTES:
            await ws.send(json.dumps({
                "id": msg_id, "status": "error",
                "error": f"File too large: {total_size} bytes exceeds limit of {MAX_UPLOAD_BYTES} bytes",
            }))
            return

        attachments_dir = str(file_system.get_attachments_dir())
        os.makedirs(attachments_dir, exist_ok=True)
        temp_path = os.path.join(attachments_dir, f".tmp_{msg_id}")

        _pending_uploads[msg_id] = {
            "file_name": safe_name,
            "total_size": total_size,
            "temp_path": temp_path,
            "hasher": hashlib.sha256(),
            "bytes_received": 0,
            "file_handle": open(temp_path, "wb"),
        }
        logging.info(f"[FileUpload] START {safe_name} (expected {total_size} bytes)")

    elif msg_type == "file_upload_chunk":
        upload = _pending_uploads.get(msg_id)
        if not upload:
            logging.warning(f"[FileUpload] CHUNK for unknown upload {msg_id}")
            await ws.send(json.dumps({
                "id": msg_id, "status": "error",
                "error": "No pending upload for this id. Upload may have been cancelled or timed out.",
            }))
            return
        # Enforce size limit even if total_size header was wrong/missing
        if upload["bytes_received"] + len(payload) > MAX_UPLOAD_BYTES:
            try:
                upload["file_handle"].close()
            except Exception:
                pass
            try:
                os.unlink(upload["temp_path"])
            except OSError:
                pass
            _pending_uploads.pop(msg_id, None)
            await ws.send(json.dumps({
                "id": msg_id, "status": "error",
                "error": f"Upload exceeded size limit of {MAX_UPLOAD_BYTES} bytes",
            }))
            return
        upload["file_handle"].write(payload)
        upload["hasher"].update(payload)
        upload["bytes_received"] += len(payload)

    elif msg_type == "file_upload_end":
        upload = _pending_uploads.pop(msg_id, None)
        if not upload:
            await ws.send(json.dumps({"id": msg_id, "status": "error", "error": "No pending upload"}))
            return

        try:
            upload["file_handle"].close()
        except Exception:
            pass
        expected_sha = header.get("sha256", "")
        actual_sha = upload["hasher"].hexdigest()

        if expected_sha and actual_sha != expected_sha:
            try:
                os.unlink(upload["temp_path"])
            except OSError:
                pass
            await ws.send(json.dumps({
                "id": msg_id,
                "status": "error",
                "error": f"Checksum mismatch: expected {expected_sha}, got {actual_sha}",
            }))
            logging.error(f"[FileUpload] Checksum mismatch for {upload['file_name']}")
            return

        final_dir = str(file_system.get_attachments_dir())
        final_path = os.path.join(final_dir, upload["file_name"])
        try:
            os.replace(upload["temp_path"], final_path)
        except OSError as e:
            await ws.send(json.dumps({"id": msg_id, "status": "error", "error": str(e)}))
            return

        file_system._invalidate_cache()

        logging.info(
            f"[FileUpload] COMPLETE {upload['file_name']} "
            f"({upload['bytes_received']} bytes, sha256={actual_sha[:12]}...)"
        )
        await ws.send(json.dumps({
            "id": msg_id,
            "status": "success",
            "file_name": upload["file_name"],
            "bytes_received": upload["bytes_received"],
        }))

async def stream_frames_direct(
    base_ws_uri,
    stream_socket_id,
    get_latest_frame,
    get_active_capture_run_ids,
    interval=0.5,
    retry_delay=5.0,
):
    stream_uri = base_ws_uri + stream_socket_id + "-stream"
    last_hash_by_run = {}
    last_sent_ts_by_run = {}
    logging.info(f"[Direct] Starting dedicated stream loop to: {stream_uri}")

    while True:
        try:
            async with websockets.connect(stream_uri) as ws:
                logging.info(f"[Direct] Connected to streaming endpoint: {stream_uri}")
                while True:
                    try:
                        run_ids = await asyncio.get_running_loop().run_in_executor(
                            None, get_active_capture_run_ids
                        )
                    except Exception:
                        run_ids = []

                    for run_id in run_ids:
                        try:
                            frame_bytes = await asyncio.get_running_loop().run_in_executor(
                                None, lambda rid=run_id: get_latest_frame(rid)
                            )
                        except Exception:
                            frame_bytes = None

                        now = time.time()
                        if frame_bytes:
                            h = hashlib.sha256(frame_bytes).hexdigest()
                            if h != last_hash_by_run.get(run_id):
                                seq = int(last_sent_ts_by_run.get(run_id) or now)
                                header = {"id": run_id, "type": "screenshot", "seq": seq}
                                envelope = _make_envelope(header, frame_bytes)
                                await ws.send(envelope)
                                last_hash_by_run[run_id] = h
                                last_sent_ts_by_run[run_id] = now
                    await asyncio.sleep(interval)
        except (websockets.exceptions.WebSocketException, OSError) as e:
            logging.error(f"[Direct] Stream connection failed: {e}. Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
        except Exception:
            logging.exception("[Direct] Unexpected error in stream loop")
            await asyncio.sleep(retry_delay)

async def stream_frames_multiplex(ws, get_latest_frame, get_active_capture_run_ids, interval=1.0):
    last_hash_by_run = {}
    logging.info("[Manager] Starting multiplexed stream for active run_ids")

    while True:
        try:
            try:
                run_ids = await asyncio.get_running_loop().run_in_executor(
                    None, get_active_capture_run_ids
                )
            except Exception:
                run_ids = []

            for run_id in run_ids:
                try:
                    frame = await asyncio.get_running_loop().run_in_executor(
                        None, lambda rid=run_id: get_latest_frame(rid)
                    )
                except Exception:
                    frame = None
                if not frame:
                    continue
                h = hashlib.sha256(frame).hexdigest()
                if h != last_hash_by_run.get(run_id):
                    header = {"id": run_id, "type": "screenshot", "seq": int(time.time())}
                    envelope = _make_envelope(header, frame)
                    await ws.send(envelope)
                    last_hash_by_run[run_id] = h

            # Cleanup stale hash entries
            active = set(run_ids)
            for stale in list(last_hash_by_run.keys()):
                if stale not in active:
                    last_hash_by_run.pop(stale, None)
            await asyncio.sleep(interval)
        except websockets.exceptions.ConnectionClosed:
            logging.warning("[Manager] WebSocket closed, stopping multiplex stream task.")
            break
        except Exception:
            logging.exception("[Manager] Error in multiplex stream")
            break


async def call_maybe_blocking(func, *args, **kwargs):
    if asyncio.iscoroutinefunction(func):
        return await func(*args, **kwargs)
    return await asyncio.to_thread(func, *args, **kwargs)


def _parse_boolish(value, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _normalize_accessibility_config(raw_config: Any) -> dict:
    config = raw_config if isinstance(raw_config, dict) else {}
    enabled = _parse_boolish(config.get("enabled"), False)
    policy = str(config.get("policy") or "").strip().lower()
    if not policy:
        if _parse_boolish(config.get("audit_after_each_step"), False):
            policy = "after_each_step"
        elif config.get("checkpoint_steps"):
            policy = "selected_steps"
        elif enabled:
            policy = "final_only"
        else:
            policy = "off"
    if policy not in {"off", "final_only", "after_each_step", "selected_steps"}:
        policy = "final_only" if enabled else "off"

    checkpoint_steps = set()
    for step in config.get("checkpoint_steps", []) or []:
        try:
            checkpoint_steps.add(int(step))
        except (TypeError, ValueError):
            continue

    return {
        "enabled": enabled and policy != "off",
        "policy": policy,
        "checkpoint_steps": checkpoint_steps,
        "audit_name": str(config.get("audit_name") or "Bulk accessibility audit"),
        "standard_profile": str(config.get("standard_profile") or "wcag22aa"),
        "scope_selector": str(config.get("scope_selector") or ""),
        "include_best_practices": "true" if _parse_boolish(config.get("include_best_practices"), True) else "false",
        "include_experimental": "true" if _parse_boolish(config.get("include_experimental"), False) else "false",
        "include_manual_placeholders": "true" if _parse_boolish(config.get("include_manual_placeholders"), True) else "false",
        "viewport_profile": str(config.get("viewport_profile") or "desktop,mobile"),
        "wait_for_network_idle": "true" if _parse_boolish(config.get("wait_for_network_idle"), True) else "false",
    }


def _get_accessibility_bindings() -> tuple:
    if _AGENT_MODULE is None:
        return None, "Agent module is not registered in SDK runtime."

    required = (
        "driver",
        "_create_session",
        "append_accessibility_audit_checkpoint",
        "finalize_accessibility_audit_session",
    )
    missing = [name for name in required if not hasattr(_AGENT_MODULE, name)]
    if missing:
        return None, "Agent module is missing accessibility hooks: {}".format(", ".join(sorted(missing)))
    return {
        "driver_store": getattr(_AGENT_MODULE, "driver"),
        "create_session": getattr(_AGENT_MODULE, "_create_session"),
        "append_checkpoint": getattr(_AGENT_MODULE, "append_accessibility_audit_checkpoint"),
        "finalize_session": getattr(_AGENT_MODULE, "finalize_accessibility_audit_session"),
    }, None


def _bulk_step_label(command: dict, func_name: str, index: int) -> str:
    label = command.get("label") or command.get("name") or command.get("step_label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return "{} {}".format(func_name.replace("_", " ").strip() or "Step", index)


def _bulk_should_checkpoint_after(policy: str, checkpoint_steps: set, command: dict, step_index: int) -> bool:
    if command.get("function") == "stop_driver":
        return False
    audit_after = command.get("audit_after")
    if audit_after is not None:
        return _parse_boolish(audit_after, False)
    if policy == "after_each_step":
        return True
    if policy == "selected_steps":
        return step_index in checkpoint_steps
    return False


def _bulk_should_checkpoint_before_stop(policy: str, checkpoint_steps: set, command: dict, step_index: int) -> bool:
    if command.get("function") != "stop_driver":
        return False
    if command.get("audit_before") is not None:
        return _parse_boolish(command.get("audit_before"), False)
    return policy in {"final_only", "after_each_step"} or step_index in checkpoint_steps


def _build_accessibility_step_result(index: int, label: str, func_name: str, status: str, result: Any = None, error: Optional[str] = None) -> dict:
    payload = {
        "step_index": index,
        "label": label,
        "action": func_name,
        "status": status,
    }
    if error is not None:
        payload["error"] = error
    else:
        payload["result"] = result
    return payload


def _driver_state_for_run(bindings: dict, run_id: str):
    driver_store = bindings["driver_store"]
    if not isinstance(driver_store, dict):
        return None
    return driver_store.get(run_id)


def _ensure_accessibility_session(bindings: dict, run_id: str, config: dict, existing_session: Optional[dict]):
    if existing_session is not None:
        return existing_session, None
    driver_state = _driver_state_for_run(bindings, run_id)
    if not driver_state:
        return None, "No active driver found for run id {}".format(run_id)
    session = bindings["create_session"](
        driver_state=driver_state,
        audit_name=config["audit_name"],
        standard_profile=config["standard_profile"],
        scope_selector=config["scope_selector"],
        include_best_practices=config["include_best_practices"],
        include_experimental=config["include_experimental"],
        include_manual_placeholders=config["include_manual_placeholders"],
        viewport_profile=config["viewport_profile"],
        wait_for_network_idle=config["wait_for_network_idle"],
        _run_test_id=run_id,
    )
    return session, None


async def _append_accessibility_checkpoint(bindings: dict, session: Optional[dict], run_id: str, config: dict, label: str, step_index: int):
    session, error = _ensure_accessibility_session(bindings, run_id, config, session)
    if session is None:
        return None, {
            "status": "skipped",
            "reason": error,
            "journey_step_index": step_index,
            "journey_step_label": label,
        }
    checkpoint = await bindings["append_checkpoint"](session, label, step_index)
    return session, checkpoint


async def _finalize_accessibility_session(bindings: dict, session: Optional[dict], step_results: list):
    if session is None:
        return {"status": "skipped", "reason": "Accessibility session was never started because no audit checkpoint could run."}
    session["scenario_steps_executed"] = step_results
    session["execution_notes"].append("Bulk steps executed: {}".format(len(step_results)))
    final_result = await bindings["finalize_session"](session)
    if isinstance(final_result, str):
        try:
            return json.loads(final_result)
        except json.JSONDecodeError:
            return {"status": "error", "error": "Accessibility finalizer returned invalid JSON."}
    if isinstance(final_result, dict):
        return final_result
    return {"status": "error", "error": "Accessibility finalizer returned unsupported payload."}


async def execute_macro_bulk_with_accessibility(
    commands: list,
    FUNCTION_MAP: dict,
    run_id: str = "1",
    accessibility: Optional[dict] = None,
) -> dict:
    config = _normalize_accessibility_config(accessibility)
    if not config["enabled"]:
        return await execute_macro_bulk(commands, FUNCTION_MAP, run_id=run_id)

    bindings, bindings_error = _get_accessibility_bindings()
    if bindings is None:
        bulk_result = await execute_macro_bulk(commands, FUNCTION_MAP, run_id=run_id)
        bulk_result["accessibility"] = {"status": "error", "error": bindings_error}
        return bulk_result

    executed_lines = 0
    results = []
    driver_created_for = set()
    step_outputs: list = []
    session = None
    step_results = []
    print(f"Executing macro with {len(commands)} commands and accessibility enabled, run_id={run_id}")

    async def _complete_with_result(base_result: dict) -> dict:
        base_result["accessibility"] = await _finalize_accessibility_session(bindings, session, step_results)
        return base_result

    try:
        for i, command in enumerate(commands):
            func_name = command.get("function")
            args = command.get("args", []) or []
            kwargs = command.get("kwargs", {}) or {}
            command_run_id = kwargs.get("_run_test_id") or run_id
            step_index = i + 1
            step_label = _bulk_step_label(command, func_name or "step", step_index)

            if step_outputs:
                args, kwargs = _resolve_step_output_vars_in_command(args, kwargs, step_outputs)

            if _bulk_should_checkpoint_before_stop(config["policy"], config["checkpoint_steps"], command, step_index):
                session, checkpoint = await _append_accessibility_checkpoint(
                    bindings, session, command_run_id, config, "{} (before stop)".format(step_label), step_index
                )
                if checkpoint and checkpoint.get("status") == "skipped" and session is not None:
                    session["execution_notes"].append(checkpoint["reason"])

            if func_name == "create_driver":
                driver_created_for.add(command_run_id)
            if func_name == "stop_driver":
                driver_created_for.discard(command_run_id)

            if func_name not in FUNCTION_MAP:
                error_msg = f"Unknown function: {func_name}"
                results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "error", "output": error_msg})
                step_results.append(_build_accessibility_step_result(step_index, step_label, func_name or "", "error", error=error_msg))
                return await _complete_with_result({
                    "status": "error",
                    "executed_lines": executed_lines,
                    "failed_index": i,
                    "failed_function": func_name,
                    "error_details": error_msg,
                    "message": f"Macro halted at index {i} due to error.",
                    "results": results
                })

            try:
                if "_run_test_id" not in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["_run_test_id"] = command_run_id

                result = await call_maybe_blocking(FUNCTION_MAP[func_name], *args, **kwargs)
                if isinstance(result, dict) and result.get("status") == "error":
                    error_msg = result.get("error", "Unknown error from function")
                    results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "error", "output": error_msg})
                    step_results.append(_build_accessibility_step_result(step_index, step_label, func_name, "error", error=error_msg))
                    session, checkpoint = await _append_accessibility_checkpoint(bindings, session, command_run_id, config, step_label, step_index)
                    if checkpoint and checkpoint.get("status") == "skipped" and session is not None:
                        session["execution_notes"].append(checkpoint["reason"])
                    return await _complete_with_result({
                        "status": "error",
                        "executed_lines": executed_lines,
                        "failed_index": i,
                        "failed_function": func_name,
                        "error_details": error_msg,
                        "message": f"Macro halted at index {i} due to error.",
                        "results": results
                    })
                element_hint = _extract_element_hint(args, kwargs, result)
                rich_output_enabled = os.getenv("BARKO_RICH_BULK_OUTPUT", "1").lower() not in ("0", "false", "no")
                raw_output = result if isinstance(result, (str, dict, list)) else str(result)
                if rich_output_enabled:
                    output = _format_step_output(func_name, args, kwargs, raw_output)
                else:
                    output = raw_output if isinstance(raw_output, str) else str(raw_output)
                results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "success", "output": output, "raw_output": raw_output})
                step_outputs.append(_output_to_str(raw_output))
                step_results.append(_build_accessibility_step_result(step_index, step_label, func_name, "success", result=raw_output))
            except Exception as e:
                logging.error(f"[BulkExec] Step {i} ({func_name}) failed: {e}")
                error_msg = str(e)
                results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "error", "output": error_msg})
                step_results.append(_build_accessibility_step_result(step_index, step_label, func_name or "", "error", error=error_msg))
                session, checkpoint = await _append_accessibility_checkpoint(bindings, session, command_run_id, config, step_label, step_index)
                if checkpoint and checkpoint.get("status") == "skipped" and session is not None:
                    session["execution_notes"].append(checkpoint["reason"])
                return await _complete_with_result({
                    "status": "error",
                    "executed_lines": executed_lines,
                    "failed_index": i,
                    "failed_function": func_name,
                    "error_details": error_msg,
                    "message": f"Macro halted at index {i} due to error.",
                    "results": results
                })

            executed_lines += 1

            if func_name != "stop_driver":
                try:
                    await streaming.capture_step_frame_async(
                        run_id=command_run_id,
                        step_index=i,
                        func_name=func_name,
                        element_hint=element_hint,
                        step_result=result,
                    )
                    print(f"Captured frame for step {i} ({func_name})")
                except Exception:
                    print(f"Warning: Failed to capture frame for step {i} ({func_name})")
                    pass

            if _bulk_should_checkpoint_after(config["policy"], config["checkpoint_steps"], command, step_index):
                session, checkpoint = await _append_accessibility_checkpoint(bindings, session, command_run_id, config, step_label, step_index)
                if checkpoint and checkpoint.get("status") == "skipped" and session is not None:
                    session["execution_notes"].append(checkpoint["reason"])

        if step_results:
            last_step = step_results[-1]
            last_audited_index = session["journey_steps"][-1]["journey_step_index"] if session and session["journey_steps"] else None
            if last_step["action"] != "stop_driver" and config["policy"] == "final_only" and last_audited_index != last_step["step_index"]:
                session, checkpoint = await _append_accessibility_checkpoint(
                    bindings, session, run_id, config, last_step["label"], last_step["step_index"]
                )
                if checkpoint and checkpoint.get("status") == "skipped" and session is not None:
                    session["execution_notes"].append(checkpoint["reason"])

        return await _complete_with_result({
            "status": "success",
            "executed_lines": executed_lines,
            "failed_index": None,
            "error_details": None,
            "message": f"Successfully executed {executed_lines} commands sequentially.",
            "results": results
        })
    except Exception as e:
        logging.error(f"[BulkExec] Fatal error in macro execution: {e}")
        return await _complete_with_result({
            "status": "error",
            "executed_lines": executed_lines,
            "failed_index": None,
            "failed_function": None,
            "error_details": str(e),
            "message": "Macro execution failed due to an unexpected error.",
            "results": results
        })
    finally:
        if driver_created_for:
            for created_run_id in sorted(driver_created_for):
                logging.info(
                    f"[BulkExec] macro failure for ({created_run_id}) ."
                )

async def execute_macro_bulk(commands: list, FUNCTION_MAP: dict, run_id: str = "1") -> dict:
    executed_lines = 0
    results = []
    driver_created_for = set()
    # Track each step's raw output (as a string) so that subsequent steps can
    # reference it via {{step_output:N:REGEX}} placeholders in their args/kwargs.
    step_outputs: list = []
    print(f"Executing macro with {len(commands)} commands, run_id={run_id}")

    try:
        for i, command in enumerate(commands):
            func_name = command.get("function")
            args = command.get("args", []) or []
            kwargs = command.get("kwargs", {}) or {}
            command_run_id = kwargs.get("_run_test_id") or run_id

            # Resolve {{step_output:N:REGEX}} placeholders before executing
            if step_outputs:
                args, kwargs = _resolve_step_output_vars_in_command(args, kwargs, step_outputs)

            if func_name == "create_driver":
                driver_created_for.add(command_run_id)
            if func_name == "stop_driver":
                driver_created_for.discard(command_run_id)

            if func_name not in FUNCTION_MAP:
                error_msg = f"Unknown function: {func_name}"
                results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "error", "output": error_msg})
                return {
                    "status": "error",
                    "executed_lines": executed_lines,
                    "failed_index": i,
                    "failed_function": func_name,
                    "error_details": error_msg,
                    "message": f"Macro halted at index {i} due to error.",
                    "results": results
                }

            try:
                if "_run_test_id" not in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["_run_test_id"] = command_run_id

                result = await call_maybe_blocking(FUNCTION_MAP[func_name], *args, **kwargs)
                if isinstance(result, dict) and result.get("status") == "error":
                    error_msg = result.get("error", "Unknown error from function")
                    results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "error", "output": error_msg})
                    return {
                        "status": "error",
                        "executed_lines": executed_lines,
                        "failed_index": i,
                        "failed_function": func_name,
                        "error_details": error_msg,
                        "message": f"Macro halted at index {i} due to error.",
                        "results": results
                    }
                element_hint = _extract_element_hint(args, kwargs, result)
                rich_output_enabled = os.getenv("BARKO_RICH_BULK_OUTPUT", "1").lower() not in ("0", "false", "no")
                raw_output = result if isinstance(result, (str, dict, list)) else str(result)
                if rich_output_enabled:
                    output = _format_step_output(func_name, args, kwargs, raw_output)
                else:
                    output = raw_output if isinstance(raw_output, str) else str(raw_output)
                results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "success", "output": output, "raw_output": raw_output})
                # Track raw output for {{step_output:N:REGEX}} resolution in later steps.
                # Use the raw string form of the result (not the display-formatted output).
                step_outputs.append(_output_to_str(raw_output))
            except Exception as e:
                logging.error(f"[BulkExec] Step {i} ({func_name}) failed: {e}")
                error_msg = str(e)
                results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "error", "output": error_msg})
                return {
                    "status": "error",
                    "executed_lines": executed_lines,
                    "failed_index": i,
                    "failed_function": func_name,
                    "error_details": error_msg,
                    "message": f"Macro halted at index {i} due to error.",
                    "results": results
                }

            executed_lines += 1

            # Capture a persistent frame after each step (skip stop_driver since browser is dead)
            if func_name != "stop_driver":
                try:
                    await streaming.capture_step_frame_async(
                        run_id=command_run_id,
                        step_index=i,
                        func_name=func_name,
                        element_hint=element_hint,
                        step_result=result,
                    )
                    print(f"Captured frame for step {i} ({func_name})")  # Simple print to avoid logging issues in some environments
                except Exception:
                    print(f"Warning: Failed to capture frame for step {i} ({func_name})")  # Avoid logging module in case of setup issues
                    pass  # Non-fatal: recording is best-effort

        return {
            "status": "success",
            "executed_lines": executed_lines,
            "failed_index": None,
            "error_details": None,
            "message": f"Successfully executed {executed_lines} commands sequentially.",
            "results": results
        }
    except Exception as e:
        logging.error(f"[BulkExec] Fatal error in macro execution: {e}")
        return {
            "status": "error",
            "executed_lines": executed_lines,
            "failed_index": None,
            "failed_function": None,
            "error_details": str(e),
            "message": "Macro execution failed due to an unexpected error.",
            "results": results
        }
    finally:
        if driver_created_for:
            for created_run_id in sorted(driver_created_for):
                logging.info(
                    f"[BulkExec] macro failure for ({created_run_id}) ."
                )
    

async def handle_message(message, FUNCTION_MAP, SYSTEM_FUNCTIONS):
    response_dict = {}
    message_id = None
    try:
        data = json.loads(message)
        message_id = data.get("id") or (data.get("kwargs", {}) or {}).get("_run_test_id")
        function_name = data.get("function")
        args = data.get("args", []) or []
        kwargs = data.get("kwargs", {}) or {}

        if not message_id:
            return json.dumps(
                {"status": "error", "error": "Missing required envelope id (run_id)."}
            )

        response_dict = {"id": message_id}

        if function_name == "list_available_methods":
            method_details = []
            for name, func in FUNCTION_MAP.items():
                sig = inspect.signature(func)
                arg_names = [p.name for p in sig.parameters.values() if p.name != "_run_test_id"]
                # Collect default values for optional parameters
                defaults = {}
                for p in sig.parameters.values():
                    if p.name != "_run_test_id" and p.default is not inspect.Parameter.empty:
                        defaults[p.name] = p.default
                detail = {"name": name, "args": arg_names, "doc": func.__doc__ or ""}
                if defaults:
                    detail["defaults"] = defaults
                method_details.append(detail)
            system_methods = [
                {"name": name, "args": [], "doc": "[SYSTEM]"} 
                for name in SYSTEM_FUNCTIONS.keys()
            ]
            response_dict.update({
                "status": "success", 
                "methods": method_details,
                "system_methods": system_methods
            })
            return json.dumps(response_dict)

        if function_name == "execute_macro_bulk":
            commands = args[0] if args else kwargs.get("commands", [])
            _run_test_id = kwargs.get("_run_test_id") or message_id
            accessibility = kwargs.get("accessibility") or kwargs.get("a11y")
            result = await execute_macro_bulk_with_accessibility(
                commands,
                FUNCTION_MAP,
                run_id=_run_test_id,
                accessibility=accessibility,
            )
            response_dict.update(result)
            return json.dumps(response_dict)

        if function_name in SYSTEM_FUNCTIONS:
            if "_run_test_id" not in kwargs:
                sig = inspect.signature(SYSTEM_FUNCTIONS[function_name])
                if "_run_test_id" in sig.parameters:
                    kwargs = dict(kwargs)
                    kwargs["_run_test_id"] = message_id
            result = await call_maybe_blocking(SYSTEM_FUNCTIONS[function_name], *args, **kwargs)
            response_dict.update({"status": "success", "result": result})
            logging.info(f"Executed SYSTEM function: {function_name}")
            return json.dumps(response_dict)

        if function_name in FUNCTION_MAP:
            sig = inspect.signature(FUNCTION_MAP[function_name])
            if "_run_test_id" in sig.parameters and "_run_test_id" not in kwargs:
                kwargs = dict(kwargs)
                kwargs["_run_test_id"] = message_id
            result = await call_maybe_blocking(FUNCTION_MAP[function_name], *args, **kwargs)
            response_dict.update({"status": "success", "result": result})
        else:
            response_dict.update({"status": "error", "error": f"Unknown function: {function_name}"})

    except Exception as e:
        logging.error(f"Error handling message: {e}")
        response_dict = {"status": "error", "error": str(e), "id": message_id}

    return json.dumps(response_dict)

async def handle_and_send(message, ws, SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS):
    try:
        async with SEM:
            response = await handle_message(message, FUNCTION_MAP, SYSTEM_FUNCTIONS)
            await ws.send(response)
    except Exception:
        logging.exception("Failed to send response")

async def command_loop(ws, SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS):
    while True:
        try:
            msg = await ws.recv()
        except websockets.exceptions.ConnectionClosedOK:
            logging.info("Command connection closed cleanly.")
            return
        # Binary messages are file upload envelopes
        if isinstance(msg, bytes):
            try:
                header, payload = _parse_envelope(msg)
                msg_type = header.get("type", "")
                if msg_type.startswith("file_upload_"):
                    await handle_file_upload_envelope(header, payload, ws)
                else:
                    logging.warning(f"[WS] Unknown binary envelope type: {msg_type}")
            except Exception as e:
                logging.error(f"[WS] Error handling binary message: {e}")
        else:
            asyncio.create_task(handle_and_send(msg, ws, SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS))

def _build_uri(base_or_id: str) -> str:
    if base_or_id.startswith("ws://") or base_or_id.startswith("wss://"):
        return base_or_id
    default_base = os.getenv("DEFAULT_WS_BASE", "wss://beta.barkoagent.com/ws/")
    return f"{default_base.rstrip('/')}/{base_or_id.lstrip('/')}"

async def connect_to_backend(uri, connection_type, SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS, get_latest_frame, get_active_capture_run_ids):
    enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() in ("1", "true", "yes")
    logging.info(f"Connecting to Backend ({connection_type}): {uri}")

    while True:
        try:
            async with websockets.connect(uri) as ws:
                logging.info("Command connection established.")
                tasks = [asyncio.create_task(command_loop(ws, SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS))]
                if connection_type == "manager" and enable_streaming:
                    tasks.append(
                        asyncio.create_task(
                            stream_frames_multiplex(ws, get_latest_frame, get_active_capture_run_ids)
                        )
                    )
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                for task in pending:
                    task.cancel()
        except Exception as e:
            logging.error(f"Connection lost: {e}. Reconnecting in 5s...")
        # Clean up any in-progress uploads (file handles + temp files)
        _cleanup_pending_uploads()
        await asyncio.sleep(5)

async def main_connect_ws(agent_func):
    setup_logging()
    global _AGENT_MODULE
    _AGENT_MODULE = agent_func
    SEM = get_semaphore()
    FUNCTION_MAP = build_function_map(agent_func)
    SYSTEM_FUNCTIONS = build_system_functions()
    from .streaming import get_latest_frame, get_active_capture_run_ids

    enable_improvements = os.getenv("ENABLE_IMPROVEMENTS", "false").lower() in ("1", "true", "yes")
    if enable_improvements:
        from .improvements import ImprovementsManager
        improvements_manager = ImprovementsManager(FUNCTION_MAP, agent_func)
        improvements_manager.load()
        FUNCTION_MAP["get_agent_source"]  = improvements_manager.get_agent_source
        FUNCTION_MAP["add_improvement"]   = improvements_manager.add_improvement
        FUNCTION_MAP["list_improvements"] = improvements_manager.list_improvements
        logging.info("[Improvements] Self-improvement mode enabled.")

    raw_uri = os.getenv("BACKEND_WS_URI", "default_client_id")
    full_uri = _build_uri(raw_uri)
    conn_type = os.getenv("AGENT_CONNECTION_TYPE", "manager").lower()
    run_id = os.getenv("STREAMING_RUN_ID", "1")
    enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() in ("1", "true", "yes")

    logging.info(f"Agent starting. Mode: {conn_type.upper()}")

    if conn_type == "direct":
        tasks = [
            asyncio.create_task(
                connect_to_backend(
                    full_uri, "direct", SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS, get_latest_frame, get_active_capture_run_ids
                )
            ),
        ]
        if enable_streaming:
            tasks.append(
                asyncio.create_task(
                    stream_frames_direct(
                        full_uri, run_id, get_latest_frame, get_active_capture_run_ids
                    )
                )
            )
        await asyncio.gather(*tasks)
    else:
        await connect_to_backend(
            full_uri, "manager", SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS, get_latest_frame, get_active_capture_run_ids
        )
