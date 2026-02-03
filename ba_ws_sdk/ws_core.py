# baweb_ws_sdk/ws_core.py

import asyncio
import inspect
import json
import logging
import time
import websockets
import os
import struct
import hashlib
from . import streaming

from typing import Optional, Callable

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
    return {
        name: obj
        for name, obj in inspect.getmembers(agent_func, inspect.isfunction)
        if not name.startswith("_")
    }

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

async def stream_frames_direct(base_ws_uri, run_id, get_latest_frame, interval=0.5, retry_delay=5.0):
    stream_uri = base_ws_uri + run_id + "-stream"
    last_hash = None
    last_sent_ts = None
    logging.info(f"[Direct] Starting dedicated stream loop to: {stream_uri}")

    while True:
        try:
            async with websockets.connect(stream_uri) as ws:
                logging.info(f"[Direct] Connected to streaming endpoint: {stream_uri}")
                while True:
                    try:
                        frame_bytes = await asyncio.get_running_loop().run_in_executor(
                            None, lambda: get_latest_frame(run_id)
                        )
                    except Exception:
                        frame_bytes = None

                    now = time.time()
                    if frame_bytes:
                        h = hashlib.sha256(frame_bytes).hexdigest()
                        if h != last_hash:
                            seq = int(last_sent_ts or now)
                            header = {"id": run_id, "type": "screenshot", "seq": seq}
                            envelope = _make_envelope(header, frame_bytes)
                            await ws.send(envelope)
                            last_hash = h
                            last_sent_ts = now
                    await asyncio.sleep(interval)
        except (websockets.exceptions.WebSocketException, OSError) as e:
            logging.error(f"[Direct] Stream connection failed: {e}. Retrying in {retry_delay}s...")
            await asyncio.sleep(retry_delay)
        except Exception:
            logging.exception("[Direct] Unexpected error in stream loop")
            await asyncio.sleep(retry_delay)

async def stream_frames_multiplex(ws, run_id, get_latest_frame, interval=1.0):
    last_hash = None
    logging.info(f"[Manager] Starting multiplexed stream for run_id: {run_id}")

    while True:
        try:
            frame = await asyncio.get_running_loop().run_in_executor(
                None, lambda: get_latest_frame(run_id)
            )
            if frame:
                h = hashlib.sha256(frame).hexdigest()
                if h != last_hash:
                    header = {"id": run_id, "type": "screenshot", "seq": int(time.time())}
                    envelope = _make_envelope(header, frame)
                    await ws.send(envelope)
                    last_hash = h
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

async def handle_message(message, FUNCTION_MAP, SYSTEM_FUNCTIONS):
    response_dict = {}
    message_id = None
    try:
        data = json.loads(message)
        message_id = data.get("id") or data.get("kwargs", {}).get("_run_test_id")
        function_name = data.get("function")
        args = data.get("args", []) or []
        kwargs = data.get("kwargs", {}) or {}

        response_dict = {"id": message_id} if message_id else {}

        if function_name == "list_available_methods":
            method_details = []
            for name, func in FUNCTION_MAP.items():
                sig = inspect.signature(func)
                arg_names = [p.name for p in sig.parameters.values() if p.name != "_run_test_id"]
                method_details.append({"name": name, "args": arg_names, "doc": func.__doc__ or ""})
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

        if function_name in SYSTEM_FUNCTIONS:
            result = await call_maybe_blocking(SYSTEM_FUNCTIONS[function_name], *args, **kwargs)
            response_dict.update({"status": "success", "result": result})
            logging.info(f"Executed SYSTEM function: {function_name}")
            return json.dumps(response_dict)

        if function_name in FUNCTION_MAP:
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
        msg = await ws.recv()
        asyncio.create_task(handle_and_send(msg, ws, SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS))

def _build_uri(base_or_id: str) -> str:
    if base_or_id.startswith("ws://") or base_or_id.startswith("wss://"):
        return base_or_id
    default_base = os.getenv("DEFAULT_WS_BASE", "wss://beta.barkoagent.com/ws/")
    return f"{default_base.rstrip('/')}/{base_or_id.lstrip('/')}"

async def connect_to_backend(uri, connection_type, SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS, get_latest_frame):
    run_id = os.getenv("STREAMING_RUN_ID", "1")
    enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() in ("1", "true", "yes")
    logging.info(f"Connecting to Backend ({connection_type}): {uri}")

    while True:
        try:
            async with websockets.connect(uri) as ws:
                logging.info("Command connection established.")
                tasks = [asyncio.create_task(command_loop(ws, SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS))]
                if connection_type == "manager" and enable_streaming:
                    tasks.append(asyncio.create_task(stream_frames_multiplex(ws, run_id, get_latest_frame)))
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
                for task in pending: task.cancel()
        except Exception as e:
            logging.error(f"Connection lost: {e}. Reconnecting in 5s...")
        await asyncio.sleep(5)

async def main_connect_ws(agent_func):
    setup_logging()
    SEM = get_semaphore()
    FUNCTION_MAP = build_function_map(agent_func)
    SYSTEM_FUNCTIONS = build_system_functions()
    from .streaming import get_latest_frame

    raw_uri = os.getenv("BACKEND_WS_URI", "default_client_id")
    full_uri = _build_uri(raw_uri)
    conn_type = os.getenv("AGENT_CONNECTION_TYPE", "manager").lower()
    run_id = os.getenv("STREAMING_RUN_ID", "1")
    enable_streaming = os.getenv("ENABLE_STREAMING", "true").lower() in ("1", "true", "yes")

    logging.info(f"Agent starting. Mode: {conn_type.upper()}")

    if conn_type == "direct":
        tasks = [
            asyncio.create_task(connect_to_backend(full_uri, "direct", SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS, get_latest_frame)),
        ]
        if enable_streaming:
            tasks.append(
                asyncio.create_task(stream_frames_direct(full_uri, run_id, get_latest_frame))
            )
        await asyncio.gather(*tasks)
    else:
        await connect_to_backend(full_uri, "manager", SEM, FUNCTION_MAP, SYSTEM_FUNCTIONS, get_latest_frame)
