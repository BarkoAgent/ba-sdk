# streaming.py
import base64
import threading
import time
import logging
import os
import inspect
from typing import Any, Dict, Optional, Tuple
import numpy as np
import cv2
import asyncio

# Configuration (env-overrideable)
RECORDING_TTL_SECONDS = int(os.getenv("RECORDING_TTL_SECONDS", "300"))
MAX_ACTIVE_RECORDINGS = int(os.getenv("MAX_ACTIVE_RECORDINGS", "3"))
MAX_FRAMES_PER_RECORDING = int(os.getenv("MAX_FRAMES_PER_RECORDING", "1000"))
RECORDING_GC_INTERVAL = int(os.getenv("RECORDING_GC_INTERVAL", "30"))
ENABLE_ELEMENT_BBOX = os.getenv("ENABLE_ELEMENT_BBOX", "true").lower() in ("1", "true", "yes")
STEP_CAPTURE_JPEG_QUALITY = int(os.getenv("STEP_CAPTURE_JPEG_QUALITY", "80"))

# Globals
_STREAM_THREADS = {}       # run_id -> Thread
_STREAM_TASKS: Dict[str, asyncio.Task] = {}            # run_id -> asyncio.Task
_STREAM_FLAGS = {}         # run_id -> threading.Event (stop flag)
_LATEST_FRAMES = {}        # run_id -> (jpeg_bytes, timestamp)
_RECORDING_FLAGS = set()   # set of run_ids currently recording
_RECORDED_FRAMES = {}      # run_id -> list of {seq, timestamp, data (bytes)}
_SEQ_COUNTERS = {}         # run_id -> next seq number
_ACKED_UP_TO = {}          # run_id -> last acked seq (-1 means none acked)
_LAST_FRAME_AT = {}        # run_id -> timestamp of last frame capture
_CAPTURE_DRIVERS = {}      # run_id -> driver reference (for per-step capture)
_LOCK = threading.Lock()
_GC_STARTED = False

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logging.getLogger("urllib3").setLevel(logging.ERROR)


def register_driver(run_id: str, driver) -> None:
    """Register a driver so capture_step_frame can take screenshots from it."""
    with _LOCK:
        _CAPTURE_DRIVERS[run_id] = driver
    logging.info(f"Registered driver for capture: run_id={run_id}")


def unregister_driver(run_id: str) -> None:
    """Unregister a driver when it's being stopped."""
    with _LOCK:
        _CAPTURE_DRIVERS.pop(run_id, None)
    logging.info(f"Unregistered driver for capture: run_id={run_id}")


def _append_recorded_frame_locked(
    run_id: str,
    jpeg_bytes: bytes,
    trigger: str = "timer",
    step_index: int = None,
    func_name: str = None,
) -> None:
    if run_id not in _RECORDING_FLAGS:
        return
    if len(_RECORDED_FRAMES.get(run_id, [])) >= MAX_FRAMES_PER_RECORDING:
        logging.warning(
            f"[LIMIT] Recording {run_id} exceeded MAX_FRAMES_PER_RECORDING ({MAX_FRAMES_PER_RECORDING}). Auto-stopping."
        )
        _RECORDING_FLAGS.discard(run_id)
        return
    if run_id not in _RECORDED_FRAMES:
        _RECORDED_FRAMES[run_id] = []
    if run_id not in _SEQ_COUNTERS:
        _SEQ_COUNTERS[run_id] = 0
    now = time.time()
    seq = _SEQ_COUNTERS[run_id]
    _RECORDED_FRAMES[run_id].append(
        {
            "seq": seq,
            "timestamp": now,
            "data": jpeg_bytes,
            "trigger": trigger,
            "step_index": step_index,
            "func_name": func_name,
        }
    )
    _SEQ_COUNTERS[run_id] = seq + 1
    _LAST_FRAME_AT[run_id] = now


def _append_recorded_timer_frame_for_linked_runs_locked(source_run_id: str, jpeg_bytes: bytes) -> None:
    """
    Append timer frames to all run_ids that share the same driver object.
    This keeps recording working when callers start recording with a different
    run_id than the one used to start the stream.
    """
    _append_recorded_frame_locked(run_id=source_run_id, jpeg_bytes=jpeg_bytes, trigger="timer")

    source_driver = _CAPTURE_DRIVERS.get(source_run_id)
    if source_driver is None:
        return

    for run_id, driver in _CAPTURE_DRIVERS.items():
        if run_id == source_run_id:
            continue
        if driver is not source_driver:
            continue
        _append_recorded_frame_locked(run_id=run_id, jpeg_bytes=jpeg_bytes, trigger="timer")


def _decode_image_bytes(image_bytes: bytes):
    nparr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Failed to decode image bytes")
    return img


def _normalized_bbox(raw_bbox: Any) -> Optional[dict]:
    if not isinstance(raw_bbox, dict):
        return None
    if not all(k in raw_bbox for k in ("x", "y", "width", "height")):
        return None
    try:
        x = float(raw_bbox["x"])
        y = float(raw_bbox["y"])
        width = float(raw_bbox["width"])
        height = float(raw_bbox["height"])
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return {"x": x, "y": y, "width": width, "height": height}


def _bbox_from_payload(payload: Any) -> Optional[dict]:
    if not isinstance(payload, dict):
        return None
    if isinstance(payload.get("bounding_box"), dict):
        return _normalized_bbox(payload["bounding_box"])
    return _normalized_bbox(payload)


def _draw_bbox(img, bbox: dict, scale: float = 1.0, label: Optional[str] = None) -> None:
    if scale <= 0:
        scale = 1.0
    height, width = img.shape[:2]

    x1 = int(round(bbox["x"] * scale))
    y1 = int(round(bbox["y"] * scale))
    x2 = int(round((bbox["x"] + bbox["width"]) * scale))
    y2 = int(round((bbox["y"] + bbox["height"]) * scale))

    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))

    if x2 <= x1 or y2 <= y1:
        return

    color = (0, 255, 0)
    thickness = max(2, int(min(width, height) * 0.003))
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

    if label:
        text_origin = (x1, max(15, y1 - 8))
        cv2.putText(img, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2, cv2.LINE_AA)


def _image_to_jpeg_bytes(
    image_bytes: bytes,
    quality: int = 80,
    bbox: Optional[dict] = None,
    bbox_scale: float = 1.0,
    label: Optional[str] = None,
) -> bytes:
    img = _decode_image_bytes(image_bytes)
    if bbox:
        _draw_bbox(img, bbox, scale=bbox_scale, label=label)
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise RuntimeError("Failed to encode JPEG")
    return enc.tobytes()


def _png_to_jpeg_bytes(png_bytes: bytes, quality: int = 80) -> bytes:
    """
    Convert PNG bytes (Selenium's get_screenshot_as_png) to JPEG bytes.
    """
    return _image_to_jpeg_bytes(png_bytes, quality=quality)


def _get_selenium_driver(driver):
    if driver is None:
        return None
    if isinstance(driver, dict):
        raw_driver = driver.get("driver")
        if raw_driver is not None and hasattr(raw_driver, "get_screenshot_as_png"):
            return raw_driver
        return None
    if hasattr(driver, "get_screenshot_as_png"):
        return driver
    if hasattr(driver, "get_driver") and callable(driver.get_driver):
        try:
            raw_driver = driver.get_driver()
        except Exception as exc:
            logging.debug(f"Failed to unwrap Selenium driver: {exc}")
            return None
        if raw_driver is not None and hasattr(raw_driver, "get_screenshot_as_png"):
            return raw_driver
    return None


def _get_playwright_page(driver):
    if driver is None:
        return None
    page = None
    if isinstance(driver, dict):
        page = driver.get("page")
    elif hasattr(driver, "page"):
        page = getattr(driver, "page")
        if callable(page):
            try:
                page = page()
            except Exception:
                page = None
    elif hasattr(driver, "get_page") and callable(driver.get_page):
        try:
            page = driver.get_page()
        except Exception:
            page = None
    if page is not None and hasattr(page, "screenshot"):
        return page
    return None


def _extract_locator(element_hint: Optional[dict]) -> Tuple[Optional[str], Optional[str]]:
    if not isinstance(element_hint, dict):
        return None, None
    locator_type = element_hint.get("locator_type") or element_hint.get("by")
    locator = element_hint.get("locator") or element_hint.get("selector") or element_hint.get("value")
    if locator is None:
        return (str(locator_type).strip().lower() if locator_type else None), None
    locator_value = str(locator).strip()
    if not locator_value:
        return (str(locator_type).strip().lower() if locator_type else None), None
    return (str(locator_type).strip().lower() if locator_type else None), locator_value


def _is_xpath(locator: str) -> bool:
    text = locator.strip()
    return text.startswith(("xpath=", "//", "(//", "/", "./"))


def _is_simple_token(locator: str) -> bool:
    text = locator.strip()
    if not text:
        return False
    if any(ch in text for ch in (" ", "/", "=", "[", "]", ">", ":", "(", ")")):
        return False
    return True


def _candidate_selenium_locators(locator_type: Optional[str], locator: str):
    by_map = {
        "id": "id",
        "name": "name",
        "xpath": "xpath",
        "css selector": "css selector",
        "css_selector": "css selector",
        "css": "css selector",
        "class name": "class name",
        "class_name": "class name",
        "tag name": "tag name",
        "tag_name": "tag name",
        "link text": "link text",
        "partial link text": "partial link text",
        "partial_link_text": "partial link text",
        "text": "link text",
    }
    candidates = []
    locator_text = locator.strip()

    def _add(by: str, value: str):
        key = (by, value)
        if key not in candidates:
            candidates.append(key)

    if locator_type and locator_type in by_map:
        _add(by_map[locator_type], locator_text)

    if locator_text.startswith("xpath="):
        _add("xpath", locator_text[len("xpath="):])
    if locator_text.startswith("css="):
        _add("css selector", locator_text[len("css="):])
    if _is_xpath(locator_text):
        cleaned = locator_text[len("xpath="):] if locator_text.startswith("xpath=") else locator_text
        _add("xpath", cleaned)
    else:
        if _is_simple_token(locator_text):
            _add("id", locator_text)
            _add("name", locator_text)
        _add("css selector", locator_text)

    return candidates


def _candidate_playwright_selectors(locator_type: Optional[str], locator: str):
    candidates = []
    locator_text = locator.strip()

    def _add(selector: str):
        if selector and selector not in candidates:
            candidates.append(selector)

    if locator_type in ("xpath",):
        _add(f"xpath={locator_text}")
    elif locator_type in ("css selector", "css_selector", "css"):
        _add(f"css={locator_text}")
    elif locator_type == "id":
        _add(f"css=[id=\"{locator_text}\"]")
    elif locator_type == "name":
        _add(f"css=[name=\"{locator_text}\"]")
    elif locator_type in ("class name", "class_name"):
        _add(f"css=.{locator_text}")
    elif locator_type in ("text", "link text", "partial link text", "partial_link_text"):
        _add(f"text={locator_text}")
    elif locator_type == "role":
        _add(f"role={locator_text}")

    if locator_text.startswith(("css=", "xpath=", "text=", "role=")):
        _add(locator_text)
    elif _is_xpath(locator_text):
        cleaned = locator_text[len("xpath="):] if locator_text.startswith("xpath=") else locator_text
        _add(f"xpath={cleaned}")
    else:
        if _is_simple_token(locator_text):
            _add(f"css=[id=\"{locator_text}\"]")
            _add(f"css=[name=\"{locator_text}\"]")
        _add(locator_text)
        _add(f"css={locator_text}")

    return candidates


def _get_selenium_device_pixel_ratio(selenium_driver) -> float:
    try:
        raw = selenium_driver.execute_script("return window.devicePixelRatio || 1;")
        value = float(raw)
        return value if value > 0 else 1.0
    except Exception:
        return 1.0


async def _get_playwright_device_pixel_ratio(page) -> float:
    try:
        raw = await page.evaluate("() => window.devicePixelRatio || 1")
        value = float(raw)
        return value if value > 0 else 1.0
    except Exception:
        return 1.0


async def _playwright_screenshot_bytes(page) -> Tuple[bytes, bool]:
    try:
        return await page.screenshot(scale="css"), True
    except TypeError:
        return await page.screenshot(), False


def _is_target_closed_error(exc: Exception) -> bool:
    message = str(exc).lower()
    name = exc.__class__.__name__.lower()
    return "targetclosed" in name or "target page, context or browser has been closed" in message


def _extract_selenium_bbox_from_result(step_result: Any) -> Optional[dict]:
    direct = _bbox_from_payload(step_result)
    if direct:
        return direct
    rect = getattr(step_result, "rect", None)
    return _normalized_bbox(rect)


async def _extract_playwright_bbox_from_result(step_result: Any) -> Optional[dict]:
    direct = _bbox_from_payload(step_result)
    if direct:
        return direct
    method = getattr(step_result, "bounding_box", None)
    if not callable(method):
        return None
    maybe_bbox = method()
    if inspect.isawaitable(maybe_bbox):
        maybe_bbox = await maybe_bbox
    return _normalized_bbox(maybe_bbox)


def _resolve_selenium_bbox(selenium_driver, element_hint: Optional[dict], step_result: Any) -> Tuple[Optional[dict], float]:
    bbox = _bbox_from_payload(element_hint) or _extract_selenium_bbox_from_result(step_result)
    dpr = _get_selenium_device_pixel_ratio(selenium_driver)
    if bbox:
        return bbox, dpr

    locator_type, locator = _extract_locator(element_hint)
    if not locator:
        return None, dpr

    for by, locator_value in _candidate_selenium_locators(locator_type, locator):
        try:
            element = selenium_driver.find_element(by=by, value=locator_value)
            maybe_bbox = _normalized_bbox(getattr(element, "rect", None))
            if maybe_bbox:
                logging.debug(f"[StepCapture] Selenium bbox resolved with by={by}")
                return maybe_bbox, dpr
        except Exception as exc:
            logging.debug(f"[StepCapture] Selenium element not found by={by}: {exc}")
    return None, dpr


async def _resolve_playwright_bbox(page, element_hint: Optional[dict], step_result: Any) -> Tuple[Optional[dict], float]:
    bbox = _bbox_from_payload(element_hint)
    if not bbox:
        bbox = await _extract_playwright_bbox_from_result(step_result)

    dpr = await _get_playwright_device_pixel_ratio(page)
    if bbox:
        return bbox, dpr

    locator_type, locator = _extract_locator(element_hint)
    if not locator:
        return None, dpr

    for selector in _candidate_playwright_selectors(locator_type, locator):
        try:
            el = page.locator(selector).first

            # Check quickly if element exists (no long auto-wait)
            if await el.count() == 0:
                continue

            maybe_bbox = await el.bounding_box(timeout=100)  # 100ms max
            normalized = _normalized_bbox(maybe_bbox)

            if normalized:
                logging.debug(f"[StepCapture] Playwright bbox resolved with selector={selector}")
                return normalized, dpr

        except Exception as exc:
            logging.debug(f"[StepCapture] Playwright element not found selector={selector}: {exc}")

    return None, dpr

def _append_step_capture(run_id: str, jpeg_bytes: bytes, step_index: int = None, func_name: str = None) -> None:
    with _LOCK:
        _append_recorded_frame_locked(
            run_id=run_id,
            jpeg_bytes=jpeg_bytes,
            trigger="step",
            step_index=step_index,
            func_name=func_name,
        )


def _capture_step_frame_selenium(
    run_id: str,
    selenium_driver,
    step_index: int = None,
    func_name: str = None,
    element_hint: Optional[dict] = None,
    step_result: Any = None,
) -> None:
    png_bytes = selenium_driver.get_screenshot_as_png()
    bbox = None
    dpr = 1.0
    if ENABLE_ELEMENT_BBOX:
        bbox, dpr = _resolve_selenium_bbox(selenium_driver, element_hint, step_result)
    jpeg_bytes = _image_to_jpeg_bytes(
        png_bytes,
        quality=STEP_CAPTURE_JPEG_QUALITY,
        bbox=bbox,
        bbox_scale=dpr,
        label=func_name if bbox else None,
    )
    _append_step_capture(run_id, jpeg_bytes, step_index=step_index, func_name=func_name)


async def _capture_step_frame_playwright(
    run_id: str,
    page,
    step_index: int = None,
    func_name: str = None,
    element_hint: Optional[dict] = None,
    step_result: Any = None,
) -> None:
    png_bytes, is_css_scaled = await _playwright_screenshot_bytes(page)
    bbox = None
    dpr = 1.0
    if ENABLE_ELEMENT_BBOX:
        bbox, dpr = await _resolve_playwright_bbox(page, element_hint, step_result)
    bbox_scale = 1.0 if is_css_scaled else dpr
    jpeg_bytes = _image_to_jpeg_bytes(
        png_bytes,
        quality=STEP_CAPTURE_JPEG_QUALITY,
        bbox=bbox,
        bbox_scale=bbox_scale,
        label=func_name if bbox else None,
    )
    _append_step_capture(run_id, jpeg_bytes, step_index=step_index, func_name=func_name)


async def capture_step_frame_async(
    run_id: str,
    step_index: int = None,
    func_name: str = None,
    element_hint: Optional[dict] = None,
    step_result: Any = None,
) -> None:
    """
    Take a fresh screenshot from the run-owned driver and append it only to
    that run's recording bucket. Supports Selenium and Playwright drivers.
    """
    with _LOCK:
        if run_id not in _RECORDING_FLAGS:
            return
        driver = _CAPTURE_DRIVERS.get(run_id)
        if driver is None:
            return

    selenium_driver = _get_selenium_driver(driver)
    if selenium_driver is not None:
        try:
            await asyncio.to_thread(
                _capture_step_frame_selenium,
                run_id,
                selenium_driver,
                step_index,
                func_name,
                element_hint,
                step_result,
            )
        except Exception as exc:
            logging.warning(f"[StepCapture] Failed to capture Selenium frame for step {step_index} ({func_name}): {exc}")
        return

    page = _get_playwright_page(driver)
    if page is not None:
        try:
            await _capture_step_frame_playwright(
                run_id,
                page,
                step_index=step_index,
                func_name=func_name,
                element_hint=element_hint,
                step_result=step_result,
            )
        except Exception as exc:
            logging.warning(f"[StepCapture] Failed to capture Playwright frame for step {step_index} ({func_name}): {exc}")
        return

    logging.debug(f"[StepCapture] Unsupported driver type for run_id={run_id}")


def capture_step_frame(
    run_id: str,
    step_index: int = None,
    func_name: str = None,
    element_hint: Optional[dict] = None,
    step_result: Any = None,
) -> None:
    """
    Backward-compatible sync wrapper around capture_step_frame_async.
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        loop.create_task(
            capture_step_frame_async(
                run_id=run_id,
                step_index=step_index,
                func_name=func_name,
                element_hint=element_hint,
                step_result=step_result,
            )
        )
        return

    asyncio.run(
        capture_step_frame_async(
            run_id=run_id,
            step_index=step_index,
            func_name=func_name,
            element_hint=element_hint,
            step_result=step_result,
        )
    )


def capture_step_frame_sync(
    run_id: str,
    func_name: str = None,
    element_hint: Optional[dict] = None,
    step_index: int = None,
    step_result: Any = None,
) -> None:
    with _LOCK:
        if run_id not in _RECORDING_FLAGS:
            return
        driver = _CAPTURE_DRIVERS.get(run_id)
        if driver is None:
            return
    selenium_driver = _get_selenium_driver(driver)
    if selenium_driver is None:
        logging.debug(f"[StepCapture] capture_step_frame_sync: no Selenium driver for run_id={run_id}")
        return
    try:
        _capture_step_frame_selenium(
            run_id=run_id,
            selenium_driver=selenium_driver,
            step_index=step_index,
            func_name=func_name,
            element_hint=element_hint,
            step_result=step_result,
        )
    except Exception as exc:
        logging.warning(f"[StepCapture] capture_step_frame_sync failed ({func_name}): {exc}")


def drop_recent_timer_frames(run_id: str, within_seconds: float = 1.5) -> None:
    cutoff = time.time() - within_seconds
    removed = 0
    with _LOCK:
        frames = _RECORDED_FRAMES.get(run_id)
        if not frames:
            return

        while frames and frames[-1].get("trigger") == "timer" and frames[-1]["timestamp"] >= cutoff:
            frames.pop()
            removed += 1

        step_idx = None
        for i in range(len(frames) - 1, -1, -1):
            if frames[i].get("trigger") == "step":
                step_idx = i
                break
        if step_idx is not None and step_idx > 0:
            to_remove = []
            i = step_idx - 1
            while i >= 0 and frames[i].get("trigger") == "timer" and frames[i]["timestamp"] >= cutoff:
                to_remove.append(i)
                i -= 1
            for idx in sorted(to_remove, reverse=True):
                frames.pop(idx)
            removed += len(to_remove)

    if removed:
        logging.debug(f"[StepCapture] Dropped {removed} recent timer frame(s) for {run_id} to prevent duplicate")


async def capture_error_frame_async(run_id: str, func_name: str) -> None:
    recording_active = run_id in _RECORDING_FLAGS
    logging.info(
        f"[ErrorCapture] {func_name} failed — capturing error screenshot "
        f"(run_id={run_id}, recording_active={recording_active})"
    )
    try:
        if not recording_active:
            logging.warning(
                f"[ErrorCapture] Recording not active for {run_id}; "
                "activating it so the error frame can be stored."
            )
            start_recording(run_id)
        await capture_step_frame_async(
            run_id=run_id,
            func_name=f"{func_name} (error)",
            element_hint=None,
        )
        logging.info(f"[ErrorCapture] Error screenshot stored for {func_name}")
    except Exception as exc:
        logging.warning(f"[ErrorCapture] Error screenshot failed for {func_name}: {exc}")


def capture_error_frame(run_id: str, func_name: str) -> None:
    recording_active = run_id in _RECORDING_FLAGS
    logging.info(
        f"[ErrorCapture] {func_name} failed — capturing error screenshot "
        f"(run_id={run_id}, recording_active={recording_active})"
    )
    try:
        if not recording_active:
            logging.warning(
                f"[ErrorCapture] Recording not active for {run_id}; "
                "activating it so the error frame can be stored."
            )
            start_recording(run_id)
        driver_obj = _CAPTURE_DRIVERS.get(run_id)
        if driver_obj is None:
            return
        selenium_driver = _get_selenium_driver(driver_obj)
        if selenium_driver is None:
            return
        _capture_step_frame_selenium(
            run_id=run_id,
            selenium_driver=selenium_driver,
            func_name=f"{func_name} (error)",
            element_hint=None,
        )
        logging.info(f"[ErrorCapture] Error screenshot stored for {func_name}")
    except Exception as exc:
        logging.warning(f"[ErrorCapture] Error screenshot failed for {func_name}: {exc}")


def _stream_worker(run_id: str, driver, fps: float, jpeg_quality: int, stop_event: threading.Event, stop_timeout: Optional[float] = None):
    """
    Worker thread: captures screenshots, converts to jpeg, stores latest bytes.
    """
    interval = 1.0 / max(0.1, fps)
    logging.info(f"Stream worker started for run_id={run_id} fps={fps}")
    time_started = time.time()
    selenium_driver = _get_selenium_driver(driver)
    if selenium_driver is None:
        logging.error(f"Stream worker requires Selenium-compatible driver for run_id={run_id}")
        return
    try:
        while not stop_event.is_set() and (stop_timeout is None or (time.time() - time_started) < stop_timeout):
            try:
                png_bytes = selenium_driver.get_screenshot_as_png()
            except Exception:
                logging.exception(f"Failed to capture screenshot for run {run_id}")
                stop_event.wait(interval)
                stop_stream(run_id)
                continue

            try:
                jpeg_bytes = _png_to_jpeg_bytes(png_bytes, quality=jpeg_quality)
            except Exception:
                logging.exception(f"PNG->JPEG conversion failed for run {run_id}; storing PNG as fallback")
                jpeg_bytes = png_bytes

            with _LOCK:
                _LATEST_FRAMES[run_id] = (jpeg_bytes, time.time())
                _append_recorded_timer_frame_for_linked_runs_locked(run_id, jpeg_bytes)

            stop_event.wait(interval)
    except Exception:
        logging.exception("Unexpected exception in stream worker")
    finally:
        logging.info(f"Stream worker exiting for run_id={run_id}")


def start_stream(driver, run_id: str = "1", fps: float = 1.0, jpeg_quality: int = 70, stop_timeout: Optional[float] = 180.0) -> None:
    """
    Start streaming for `run_id`.
    - Selenium drivers use a background thread.
    - Playwright drivers schedule async worker on current event loop.
    """
    playwright_page = _get_playwright_page(driver)
    if playwright_page is not None:
        with _LOCK:
            thread = _STREAM_THREADS.get(run_id)
        if thread and thread.is_alive():
            stop_stream(run_id)

        # Playwright async API must execute on a running event loop.
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            loop.create_task(
                astart_stream(
                    driver=driver,
                    run_id=run_id,
                    fps=fps,
                    jpeg_quality=jpeg_quality,
                    stop_after=stop_timeout,
                )
            )
            logging.info(f"Scheduled async Playwright stream for run_id={run_id}")
        else:
            logging.error(
                f"Cannot start Playwright stream for run_id={run_id}: no running event loop. "
                "Use `await astart_stream(...)`."
            )
            register_driver(run_id, driver)
        return

    _cancel_stream_task_if_any(run_id)

    with _LOCK:
        thread = _STREAM_THREADS.get(run_id)
    if thread and thread.is_alive():
        logging.info(f"Stream already running for run_id={run_id}. Restarting stream.")
        stop_stream(run_id)

    with _LOCK:
        # Double-check in case another caller started a thread meanwhile.
        thread = _STREAM_THREADS.get(run_id)
        if thread and thread.is_alive():
            logging.warning(
                f"Unable to start new stream for run_id={run_id} because an existing thread is still alive"
            )
            return

        stop_event = threading.Event()
        thread = threading.Thread(
            target=_stream_worker,
            args=(run_id, driver, fps, jpeg_quality, stop_event, stop_timeout),
            daemon=True,
            name=f"stream-{run_id}"
        )
        _STREAM_THREADS[run_id] = thread
        _STREAM_FLAGS[run_id] = stop_event
        thread.start()
        logging.info(f"Started streaming thread for run_id={run_id}")

    register_driver(run_id, driver)


def stop_stream(run_id: str) -> None:
    """
    Stop stream thread for run_id and cleanup.
    """
    with _LOCK:
        stop_event = _STREAM_FLAGS.get(run_id)
        thread = _STREAM_THREADS.get(run_id)

    if stop_event:
        stop_event.set()
        if thread:
            thread.join(timeout=2.0)

    with _LOCK:
        _STREAM_FLAGS.pop(run_id, None)
        _STREAM_THREADS.pop(run_id, None)
        _LATEST_FRAMES.pop(run_id, None)
        # We generally do NOT clear recorded frames here,
        # so they can be retrieved after driver stops.
        # user can call clear_recorded_frames explicitly.

    unregister_driver(run_id)
    logging.info(f"Stopped stream for run_id={run_id}")


def get_latest_frame(run_id: str) -> Optional[bytes]:
    """
    Return latest frame bytes for run_id (JPEG bytes preferably), or None.
    """
    with _LOCK:
        item = _LATEST_FRAMES.get(run_id)
        if item is None:
            driver = _CAPTURE_DRIVERS.get(run_id)
            if driver is not None:
                for other_run_id, other_driver in _CAPTURE_DRIVERS.items():
                    if other_run_id == run_id:
                        continue
                    if other_driver is not driver:
                        continue
                    item = _LATEST_FRAMES.get(other_run_id)
                    if item is not None:
                        break
        if item is None:
            return None
        return item[0]


def get_active_capture_run_ids():
    """Return run_ids that currently own capture state or latest frames."""
    with _LOCK:
        run_ids = set(_CAPTURE_DRIVERS.keys()) | set(_LATEST_FRAMES.keys()) | set(_RECORDING_FLAGS)
    return list(run_ids)


# -------------------------------------------------------------------------
# Recording / Persistence Logic
# -------------------------------------------------------------------------

def start_recording(run_id: str) -> None:
    """
    Enable recording (persistence) of frames for this run_id.
    """
    with _LOCK:
        # Hard limit: max active recordings
        if len(_RECORDING_FLAGS) >= MAX_ACTIVE_RECORDINGS:
            logging.warning(f"[LIMIT] Cannot start recording {run_id}: MAX_ACTIVE_RECORDINGS ({MAX_ACTIVE_RECORDINGS}) reached")
            return
        
        _RECORDING_FLAGS.add(run_id)
        if run_id not in _CAPTURE_DRIVERS and len(_CAPTURE_DRIVERS) == 1:
            existing_run_id, existing_driver = next(iter(_CAPTURE_DRIVERS.items()))
            _CAPTURE_DRIVERS[run_id] = existing_driver
            logging.info(
                f"Aliased capture driver from run_id={existing_run_id} to recording run_id={run_id}"
            )
        if run_id not in _RECORDED_FRAMES:
            _RECORDED_FRAMES[run_id] = []
        if run_id not in _SEQ_COUNTERS:
            _SEQ_COUNTERS[run_id] = 0
        if run_id not in _ACKED_UP_TO:
            _ACKED_UP_TO[run_id] = -1
        _LAST_FRAME_AT[run_id] = time.time()
    logging.info(f"Started recording frames for run_id={run_id}")

def stop_recording(run_id: str) -> None:
    """
    Disable recording of frames for this run_id.
    """
    with _LOCK:
        _RECORDING_FLAGS.discard(run_id)
    logging.info(f"Stopped recording frames for run_id={run_id}")

def get_recorded_frames(run_id: str, since_seq: int = 0, limit: int = 50):
    """
    Return frames where seq >= since_seq and seq > last_acked.
    Returns at most `limit` frames, ordered by seq ascending.
    """
    with _LOCK:
        frames = _RECORDED_FRAMES.get(run_id, [])
        acked = _ACKED_UP_TO.get(run_id, -1)
        
        # Filter: seq > acked AND seq >= since_seq
        filtered = [
            f for f in frames 
            if f["seq"] > acked and f["seq"] >= since_seq
        ]
        
        # Sort by seq and limit
        filtered.sort(key=lambda f: f["seq"])
        return filtered[:limit]


def ack_recorded_frames(run_id: str, up_to_seq: int) -> None:
    """
    Acknowledge frames up to and including up_to_seq.
    These frames will be excluded from future get_recorded_frames calls
    and can be garbage collected.
    """
    with _LOCK:
        _ACKED_UP_TO[run_id] = max(
            _ACKED_UP_TO.get(run_id, -1),
            up_to_seq
        )
        
        # Actually remove acked frames to free memory
        if run_id in _RECORDED_FRAMES:
            _RECORDED_FRAMES[run_id] = [
                f for f in _RECORDED_FRAMES[run_id]
                if f["seq"] > up_to_seq
            ]
    
    logging.info(f"ACK'd frames up to seq={up_to_seq} for run_id={run_id}")


def clear_recorded_frames(run_id: str) -> None:
    """
    Clear all recorded frames and reset seq counter for the given run_id.
    """
    with _LOCK:
        _RECORDING_FLAGS.discard(run_id)
        _RECORDED_FRAMES.pop(run_id, None)
        _SEQ_COUNTERS.pop(run_id, None)
        _ACKED_UP_TO.pop(run_id, None)
        _LAST_FRAME_AT.pop(run_id, None)
    logging.info(f"Cleared all state for run_id={run_id}")


# -------------------------------------------------------------------------
# TTL-Based Garbage Collection
# -------------------------------------------------------------------------

def _recording_gc():
    """Background thread that cleans up stale recordings."""
    while True:
        now = time.time()
        with _LOCK:
            for run_id in list(_RECORDING_FLAGS):
                last_activity = _LAST_FRAME_AT.get(run_id, 0)
                if now - last_activity > RECORDING_TTL_SECONDS:
                    _RECORDING_FLAGS.discard(run_id)
                    _RECORDED_FRAMES.pop(run_id, None)
                    _SEQ_COUNTERS.pop(run_id, None)
                    _ACKED_UP_TO.pop(run_id, None)
                    _LAST_FRAME_AT.pop(run_id, None)
                    logging.warning(f"[GC] Cleaned up inactive recording run_id={run_id} (TTL expired)")
        time.sleep(RECORDING_GC_INTERVAL)


def _start_gc_once():
    global _GC_STARTED
    if not _GC_STARTED:
        _GC_STARTED = True
        threading.Thread(target=_recording_gc, daemon=True, name="recording-gc").start()
        logging.info("Recording GC thread started")


_start_gc_once()

async def _start_frame_recording(_run_test_id='1') -> str:
    """
    [SYSTEM] Starts persisting (recording) frames for the given run_id.
    Not exposed to users - called by backend only.
    """
    await asyncio.to_thread(start_recording, _run_test_id)
    return "recording started"


async def _stop_frame_recording(_run_test_id='1') -> str:
    """
    [SYSTEM] Stops persisting (recording) frames for the given run_id.
    Not exposed to users - called by backend only.
    """
    await asyncio.to_thread(stop_recording, _run_test_id)
    return "recording stopped"


async def _get_recorded_frames(_run_test_id='1', since_seq: int = 0, limit: int = 50):
    """
    [SYSTEM] Retrieves persisted frames with cursor-based pagination.
    Not exposed to users - called by backend only.
    
    Args:
        _run_test_id: Execution identifier
        since_seq: Return only frames with seq >= since_seq (default: 0)
        limit: Max frames to return (default: 50)
    
    Returns:
        List of dicts: [{'seq': int, 'timestamp': float, 'data': base64_str}, ...]
    """
    try:
        if since_seq == "":
            since_seq = 0
        else:
            since_seq = int(since_seq)
            
        if limit == "":
            limit = 50
        else:
            limit = int(limit)
    except (ValueError, TypeError):
        since_seq = 0
        limit = 50

    frames = await asyncio.to_thread(get_recorded_frames, _run_test_id, since_seq, limit)
    
    result = []
    for frame in frames:
        b64_str = base64.b64encode(frame["data"]).decode('utf-8')
        result.append({
            'seq': frame["seq"],
            'timestamp': frame["timestamp"],
            'data': b64_str,
            'trigger': frame.get("trigger", "timer"),
            'step_index': frame.get("step_index"),
            'func_name': frame.get("func_name")
        })
    
    return result


async def _ack_recorded_frames(_run_test_id='1', up_to_seq: int = 0) -> str:
    """
    [SYSTEM] Acknowledge frames up to and including up_to_seq.
    ACK'd frames are freed from memory and will not be returned in future calls.
    Not exposed to users - called by backend only.
    
    Args:
        _run_test_id: Execution identifier
        up_to_seq: ACK all frames with seq <= up_to_seq
    """
    try:
        if up_to_seq == "":
            up_to_seq = 0
        else:
            up_to_seq = int(up_to_seq)
    except (ValueError, TypeError):
        up_to_seq = 0

    await asyncio.to_thread(ack_recorded_frames, _run_test_id, up_to_seq)
    return "frames acknowledged"


async def _clear_frame_recording(_run_test_id='1') -> str:
    """
    [SYSTEM] Clears all persisted frames and resets seq counter for the given run_id.
    Not exposed to users - called by backend only.
    """
    await asyncio.to_thread(clear_recorded_frames, _run_test_id)
    return "recording cleared"


async def _astream_worker(
    run_id: str,
    driver,
    fps: float,
    jpeg_quality: int,
    stop_after: Optional[float] = None,
):
    """
    Async worker that captures screenshots from `driver` periodically.
    Expects a Playwright page-like object with awaitable `screenshot()`.
    Cancels itself when task is cancelled or stop_after exceeded.
    """
    interval = 1.0 / max(0.1, fps)
    logging.info(f"Stream worker started for run_id={run_id} fps={fps}")
    started_at = time.time()
    page = _get_playwright_page(driver)
    if page is None:
        logging.error(f"[{run_id}] Async stream worker requires Playwright page-compatible driver")
        return
    try:
        while True:
            # stop after timeout if requested
            if stop_after is not None and (time.time() - started_at) >= stop_after:
                logging.info(f"[{run_id}] stop_after reached ({stop_after}s). Exiting worker.")
                return

            if hasattr(page, "is_closed"):
                try:
                    if page.is_closed():
                        logging.info(f"[{run_id}] Playwright page closed. Exiting worker.")
                        return
                except Exception:
                    pass

            # allow cancellation
            await asyncio.sleep(0)  # cooperative cancellation point

            screenshot_timeout = max(5.0, interval * 2)
            try:
                # Await screenshot with a timeout
                png_bytes, _ = await asyncio.wait_for(_playwright_screenshot_bytes(page), timeout=screenshot_timeout)
            except asyncio.TimeoutError:
                logging.warning(f"[{run_id}] screenshot timed out after {screenshot_timeout}s")
                # avoid tight loop
                await asyncio.sleep(interval)
                continue
            except asyncio.CancelledError:
                # propagate cancellation cleanly
                logging.info(f"[{run_id}] worker cancelled during screenshot.")
                raise
            except Exception as exc:
                if _is_target_closed_error(exc):
                    logging.info(f"[{run_id}] Target closed while capturing screenshot. Exiting worker.")
                    return
                logging.exception(f"[{run_id}] Failed to capture screenshot; stopping worker.")
                return

            # convert to jpeg (with fallback to original bytes on failure)
            try:
                jpeg_bytes = _png_to_jpeg_bytes(png_bytes, quality=jpeg_quality)
            except Exception:
                logging.exception(f"[{run_id}] PNG->JPEG conversion failed; storing PNG as fallback")
                jpeg_bytes = png_bytes

            # store latest frame and record if enabled
            with _LOCK:
                _LATEST_FRAMES[run_id] = (jpeg_bytes, time.time())
                _append_recorded_timer_frame_for_linked_runs_locked(run_id, jpeg_bytes)

            # sleep for interval (cooperative; allows cancellation)
            try:
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                logging.info(f"[{run_id}] worker cancelled during sleep.")
                raise

    except asyncio.CancelledError:
        logging.info(f"[{run_id}] worker received cancellation.")
        raise
    except Exception:
        logging.exception(f"[{run_id}] Unexpected exception in stream worker for run_id={run_id}")
    finally:
        logging.info(f"Stream worker exiting for run_id={run_id}")


def _cancel_stream_task_if_any(run_id: str) -> None:
    task = _STREAM_TASKS.get(run_id)
    if task and not task.done():
        task.cancel()


async def astart_stream(
    driver,
    run_id: str = "1",
    fps: float = 1.0,
    jpeg_quality: int = 70,
    stop_after: Optional[float] = 180.0,
) -> None:
    """
    Start an async streaming task for `run_id`. No-op if already running.
    If an existing task is found and is still running, returns without starting a new one.
    """
    page = _get_playwright_page(driver)
    if page is None:
        logging.error(f"Cannot start async stream for run_id={run_id}: Playwright page not found in driver")
        register_driver(run_id, driver)
        return

    with _LOCK:
        thread = _STREAM_THREADS.get(run_id)
    if thread and thread.is_alive():
        await asyncio.to_thread(stop_stream, run_id)

    existing = _STREAM_TASKS.get(run_id)
    if existing and not existing.done():
        logging.info(f"Stream already running for run_id={run_id}; start_stream no-op.")
        return

    logging.info(f"Starting stream task for run_id={run_id} fps={fps} quality={jpeg_quality}")
    task = asyncio.create_task(_astream_worker(run_id, driver, fps, jpeg_quality, stop_after), name=f"stream-{run_id}")
    _STREAM_TASKS[run_id] = task

    register_driver(run_id, driver)

    # optional: attach done callback to clean up dicts when task finishes
    def _on_done(t: asyncio.Task, rid=run_id):
        logging.info(f"Stream task done for run_id={rid}. Cleaning up.")
        with _LOCK:
            _STREAM_TASKS.pop(rid, None)
            _LATEST_FRAMES.pop(rid, None)
        unregister_driver(rid)

    task.add_done_callback(_on_done)

async def astop_stream(run_id: str, cancel_timeout: float = 2.0) -> None:
    """
    Stop the stream task for `run_id` and cleanup.
    Attempts graceful cancellation and waits up to `cancel_timeout` seconds.
    """
    task = _STREAM_TASKS.get(run_id)

    if not task:
        logging.info(f"No active stream task for run_id={run_id}")
        # ensure any leftover frames removed
        with _LOCK:
            _LATEST_FRAMES.pop(run_id, None)
            _STREAM_TASKS.pop(run_id, None)
        return

    if task.done():
        logging.info(f"Stream task already done for run_id={run_id}")
        with _LOCK:
            _STREAM_TASKS.pop(run_id, None)
            _LATEST_FRAMES.pop(run_id, None)
        unregister_driver(run_id)
        return

    logging.info(f"Stopping stream task for run_id={run_id}")
    task.cancel()
    try:
        await asyncio.wait_for(asyncio.shield(task), timeout=cancel_timeout)
    except asyncio.TimeoutError:
        logging.warning(f"Timeout while waiting for task cancellation for run_id={run_id}")
    except Exception:
        logging.exception(f"Exception while stopping task for run_id={run_id}")
    finally:
        with _LOCK:
            _STREAM_TASKS.pop(run_id, None)
            _LATEST_FRAMES.pop(run_id, None)
        unregister_driver(run_id)
        logging.info(f"Stopped stream for run_id={run_id}")
