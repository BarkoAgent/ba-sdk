"""Unit tests for ba_ws_sdk.streaming helper functions and recording logic."""

import threading
import time
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from ba_ws_sdk import streaming


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def _reset_streaming_globals():
    """Reset all module-level state between tests."""
    with streaming._LOCK:
        streaming._STREAM_THREADS.clear()
        streaming._STREAM_TASKS.clear()
        streaming._STREAM_FLAGS.clear()
        streaming._LATEST_FRAMES.clear()
        streaming._RECORDING_FLAGS.clear()
        streaming._RECORDED_FRAMES.clear()
        streaming._SEQ_COUNTERS.clear()
        streaming._ACKED_UP_TO.clear()
        streaming._LAST_FRAME_AT.clear()
        streaming._CAPTURE_DRIVERS.clear()
    yield
    with streaming._LOCK:
        streaming._STREAM_THREADS.clear()
        streaming._STREAM_TASKS.clear()
        streaming._STREAM_FLAGS.clear()
        streaming._LATEST_FRAMES.clear()
        streaming._RECORDING_FLAGS.clear()
        streaming._RECORDED_FRAMES.clear()
        streaming._SEQ_COUNTERS.clear()
        streaming._ACKED_UP_TO.clear()
        streaming._LAST_FRAME_AT.clear()
        streaming._CAPTURE_DRIVERS.clear()


def _make_png_bytes(width=10, height=10):
    """Create a minimal valid PNG via OpenCV."""
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    img[2:8, 2:8] = (0, 128, 255)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return buf.tobytes()


def _make_jpeg_bytes(width=10, height=10):
    import cv2

    img = np.zeros((height, width, 3), dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


# ---------------------------------------------------------------------------
# register / unregister driver
# ---------------------------------------------------------------------------
class TestDriverRegistration:
    def test_register_and_unregister(self):
        driver = MagicMock()
        streaming.register_driver("r1", driver)
        assert streaming._CAPTURE_DRIVERS["r1"] is driver

        streaming.unregister_driver("r1")
        assert "r1" not in streaming._CAPTURE_DRIVERS

    def test_unregister_nonexistent(self):
        streaming.unregister_driver("nope")  # should not raise


# ---------------------------------------------------------------------------
# _normalized_bbox
# ---------------------------------------------------------------------------
class TestNormalizedBbox:
    def test_valid(self):
        bbox = streaming._normalized_bbox({"x": 1, "y": 2, "width": 100, "height": 50})
        assert bbox == {"x": 1.0, "y": 2.0, "width": 100.0, "height": 50.0}

    def test_missing_key(self):
        assert streaming._normalized_bbox({"x": 1, "y": 2}) is None

    def test_zero_dimension(self):
        assert streaming._normalized_bbox({"x": 0, "y": 0, "width": 0, "height": 10}) is None

    def test_negative_dimension(self):
        assert streaming._normalized_bbox({"x": 0, "y": 0, "width": -5, "height": 10}) is None

    def test_non_dict(self):
        assert streaming._normalized_bbox("not a dict") is None
        assert streaming._normalized_bbox(None) is None

    def test_non_numeric_values(self):
        assert streaming._normalized_bbox({"x": "a", "y": 0, "width": 10, "height": 10}) is None


# ---------------------------------------------------------------------------
# _bbox_from_payload
# ---------------------------------------------------------------------------
class TestBboxFromPayload:
    def test_nested_bounding_box(self):
        payload = {"bounding_box": {"x": 1, "y": 2, "width": 10, "height": 20}}
        bbox = streaming._bbox_from_payload(payload)
        assert bbox is not None
        assert bbox["x"] == 1.0

    def test_flat_payload(self):
        payload = {"x": 5, "y": 10, "width": 50, "height": 30}
        bbox = streaming._bbox_from_payload(payload)
        assert bbox is not None

    def test_non_dict(self):
        assert streaming._bbox_from_payload("nope") is None
        assert streaming._bbox_from_payload(None) is None


# ---------------------------------------------------------------------------
# Image processing
# ---------------------------------------------------------------------------
class TestImageProcessing:
    def test_decode_image_bytes(self):
        png = _make_png_bytes()
        img = streaming._decode_image_bytes(png)
        assert img is not None
        assert img.shape[0] == 10  # height
        assert img.shape[1] == 10  # width

    def test_decode_invalid_bytes(self):
        with pytest.raises(ValueError, match="Failed to decode"):
            streaming._decode_image_bytes(b"not an image")

    def test_png_to_jpeg_bytes(self):
        png = _make_png_bytes()
        jpeg = streaming._png_to_jpeg_bytes(png)
        assert len(jpeg) > 0
        # JPEG magic bytes
        assert jpeg[:2] == b"\xff\xd8"

    def test_image_to_jpeg_with_bbox(self):
        png = _make_png_bytes(width=100, height=100)
        bbox = {"x": 10, "y": 10, "width": 30, "height": 30}
        jpeg = streaming._image_to_jpeg_bytes(png, bbox=bbox, label="test")
        assert jpeg[:2] == b"\xff\xd8"


# ---------------------------------------------------------------------------
# _draw_bbox
# ---------------------------------------------------------------------------
class TestDrawBbox:
    def test_draws_without_error(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        streaming._draw_bbox(img, {"x": 10, "y": 10, "width": 30, "height": 30})
        # At minimum the image should have some non-zero pixels (the rectangle)
        assert img.sum() > 0

    def test_with_label(self):
        img = np.zeros((100, 100, 3), dtype=np.uint8)
        streaming._draw_bbox(img, {"x": 10, "y": 20, "width": 30, "height": 30}, label="click")
        assert img.sum() > 0

    def test_zero_scale(self):
        """scale <= 0 should default to 1.0 and not crash."""
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        streaming._draw_bbox(img, {"x": 5, "y": 5, "width": 10, "height": 10}, scale=0)

    def test_bbox_clipped_to_image(self):
        img = np.zeros((50, 50, 3), dtype=np.uint8)
        streaming._draw_bbox(img, {"x": 40, "y": 40, "width": 100, "height": 100})
        # Should not raise


# ---------------------------------------------------------------------------
# _extract_locator
# ---------------------------------------------------------------------------
class TestExtractLocator:
    def test_locator_type_and_locator(self):
        lt, lv = streaming._extract_locator({"locator_type": "xpath", "locator": "//div"})
        assert lt == "xpath"
        assert lv == "//div"

    def test_by_and_selector(self):
        lt, lv = streaming._extract_locator({"by": "CSS", "selector": ".btn"})
        assert lt == "css"
        assert lv == ".btn"

    def test_none_input(self):
        assert streaming._extract_locator(None) == (None, None)

    def test_empty_dict(self):
        assert streaming._extract_locator({}) == (None, None)


# ---------------------------------------------------------------------------
# Locator helpers
# ---------------------------------------------------------------------------
class TestIsXpath:
    def test_xpath_prefix(self):
        assert streaming._is_xpath("//div") is True
        assert streaming._is_xpath("(//div)") is True
        assert streaming._is_xpath("xpath=//div") is True

    def test_not_xpath(self):
        assert streaming._is_xpath(".class") is False
        assert streaming._is_xpath("#id") is False


class TestIsSimpleToken:
    def test_simple(self):
        assert streaming._is_simple_token("myButton") is True

    def test_not_simple(self):
        assert streaming._is_simple_token("//div[@id]") is False
        assert streaming._is_simple_token("a b") is False
        assert streaming._is_simple_token("") is False


# ---------------------------------------------------------------------------
# _candidate_selenium_locators
# ---------------------------------------------------------------------------
class TestCandidateSeleniumLocators:
    def test_explicit_type(self):
        candidates = streaming._candidate_selenium_locators("id", "myBtn")
        assert ("id", "myBtn") in candidates

    def test_xpath_prefix(self):
        candidates = streaming._candidate_selenium_locators(None, "xpath=//div")
        assert ("xpath", "//div") in candidates

    def test_css_prefix(self):
        candidates = streaming._candidate_selenium_locators(None, "css=.btn")
        assert ("css selector", ".btn") in candidates

    def test_simple_token_generates_id_and_name(self):
        candidates = streaming._candidate_selenium_locators(None, "submitBtn")
        bys = [c[0] for c in candidates]
        assert "id" in bys
        assert "name" in bys


# ---------------------------------------------------------------------------
# _candidate_playwright_selectors
# ---------------------------------------------------------------------------
class TestCandidatePlaywrightSelectors:
    def test_xpath_type(self):
        sels = streaming._candidate_playwright_selectors("xpath", "//div")
        assert "xpath=//div" in sels

    def test_css_type(self):
        sels = streaming._candidate_playwright_selectors("css", ".btn")
        assert "css=.btn" in sels

    def test_id_type(self):
        sels = streaming._candidate_playwright_selectors("id", "myId")
        assert 'css=[id="myId"]' in sels

    def test_text_type(self):
        sels = streaming._candidate_playwright_selectors("text", "Submit")
        assert "text=Submit" in sels

    def test_no_type_simple_token(self):
        sels = streaming._candidate_playwright_selectors(None, "myBtn")
        assert any("myBtn" in s for s in sels)


# ---------------------------------------------------------------------------
# Recording lifecycle
# ---------------------------------------------------------------------------
class TestRecordingLifecycle:
    def test_start_stop_recording(self):
        streaming.start_recording("r1")
        assert "r1" in streaming._RECORDING_FLAGS
        assert "r1" in streaming._RECORDED_FRAMES

        streaming.stop_recording("r1")
        assert "r1" not in streaming._RECORDING_FLAGS

    def test_max_active_recordings(self):
        for i in range(streaming.MAX_ACTIVE_RECORDINGS):
            streaming.start_recording(f"r{i}")
        streaming.start_recording("overflow")
        assert "overflow" not in streaming._RECORDING_FLAGS

    def test_clear_recorded_frames(self):
        streaming.start_recording("r1")
        streaming.clear_recorded_frames("r1")
        assert "r1" not in streaming._RECORDING_FLAGS
        assert "r1" not in streaming._RECORDED_FRAMES

    def test_start_recording_aliases_driver(self):
        driver = MagicMock()
        streaming.register_driver("stream-run", driver)
        streaming.start_recording("recording-run")
        # Should alias the only registered driver
        assert streaming._CAPTURE_DRIVERS.get("recording-run") is driver


# ---------------------------------------------------------------------------
# Frame recording
# ---------------------------------------------------------------------------
class TestFrameRecording:
    def test_append_recorded_frame(self):
        streaming.start_recording("r1")
        with streaming._LOCK:
            streaming._append_recorded_frame_locked("r1", b"frame1")
            streaming._append_recorded_frame_locked("r1", b"frame2")
        frames = streaming.get_recorded_frames("r1")
        assert len(frames) == 2
        assert frames[0]["seq"] == 0
        assert frames[1]["seq"] == 1

    def test_no_append_without_recording(self):
        with streaming._LOCK:
            streaming._append_recorded_frame_locked("r1", b"frame")
        assert streaming._RECORDED_FRAMES.get("r1") is None

    def test_get_recorded_frames_since_seq(self):
        streaming.start_recording("r1")
        with streaming._LOCK:
            for i in range(5):
                streaming._append_recorded_frame_locked("r1", f"frame{i}".encode())
        frames = streaming.get_recorded_frames("r1", since_seq=3)
        assert len(frames) == 2
        assert frames[0]["seq"] == 3

    def test_get_recorded_frames_limit(self):
        streaming.start_recording("r1")
        with streaming._LOCK:
            for i in range(10):
                streaming._append_recorded_frame_locked("r1", f"frame{i}".encode())
        frames = streaming.get_recorded_frames("r1", limit=3)
        assert len(frames) == 3

    def test_ack_frames(self):
        streaming.start_recording("r1")
        with streaming._LOCK:
            for i in range(5):
                streaming._append_recorded_frame_locked("r1", f"frame{i}".encode())

        streaming.ack_recorded_frames("r1", up_to_seq=2)
        frames = streaming.get_recorded_frames("r1")
        assert all(f["seq"] > 2 for f in frames)
        assert len(frames) == 2

    def test_append_step_frame(self):
        streaming.start_recording("r1")
        streaming._append_step_capture("r1", b"step-frame", step_index=0, func_name="click")
        frames = streaming.get_recorded_frames("r1")
        assert len(frames) == 1
        assert frames[0]["trigger"] == "step"
        assert frames[0]["func_name"] == "click"

    def test_max_frames_auto_stops(self):
        streaming.start_recording("r1")
        original_max = streaming.MAX_FRAMES_PER_RECORDING
        streaming.MAX_FRAMES_PER_RECORDING = 3
        try:
            with streaming._LOCK:
                for i in range(5):
                    streaming._append_recorded_frame_locked("r1", f"f{i}".encode())
            assert "r1" not in streaming._RECORDING_FLAGS
            assert len(streaming._RECORDED_FRAMES["r1"]) == 3
        finally:
            streaming.MAX_FRAMES_PER_RECORDING = original_max


# ---------------------------------------------------------------------------
# Linked run recording
# ---------------------------------------------------------------------------
class TestLinkedRunRecording:
    def test_linked_drivers_get_timer_frames(self):
        driver = MagicMock()
        streaming.register_driver("r1", driver)
        streaming.register_driver("r2", driver)  # same driver object
        streaming.start_recording("r1")
        streaming.start_recording("r2")

        with streaming._LOCK:
            streaming._append_recorded_timer_frame_for_linked_runs_locked("r1", b"frame")

        f1 = streaming.get_recorded_frames("r1")
        f2 = streaming.get_recorded_frames("r2")
        assert len(f1) == 1
        assert len(f2) == 1


# ---------------------------------------------------------------------------
# get_latest_frame / get_active_capture_run_ids
# ---------------------------------------------------------------------------
class TestLatestFrameAndActiveIds:
    def test_get_latest_frame(self):
        with streaming._LOCK:
            streaming._LATEST_FRAMES["r1"] = (b"jpeg-data", time.time())
        assert streaming.get_latest_frame("r1") == b"jpeg-data"

    def test_get_latest_frame_none(self):
        assert streaming.get_latest_frame("nonexistent") is None

    def test_get_latest_frame_from_linked_driver(self):
        driver = MagicMock()
        streaming.register_driver("r1", driver)
        streaming.register_driver("r2", driver)
        with streaming._LOCK:
            streaming._LATEST_FRAMES["r1"] = (b"shared-frame", time.time())
        # r2 shares the same driver, should get r1's frame
        assert streaming.get_latest_frame("r2") == b"shared-frame"

    def test_get_active_capture_run_ids(self):
        streaming.register_driver("r1", MagicMock())
        streaming.start_recording("r2")
        with streaming._LOCK:
            streaming._LATEST_FRAMES["r3"] = (b"data", time.time())

        ids = set(streaming.get_active_capture_run_ids())
        assert "r1" in ids
        assert "r2" in ids
        assert "r3" in ids


# ---------------------------------------------------------------------------
# Driver detection helpers
# ---------------------------------------------------------------------------
class TestGetSeleniumDriver:
    def test_direct_driver(self):
        driver = MagicMock()
        driver.get_screenshot_as_png = MagicMock()
        assert streaming._get_selenium_driver(driver) is driver

    def test_dict_driver(self):
        raw = MagicMock()
        raw.get_screenshot_as_png = MagicMock()
        assert streaming._get_selenium_driver({"driver": raw}) is raw

    def test_wrapper_driver(self):
        raw = MagicMock()
        raw.get_screenshot_as_png = MagicMock()
        wrapper = MagicMock()
        wrapper.get_driver = MagicMock(return_value=raw)
        del wrapper.get_screenshot_as_png  # ensure it doesn't match directly
        assert streaming._get_selenium_driver(wrapper) is raw

    def test_none(self):
        assert streaming._get_selenium_driver(None) is None


class TestGetPlaywrightPage:
    def test_dict_page(self):
        page = MagicMock()
        page.screenshot = AsyncMock()
        assert streaming._get_playwright_page({"page": page}) is page

    def test_attribute_page(self):
        """Driver with a non-callable .page attribute holding a page object."""

        class FakePage:
            screenshot = AsyncMock()

        page = FakePage()

        class FakeDriver:
            pass

        driver = FakeDriver()
        driver.page = page
        assert streaming._get_playwright_page(driver) is page

    def test_none(self):
        assert streaming._get_playwright_page(None) is None


# ---------------------------------------------------------------------------
# System functions (async wrappers)
# ---------------------------------------------------------------------------
class TestSystemFunctions:
    @pytest.mark.asyncio
    async def test_start_frame_recording(self):
        result = await streaming._start_frame_recording("test-run")
        assert result == "recording started"
        assert "test-run" in streaming._RECORDING_FLAGS

    @pytest.mark.asyncio
    async def test_stop_frame_recording(self):
        streaming.start_recording("test-run")
        result = await streaming._stop_frame_recording("test-run")
        assert result == "recording stopped"
        assert "test-run" not in streaming._RECORDING_FLAGS

    @pytest.mark.asyncio
    async def test_get_recorded_frames(self):
        streaming.start_recording("test-run")
        with streaming._LOCK:
            streaming._append_recorded_frame_locked("test-run", b"frame-data")
        result = await streaming._get_recorded_frames("test-run")
        assert len(result) == 1
        assert result[0]["seq"] == 0
        assert isinstance(result[0]["data"], str)  # base64 encoded

    @pytest.mark.asyncio
    async def test_ack_recorded_frames(self):
        streaming.start_recording("test-run")
        with streaming._LOCK:
            streaming._append_recorded_frame_locked("test-run", b"f1")
            streaming._append_recorded_frame_locked("test-run", b"f2")
        result = await streaming._ack_recorded_frames("test-run", up_to_seq=0)
        assert result == "frames acknowledged"
        frames = streaming.get_recorded_frames("test-run")
        assert len(frames) == 1

    @pytest.mark.asyncio
    async def test_clear_frame_recording(self):
        streaming.start_recording("test-run")
        result = await streaming._clear_frame_recording("test-run")
        assert result == "recording cleared"
        assert "test-run" not in streaming._RECORDED_FRAMES

    @pytest.mark.asyncio
    async def test_get_recorded_frames_empty_string_params(self):
        """System functions receive strings from JSON; empty strings should be handled."""
        streaming.start_recording("test-run")
        result = await streaming._get_recorded_frames("test-run", since_seq="", limit="")
        assert isinstance(result, list)


# ---------------------------------------------------------------------------
# stop_stream
# ---------------------------------------------------------------------------
class TestStopStream:
    def test_stop_nonexistent(self):
        streaming.stop_stream("nope")  # should not raise

    def test_stop_cleans_up(self):
        streaming.register_driver("r1", MagicMock())
        with streaming._LOCK:
            streaming._LATEST_FRAMES["r1"] = (b"data", time.time())
            stop_event = threading.Event()
            stop_event.set()
            streaming._STREAM_FLAGS["r1"] = stop_event
        streaming.stop_stream("r1")
        assert "r1" not in streaming._STREAM_FLAGS
        assert "r1" not in streaming._LATEST_FRAMES
        assert "r1" not in streaming._CAPTURE_DRIVERS


# ---------------------------------------------------------------------------
# _is_target_closed_error
# ---------------------------------------------------------------------------
class TestIsTargetClosedError:
    def test_matching(self):
        class TargetClosedError(Exception):
            pass

        assert streaming._is_target_closed_error(
            Exception("Target page, context or browser has been closed")
        ) is True

    def test_non_matching(self):
        assert streaming._is_target_closed_error(ValueError("something else")) is False
