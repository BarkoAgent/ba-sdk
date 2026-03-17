"""Unit tests for ba_ws_sdk.ws_core helper functions and core logic."""

import json
import struct
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ba_ws_sdk.ws_core import (
    _build_uri,
    _extract_element_hint,
    _format_locator,
    _format_step_output,
    _is_sensitive_field,
    _make_envelope,
    _preview_value,
    build_function_map,
    build_system_functions,
    call_maybe_blocking,
    execute_macro_bulk,
    handle_message,
)


# ---------------------------------------------------------------------------
# _is_sensitive_field
# ---------------------------------------------------------------------------
class TestIsSensitiveField:
    def test_password_variants(self):
        assert _is_sensitive_field("password") is True
        assert _is_sensitive_field("PASSWORD") is True
        assert _is_sensitive_field("user_password") is True
        assert _is_sensitive_field("passwd") is True

    def test_token_and_secret(self):
        assert _is_sensitive_field("auth_token") is True
        assert _is_sensitive_field("SECRET_KEY") is True
        assert _is_sensitive_field("api_key") is True
        assert _is_sensitive_field("apikey") is True

    def test_non_sensitive(self):
        assert _is_sensitive_field("username") is False
        assert _is_sensitive_field("email") is False
        assert _is_sensitive_field("locator") is False


# ---------------------------------------------------------------------------
# _preview_value
# ---------------------------------------------------------------------------
class TestPreviewValue:
    def test_short_value(self):
        assert _preview_value("hello") == "hello"

    def test_long_value_truncated(self):
        long_str = "x" * 100
        result = _preview_value(long_str, max_len=10)
        assert result == "x" * 10 + "..."

    def test_non_string(self):
        assert _preview_value(42) == "42"

    def test_exact_boundary(self):
        assert _preview_value("abcde", max_len=5) == "abcde"
        assert _preview_value("abcdef", max_len=5) == "abcde..."


# ---------------------------------------------------------------------------
# _format_locator
# ---------------------------------------------------------------------------
class TestFormatLocator:
    def test_with_type_and_locator(self):
        assert _format_locator({"locator_type": "xpath", "locator": "//div"}) == "xpath://div"

    def test_locator_only(self):
        assert _format_locator({"locator": "#my-id"}) == "#my-id"

    def test_empty(self):
        assert _format_locator({}) == ""

    def test_locator_type_without_locator(self):
        assert _format_locator({"locator_type": "css"}) == ""


# ---------------------------------------------------------------------------
# _extract_element_hint
# ---------------------------------------------------------------------------
class TestExtractElementHint:
    def test_kwargs_locator_type_and_locator(self):
        hint = _extract_element_hint([], {"locator_type": "id", "locator": "my-btn"}, None)
        assert hint == {"locator_type": "id", "locator": "my-btn"}

    def test_kwargs_locator_only(self):
        hint = _extract_element_hint([], {"locator": "#container"}, None)
        assert hint == {"locator": "#container"}

    def test_positional_type_value(self):
        hint = _extract_element_hint(["id", "submit-btn"], {}, None)
        assert hint["locator_type"] == "id"
        assert hint["locator"] == "submit-btn"

    def test_positional_xpath_string(self):
        hint = _extract_element_hint(["//div[@class='x']"], {}, None)
        assert hint["locator"] == "//div[@class='x']"

    def test_bounding_box_from_result(self):
        result = {"x": 10, "y": 20, "width": 100, "height": 50}
        hint = _extract_element_hint([], {}, result)
        assert hint["bounding_box"] == result

    def test_nested_bounding_box(self):
        result = {"bounding_box": {"x": 5, "y": 5, "width": 50, "height": 50}}
        hint = _extract_element_hint([], {}, result)
        assert hint["bounding_box"] == {"x": 5, "y": 5, "width": 50, "height": 50}

    def test_returns_none_for_empty(self):
        assert _extract_element_hint([], {}, None) is None

    def test_nested_dict_locator_in_kwargs(self):
        hint = _extract_element_hint([], {"locator": {"type": "css", "selector": ".btn"}}, None)
        assert hint is not None
        assert hint["locator"] == ".btn"

    def test_dict_arg_with_locator(self):
        hint = _extract_element_hint([{"locator": "//input"}], {}, None)
        assert hint["locator"] == "//input"

    def test_url_not_treated_as_locator(self):
        hint = _extract_element_hint(["https://example.com"], {}, None)
        assert hint is None


# ---------------------------------------------------------------------------
# _format_step_output
# ---------------------------------------------------------------------------
class TestFormatStepOutput:
    def test_create_driver(self):
        assert _format_step_output("create_driver", [], {}, None) == "create_driver"

    def test_stop_driver(self):
        assert _format_step_output("stop_driver", [], {}, None) == "stop_driver"

    def test_navigate_to_url_kwargs(self):
        out = _format_step_output("navigate_to_url", [], {"url": "https://example.com"}, None)
        assert "navigate_to_url" in out
        assert "https://example.com" in out

    def test_navigate_to_url_args(self):
        out = _format_step_output("navigate_to_url", ["https://test.com"], {}, None)
        assert "https://test.com" in out

    def test_click_with_locator(self):
        out = _format_step_output("click", [], {"locator_type": "css", "locator": ".btn"}, None)
        assert "click" in out
        assert "css:.btn" in out

    def test_send_keys_redacts_password(self):
        out = _format_step_output(
            "send_keys", [], {"locator": "#pass", "key": "password", "value": "s3cret"}, None
        )
        assert "<redacted>" in out
        assert "s3cret" not in out

    def test_send_keys_shows_value(self):
        out = _format_step_output("send_keys", [], {"locator": "#user", "key": "user", "value": "alice"}, None)
        assert "alice" in out

    def test_get_page_html(self):
        html = "<html><body>hi</body></html>"
        out = _format_step_output("get_page_html", [], {}, html)
        assert "captured chars=" in out

    def test_exists_with_locator(self):
        out = _format_step_output("exists", [], {"locator": "#el"}, None)
        assert "exists" in out

    def test_generic_function(self):
        out = _format_step_output("some_func", [], {}, "done")
        assert "some_func" in out
        assert "done" in out

    def test_generic_function_success_result(self):
        out = _format_step_output("some_func", [], {}, "success")
        assert out == "some_func"


# ---------------------------------------------------------------------------
# _build_uri
# ---------------------------------------------------------------------------
class TestBuildUri:
    def test_full_ws_uri_passed_through(self):
        assert _build_uri("ws://localhost:8080/ws") == "ws://localhost:8080/ws"

    def test_wss_uri_passed_through(self):
        assert _build_uri("wss://example.com/ws") == "wss://example.com/ws"

    def test_id_uses_default_base(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_WS_BASE", "ws://localhost:9000/ws/")
        assert _build_uri("my-agent-id") == "ws://localhost:9000/ws/my-agent-id"

    def test_id_strips_slashes(self, monkeypatch):
        monkeypatch.setenv("DEFAULT_WS_BASE", "ws://localhost:9000/ws/")
        assert _build_uri("/my-id") == "ws://localhost:9000/ws/my-id"


# ---------------------------------------------------------------------------
# _make_envelope
# ---------------------------------------------------------------------------
class TestMakeEnvelope:
    def test_structure(self):
        header = {"id": "run1", "type": "screenshot", "seq": 0}
        payload = b"\x89PNG fake data"
        envelope = _make_envelope(header, payload)

        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        expected_len = len(header_json)

        # First 4 bytes = big-endian uint32 header length
        stored_len = struct.unpack(">I", envelope[:4])[0]
        assert stored_len == expected_len

        # Next bytes = header JSON
        assert envelope[4 : 4 + expected_len] == header_json

        # Remaining = payload
        assert envelope[4 + expected_len :] == payload


# ---------------------------------------------------------------------------
# build_function_map
# ---------------------------------------------------------------------------
class TestBuildFunctionMap:
    def test_collects_public_functions(self):
        module = MagicMock()
        module.public_func = lambda: None
        module._private_func = lambda: None
        module.some_var = "not a function"

        import types

        # inspect.getmembers needs real functions
        module.public_func = types.FunctionType(
            compile("pass", "<test>", "exec"), {}, "public_func"
        )
        module._private_func = types.FunctionType(
            compile("pass", "<test>", "exec"), {}, "_private_func"
        )

        fmap = build_function_map(module)
        assert "public_func" in fmap
        assert "_private_func" not in fmap


# ---------------------------------------------------------------------------
# build_system_functions
# ---------------------------------------------------------------------------
class TestBuildSystemFunctions:
    def test_keys(self):
        sys_funcs = build_system_functions()
        expected_keys = {
            "_start_frame_recording",
            "_stop_frame_recording",
            "_get_recorded_frames",
            "_ack_recorded_frames",
            "_clear_frame_recording",
        }
        assert set(sys_funcs.keys()) == expected_keys


# ---------------------------------------------------------------------------
# call_maybe_blocking
# ---------------------------------------------------------------------------
class TestCallMaybeBlocking:
    @pytest.mark.asyncio
    async def test_async_function(self):
        async def afunc(x):
            return x * 2

        result = await call_maybe_blocking(afunc, 5)
        assert result == 10

    @pytest.mark.asyncio
    async def test_sync_function(self):
        def sfunc(x):
            return x + 1

        result = await call_maybe_blocking(sfunc, 5)
        assert result == 6


# ---------------------------------------------------------------------------
# execute_macro_bulk
# ---------------------------------------------------------------------------
class TestExecuteMacroBulk:
    @pytest.mark.asyncio
    async def test_success(self):
        func_map = {"my_func": lambda **kwargs: "ok"}
        commands = [
            {"function": "my_func", "kwargs": {}},
            {"function": "my_func", "kwargs": {}},
        ]
        with patch("ba_ws_sdk.ws_core.streaming") as mock_streaming:
            mock_streaming.capture_step_frame_async = AsyncMock()
            result = await execute_macro_bulk(commands, func_map, run_id="test-1")

        assert result["status"] == "success"
        assert result["executed_lines"] == 2

    @pytest.mark.asyncio
    async def test_unknown_function(self):
        result = await execute_macro_bulk(
            [{"function": "nonexistent", "kwargs": {}}], {}, run_id="test-1"
        )
        assert result["status"] == "error"
        assert result["failed_index"] == 0
        assert "Unknown function" in result["error_details"]

    @pytest.mark.asyncio
    async def test_function_raises(self):
        def failing(**kwargs):
            raise ValueError("boom")

        func_map = {"fail_func": failing}
        commands = [{"function": "fail_func", "kwargs": {}}]

        with patch("ba_ws_sdk.ws_core.streaming") as mock_streaming:
            mock_streaming.capture_step_frame_async = AsyncMock()
            result = await execute_macro_bulk(commands, func_map, run_id="test-1")

        assert result["status"] == "error"
        assert "boom" in result["error_details"]

    @pytest.mark.asyncio
    async def test_error_dict_result(self):
        def err_func(**kwargs):
            return {"status": "error", "error": "element not found"}

        func_map = {"err_func": err_func}
        commands = [{"function": "err_func", "kwargs": {}}]

        with patch("ba_ws_sdk.ws_core.streaming") as mock_streaming:
            mock_streaming.capture_step_frame_async = AsyncMock()
            result = await execute_macro_bulk(commands, func_map, run_id="test-1")

        assert result["status"] == "error"
        assert result["failed_index"] == 0

    @pytest.mark.asyncio
    async def test_halts_on_first_error(self):
        call_count = 0

        def counting(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("stop here")
            return "ok"

        func_map = {"counter": counting}
        commands = [
            {"function": "counter", "kwargs": {}},
            {"function": "counter", "kwargs": {}},
            {"function": "counter", "kwargs": {}},
        ]

        with patch("ba_ws_sdk.ws_core.streaming") as mock_streaming:
            mock_streaming.capture_step_frame_async = AsyncMock()
            result = await execute_macro_bulk(commands, func_map, run_id="test-1")

        assert result["status"] == "error"
        assert result["executed_lines"] == 1
        assert result["failed_index"] == 1


# ---------------------------------------------------------------------------
# handle_message
# ---------------------------------------------------------------------------
class TestHandleMessage:
    @pytest.mark.asyncio
    async def test_missing_id_returns_error(self):
        msg = json.dumps({"function": "foo"})
        resp = await handle_message(msg, {}, {})
        data = json.loads(resp)
        assert data["status"] == "error"
        assert "Missing" in data["error"]

    @pytest.mark.asyncio
    async def test_unknown_function(self):
        msg = json.dumps({"id": "r1", "function": "nope"})
        resp = await handle_message(msg, {}, {})
        data = json.loads(resp)
        assert data["status"] == "error"
        assert data["id"] == "r1"

    @pytest.mark.asyncio
    async def test_known_function_called(self):
        def greet(name="world", **kwargs):
            return f"hello {name}"

        fmap = {"greet": greet}
        msg = json.dumps({"id": "r1", "function": "greet", "kwargs": {"name": "alice"}})
        resp = await handle_message(msg, fmap, {})
        data = json.loads(resp)
        assert data["status"] == "success"
        assert data["result"] == "hello alice"

    @pytest.mark.asyncio
    async def test_list_available_methods(self):
        def my_func(a, b):
            pass

        fmap = {"my_func": my_func}
        sys_funcs = {"_sys": lambda: None}
        msg = json.dumps({"id": "r1", "function": "list_available_methods"})
        resp = await handle_message(msg, fmap, sys_funcs)
        data = json.loads(resp)
        assert data["status"] == "success"
        assert any(m["name"] == "my_func" for m in data["methods"])
        assert any(m["name"] == "_sys" for m in data["system_methods"])

    @pytest.mark.asyncio
    async def test_execute_macro_bulk_route(self):
        def noop(**kwargs):
            return "ok"

        fmap = {"noop": noop}
        commands = [{"function": "noop", "kwargs": {}}]
        msg = json.dumps({
            "id": "r1",
            "function": "execute_macro_bulk",
            "args": [commands],
            "kwargs": {"_run_test_id": "r1"},
        })

        with patch("ba_ws_sdk.ws_core.streaming") as mock_streaming:
            mock_streaming.capture_step_frame_async = AsyncMock()
            resp = await handle_message(msg, fmap, {})

        data = json.loads(resp)
        assert data["status"] == "success"
        assert data["executed_lines"] == 1

    @pytest.mark.asyncio
    async def test_system_function_called(self):
        async def sys_fn(_run_test_id="1"):
            return "sys ok"

        sys_funcs = {"_my_sys": sys_fn}
        msg = json.dumps({"id": "r1", "function": "_my_sys"})
        resp = await handle_message(msg, {}, sys_funcs)
        data = json.loads(resp)
        assert data["status"] == "success"
        assert data["result"] == "sys ok"

    @pytest.mark.asyncio
    async def test_invalid_json(self):
        resp = await handle_message("not json", {}, {})
        data = json.loads(resp)
        assert data["status"] == "error"

    @pytest.mark.asyncio
    async def test_run_test_id_from_kwargs_fallback(self):
        """When envelope 'id' is missing, _run_test_id from kwargs is used."""
        def echo(**kwargs):
            return kwargs.get("_run_test_id")

        fmap = {"echo": echo}
        msg = json.dumps({
            "function": "echo",
            "kwargs": {"_run_test_id": "custom-id"},
        })
        resp = await handle_message(msg, fmap, {})
        data = json.loads(resp)
        assert data["id"] == "custom-id"
        assert data["result"] == "custom-id"
