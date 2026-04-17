"""
Drop-in accessibility adapter for ba_ws_sdk.

Owns the boundary between the SDK and the a11y module internals.
ws_core imports from here rather than carrying a11y logic itself,
so the a11y session lifecycle stays out of the core message loop.

Usage (any compatible agent):
    1. Install the a11y package and set A11Y_ENABLED=true
    2. Expose a `driver` dict on your agent module
    The SDK handles the rest automatically via this adapter.
"""
import json
import logging
import os
from typing import Any, Callable, NamedTuple, Optional

# A11y functions get a dedicated longer timeout because page scans + evaluators
# can take significantly more time than standard agent commands.
_A11Y_FUNCTION_NAMES = frozenset({"run_accessibility_audit", "run_accessibility_test_case"})
_A11Y_TIMEOUT_S = float(os.getenv("A11Y_COMMAND_TIMEOUT_MS", "120000")) / 1000

_A11Y_AVAILABLE = False
_create_session = None
_append_checkpoint = None
_finalize_session = None

try:
    from a11y.runner import (
        _create_session,
        append_accessibility_audit_checkpoint as _append_checkpoint,
        finalize_accessibility_audit_session as _finalize_session,
    )
    _A11Y_AVAILABLE = True
except ImportError as _import_err:
    logging.warning("a11y module not available in SDK adapter: %s", _import_err)


# ─── Agent module registration ────────────────────────────────────────────────

_AGENT_MODULE = None


def register_agent_module(agent_func) -> None:
    global _AGENT_MODULE
    _AGENT_MODULE = agent_func


def get_bindings(driver_store: dict):
    """
    Returns accessibility hook bindings for bulk execution.

    Returns (bindings_dict, None) on success, or (None, error_msg) on failure.
    The bindings dict contains: driver_store, create_session, append_checkpoint,
    finalize_session — all pointing directly to a11y.runner internals.
    """
    if not _A11Y_AVAILABLE:
        return None, (
            "a11y module is not available. "
            "Install the a11y package and ensure A11Y_ENABLED=true."
        )
    return {
        "driver_store": driver_store,
        "create_session": _create_session,
        "append_checkpoint": _append_checkpoint,
        "finalize_session": _finalize_session,
    }, None


def _get_accessibility_bindings() -> tuple:
    if _AGENT_MODULE is None:
        return None, "Agent module is not registered in SDK runtime."
    driver_store = getattr(_AGENT_MODULE, "driver", None)
    if driver_store is None:
        return None, "Agent module is missing 'driver' state."
    return get_bindings(driver_store)


# ─── Config normalization ─────────────────────────────────────────────────────

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
            policy = "after_navigation"
        else:
            policy = "off"
    if policy not in {"off", "final_only", "after_each_step", "selected_steps", "after_navigation"}:
        policy = "after_navigation" if enabled else "off"

    checkpoint_steps = set()
    for step in config.get("checkpoint_steps", []) or []:
        try:
            checkpoint_steps.add(int(step))
        except (TypeError, ValueError):
            continue

    raw_network_idle = config.get("wait_for_network_idle")
    if raw_network_idle is None:
        network_idle_mode = "navigation_only" if policy in {"after_each_step", "selected_steps", "after_navigation"} else "always"
    else:
        text = str(raw_network_idle).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            network_idle_mode = "always"
        elif text in {"0", "false", "no", "off"}:
            network_idle_mode = "never"
        elif text in {"always", "navigation_only", "never"}:
            network_idle_mode = text
        else:
            network_idle_mode = "always"

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
        "wait_for_network_idle": network_idle_mode,
        "axe_full_scan": "true" if _parse_boolish(config.get("axe_full_scan", config.get("full_scan")), False) else "false",
        "axe_custom_tags": str(config.get("axe_custom_tags", config.get("custom_tags")) or ""),
        "axe_exclude_tags": str(config.get("axe_exclude_tags", config.get("exclude_tags")) or ""),
        "axe_enabled_rules": str(config.get("axe_enabled_rules", config.get("enabled_rules")) or ""),
        "axe_disabled_rules": str(config.get("axe_disabled_rules", config.get("disabled_rules")) or ""),
        "axe_include_iframes": "true" if _parse_boolish(config.get("axe_include_iframes", config.get("include_iframes")), True) else "false",
        "axe_include_selectors": "true" if _parse_boolish(config.get("axe_include_selectors", config.get("include_selectors")), True) else "false",
        "axe_include_ancestry": "true" if _parse_boolish(config.get("axe_include_ancestry", config.get("include_ancestry")), True) else "false",
        "axe_result_types": str(config.get("axe_result_types", config.get("result_types")) or ""),
        "axe_reporter": str(config.get("axe_reporter", config.get("reporter")) or "v2"),
    }


# ─── Bulk-step helpers ────────────────────────────────────────────────────────

_AUDIT_SKIP_FUNCTIONS = frozenset({"create_driver", "stop_driver"})

_NAVIGATION_FUNCTIONS = frozenset({
    "navigate_to_url",
    "refresh_page",
    "change_windows_tabs",
    "change_frame_by_id",
    "change_frame_by_locator",
    "change_frame_to_original",
})


def _bulk_step_label(command: dict, func_name: str, index: int) -> str:
    label = command.get("label") or command.get("name") or command.get("step_label")
    if isinstance(label, str) and label.strip():
        return label.strip()
    return "{} {}".format(func_name.replace("_", " ").strip() or "Step", index)


def _bulk_checkpoint_kind(func_name: str) -> str:
    return "navigation" if func_name in _NAVIGATION_FUNCTIONS else "step"


def _bulk_should_checkpoint_after(policy: str, checkpoint_steps: set, command: dict, step_index: int) -> bool:
    func_name = command.get("function") or ""
    # Setup/teardown functions produce no user-meaningful page state to audit.
    if func_name in _AUDIT_SKIP_FUNCTIONS:
        return False
    if policy == "after_each_step":
        return True
    if policy == "after_navigation":
        return func_name in _NAVIGATION_FUNCTIONS
    if policy == "selected_steps":
        return step_index in checkpoint_steps
    # final_only and off: checkpointing is handled separately at the end of the run.
    return False


def _bulk_should_checkpoint_before_stop(policy: str, checkpoint_steps: set, command: dict, step_index: int) -> bool:
    return command.get("function") == "stop_driver"


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


# ─── Session lifecycle ────────────────────────────────────────────────────────

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
        axe_full_scan=config["axe_full_scan"],
        axe_custom_tags=config["axe_custom_tags"],
        axe_exclude_tags=config["axe_exclude_tags"],
        axe_enabled_rules=config["axe_enabled_rules"],
        axe_disabled_rules=config["axe_disabled_rules"],
        axe_include_iframes=config["axe_include_iframes"],
        axe_include_selectors=config["axe_include_selectors"],
        axe_include_ancestry=config["axe_include_ancestry"],
        axe_result_types=config["axe_result_types"],
        axe_reporter=config["axe_reporter"],
        _run_test_id=run_id,
    )
    return session, None


async def _append_accessibility_checkpoint(
    bindings: dict,
    session: Optional[dict],
    run_id: str,
    config: dict,
    label: str,
    step_index: int,
    checkpoint_kind: str = "step",
):
    session, error = _ensure_accessibility_session(bindings, run_id, config, session)
    if session is None:
        return None, {
            "status": "skipped",
            "reason": error,
            "journey_step_index": step_index,
            "journey_step_label": label,
        }
    checkpoint = await bindings["append_checkpoint"](session, label, step_index, checkpoint_kind=checkpoint_kind)
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


# ─── Core operations injection ────────────────────────────────────────────────

class CoreOps(NamedTuple):
    """ws_core primitives injected into the a11y executor to avoid circular imports."""
    execute_macro_bulk: Callable
    call_maybe_blocking: Callable
    resolve_step_output_vars: Callable
    extract_element_hint: Callable
    format_step_output: Callable
    output_to_str: Callable
    capture_step_frame: Callable


# ─── Accessible bulk executor ─────────────────────────────────────────────────

async def execute_macro_bulk_with_accessibility(
    commands: list,
    FUNCTION_MAP: dict,
    core_ops: CoreOps,
    run_id: str = "1",
    accessibility: Optional[dict] = None,
) -> dict:
    config = _normalize_accessibility_config(accessibility)
    if not config["enabled"]:
        return await core_ops.execute_macro_bulk(commands, FUNCTION_MAP, run_id=run_id)

    bindings, bindings_error = _get_accessibility_bindings()
    if bindings is None:
        bulk_result = await core_ops.execute_macro_bulk(commands, FUNCTION_MAP, run_id=run_id)
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
            checkpoint_kind = _bulk_checkpoint_kind(func_name or "")

            if step_outputs:
                args, kwargs = core_ops.resolve_step_output_vars(args, kwargs, step_outputs)

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

            # Capture URL before the call so we can detect click-triggered navigations.
            pre_call_url = None
            if config["policy"] == "after_navigation" and func_name not in _NAVIGATION_FUNCTIONS:
                try:
                    _pre_page = bindings["driver_store"].get(command_run_id, {}).get("page")
                    if _pre_page is not None:
                        pre_call_url = _pre_page.url
                except Exception:
                    pre_call_url = None

            try:
                if "_run_test_id" not in kwargs:
                    kwargs = dict(kwargs)
                    kwargs["_run_test_id"] = command_run_id

                result = await core_ops.call_maybe_blocking(FUNCTION_MAP[func_name], *args, **kwargs)
                if isinstance(result, dict) and result.get("status") == "error":
                    error_msg = result.get("error", "Unknown error from function")
                    results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "error", "output": error_msg})
                    step_results.append(_build_accessibility_step_result(step_index, step_label, func_name, "error", error=error_msg))
                    session, checkpoint = await _append_accessibility_checkpoint(bindings, session, command_run_id, config, step_label, step_index, checkpoint_kind=checkpoint_kind)
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
                element_hint = core_ops.extract_element_hint(args, kwargs, result)
                rich_output_enabled = os.getenv("BARKO_RICH_BULK_OUTPUT", "1").lower() not in ("0", "false", "no")
                raw_output = result if isinstance(result, (str, dict, list)) else str(result)
                if rich_output_enabled:
                    output = core_ops.format_step_output(func_name, args, kwargs, raw_output)
                else:
                    output = raw_output if isinstance(raw_output, str) else str(raw_output)
                results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "success", "output": output, "raw_output": raw_output})
                step_outputs.append(core_ops.output_to_str(raw_output))
                step_results.append(_build_accessibility_step_result(step_index, step_label, func_name, "success", result=raw_output))
            except Exception as e:
                logging.error(f"[BulkExec] Step {i} ({func_name}) failed: {e}")
                error_msg = str(e)
                results.append({"index": i, "function": func_name, "args": args, "kwargs": kwargs, "status": "error", "output": error_msg})
                step_results.append(_build_accessibility_step_result(step_index, step_label, func_name or "", "error", error=error_msg))
                session, checkpoint = await _append_accessibility_checkpoint(bindings, session, command_run_id, config, step_label, step_index, checkpoint_kind=checkpoint_kind)
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
                    await core_ops.capture_step_frame(
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

            # Detect URL changes caused by clicks or other non-navigation functions.
            url_changed = False
            if pre_call_url is not None:
                try:
                    _post_page = bindings["driver_store"].get(command_run_id, {}).get("page")
                    if _post_page is not None and _post_page.url != pre_call_url:
                        url_changed = True
                except Exception:
                    pass

            if _bulk_should_checkpoint_after(config["policy"], config["checkpoint_steps"], command, step_index) or url_changed:
                session, checkpoint = await _append_accessibility_checkpoint(bindings, session, command_run_id, config, step_label, step_index, checkpoint_kind="navigation" if url_changed else checkpoint_kind)
                if checkpoint and checkpoint.get("status") == "skipped" and session is not None:
                    session["execution_notes"].append(checkpoint["reason"])

        if step_results:
            last_step = step_results[-1]
            last_audited_index = session["journey_steps"][-1]["journey_step_index"] if session and session["journey_steps"] else None
            if last_step["action"] != "stop_driver" and config["policy"] in {"final_only", "after_navigation"} and last_audited_index != last_step["step_index"]:
                session, checkpoint = await _append_accessibility_checkpoint(
                    bindings, session, run_id, config, last_step["label"], last_step["step_index"], checkpoint_kind="step"
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
                    f"[BulkExec] macro completed with open driver ({created_run_id})."
                )
