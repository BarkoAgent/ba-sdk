"""
Drop-in accessibility adapter for ba_ws_sdk.

Owns the boundary between the SDK and the a11y module internals.
ws_core imports bindings from here rather than reaching into agent_func,
so the a11y session lifecycle is never exposed in the FUNCTION_MAP.

Usage (any compatible agent):
    1. Install the a11y package and set A11Y_ENABLED=true
    2. Expose a `driver` dict on your agent module
    The SDK handles the rest automatically via this adapter.
"""
import logging

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


def get_bindings(driver_store: dict):
    """
    Returns accessibility hook bindings for ws_core bulk execution.

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
