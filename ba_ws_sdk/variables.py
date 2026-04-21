# ba_ws_sdk/variables.py
"""
Cross-test variable passing for Barko agents.
Allows dependency tests to export named values that dependent tests can import.
"""

import logging
import threading

_lock = threading.Lock()

# run_id -> {name: value}
_variables: dict = {}

# run_id -> {name: value}  (exported at end of run, consumed by dependent runs)
_exported: dict = {}


def init_run(run_id: str, imported: dict = None) -> None:
    """
    Called at create_driver time. Optionally pre-populates variables
    exported by dependency tests.
    """
    with _lock:
        _variables[run_id] = dict(imported or {})
    logging.info(f"[Variables] Initialized run_id={run_id} with {len(_variables[run_id])} imported vars")


def set_variable(name: str, value: str, _run_test_id: str = "1") -> str:
    """
    Store a named variable for this test run.
    The value can be retrieved within this run via return_variable()
    and is automatically exported for dependent tests to use.
    """
    with _lock:
        if _run_test_id not in _variables:
            _variables[_run_test_id] = {}
        _variables[_run_test_id][name] = value
    logging.info(f"[Variables] set_variable({name!r}) = {value!r} for run_id={_run_test_id}")
    return f"variable '{name}' set to '{value}'"


def return_variable(name: str, _run_test_id: str = "1") -> str:
    """
    Retrieve a named variable previously set in this run or imported
    from a dependency test.
    """
    with _lock:
        store = _variables.get(_run_test_id, {})
        if name not in store:
            raise KeyError(f"Variable '{name}' not found for run_id={_run_test_id}. "
                           f"Available: {list(store.keys())}")
        value = store[name]
    logging.info(f"[Variables] return_variable({name!r}) = {value!r} for run_id={_run_test_id}")
    return value


def export_run(run_id: str) -> dict:
    """
    Export all variables set during a run so dependent tests can import them.
    Called at stop_driver time.
    """
    with _lock:
        exported = dict(_variables.get(run_id, {}))
        _exported[run_id] = exported
    return exported


def get_all(run_id: str) -> dict:
    """Return a copy of all variables currently set for a run."""
    with _lock:
        return dict(_variables.get(run_id, {}))


def cleanup_run(run_id: str) -> None:
    """Remove variable state for a finished run."""
    with _lock:
        _variables.pop(run_id, None)
