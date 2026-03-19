"""
improvements.py — Self-improvement support for Barko agents.

When ENABLE_IMPROVEMENTS=true, the SDK injects three extra functions into
any agent's FUNCTION_MAP:

    get_agent_source()            — read agent_func.py so the agent understands
                                    current behaviour before writing new code
    add_improvement(name, code)   — define a new function, persisted to
                                    extended_functions.py
    list_improvements()           — list all functions added via improvements

New functions are persisted to ``extended_functions.py`` (next to
``agent_func.py``) and are executed inside the agent_func module's own
namespace, so they share all its globals (driver, test_variables, etc.)
exactly like any native function would.

Core functions (those present when the process started) can never be
overwritten.
"""

import ast
import json
import logging
import os

logger = logging.getLogger(__name__)

EXTENDED_FILE_NAME = "extended_functions.py"
REGISTRY_FILE_NAME = "improvements_registry.json"


class ImprovementsManager:
    """
    Manages dynamically added agent functions stored in ``extended_functions.py``.

    Functions are exec'd into the agent_func module's namespace so they share
    all its globals (driver, test_variables, imports, helpers, etc.).

    Usage (done automatically by the SDK when ENABLE_IMPROVEMENTS is set):

        manager = ImprovementsManager(function_map, agent_func_module)
        manager.load()
        function_map["get_agent_source"]  = manager.get_agent_source
        function_map["add_improvement"]   = manager.add_improvement
        function_map["list_improvements"] = manager.list_improvements
    """

    def __init__(self, function_map: dict, agent_func_module):
        # Snapshot of core function names — protected for the lifetime of the process.
        self._core_functions: set = set(function_map.keys())
        # Live reference; mutated to register new functions.
        self._function_map: dict = function_map
        # The actual agent_func module — new functions are exec'd into its namespace.
        self._agent_module = agent_func_module

        agent_func_dir = os.path.dirname(os.path.abspath(agent_func_module.__file__))
        self._agent_func_path: str = os.path.abspath(agent_func_module.__file__)
        self._extended_path: str = os.path.join(agent_func_dir, EXTENDED_FILE_NAME)
        self._registry_path: str = os.path.join(agent_func_dir, REGISTRY_FILE_NAME)
        self._agent_dir: str = os.path.realpath(agent_func_dir)
        self._allow_insecure: bool = os.getenv("ALLOW_INSECURE", "false").lower() in ("1", "true", "yes")

        if self._allow_insecure:
            logger.warning(
                "[Improvements] ALLOW_INSECURE is enabled — security checks are disabled. "
                "Improvement functions may use any import or system call."
            )
        else:
            # Inject a restricted open into the module namespace so all improvement
            # functions (and any future ones) can only write inside the agent folder.
            self._agent_module.__dict__["open"] = self._make_restricted_open()

    # ------------------------------------------------------------------
    # Public API (called by the SDK at startup)
    # ------------------------------------------------------------------

    def load(self) -> None:
        """
        Load extended_functions.py (if it exists) by exec-ing it into the
        agent_func module's namespace, then register all new public functions.
        """
        if not os.path.exists(self._extended_path):
            logger.info("[Improvements] No %s found — starting fresh.", EXTENDED_FILE_NAME)
            return

        try:
            with open(self._extended_path, "r", encoding="utf-8") as fh:
                source = fh.read()

            exec(compile(source, self._extended_path, "exec"), self._agent_module.__dict__)  # noqa: S102

            loaded = 0
            import inspect
            for name, obj in inspect.getmembers(
                self._agent_module,
                lambda o: inspect.isfunction(o) or inspect.iscoroutinefunction(o),
            ):
                if name.startswith("_") or name in self._core_functions:
                    continue
                if name not in self._function_map:
                    self._function_map[name] = obj
                    loaded += 1

            logger.info(
                "[Improvements] Loaded %d extended function(s) from %s.",
                loaded,
                self._extended_path,
            )
        except Exception as exc:
            logger.error("[Improvements] Failed to load %s: %s", EXTENDED_FILE_NAME, exc)

    # ------------------------------------------------------------------
    # Agent-facing callables (injected into FUNCTION_MAP)
    # ------------------------------------------------------------------

    def get_agent_source(self, _run_test_id: str = "1") -> str:
        """
        Return the full source code of agent_func.py.

        Always call this before add_improvement so you understand the existing
        functions, coding style, global variables, and imports available.

        Usage:
            get_agent_source({})
        """
        try:
            with open(self._agent_func_path, "r", encoding="utf-8") as fh:
                return fh.read()
        except Exception as exc:
            return f"Error: Could not read agent source: {exc}"

    def add_improvement(self, name: str, code: str, _run_test_id: str = "1") -> str:
        """
        Add a new function to this agent at runtime.

        The function is persisted to extended_functions.py and executed inside
        the agent_func module's namespace, giving it access to all globals
        (driver, test_variables, imports, helpers, etc.).

        Before calling this, ALWAYS, ALWAYS call get_agent_source() first to read and
        understand the existing behaviour, patterns, and available globals!!

        Core functions that existed at startup can never be overwritten.

        Usage:
            add_improvement({'name': 'my_func', 'code': 'def my_func(x, _run_test_id=\"1\"):\\n    return x'})

        Parameters:
            name: The exact top-level function name defined in the code.
            code: Valid Python source containing ``def <name>(..., _run_test_id='1')``.
        """
        # --- Guard: never overwrite a core function ---
        if name in self._core_functions:
            return f"Error: '{name}' is a core function and cannot be overwritten."

        # --- Syntax check ---
        try:
            tree = ast.parse(code)
        except SyntaxError as exc:
            return f"Error: Syntax error in provided code: {exc}"

        # --- Security scan (skipped when ALLOW_INSECURE=true) ---
        if not self._allow_insecure:
            danger = self._check_dangerous_patterns(tree)
            if danger:
                return f"Error: Forbidden pattern in code — {danger}"

        # --- Confirm a top-level def with the expected name exists ---
        top_level_funcs = [
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.col_offset == 0
        ]
        if name not in top_level_funcs:
            return (
                f"Error: Code does not define a top-level function named '{name}'. "
                f"Found: {top_level_funcs or 'none'}."
            )

        # --- Enforce _run_test_id parameter ---
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == name
                and node.col_offset == 0
            ):
                param_names = (
                    [arg.arg for arg in node.args.args]
                    + [arg.arg for arg in node.args.kwonlyargs]
                )
                if "_run_test_id" not in param_names:
                    return (
                        f"Error: Function '{name}' must include '_run_test_id' as a parameter "
                        f"(e.g. `def {name}(..., _run_test_id='1')`)."
                    )
                break

        # --- Execute inside the agent_func module namespace ---
        # This gives the new function access to all existing globals:
        # driver, test_variables, imports, helper functions, etc.
        try:
            exec(compile(tree, "<improvements>", "exec"), self._agent_module.__dict__)  # noqa: S102
        except Exception as exc:
            return f"Error: Failed to execute function code: {exc}"

        func_obj = self._agent_module.__dict__.get(name)
        if not callable(func_obj):
            return f"Error: '{name}' did not produce a callable after execution."

        # --- Persist to extended_functions.py ---
        try:
            self._persist(name, code)
        except Exception as exc:
            return f"Error: Failed to persist function to {EXTENDED_FILE_NAME}: {exc}"

        # --- Update registry and live function map ---
        try:
            self._register(name)
        except Exception as exc:
            logger.warning("[Improvements] Could not update registry for '%s': %s", name, exc)

        self._function_map[name] = func_obj
        logger.info("[Improvements] Registered function '%s' in %s.", name, EXTENDED_FILE_NAME)
        return f"Success: Function '{name}' added and persisted to {EXTENDED_FILE_NAME}."

    def list_improvements(self, _run_test_id: str = "1") -> dict:
        """
        List all functions that were added via add_improvement.

        Returns a dict with keys:
            improvements: {name: docstring}
            count: int
        """
        registry = self._read_registry()
        improvements = {}
        for fname in registry:
            func = self._function_map.get(fname)
            improvements[fname] = (func.__doc__ or "").strip() if func else "(not loaded)"
        return {"improvements": improvements, "count": len(improvements)}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _check_dangerous_patterns(tree: ast.AST) -> str:
        """
        Walk the AST and return a human-readable error string if any forbidden
        pattern is found, or an empty string if the code is clean.

        Blocked patterns
        ----------------
        Imports   : subprocess, pty, ctypes, signal, multiprocessing, socket,
                    importlib (dynamic import abuse)
        os calls  : os.system, os.popen, os.exec*, os.spawn*, os.fork,
                    os.kill, os.killpg
        Builtins  : eval(), exec(), __import__(), compile()
        """
        _BLOCKED_MODULES = frozenset({
            "subprocess", "pty", "ctypes", "signal",
            "multiprocessing", "socket", "importlib",
        })
        _BLOCKED_OS_ATTRS = frozenset({
            "system", "popen", "execv", "execve", "execvp", "execvpe",
            "spawnl", "spawnle", "spawnlp", "spawnlpe",
            "spawnv", "spawnve", "spawnvp", "spawnvpe",
            "fork", "forkpty", "kill", "killpg", "popen2",
        })
        _BLOCKED_BUILTINS = frozenset({"eval", "exec", "__import__", "compile"})

        for node in ast.walk(tree):
            # import subprocess / import pty / etc.
            if isinstance(node, ast.Import):
                for alias in node.names:
                    root = alias.name.split(".")[0]
                    if root in _BLOCKED_MODULES:
                        return f"'import {alias.name}' is not allowed."

            # from subprocess import ... / from os import system / etc.
            if isinstance(node, ast.ImportFrom):
                root = (node.module or "").split(".")[0]
                if root in _BLOCKED_MODULES:
                    return f"'from {node.module} import ...' is not allowed."
                if root == "os":
                    for alias in node.names:
                        if alias.name in _BLOCKED_OS_ATTRS:
                            return f"'from os import {alias.name}' is not allowed."

            # Direct calls: eval(...), exec(...), __import__(...), compile(...)
            if isinstance(node, ast.Call):
                # eval() / exec() / __import__() / compile() as bare names
                if isinstance(node.func, ast.Name) and node.func.id in _BLOCKED_BUILTINS:
                    return f"'{node.func.id}(...)' is not allowed."
                # os.system(...) / os.popen(...) / os.fork() / etc.
                if (
                    isinstance(node.func, ast.Attribute)
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "os"
                    and node.func.attr in _BLOCKED_OS_ATTRS
                ):
                    return f"'os.{node.func.attr}(...)' is not allowed."

        return ""

    def _make_restricted_open(self):
        """
        Return an open() replacement that blocks write operations targeting
        paths outside the agent directory.

        Read-only operations are never restricted.
        Write modes: any mode string containing 'w', 'a', 'x', or '+'.
        """
        import builtins
        _real_open = builtins.open
        agent_dir = self._agent_dir

        def _restricted_open(file, mode="r", *args, **kwargs):
            _WRITE_CHARS = frozenset("wax+")
            if _WRITE_CHARS.intersection(str(mode)):
                resolved = os.path.realpath(os.path.abspath(str(file)))
                if not (resolved == agent_dir or resolved.startswith(agent_dir + os.sep)):
                    raise PermissionError(
                        f"Improvement functions may only write files inside the agent "
                        f"directory ('{agent_dir}'). Blocked path: '{file}'"
                    )
            return _real_open(file, mode, *args, **kwargs)

        return _restricted_open

    def _persist(self, name: str, code: str) -> None:
        """Append (or replace) a function definition in extended_functions.py."""
        existing = ""
        if os.path.exists(self._extended_path):
            with open(self._extended_path, "r", encoding="utf-8") as fh:
                existing = fh.read()

        existing = self._remove_function_def(existing, name)
        separator = "\n\n" if existing.strip() else ""
        new_content = existing.rstrip() + separator + code.strip() + "\n"

        with open(self._extended_path, "w", encoding="utf-8") as fh:
            fh.write(new_content)

    def _read_registry(self) -> list:
        if not os.path.exists(self._registry_path):
            return []
        try:
            with open(self._registry_path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            return data if isinstance(data, list) else []
        except Exception:
            return []

    def _register(self, name: str) -> None:
        registry = self._read_registry()
        if name not in registry:
            registry.append(name)
        with open(self._registry_path, "w", encoding="utf-8") as fh:
            json.dump(registry, fh, indent=2)

    @staticmethod
    def _remove_function_def(source: str, func_name: str) -> str:
        """
        Remove a top-level function definition by name from Python source.
        Uses AST end_lineno (Python 3.8+) for accurate removal.
        Returns source unchanged if the function is not found.
        """
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return source

        lines = source.splitlines(keepends=True)
        for node in ast.walk(tree):
            if (
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                and node.name == func_name
                and node.col_offset == 0
            ):
                start = node.lineno - 1      # inclusive, 0-indexed
                end = node.end_lineno        # 1-indexed inclusive → 0-indexed exclusive
                while end < len(lines) and lines[end].strip() == "":
                    end += 1
                return "".join(lines[:start] + lines[end:])
        return source
