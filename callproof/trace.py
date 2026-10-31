"""Pytest plugin recording repo-internal call edges while selected tests run.

Usage: ``pytest -p callproof.trace --callproof-root REPO --callproof-out edges.json``.
Only the call phase of each test is traced, and only calls whose caller and callee frames both
live under the repository root are kept.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

_TOOL = sys.monitoring.PROFILER_ID
_SYNTHETIC = ("<lambda>", "<genexpr>", "<listcomp>", "<dictcomp>", "<setcomp>")
_EXCLUDED_PARTS = {"site-packages", "dist-packages", ".venv", "venv", "__pycache__"}
_TEST_NAMES = {"tests", "test", "testing"}


def collapse_qualname(qualname: str) -> tuple[str, bool]:
    """Fold nested defs, lambdas and comprehensions into the nearest enclosing named def or class."""
    parts = qualname.split(".")
    kept: list[str] = []
    for part in parts:
        if part == "<locals>" or part in _SYNTHETIC:
            break
        kept.append(part)
    collapsed = ".".join(kept) or "<module>"
    return collapsed, collapsed != qualname


def is_test_path(path: str) -> bool:
    pure = Path(path)
    name = pure.name
    return (
        name == "conftest.py"
        or name.startswith("test_")
        or name.endswith("_test.py")
        or bool(_TEST_NAMES.intersection(pure.parts[:-1]))
    )


class CallTracer:
    def __init__(self, root: Path) -> None:
        self.root = str(root.resolve()) + "/"
        self.active = False
        self.edges: Counter[tuple[str, str, str, str, bool]] = Counter()
        self.executed: dict[tuple[str, str], bool] = {}
        self._codes: dict[object, tuple[str, str, bool] | None] = {}

    def identify(self, code) -> tuple[str, str, bool] | None:
        try:
            return self._codes[code]
        except KeyError:
            pass
        filename = code.co_filename
        result = None
        if filename.startswith(self.root):
            rel = filename[len(self.root):]
            if not _EXCLUDED_PARTS.intersection(rel.split("/")) and rel.endswith(".py"):
                qualname, collapsed = collapse_qualname(code.co_qualname)
                result = (rel, qualname, collapsed)
        self._codes[code] = result
        return result

    def _on_enter(self, code, offset, *_):
        if self.identify(code) is None:
            return sys.monitoring.DISABLE
        if not self.active:
            return None
        callee = self.identify(code)
        self.executed[callee[:2]] = callee[2] or self.executed.get(callee[:2], False)
        caller_frame = sys._getframe(1).f_back
        caller = self.identify(caller_frame.f_code) if caller_frame else None
        if caller is not None:
            collapsed = caller[2] or callee[2]
            if not (collapsed and caller[:2] == callee[:2]):
                self.edges[(*caller[:2], *callee[:2], collapsed)] += 1
        return None

    def start(self) -> None:
        sys.monitoring.use_tool_id(_TOOL, "callproof")
        events = sys.monitoring.events
        for event in (events.PY_START, events.PY_RESUME):
            sys.monitoring.register_callback(_TOOL, event, self._on_enter)
        sys.monitoring.set_events(_TOOL, events.PY_START | events.PY_RESUME)

    def stop(self) -> None:
        sys.monitoring.set_events(_TOOL, 0)
        sys.monitoring.free_tool_id(_TOOL)

    def dump(self) -> dict:
        edges = [
            [cp, cq, ep, eq, count, is_test_path(cp) or is_test_path(ep), collapsed]
            for (cp, cq, ep, eq, collapsed), count in sorted(self.edges.items())
        ]
        executed = [[p, q, flag] for (p, q), flag in sorted(self.executed.items())]
        return {"edges": edges, "executed": executed}


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("callproof")
    group.addoption("--callproof-root", help="repository root whose code is traced")
    group.addoption("--callproof-out", help="where to write the recorded edges (JSON)")


def pytest_configure(config: pytest.Config) -> None:
    root, out = config.getoption("--callproof-root"), config.getoption("--callproof-out")
    if root and out:
        config._callproof = CallTracer(Path(root))
        config._callproof_out = Path(out)
        config._callproof.start()


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item):
    tracer = getattr(item.config, "_callproof", None)
    if tracer is None:
        return (yield)
    tracer.active = True
    try:
        return (yield)
    finally:
        tracer.active = False


def pytest_unconfigure(config: pytest.Config) -> None:
    tracer = getattr(config, "_callproof", None)
    if tracer is not None:
        tracer.stop()
        config._callproof_out.write_text(json.dumps(tracer.dump()))
