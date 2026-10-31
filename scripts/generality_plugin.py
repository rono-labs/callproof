"""Pytest plugin for the generality experiment: deterministic test sampling and per-test edge attribution.

Loaded next to ``callproof.trace``. ``GENERALITY_MAX_TESTS`` caps the number of tests (evenly spaced in
collection order) and ``GENERALITY_PER_TEST_OUT`` names the JSON file receiving, per test id, the
traced edges that test contributed.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

_per_test: dict[str, list[list]] = {}


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    cap = int(os.environ.get("GENERALITY_MAX_TESTS", "0"))
    if cap and len(items) > cap:
        keep = {(i * len(items)) // cap for i in range(cap)}
        dropped = [item for i, item in enumerate(items) if i not in keep]
        config.hook.pytest_deselected(items=dropped)
        items[:] = [item for i, item in enumerate(items) if i in keep]


@pytest.hookimpl(wrapper=True)
def pytest_runtest_call(item: pytest.Item):
    tracer = getattr(item.config, "_callproof", None)
    before = dict(tracer.edges) if tracer is not None else {}
    try:
        return (yield)
    finally:
        if tracer is not None:
            _per_test[item.nodeid] = [
                list(key) for key, count in tracer.edges.items() if before.get(key) != count
            ]


def pytest_unconfigure(config: pytest.Config) -> None:
    out = os.environ.get("GENERALITY_PER_TEST_OUT")
    if out:
        Path(out).write_text(json.dumps(_per_test))
