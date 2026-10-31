"""Pytest plugin writing one outcome per test node: passed, failed, error or skipped."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.getgroup("callproof").addoption("--outcomes-out", help="where to write node outcomes (JSON)")


def pytest_configure(config: pytest.Config) -> None:
    config._outcomes = {}


def _invocation_relative(config: pytest.Config, nodeid: str) -> str:
    file, sep, rest = nodeid.partition("::")
    absolute = config.rootpath / file
    return os.path.relpath(absolute, config.invocation_params.dir) + sep + rest


def pytest_runtest_logreport(report: pytest.TestReport) -> None:
    outcomes = _current.outcomes
    report = _Relocated(report, _current.config)
    if report.when == "call":
        outcomes[report.nodeid] = report.outcome
    elif report.when == "setup" and report.outcome != "passed":
        outcomes[report.nodeid] = "skipped" if report.outcome == "skipped" else "error"
    elif report.when == "teardown" and report.outcome == "failed" and outcomes.get(report.nodeid) == "passed":
        outcomes[report.nodeid] = "error"


class _Relocated:
    def __init__(self, report: pytest.TestReport, config: pytest.Config) -> None:
        self.when, self.outcome = report.when, report.outcome
        self.nodeid = _invocation_relative(config, report.nodeid)


class _Current:
    outcomes: dict[str, str] = {}
    config: pytest.Config


_current = _Current()


def pytest_sessionstart(session: pytest.Session) -> None:
    _current.outcomes = session.config._outcomes
    _current.config = session.config


def pytest_sessionfinish(session: pytest.Session) -> None:
    out = session.config.getoption("--outcomes-out")
    if out:
        Path(out).write_text(json.dumps(session.config._outcomes))
