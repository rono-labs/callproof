"""Minimal stand-in for inline-snapshot: a snapshot is the literal it wraps, compared by equality."""

from __future__ import annotations


Snapshot = object


def snapshot(value=None):
    return value


def external(*args, **kwargs):
    raise NotImplementedError("external snapshots are not available in this stand-in")
