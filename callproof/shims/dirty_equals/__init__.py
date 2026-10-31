"""Minimal stand-in for dirty-equals covering the matchers used by the benchmark repositories."""

from __future__ import annotations

import re
from typing import Any


class DirtyEquals:
    def equals(self, other: Any) -> bool:
        raise NotImplementedError

    def __eq__(self, other: Any) -> bool:
        return self.equals(other)

    def __ne__(self, other: Any) -> bool:
        return not self.equals(other)

    __hash__ = None

    def __or__(self, other: Any) -> DirtyEquals:
        return _Combined(self, other, any)

    def __and__(self, other: Any) -> DirtyEquals:
        return _Combined(self, other, all)

    def __repr__(self) -> str:
        return type(self).__name__ + "()"


class _Combined(DirtyEquals):
    def __init__(self, left: Any, right: Any, combine) -> None:
        self.parts, self.combine = (left, right), combine

    def equals(self, other: Any) -> bool:
        return self.combine(part == other for part in self.parts)


class AnyThing(DirtyEquals):
    def equals(self, other: Any) -> bool:
        return True


class IsOneOf(DirtyEquals):
    def __init__(self, *expected: Any) -> None:
        self.expected = expected

    def equals(self, other: Any) -> bool:
        return any(other == item for item in self.expected)


class IsDict(DirtyEquals):
    def __init__(self, expected: dict | None = None, **kwargs: Any) -> None:
        self.expected = {**(expected or {}), **kwargs}

    def equals(self, other: Any) -> bool:
        return isinstance(other, dict) and other == self.expected


class IsPartialDict(IsDict):
    def equals(self, other: Any) -> bool:
        return isinstance(other, dict) and all(k in other and other[k] == v for k, v in self.expected.items())


class IsStr(DirtyEquals):
    def __init__(self, regex: str | None = None, **_: Any) -> None:
        self.regex = regex

    def equals(self, other: Any) -> bool:
        return isinstance(other, str) and (self.regex is None or re.fullmatch(self.regex, other) is not None)


class IsInt(DirtyEquals):
    def __init__(self, *_: Any, **__: Any) -> None:
        pass

    def equals(self, other: Any) -> bool:
        return isinstance(other, int) and not isinstance(other, bool)


class IsList(DirtyEquals):
    def __init__(self, *items: Any, length: int | None = None) -> None:
        self.items, self.length = items, length

    def equals(self, other: Any) -> bool:
        if not isinstance(other, list):
            return False
        if self.length is not None and len(other) != self.length:
            return False
        return not self.items or list(self.items) == other
