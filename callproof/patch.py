"""Map the lines edited by a unified diff onto the Python functions and classes that contain them."""

from __future__ import annotations

import ast
import re
from collections.abc import Callable
from dataclasses import dataclass, field

MODULE = "<module>"
_HUNK = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_Scope = tuple[int, int, str, str]


@dataclass(frozen=True)
class GoldFunction:
    path: str
    qualname: str
    kind: str
    node_type: str
    collapsed_qualname: str


@dataclass
class FileEdit:
    old_path: str | None
    new_path: str | None
    removed: list[int] = field(default_factory=list)
    added: list[tuple[int, str]] = field(default_factory=list)
    removed_text: dict[int, str] = field(default_factory=dict)

    @property
    def path(self) -> str:
        return self.new_path or self.old_path or ""


def _strip_prefix(raw: str) -> str | None:
    raw = raw.split("\t")[0].strip()
    if raw == "/dev/null":
        return None
    return raw[2:] if raw[:2] in ("a/", "b/") else raw


def parse_diff(text: str) -> list[FileEdit]:
    """Parse a unified diff into per-file lists of removed (old numbering) and added (new numbering) lines."""
    edits: list[FileEdit] = []
    current: FileEdit | None = None
    old_path: str | None = None
    have_old = False
    old_no = new_no = old_left = new_left = 0
    for line in text.splitlines():
        if (old_left or new_left) and current is not None:
            if line.startswith("+"):
                current.added.append((new_no, line[1:]))
                new_no += 1
                new_left -= 1
            elif line.startswith("-"):
                current.removed.append(old_no)
                current.removed_text[old_no] = line[1:]
                old_no += 1
                old_left -= 1
            elif not line.startswith("\\"):
                old_no += 1
                new_no += 1
                old_left -= 1
                new_left -= 1
        elif line.startswith("--- "):
            old_path, have_old = _strip_prefix(line[4:]), True
        elif line.startswith("+++ ") and have_old:
            current = FileEdit(old_path, _strip_prefix(line[4:]))
            edits.append(current)
            have_old = False
        elif (match := _HUNK.match(line)) and current is not None:
            old_no, new_no = int(match[1]), int(match[3])
            old_left = int(match[2] if match[2] is not None else 1)
            new_left = int(match[4] if match[4] is not None else 1)
    return edits


def _scopes(source: str) -> list[_Scope]:
    tree = ast.parse(source)
    scopes: list[_Scope] = []

    def visit(node: ast.AST, prefix: str, in_function: bool) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                start = min([child.lineno, *(d.lineno for d in child.decorator_list)])
                qualname = f"{prefix}{child.name}"
                kind = "class" if isinstance(child, ast.ClassDef) else "function"
                scopes.append((start, child.end_lineno or child.lineno, qualname, kind))
                inner = f"{qualname}.<locals>." if kind == "function" else f"{qualname}."
                visit(child, inner, in_function or kind == "function")
            else:
                visit(child, prefix, in_function)

    visit(tree, "", False)
    return scopes


def _collapse(qualname: str) -> str:
    return qualname.split(".<locals>")[0]


def _innermost(scopes: list[_Scope], line: int) -> _Scope | None:
    best: _Scope | None = None
    for scope in scopes:
        if scope[0] <= line <= scope[1] and (best is None or scope[0] >= best[0] and scope[1] <= best[1]):
            best = scope
    return best


def _is_code(text: str) -> bool:
    stripped = text.strip()
    return bool(stripped) and not stripped.startswith("#")


def extract_gold_functions(
    patch: str,
    read_base: Callable[[str], str | None],
    read_head: Callable[[str], str | None],
) -> tuple[list[GoldFunction], list[str]]:
    """Return the functions edited by `patch` and the non-Python files it touches.

    `read_base` and `read_head` return a file's text before and after the patch (None when absent).
    Removed lines are located in the base file, added lines in the patched file. A function is
    "modified" if its qualname exists in the base file and "added" otherwise. Blank and comment-only
    lines are ignored.
    """
    found: dict[tuple[str, str], GoldFunction] = {}
    non_py: list[str] = []
    for edit in parse_diff(patch):
        path = edit.path
        if not path.endswith(".py"):
            non_py.append(path)
            continue
        base_src = read_base(edit.old_path) if edit.old_path else None
        head_src = read_head(edit.new_path) if edit.new_path else None
        base_scopes = _safe_scopes(base_src)
        head_scopes = _safe_scopes(head_src)
        base_names = {scope[2] for scope in base_scopes}

        def record(scope: _Scope | None, in_base: bool) -> None:
            qualname, node_type = (scope[2], scope[3]) if scope else (MODULE, "module")
            existed = qualname in base_names or (qualname == MODULE and base_src is not None)
            kind = "modified" if in_base or existed else "added"
            key = (path, qualname)
            if key not in found or kind == "modified":
                found[key] = GoldFunction(path, qualname, kind, node_type, _collapse(qualname))

        for line in edit.removed:
            if _is_code(edit.removed_text.get(line, "")):
                record(_innermost(base_scopes, line), True)
        for line, text in edit.added:
            if _is_code(text):
                record(_innermost(head_scopes, line), False)
    return sorted(found.values(), key=lambda f: (f.path, f.qualname)), non_py


def _safe_scopes(source: str | None) -> list[_Scope]:
    if source is None:
        return []
    try:
        return _scopes(source)
    except SyntaxError:
        return []
