"""Repository tools for the localization agent: file reading, code search and the competition's graph tools."""

from __future__ import annotations

import ast
import fnmatch
import json
import os
import re
import shutil
import tarfile
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np

MAX_FILE_LINES = 150
MAX_FILE_CHARS = 10_000
MAX_SEARCH_MATCHES = 40
MAX_LINE_CHARS = 200
_NULL_STRINGS = {"", "none", "null", "nil"}


def graph_key(repo: str, base_commit: str) -> str:
    return f"{repo.rsplit('/', 1)[-1]}_{base_commit}"


def ensure_snapshot(snapshot_tgz: Path, destination: Path) -> Path:
    """Extract a task snapshot to `destination` unless it already exists (atomic, safe for concurrent users)."""
    if destination.exists():
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / f".tmp_{destination.name}_{os.getpid()}"
    staging.mkdir(parents=True, exist_ok=True)
    with tarfile.open(snapshot_tgz) as archive:
        archive.extractall(staging, filter="data")
    try:
        staging.rename(destination)
    except OSError:
        shutil.rmtree(staging, ignore_errors=True)
    return destination


def _truncate_items(items: list[str], limit: int) -> tuple[list[str], bool]:
    kept: list[str] = []
    used = 0
    for index, item in enumerate(items):
        if index == 0 and len(item) > limit:
            return [item[:limit]], True
        if index > 0 and used + len(item) > limit:
            return kept, True
        kept.append(item)
        used += len(item)
    return kept, False


def _allocate_budgets(lengths: list[int], limit: int) -> list[int]:
    """Split `limit` characters across items, rolling unused share over to larger items."""
    if sum(lengths) <= limit:
        return list(lengths)
    remaining = limit
    open_items = list(range(len(lengths)))
    budgets = [0] * len(lengths)
    while open_items:
        share = remaining // len(open_items)
        fitting = [i for i in open_items if lengths[i] <= share]
        if not fitting:
            extra = remaining % len(open_items)
            for rank, i in enumerate(open_items):
                budgets[i] = share + (rank < extra)
            break
        for i in fitting:
            budgets[i] = lengths[i]
            remaining -= lengths[i]
        open_items = [i for i in open_items if i not in set(fitting)]
    return budgets


def resolve_node_name(target: str, graph: Any) -> str | None:
    """Resolve a short or partially qualified symbol to a node id (exact, suffix, case-insensitive, substring)."""
    query = str(target).strip()
    if not query or not graph:
        return None
    bare = query.split(";", 1)[1] if ";" in query else query
    nodes = [str(n) for n in (graph.nodes() if hasattr(graph, "nodes") and callable(graph.nodes) else graph)]
    if not nodes:
        return None
    node_set = set(nodes)
    if query in node_set:
        return query
    if bare in node_set:
        return bare
    if "::" in bare:
        return _resolve_location(bare, node_set)

    def best(candidates: list[str]) -> str:
        return min(candidates, key=lambda n: (len(n.split(".")), len(n), n))

    suffixes = (f".{bare}", f";{bare}", f"/{bare}")
    if matches := [n for n in nodes if n.endswith(suffixes)]:
        return best(matches)
    lower_bare, lower_query = bare.lower(), query.lower()
    if matches := [n for n in nodes if n.lower() in (lower_query, lower_bare)]:
        return best(matches)
    lower_suffixes = tuple(s.lower() for s in suffixes)
    if matches := [n for n in nodes if n.lower().endswith(lower_suffixes)]:
        return best(matches)
    if not any(c.isspace() for c in bare):
        if matches := [n for n in nodes if bare in n or lower_bare in n.lower()]:
            return best(matches)
    return None


def _resolve_location(location: str, node_set: set[str]) -> str | None:
    """Map `path/to/mod.py::Class.method` to a dotted node id, dropping leading path parts for src layouts."""
    path, qualname = (part.strip() for part in location.split("::", 1))
    parts = path.removesuffix(".py").split("/")
    if parts and parts[-1] == "__init__":
        parts.pop()
    for start in range(len(parts) + 1):
        candidate = ".".join([*parts[start:], qualname])
        if candidate in node_set:
            return candidate
    return None


def _coerce_int(value: Any, default: int, name: str) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got boolean {value}")
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in _NULL_STRINGS:
            return default
        try:
            return int(float(text))
        except ValueError:
            pass
    raise ValueError(f"Invalid integer for {name}: {value!r}")


def _ok(**fields: Any) -> str:
    return json.dumps({"status": "ok", **fields}, ensure_ascii=False)


def _error(error_type: str, message: str) -> str:
    return json.dumps(
        {"status": "error", "error_type": error_type, "error_message": message}, ensure_ascii=False
    )


TYPE_ORDER = {"calls": 0, "inherits": 1, "contains": 2, "decorated_by": 3, "references": 4, "imports": 5}


class CodeGraph:
    """The competition's code graph, loaded from an arbitrary graph JSON path (node-link, schema of `swegemma`)."""

    def __init__(self, graph_path: Path, embeddings_path: Path | None, repo_key: str, max_chars: int = 5000):
        with open(graph_path, encoding="utf-8") as handle:
            data = json.load(handle)
        self.graph = nx.node_link_graph(data, edges="edges" if "edges" in data else "links", multigraph=True)
        self.embeddings_path = embeddings_path
        self.repo_key = repo_key
        self.max_chars = max_chars
        self.typed = False

    @cached_property
    def _similarity_index(self) -> tuple[list[str], list[str], np.ndarray]:
        table: dict[str, np.ndarray] = {}
        if self.embeddings_path and self.embeddings_path.exists():
            with np.load(self.embeddings_path, allow_pickle=False) as archive:
                table = {key: archive[key].astype(np.float32) for key in archive.files}
        dim = next((len(v) for v in table.values()), 256)
        names, texts, rows = [], [], []
        for node, attrs in self.graph.nodes(data=True):
            names.append(str(attrs.get("name", node)))
            texts.append(attrs.get("text") or "")
            rows.append(table.get(str(node), np.zeros(dim, dtype=np.float32)))
        matrix = np.stack(rows) if rows else np.zeros((0, dim), dtype=np.float32)
        return names, texts, matrix

    def get_code_neighbors(self, node: str, edge_type: str | None = None, max_neighbors: int = 50) -> str:
        try:
            limit = _coerce_int(max_neighbors, 50, "max_neighbors")
            if limit <= 0:
                raise ValueError("max_neighbors must be greater than 0.")
            if isinstance(edge_type, str) and edge_type.strip().lower() in _NULL_STRINGS:
                edge_type = None
            target = node.split(";", 1)[1] if ";" in node else node
            if not self.graph.has_node(target):
                target = resolve_node_name(target, self.graph)
            if target is None:
                return _error("NodeNotFound", f"No graph node matches {node!r}.")
            neighbors: list[str] = []
            if target is not None:
                wanted = None if edge_type is None else str(edge_type).lower()

                def keep(data: dict) -> bool:
                    relation = data.get("type") or data.get("relation")
                    return wanted is None or (relation is not None and str(relation).lower() == wanted)

                if self.typed:
                    return self._typed_neighbors(node, target, keep, limit)
                linked = [v for _, v, d in self.graph.out_edges(target, data=True) if keep(d)]
                linked += [u for u, _, d in self.graph.in_edges(target, data=True) if keep(d)]
                for other in linked:
                    name = str(other) if str(other).startswith(f"{self.repo_key};") else f"{self.repo_key};{other}"
                    if name not in neighbors:
                        neighbors.append(name)
                neighbors.sort()
                neighbors = neighbors[:limit]
            neighbors, truncated = _truncate_items(neighbors, self.max_chars)
            return _ok(node=node, neighbors=neighbors, count=len(neighbors), is_truncated=truncated)
        except Exception as exc:
            return _error("GraphLookupError", str(exc))

    def _typed_neighbors(self, node: str, target: str, keep: Callable[[dict], bool], limit: int) -> str:
        """Direction- and type-aware variant: outgoing and incoming edges listed separately, calls first."""

        def listing(edges: list[tuple[str, dict]]) -> list[str]:
            rows = sorted({(TYPE_ORDER.get(d.get("type"), len(TYPE_ORDER)), str(other), d.get("type") or "") for other, d in edges if keep(d)})
            return [f"{name} [{kind}]" for _, name, kind in rows][:limit]

        outgoing = listing([(v, d) for _, v, d in self.graph.out_edges(target, data=True)])
        incoming = listing([(u, d) for u, _, d in self.graph.in_edges(target, data=True)])
        outgoing, out_truncated = _truncate_items(outgoing, self.max_chars // 2)
        incoming, in_truncated = _truncate_items(incoming, self.max_chars // 2)
        return _ok(node=target, outgoing=outgoing, incoming=incoming, count=len(outgoing) + len(incoming), is_truncated=out_truncated or in_truncated)

    def get_code_subgraph(self, nodes: list[str] | str) -> str:
        try:
            if isinstance(nodes, str):
                text = nodes.strip()
                parsed = None
                if text.startswith("["):
                    try:
                        parsed = json.loads(text)
                    except json.JSONDecodeError:
                        parsed = None
                nodes = (
                    [str(x) for x in parsed]
                    if isinstance(parsed, list)
                    else [p.strip().strip("\"'") for p in text.strip("[]").split(",") if p.strip().strip("\"'")]
                )
            if not nodes:
                raise ValueError("subset_nodes cannot be empty.")
            valid: list[str] = []
            for raw in nodes:
                name = str(raw)
                if self.graph.has_node(name):
                    valid.append(name)
                elif ";" in name and self.graph.has_node(name.split(";", 1)[1]):
                    valid.append(name.split(";", 1)[1])
                elif (resolved := resolve_node_name(name, self.graph)) is not None:
                    valid.append(resolved)
            sub = self.graph.subgraph(valid)
            edges = [
                {"from": str(u), "to": str(v), "type": d.get("type") or d.get("relation", "connected")}
                for u, v, d in sub.edges(data=True)
            ]
            names = [str(n) for n in sub.nodes()]
            total_nodes, total_edges = len(names), len(edges)
            kept_nodes, truncated = _truncate_items(names, self.max_chars // 2 if edges else self.max_chars)
            edge_budget = max(0, self.max_chars - sum(len(n) for n in kept_nodes))
            kept_edges, used = [], 0
            for edge in edges:
                size = len(edge["from"]) + len(edge["to"]) + len(str(edge["type"]))
                if used + size > edge_budget:
                    truncated = True
                    break
                kept_edges.append(edge)
                used += size
            return _ok(
                nodes=kept_nodes,
                edges=kept_edges,
                node_count=total_nodes,
                edge_count=total_edges,
                is_truncated=truncated,
            )
        except Exception as exc:
            return _error("SubgraphExtractionError", str(exc))

    def search_similar_code(self, query: str, k: int = 10) -> str:
        try:
            count = _coerce_int(k, 10, "k")
            if count <= 0:
                raise ValueError("k must be greater than 0.")
            query = query.split(";", 1)[1] if ";" in query else query
            names, texts, embeddings = self._similarity_index
            hits = self._rank(query, count, names, texts, embeddings)
            codes = [str(h["code"] or "") for h in hits]
            lengths = [len(c) for c in codes]
            truncated = sum(lengths) > self.max_chars
            budgets = _allocate_budgets(lengths, self.max_chars)
            results = [
                {"node_name": h["node_name"], "code": codes[i][: budgets[i]], "similarity": round(float(h["similarity"]), 4)}
                for i, h in enumerate(hits)
            ]
            return _ok(query=query, results=results, count=len(results), is_truncated=truncated)
        except Exception as exc:
            return _error("SimilaritySearchError", str(exc))

    def _rank(self, query: str, k: int, names: list[str], texts: list[str], embeddings: np.ndarray) -> list[dict]:
        if not names:
            return []
        target = query
        if target not in names and not any(c.isspace() for c in target):
            target = resolve_node_name(target, names) or target
        if target in names:
            index = names.index(target)
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            norms[norms == 0] = 1e-12
            unit = embeddings / norms
            sims = unit @ (embeddings[index] / (np.linalg.norm(embeddings[index]) or 1e-12))
            order = sorted((j for j in range(len(names)) if j != index), key=lambda j: (-float(sims[j]), names[j]))
            return [{"node_name": names[j], "code": texts[j], "similarity": float(sims[j])} for j in order[: min(k, len(names) - 1)]]
        return self._lexical(query, k, names, texts)

    @staticmethod
    def _lexical(query: str, k: int, names: list[str], texts: list[str]) -> list[dict]:
        tokens = [t for t in re.findall(r"[a-z0-9_]+", query.strip().lower()) if len(t) >= 2]
        if not tokens:
            return []
        ceiling = len(tokens) * 4.0 + 0.5
        scored: list[tuple[float, int]] = []
        for index, name in enumerate(names):
            low_name, low_text = name.lower(), texts[index].lower()
            score = sum((tok in low_name) * 3.0 + (tok in low_text) * 1.0 for tok in tokens)
            if score > 0:
                if not low_name.startswith(("test.", "tests.")) and ".test_" not in low_name:
                    score += 0.5
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], names[item[1]]))
        return [
            {"node_name": names[i], "code": texts[i], "similarity": round(min(1.0, s / ceiling), 4)}
            for s, i in scored[:k]
        ]


@dataclass
class Repo:
    """An extracted task snapshot: file access, Python search, and mapping between graph node ids and `path::qualname`."""

    root: Path
    _definitions: dict[str, set[str]] = field(default_factory=dict, repr=False)

    @cached_property
    def python_files(self) -> list[str]:
        skipped = {".git", ".tox", "node_modules", ".venv", "venv", "__pycache__"}
        return sorted(
            p.relative_to(self.root).as_posix()
            for p in self.root.rglob("*.py")
            if not skipped.intersection(p.relative_to(self.root).parts) and p.is_file()
        )

    @cached_property
    def _modules(self) -> dict[str, list[str]]:
        index: dict[str, list[str]] = {}
        for path in self.python_files:
            parts = path[:-3].split("/")
            if parts[-1] == "__init__":
                parts = parts[:-1]
            for start in range(len(parts)):
                index.setdefault(".".join(parts[start:]), []).append(path)
        for paths in index.values():
            paths.sort(key=lambda p: (len(p), p))
        return index

    def _defined(self, path: str) -> set[str]:
        if path not in self._definitions:
            names: set[str] = set()
            try:
                tree = ast.parse((self.root / path).read_text(encoding="utf-8", errors="replace"))
            except SyntaxError:
                tree = None

            def visit(node: ast.AST, prefix: str) -> None:
                for child in ast.iter_child_nodes(node):
                    if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        names.add(prefix + child.name)
                        visit(child, f"{prefix}{child.name}.")
                    else:
                        visit(child, prefix)

            if tree is not None:
                visit(tree, "")
            self._definitions[path] = names
        return self._definitions[path]

    def locate(self, node_id: str) -> tuple[str, str] | None:
        """Map a graph node id such as `pkg.mod.Class.method` to `(path, qualname)` using the longest module prefix."""
        parts = node_id.split(";", 1)[-1].split(".")
        for split in range(len(parts) - 1, 0, -1):
            paths = self._modules.get(".".join(parts[:split]))
            if not paths:
                continue
            qualname = ".".join(parts[split:])
            for path in paths:
                if qualname in self._defined(path):
                    return path, qualname
            return paths[0], qualname
        return None

    def normalize(self, answer: str) -> str | None:
        """Turn a model answer (`path::qualname` or dotted node id) into canonical `path::qualname`."""
        answer = answer.strip().strip("`'\"")
        if "::" in answer:
            path, qualname = answer.split("::", 1)
            path = re.sub(r"^(\./|/?workspace/|/)", "", path.strip())
            return f"{path}::{qualname.strip()}" if path and qualname.strip() else None
        located = self.locate(answer)
        return f"{located[0]}::{located[1]}" if located else None

    def read_file(self, filepath: str, start_line: Any = None, end_line: Any = None) -> str:
        try:
            relative = re.sub(r"^(/?workspace/|/)", "", str(filepath).strip())
            if ".." in Path(relative).parts:
                raise ValueError("Path traversal is not allowed")
            target = self.root / relative
            if not target.is_file():
                return _error("FileNotFoundError", f"File not found: {relative}")
            lines = target.read_text(encoding="utf-8", errors="replace").splitlines()
            first = max(1, _coerce_int(start_line, 1, "start_line"))
            last = min(len(lines), _coerce_int(end_line, len(lines), "end_line"))
            kept: list[str] = []
            used = 0
            for number in range(first, min(last, first + MAX_FILE_LINES - 1) + 1):
                row = f"{number}\t{lines[number - 1]}"
                if used + len(row) > MAX_FILE_CHARS:
                    break
                kept.append(row)
                used += len(row) + 1
            reached = first + len(kept) - 1
            return _ok(
                filepath=relative,
                content="\n".join(kept),
                start_line=first,
                end_line=reached,
                total_lines=len(lines),
                is_truncated=reached < last,
            )
        except Exception as exc:
            return _error("FileReadError", str(exc))

    def search(self, pattern: str, path_glob: str | None = None) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return _error("InvalidPattern", str(exc))
        glob = path_glob if path_glob and path_glob.strip().lower() not in _NULL_STRINGS else "*.py"
        matches: list[str] = []
        files_hit: set[str] = set()
        for path in self.python_files:
            if not (fnmatch.fnmatch(path, glob) or fnmatch.fnmatch(path.rsplit("/", 1)[-1], glob)):
                continue
            try:
                text = (self.root / path).read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for number, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{path}:{number}:{line.strip()[:MAX_LINE_CHARS]}")
                    files_hit.add(path)
            if len(matches) > MAX_SEARCH_MATCHES * 4:
                break
        return _ok(
            matches=matches[:MAX_SEARCH_MATCHES],
            count=len(matches),
            files=len(files_hit),
            is_truncated=len(matches) > MAX_SEARCH_MATCHES,
        )


def _function(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {"type": "object", "properties": properties, "required": required},
        },
    }


_STR, _INT = {"type": "string"}, {"type": "integer"}

SCHEMAS: dict[str, dict] = {
    "read_file": _function(
        "read_file",
        "Read a repository file with line numbers (max 150 lines per call).",
        {"filepath": _STR, "start_line": _INT, "end_line": _INT},
        ["filepath"],
    ),
    "search": _function(
        "search",
        "Regex search over repository Python files. Returns path:line:text matches.",
        {"pattern": _STR, "path_glob": {"type": "string", "description": "e.g. fastapi/*.py"}},
        ["pattern"],
    ),
    "get_code_neighbors": _function(
        "get_code_neighbors",
        "Callers and callees of a symbol in the repo call graph. node is a dotted name or suffix such as 'Class.method'.",
        {"node": _STR, "edge_type": _STR, "max_neighbors": _INT},
        ["node"],
    ),
    "get_code_subgraph": _function(
        "get_code_subgraph",
        "Nodes and edges connecting a set of symbols in the call graph.",
        {"nodes": {"type": "array", "items": _STR}},
        ["nodes"],
    ),
    "search_similar_code": _function(
        "search_similar_code",
        "Functions most similar to a symbol (by embedding) with code snippets. Pass a symbol name, not a sentence.",
        {"query": _STR, "k": _INT},
        ["query"],
    ),
}
GRAPH_TOOLS = ("get_code_neighbors", "get_code_subgraph", "search_similar_code")
TOOL_SETS = {
    "search": ("read_file", "search"),
    "graph": ("read_file", *GRAPH_TOOLS),
    "walk": ("read_file", "get_code_neighbors", "get_code_subgraph"),
    "both": ("read_file", "search", *GRAPH_TOOLS),
}


class ToolBox:
    """Dispatches model tool calls for one task and reports whether each call was well-formed."""

    def __init__(self, repo: Repo, graph: CodeGraph | None, tool_set: str):
        self.repo = repo
        self.names = TOOL_SETS[tool_set]
        self._handlers: dict[str, Callable[..., str]] = {"read_file": repo.read_file, "search": repo.search}
        if graph is not None:
            self._handlers |= {name: getattr(graph, name) for name in GRAPH_TOOLS}

    @property
    def schemas(self) -> list[dict]:
        return [SCHEMAS[name] for name in self.names]

    def call(self, name: str, arguments: Any) -> tuple[str, bool]:
        """Run a tool; the flag is False when the call itself was invalid (unknown tool or bad arguments)."""
        if name not in self.names:
            return _error("UnknownTool", f"Unknown tool {name!r}. Available: {', '.join(self.names)}"), False
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments or "{}")
            except json.JSONDecodeError as exc:
                return _error("InvalidArguments", f"Arguments are not valid JSON: {exc}"), False
        if not isinstance(arguments, dict):
            return _error("InvalidArguments", "Arguments must be a JSON object"), False
        schema = SCHEMAS[name]["function"]["parameters"]
        missing = [p for p in schema["required"] if p not in arguments]
        unknown = [p for p in arguments if p not in schema["properties"]]
        if missing or unknown:
            return _error("InvalidArguments", f"missing={missing} unexpected={unknown}"), False
        return self._handlers[name](**arguments), True
