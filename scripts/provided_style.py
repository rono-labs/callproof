"""A reimplementation of the competition graph's documented rules, for use as a baseline on other repos.

Nodes follow the ``harness`` id convention of :mod:`callproof.naming` (no async definitions, no module
nodes). Edges are undirected, stored as one ``calls`` edge per direction, and come from four name-based
rules: a class contains its methods, a class inherits from a base named in its ``bases``, a definition
instantiates a class it calls by plain name, and a definition calls a function it calls by plain name.
Method calls through ``self`` or other attributes are not edges.
"""

from __future__ import annotations

import ast
from collections import defaultdict

import networkx as nx

from callproof.naming import Definition, RepoIndex


class _Resolver:
    """Resolves a plain name used in a module to a definition: same module, then its imports, then a unique name."""

    def __init__(self, index: RepoIndex) -> None:
        self.by_module: dict[tuple[str, str], Definition] = {}
        self.by_name: dict[str, list[Definition]] = defaultdict(list)
        for definition in index.definitions:
            if definition.harness_id is None or "." in definition.qualname:
                continue
            self.by_module[(index.modules[definition.path], definition.qualname)] = definition
            self.by_name[definition.qualname].append(definition)
        self.imports: dict[str, dict[str, str]] = {}
        for path, tree in index.trees.items():
            module = index.modules[path]
            package = module if path.endswith("__init__.py") else module.rpartition(".")[0]
            table: dict[str, str] = {}
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    base = _absolute(node, package)
                    for alias in node.names:
                        table[alias.asname or alias.name] = f"{base}.{alias.name}" if base else alias.name
            self.imports[module] = table

    def resolve(self, module: str, name: str) -> Definition | None:
        local = self.by_module.get((module, name))
        if local is not None:
            return local
        target = self.imports.get(module, {}).get(name)
        if target is not None:
            owner, _, original = target.rpartition(".")
            imported = self.by_module.get((owner, original))
            if imported is not None:
                return imported
        candidates = self.by_name.get(name, [])
        return candidates[0] if len(candidates) == 1 else None


def _absolute(node: ast.ImportFrom, package: str) -> str:
    if node.level == 0:
        return node.module or ""
    parts = package.split(".") if package else []
    parts = parts[: max(len(parts) - (node.level - 1), 0)]
    if node.module:
        parts.append(node.module)
    return ".".join(parts)


def _terminal(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    return None


def build_provided_style(index: RepoIndex, repo_name: str = "") -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph(repo_name=repo_name)
    resolver = _Resolver(index)
    nodes = [d for d in index.definitions if d.harness_id is not None]
    graph.add_nodes_from(d.harness_id for d in nodes)

    def link(a: str | None, b: str | None) -> None:
        if a and b and a != b:
            graph.add_edge(a, b, type="calls")
            graph.add_edge(b, a, type="calls")

    for definition in nodes:
        node = index.ast_nodes[(definition.path, definition.qualname)]
        module = index.modules[definition.path]
        if definition.kind == "class":
            for base in node.bases:
                name = _terminal(base)
                target = resolver.resolve(module, name) if name else None
                link(definition.harness_id, target.harness_id if target else None)
            for child in node.body:
                if isinstance(child, ast.FunctionDef):
                    method = f"{definition.harness_id}.{child.name}"
                    if method in graph:
                        link(definition.harness_id, method)
            continue
        for sub in ast.walk(node):
            if isinstance(sub, ast.Call) and isinstance(sub.func, ast.Name):
                target = resolver.resolve(module, sub.func.id)
                link(definition.harness_id, target.harness_id if target else None)
    return graph
