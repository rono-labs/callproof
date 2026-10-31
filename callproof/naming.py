"""Neutral identifiers for definitions, and the node ids that code graphs give them.

A definition is identified neutrally by its repo-relative POSIX path and its Python
``__qualname__`` (``"<module>"`` for module-level code, ``"f.<locals>.g"`` for nested functions).
Tracers that fold nested scopes into the enclosing named definition produce *collapsed* qualnames,
which simply stop before the first ``<locals>``.

Two node-id conventions are supported.

``harness`` -- the convention of the competition's provided graphs, reverse-engineered from all
127 of them (the predicted id set equals the provided node set exactly on every graph):

* Every ``*.py`` file outside ``.git`` is parsed; files with syntax errors are skipped.
* Module names are relative to the deepest *source root* above the file. Source roots are the
  repository root plus the parent directory of every topmost package (a directory with
  ``__init__.py`` whose parent has none). Hence ``src/requests/models.py -> requests.models``,
  ``docs_src/events/tutorial001.py -> events.tutorial001`` once ``docs_src/events`` is a package, and
  ``scripts/people.py -> scripts.people`` when ``scripts`` is not.
* Async functions are not nodes (their nested sync functions are). There are no module nodes.
* A class is ``<module>.<ClassName>``, whatever encloses it.
* A function is ``<module>.<Class>.<name>`` when a "current class" is set, else ``<module>.<name>``.
  The current class is set on entering a class body and cleared, not restored, on leaving it, so a
  function nested in a method keeps the method's class (``Table.render.helper -> Table.helper``) and
  methods that follow a nested class lose their class (``TestRequests.test_x -> test_requests.test_x``).
* Distinct definitions mapping to one id collapse into a single node holding the last one's text.

``callproof`` -- the convention of graphs built by :mod:`callproof.build`: the same module names,
then the qualname with ``<locals>`` segments dropped (``Console.export_svg.make_tag``). Module nodes
use the bare module name (see :func:`module_node_id` for the rare clash). Ids therefore coincide with ``harness`` ids for every top-level function,
class and method that is not affected by the flattening rules above.
"""

from __future__ import annotations

import ast
import os
from collections import defaultdict
from collections.abc import Iterable, Iterator
from itertools import pairwise
from dataclasses import dataclass
from pathlib import Path

MODULE_QUALNAME = "<module>"
LOCALS = "<locals>"

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef
DefinitionNode = FunctionNode | ast.ClassDef


@dataclass(frozen=True)
class Definition:
    path: str
    qualname: str
    kind: str
    lineno: int
    end_lineno: int
    harness_id: str | None
    node_id: str

    @property
    def is_async(self) -> bool:
        return self.kind.startswith("async")

    @property
    def collapsed_qualname(self) -> str:
        return collapse_qualname(self.qualname)


def collapse_qualname(qualname: str) -> str:
    return qualname.split(f".{LOCALS}", 1)[0]


def drop_locals(qualname: str) -> str:
    return ".".join(part for part in qualname.split(".") if part != LOCALS)


def iter_python_files(root: Path) -> Iterator[str]:
    """Repo-relative paths of all ``*.py`` files outside ``.git``, in a stable order."""
    for directory, subdirs, files in os.walk(root):
        subdirs[:] = sorted(d for d in subdirs if d != ".git")
        rel = Path(directory).relative_to(root)
        for name in sorted(files):
            if name.endswith(".py"):
                yield (rel / name).as_posix()


def source_roots(paths: Iterable[str]) -> set[str]:
    """The repository root ("") plus the parent of every topmost package among `paths`."""
    package_dirs = {path.rpartition("/")[0] for path in paths if path.endswith("/__init__.py")}
    roots = {""}
    for directory in package_dirs:
        parent = directory.rpartition("/")[0]
        if parent not in package_dirs:
            roots.add(parent)
    return roots


def module_name(path: str, roots: set[str]) -> str:
    parts = path.removesuffix(".py").split("/")
    start = max(k for k in range(len(parts)) if "/".join(parts[:k]) in roots)
    parts = parts[start:]
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def join(module: str, local: str) -> str:
    return f"{module}.{local}" if module else local


def module_node_id(module: str, definition_ids: set[str]) -> str:
    """A module's node id: its name, unless a definition already owns that id (``pkg/__init__.py``
    defining ``b`` next to ``pkg/b.py``) or the module is a top-level ``__init__.py``."""
    if module and module not in definition_ids:
        return module
    return join(module, MODULE_QUALNAME)


def harness_id_from_qualname(module: str, qualname: str, is_class: bool) -> str:
    """Harness id of a definition known only by qualname, assuming no earlier nested class reset the context.

    Segments followed by ``<locals>`` are functions; the others are classes.
    """
    parts = qualname.split(".")
    name = parts[-1]
    if is_class:
        return join(module, name)
    classes = [part for part, following in pairwise(parts) if LOCALS not in (part, following)]
    return join(module, f"{classes[-1]}.{name}" if classes else name)


class _Collector(ast.NodeVisitor):
    """Walks one module recording each definition's qualname and its id under both conventions."""

    def __init__(self, path: str, module: str) -> None:
        self.path = path
        self.module = module
        self.qualname: list[str] = []
        self.class_stack: list[bool] = []
        self.current_class: str | None = None
        self.found: list[tuple[Definition, DefinitionNode]] = []

    def _record(self, node: DefinitionNode, kind: str, harness_local: str | None) -> str:
        qualname = ".".join([*self.qualname, node.name])
        definition = Definition(
            path=self.path,
            qualname=qualname,
            kind=kind,
            lineno=min([node.lineno, *(d.lineno for d in node.decorator_list)]),
            end_lineno=node.end_lineno or node.lineno,
            harness_id=None if harness_local is None else join(self.module, harness_local),
            node_id=join(self.module, drop_locals(qualname)),
        )
        self.found.append((definition, node))
        return qualname

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        qualname = self._record(node, "class", node.name)
        self.current_class = node.name
        self._descend(node, qualname, is_class=True)
        self.current_class = None

    def visit_FunctionDef(self, node: FunctionNode) -> None:
        in_class = bool(self.class_stack) and self.class_stack[-1]
        kind = "method" if in_class else "function"
        if isinstance(node, ast.AsyncFunctionDef):
            kind, harness_local = f"async_{kind}", None
        else:
            harness_local = f"{self.current_class}.{node.name}" if self.current_class else node.name
        qualname = self._record(node, kind, harness_local)
        self._descend(node, f"{qualname}.{LOCALS}", is_class=False)

    visit_AsyncFunctionDef = visit_FunctionDef

    def _descend(self, node: DefinitionNode, qualname: str, is_class: bool) -> None:
        saved = self.qualname
        self.qualname = qualname.split(".")
        self.class_stack.append(is_class)
        self.generic_visit(node)
        self.class_stack.pop()
        self.qualname = saved


def collect_definitions(tree: ast.Module, path: str, module: str) -> list[tuple[Definition, DefinitionNode]]:
    collector = _Collector(path, module)
    collector.visit(tree)
    return collector.found


@dataclass
class ParsedModule:
    path: str
    name: str
    source: str
    tree: ast.Module


def parse_checkout(root: Path) -> list[ParsedModule]:
    """Parse every Python file of a checkout, skipping files that do not parse (as the provided graphs do)."""
    paths = list(iter_python_files(root))
    roots = source_roots(paths)
    modules = []
    for path in paths:
        source = (root / path).read_bytes()
        try:
            tree = ast.parse(source, filename=path)
        except (SyntaxError, ValueError):
            continue
        modules.append(ParsedModule(path, module_name(path, roots), source.decode("utf-8", "replace"), tree))
    return modules


class RepoIndex:
    """Every definition of a checkout, addressable by (path, qualname) and by node id in either convention."""

    def __init__(self, modules: list[ParsedModule]) -> None:
        self.modules = {m.path: m.name for m in modules}
        self.trees = {m.path: m.tree for m in modules}
        self.definitions: list[Definition] = []
        self.ast_nodes: dict[tuple[str, str], DefinitionNode] = {}
        for module in modules:
            for definition, node in collect_definitions(module.tree, module.path, module.name):
                self.definitions.append(definition)
                self.ast_nodes[(definition.path, definition.qualname)] = node
        self._by_key = {(d.path, d.qualname): d for d in self.definitions}
        self._by_harness: dict[str, list[Definition]] = defaultdict(list)
        self._by_node: dict[str, list[Definition]] = defaultdict(list)
        for d in self.definitions:
            if d.harness_id is not None:
                self._by_harness[d.harness_id].append(d)
            self._by_node[d.node_id].append(d)
        self._def_ids = set(self._by_node)

    @classmethod
    def from_checkout(cls, root: Path) -> RepoIndex:
        return cls(parse_checkout(root))

    def lookup(self, path: str, qualname: str) -> Definition | None:
        return self._by_key.get((path, qualname))

    def harness_id(self, path: str, qualname: str, is_class: bool = False) -> str | None:
        """Provided-graph id for a definition; None for module-level code and async functions.

        Definitions absent from the checkout (e.g. added by a patch) get the id the convention would assign.
        """
        if qualname == MODULE_QUALNAME or not path.endswith(".py"):
            return None
        definition = self.lookup(path, qualname)
        if definition is not None:
            return definition.harness_id
        return harness_id_from_qualname(self.module_of(path), qualname, is_class)

    def node_id(self, path: str, qualname: str) -> str:
        """Callproof-graph id for a definition (the module node for ``<module>``)."""
        module = self.module_of(path)
        if qualname == MODULE_QUALNAME:
            return module_node_id(module, self._def_ids)
        return join(module, drop_locals(qualname))

    def module_of(self, path: str) -> str:
        if path in self.modules:
            return self.modules[path]
        return module_name(path, source_roots([*self.modules, path]))

    def from_harness_id(self, node_id: str) -> list[Definition]:
        return self._by_harness.get(node_id, [])

    def from_node_id(self, node_id: str) -> list[Definition]:
        return self._by_node.get(node_id, [])

    def harness_collisions(self) -> dict[str, list[Definition]]:
        return {k: v for k, v in self._by_harness.items() if len(v) > 1}
