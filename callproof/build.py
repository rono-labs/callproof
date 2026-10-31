"""Static code-graph builder emitting the competition's graph format.

Nodes are every module, class and function (sync or async, at any nesting depth) of a checkout,
with ids from :mod:`callproof.naming` (``callproof`` convention) and node attributes ``name``,
``text``, ``kind``, ``path``, ``qualname``, ``lineno`` and ``end_lineno``.

Edges are directed and typed; ``u -> v`` means:

``calls``
    Code in ``u`` calls ``v``: a function or method, a class (instantiation; an extra edge goes to
    the ``__init__`` found on the class's MRO), an instance's ``__call__``, or a property getter that
    ``u`` reads. ``u`` is the innermost enclosing function, else the class body, else the module;
    decorators and default values belong to the scope that evaluates them.
``inherits``
    Class ``u`` lists class ``v`` among its bases.
``imports``
    An import statement in ``u`` binds ``v`` (a definition or a module).
``contains``
    ``v`` is defined directly inside ``u`` (module, class or function).
``decorated_by``
    Definition ``u`` is decorated by ``v``, or by a call to ``v`` (``@app.get("/")`` -> ``FastAPI.get``).
``references``
    Code in ``u`` uses the function or class ``v`` as a value without calling it there
    (callbacks, ``Depends(get_db)``, ``isinstance(x, Foo)``). Annotations are not references.

Every edge carries a ``resolution`` attribute saying how its target was found:

``static``
    Scoping, imports (re-exports through ``__init__``, aliases, relative and star imports), class
    hierarchies (C3 MRO, ``super()``) and light type inference: ``x = Foo()``, ``self.x = Foo()``,
    ``x: Foo``, ``def f(x: Foo)``, ``-> Foo``, ``Optional``/``Union``/``Annotated`` and string
    annotations, properties, and element types of ``list[Foo]``-style containers and of
    ``__iter__`` annotated as ``Iterator[Foo]`` (``for x in ...`` and comprehensions).
``protocol``
    A call Python makes implicitly on a value of known class: ``len``/``iter``/``str``/``repr``/
    ``hash``-style builtins, ``for`` (``__iter__``), ``with`` (``__enter__``/``__exit__``), binary
    and augmented operators, comparisons, ``in``, subscripts and f-string fields.
``override``
    Class hierarchy analysis: a method call on a value of class ``C`` also targets the methods of
    that name defined in subclasses of ``C``.
``unique-name``
    Fallback for a method call on a value of unknown type: the only repository method of that
    name, provided the name is not also a method of a builtin type.
"""

from __future__ import annotations

import argparse
import ast
import sys
import textwrap
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx

from callproof.graphio import save_graph
from callproof.naming import (
    LOCALS,
    MODULE_QUALNAME,
    DefinitionNode,
    ParsedModule,
    collect_definitions,
    drop_locals,
    join,
    module_node_id,
    parse_checkout,
)

CALLS = "calls"
INHERITS = "inherits"
IMPORTS = "imports"
CONTAINS = "contains"
DECORATED_BY = "decorated_by"
REFERENCES = "references"

STATIC = "static"
PROTOCOL = "protocol"
OVERRIDE = "override"
UNIQUE_NAME = "unique-name"

_MAX_VALUES = 8
_PROPERTY_DECORATORS = {"property", "cached_property", "setter", "deleter"}
_TRANSPARENT_GENERICS = {"Optional", "Union", "Annotated", "Final", "ClassVar", "Required", "NotRequired"}
_TYPE_GENERICS = {"type", "Type"}
_CONTAINER_GENERICS = {
    *("list", "List", "Sequence", "MutableSequence", "Collection", "Iterable", "Iterator", "Generator"),
    *("AsyncIterable", "AsyncIterator", "AsyncGenerator", "set", "Set", "frozenset", "FrozenSet"),
    *("deque", "Deque", "tuple", "Tuple"),
}
_BUILTIN_PROTOCOLS = {
    "len": "__len__",
    "iter": "__iter__",
    "next": "__next__",
    "str": "__str__",
    "repr": "__repr__",
    "hash": "__hash__",
    "bool": "__bool__",
    "reversed": "__reversed__",
    "format": "__format__",
    "abs": "__abs__",
    "int": "__int__",
    "float": "__float__",
}
_BINARY_PROTOCOLS = {
    ast.Add: "__add__",
    ast.Sub: "__sub__",
    ast.Mult: "__mul__",
    ast.MatMult: "__matmul__",
    ast.Div: "__truediv__",
    ast.FloorDiv: "__floordiv__",
    ast.Mod: "__mod__",
    ast.Pow: "__pow__",
    ast.LShift: "__lshift__",
    ast.RShift: "__rshift__",
    ast.BitOr: "__or__",
    ast.BitAnd: "__and__",
    ast.BitXor: "__xor__",
}
_COMPARE_PROTOCOLS = {
    ast.Eq: "__eq__",
    ast.NotEq: "__ne__",
    ast.Lt: "__lt__",
    ast.LtE: "__le__",
    ast.Gt: "__gt__",
    ast.GtE: "__ge__",
}
_SUBSCRIPT_PROTOCOLS = {ast.Load: "__getitem__", ast.Store: "__setitem__", ast.Del: "__delitem__"}
_FUNCTION_TYPES = (ast.FunctionDef, ast.AsyncFunctionDef)
_BUILTIN_METHODS = frozenset(
    name for kind in (str, bytes, list, tuple, dict, set, frozenset, int, float, object) for name in dir(kind)
)
_RECEIVER_TYPES = (ast.Name, ast.Attribute, ast.Call, ast.Subscript)
_SCOPE_TYPES = (*_FUNCTION_TYPES, ast.ClassDef)


@dataclass(frozen=True)
class Value:
    """What an expression may evaluate to, as far as the repository's own code is concerned."""

    kind: str  # "def" | "module" | "instance" | "iterable" (of instances) | "super" | "property" | "external"
    target: str


@dataclass(frozen=True)
class Binding:
    kind: str  # "value" | "annotation" | "element" | "import" | "import_from" | "def" | "self" | "cls"
    scope: Scope | None = None
    expr: ast.expr | None = None
    target: str = ""
    attr: str = ""


@dataclass(eq=False)
class Scope:
    node_id: str
    kind: str
    module: ModuleScope | None
    parent: Scope | None
    node: ast.AST
    qualname: str
    bindings: dict[str, list[Binding]] = field(default_factory=lambda: defaultdict(list))
    owner: Scope | None = None
    attributes: dict[str, list[Binding]] = field(default_factory=lambda: defaultdict(list))
    children: list[Scope] = field(default_factory=list)

    @property
    def is_class(self) -> bool:
        return self.kind == "class"

    @property
    def is_function(self) -> bool:
        return self.kind not in ("class", "module")


@dataclass(eq=False)
class ModuleScope(Scope):
    path: str = ""
    name: str = ""
    is_package: bool = False
    star_imports: list[str] = field(default_factory=list)


def _decorator_names(node: DefinitionNode) -> set[str]:
    names = set()
    for decorator in node.decorator_list:
        target = decorator.func if isinstance(decorator, ast.Call) else decorator
        if isinstance(target, ast.Name):
            names.add(target.id)
        elif isinstance(target, ast.Attribute):
            names.add(target.attr)
    return names


def _is_property(node: ast.AST) -> bool:
    return isinstance(node, _FUNCTION_TYPES) and bool(_decorator_names(node) & _PROPERTY_DECORATORS)


def _subscript_head(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _function_kind(node: ast.AST, in_class: bool) -> str:
    kind = "method" if in_class else "function"
    return f"async_{kind}" if isinstance(node, ast.AsyncFunctionDef) else kind


class GraphBuilder:
    """Builds the callproof graph of one checkout. Use :func:`build_graph`."""

    def __init__(self, modules: list[ParsedModule], repo_name: str = "") -> None:
        self.repo_name = repo_name
        self.parsed = modules
        self.graph = nx.MultiDiGraph(repo_name=repo_name)
        self.modules: dict[str, list[ModuleScope]] = defaultdict(list)
        self.scopes: dict[str, Scope] = {}
        self.class_scopes: list[Scope] = []
        self.edges: dict[tuple[str, str, str], str] = {}
        self._cache: dict[tuple[int, int], frozenset[Value]] = {}
        self._evaluating: set[tuple[int, int]] = set()
        self._resolving: set[tuple[int, str]] = set()
        self._mro: dict[str, list[Scope]] = {}
        self._methods_by_name: dict[str, list[str]] = defaultdict(list)
        self._subclasses: dict[str, list[Scope]] | None = None

    def build(self) -> nx.MultiDiGraph:
        def_ids = {d.node_id for m in self.parsed for d, _ in collect_definitions(m.tree, m.path, m.name)}
        for parsed in self.parsed:
            self._add_module(parsed, module_node_id(parsed.name, def_ids))
        for scope in self.class_scopes:
            for name, bindings in scope.bindings.items():
                if any(b.kind == "def" and self.scopes[b.target].is_function for b in bindings):
                    self._methods_by_name[name].append(scope.bindings[name][-1].target)
        for scopes in self.modules.values():
            for module in scopes:
                self._link_scope(module)
        for (source, target, kind), resolution in self.edges.items():
            self.graph.add_edge(source, target, type=kind, resolution=resolution)
        return self.graph

    # Collection: scopes, nodes and bindings.

    def _add_module(self, parsed: ParsedModule, node_id: str) -> None:
        module = ModuleScope(
            node_id=node_id,
            kind="module",
            module=None,
            parent=None,
            node=parsed.tree,
            qualname=MODULE_QUALNAME,
            path=parsed.path,
            name=parsed.name,
            is_package=parsed.path.endswith("__init__.py"),
        )
        module.module = module
        self.modules[parsed.name].append(module)
        self.scopes[node_id] = module
        lines = parsed.source.splitlines()
        self._add_node(module, _module_text(parsed.tree), parsed.path, 1, len(lines))
        self._collect(module, parsed.tree.body, lines)

    def _add_node(self, scope: Scope, text: str, path: str, lineno: int, end_lineno: int) -> None:
        self.graph.add_node(
            scope.node_id,
            name=scope.node_id,
            text=text,
            kind=scope.kind,
            path=path,
            qualname=scope.qualname,
            lineno=lineno,
            end_lineno=end_lineno,
        )

    def _new_scope(self, node: DefinitionNode, parent: Scope, lines: list[str]) -> Scope:
        module = parent.module
        qualname = _child_qualname(parent, node.name)
        kind = "class" if isinstance(node, ast.ClassDef) else _function_kind(node, parent.is_class)
        scope = Scope(
            node_id=join(module.name, drop_locals(qualname)),
            kind=kind,
            module=module,
            parent=parent,
            node=node,
            qualname=qualname,
            owner=parent if parent.is_class else None,
        )
        self.scopes[scope.node_id] = scope
        start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
        end = node.end_lineno or node.lineno
        text = _class_text(node) if isinstance(node, ast.ClassDef) else textwrap.dedent("\n".join(lines[start - 1 : end]))
        self._add_node(scope, text, module.path, start, end)
        self._add_edge(parent.node_id, scope.node_id, CONTAINS)
        parent.children.append(scope)
        if scope.is_class:
            self.class_scopes.append(scope)
        return scope

    def _collect(self, scope: Scope, body: list[ast.stmt], lines: list[str]) -> None:
        for stmt in body:
            self._collect_statement(scope, stmt, lines)

    def _collect_statement(self, scope: Scope, stmt: ast.stmt, lines: list[str]) -> None:
        if isinstance(stmt, _SCOPE_TYPES):
            child = self._new_scope(stmt, scope, lines)
            scope.bindings[stmt.name].append(Binding("def", target=child.node_id))
            if isinstance(stmt, _FUNCTION_TYPES):
                self._bind_parameters(child, stmt)
            self._collect(child, stmt.body, lines)
            return
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                if alias.asname:
                    scope.bindings[alias.asname].append(Binding("import", scope, target=alias.name))
                else:
                    top = alias.name.split(".")[0]
                    scope.bindings[top].append(Binding("import", scope, target=top))
        elif isinstance(stmt, ast.ImportFrom):
            source = self._absolute_module(scope.module, stmt.module, stmt.level)
            for alias in stmt.names:
                if alias.name == "*":
                    scope.module.star_imports.append(source)
                else:
                    scope.bindings[alias.asname or alias.name].append(
                        Binding("import_from", scope, target=source, attr=alias.name)
                    )
        elif isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                self._bind_target(scope, target, Binding("value", scope, stmt.value))
        elif isinstance(stmt, ast.AnnAssign):
            self._bind_target(scope, stmt.target, Binding("annotation", scope, stmt.annotation))
            if stmt.value is not None:
                self._bind_target(scope, stmt.target, Binding("value", scope, stmt.value))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            for item in stmt.items:
                if item.optional_vars is not None:
                    self._bind_target(scope, item.optional_vars, Binding("value", scope, item.context_expr))
        elif isinstance(stmt, (ast.For, ast.AsyncFor)):
            self._bind_target(scope, stmt.target, Binding("element", scope, stmt.iter))
        for expr in _expressions(stmt):
            for node in ast.walk(expr):
                if isinstance(node, ast.comprehension):
                    self._bind_target(scope, node.target, Binding("element", scope, node.iter))
        for name in ("body", "orelse", "finalbody", "handlers", "cases"):
            for child in getattr(stmt, name, ()):
                if isinstance(child, ast.stmt):
                    self._collect_statement(scope, child, lines)
                else:
                    self._collect(scope, child.body, lines)

    def _bind_target(self, scope: Scope, target: ast.expr, binding: Binding) -> None:
        if isinstance(target, ast.Name):
            scope.bindings[target.id].append(binding)
        elif (
            isinstance(target, ast.Attribute)
            and isinstance(target.value, ast.Name)
            and scope.owner is not None
            and target.value.id == self._self_name(scope)
        ):
            scope.owner.attributes[target.attr].append(binding)

    @staticmethod
    def _self_name(scope: Scope) -> str | None:
        node = scope.node
        if not isinstance(node, _FUNCTION_TYPES) or "staticmethod" in _decorator_names(node):
            return None
        positional = [*node.args.posonlyargs, *node.args.args]
        return positional[0].arg if positional else None

    def _bind_parameters(self, scope: Scope, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        arguments = [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        first = self._self_name(scope)
        for arg in arguments:
            if arg.arg == first and scope.owner is not None:
                is_class = "classmethod" in _decorator_names(node) or node.name in ("__new__", "__init_subclass__")
                scope.bindings[arg.arg].append(Binding("cls" if is_class else "self", target=scope.owner.node_id))
            elif arg.annotation is not None:
                scope.bindings[arg.arg].append(Binding("annotation", scope.parent, arg.annotation))

    def _absolute_module(self, module: ModuleScope, name: str | None, level: int) -> str:
        if level == 0:
            return name or ""
        parts = module.name.split(".")
        package = parts if module.is_package else parts[:-1]
        base = package[: len(package) - (level - 1)] if level > 1 else package
        return ".".join([*base, *([name] if name else [])])

    # Evaluation.

    def _module(self, name: str, near: ModuleScope | None = None) -> ModuleScope | None:
        candidates = self.modules.get(name)
        if not candidates:
            return None
        if near is None or len(candidates) == 1:
            return candidates[0]
        return max(candidates, key=lambda m: len(_common_prefix(m.path, near.path)))

    def _binding_values(self, binding: Binding) -> frozenset[Value]:
        if binding.kind == "def":
            return frozenset({Value("def", binding.target)})
        if binding.kind == "self":
            return frozenset({Value("instance", binding.target)})
        if binding.kind == "cls":
            return frozenset({Value("def", binding.target)})
        if binding.kind == "import":
            module = self._module(binding.target, binding.scope.module)
            return frozenset({Value("module", module.name) if module else Value("external", binding.target)})
        if binding.kind == "import_from":
            return self._import_values(binding.target, binding.attr, binding.scope.module)
        if binding.kind == "annotation":
            return self._annotation(binding.expr, binding.scope)
        if binding.kind == "element":
            return self._elements(self._eval(binding.expr, binding.scope))
        return self._eval(binding.expr, binding.scope)

    def _elements(self, values: frozenset[Value]) -> frozenset[Value]:
        """What iterating over `values` yields: container elements, or what ``__iter__`` is annotated to yield."""
        found = set()
        for value in values:
            if value.kind == "iterable":
                found.add(Value("instance", value.target))
            elif value.kind == "instance":
                for method in self._member(value.target, "__iter__", instance=True):
                    found.update(Value("instance", r.target) for r in self._call_result(method) if r.kind == "iterable")
        return frozenset(found)

    def _import_values(self, module_name: str, attr: str, near: ModuleScope | None) -> frozenset[Value]:
        module = self._module(module_name, near)
        if module is not None:
            values = self._module_attribute(module, attr)
            if values:
                return values
        submodule = self._module(join(module_name, attr), near)
        if submodule is not None:
            return frozenset({Value("module", submodule.name)})
        return frozenset() if module is not None else frozenset({Value("external", join(module_name, attr))})

    def _module_attribute(self, module: ModuleScope, attr: str) -> frozenset[Value]:
        key = (id(module), attr)
        if key in self._resolving:
            return frozenset()
        self._resolving.add(key)
        try:
            bindings = module.bindings.get(attr)
            if bindings:
                return self._union(self._binding_values(b) for b in bindings)
            submodule = self._module(join(module.name, attr), module)
            if submodule is not None:
                return frozenset({Value("module", submodule.name)})
            for source in module.star_imports:
                star = self._module(source, module)
                if star is not None and not attr.startswith("_"):
                    values = self._module_attribute(star, attr)
                    if values:
                        return values
            return frozenset()
        finally:
            self._resolving.discard(key)

    def _lookup(self, name: str, scope: Scope) -> frozenset[Value]:
        current: Scope | None = scope
        while current is not None:
            if current is scope or not current.is_class:
                bindings = current.bindings.get(name)
                if bindings:
                    return self._union(self._binding_values(b) for b in bindings)
                if current.kind == "module":
                    return self._module_attribute(current, name)
            current = current.parent
        return frozenset()

    def _eval(self, expr: ast.expr | None, scope: Scope) -> frozenset[Value]:
        if expr is None:
            return frozenset()
        key = (id(expr), id(scope))
        if key in self._cache:
            return self._cache[key]
        if key in self._evaluating:
            return frozenset()
        self._evaluating.add(key)
        try:
            values = self._union(self._settle(v) for v in self._eval_uncached(expr, scope))
        finally:
            self._evaluating.discard(key)
        self._cache[key] = values
        return values

    def _settle(self, value: Value) -> frozenset[Value]:
        """Replace a property by the instances its getter is annotated to return."""
        if value.kind != "property":
            return frozenset({value})
        getter = self.scopes[value.target]
        return self._annotation(getter.node.returns, getter.parent)

    def _eval_uncached(self, expr: ast.expr, scope: Scope) -> frozenset[Value]:
        if isinstance(expr, ast.Name):
            return self._lookup(expr.id, scope)
        if isinstance(expr, ast.Attribute):
            return self._union(self._attribute(v, expr.attr) for v in self._eval(expr.value, scope))
        if isinstance(expr, ast.Call):
            if isinstance(expr.func, ast.Name) and expr.func.id == "super" and not self._lookup("super", scope):
                owner = self._enclosing_method(scope)
                return frozenset({Value("super", owner.owner.node_id)}) if owner else frozenset()
            return self._union(self._call_result(v) for v in self._eval(expr.func, scope))
        if isinstance(expr, ast.Await):
            return self._eval(expr.value, scope)
        if isinstance(expr, ast.BoolOp):
            return self._union(self._eval(v, scope) for v in expr.values)
        if isinstance(expr, ast.IfExp):
            return self._eval(expr.body, scope) | self._eval(expr.orelse, scope)
        if isinstance(expr, ast.NamedExpr):
            return self._eval(expr.value, scope)
        if isinstance(expr, ast.Subscript):
            found = set()
            for value in self._eval(expr.value, scope):
                if value.kind == "iterable":
                    found.add(Value("instance", value.target))
                elif value.kind == "instance":
                    for method in self._member(value.target, "__getitem__", instance=True):
                        found.update(self._call_result(method))
            return frozenset(found)
        return frozenset()

    @staticmethod
    def _enclosing_method(scope: Scope) -> Scope | None:
        current: Scope | None = scope
        while current is not None and current.owner is None:
            current = current.parent if current.is_function else None
        return current

    def _call_result(self, value: Value) -> frozenset[Value]:
        if value.kind == "def":
            target = self.scopes[value.target]
            if target.is_class:
                return frozenset({Value("instance", target.node_id)})
            return self._annotation(target.node.returns, target.parent)
        if value.kind == "instance":
            call = self._member(value.target, "__call__", instance=True)
            return self._union(self._call_result(v) for v in call if v.kind == "def")
        if value.kind == "external":
            return frozenset({value})
        return frozenset()

    def _annotation(self, expr: ast.expr | None, scope: Scope | None) -> frozenset[Value]:
        if expr is None or scope is None:
            return frozenset()
        if isinstance(expr, ast.Constant) and isinstance(expr.value, str):
            try:
                parsed = ast.parse(expr.value, mode="eval").body
            except SyntaxError:
                return frozenset()
            return self._annotation(parsed, scope)
        if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
            return self._annotation(expr.left, scope) | self._annotation(expr.right, scope)
        if isinstance(expr, ast.Subscript):
            head = _subscript_head(expr.value)
            items = expr.slice.elts if isinstance(expr.slice, ast.Tuple) else [expr.slice]
            if head in _TRANSPARENT_GENERICS:
                selected = items[:1] if head == "Annotated" else items
                return self._union(self._annotation(item, scope) for item in selected)
            if head in _TYPE_GENERICS:
                return frozenset(v for v in self._eval(items[0], scope) if v.kind == "def")
            if head in _CONTAINER_GENERICS:
                elements = self._annotation(items[0], scope)
                return frozenset(Value("iterable", v.target) for v in elements if v.kind == "instance")
            return frozenset()
        values = set()
        for value in self._eval(expr, scope):
            if value.kind == "external":
                values.add(value)
            elif value.kind == "def" and self.scopes[value.target].is_class:
                values.add(Value("instance", value.target))
        return frozenset(values)

    def _attribute(self, value: Value, attr: str) -> frozenset[Value]:
        if value.kind == "module":
            module = self._module(value.target)
            return self._module_attribute(module, attr) if module else frozenset()
        if value.kind == "def" and self.scopes[value.target].is_class:
            return self._member(value.target, attr, instance=False)
        if value.kind == "instance":
            return self._member(value.target, attr, instance=True)
        if value.kind == "external":
            return frozenset({Value("external", join(value.target, attr))})
        if value.kind == "super":
            return self._member(value.target, attr, instance=True, skip_self=True)
        return frozenset()

    def _member(self, class_id: str, attr: str, instance: bool, skip_self: bool = False) -> frozenset[Value]:
        mro = self.mro(class_id)
        for cls in mro[1:] if skip_self else mro:
            bindings = cls.bindings.get(attr) or (cls.attributes.get(attr) if instance else None)
            if not bindings:
                continue
            values = self._union(self._binding_values(b) for b in bindings)
            if instance:
                values = frozenset(
                    Value("property", v.target) if v.kind == "def" and _is_property(self.scopes[v.target].node) else v
                    for v in values
                )
            return values
        return frozenset()

    def mro(self, class_id: str) -> list[Scope]:
        """C3 linearisation over the repository's classes (external bases are ignored)."""
        if class_id in self._mro:
            return self._mro[class_id]
        self._mro[class_id] = [self.scopes[class_id]]
        cls = self.scopes[class_id]
        bases = [b.target for b in self.bases(cls)]
        sequences = [list(self.mro(b)) for b in bases if b != class_id] + [[self.scopes[b] for b in bases]]
        result = [cls]
        while any(sequences):
            sequences = [s for s in sequences if s]
            head = next((s[0] for s in sequences if not any(s[0] in t[1:] for t in sequences)), sequences[0][0])
            result.append(head)
            sequences = [[c for c in s if c is not head] for s in sequences]
        self._mro[class_id] = result
        return result

    def bases(self, cls: Scope) -> list[Value]:
        found = []
        for base in cls.node.bases:
            expr = base.value if isinstance(base, ast.Subscript) else base
            for value in sorted(self._eval(expr, cls.parent), key=lambda v: v.target):
                if value.kind == "def" and self.scopes[value.target].is_class and value not in found:
                    found.append(value)
        return found

    @staticmethod
    def _union(groups: Iterable[frozenset[Value]]) -> frozenset[Value]:
        merged: set[Value] = set()
        for group in groups:
            merged |= group
            if len(merged) >= _MAX_VALUES:
                break
        return frozenset(merged)

    # Linking: edges out of each scope.

    def _add_edge(self, source: str, target: str, kind: str, resolution: str = STATIC) -> None:
        key = (source, target, kind)
        if self.edges.get(key) != STATIC:
            self.edges[key] = resolution

    def _link_scope(self, scope: Scope) -> None:
        node = scope.node
        if scope.is_class:
            for base in self.bases(scope):
                self._add_edge(scope.node_id, base.target, INHERITS)
        for statement in node.body:
            self._link_node(scope, statement)
        for child in scope.children:
            self._link_scope(child)

    def _link_node(self, scope: Scope, node: ast.AST) -> None:
        if isinstance(node, _SCOPE_TYPES):
            child_id = join(scope.module.name, drop_locals(_child_qualname(scope, node.name)))
            for decorator in node.decorator_list:
                if isinstance(decorator, ast.Call):
                    self._link_node(scope, decorator)
                else:
                    self._link_call(scope, ast.Call(decorator, [], []))
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                for value in self._eval(target, scope):
                    if value.kind == "def":
                        self._add_edge(child_id, value.target, DECORATED_BY)
            if isinstance(node, ast.ClassDef):
                for expr in [*node.bases, *node.keywords]:
                    self._link_node(scope, expr)
            else:
                for expr in [*node.args.defaults, *node.args.kw_defaults]:
                    if expr is not None:
                        self._link_node(scope, expr)
            return
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            self._link_import(scope, node)
            return
        if isinstance(node, ast.AnnAssign):
            if node.value is not None:
                self._link_node(scope, node.value)
            self._link_node(scope, node.target)
            return
        self._link_protocols(scope, node)
        if isinstance(node, ast.AugAssign):
            for value in self._raw_eval(node.target, scope):
                if value.kind == "property":
                    self._add_edge(scope.node_id, value.target, CALLS)
        if isinstance(node, ast.Call):
            self._link_call(scope, node)
            for child in [*node.args, *node.keywords]:
                self._link_node(scope, child)
            func = node.func
            if isinstance(func, ast.Attribute):
                self._link_node(scope, func.value)
            elif not isinstance(func, ast.Name):
                self._link_node(scope, func)
            return
        if isinstance(node, (ast.Name, ast.Attribute)) and isinstance(node.ctx, ast.Load):
            for value in self._raw_eval(node, scope):
                if value.kind == "def":
                    self._add_edge(scope.node_id, value.target, REFERENCES)
                elif value.kind == "property":
                    self._add_edge(scope.node_id, value.target, CALLS)
            if isinstance(node, ast.Attribute):
                self._link_node(scope, node.value)
            return
        for child in ast.iter_child_nodes(node):
            self._link_node(scope, child)

    def _link_protocols(self, scope: Scope, node: ast.AST) -> None:
        """Calls Python makes implicitly: ``len(x)``, ``for``, ``with``, operators, subscripts, f-strings."""
        for operand, groups in self._protocol_uses(scope, node):
            for value in self._eval(operand, scope):
                if value.kind != "instance":
                    continue
                for alternatives in groups:
                    for dunder in alternatives:
                        methods = [m for m in self._member(value.target, dunder, instance=True) if m.kind == "def"]
                        for method in methods:
                            self._add_edge(scope.node_id, method.target, CALLS, PROTOCOL)
                        if methods:
                            break

    def _protocol_uses(self, scope: Scope, node: ast.AST) -> list[tuple[ast.expr, list[tuple[str, ...]]]]:
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.args:
            dunder = _BUILTIN_PROTOCOLS.get(node.func.id)
            if dunder and not self._lookup(node.func.id, scope):
                return [(node.args[0], [(dunder,)])]
        elif isinstance(node, (ast.For, ast.AsyncFor, ast.comprehension)):
            asynchronous = isinstance(node, ast.AsyncFor) or getattr(node, "is_async", False)
            return [(node.iter, [("__aiter__",) if asynchronous else ("__iter__",)])]
        elif isinstance(node, (ast.With, ast.AsyncWith)):
            names = ("__aenter__", "__aexit__") if isinstance(node, ast.AsyncWith) else ("__enter__", "__exit__")
            return [(item.context_expr, [(n,) for n in names]) for item in node.items]
        elif isinstance(node, (ast.BinOp, ast.AugAssign)) and type(node.op) in _BINARY_PROTOCOLS:
            dunder = _BINARY_PROTOCOLS[type(node.op)]
            if isinstance(node, ast.BinOp):
                return [(node.left, [(dunder,)])]
            return [(node.target, [(f"__i{dunder[2:]}", dunder)])]
        elif isinstance(node, ast.Compare):
            uses = []
            for left, op, right in zip([node.left, *node.comparators], node.ops, node.comparators):
                if isinstance(op, (ast.In, ast.NotIn)):
                    uses.append((right, [("__contains__",)]))
                elif type(op) in _COMPARE_PROTOCOLS:
                    uses.append((left, [(_COMPARE_PROTOCOLS[type(op)],)]))
            return uses
        elif isinstance(node, ast.Subscript) and type(node.ctx) in _SUBSCRIPT_PROTOCOLS:
            return [(node.value, [(_SUBSCRIPT_PROTOCOLS[type(node.ctx)],)])]
        elif isinstance(node, ast.FormattedValue):
            return [(node.value, [("__format__", "__str__")])]
        return []

    def _raw_eval(self, expr: ast.expr, scope: Scope) -> frozenset[Value]:
        """Like `_eval`, but keeps property values so that property reads can be linked."""
        if isinstance(expr, ast.Attribute):
            return self._union(self._attribute(v, expr.attr) for v in self._eval(expr.value, scope))
        return self._eval(expr, scope)

    def _link_call(self, scope: Scope, call: ast.Call) -> None:
        func = call.func
        values = self._raw_eval(func, scope)
        for value in values:
            for target in self._callees(value):
                self._add_edge(scope.node_id, target, CALLS)
        if isinstance(func, ast.Attribute):
            for receiver in self._eval(func.value, scope):
                if receiver.kind == "instance":
                    for override in self._overrides(receiver.target, func.attr):
                        self._add_edge(scope.node_id, override, CALLS, OVERRIDE)
        if (
            not values
            and isinstance(func, ast.Attribute)
            and isinstance(func.value, _RECEIVER_TYPES)
            and func.attr not in _BUILTIN_METHODS
            and not self._eval(func.value, scope)
        ):
            candidates = self._methods_by_name.get(func.attr, [])
            if len(candidates) == 1:
                self._add_edge(scope.node_id, candidates[0], CALLS, UNIQUE_NAME)

    def _overrides(self, class_id: str, name: str) -> list[str]:
        """Methods called `name` defined by strict subclasses of `class_id` (class hierarchy analysis)."""
        if self._subclasses is None:
            self._subclasses = defaultdict(list)
            for cls in self.class_scopes:
                for base in self.bases(cls):
                    self._subclasses[base.target].append(cls)
        found, stack, seen = [], list(self._subclasses.get(class_id, ())), set()
        while stack:
            cls = stack.pop()
            if cls.node_id in seen:
                continue
            seen.add(cls.node_id)
            stack.extend(self._subclasses.get(cls.node_id, ()))
            found += [b.target for b in cls.bindings.get(name, ()) if b.kind == "def" and self.scopes[b.target].is_function]
        return found

    def _callees(self, value: Value) -> list[str]:
        if value.kind == "property":
            return [value.target]
        if value.kind == "def":
            target = self.scopes[value.target]
            if not target.is_class:
                return [target.node_id]
            return [target.node_id, *(v.target for v in self._member(target.node_id, "__init__", instance=True) if v.kind == "def")]
        if value.kind == "instance":
            return [v.target for v in self._member(value.target, "__call__", instance=True) if v.kind == "def"]
        return []

    def _link_import(self, scope: Scope, node: ast.Import | ast.ImportFrom) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                module = self._module(alias.name, scope.module)
                if module is not None:
                    self._add_edge(scope.node_id, module.node_id, IMPORTS)
            return
        source = self._absolute_module(scope.module, node.module, node.level)
        for alias in node.names:
            if alias.name == "*":
                module = self._module(source, scope.module)
                values = frozenset({Value("module", module.name)}) if module else frozenset()
            else:
                values = self._import_values(source, alias.name, scope.module)
            for value in values:
                if value.kind == "def":
                    self._add_edge(scope.node_id, value.target, IMPORTS)
                elif value.kind == "module":
                    module = self._module(value.target, scope.module)
                    self._add_edge(scope.node_id, module.node_id, IMPORTS)


def _expressions(stmt: ast.stmt) -> list[ast.expr]:
    """The expressions a statement evaluates itself, leaving out nested statement blocks."""
    found = []
    for _, value in ast.iter_fields(stmt):
        for item in value if isinstance(value, list) else [value]:
            if isinstance(item, ast.expr):
                found.append(item)
            elif isinstance(item, ast.withitem):
                found.append(item.context_expr)
    return found


def _child_qualname(parent: Scope, name: str) -> str:
    if parent.is_class:
        return f"{parent.qualname}.{name}"
    if parent.is_function:
        return f"{parent.qualname}.{LOCALS}.{name}"
    return name


def _common_prefix(a: str, b: str) -> str:
    size = 0
    for x, y in zip(a, b):
        if x != y:
            break
        size += 1
    return a[:size]


def _stub(node: ast.FunctionDef | ast.AsyncFunctionDef) -> ast.stmt:
    stub = type(node)(**{f: getattr(node, f) for f in node._fields})
    stub.body = [ast.Expr(ast.Constant(...))]
    return stub


def _class_text(node: ast.ClassDef) -> str:
    """Decorators, header, docstring, class-level fields and method signatures."""
    body: list[ast.stmt] = []
    docstring = ast.get_docstring(node)
    if docstring is not None:
        body.append(ast.Expr(ast.Constant(docstring)))
    for stmt in node.body:
        if isinstance(stmt, _FUNCTION_TYPES):
            body.append(_stub(stmt))
        elif isinstance(stmt, ast.ClassDef):
            body.append(ast.ClassDef(stmt.name, stmt.bases, stmt.keywords, [ast.Expr(ast.Constant(...))], stmt.decorator_list, []))
        elif isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            body.append(stmt)
    summary = ast.ClassDef(node.name, node.bases, node.keywords, body or [ast.Pass()], node.decorator_list, [])
    return ast.unparse(ast.fix_missing_locations(summary))


def _module_text(tree: ast.Module) -> str:
    """Docstring, imports, top-level assignments and stubs of top-level definitions."""
    body: list[ast.stmt] = []
    docstring = ast.get_docstring(tree)
    if docstring is not None:
        body.append(ast.Expr(ast.Constant(docstring)))
    for stmt in tree.body:
        if isinstance(stmt, _FUNCTION_TYPES):
            body.append(_stub(stmt))
        elif isinstance(stmt, ast.ClassDef):
            body.append(ast.ClassDef(stmt.name, stmt.bases, stmt.keywords, [ast.Expr(ast.Constant(...))], stmt.decorator_list, []))
        elif isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Assign, ast.AnnAssign)):
            body.append(stmt)
    return ast.unparse(ast.fix_missing_locations(ast.Module(body, [])))


def build_graph(root: Path, repo_name: str = "") -> nx.MultiDiGraph:
    """Build the callproof graph of the checkout at `root`."""
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 20_000))
    return GraphBuilder(parse_checkout(root), repo_name).build()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("root", type=Path, help="repository checkout")
    parser.add_argument("output", type=Path, help="graph JSON to write")
    parser.add_argument("--repo-name", default="", help="graph-level repo_name, e.g. fastapi_<commit>")
    args = parser.parse_args()
    save_graph(build_graph(args.root, args.repo_name), args.output)


if __name__ == "__main__":
    main()
