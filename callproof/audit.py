"""Metrics for code graphs: structure, agreement with the source text, and agreement with traced executions.

Graphs with different id conventions are compared in a neutral *key space*: a key is
``(path, collapsed qualname)``, the granularity at which the tracer records calls (nested functions,
lambdas and comprehensions are folded into the enclosing named definition, module-level code is
``<module>``). Each graph node maps to the keys of the definitions it stands for, so a provided node
that merges two colliding definitions maps to both, and a nested function maps to its enclosing one.

Counts are returned as flat dicts of integers so that per-graph results can be summed and ratios
taken afterwards (micro-averages).
"""

from __future__ import annotations

import ast
from collections import Counter, defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass

import networkx as nx

from callproof.build import INHERITS
from callproof.naming import MODULE_QUALNAME, Definition, RepoIndex, collapse_qualname
from callproof.trace import is_test_path

Key = tuple[str, str]
NON_LIBRARY_DIRS = {"docs", "docs_src", "examples", "scripts", "benchmarks", "tools", "tests", "test"}
CONSTRUCTOR_NAMES = {"__init__", "__new__", "__post_init__"}
EDGE_CATEGORIES = ("self_method", "constructor", "module_function", "cross_module", "async", "via_test", "nested")


@dataclass
class GraphView:
    """A graph together with the neutral keys of the definitions behind each node."""

    graph: nx.MultiDiGraph
    keys: dict[str, frozenset[Key]]
    definitions: dict[str, list[Definition]]

    @classmethod
    def harness(cls, graph: nx.MultiDiGraph, index: RepoIndex) -> GraphView:
        definitions = {node: index.from_harness_id(node) for node in graph}
        keys = {node: frozenset((d.path, d.collapsed_qualname) for d in defs) for node, defs in definitions.items()}
        return cls(graph, keys, definitions)

    @classmethod
    def callproof(cls, graph: nx.MultiDiGraph, index: RepoIndex) -> GraphView:
        definitions, keys = {}, {}
        for node, data in graph.nodes(data=True):
            definitions[node] = index.from_node_id(node)
            keys[node] = frozenset({(data["path"], collapse_qualname(data["qualname"]))})
        return cls(graph, keys, definitions)

    def key_nodes(self) -> dict[Key, set[str]]:
        found: dict[Key, set[str]] = defaultdict(set)
        for node, keys in self.keys.items():
            for key in keys:
                found[key].add(node)
        return found

    def key_edges(self, types: Iterable[str] | None = None) -> set[tuple[Key, Key]]:
        wanted = set(types) if types is not None else None
        edges = set()
        for u, v, data in self.graph.edges(data=True):
            if wanted is None or data.get("type") in wanted:
                edges.update((a, b) for a in self.keys[u] for b in self.keys[v])
        return edges


# Structure.


def structure_stats(view: GraphView, index: RepoIndex, harness: bool) -> dict[str, int]:
    """Node and edge counts, direction symmetry, and coverage of the checkout's definitions."""
    graph = view.graph
    stats: Counter[str] = Counter(nodes=graph.number_of_nodes(), edges=graph.number_of_edges())
    pairs = Counter((u, v) for u, v in graph.edges() if u != v)
    stats["self_loops"] = sum(1 for u, v in graph.edges() if u == v)
    stats["distinct_pairs"] = len(pairs)
    stats["pairs_with_reverse"] = sum(1 for u, v in pairs if (v, u) in pairs)
    neighbours: dict[str, set[str]] = defaultdict(set)
    for u, v in pairs:
        neighbours[u].add(v)
        neighbours[v].add(u)
    stats["nodes_over_50_neighbours"] = sum(1 for n in neighbours.values() if len(n) > 50)
    stats["parallel_edges"] = sum(n - 1 for n in Counter((u, v, d.get("type")) for u, v, d in graph.edges(data=True)).values())
    for _, _, data in graph.edges(data=True):
        stats[f"edges_{data.get('type')}"] += 1
        if "resolution" in data:
            stats[f"resolution_{data['resolution']}"] += 1

    present = set(graph)
    for definition in index.definitions:
        node = definition.harness_id if harness else definition.node_id
        stats["defs"] += 1
        stats[f"defs_{definition.kind}"] += 1
        stats["defs_with_node"] += node in present
        stats[f"defs_{definition.kind}_with_node"] += node in present
        stats["defs_nested"] += "<locals>" in definition.qualname
        if definition.is_async:
            stats["async_defs"] += 1
            stats["async_defs_with_node"] += node in present
        if harness and definition.harness_id is not None:
            stats["defs_id_not_dotted_qualname"] += definition.harness_id != definition.node_id
            top_level_method = definition.kind == "method" and definition.qualname.count(".") == 1
            stats["methods_lost_class"] += top_level_method and definition.harness_id != definition.node_id
    owners = Counter(d.harness_id if harness else d.node_id for d in index.definitions)
    stats["ids_shared"] = sum(1 for n in owners.values() if n > 1)
    stats["defs_sharing_id"] = sum(n for node, n in owners.items() if n > 1 and node is not None)
    stats["modules"] = len(index.modules)
    if not harness:
        stats["modules_with_node"] = sum(index.node_id(path, MODULE_QUALNAME) in present for path in index.modules)
    return dict(stats)


def is_library_path(path: str) -> bool:
    """Code shipped by the project, as opposed to tests, docs examples and maintenance scripts."""
    return not is_test_path(path) and path.split("/", 1)[0] not in NON_LIBRARY_DIRS


def isolation_stats(view: GraphView, call_pairs: set[frozenset[str]]) -> dict[str, int]:
    """Function and method nodes that take part in no call relation, overall and in library code."""
    linked = {node for pair in call_pairs for node in pair}
    stats: Counter[str] = Counter()
    for node, defs in view.definitions.items():
        if not {d.kind for d in defs} & {"function", "method", "async_function", "async_method"}:
            continue
        for prefix in ("", "lib_") if any(is_library_path(d.path) for d in defs) else ("",):
            stats[f"{prefix}callable_nodes"] += 1
            stats[f"{prefix}callable_nodes_without_calls"] += node not in linked
    return dict(stats)


# Agreement with the source text.


@dataclass(frozen=True)
class SourceFacts:
    """Names a definition's code calls, loads and inherits from (by terminal name)."""

    called: frozenset[str]
    loaded: frozenset[str]
    bases: frozenset[str]


def _terminal(expr: ast.expr) -> str | None:
    if isinstance(expr, ast.Name):
        return expr.id
    if isinstance(expr, ast.Attribute):
        return expr.attr
    if isinstance(expr, ast.Subscript):
        return _terminal(expr.value)
    return None


def _module_level(tree: ast.Module) -> list[ast.AST]:
    """Module statements with definition bodies left out (their decorators, bases and defaults stay)."""
    nodes: list[ast.AST] = []
    for stmt in tree.body:
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
            nodes += [*stmt.decorator_list, *stmt.args.defaults]
        elif isinstance(stmt, ast.ClassDef):
            nodes += [*stmt.decorator_list, *stmt.bases, *stmt.keywords]
        else:
            nodes.append(stmt)
    return nodes


def source_facts(node: ast.AST) -> SourceFacts:
    called, loaded = set(), set()
    roots = _module_level(node) if isinstance(node, ast.Module) else [node]
    for child in (n for root in roots for n in ast.walk(root)):
        if isinstance(child, ast.Call) and (name := _terminal(child.func)):
            called.add(name)
        elif isinstance(child, (ast.Name, ast.Attribute)) and isinstance(child.ctx, ast.Load):
            loaded.add(_terminal(child))
    bases = {_terminal(b) for b in node.bases} - {None} if isinstance(node, ast.ClassDef) else set()
    return SourceFacts(frozenset(called), frozenset(loaded), frozenset(bases))


class SourceChecker:
    """Explains an edge ``u -> v`` by what ``u``'s source does with ``v``'s name."""

    RELATIONS = ("contains", "inherits", "instantiates", "calls", "references")

    def __init__(self, view: GraphView, index: RepoIndex) -> None:
        self.view = view
        self.index = index
        self._facts: dict[str, list[tuple[Definition, SourceFacts]]] = {}

    def facts(self, node: str) -> list[tuple[Definition, SourceFacts]]:
        if node not in self._facts:
            data = self.view.graph.nodes[node]
            if data.get("kind") == "module":
                module = Definition(data["path"], MODULE_QUALNAME, "module", 1, data.get("end_lineno", 1), None, node)
                self._facts[node] = [(module, source_facts(self.index.trees[data["path"]]))]
            else:
                self._facts[node] = [
                    (d, source_facts(self.index.ast_nodes[(d.path, d.qualname)])) for d in self.view.definitions.get(node, [])
                ]
        return self._facts[node]

    def relation(self, source: str, target: str) -> str | None:
        """The strongest relation of `source` towards `target` supported by the source text, if any."""
        targets = self.view.definitions.get(target, [])
        if not targets:
            return None
        name = targets[0].qualname.rsplit(".", 1)[-1]
        target_is_class = any(d.kind == "class" for d in targets)
        if target.startswith(f"{source}.") and any(d.kind == "class" for d, _ in self.facts(source)):
            return "contains"
        for definition, facts in self.facts(source):
            if definition.kind == "class" and target_is_class and name in facts.bases:
                return "inherits"
        if any(name in facts.called for _, facts in self.facts(source)):
            return "instantiates" if target_is_class else "calls"
        if target.rsplit(".", 1)[-1] == "__init__":
            owner = target.rsplit(".", 2)[-2]
            if any(owner in facts.called for _, facts in self.facts(source)):
                return "instantiates"
        if any(name in facts.loaded for _, facts in self.facts(source)):
            return "references"
        return None


def direction_stats(view: GraphView, checker: SourceChecker, types: Iterable[str] | None = None) -> tuple[dict[str, int], list[dict]]:
    """Classify each directed edge by the relation its source text supports, forwards or only backwards."""
    wanted = set(types) if types is not None else None
    stats: Counter[str] = Counter()
    records = []
    seen = set()
    for u, v, data in view.graph.edges(data=True):
        if u == v or (wanted is not None and data.get("type") not in wanted) or (u, v) in seen:
            continue
        seen.add((u, v))
        forward, backward = checker.relation(u, v), checker.relation(v, u)
        stats["edges_checked"] += 1
        if forward:
            stats["forward_supported"] += 1
            stats[f"forward_{forward}"] += 1
        elif backward:
            stats["only_reverse_supported"] += 1
            stats[f"reverse_{backward}"] += 1
        else:
            stats["unsupported"] += 1
        records.append({"source": u, "target": v, "type": data.get("type"), "forward": forward, "backward": backward})
    return dict(stats), records


def pair_semantics(view: GraphView, checker: SourceChecker) -> tuple[dict[str, int], set[frozenset[str]]]:
    """Label each unordered connected pair with the relation that explains it (either direction).

    Returns the label counts and the pairs explained by a call or an instantiation.
    """
    stats: Counter[str] = Counter()
    call_pairs = set()
    for pair in {frozenset((u, v)) for u, v in view.graph.edges() if u != v}:
        u, v = sorted(pair)
        relations = {checker.relation(u, v), checker.relation(v, u)} - {None}
        label = next((r for r in SourceChecker.RELATIONS if r in relations), "unexplained")
        stats[f"pairs_{label}"] += 1
        if label in ("calls", "instantiates"):
            call_pairs.add(pair)
    stats["pairs"] = sum(stats.values())
    return dict(stats), call_pairs


# Agreement with traced executions.


def parse_test_id(test_id: str) -> Key:
    path, _, rest = test_id.partition("::")
    return path, ".".join(part.split("[", 1)[0] for part in rest.split("::") if part)


def class_ancestors(view: GraphView) -> dict[Key, set[Key]]:
    """Transitive base classes per class key, from ``inherits`` edges."""
    parents: dict[Key, set[Key]] = defaultdict(set)
    for u, v, data in view.graph.edges(data=True):
        if data.get("type") == INHERITS:
            for a in view.keys[u]:
                parents[a].update(view.keys[v])
    closure: dict[Key, set[Key]] = {}
    for start, bases in parents.items():
        seen, stack = set(), list(bases)
        while stack:
            key = stack.pop()
            if key not in seen:
                seen.add(key)
                stack.extend(parents.get(key, ()))
        closure[start] = seen
    return closure


def edge_categories(edge: list, index: RepoIndex, ancestors: dict[Key, set[Key]]) -> list[str]:
    caller_path, caller, callee_path, callee, _count, via_test, nested = edge[:7]
    categories = []
    caller_class, _, _ = caller.rpartition(".")
    callee_class, _, callee_name = callee.rpartition(".")
    callee_def = index.lookup(callee_path, callee)
    caller_def = index.lookup(caller_path, caller)
    if callee_class and caller_class and callee_name not in CONSTRUCTOR_NAMES:
        same = (caller_path, caller_class) == (callee_path, callee_class)
        if same or (callee_path, callee_class) in ancestors.get((caller_path, caller_class), ()):
            categories.append("self_method")
    if callee_name in CONSTRUCTOR_NAMES or (callee_def is not None and callee_def.kind == "class"):
        categories.append("constructor")
    if callee_def is not None and callee_def.kind in ("function", "async_function") and "." not in callee:
        categories.append("module_function")
    if caller_path != callee_path:
        categories.append("cross_module")
    if any(d is not None and d.is_async for d in (caller_def, callee_def)):
        categories.append("async")
    if via_test:
        categories.append("via_test")
    if nested:
        categories.append("nested")
    return categories


def dynamic_edge_stats(
    edges: list[list], view: GraphView, index: RepoIndex, ancestors: dict[Key, set[Key]], types: Iterable[str] | None = None
) -> dict[str, int]:
    """Recall of traced caller -> callee pairs: as an edge in either direction, and in the right direction."""
    directed = view.key_edges(types)
    nodes = view.key_nodes()
    stats: Counter[str] = Counter()
    for edge in edges:
        caller, callee = (edge[0], edge[1]), (edge[2], edge[3])
        if caller == callee:
            continue
        forward = (caller, callee) in directed
        either = forward or (callee, caller) in directed
        mapped = caller in nodes and callee in nodes
        categories = edge_categories(edge, index, ancestors)
        if "constructor" in categories:
            owner = (callee[0], callee[1].rpartition(".")[0]) if callee[1].rpartition(".")[2] in CONSTRUCTOR_NAMES else callee
            stats["constructor_class_either"] += either or (caller, owner) in directed or (owner, caller) in directed
        for category in ("all", *categories):
            stats[f"{category}_n"] += 1
            stats[f"{category}_mapped"] += mapped
            stats[f"{category}_either"] += either
            stats[f"{category}_forward"] += forward
    return dict(stats)


def node_coverage(keys: Iterable[Key], view: GraphView, prefix: str) -> dict[str, int]:
    nodes = view.key_nodes()
    stats: Counter[str] = Counter()
    for key in keys:
        stats[f"{prefix}_n"] += 1
        stats[f"{prefix}_covered"] += key in nodes
    return dict(stats)


def reachability(
    view: GraphView, sources: Iterable[Key], targets: Iterable[Key], max_hops: int = 3, types: Iterable[str] | None = None
) -> dict[str, int]:
    """How many target keys lie within k undirected hops of any source key, for k = 1..max_hops."""
    wanted = set(types) if types is not None else None
    adjacency: dict[str, set[str]] = defaultdict(set)
    for u, v, data in view.graph.edges(data=True):
        if wanted is None or data.get("type") in wanted:
            adjacency[u].add(v)
            adjacency[v].add(u)
    nodes = view.key_nodes()
    start = {n for key in sources for n in nodes.get(key, ())}
    distance = {n: 0 for n in start}
    queue = deque(start)
    while queue:
        node = queue.popleft()
        if distance[node] == max_hops:
            continue
        for neighbour in adjacency[node]:
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    stats: Counter[str] = Counter()
    for key in targets:
        stats["targets"] += 1
        hops = min((distance[n] for n in nodes.get(key, ()) if n in distance), default=None)
        for k in range(1, max_hops + 1):
            stats[f"within_{k}"] += hops is not None and hops <= k
    stats["sources_found"] = len(start)
    return dict(stats)


def groundtruth_stats(record: dict, view: GraphView, index: RepoIndex, ancestors: dict[Key, set[Key]], types: Iterable[str] | None = None) -> dict[str, int]:
    """All execution-based metrics of one task for one graph."""
    stats: Counter[str] = Counter()
    gold = record["gold_functions"]
    for kind in ("modified", "added"):
        defs = [(g["path"], g["collapsed_qualname"]) for g in gold if g["kind"] == kind and g["node_type"] != "module"]
        stats.update(node_coverage(defs, view, f"gold_{kind}"))
        modules = [(g["path"], MODULE_QUALNAME) for g in gold if g["kind"] == kind and g["node_type"] == "module"]
        stats.update(node_coverage(modules, view, f"gold_{kind}_module"))
    stats.update({f"dyn_{k}": v for k, v in dynamic_edge_stats(record["dyn_edges"], view, index, ancestors, types).items()})
    executed = [(p, q) for p, q, *_ in record["executed"]]
    stats.update(node_coverage(executed, view, "executed"))
    stats.update(node_coverage([k for k in executed if k[1] != MODULE_QUALNAME], view, "executed_defs"))
    tests = [parse_test_id(t) for t in record["f2p_tests"]]
    stats.update(node_coverage(tests, view, "f2p_tests"))
    targets = [(g["path"], g["collapsed_qualname"]) for g in gold if g["kind"] == "modified" and g["node_type"] != "module"]
    test_code = [key for key in executed if is_test_path(key[0])]
    for name, sources in (("reach", tests), ("reachexec", test_code)):
        if sources and targets:
            reach = reachability(view, sources, targets, types=types)
            stats[f"{name}_tasks"] += 1
            if reach.pop("sources_found"):
                stats[f"{name}_tasks_with_source"] += 1
                stats.update({f"{name}_{k}": v for k, v in reach.items()})
    return dict(stats)
