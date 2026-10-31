"""Precision proxy, density-normalised reachability and a model-free graph walker, provided graph vs Callproof.

Writes ``results/audit/EXTRA.md``, the per-task and summary CSVs next to it and ``project/docs/reachability.png``.

1. Precision proxy: for definitions executed during a task's F2P tests, the share of their static out-neighbours
   that also ran. Unexecuted-but-real calls count as misses, so the share is a lower bound on precision.
2. Reachability: share of gold modified definitions within k undirected hops of a start node, with the number of
   nodes the k-hop ball contains (the cost of the walk).
3. Walker: breadth-first expansion from the top BM25 hits using exactly what ``get_code_neighbors`` returns.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from collections import defaultdict, deque
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from callproof import audit  # noqa: E402
from callproof.dataset import DEFAULT_DATA_DIR, Dataset  # noqa: E402
from callproof.graphio import load_graph  # noqa: E402
from callproof.naming import MODULE_QUALNAME, RepoIndex, collapse_qualname, parse_checkout  # noqa: E402
from callproof.retrieve import RepoIndex as BM25Index  # noqa: E402
from callproof.retrieve import issue_query  # noqa: E402
from callproof.tools import CodeGraph, graph_key  # noqa: E402
from callproof.trace import is_test_path  # noqa: E402

MAX_HOPS = 4
SEEDS = 3
BUDGETS = (5, 10, 20)
BOOTSTRAP_DRAWS = 10_000
CALLS_CIC = ("calls", "contains", "inherits")
REACH_VARIANTS = {
    "provided": ("prov", None, False),
    "callproof calls": ("ours", ("calls",), False),
    "callproof calls+contains+inherits": ("ours", CALLS_CIC, False),
    "callproof all edges": ("ours", None, False),
    "callproof calls, directed": ("ours", ("calls",), True),
}
WALK_VARIANTS = {
    "provided": ("prov", None),
    "callproof all edges": ("ours", None),
    "callproof calls+contains+inherits": ("ours", CALLS_CIC),
}
GRAPH_LABELS = {"prov": "provided (all neighbours)", "ours": "callproof (calls edges)"}
SOURCES = {"f2p": "F2P test present at base", "exec": "executed test code present at base"}

Key = audit.Key


def adjacency(graph: nx.MultiDiGraph, types: Iterable[str] | None, directed: bool) -> dict[str, set[str]]:
    wanted = set(types) if types is not None else None
    found: dict[str, set[str]] = defaultdict(set)
    for u, v, kind in graph.edges(data="type"):
        if u != v and (wanted is None or kind in wanted):
            found[u].add(v)
            if not directed:
                found[v].add(u)
    return found


def hop_distances(adj: dict[str, set[str]], start: Iterable[str], max_hops: int) -> dict[str, int]:
    distance = dict.fromkeys(start, 0)
    queue = deque(distance)
    while queue:
        node = queue.popleft()
        if distance[node] == max_hops:
            continue
        for neighbour in adj.get(node, ()):
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)
    return distance


def nodes_of(key_nodes: dict[Key, set[str]], keys: Iterable[Key]) -> set[str]:
    return {n for key in keys for n in key_nodes.get(key, ())}


def task_keys(record: dict) -> dict[str, list[Key]]:
    gold = [(g["path"], g["collapsed_qualname"]) for g in record["gold_functions"] if g["kind"] == "modified" and g["node_type"] != "module"]
    executed = [(p, q) for p, q, *_ in record["executed"]]
    return {
        "gold": gold,
        "executed": executed,
        "f2p": [audit.parse_test_id(t) for t in record["f2p_tests"]],
        "exec": [k for k in executed if is_test_path(k[0])],
    }


def is_traced(record: dict) -> bool:
    return any(e[:2] != e[2:4] for e in record["dyn_edges"])


def precision_rows(task: dict, keys: dict[str, list[Key]], views: dict[str, audit.GraphView]) -> list[dict]:
    executed = set(keys["executed"])
    rows = []
    for name, view in views.items():
        types = None if name == "prov" else ("calls",)
        out = adjacency(view.graph, types, directed=True)
        ran = {n for n, ks in view.keys.items() if ks & executed}
        sources = {n for n in ran if any(q != MODULE_QUALNAME for _, q in view.keys[n])}
        edges = sum(len(out.get(n, ())) for n in sources)
        hits = sum(1 for n in sources for m in out.get(n, ()) if m in ran)
        rows.append({"graph": GRAPH_LABELS[name], "instance_id": task["instance_id"], "executed_nodes": len(sources), "edges": edges, "hits": hits})
    return rows


def reach_rows(task: dict, keys: dict[str, list[Key]], views: dict[str, audit.GraphView], adjs: dict[str, dict]) -> list[dict]:
    rows = []
    for source, _label in SOURCES.items():
        for variant, (name, _types, _directed) in REACH_VARIANTS.items():
            key_nodes = views[name].key_nodes()
            start = nodes_of(key_nodes, keys[source])
            gold_nodes = [key_nodes.get(k, set()) for k in keys["gold"]]
            distance = hop_distances(adjs[variant], start, MAX_HOPS)
            base = {"instance_id": task["instance_id"], "source": source, "variant": variant, "start_nodes": len(start), "targets": len(gold_nodes)}
            for k in range(1, MAX_HOPS + 1):
                reached = sum(any(distance.get(n, MAX_HOPS + 1) <= k for n in nodes) for nodes in gold_nodes)
                ball = sum(1 for d in distance.values() if d <= k)
                rows.append({**base, "k": k, "reached": reached, "ball": ball})
    return rows


def bm25_seeds(task: dict, dataset: Dataset, provided: nx.MultiDiGraph) -> list[Key]:
    index = BM25Index(provided, dataset.repo(task))
    locations = index.top_locations(index.bm25_scores(issue_query(task)), SEEDS, exclude_tests=True)
    return [(path, collapse_qualname(qualname)) for path, qualname in (loc.split("::", 1) for loc in locations)]


def walk(code_graph: CodeGraph, start: list[str], budget: int) -> list[frozenset[str]]:
    """Breadth-first expansion through `get_code_neighbors`; element i is the set of nodes seen after i calls."""
    prefix = f"{code_graph.repo_key};"
    visited = dict.fromkeys(start)
    queue = deque(start)
    expanded: set[str] = set()
    trail = [frozenset(visited)]
    while queue and len(trail) <= budget:
        node = queue.popleft()
        if node in expanded:
            continue
        expanded.add(node)
        reply = json.loads(code_graph.get_code_neighbors(f"{prefix}{node}"))
        for name in reply.get("neighbors", []):
            name = name.removeprefix(prefix)
            if name not in visited:
                visited[name] = None
                queue.append(name)
        trail.append(frozenset(visited))
    return trail


def walker_rows(task: dict, keys: dict[str, list[Key]], views: dict[str, audit.GraphView], code_graphs: dict[str, CodeGraph], seeds: list[Key]) -> list[dict]:
    rows = []
    for variant, (name, types) in WALK_VARIANTS.items():
        code_graph = code_graphs[name]
        if types is not None:
            wanted = set(types)
            code_graph = CodeGraph.__new__(CodeGraph)
            code_graph.__dict__.update(code_graphs[name].__dict__)
            full = code_graphs[name].graph
            code_graph.graph = nx.subgraph_view(full, filter_edge=lambda u, v, k, g=full: g[u][v][k].get("type") in wanted)
        key_nodes = views[name].key_nodes()
        start = sorted(nodes_of(key_nodes, seeds))
        gold = nodes_of(key_nodes, keys["gold"])
        trail = walk(code_graph, start, max(BUDGETS))
        for budget in (0, *BUDGETS):
            visited = trail[min(budget, len(trail) - 1)]
            rows.append({
                "instance_id": task["instance_id"], "variant": variant, "budget": budget, "hit": bool(gold & visited),
                "listed": len(visited), "gold_in_graph": bool(gold), "seed_nodes": len(start),
            })
    return rows


def analyse_commit(dataset: Dataset, tasks: list[dict], graphs_dir: Path, groundtruth_dir: Path) -> dict:
    first = tasks[0]
    key = graph_key(first["repo"], first["base_commit"])
    index = RepoIndex(parse_checkout(dataset.repo(first).root))
    provided = load_graph(dataset.graph_path(first))
    ours = load_graph(graphs_dir / f"{key}.json")
    views = {"prov": audit.GraphView.harness(provided, index), "ours": audit.GraphView.callproof(ours, index)}
    adjs = {variant: adjacency(views[name].graph, types, directed) for variant, (name, types, directed) in REACH_VARIANTS.items()}
    code_graphs = {"prov": CodeGraph(dataset.graph_path(first), None, key), "ours": CodeGraph(graphs_dir / f"{key}.json", None, key)}
    result: dict[str, list] = {"precision": [], "reach": [], "walker": []}
    for task in tasks:
        path = groundtruth_dir / f"{task['instance_id']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        keys = task_keys(record)
        if keys["gold"]:
            result["walker"] += walker_rows(task, keys, views, code_graphs, bm25_seeds(task, dataset, provided))
        if record.get("status") == "ok" and is_traced(record):
            result["precision"] += precision_rows(task, keys, views)
            if keys["gold"]:
                result["reach"] += reach_rows(task, keys, views, adjs)
    return result


def bootstrap(per_task: np.ndarray, statistic, rng: np.random.Generator) -> tuple[float, float, float]:
    """Point estimate and percentile 95% interval of `statistic(rows)` over tasks resampled with replacement."""
    picks = rng.integers(0, len(per_task), size=(BOOTSTRAP_DRAWS, len(per_task)))
    draws = np.array([statistic(per_task[p]) for p in picks])
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(statistic(per_task)), float(low), float(high)


def pooled_share(rows: np.ndarray) -> float:
    return rows[:, 1].sum() / max(rows[:, 0].sum(), 1)


def macro_share(rows: np.ndarray) -> float:
    valid = rows[rows[:, 0] > 0]
    return float((valid[:, 1] / valid[:, 0]).mean()) if len(valid) else float("nan")


def summarise_precision(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = []
    for name, group in frame.groupby("graph", sort=False):
        counts = group[["edges", "hits"]].to_numpy(float)
        pooled = bootstrap(counts, pooled_share, rng)
        macro = bootstrap(counts, macro_share, rng)
        out.append({
            "graph": name, "tasks": len(group), "edges": int(group.edges.sum()), "executed_node_visits": int(group.executed_nodes.sum()),
            "mean_out_degree_executed": group.edges.sum() / group.executed_nodes.sum(),
            "pooled": pooled[0], "pooled_lo": pooled[1], "pooled_hi": pooled[2],
            "macro": macro[0], "macro_lo": macro[1], "macro_hi": macro[2],
        })
    return pd.DataFrame(out)


def summarise_reach(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    keep = frame.groupby(["source", "instance_id"]).apply(lambda g: (g.start_nodes > 0).all(), include_groups=False)
    frame = frame[frame.set_index(["source", "instance_id"]).index.map(keep)]
    out = []
    for (source, variant, k), group in frame.groupby(["source", "variant", "k"], sort=False):
        counts = group[["targets", "reached"]].to_numpy(float)
        macro = bootstrap(counts, macro_share, rng)
        out.append({
            "source": source, "variant": variant, "k": k, "tasks": len(group), "gold": int(group.targets.sum()),
            "pooled": group.reached.sum() / group.targets.sum(), "macro": macro[0], "macro_lo": macro[1], "macro_hi": macro[2],
            "mean_ball": group.ball.mean(), "gold_per_1000_nodes": 1000 * group.reached.sum() / group.ball.sum(),
        })
    return pd.DataFrame(out)


def summarise_walker(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    out = []
    for (variant, budget), group in frame.groupby(["variant", "budget"], sort=False):
        hits = group[["hit"]].to_numpy(float)
        point, low, high = bootstrap(hits, lambda rows: rows[:, 0].mean(), rng)
        out.append({
            "variant": variant, "budget": budget, "tasks": len(group), "hit_rate": point, "lo": low, "hi": high,
            "mean_listed": group.listed.mean(), "median_listed": group.listed.median(),
            "gold_node_exists": group.gold_in_graph.mean(),
        })
    return pd.DataFrame(out)


def reach_figure(summary: pd.DataFrame, path: Path) -> None:
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(10, 4), sharey=True)
    for axis, (source, label) in zip(axes, SOURCES.items()):
        for variant, group in summary[summary.source == source].groupby("variant", sort=False):
            axis.plot(group.mean_ball, group.macro, marker="o", label=variant)
            for _, row in group.iterrows():
                axis.annotate(str(int(row.k)), (row.mean_ball, row.macro), textcoords="offset points", xytext=(4, -9), fontsize=8)
        axis.set_xscale("log")
        axis.set_xlabel("mean nodes within k hops (log scale)")
        axis.set_title(f"start: {label}", fontsize=9)
        axis.grid(alpha=0.3)
    axes[0].set_ylabel("gold modified definitions reached (task macro average)")
    axes[0].legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)


def pct(x: float) -> str:
    return f"{100 * x:.1f}%"


def report(precision: pd.DataFrame, reach: pd.DataFrame, walker: pd.DataFrame, n_walk_tasks: int) -> str:
    lines = [
        "# Extra analyses: precision proxy, normalised reachability, model-free walker",
        "",
        "## 1. Precision proxy (static out-neighbours of executed definitions that also ran)",
        "",
        "Bias: a real call that the F2P tests simply do not exercise counts as a miss, so every share is a lower bound on",
        "precision; the provided graph is symmetric, so its out-neighbours are all neighbours (callers included).",
        "Callproof uses `calls` edges only. Intervals are task-bootstrap 95%.",
        "",
        "| graph | tasks | edges from executed nodes | mean out-degree | pooled | macro |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in precision.iterrows():
        lines.append(
            f"| {r.graph} | {r.tasks} | {r.edges} | {r.mean_out_degree_executed:.1f} | "
            f"{pct(r.pooled)} [{pct(r.pooled_lo)}, {pct(r.pooled_hi)}] | {pct(r.macro)} [{pct(r.macro_lo)}, {pct(r.macro_hi)}] |"
        )
    lines += ["", "## 2. Reachability of gold modified definitions within k undirected hops", ""]
    lines += ["Tasks are those whose start nodes exist in every variant, so all rows share one task set.", ""]
    for source, label in SOURCES.items():
        part = reach[reach.source == source]
        lines += [
            f"Start: {label} ({part.tasks.max()} tasks, {part[part.k == 1].gold.max()} gold definitions).",
            "",
            "| variant | k | pooled | macro [95% CI] | mean nodes in ball | gold reached per 1,000 nodes |",
            "|---|---|---|---|---|---|",
        ]
        for _, r in part.iterrows():
            lines.append(
                f"| {r.variant} | {r.k} | {pct(r.pooled)} | {pct(r.macro)} [{pct(r.macro_lo)}, {pct(r.macro_hi)}] | "
                f"{r.mean_ball:.0f} | {r.gold_per_1000_nodes:.2f} |"
            )
        lines.append("")
    lines += [
        f"## 3. Model-free walker ({n_walk_tasks} tasks with a modified gold function)",
        "",
        f"Start: top-{SEEDS} BM25 function hits for the issue text (tests and docs_src excluded; one seed list from the provided",
        "graph's index, shared by all variants). Expansion: breadth-first through `CodeGraph.get_code_neighbors` (merged,",
        "alphabetical, 50-neighbour and 5,000-character caps, no edge types), one call per expanded node. Hit: any gold",
        "modified definition among the seed or listed nodes. Budget 0 is the seeds alone. Listed = distinct nodes seen.",
        "",
        "| variant | budget | tasks hit [95% CI] | mean listed | median listed | tasks whose gold has a node |",
        "|---|---|---|---|---|---|",
    ]
    for _, r in walker.iterrows():
        lines.append(
            f"| {r.variant} | {r.budget} | {pct(r.hit_rate)} [{pct(r.lo)}, {pct(r.hi)}] | {r.mean_listed:.0f} | {r.median_listed:.0f} | {pct(r.gold_node_exists)} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--figure", type=Path, default=Path("project/docs/reachability.png"))
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--only", nargs="*", help="instance ids to analyse (default: all)")
    args = parser.parse_args()

    dataset = Dataset(args.data_dir)
    by_commit: dict[str, list[dict]] = defaultdict(list)
    for task in dataset.tasks(args.only):
        by_commit[graph_key(task["repo"], task["base_commit"])].append(task)
    gathered: dict[str, list] = {"precision": [], "reach": [], "walker": []}
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(analyse_commit, dataset, tasks, args.results / "graphs", args.results / "groundtruth"): key
            for key, tasks in sorted(by_commit.items())
        }
        for done, future in enumerate(as_completed(futures), 1):
            try:
                for name, rows in future.result().items():
                    gathered[name] += rows
                print(f"[{done}/{len(futures)}] {futures[future]}", flush=True)
            except Exception:
                print(f"[{done}/{len(futures)}] {futures[future]} FAILED\n{traceback.format_exc()}", flush=True)

    out = args.results / "audit"
    frames = {name: pd.DataFrame(rows).sort_values(["instance_id"], kind="stable") for name, rows in gathered.items()}
    summaries = {
        "precision": summarise_precision(frames["precision"]),
        "reach": summarise_reach(frames["reach"]),
        "walker": summarise_walker(frames["walker"]),
    }
    for name, frame in frames.items():
        frame.to_csv(out / f"extra_{name}_by_task.csv", index=False)
        summaries[name].to_csv(out / f"extra_{name}.csv", index=False)
    reach_figure(summaries["reach"], args.figure)
    text = report(summaries["precision"], summaries["reach"], summaries["walker"], frames["walker"].instance_id.nunique())
    (out / "EXTRA.md").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
