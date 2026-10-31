"""Build callproof graphs for every task snapshot and compare them with the provided graphs.

Writes ``results/graphs/<repo>_<commit>.json`` (our graphs), ``results/audit/static_summary.csv``
(one row per graph plus an aggregate row), ``results/audit/groundtruth_summary.csv`` (one row per
task with traced ground truth), ``results/audit/edge_sample.jsonl`` (random provided edges with their
source-text verdicts, for hand inspection) and ``results/audit/STATIC.md``.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from callproof import audit  # noqa: E402
from callproof.build import CALLS, GraphBuilder  # noqa: E402
from callproof.dataset import DEFAULT_DATA_DIR, Dataset  # noqa: E402
from callproof.graphio import load_graph, save_graph  # noqa: E402
from callproof.naming import RepoIndex, parse_checkout  # noqa: E402
from callproof.tools import graph_key  # noqa: E402

SAMPLE_PER_GRAPH = 2
SAMPLE_SIZE = 30


def embedding_coverage(npz: Path, nodes: set[str]) -> int:
    with np.load(npz) as data:
        return len(nodes & set(data.files))


def embedding_spread(npz: Path, size: int = 300, seed: int = 0) -> float:
    """Median cosine similarity between random pairs of provided node embeddings."""
    with np.load(npz) as data:
        names = sorted(data.files)
        chosen = np.random.default_rng(seed).choice(len(names), min(size, len(names)), replace=False)
        vectors = np.stack([data[names[i]] for i in chosen]).astype(np.float64)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = vectors @ vectors.T
    return float(np.median(similarities[np.triu_indices(len(vectors), 1)]))


def audit_commit(dataset: Dataset, tasks: list[dict], graphs_out: Path, groundtruth_dir: Path, seed: int) -> dict:
    task = tasks[0]
    key = graph_key(task["repo"], task["base_commit"])
    root = dataset.repo(task).root
    started = time.perf_counter()
    modules = parse_checkout(root)
    ours = GraphBuilder(modules, key).build()
    build_seconds = time.perf_counter() - started
    save_graph(ours, graphs_out / f"{key}.json")

    index = RepoIndex(modules)
    provided = load_graph(dataset.graph_path(task))
    views = {"prov": audit.GraphView.harness(provided, index), "ours": audit.GraphView.callproof(ours, index)}
    npz = dataset.data_dir / "embeddings" / f"{key}.npz"

    row: dict = {
        "graph": key,
        "repo": task["repo"].split("/")[1],
        "build_seconds": round(build_seconds, 2),
        "embedding_median_pair_cosine": round(embedding_spread(npz), 4),
    }
    sample = []
    for name, view in views.items():
        checker = audit.SourceChecker(view, index)
        stats = audit.structure_stats(view, index, harness=name == "prov")
        semantics, call_pairs = audit.pair_semantics(view, checker)
        if name == "ours":
            call_pairs = {frozenset((u, v)) for u, v, d in ours.edges(data=True) if d["type"] == CALLS and u != v}
        stats.update(semantics)
        stats.update(audit.isolation_stats(view, call_pairs))
        directions, records = audit.direction_stats(view, checker, None if name == "prov" else [CALLS])
        stats.update({f"dir_{k}": v for k, v in directions.items()})
        stats["embedded_nodes"] = embedding_coverage(npz, set(view.graph))
        row.update({f"{name}_{k}": v for k, v in stats.items()})
        if name == "prov":
            rng = random.Random(f"{seed}:{key}")
            sample = [{"graph": key, **r} for r in rng.sample(records, min(SAMPLE_PER_GRAPH, len(records)))]
            for record in sample:
                for end in ("source", "target"):
                    defs = view.definitions.get(record[end], [])
                    record[f"{end}_defs"] = [f"{d.path}::{d.qualname} ({d.kind})" for d in defs]

    ancestors = audit.class_ancestors(views["ours"])
    task_rows = []
    for task in tasks:
        path = groundtruth_dir / f"{task['instance_id']}.json"
        if not path.exists():
            continue
        record = json.loads(path.read_text())
        task_row = {"instance_id": task["instance_id"], "graph": key, "status": record.get("status", "")}
        task_row.update({f"prov_{k}": v for k, v in audit.groundtruth_stats(record, views["prov"], index, ancestors).items()})
        task_row.update({f"ours_{k}": v for k, v in audit.groundtruth_stats(record, views["ours"], index, ancestors).items()})
        calls_only = audit.groundtruth_stats(record, views["ours"], index, ancestors, types=[CALLS])
        task_row.update({f"ourscalls_{k}": v for k, v in calls_only.items() if k.startswith(("dyn_", "reach_"))})
        task_rows.append(task_row)
    return {"row": row, "tasks": task_rows, "sample": sample}


def write_csv(path: Path, rows: list[dict], total_label: str) -> dict:
    columns = list(dict.fromkeys(k for row in rows for k in row))
    numeric = [c for c in columns if all(isinstance(r.get(c, 0), (int, float)) for r in rows)]
    total = {c: sum(r.get(c, 0) for r in rows) for c in numeric}
    total[columns[0]] = total_label
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, restval=0)
        writer.writeheader()
        writer.writerows(rows)
        writer.writerow(total)
    return total


def ratio(total: dict, numerator: str, denominator: str) -> str:
    den = total.get(denominator, 0)
    return f"{100 * total.get(numerator, 0) / den:.1f}% ({total.get(numerator, 0)}/{den})" if den else "n/a"


def static_table(total: dict) -> list[str]:
    def both(fmt):
        return f"| {fmt('prov')} | {fmt('ours')} |"

    lines = ["| metric | provided | callproof |", "|---|---|---|"]
    rows = [
        ("nodes", lambda p: f"{total[f'{p}_nodes']}"),
        ("edges", lambda p: f"{total[f'{p}_edges']}"),
        ("definitions with a node", lambda p: ratio(total, f"{p}_defs_with_node", f"{p}_defs")),
        ("async definitions with a node", lambda p: ratio(total, f"{p}_async_defs_with_node", f"{p}_async_defs")),
        ("modules with a node", lambda p: ratio(total, f"{p}_modules_with_node", f"{p}_modules")),
        ("definitions sharing an id", lambda p: ratio(total, f"{p}_defs_sharing_id", f"{p}_defs")),
        ("definitions whose id is not module + dotted qualname", lambda p: ratio(total, f"{p}_defs_id_not_dotted_qualname", f"{p}_defs")),
        ("top-level-class methods whose id lost the class", lambda p: ratio(total, f"{p}_methods_lost_class", f"{p}_defs_method")),
        ("nodes with >50 neighbours (truncated by get_code_neighbors)", lambda p: ratio(total, f"{p}_nodes_over_50_neighbours", f"{p}_nodes")),
        ("connected pairs also linked in reverse", lambda p: ratio(total, f"{p}_pairs_with_reverse", f"{p}_distinct_pairs")),
        ("callable nodes in no call relation", lambda p: ratio(total, f"{p}_callable_nodes_without_calls", f"{p}_callable_nodes")),
        ("... library code only", lambda p: ratio(total, f"{p}_lib_callable_nodes_without_calls", f"{p}_lib_callable_nodes")),
        ("checked edges whose source supports their direction", lambda p: ratio(total, f"{p}_dir_forward_supported", f"{p}_dir_edges_checked")),
        ("checked edges supported only in reverse", lambda p: ratio(total, f"{p}_dir_only_reverse_supported", f"{p}_dir_edges_checked")),
        ("checked edges unsupported by source text", lambda p: ratio(total, f"{p}_dir_unsupported", f"{p}_dir_edges_checked")),
        ("nodes with an embedding", lambda p: ratio(total, f"{p}_embedded_nodes", f"{p}_nodes")),
    ]
    for label, fmt in rows:
        lines.append(f"| {label} " + both(fmt))
    return lines


def groundtruth_table(total: dict) -> list[str]:
    lines = ["| metric | provided | callproof (all edges) | callproof (calls only) |", "|---|---|---|---|"]

    def add(label: str, num: str, den: str, calls_variant: bool = False) -> None:
        third = ratio(total, f"ourscalls_{num}", f"ourscalls_{den}") if calls_variant else ""
        lines.append(f"| {label} | {ratio(total, f'prov_{num}', f'prov_{den}')} | {ratio(total, f'ours_{num}', f'ours_{den}')} | {third} |")

    add("gold modified definitions with a node", "gold_modified_covered", "gold_modified_n")
    add("gold added definitions with a node", "gold_added_covered", "gold_added_n")
    add("gold modified module-level edits with a node", "gold_modified_module_covered", "gold_modified_module_n")
    add("executed definitions with a node", "executed_defs_covered", "executed_defs_n")
    add("executed (incl. module code) with a node", "executed_covered", "executed_n")
    add("F2P tests present at base with a node", "f2p_tests_covered", "f2p_tests_n")
    for category in ("all", *audit.EDGE_CATEGORIES):
        add(f"dynamic edges [{category}]: both ends are nodes", f"dyn_{category}_mapped", f"dyn_{category}_n")
        add(f"dynamic edges [{category}]: linked, either direction", f"dyn_{category}_either", f"dyn_{category}_n", True)
        add(f"dynamic edges [{category}]: linked, correct direction", f"dyn_{category}_forward", f"dyn_{category}_n", True)
    add("constructor calls linked to the __init__ or to the class", "dyn_constructor_class_either", "dyn_constructor_n", True)
    add("tasks with an F2P test present at base", "reach_tasks_with_source", "reach_tasks")
    for k in (1, 2, 3):
        add(f"gold modified defs within {k} hops of an F2P test (tasks with one)", f"reach_within_{k}", "reach_targets", True)
    add("tasks with executed test code present at base", "reachexec_tasks_with_source", "reachexec_tasks")
    for k in (1, 2, 3):
        add(f"gold modified defs within {k} hops of executed test code", f"reachexec_within_{k}", "reachexec_targets", True)
    return lines


def sample_table(sample: list[dict], size: int, seed: int) -> list[str]:
    chosen = random.Random(seed).sample(sample, min(size, len(sample)))
    lines = ["| source | target | relation u -> v | relation v -> u |", "|---|---|---|---|"]
    for record in sorted(chosen, key=lambda r: (r["source"], r["target"])):
        lines.append(f"| `{record['source']}` | `{record['target']}` | {record['forward'] or '-'} | {record['backward'] or '-'} |")
    return lines


def write_report(
    path: Path,
    static_total: dict,
    gt_total: dict | None,
    sample: list[dict],
    n_graphs: int,
    n_tasks: int,
    seed: int,
    max_build: float,
) -> None:
    lines = [
        "# Static audit: provided graphs vs callproof graphs",
        "",
        f"{n_graphs} graphs (one per distinct base commit). Percentages are micro-averages over all graphs.",
        f"Mean build time {static_total['build_seconds'] / n_graphs:.1f}s per checkout (max {max_build:.1f}s, under parallel load).",
        f"Provided embeddings are compressed: the median cosine similarity between random node pairs is"
        f" {static_total['embedding_median_pair_cosine'] / n_graphs:.3f} (mean over graphs).",
        "Edges are checked against the source text of their endpoints: an edge u -> v is *supported* when u's",
        "code contains v (class -> method), inherits from, instantiates, calls or references v by name. The check",
        "is name-based, so resolved edges it cannot see (inherited `__init__`, `cls(...)`, `obj(...)` -> `__call__`)",
        "count as unsupported.",
        "",
        *static_table(static_total),
        "",
        "Provided-graph pair semantics (unordered connected pairs, strongest explanation):",
        "",
    ]
    pairs = static_total.get("prov_pairs", 0)
    for label in (*audit.SourceChecker.RELATIONS, "unexplained"):
        lines.append(f"- {label}: {ratio(static_total, f'prov_pairs_{label}', 'prov_pairs')}")
    edge_types = ", ".join(
        f"{k.removeprefix('ours_edges_')}={v}" for k, v in sorted(static_total.items()) if k.startswith("ours_edges_")
    )
    resolutions = ", ".join(
        f"{k.removeprefix('ours_resolution_')}={v}" for k, v in sorted(static_total.items()) if k.startswith("ours_resolution_")
    )
    lines += [
        "",
        f"Total provided pairs: {pairs}. Callproof edge types: {edge_types}. Edge resolutions: {resolutions}.",
        "",
        f"Random provided edges for hand inspection ({SAMPLE_SIZE} of {len(sample)} sampled):",
        "",
        *sample_table(sample, SAMPLE_SIZE, seed),
    ]
    if gt_total is not None:
        lines += [
            "",
            f"## Agreement with traced executions ({n_tasks} tasks with ground truth)",
            "",
            "Dynamic edges are compared at the tracer's granularity (nested functions folded into their",
            "enclosing named definition). Categories overlap.",
            "",
            *groundtruth_table(gt_total),
        ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--jobs", type=int, default=16)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--groundtruth", type=Path, default=Path("results/groundtruth"))
    parser.add_argument("--only", nargs="*", help="instance ids to audit (default: all)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR, help="competition data")
    args = parser.parse_args()

    dataset = Dataset(args.data_dir)
    by_commit: dict[str, list[dict]] = defaultdict(list)
    for task in dataset.tasks(args.only):
        by_commit[graph_key(task["repo"], task["base_commit"])].append(task)
    graphs_out, audit_out = args.results / "graphs", args.results / "audit"
    graphs_out.mkdir(parents=True, exist_ok=True)
    audit_out.mkdir(parents=True, exist_ok=True)

    rows, task_rows, sample = [], [], []
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        futures = {
            pool.submit(audit_commit, dataset, tasks, graphs_out, args.groundtruth, args.seed): key
            for key, tasks in sorted(by_commit.items())
        }
        for done, future in enumerate(as_completed(futures), 1):
            key = futures[future]
            try:
                result = future.result()
            except Exception:
                print(f"[{done}/{len(futures)}] {key} FAILED\n{traceback.format_exc()}", flush=True)
                continue
            rows.append(result["row"])
            task_rows.extend(result["tasks"])
            sample.extend(result["sample"])
            print(f"[{done}/{len(futures)}] {key} built in {result['row']['build_seconds']}s", flush=True)

    rows.sort(key=lambda r: r["graph"])
    task_rows.sort(key=lambda r: r["instance_id"])
    static_total = write_csv(audit_out / "static_summary.csv", rows, "ALL")
    gt_total = write_csv(audit_out / "groundtruth_summary.csv", task_rows, "ALL") if task_rows else None
    with open(audit_out / "edge_sample.jsonl", "w", encoding="utf-8") as handle:
        ordered = sorted(sample, key=lambda r: (r["graph"], r["source"], r["target"]))
        handle.writelines(json.dumps(record) + "\n" for record in ordered)
    max_build = max(r["build_seconds"] for r in rows)
    write_report(audit_out / "STATIC.md", static_total, gt_total, sample, len(rows), len(task_rows), args.seed, max_build)
    print(Counter(r["repo"] for r in rows), flush=True)


if __name__ == "__main__":
    main()
