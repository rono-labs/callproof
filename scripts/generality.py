"""Does edge recall generalize beyond the competition's repositories?

Clones six pure-Python repositories at pinned tags, traces their pytest suites with
:mod:`callproof.trace`, builds the callproof graph and a provided-style baseline graph for each
checkout, and measures how many traced repo-to-repo calls each graph links. Writes
``results/generality/GENERALITY.md`` and ``summary.csv``.

Stages (all by default): ``setup`` (clone, venv, install), ``trace``, ``report``.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from callproof import audit  # noqa: E402
from callproof.build import GraphBuilder  # noqa: E402
from callproof.naming import RepoIndex, parse_checkout  # noqa: E402
from provided_style import build_provided_style  # noqa: E402

WORK = Path(".cache/work/general")
RESULTS = Path("results/generality")
LOGS = Path("logs/generality")
MAX_TESTS = 300
TEST_TIMEOUT_SECONDS = 1800
BOOTSTRAPS = 1000
CATEGORIES = ("all", *audit.EDGE_CATEGORIES)
SYSTEMS = ("callproof", "baseline")


@dataclass(frozen=True)
class Repo:
    name: str
    url: str
    tag: str
    test_deps: tuple[str, ...] = ()
    test_args: tuple[str, ...] = ("tests",)


REPOS = (
    Repo("click", "https://github.com/pallets/click", "8.5.0"),
    Repo("attrs", "https://github.com/python-attrs/attrs", "26.1.0", ("hypothesis",)),
    Repo("more-itertools", "https://github.com/more-itertools/more-itertools", "v11.2.1"),
    Repo("packaging", "https://github.com/pypa/packaging", "26.3", ("pretend", "tomli-w", "hypothesis")),
    Repo("itsdangerous", "https://github.com/pallets/itsdangerous", "2.2.0", ("freezegun",)),
    Repo("h11", "https://github.com/python-hyper/h11", "v0.16.0", test_args=("h11/tests",)),
)


def run(command: list[str], log: Path, cwd: Path | None = None, env: dict | None = None, timeout: int | None = None) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    with open(log, "ab") as handle:
        handle.write(f"$ {' '.join(command)}\n".encode())
        handle.flush()
        done = subprocess.run(command, cwd=cwd, env=env, stdout=handle, stderr=subprocess.STDOUT, timeout=timeout)
    return done.returncode


def checkout_dir(repo: Repo) -> Path:
    return WORK / "repos" / repo.name


def venv_dir(repo: Repo) -> Path:
    return WORK / "envs" / repo.name


def setup_repo(repo: Repo) -> dict:
    log = LOGS / f"{repo.name}.setup.log"
    root, env = checkout_dir(repo), venv_dir(repo)
    if not root.exists():
        run(["git", "clone", "--quiet", "--depth", "1", "--branch", repo.tag, repo.url, str(root)], log)
    if not env.exists():
        run(["uv", "venv", "--quiet", "--python", "3.13", str(env)], log)
    python = str(env / "bin" / "python")
    status = run(["uv", "pip", "install", "--quiet", "--python", python, "-e", str(root), "pytest", *repo.test_deps], log)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True).stdout.strip()
    return {"repo": repo.name, "tag": repo.tag, "sha": sha, "install_status": status}


def trace_repo(repo: Repo) -> dict:
    root, env = checkout_dir(repo).resolve(), venv_dir(repo).resolve()
    out = (WORK / "traces").resolve()
    out.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join([str(PROJECT), str(scripts)]),
        "GENERALITY_MAX_TESTS": str(MAX_TESTS),
        "GENERALITY_PER_TEST_OUT": str(out / f"{repo.name}.per_test.json"),
    }
    command = [
        str(env / "bin" / "python"), "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "addopts=",
        "-p", "callproof.trace", "-p", "generality_plugin",
        "--callproof-root", str(root), "--callproof-out", str(out / f"{repo.name}.edges.json"), *repo.test_args,
    ]
    started = time.perf_counter()
    status = run(command, LOGS / f"{repo.name}.trace.log", cwd=root, env=environment, timeout=TEST_TIMEOUT_SECONDS)
    return {"repo": repo.name, "pytest_status": status, "seconds": round(time.perf_counter() - started, 1)}


@dataclass
class RepoResult:
    name: str
    n_tests: int
    build_seconds: dict[str, float]
    graph_sizes: dict[str, tuple[int, int]]
    edges: list[list]
    flags: dict[str, dict[str, np.ndarray]]
    tests: np.ndarray
    categories: dict[str, np.ndarray] = field(default_factory=dict)


def edge_flags(edges: list[list], view: audit.GraphView) -> dict[str, np.ndarray]:
    directed, nodes = view.key_edges(), view.key_nodes()
    mapped, either, forward = [], [], []
    for edge in edges:
        caller, callee = (edge[0], edge[1]), (edge[2], edge[3])
        fwd = (caller, callee) in directed
        mapped.append(caller in nodes and callee in nodes)
        forward.append(fwd)
        either.append(fwd or (callee, caller) in directed)
    return {"mapped": np.array(mapped), "either": np.array(either), "forward": np.array(forward)}


def evaluate_repo(repo: Repo) -> RepoResult:
    root = checkout_dir(repo)
    traces = WORK / "traces"
    raw = json.loads((traces / f"{repo.name}.edges.json").read_text())["edges"]
    per_test = json.loads((traces / f"{repo.name}.per_test.json").read_text())

    started = time.perf_counter()
    modules = parse_checkout(root)
    parse_seconds = time.perf_counter() - started
    started = time.perf_counter()
    ours = GraphBuilder(modules, repo.name).build()
    ours_seconds = parse_seconds + time.perf_counter() - started
    started = time.perf_counter()
    index = RepoIndex(modules)
    baseline = build_provided_style(index, repo.name)
    baseline_seconds = parse_seconds + time.perf_counter() - started

    views = {"callproof": audit.GraphView.callproof(ours, index), "baseline": audit.GraphView.harness(baseline, index)}
    ancestors = audit.class_ancestors(views["callproof"])

    edges = [e for e in raw if (e[0], e[1]) != (e[2], e[3])]
    position = {tuple(e[:4]) + (e[6],): i for i, e in enumerate(edges)}
    tests = np.zeros((len(per_test), len(edges)), dtype=bool)
    for row, keys in enumerate(per_test.values()):
        for key in keys:
            tests[row, position[tuple(key)]] = True
    categories = {name: np.zeros(len(edges), dtype=bool) for name in CATEGORIES}
    for i, edge in enumerate(edges):
        for name in audit.edge_categories(edge, index, ancestors):
            categories[name][i] = True
    categories["all"] = ~categories["via_test"]

    flags = {system: edge_flags(edges, view) for system, view in views.items()}
    check = audit.dynamic_edge_stats(edges, views["callproof"], index, ancestors)
    assert check["all_either"] == flags["callproof"]["either"].sum(), "per-edge flags disagree with audit"
    return RepoResult(
        name=repo.name,
        n_tests=len(per_test),
        build_seconds={"callproof": ours_seconds, "baseline": baseline_seconds},
        graph_sizes={s: (v.graph.number_of_nodes(), v.graph.number_of_edges()) for s, v in views.items()},
        edges=edges,
        flags=flags,
        tests=tests,
        categories=categories,
    )


def counts(result: RepoResult, category: str, system: str, edge_mask: np.ndarray | None = None) -> dict[str, int]:
    select = result.categories[category] if edge_mask is None else result.categories[category] & edge_mask
    return {"n": int(select.sum()), **{k: int((v & select).sum()) for k, v in result.flags[system].items()}}


def bootstrap(results: list[RepoResult], category: str, system: str, metric: str, seed: int = 0) -> tuple[float, float]:
    """95% interval of pooled recall, resampling tests within each repository."""
    rng = np.random.default_rng(seed)
    prepared = [
        (r.tests.astype(np.uint8), r.categories[category], r.flags[system][metric])
        for r in results
        if r.categories[category].any() and len(r.tests)
    ]
    draws = []
    for _ in range(BOOTSTRAPS):
        hit = total = 0
        for matrix, select, flag in prepared:
            chosen = matrix[rng.integers(0, len(matrix), len(matrix))].max(axis=0).astype(bool) & select
            total += chosen.sum()
            hit += (chosen & flag).sum()
        if total:
            draws.append(hit / total)
    low, high = np.percentile(draws, [2.5, 97.5])
    return float(low), float(high)


def percent(hit: int, n: int) -> str:
    return f"{100 * hit / n:.1f}% ({hit}/{n})" if n else "n/a"


def write_reports(results: list[RepoResult], pins: list[dict], traced: dict[str, dict]) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    rows = []
    for result in results:
        for category in CATEGORIES:
            for system in SYSTEMS:
                rows.append({"repo": result.name, "category": category, "system": system, **counts(result, category, system)})
    pooled_rows = []
    for category in CATEGORIES:
        for system in SYSTEMS:
            totals = Counter()
            for result in results:
                totals.update(counts(result, category, system))
            row = {"repo": "pooled", "category": category, "system": system, **totals}
            for metric in ("either", "forward"):
                low, high = bootstrap(results, category, system, metric)
                row[f"{metric}_ci_low"], row[f"{metric}_ci_high"] = round(low, 4), round(high, 4)
            pooled_rows.append(row)
    fields = ["repo", "category", "system", "n", "mapped", "either", "forward", "either_ci_low", "either_ci_high", "forward_ci_low", "forward_ci_high"]
    with open(RESULTS / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, fields, restval="")
        writer.writeheader()
        writer.writerows([*rows, *pooled_rows])

    lines = ["# Generality: edge recall on six repositories outside the competition", "", "## Repositories", "",
             "| repo | tag | commit | tests traced | traced edges (non-test) | callproof graph (nodes/edges) | baseline graph (nodes/edges) | build s (callproof / baseline) | pytest exit |",
             "|---|---|---|---|---|---|---|---|---|"]
    for pin, result in zip(pins, results):
        n = int(result.categories["all"].sum())
        lines.append(
            f"| {result.name} | {pin['tag']} | `{pin['sha'][:12]}` | {result.n_tests} | {n} | "
            f"{result.graph_sizes['callproof'][0]}/{result.graph_sizes['callproof'][1]} | "
            f"{result.graph_sizes['baseline'][0]}/{result.graph_sizes['baseline'][1]} | "
            f"{result.build_seconds['callproof']:.2f} / {result.build_seconds['baseline']:.2f} | {traced[result.name]['pytest_status']} |")
    lines += ["", "## Pooled recall (95% bootstrap CI over tests, resampled within each repository)", "",
              "| call kind | traced edges | baseline: both ends nodes | baseline: linked either way | baseline: correct direction | callproof: both ends nodes | callproof: linked either way | callproof: correct direction |",
              "|---|---|---|---|---|---|---|---|"]
    by_key = {(r["category"], r["system"]): r for r in pooled_rows}
    for category in CATEGORIES:
        b, c = by_key[(category, "baseline")], by_key[(category, "callproof")]
        cells = [percent(s["mapped"], s["n"]) for s in (b,)]
        for s in (b, c):
            if s is c:
                cells.append(percent(s["mapped"], s["n"]))
            for metric in ("either", "forward"):
                cells.append(f"{percent(s[metric], s['n'])} [{100 * s[f'{metric}_ci_low']:.1f}, {100 * s[f'{metric}_ci_high']:.1f}]" if s["n"] else "n/a")
        lines.append(f"| {category} | {b['n']} | " + " | ".join(cells) + " |")
    for result in results:
        lines += ["", f"### {result.name}", "", "| call kind | traced edges | baseline either / forward | callproof either / forward |", "|---|---|---|---|"]
        for category in CATEGORIES:
            b, c = counts(result, category, "baseline"), counts(result, category, "callproof")
            lines.append(f"| {category} | {c['n']} | {percent(b['either'], b['n'])} / {percent(b['forward'], b['n'])} | {percent(c['either'], c['n'])} / {percent(c['forward'], c['n'])} |")
    lines += ["", "## Method notes", "",
              "- Traced edges: `callproof.trace` under `pytest` on Python 3.13 (call phase only), unique caller/callee pairs over all sampled tests, nested defs folded into their enclosing definition. `all` excludes edges with a test-code endpoint (`via_test`, reported as its own row); self-edges are dropped. Categories overlap.",
              f"- Suites larger than {MAX_TESTS} tests are sampled at evenly spaced positions in collection order.",
              "- Metrics are those of `callproof.audit` (both ends are nodes; linked in either direction; linked in the traced direction), computed per edge so tests can be resampled.",
              "- Baseline: a reimplementation (`project/scripts/provided_style.py`) of the provided graphs' documented rules, not their actual builder: harness node ids, undirected `calls` edges for containment, inheritance, instantiation and plain-name calls, no async definitions, no module nodes, no `self.method` or attribute calls. Plain names resolve by same module, then `from` imports, then a unique repository-wide name.",
              "- Bootstrap: 1000 resamples of tests within each repository; each resample's edge set is the union of its tests' edges."]
    (RESULTS / "GENERALITY.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("stages", nargs="*", default=["setup", "trace", "report"], choices=["setup", "trace", "report"])
    parser.add_argument("--only", nargs="+", metavar="REPO", help="restrict setup and trace to these repositories")
    parser.add_argument("--jobs", type=int, default=3, help="repositories processed concurrently (at most 6)")
    args = parser.parse_args()
    jobs = min(args.jobs, 6)
    manifest = WORK / "manifest.json"
    state = json.loads(manifest.read_text()) if manifest.exists() else {"pins": [], "traced": {}}
    WORK.mkdir(parents=True, exist_ok=True)
    chosen = [r for r in REPOS if not args.only or r.name in args.only]
    with ThreadPoolExecutor(jobs) as pool:
        if "setup" in args.stages:
            pins = {p["repo"]: p for p in state["pins"]} | {p["repo"]: p for p in pool.map(setup_repo, chosen)}
            state["pins"] = [pins[r.name] for r in REPOS if r.name in pins]
        if "trace" in args.stages:
            state["traced"] |= {r["repo"]: r for r in pool.map(trace_repo, chosen)}
    manifest.write_text(json.dumps(state, indent=1))
    if "report" in args.stages:
        results = [evaluate_repo(repo) for repo in REPOS]
        write_reports(results, state["pins"], state["traced"])


if __name__ == "__main__":
    main()
