"""Build and evaluate CallproofLP, the call-edge link-prediction benchmark.

    linkpred.py build   # results/groundtruth + provided graphs -> results/linkpred/{edges,static_calls}.jsonl
    linkpred.py eval    # baselines on the test splits -> results/linkpred/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from callproof import linkpred  # noqa: E402
from callproof.dataset import Dataset  # noqa: E402
from callproof.graphio import load_graph  # noqa: E402
from callproof.naming import RepoIndex  # noqa: E402
from callproof.tools import graph_key  # noqa: E402

ROOT = Path.cwd()
OUT = ROOT / "results" / "linkpred"
GROUNDTRUTH = ROOT / "results" / "groundtruth"
CALLPROOF_GRAPHS = ROOT / "results" / "graphs"


def build_commit(tasks: list[dict]) -> tuple[list[dict], list[tuple[str, str]]]:
    dataset = Dataset()
    task = tasks[0]
    commit = graph_key(task["repo"], task["base_commit"])
    index = RepoIndex.from_checkout(dataset.repo(task).root)
    provided = load_graph(dataset.graph_path(task))
    edges = []
    for item in tasks:
        record = GROUNDTRUTH / f"{item['instance_id']}.json"
        if record.exists():
            edges += json.loads(record.read_text())["dyn_edges"]
    examples = linkpred.build_examples(commit, task["repo"].split("/")[1], edges, index, provided)
    ours = load_graph(CALLPROOF_GRAPHS / f"{commit}.json")
    return examples, [(commit, u, v) for u, v in linkpred.static_call_pairs(ours, index, provided)]


def build(workers: int) -> None:
    by_commit = defaultdict(list)
    for task in Dataset().tasks():
        by_commit[graph_key(task["repo"], task["base_commit"])].append(task)
    OUT.mkdir(parents=True, exist_ok=True)
    examples, static = [], []
    with ProcessPoolExecutor(workers) as pool:
        for rows, pairs in pool.map(build_commit, [by_commit[k] for k in sorted(by_commit)]):
            examples += rows
            static += pairs
    count = linkpred.write_jsonl(OUT / "edges.jsonl", examples)
    linkpred.write_jsonl(OUT / "static_calls.jsonl.gz", ([c, u, v] for c, u, v in static))
    print(f"{count} positives, {len({e['commit'] for e in examples})} commits, "
          f"{len({(e['commit'], e['source']) for e in examples})} sources, {len(static)} static edges")


def load_contexts(examples: list[dict]) -> dict[str, linkpred.Context]:
    dataset = Dataset()
    static = defaultdict(list)
    for commit, u, v in linkpred.read_jsonl(OUT / "static_calls.jsonl.gz"):
        static[commit].append((u, v))
    return {
        commit: linkpred.load_context(
            commit, dataset.data_dir / "graphs" / f"{commit}.json", dataset.data_dir / "embeddings" / f"{commit}.npz", static[commit]
        )
        for commit in sorted({e["commit"] for e in examples})
    }


def fmt(metrics: dict[str, float]) -> str:
    if not metrics:
        return "| n/a | n/a | n/a | n/a | n/a | n/a |"
    keys = ("macro_mrr", "macro_hits@1", "macro_hits@10", "pooled_mrr", "pooled_hits@1", "pooled_hits@10")
    return "| " + " | ".join(f"{metrics[k]:.3f}" for k in keys) + " |"


def evaluate_split(contexts, examples, split_key: str) -> list[str]:
    train = [e for e in examples if e[split_key] == "train"]
    test = [e for e in examples if e[split_key] == "test"]
    x, y = linkpred.training_matrix(contexts, train)
    model = linkpred.LogisticScorer.fit(x, y)
    scorers = dict(linkpred.BASELINES)
    scorers["logistic regression"] = lambda ctx, src: model.score(linkpred.feature_matrix(ctx, src))
    lines = [
        f"train: {len(train)} pairs / {len({e['commit'] for e in train})} commits; "
        f"test: {len(test)} pairs / {len({e['commit'] for e in test})} commits / "
        f"{len({(e['commit'], e['source']) for e in test})} sources; hard (not linked in provided graph): "
        f"{sum(not e['linked'] for e in test)} pairs",
        "",
        "LR weights (standardised): " + ", ".join(f"{n}={w:+.2f}" for n, w in zip(linkpred.FEATURES, model.weights)),
        "",
    ]
    for label, subset in (("all test pairs", test), ("hard subset (not linked in provided graph)", [e for e in test if not e["linked"]])):
        results = linkpred.evaluate(contexts, subset, scorers, known_examples=test)
        lines += [
            f"**{label}**", "",
            "| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |",
            "|---|---|---|---|---|---|---|",
        ]
        lines += [f"| {name} {fmt(metrics)}" for name, metrics in results.items()]
        lines.append("")
    return lines


def evaluate() -> None:
    examples = linkpred.read_jsonl(OUT / "edges.jsonl")
    contexts = load_contexts(examples)
    lines = ["# CallproofLP results", "",
             "Filtered ranking over all nodes of the provided graph of the commit; ties broken in expectation. "
             "Macro = mean over (commit, source) of the mean over its callees; pooled = mean over pairs.", ""]
    lines += ["## Split by repository (train: rich, requests, httpx; test: fastapi)", ""]
    lines += evaluate_split(contexts, examples, "split_repo")
    lines += ["## Random-commit split (30% of commits held out, seed 0)", ""]
    lines += evaluate_split(contexts, examples, "split_random")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build").add_argument("--workers", type=int, default=2)
    sub.add_parser("eval")
    args = parser.parse_args()
    build(args.workers) if args.command == "build" else evaluate()


if __name__ == "__main__":
    main()
