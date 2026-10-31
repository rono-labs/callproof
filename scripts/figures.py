"""Bootstrap confidence intervals over tasks and the recall-by-call-kind figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

KINDS = {
    "all": "All traced calls",
    "self_method": "self/cls method",
    "constructor": "Constructor to __init__",
    "module_function": "Module function",
    "cross_module": "Cross-module",
    "async": "Async",
    "nested": "Nested def",
}
GRAPHS = {"prov": "Provided graph", "ours": "Callproof"}


def pooled_recall(frame: pd.DataFrame, graph: str, kind: str, rng: np.random.Generator, draws: int) -> tuple[float, float, float, int, int]:
    hits = frame[f"{graph}_dyn_{kind}_forward"].to_numpy(float)
    totals = frame[f"{graph}_dyn_{kind}_n"].to_numpy(float)
    keep = totals > 0
    hits, totals = hits[keep], totals[keep]
    point = hits.sum() / totals.sum()
    picks = rng.integers(0, len(totals), size=(draws, len(totals)))
    samples = hits[picks].sum(axis=1) / np.maximum(totals[picks].sum(axis=1), 1)
    low, high = np.percentile(samples, [2.5, 97.5])
    return point, low, high, int(totals.sum()), int(keep.sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summary", type=Path, default=Path("results/audit/groundtruth_summary.csv"))
    parser.add_argument("--table", type=Path, default=Path("results/audit/recall_ci.csv"))
    parser.add_argument("--figure", type=Path, default=Path("project/docs/recall_by_kind.png"))
    parser.add_argument("--draws", type=int, default=10000)
    args = parser.parse_args()

    frame = pd.read_csv(args.summary)
    frame = frame[(frame["instance_id"] != "ALL") & (frame["prov_dyn_all_n"].fillna(0) > 0)]
    rng = np.random.default_rng(0)
    rows = []
    for kind in KINDS:
        for graph in GRAPHS:
            point, low, high, edges, tasks = pooled_recall(frame, graph, kind, rng, args.draws)
            rows.append({"kind": kind, "graph": graph, "recall": point, "ci_low": low, "ci_high": high, "edges": edges, "tasks": tasks})
    table = pd.DataFrame(rows)
    args.table.parent.mkdir(parents=True, exist_ok=True)
    table.round(4).to_csv(args.table, index=False)
    print(table.round(3).to_string(index=False))

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=160)
    positions = np.arange(len(KINDS))
    width = 0.38
    colors = {"prov": "#9aa0a6", "ours": "#1a73e8"}
    for offset, graph in zip((-width / 2, width / 2), GRAPHS):
        part = table[table.graph == graph].set_index("kind").loc[list(KINDS)]
        errors = np.vstack([part.recall - part.ci_low, part.ci_high - part.recall]) * 100
        bars = ax.bar(positions + offset, part.recall * 100, width, yerr=errors, capsize=3, color=colors[graph], label=GRAPHS[graph])
        ax.bar_label(bars, labels=[f"{v:.0f}" for v in part.recall * 100], padding=2, fontsize=8)
    edges = table[table.graph == "ours"].set_index("kind").loc[list(KINDS), "edges"]
    ax.set_xticks(positions, [f"{KINDS[k]}\nn={edges[k]:,}" for k in KINDS], fontsize=8)
    ax.set_ylabel("Traced call edges linked, correct direction (%)")
    ax.set_ylim(0, 110)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    ax.set_title("Which executed calls does the code graph contain? (114 tasks with traced edges, 95% task-bootstrap CI)", fontsize=10)
    fig.tight_layout()
    args.figure.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.figure)


if __name__ == "__main__":
    main()
