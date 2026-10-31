"""Paired comparison of localization-agent runs on the tasks every run finished."""

from __future__ import annotations

import argparse
import math
import json
from pathlib import Path

import numpy as np

from callproof import patch as patch_tools
from callproof.dataset import Dataset
from callproof.localize_eval import load_gold, read_episodes, score
from callproof.paths import resolve_data


def surfaced(episode: dict, gold: list[dict]) -> bool:
    """True when a tool result named an edited function (dotted suffix match on Class.method)."""
    wanted = {g["qualname"] for g in gold if g["kind"] != "added" and g["qualname"] != patch_tools.MODULE}
    texts = [m.get("content") or "" for m in episode.get("messages", []) if m.get("role") == "tool"]
    return any(f".{name}\"" in text or f".{name} [" in text or f".{name}," in text for name in wanted for text in texts)


def read_gold_file(episode: dict, gold: list[dict]) -> bool:
    """True when the agent opened a file that holds an edited function."""
    wanted = {g["path"] for g in gold if g["kind"] != "added" and g["qualname"] != patch_tools.MODULE}
    opened = set()
    for call in episode.get("tool_calls", []):
        if call.get("name") == "read_file":
            try:
                opened.add(json.loads(call.get("arguments") or "{}").get("filepath"))
            except json.JSONDecodeError:
                pass
    return bool(wanted & opened)


def graph_calls(episode: dict) -> tuple[int, int]:
    calls = episode.get("tool_calls", [])
    return sum(c.get("name") != "read_file" for c in calls), len(calls)


def graph_misses(episode: dict) -> tuple[int, int]:
    """Graph tool results that came back empty or as NodeNotFound, and all graph tool results."""
    misses = total = 0
    for message in episode.get("messages", []):
        if message.get("role") != "tool":
            continue
        try:
            result = json.loads(message.get("content") or "")
        except json.JSONDecodeError:
            continue
        if not isinstance(result, dict) or not ({"neighbors", "outgoing", "nodes"} & result.keys() or result.get("error_type") == "NodeNotFound"):
            continue
        total += 1
        misses += result.get("error_type") == "NodeNotFound" or not (result.get("count") or result.get("node_count"))
    return misses, total


def sign_test(wins: int, losses: int) -> float:
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--data-dir", type=Path)
    args = parser.parse_args()
    dataset = Dataset(resolve_data(args.data_dir), None)
    tasks = {t["instance_id"]: t for t in dataset.tasks(None)}
    runs = {run.name: {e["instance_id"]: e for e in read_episodes(run)} for run in args.runs}
    shared = sorted(set.intersection(*(set(r) for r in runs.values())))
    rows: dict[str, dict[str, list[bool]]] = {}
    for name, episodes in runs.items():
        metrics: dict[str, list[bool]] = {"func@1": [], "func@5": [], "file@5": [], "surfaced": [], "read_gold_file": [], "empty": []}
        graph_total = all_total = miss_total = result_total = 0
        for instance in shared:
            gold, _ = load_gold(tasks[instance], dataset)
            if not any(g["kind"] != "added" and g["qualname"] != patch_tools.MODULE for g in gold):
                continue
            episode = episodes[instance]
            hits = score(episode["predictions"], gold)
            metrics["func@1"].append(hits["function"][1])
            metrics["func@5"].append(hits["function"][5])
            metrics["file@5"].append(hits.get("file", {}).get(5, False))
            metrics["surfaced"].append(surfaced(episode, gold))
            metrics["read_gold_file"].append(read_gold_file(episode, gold))
            metrics["empty"].append(not episode["predictions"])
            graph, total = graph_calls(episode)
            graph_total, all_total = graph_total + graph, all_total + total
            misses, results = graph_misses(episode)
            miss_total, result_total = miss_total + misses, result_total + results
        rows[name] = metrics
        print(f"{name}: {all_total} tool calls, {graph_total} graph calls ({graph_total / max(all_total, 1):.1%}), "
              f"{miss_total}/{result_total} graph results empty or not found")
    n = len(next(iter(rows.values()))["func@1"])
    print(f"{len(shared)} shared tasks, {n} with a modified gold function")
    for name, metrics in rows.items():
        print(name, {k: round(float(np.mean(v)), 3) for k, v in metrics.items()})
    names = list(rows)
    for other in names[1:]:
        for metric in ("func@1", "func@5", "file@5", "surfaced", "read_gold_file"):
            a, b = np.array(rows[names[0]][metric]), np.array(rows[other][metric])
            wins, losses = int((b & ~a).sum()), int((a & ~b).sum())
            print(f"{other} vs {names[0]} {metric}: +{wins} / -{losses}, sign test p={sign_test(wins, losses):.3f}")
    print(json.dumps({name: {k: float(np.mean(v)) for k, v in m.items()} for name, m in rows.items()}))


if __name__ == "__main__":
    main()
