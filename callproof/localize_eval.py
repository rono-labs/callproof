"""Score localization runs: Acc@1 / Acc@5 at function and file level against gold edited functions."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from collections import Counter
from pathlib import Path

from callproof import patch as patch_tools
from callproof.dataset import DEFAULT_DATA_DIR, Dataset

GROUNDTRUTH_DIR = Path("results/groundtruth")
KS = (1, 5)


def _split_location(location: str) -> tuple[str, str]:
    path, _, qualname = location.partition("::")
    return path, qualname.split(".<locals>")[0]


def _read_head(base_files: dict[str, str], diff: str) -> dict[str, str]:
    """Apply `diff` to the touched Python files (outside any repository) and return the patched texts."""
    with tempfile.TemporaryDirectory() as scratch:
        for path, text in base_files.items():
            target = Path(scratch) / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text, encoding="utf-8")
        subprocess.run(["git", "apply", "--include=*.py", "-"], input=diff.encode(), cwd=scratch, capture_output=True, check=False)
        return {
            edit.path: (Path(scratch) / edit.path).read_text(encoding="utf-8", errors="replace")
            for edit in patch_tools.parse_diff(diff)
            if edit.path.endswith(".py") and (Path(scratch) / edit.path).is_file()
        }


def derive_gold(task: dict, dataset: Dataset) -> list[dict]:
    """Gold functions from the task patch: removed lines in the base file, added lines in the patched file."""
    root = dataset.repo(task).root
    edits = [e for e in patch_tools.parse_diff(task["patch"]) if e.path.endswith(".py")]
    base = {
        e.old_path: (root / e.old_path).read_text(encoding="utf-8", errors="replace")
        for e in edits
        if e.old_path and (root / e.old_path).is_file()
    }
    head = _read_head(base, task["patch"])
    found, _ = patch_tools.extract_gold_functions(task["patch"], base.get, head.get)
    return [{"path": g.path, "qualname": g.collapsed_qualname, "kind": g.kind, "node_type": g.node_type} for g in found]


def load_gold(task: dict, dataset: Dataset, groundtruth_dir: Path | None = GROUNDTRUTH_DIR) -> tuple[list[dict], str]:
    path = groundtruth_dir / f"{task['instance_id']}.json" if groundtruth_dir else None
    if path and path.exists():
        gold = json.loads(path.read_text(encoding="utf-8"))["gold_functions"]
        return [{**g, "qualname": g["qualname"].split(".<locals>")[0]} for g in gold], "groundtruth"
    return derive_gold(task, dataset), "patch"


def score(predictions: list[str], gold: list[dict]) -> dict[str, dict[int, bool]]:
    """Hit@k flags. Function gold excludes added and module-level edits; file gold excludes files created by the patch."""
    gold_functions = {(g["path"], g["qualname"]) for g in gold if g["kind"] != "added" and g["qualname"] != patch_tools.MODULE}
    gold_files = {g["path"] for g in gold if g["kind"] != "added"}
    ranked = [_split_location(p) for p in predictions]
    ranked_files = list(dict.fromkeys(path for path, _ in ranked))
    result: dict[str, dict[int, bool]] = {}
    if gold_functions:
        result["function"] = {k: any(loc in gold_functions for loc in ranked[:k]) for k in KS}
    if gold_files:
        result["file"] = {k: any(path in gold_files for path in ranked_files[:k]) for k in KS}
    return result


def read_episodes(run_dir: Path) -> list[dict]:
    with open(run_dir / "episodes.jsonl", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate(run_dir: Path, dataset: Dataset, groundtruth_dir: Path | None = GROUNDTRUTH_DIR) -> dict:
    tasks = {t["instance_id"]: t for t in dataset.tasks()}
    hits: dict[str, list[dict[int, bool]]] = {"function": [], "file": []}
    sources: Counter[str] = Counter()
    episodes = [e for e in read_episodes(run_dir) if e["instance_id"] in tasks]
    for episode in episodes:
        gold, source = load_gold(tasks[episode["instance_id"]], dataset, groundtruth_dir)
        sources[source] += 1
        for level, flags in score(episode["predictions"], gold).items():
            hits[level].append(flags)
    metrics: dict = {"run": run_dir.name, "episodes": len(episodes), "gold_source": dict(sources)}
    for level, rows in hits.items():
        metrics[level] = {"n": len(rows), **{f"acc@{k}": round(sum(r[k] for r in rows) / len(rows), 4) if rows else None for k in KS}}
    calls = [c for e in episodes for c in e.get("tool_calls", [])]
    if calls:
        metrics["tool_calls"] = {
            "total": len(calls),
            "valid_rate": round(sum(c["valid"] for c in calls) / len(calls), 4),
            "ok_rate": round(sum(c["ok"] for c in calls) / len(calls), 4),
        }
        metrics["mean_wall_seconds"] = round(sum(e["wall_seconds"] for e in episodes) / len(episodes), 1)
        metrics["answered"] = round(sum(bool(e["predictions"]) for e in episodes) / len(episodes), 4)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories containing episodes.jsonl")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--groundtruth-dir", type=Path, default=GROUNDTRUTH_DIR)
    parser.add_argument("--patch-gold", action="store_true", help="ignore groundtruth files and derive gold from the patch")
    args = parser.parse_args()
    dataset = Dataset(args.data_dir)
    for run_dir in args.runs:
        metrics = evaluate(run_dir, dataset, None if args.patch_gold else args.groundtruth_dir)
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(json.dumps(metrics))


if __name__ == "__main__":
    main()
