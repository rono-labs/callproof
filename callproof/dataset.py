"""Access to the competition data directory: tasks, snapshots, graphs and embeddings."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from callproof.tools import CodeGraph, Repo, ensure_snapshot, graph_key

DEFAULT_DATA_DIR = Path(".cache/compdata/unz/root/.cache/kagglehub/competitions/gemma-4-developer-agent")
DEFAULT_SNAPSHOT_DIR = Path(".cache/work/snap")


@dataclass(frozen=True)
class Dataset:
    data_dir: Path = DEFAULT_DATA_DIR
    graph_dir: Path | None = None
    snapshot_dir: Path = DEFAULT_SNAPSHOT_DIR

    def tasks(self, ids: list[str] | None = None) -> list[dict]:
        with open(self.data_dir / "tasks.jsonl", encoding="utf-8") as handle:
            tasks = [json.loads(line) for line in handle if line.strip()]
        return [t for t in tasks if ids is None or t["instance_id"] in ids]

    def repo(self, task: dict) -> Repo:
        snapshot = self.data_dir / "snapshots" / f"{task['instance_id']}.tgz"
        return Repo(ensure_snapshot(snapshot, self.snapshot_dir / task["instance_id"]))

    def graph_path(self, task: dict) -> Path:
        key = graph_key(task["repo"], task["base_commit"])
        return (self.graph_dir or self.data_dir / "graphs") / f"{key}.json"

    def code_graph(self, task: dict, max_chars: int = 5000) -> CodeGraph:
        key = graph_key(task["repo"], task["base_commit"])
        return CodeGraph(self.graph_path(task), self.data_dir / "embeddings" / f"{key}.npz", key, max_chars)
