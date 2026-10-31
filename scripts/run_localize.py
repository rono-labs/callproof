"""Run the tool-using localization agent over the competition tasks with one or more llama-server instances."""

from __future__ import annotations

import argparse
import json
import os
import queue
import threading
import traceback
from contextlib import ExitStack
from pathlib import Path

from callproof.agent import AgentConfig, LlamaServer, LocalizationAgent
from callproof.dataset import DEFAULT_DATA_DIR, Dataset
from callproof.paths import resolve_cache
from callproof.retrieve import RepoIndex, issue_query
from callproof.tools import TOOL_SETS, ToolBox

MODEL_FILES = {"e4b": "gemma-4-E4B_q4_0-it.gguf", "e2b": "gemma-4-E2B_q4_0-it.gguf"}
MODELS_DIR = Path(os.environ.get("CALLPROOF_MODELS") or resolve_cache() / "models").expanduser()
LLAMA_SERVER = os.environ.get("LLAMA_SERVER", "llama-server")
MODELS = {name: MODELS_DIR / file for name, file in MODEL_FILES.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--model", choices=MODELS, default="e4b")
    parser.add_argument("--tools", choices=TOOL_SETS, default="both")
    parser.add_argument("--graph", default="provided", help="'provided' or a directory of <repo>_<commit>.json graphs")
    parser.add_argument("--budget", type=int, default=15)
    parser.add_argument("--anchors", type=int, default=0, help="prepend the top-N BM25 function ids from the provided graph")
    parser.add_argument("--typed-neighbors", action="store_true", help="list outgoing and incoming neighbors separately with edge types")
    parser.add_argument("--servers", type=int, default=1)
    parser.add_argument("--threads-per-server", type=int, default=8)
    parser.add_argument("--port", type=int, default=18100)
    parser.add_argument("--context-size", type=int, default=24576)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--thinking", action="store_true")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tool-chars", type=int, default=3000)
    parser.add_argument("--ids", nargs="*", help="restrict to these instance ids")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("results/loc"))
    return parser.parse_args()


def finished_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with open(path, encoding="utf-8") as handle:
        return {json.loads(line)["instance_id"] for line in handle if line.strip()}


def anchors(task: dict, dataset: Dataset, repo, count: int) -> list[str]:
    """Top BM25 function ids over the provided graph, shared by every arm so runs differ only in the graph."""
    index = RepoIndex(Dataset(dataset.data_dir, None).code_graph(task).graph, repo)
    return index.top_locations(index.bm25_scores(issue_query(task)), count, exclude_tests=True)


def main() -> None:
    args = parse_args()
    graph_dir = None if args.graph == "provided" else Path(args.graph)
    dataset = Dataset(args.data_dir, graph_dir)
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    episodes_path, errors_path = run_dir / "episodes.jsonl", run_dir / "errors.jsonl"
    config = AgentConfig(
        tool_set=args.tools,
        budget=args.budget,
        temperature=args.temperature,
        seed=args.seed,
        thinking=args.thinking,
        context_size=args.context_size,
    )
    (run_dir / "config.json").write_text(json.dumps({**vars(args), "out_dir": str(args.out_dir), "data_dir": str(args.data_dir)}, indent=2, default=str))

    done = finished_ids(episodes_path)
    pending: queue.Queue[dict] = queue.Queue()
    selected = [t for t in dataset.tasks(args.ids) if t["instance_id"] not in done]
    for task in selected[: args.limit]:
        pending.put(task)
    print(f"{pending.qsize()} tasks to run ({len(done)} already finished)", flush=True)
    write_lock = threading.Lock()

    def append(path: Path, record: dict) -> None:
        with write_lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def work(url: str) -> None:
        agent = LocalizationAgent(url, config)
        while True:
            try:
                task = pending.get_nowait()
            except queue.Empty:
                return
            try:
                repo = dataset.repo(task)
                graph = dataset.code_graph(task, args.max_tool_chars) if args.tools != "search" else None
                if graph is not None:
                    graph.typed = args.typed_neighbors
                if args.anchors:
                    task = {**task, "anchors": anchors(task, dataset, repo, args.anchors)}
                episode = agent.run(task, ToolBox(repo, graph, args.tools), repo)
                append(episodes_path, {"instance_id": task["instance_id"], "graph": args.graph, "tools": args.tools, **episode})
                print(f"{task['instance_id']}: {episode['wall_seconds']}s {len(episode['tool_calls'])} calls {episode['predictions']}", flush=True)
            except Exception:
                append(errors_path, {"instance_id": task["instance_id"], "traceback": traceback.format_exc()})
                print(f"{task['instance_id']}: ERROR", flush=True)

    with ExitStack() as stack:
        servers = [
            stack.enter_context(
                LlamaServer(Path(LLAMA_SERVER), MODELS[args.model], args.port + i, args.threads_per_server, args.context_size, Path(f"logs/llama_{args.run_name}_{i}.log"))
            )
            for i in range(args.servers)
        ]
        for server in servers:
            server.wait_ready()
        threads = [threading.Thread(target=work, args=(server.url,)) for server in servers]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()


if __name__ == "__main__":
    main()
