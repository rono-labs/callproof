"""Non-LLM localization baselines over the code graph: BM25, optional dense reranking, personalized PageRank."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import networkx as nx
import numpy as np

from callproof.dataset import DEFAULT_DATA_DIR, Dataset
from callproof.tools import Repo

_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")
_STOPWORDS = frozenset("the a an of to in is it and or for on with this that be as at by from not if are was".split())
DENSE_MODEL = "google/embeddinggemma-300m"


def tokenize(text: str) -> list[str]:
    """Lowercased identifier-aware tokens: `getRequestHandler` and `get_request_handler` both give get/request/handler."""
    return [t for w in _WORD.findall(text) if len(t := w.lower()) > 1 and t not in _STOPWORDS]


class BM25:
    def __init__(self, documents: list[list[str]], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        self.lengths = np.array([len(d) for d in documents], dtype=np.float32)
        self.mean_length = float(self.lengths.mean()) if len(documents) else 1.0
        postings: dict[str, list[tuple[int, int]]] = {}
        for index, document in enumerate(documents):
            for term, count in Counter(document).items():
                postings.setdefault(term, []).append((index, count))
        self.postings = {t: (np.array([i for i, _ in p]), np.array([c for _, c in p], dtype=np.float32)) for t, p in postings.items()}
        n = len(documents)
        self.idf = {t: float(np.log1p((n - len(p[0]) + 0.5) / (len(p[0]) + 0.5))) for t, p in self.postings.items()}

    def scores(self, query: list[str]) -> np.ndarray:
        result = np.zeros(len(self.lengths), dtype=np.float32)
        norm = self.k1 * (1 - self.b + self.b * self.lengths / self.mean_length)
        for term, weight in Counter(query).items():
            if term in self.postings:
                docs, tf = self.postings[term]
                result[docs] += self.idf[term] * (1 + np.log(weight)) * tf * (self.k1 + 1) / (tf + norm[docs])
        return result


def is_test_path(path: str) -> bool:
    parts = path.split("/")
    return any(p in ("tests", "test", "docs_src") for p in parts[:-1]) or parts[-1].startswith("test_") or parts[-1] == "conftest.py"


class RepoIndex:
    """Function nodes of one task's graph with BM25 over name + source text."""

    def __init__(self, graph: nx.MultiDiGraph, repo: Repo, edge_types: set[str] | None = None):
        self.graph, self.repo, self.edge_types = graph, repo, edge_types
        self.nodes: list[str] = []
        self.texts: list[str] = []
        for node, attrs in graph.nodes(data=True):
            text = attrs.get("text") or ""
            if not text.lstrip().startswith("class "):
                self.nodes.append(str(node))
                self.texts.append(text)
        self.bm25 = BM25([tokenize(f"{n.replace('.', ' ')} {n.rsplit('.', 1)[-1]} {t}") for n, t in zip(self.nodes, self.texts)])

    def bm25_scores(self, query: str) -> np.ndarray:
        return self.bm25.scores(tokenize(query))

    def ppr_scores(self, seed_scores: np.ndarray, seeds: int = 5, damping: float = 0.5, iterations: int = 40) -> np.ndarray:
        """Personalized PageRank on the undirected graph restarting at the top-`seeds` BM25 nodes."""
        position = {n: i for i, n in enumerate(map(str, self.graph.nodes()))}
        edges = np.array([(position[str(u)], position[str(v)]) for u, v, kind in self.graph.edges(data="type") if self.edge_types is None or kind in self.edge_types], dtype=np.int64).reshape(-1, 2)
        src = np.concatenate([edges[:, 0], edges[:, 1]])
        dst = np.concatenate([edges[:, 1], edges[:, 0]])
        degree = np.bincount(src, minlength=len(position)).astype(np.float64)
        restart = np.zeros(len(position))
        top = np.argsort(-seed_scores)[:seeds]
        for i in top:
            restart[position[self.nodes[i]]] = max(float(seed_scores[i]), 1e-9)
        restart /= restart.sum()
        rank = restart.copy()
        for _ in range(iterations):
            share = np.where(degree > 0, rank / np.maximum(degree, 1), 0.0)
            spread = np.bincount(dst, weights=share[src], minlength=len(position))
            dangling = rank[degree == 0].sum()
            rank = (1 - damping) * restart + damping * (spread + dangling * restart)
        return np.array([rank[position[n]] for n in self.nodes])

    def top_locations(self, scores: np.ndarray, limit: int = 10, exclude_tests: bool = False) -> list[str]:
        found: list[str] = []
        for i in np.argsort(-scores, kind="stable"):
            if scores[i] <= 0 or len(found) >= limit:
                break
            located = self.repo.locate(self.nodes[i])
            if located is None or (exclude_tests and is_test_path(located[0])):
                continue
            location = f"{located[0]}::{located[1]}"
            if location not in found:
                found.append(location)
        return found

    def rerank_dense(self, query: str, scores: np.ndarray, model, candidates: int = 100) -> np.ndarray:
        top = np.argsort(-scores)[:candidates]
        vectors = model.encode_document([self.texts[i][:1500] for i in top], normalize_embeddings=True)
        similarity = vectors @ model.encode_query(query, normalize_embeddings=True)
        dense = np.zeros_like(scores)
        dense[top] = similarity - similarity.min() + 1e-6
        return dense


def issue_query(task: dict, max_chars: int = 6000) -> str:
    return f"{task['problem_statement']}\n{task.get('hints_text') or ''}"[:max_chars]


def load_dense_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(DENSE_MODEL, device="cpu")


def run(method: str, dataset: Dataset, exclude_tests: bool, ids: list[str] | None = None, edge_types: set[str] | None = None) -> list[dict]:
    model = load_dense_model() if method == "dense" else None
    records = []
    for task in dataset.tasks(ids):
        repo = dataset.repo(task)
        index = RepoIndex(dataset.code_graph(task).graph, repo, edge_types)
        query = issue_query(task)
        scores = index.bm25_scores(query)
        if method == "ppr":
            scores = index.ppr_scores(scores)
        elif method == "dense":
            scores = index.rerank_dense(query, scores, model)
        records.append({"instance_id": task["instance_id"], "method": method, "predictions": index.top_locations(scores, 10, exclude_tests)})
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("bm25", "ppr", "dense"), required=True)
    parser.add_argument("--graph", default="provided", help="'provided' or a directory of <repo>_<commit>.json graphs")
    parser.add_argument("--exclude-tests", action="store_true", help="drop test and docs_src files from candidates")
    parser.add_argument("--edge-types", nargs="*", help="restrict PageRank to these edge types")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--ids", nargs="*")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("results/loc"))
    args = parser.parse_args()
    dataset = Dataset(args.data_dir, None if args.graph == "provided" else Path(args.graph))
    records = run(args.method, dataset, args.exclude_tests, args.ids, set(args.edge_types) if args.edge_types else None)
    run_dir = args.out_dir / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "episodes.jsonl", "w", encoding="utf-8") as handle:
        handle.writelines(json.dumps({**r, "graph": args.graph}) + "\n" for r in records)
    print(f"wrote {len(records)} rankings to {run_dir / 'episodes.jsonl'}")


if __name__ == "__main__":
    main()
