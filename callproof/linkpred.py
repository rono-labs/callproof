"""CallproofLP: call-edge link prediction on code graphs.

Each example is a traced caller -> callee pair, mapped to node ids of the provided code graph of one base commit.
The task ranks every node of that graph as the callee of the caller (filtered ranking: the other known callees
of the same caller are removed). Only ids and counts are released, never source text.
"""

from __future__ import annotations

import gzip
import json
import math
import random
import re
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import networkx as nx
import numpy as np

from callproof import audit
from callproof.build import CALLS
from callproof.graphio import load_graph
from callproof.naming import RepoIndex
from callproof.trace import is_test_path

TRAIN_REPOS = ("rich", "requests", "httpx")
TEST_REPO = "fastapi"
FEATURES = ("lexical", "cosine", "same_module", "same_class", "adjacent", "log_degree")
_WORD = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|\d+")


def tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD.findall(text)]


def split_by_repo(repo: str) -> str:
    return "test" if repo == TEST_REPO else "train"


def split_by_commit(commit: str, fraction: float = 0.3, seed: int = 0) -> str:
    return "test" if random.Random(f"{seed}:{commit}").random() < fraction else "train"


def module_of(node_id: str) -> str:
    """Module part of a node id: the dotted prefix before the first capitalised (class) component."""
    parts = node_id.split(".")[:-1]
    for i, part in enumerate(parts):
        if part[:1].isupper():
            return ".".join(parts[:i])
    return ".".join(parts)


def class_prefix(node_id: str) -> str:
    return node_id.rpartition(".")[0]


# Dataset construction.


def edge_kind(callee: str, callee_def_kind: str | None) -> str:
    name = callee.rpartition(".")[2]
    if name in audit.CONSTRUCTOR_NAMES or callee_def_kind == "class":
        return "constructor"
    return "method" if "." in callee else "function"


def build_examples(
    commit: str, repo: str, edges: Iterable[list], index: RepoIndex, provided: nx.MultiDiGraph
) -> list[dict]:
    """Deduplicated non-test traced pairs whose both ends exist in the provided graph."""
    pairs: dict[tuple[str, str], dict] = {}
    for caller_path, caller, callee_path, callee, count, via_test, *_ in edges:
        if via_test or is_test_path(caller_path) or is_test_path(callee_path):
            continue
        source, target = index.harness_id(caller_path, caller), index.harness_id(callee_path, callee)
        if source is None or target is None or source == target or source not in provided or target not in provided:
            continue
        definition = index.lookup(callee_path, callee)
        row = pairs.setdefault(
            (source, target),
            {"commit": commit, "repo": repo, "source": source, "target": target, "count": 0,
             "kind": edge_kind(callee, definition.kind if definition else None)},
        )
        row["count"] += count
    undirected = nx.Graph(provided)
    for (source, target), row in pairs.items():
        row["linked"] = undirected.has_edge(source, target)
        row["split_repo"] = split_by_repo(repo)
        row["split_random"] = split_by_commit(commit)
    return sorted(pairs.values(), key=lambda r: (r["source"], r["target"]))


def static_call_pairs(ours: nx.MultiDiGraph, index: RepoIndex, provided: nx.MultiDiGraph) -> list[tuple[str, str]]:
    """Callproof ``calls`` edges expressed in provided-graph ids."""
    provided_of: dict[str, set[str]] = {}
    for node in ours:
        provided_of[node] = {d.harness_id for d in index.from_node_id(node) if d.harness_id in provided}
    pairs = set()
    for u, v, data in ours.edges(data=True):
        if data.get("type") == CALLS:
            pairs.update((a, b) for a in provided_of[u] for b in provided_of[v] if a != b)
    return sorted(pairs)


def write_jsonl(path: Path, rows: Iterable[dict | list]) -> int:
    opener = gzip.open if path.suffix == ".gz" else open
    count = 0
    with opener(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


# Metrics.


def tied_metrics(score_of_target: float, others: np.ndarray, ks: Sequence[int] = (1, 10)) -> dict[str, float]:
    """Reciprocal rank and hits@k in expectation over a uniformly random order of tied candidates."""
    greater = int((others > score_of_target).sum())
    ties = int((others == score_of_target).sum())
    positions = np.arange(greater + 1, greater + ties + 2)
    out = {"mrr": float(np.mean(1.0 / positions))}
    for k in ks:
        out[f"hits@{k}"] = float(np.mean(positions <= k))
    return out


METRICS = ("mrr", "hits@1", "hits@10")


def aggregate(rows: Sequence[tuple[str, str, dict[str, float]]]) -> dict[str, float]:
    """Pooled and macro (over (commit, source)) means of per-pair metrics."""
    if not rows:
        return {}
    pooled = {m: float(np.mean([r[2][m] for r in rows])) for m in METRICS}
    groups: dict[tuple[str, str], list[dict[str, float]]] = defaultdict(list)
    for commit, source, metrics in rows:
        groups[(commit, source)].append(metrics)
    macro = {m: float(np.mean([np.mean([g[m] for g in group]) for group in groups.values()])) for m in METRICS}
    return {**{f"macro_{m}": v for m, v in macro.items()}, **{f"pooled_{m}": v for m, v in pooled.items()},
            "pairs": len(rows), "sources": len(groups)}


# Per-commit scoring context.


@dataclass
class Context:
    commit: str
    nodes: list[str]
    position: dict[str, int]
    adjacent: np.ndarray
    degree: np.ndarray
    unit: np.ndarray | None
    texts: list[str]
    token_counts: list[Counter]
    document_frequency: Counter
    name_tokens: list[list[str]]
    modules: list[str]
    classes: list[str]
    static_out: dict[int, set[int]]
    average_length: float = 1.0

    @property
    def size(self) -> int:
        return len(self.nodes)


def load_context(commit: str, graph_path: Path, embedding_path: Path, static: Iterable[tuple[str, str]]) -> Context:
    graph = load_graph(graph_path)
    nodes = list(graph)
    position = {n: i for i, n in enumerate(nodes)}
    adjacent = np.zeros((len(nodes), len(nodes)), dtype=bool)
    for u, v in graph.edges():
        adjacent[position[u], position[v]] = adjacent[position[v], position[u]] = True
    degree = adjacent.sum(1).astype(float)
    unit = None
    if embedding_path.exists():
        with np.load(embedding_path) as data:
            matrix = np.zeros((len(nodes), 256), dtype=np.float32)
            for i, node in enumerate(nodes):
                if node in data.files:
                    matrix[i] = data[node]
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        unit = matrix / np.maximum(norms, 1e-9)
    texts = [graph.nodes[n].get("text", "") or "" for n in nodes]
    token_counts = [Counter(tokenize(t)) for t in texts]
    document_frequency = Counter(tok for counts in token_counts for tok in counts)
    static_out: dict[int, set[int]] = defaultdict(set)
    for u, v in static:
        if u in position and v in position:
            static_out[position[u]].add(position[v])
    return Context(
        commit, nodes, position, adjacent, degree, unit, texts, token_counts, document_frequency,
        [tokenize(n.rpartition(".")[2]) for n in nodes], [module_of(n) for n in nodes],
        [class_prefix(n) for n in nodes], static_out,
        max(float(np.mean([sum(c.values()) for c in token_counts])), 1.0),
    )


def bm25(ctx: Context, source: int, k1: float = 1.2, b: float = 0.75) -> np.ndarray:
    """BM25 of each candidate's name tokens against the caller's source text, idf over all node texts."""
    counts = ctx.token_counts[source]
    length = sum(counts.values())
    average = ctx.average_length
    n = ctx.size
    weight = {}
    for token, tf in counts.items():
        df = ctx.document_frequency[token]
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        weight[token] = idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * length / average))
    return np.array([sum(weight.get(t, 0.0) for t in set(tokens)) for tokens in ctx.name_tokens])


def cosine(ctx: Context, source: int) -> np.ndarray:
    return np.zeros(ctx.size) if ctx.unit is None else ctx.unit @ ctx.unit[source]


def adjacency_scores(ctx: Context, source: int) -> np.ndarray:
    """Provided-graph neighbours first, then everything else by degree."""
    return ctx.adjacent[source] * (ctx.degree.max() + 1.0) + ctx.degree


def static_scores(ctx: Context, source: int) -> np.ndarray:
    scores = np.zeros(ctx.size)
    scores[list(ctx.static_out.get(source, ()))] = 1.0
    return scores + 1e-3 * ctx.adjacent[source]


def feature_matrix(ctx: Context, source: int) -> np.ndarray:
    node = ctx.nodes[source]
    module, klass = module_of(node), class_prefix(node)
    return np.column_stack([
        bm25(ctx, source),
        cosine(ctx, source),
        [m == module for m in ctx.modules],
        [c == klass for c in ctx.classes],
        ctx.adjacent[source],
        np.log1p(ctx.degree),
    ]).astype(float)


BASELINES = {
    "provided adjacency": adjacency_scores,
    "provided embeddings (cosine)": cosine,
    "lexical (BM25)": bm25,
    "callproof static calls": static_scores,
}


# Learned scorer: logistic regression in plain numpy.


@dataclass
class LogisticScorer:
    mean: np.ndarray
    scale: np.ndarray
    weights: np.ndarray
    bias: float

    @classmethod
    def fit(cls, x: np.ndarray, y: np.ndarray, steps: int = 400, l2: float = 1e-3) -> LogisticScorer:
        mean, scale = x.mean(0), np.maximum(x.std(0), 1e-9)
        z = (x - mean) / scale
        weights, bias = np.zeros(z.shape[1]), 0.0
        positive = max(y.sum(), 1.0)
        sample_weight = np.where(y > 0, 0.5 / positive, 0.5 / max((1 - y).sum(), 1.0)) * len(y)
        for _ in range(steps):
            p = 1 / (1 + np.exp(-(z @ weights + bias)))
            error = (p - y) * sample_weight
            weights -= 0.5 * (z.T @ error / len(y) + l2 * weights)
            bias -= 0.5 * error.mean()
        return cls(mean, scale, weights, bias)

    def score(self, x: np.ndarray) -> np.ndarray:
        return ((x - self.mean) / self.scale) @ self.weights + self.bias


def training_matrix(
    contexts: dict[str, Context], examples: Sequence[dict], negatives: int = 50, seed: int = 0
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    by_source: dict[tuple[str, str], set[str]] = defaultdict(set)
    for ex in examples:
        by_source[(ex["commit"], ex["source"])].add(ex["target"])
    xs, ys = [], []
    for (commit, source), targets in by_source.items():
        ctx = contexts[commit]
        features = feature_matrix(ctx, ctx.position[source])
        positive = [ctx.position[t] for t in targets]
        pool = np.setdiff1d(np.arange(ctx.size), positive + [ctx.position[source]])
        chosen = rng.choice(pool, size=min(negatives, len(pool)), replace=False)
        xs += [features[positive], features[chosen]]
        ys += [np.ones(len(positive)), np.zeros(len(chosen))]
    return np.vstack(xs), np.concatenate(ys)


# Evaluation.


def evaluate(
    contexts: dict[str, Context],
    examples: Sequence[dict],
    scorers: dict[str, Callable[[Context, int], np.ndarray]],
    known_examples: Sequence[dict] | None = None,
) -> dict[str, dict[str, float]]:
    """Filtered ranking metrics of every scorer over ``examples``.

    ``known_examples`` (default: ``examples``) are all true pairs, removed from the candidates of other targets.
    """
    known_by_source: dict[tuple[str, str], list[str]] = defaultdict(list)
    for ex in known_examples if known_examples is not None else examples:
        known_by_source[(ex["commit"], ex["source"])].append(ex["target"])
    by_source: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for ex in examples:
        by_source[(ex["commit"], ex["source"])].append(ex)
    rows: dict[str, list] = {name: [] for name in scorers}
    for (commit, source), group in by_source.items():
        ctx = contexts[commit]
        src = ctx.position[source]
        known = np.array([ctx.position[t] for t in known_by_source[(commit, source)]])
        for name, scorer in scorers.items():
            scores = np.asarray(scorer(ctx, src), dtype=float)
            for ex in group:
                target = ctx.position[ex["target"]]
                keep = np.ones(ctx.size, dtype=bool)
                keep[known] = False
                keep[src] = False
                rows[name].append((commit, source, tied_metrics(scores[target], scores[keep])))
    return {name: aggregate(r) for name, r in rows.items()}
