"""Learned baselines for CallproofLP under the protocol of linkpred.py eval.

    linkpred_learned.py [--skip-gnn]   # writes results/linkpred/learned/* and appends a section to RESULTS.md
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import lightgbm as lgb
import numpy as np
import scipy.sparse as sp
import torch
from scipy.sparse.csgraph import dijkstra

sys.path.insert(0, str(Path(__file__).resolve().parent))

from linkpred import OUT, fmt, load_contexts  # noqa: E402

from callproof import linkpred  # noqa: E402

torch.set_num_threads(4)
LEARNED = OUT / "learned"
FEATURES = ("cosine", "bm25", "same_module", "same_class", "distance", "name_overlap", "name_in_source",
            "cand_log_degree", "source_log_degree")
CAP = 6
_IDENT = re.compile(r"\w+")
_sparse: dict[str, sp.csr_matrix] = {}
_idents: dict[tuple[str, int], set[str]] = {}


def distances(ctx: linkpred.Context, source: int) -> np.ndarray:
    if ctx.commit not in _sparse:
        _sparse[ctx.commit] = sp.csr_matrix(ctx.adjacent)
    d = dijkstra(_sparse[ctx.commit], unweighted=True, indices=source, limit=CAP)
    return np.minimum(d, CAP)


def features(ctx: linkpred.Context, source: int) -> np.ndarray:
    node = ctx.nodes[source]
    module, klass = ctx.modules[source], ctx.classes[source]
    idents = _idents.setdefault((ctx.commit, source), set(_IDENT.findall(ctx.texts[source])))
    own = set(ctx.name_tokens[source])
    overlap = [len(own & set(t)) / max(len(own | set(t)), 1) for t in ctx.name_tokens]
    present = [n.rpartition(".")[2] in idents for n in ctx.nodes]
    return np.column_stack([
        linkpred.cosine(ctx, source), linkpred.bm25(ctx, source),
        [m == module for m in ctx.modules], [c == klass for c in ctx.classes],
        distances(ctx, source), overlap, present,
        np.log1p(ctx.degree), np.full(ctx.size, np.log1p(ctx.degree[source])),
    ]).astype(float)


def pair_data(contexts, examples, negatives=100, seed=0):
    rng = np.random.default_rng(seed)
    by_source = defaultdict(set)
    for ex in examples:
        by_source[(ex["commit"], ex["source"])].add(ex["target"])
    xs, ys = [], []
    for (commit, source), targets in by_source.items():
        ctx = contexts[commit]
        src = ctx.position[source]
        feats = features(ctx, src)
        pos = [ctx.position[t] for t in targets]
        pool = np.setdiff1d(np.arange(ctx.size), pos + [src])
        neg = rng.choice(pool, size=min(negatives, len(pool)), replace=False)
        xs += [feats[pos], feats[neg]]
        ys += [np.ones(len(pos)), np.zeros(len(neg))]
    return np.vstack(xs), np.concatenate(ys)


def cached(fn):
    memo = {}
    return lambda ctx, src: memo.setdefault((ctx.commit, src), fn(ctx, src))


# GraphSAGE over the provided graph.


class Sage(torch.nn.Module):
    def __init__(self, dim_in: int, hidden: int = 128):
        super().__init__()
        self.l1 = torch.nn.Linear(2 * dim_in, hidden)
        self.l2 = torch.nn.Linear(2 * hidden, hidden)
        self.bilinear = torch.nn.Parameter(torch.eye(hidden))

    def forward(self, x, adj):
        h = torch.relu(self.l1(torch.cat([x, torch.sparse.mm(adj, x)], 1)))
        return self.l2(torch.cat([h, torch.sparse.mm(adj, h)], 1))

    def logits(self, h, src):
        return (h[src] @ self.bilinear) @ h.T


def graph_tensors(ctx, keep=None):
    a = sp.csr_matrix(ctx.adjacent if keep is None else keep, dtype=np.float32)
    deg = np.maximum(np.asarray(a.sum(1)).ravel(), 1.0)
    a = sp.diags(1 / deg) @ a
    coo = a.tocoo()
    adj = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]), coo.data.astype(np.float32), a.shape)
    x = np.column_stack([ctx.unit if ctx.unit is not None else np.zeros((ctx.size, 256)), np.log1p(ctx.degree) / 5])
    return torch.tensor(x, dtype=torch.float32), adj


def pretrain(model, contexts, commits, epochs=40, seed=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    for _ in range(epochs):
        for commit in commits:
            ctx = contexts[commit]
            rows, cols = np.nonzero(np.triu(ctx.adjacent, 1))
            if len(rows) < 10:
                continue
            held = rng.random(len(rows)) < 0.2
            keep = ctx.adjacent.copy()
            keep[rows[held], cols[held]] = keep[cols[held], rows[held]] = False
            x, adj = graph_tensors(ctx, keep)
            h = model(x, adj)
            r, c = torch.tensor(rows[held]), torch.tensor(cols[held])
            neg = torch.tensor(rng.integers(0, ctx.size, len(r)))
            pos_s = (h[r] @ model.bilinear * h[c]).sum(1)
            neg_s = (h[r] @ model.bilinear * h[neg]).sum(1)
            loss = torch.nn.functional.softplus(-pos_s).mean() + torch.nn.functional.softplus(neg_s).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()


def finetune(model, contexts, train, epochs=30, lr=5e-4):
    by_commit = defaultdict(lambda: defaultdict(set))
    for ex in train:
        by_commit[ex["commit"]][ex["source"]].add(ex["target"])
    opt = torch.optim.Adam(model.parameters(), lr)
    for _ in range(epochs):
        for commit, sources in by_commit.items():
            ctx = contexts[commit]
            x, adj = graph_tensors(ctx)
            h = model(x, adj)
            loss = 0.0
            for source, targets in sources.items():
                src = ctx.position[source]
                logits = model.logits(h, src)
                logits = logits.index_fill(0, torch.tensor([src]), -1e9)
                pos = torch.tensor([ctx.position[t] for t in targets])
                for p in pos:
                    masked = logits.index_fill(0, pos[pos != p], -1e9)
                    loss = loss + (torch.logsumexp(masked, 0) - masked[p])
            loss = loss / sum(len(s) for s in sources.values())
            opt.zero_grad()
            loss.backward()
            opt.step()


def gnn_scorer(model, contexts):
    memo = {}

    def score(ctx, src):
        if ctx.commit not in memo:
            with torch.no_grad():
                memo[ctx.commit] = model(*graph_tensors(ctx))
        with torch.no_grad():
            return model.logits(memo[ctx.commit], src).numpy()

    return score


def run_split(contexts, examples, key, skip_gnn):
    train = [e for e in examples if e[key] == "train"]
    test = [e for e in examples if e[key] == "test"]
    train_commits = sorted({e["commit"] for e in train})
    scorers, extra, timings = dict(linkpred.BASELINES), {}, {}
    t0 = time.time()
    x, y = pair_data(contexts, train)
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
              "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "num_threads": 4,
              "seed": 0, "verbose": -1}
    gbm = lgb.train(params, lgb.Dataset(x, y), num_boost_round=300)
    lr = linkpred.LogisticScorer.fit(x, y)
    scorers["pair ranker: logistic regression"] = cached(lambda c, s: lr.score(features(c, s)))
    scorers["pair ranker: LightGBM"] = cached(lambda c, s: gbm.predict(features(c, s)))
    gain = gbm.feature_importance("gain")
    imp = gain / gain.sum()
    extra["lightgbm_importance"] = dict(zip(FEATURES, map(float, imp)))
    extra["logistic_weights"] = dict(zip(FEATURES, map(float, lr.weights)))
    timings["pair_ranker"] = time.time() - t0
    if not skip_gnn:
        dim = 257
        t0 = time.time()
        torch.manual_seed(0)
        ssl = Sage(dim)
        pretrain(ssl, contexts, train_commits)
        scorers["GraphSAGE self-supervised on edges"] = gnn_scorer(ssl, contexts)
        timings["gnn_ssl"] = time.time() - t0
        t0 = time.time()
        tuned = Sage(dim)
        tuned.load_state_dict(ssl.state_dict())
        finetune(tuned, contexts, train)
        scorers["GraphSAGE self-supervised + fine-tuned on labels"] = gnn_scorer(tuned, contexts)
        timings["gnn_finetune"] = time.time() - t0
        t0 = time.time()
        torch.manual_seed(0)
        scratch = Sage(dim)
        finetune(scratch, contexts, train)
        scorers["GraphSAGE trained on labels only"] = gnn_scorer(scratch, contexts)
        timings["gnn_scratch"] = time.time() - t0
    out = {"train_pairs": len(train), "test_pairs": len(test), "hard_pairs": sum(not e["linked"] for e in test),
           "timings_s": timings, **extra}
    for label, subset in (("all", test), ("hard", [e for e in test if not e["linked"]])):
        out[label] = linkpred.evaluate(contexts, subset, scorers, known_examples=test)
    return out


def section(results: dict) -> list[str]:
    lines = ["", "# Learned baselines (linkpred_learned.py)", "",
             "Same pairs, splits, candidate sets and filtered metrics as above. Pair rankers use 9 cheap features "
             f"({', '.join(FEATURES)}) with 100 random negatives per training source; GraphSAGE is 2-layer "
             "mean-aggregator (128-d, bilinear decoder) over the provided graph with provided embeddings plus log degree "
             "as input, trained on the training-split graphs only.", ""]
    for key, title in (("split_repo", "Split by repository"), ("split_random", "Random-commit split")):
        r = results[key]
        lines += [f"## {title}", "", f"train {r['train_pairs']} pairs, test {r['test_pairs']} pairs, hard {r['hard_pairs']}", "",
                  "LightGBM gain importance: " + ", ".join(f"{k}={v:.2f}" for k, v in sorted(r["lightgbm_importance"].items(), key=lambda kv: -kv[1])), ""]
        for label in ("all", "hard"):
            lines += [f"**{label} test pairs**", "",
                      "| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |",
                      "|---|---|---|---|---|---|---|"]
            lines += [f"| {name} {fmt(m)}" for name, m in r[label].items()]
            lines.append("")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gnn", action="store_true")
    args = parser.parse_args()
    examples = linkpred.read_jsonl(OUT / "edges.jsonl")
    contexts = load_contexts(examples)
    LEARNED.mkdir(exist_ok=True)
    results = {}
    for key in ("split_repo", "split_random"):
        results[key] = run_split(contexts, examples, key, args.skip_gnn)
        print(key, results[key]["timings_s"], flush=True)
    (LEARNED / "results.json").write_text(json.dumps(results, indent=1))
    text = "\n".join(section(results)) + "\n"
    (LEARNED / "section.md").write_text(text)
    base = (OUT / "RESULTS.md").read_text()
    marker = "\n# Learned baselines"
    (OUT / "RESULTS.md").write_text(base.split(marker)[0].rstrip("\n") + "\n" + text)
    print(text)


if __name__ == "__main__":
    main()
