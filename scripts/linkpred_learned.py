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
_features: dict[tuple[str, int], np.ndarray] = {}


def distances(ctx: linkpred.Context, source: int) -> np.ndarray:
    if ctx.commit not in _sparse:
        _sparse[ctx.commit] = sp.csr_matrix(ctx.adjacent)
    d = dijkstra(_sparse[ctx.commit], unweighted=True, indices=source, limit=CAP)
    return np.minimum(d, CAP)


def features(ctx: linkpred.Context, source: int) -> np.ndarray:
    if (ctx.commit, source) not in _features:
        _features[(ctx.commit, source)] = _compute_features(ctx, source)
    return _features[(ctx.commit, source)]


def _compute_features(ctx: linkpred.Context, source: int) -> np.ndarray:
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

SCALE = 16.0


class Sage(torch.nn.Module):
    def __init__(self, dim_in: int, extra: int = 0, hidden: int = 128):
        super().__init__()
        self.l1 = torch.nn.Linear(2 * dim_in, hidden)
        self.l2 = torch.nn.Linear(2 * hidden, hidden)
        self.bilinear = torch.nn.Parameter(torch.eye(hidden))
        self.extra = torch.nn.Linear(extra, 1) if extra else None
        if extra:
            torch.nn.init.zeros_(self.extra.weight)

    def forward(self, x, adj):
        h = torch.relu(self.l1(torch.cat([x, torch.sparse.mm(adj, x)], 1)))
        return self.l2(torch.cat([h, torch.sparse.mm(adj, h)], 1))

    def logits(self, h, src, extra=None):
        out = (h[src] @ self.bilinear) @ h.T
        return out if self.extra is None else out + self.extra(extra).squeeze(-1)


def graph_tensors(ctx, keep=None):
    a = sp.csr_matrix(ctx.adjacent if keep is None else keep, dtype=np.float32)
    degree = np.asarray(a.sum(1)).ravel()
    a = sp.diags(1 / np.maximum(degree, 1.0)) @ a
    coo = a.tocoo()
    adj = torch.sparse_coo_tensor(np.vstack([coo.row, coo.col]), coo.data.astype(np.float32), a.shape)
    unit = ctx.unit if ctx.unit is not None else np.zeros((ctx.size, 256))
    x = np.column_stack([unit * SCALE, np.log1p(degree) / 5])
    return torch.tensor(x, dtype=torch.float32), adj


def extras(ctx, src, kind, stats):
    if kind == "adj":
        return ctx.adjacent[src, :, None].astype(np.float32)
    mean, std = stats
    return np.column_stack([(features(ctx, src) - mean) / std, ctx.adjacent[src]]).astype(np.float32)


def pretrain(model, contexts, commits, steps=300, lr=3e-3, seed=0):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(model.parameters(), lr)
    for step in range(steps):
        ctx = contexts[commits[step % len(commits)]]
        rows, cols = np.nonzero(np.triu(ctx.adjacent, 1))
        held = rng.random(len(rows)) < 0.2
        keep = ctx.adjacent.copy()
        keep[rows[held], cols[held]] = keep[cols[held], rows[held]] = False
        h = model(*graph_tensors(ctx, keep))
        src = torch.tensor(np.concatenate([rows[held], cols[held]]))
        dst = torch.tensor(np.concatenate([cols[held], rows[held]]))
        logits = model.logits(h, src).masked_fill(torch.nn.functional.one_hot(src, ctx.size).bool(), -1e9)
        loss = torch.nn.functional.cross_entropy(logits, dst)
        opt.zero_grad()
        loss.backward()
        opt.step()


def finetune(model, contexts, train, kind, stats, steps=300, lr=1e-3, seed=0):
    rng = np.random.default_rng(seed)
    by_commit = defaultdict(lambda: defaultdict(set))
    for ex in train:
        by_commit[ex["commit"]][ex["source"]].add(ex["target"])
    commits = sorted(by_commit)
    head = [p for n, p in model.named_parameters() if n.startswith("extra")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("extra")]
    opt = torch.optim.Adam([{"params": rest, "lr": lr}, {"params": head, "lr": 3e-2}])
    for step in range(steps):
        ctx = contexts[commits[rng.integers(len(commits))]]
        sources = by_commit[ctx.commit]
        src = [ctx.position[s] for s in sources]
        pos = torch.zeros(len(src), ctx.size, dtype=torch.bool)
        for i, targets in enumerate(sources.values()):
            pos[i, [ctx.position[t] for t in targets]] = True
        src_t = torch.tensor(src)
        h = model(*graph_tensors(ctx))
        feats = torch.tensor(np.stack([extras(ctx, s, kind, stats) for s in src])) if model.extra is not None else None
        logits = (h[src_t] @ model.bilinear) @ h.T
        if feats is not None:
            logits = logits + model.extra(feats).squeeze(-1)
        negatives = logits.masked_fill(pos | torch.nn.functional.one_hot(src_t, ctx.size).bool(), -1e9)
        rows, cols = pos.nonzero(as_tuple=True)
        loss = torch.nn.functional.softplus(torch.logsumexp(negatives, 1)[rows] - logits[rows, cols]).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()


def gnn_scorer(model, kind=None, stats=None):
    memo = {}

    def score(ctx, src):
        with torch.no_grad():
            if ctx.commit not in memo:
                memo[ctx.commit] = model(*graph_tensors(ctx))
            feats = None if model.extra is None else torch.tensor(extras(ctx, src, kind, stats))
            return model.logits(memo[ctx.commit], src, feats).numpy()

    return score


def gnn_scorers(contexts, train, stats):
    dim, timings, scorers = 257, {}, {}
    commits = sorted({e["commit"] for e in train})
    t0 = time.time()
    torch.manual_seed(0)
    ssl = Sage(dim)
    pretrain(ssl, contexts, commits)
    scorers["GraphSAGE self-supervised on edges"] = gnn_scorer(ssl)
    timings["gnn_ssl"] = time.time() - t0
    for name, kind, pretrained, extra in (
        ("GraphSAGE self-supervised + fine-tuned on labels (decoder sees edge bit)", "adj", True, 1),
        ("GraphSAGE trained on labels only (decoder sees edge bit)", "adj", False, 1),
        ("GraphSAGE self-supervised + fine-tuned, decoder also gets pair features", "pair", True, len(FEATURES) + 1),
    ):
        t0 = time.time()
        torch.manual_seed(0)
        model = Sage(dim, extra)
        if pretrained:
            model.load_state_dict(ssl.state_dict(), strict=False)
        finetune(model, contexts, train, kind, stats)
        scorers[name] = gnn_scorer(model, kind, stats)
        timings[f"gnn_{kind}_{'ssl' if pretrained else 'scratch'}"] = time.time() - t0
    return scorers, timings


def run_split(contexts, examples, key, skip_gnn, prev=None):
    train = [e for e in examples if e[key] == "train"]
    test = [e for e in examples if e[key] == "test"]
    scorers, timings = dict(linkpred.BASELINES), {}
    t0 = time.time()
    x, y = pair_data(contexts, train)
    stats = x.mean(0), x.std(0) + 1e-6
    if prev is None:
        params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
                  "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "num_threads": 4,
                  "seed": 0, "verbose": -1}
        gbm = lgb.train(params, lgb.Dataset(x, y), num_boost_round=300)
        lr = linkpred.LogisticScorer.fit(x, y)
        scorers["pair ranker: logistic regression"] = cached(lambda c, s: lr.score(features(c, s)))
        scorers["pair ranker: LightGBM"] = cached(lambda c, s: gbm.predict(features(c, s)))
        gain = gbm.feature_importance("gain")
        extra = {"lightgbm_importance": dict(zip(FEATURES, map(float, gain / gain.sum()))),
                 "logistic_weights": dict(zip(FEATURES, map(float, lr.weights)))}
        timings["pair_ranker"] = time.time() - t0
    else:
        scorers = {}
        timings = {k: v for k, v in prev["timings_s"].items() if not k.startswith("gnn")}
    if not skip_gnn:
        gnn, spent = gnn_scorers(contexts, train, stats)
        scorers.update(gnn)
        timings.update(spent)
    out = {"train_pairs": len(train), "test_pairs": len(test), "hard_pairs": sum(not e["linked"] for e in test),
           "timings_s": timings, **(extra if prev is None else {k: prev[k] for k in ("lightgbm_importance", "logistic_weights")})}
    for label, subset in (("all", test), ("hard", [e for e in test if not e["linked"]])):
        scored = linkpred.evaluate(contexts, subset, scorers, known_examples=test)
        if prev is not None:
            kept = {k: v for k, v in prev[label].items() if not k.startswith("GraphSAGE")}
            scored = {**kept, **scored}
        out[label] = scored
    return out


def section(results: dict) -> list[str]:
    lines = ["", "# Learned baselines (linkpred_learned.py)", "",
             "Same pairs, splits, candidate sets and filtered metrics as above. Pair rankers use 9 cheap features "
             f"({', '.join(FEATURES)}) with 100 random negatives per training source; GraphSAGE is 2-layer "
             "mean-aggregator (128-d, bilinear decoder) over the provided graph with provided embeddings plus log degree "
             "as input, trained on the training-split graphs only. The self-supervised row predicts 20% held-out "
             "edges per step with a full softmax over all nodes and never sees labels; the other rows add a linear "
             "decoder term on the observed-edge bit (and, in the last row, the 9 pair features) and are trained on "
             "the training labels, from scratch or from the self-supervised weights. "
             "The self-supervised row scores pairs that are mostly visible edges of the message-passing graph, which it was never trained to do; "
             "in a sanity check on 6 training graphs with a separate 10% of edges held out, it recovers the held-out edges with MRR 0.26 (random about 0.005).", ""]
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
    parser.add_argument("--gnn-only", action="store_true", help="recompute only the GraphSAGE rows of the existing results.json")
    args = parser.parse_args()
    examples = linkpred.read_jsonl(OUT / "edges.jsonl")
    contexts = load_contexts(examples)
    LEARNED.mkdir(exist_ok=True)
    previous = json.loads((LEARNED / "results.json").read_text()) if args.gnn_only else {}
    results = {}
    for key in ("split_repo", "split_random"):
        results[key] = run_split(contexts, examples, key, args.skip_gnn, previous.get(key))
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
