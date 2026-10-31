"""CallproofLP on six repositories the Callproof rules were never developed on.

    linkpred_unseen.py   # needs the traces generality.py writes under $CALLPROOF_CACHE/general; writes results/linkpred/unseen/*

Candidate nodes are those of the provided-style reconstruction graph (no provided graphs or embeddings exist for
these repositories); node text is the definition's source. The cosine feature is constant zero for training and
testing alike. Pair rankers are trained on all pairs of the competition CallproofLP.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import lightgbm as lgb

sys.path.insert(0, str(Path(__file__).resolve().parent))

import linkpred_learned as learned  # noqa: E402
from generality import REPOS, WORK  # noqa: E402
from linkpred import OUT, fmt, load_contexts  # noqa: E402
from provided_style import build_provided_style  # noqa: E402

from callproof import linkpred  # noqa: E402
from callproof.build import GraphBuilder  # noqa: E402
from callproof.graphio import save_graph  # noqa: E402
from callproof.naming import RepoIndex, parse_checkout  # noqa: E402

UNSEEN = OUT / "unseen"
THREADS = 16


def with_text(graph, index, root: Path):
    lines: dict[str, list[str]] = {}
    for d in index.definitions:
        if d.harness_id in graph and not graph.nodes[d.harness_id].get("text"):
            src = lines.setdefault(d.path, (root / d.path).read_text(encoding="utf-8", errors="replace").splitlines())
            graph.nodes[d.harness_id]["text"] = "\n".join(src[d.lineno - 1:d.end_lineno])
    return graph


def build(tmp: Path):
    contexts, examples = {}, []
    for repo in REPOS:
        root = WORK / "repos" / repo.name
        modules = parse_checkout(root)
        index = RepoIndex(modules)
        provided = with_text(build_provided_style(index, repo.name), index, root)
        ours = GraphBuilder(modules, repo.name).build()
        commit = f"unseen_{repo.name}"
        raw = json.loads((WORK / "traces" / f"{repo.name}.edges.json").read_text())["edges"]
        rows = linkpred.build_examples(commit, repo.name, raw, index, provided)
        static = [(u, v) for u, v in linkpred.static_call_pairs(ours, index, provided)]
        save_graph(provided, tmp / f"{commit}.json")
        contexts[commit] = linkpred.load_context(commit, tmp / f"{commit}.json", tmp / "none.npz", static)
        examples += rows
    return contexts, examples


def main() -> None:
    started = time.time()
    train = linkpred.read_jsonl(OUT / "edges.jsonl")
    train_contexts = load_contexts(train)
    for ctx in train_contexts.values():
        ctx.unit = None
    x, y = learned.pair_data(train_contexts, train)
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20,
              "bagging_fraction": 0.8, "bagging_freq": 1, "feature_fraction": 0.8, "num_threads": THREADS,
              "seed": 0, "verbose": -1}
    gbm = lgb.train(params, lgb.Dataset(x, y), num_boost_round=300)
    lr = linkpred.LogisticScorer.fit(x, y)
    with tempfile.TemporaryDirectory() as tmp:
        contexts, examples = build(Path(tmp))
    scorers = {name: linkpred.BASELINES[name] for name in ("provided adjacency", "lexical (BM25)", "callproof static calls")}
    scorers["pair ranker: logistic regression"] = learned.cached(lambda c, s: lr.score(learned.features(c, s)))
    scorers["pair ranker: LightGBM"] = learned.cached(lambda c, s: gbm.predict(learned.features(c, s)))
    scorers["provided adjacency"] = linkpred.BASELINES["provided adjacency"]
    hard = [e for e in examples if not e["linked"]]
    out = {"train_pairs": len(train), "pairs": len(examples), "hard_pairs": len(hard),
           "all": linkpred.evaluate(contexts, examples, scorers), "hard": linkpred.evaluate(contexts, hard, scorers, known_examples=examples),
           "per_repo": {}}
    for repo in REPOS:
        commit = f"unseen_{repo.name}"
        sub = [e for e in examples if e["commit"] == commit]
        sub_hard = [e for e in sub if not e["linked"]]
        out["per_repo"][repo.name] = {
            "nodes": contexts[commit].size, "pairs": len(sub), "hard_pairs": len(sub_hard),
            "all": linkpred.evaluate(contexts, sub, scorers), "hard": linkpred.evaluate(contexts, sub_hard, scorers, known_examples=sub)}
    out["seconds"] = round(time.time() - started, 1)
    UNSEEN.mkdir(exist_ok=True)
    (UNSEEN / "results.json").write_text(json.dumps(out, indent=1))
    header = "| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |\n|---|---|---|---|---|---|---|"
    lines = ["# CallproofLP on unseen repositories", "",
             "Six repositories outside the competition (click, attrs, more-itertools, packaging, itsdangerous, h11), traced by "
             "`generality.py`. Candidates: all nodes of the provided-style reconstruction graph of each checkout; "
             "filtered ranking, ties in expectation, macro over (repo, source). Hard = pair not adjacent in that reconstruction. "
             "Cosine is constant zero (no embeddings); pair rankers are trained on all competition pairs with cosine zeroed too. "
             "No GraphSAGE (needs provided embeddings). Adjacency is that of the reconstruction, not the provided graph.", "",
             f"{out['pairs']} pairs, {out['hard_pairs']} hard, {len(contexts)} graphs; runtime {out['seconds']} s.", ""]
    for label in ("all", "hard"):
        lines += [f"## {label} pairs", "", header, *[f"| {n} {fmt(m)}" for n, m in out[label].items()], ""]
    lines += ["## Per repository (macro MRR all / hard)", "", "| repo | nodes | pairs | hard | " + " | ".join(scorers) + " |",
              "|---|---|---|---|" + "---|" * len(scorers)]
    for name, r in out["per_repo"].items():
        cells = [f"{r['all'][s].get('macro_mrr', float('nan')):.3f} / {r['hard'][s].get('macro_mrr', float('nan')):.3f}" for s in scorers]
        lines.append(f"| {name} | {r['nodes']} | {r['pairs']} | {r['hard_pairs']} | " + " | ".join(cells) + " |")
    (UNSEEN / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
