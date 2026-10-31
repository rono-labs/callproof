# CallproofLP

Call-edge link prediction on code graphs. Positives are caller -> callee pairs recorded by executing the
fail-to-pass tests of each task on the gold-patched repository, mapped to node ids of the provided code
graph of the task's base commit. Test-only edges are dropped, pairs are deduplicated per commit.

Task: given a source node, rank all nodes of the same commit's graph as its callees (filtered ranking:
other known callees of the source are removed from the candidates). Metrics: MRR, Hits@1, Hits@10, macro
over (commit, source) and pooled over pairs; ties are resolved in expectation.

## Files

- `edges.jsonl`: one positive per line.
  - `commit`: graph id `<repo>_<base commit>` (the provided graph `graphs/<commit>.json` is the candidate space)
  - `repo`, `source`, `target`: node ids in the provided graph, `count`: traced call count summed over tasks
  - `kind`: `function`, `method` or `constructor` (callee)
  - `linked`: the pair is adjacent (either direction) in the provided graph; `linked == false` is the hard subset
  - `split_repo`: `train` (rich, requests, httpx) or `test` (fastapi); `split_random`: `train` / `test` by commit (30% test, seed 0)
- `static_calls.jsonl.gz`: `[commit, source, target]` calls edges of the Callproof static builder, in provided ids (baseline input).
- `RESULTS.md`: baseline results.

## License note

Only derived labels (node ids, counts, split flags) are released; no source text. Node ids name
definitions of open-source projects (rich, requests, httpx, fastapi); node text and embeddings come from the
competition dataset and must be obtained from it under its terms.

## Reproduce

    export CALLPROOF_DATA=<competition data dir> CALLPROOF_CACHE=<cache dir>
    python project/scripts/linkpred.py build   # needs results/groundtruth and results/graphs
    python project/scripts/linkpred.py eval
