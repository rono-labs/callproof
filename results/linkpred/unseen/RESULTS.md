# CallproofLP on unseen repositories

Six repositories outside the competition (click, attrs, more-itertools, packaging, itsdangerous, h11), traced by `generality.py`. Candidates: all nodes of the provided-style reconstruction graph of each checkout; filtered ranking, ties in expectation, macro over (repo, source). Hard = pair not adjacent in that reconstruction. Cosine is constant zero (no embeddings); pair rankers are trained on all competition pairs with cosine zeroed too. No GraphSAGE (needs provided embeddings). Adjacency is that of the reconstruction, not the provided graph.

1068 pairs, 735 hard, 6 graphs; runtime 95.3 s.

## all pairs

| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |
|---|---|---|---|---|---|---|
| provided adjacency | 0.248 | 0.159 | 0.355 | 0.200 | 0.113 | 0.311 |
| lexical (BM25) | 0.197 | 0.115 | 0.363 | 0.169 | 0.093 | 0.318 |
| callproof static calls | 0.687 | 0.571 | 0.841 | 0.613 | 0.477 | 0.814 |
| pair ranker: logistic regression | 0.568 | 0.451 | 0.777 | 0.497 | 0.365 | 0.736 |
| pair ranker: LightGBM | 0.566 | 0.454 | 0.772 | 0.503 | 0.376 | 0.742 |

## hard pairs

| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |
|---|---|---|---|---|---|---|
| provided adjacency | 0.005 | 0.000 | 0.002 | 0.005 | 0.000 | 0.002 |
| lexical (BM25) | 0.143 | 0.080 | 0.272 | 0.126 | 0.069 | 0.239 |
| callproof static calls | 0.546 | 0.398 | 0.747 | 0.492 | 0.337 | 0.730 |
| pair ranker: logistic regression | 0.415 | 0.284 | 0.656 | 0.373 | 0.238 | 0.634 |
| pair ranker: LightGBM | 0.410 | 0.280 | 0.664 | 0.371 | 0.228 | 0.660 |

## Per repository (macro MRR all / hard)

| repo | nodes | pairs | hard | provided adjacency | lexical (BM25) | callproof static calls | pair ranker: logistic regression | pair ranker: LightGBM |
|---|---|---|---|---|---|---|---|---|
| click | 1573 | 234 | 166 | 0.224 / 0.003 | 0.205 / 0.119 | 0.672 / 0.527 | 0.563 / 0.430 | 0.565 / 0.411 |
| attrs | 1365 | 157 | 95 | 0.314 / 0.002 | 0.264 / 0.211 | 0.615 / 0.358 | 0.597 / 0.393 | 0.520 / 0.337 |
| more-itertools | 1322 | 85 | 38 | 0.477 / 0.002 | 0.182 / 0.110 | 0.735 / 0.452 | 0.764 / 0.532 | 0.714 / 0.446 |
| packaging | 2258 | 362 | 263 | 0.163 / 0.001 | 0.107 / 0.059 | 0.717 / 0.628 | 0.509 / 0.376 | 0.541 / 0.415 |
| itsdangerous | 136 | 87 | 60 | 0.258 / 0.023 | 0.281 / 0.265 | 0.791 / 0.710 | 0.597 / 0.577 | 0.626 / 0.565 |
| h11 | 251 | 143 | 113 | 0.139 / 0.007 | 0.269 / 0.239 | 0.623 / 0.546 | 0.445 / 0.323 | 0.485 / 0.355 |
