# CallproofLP results

Filtered ranking over all nodes of the provided graph of the commit; ties broken in expectation. Macro = mean over (commit, source) of the mean over its callees; pooled = mean over pairs.

## Split by repository (train: rich, requests, httpx; test: fastapi)

train: 5001 pairs / 50 commits; test: 2361 pairs / 55 commits / 1181 sources; hard (not linked in provided graph): 814 pairs

LR weights (standardised): lexical=+1.05, cosine=+0.89, same_module=+0.42, same_class=+0.27, adjacent=+0.39, log_degree=-0.37

**all test pairs**

| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |
|---|---|---|---|---|---|---|
| provided adjacency | 0.347 | 0.168 | 0.592 | 0.358 | 0.169 | 0.648 |
| provided embeddings (cosine) | 0.153 | 0.080 | 0.306 | 0.141 | 0.064 | 0.310 |
| lexical (BM25) | 0.280 | 0.187 | 0.465 | 0.246 | 0.155 | 0.433 |
| callproof static calls | 0.701 | 0.601 | 0.841 | 0.665 | 0.524 | 0.870 |
| logistic regression | 0.473 | 0.360 | 0.674 | 0.463 | 0.350 | 0.667 |

**hard subset (not linked in provided graph)**

| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |
|---|---|---|---|---|---|---|
| provided adjacency | 0.008 | 0.000 | 0.010 | 0.007 | 0.000 | 0.007 |
| provided embeddings (cosine) | 0.091 | 0.051 | 0.164 | 0.076 | 0.041 | 0.138 |
| lexical (BM25) | 0.142 | 0.104 | 0.213 | 0.141 | 0.101 | 0.210 |
| callproof static calls | 0.450 | 0.326 | 0.623 | 0.413 | 0.267 | 0.625 |
| logistic regression | 0.145 | 0.078 | 0.268 | 0.143 | 0.079 | 0.258 |

## Random-commit split (30% of commits held out, seed 0)

train: 5667 pairs / 80 commits; test: 1695 pairs / 25 commits / 741 sources; hard (not linked in provided graph): 1261 pairs

LR weights (standardised): lexical=+1.05, cosine=+1.69, same_module=+0.52, same_class=+0.12, adjacent=+0.49, log_degree=-0.00

**all test pairs**

| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |
|---|---|---|---|---|---|---|
| provided adjacency | 0.139 | 0.055 | 0.254 | 0.126 | 0.044 | 0.254 |
| provided embeddings (cosine) | 0.090 | 0.044 | 0.186 | 0.072 | 0.032 | 0.149 |
| lexical (BM25) | 0.227 | 0.143 | 0.385 | 0.149 | 0.086 | 0.269 |
| callproof static calls | 0.625 | 0.476 | 0.863 | 0.520 | 0.330 | 0.854 |
| logistic regression | 0.377 | 0.242 | 0.655 | 0.297 | 0.172 | 0.556 |

**hard subset (not linked in provided graph)**

| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |
|---|---|---|---|---|---|---|
| provided adjacency | 0.003 | 0.000 | 0.002 | 0.003 | 0.000 | 0.001 |
| provided embeddings (cosine) | 0.065 | 0.032 | 0.130 | 0.049 | 0.023 | 0.094 |
| lexical (BM25) | 0.184 | 0.118 | 0.303 | 0.121 | 0.072 | 0.210 |
| callproof static calls | 0.532 | 0.363 | 0.808 | 0.430 | 0.225 | 0.805 |
| logistic regression | 0.250 | 0.127 | 0.530 | 0.185 | 0.079 | 0.427 |

# Learned baselines (linkpred_learned.py)

Same pairs, splits, candidate sets and filtered metrics as above. Pair rankers use 9 cheap features (cosine, bm25, same_module, same_class, distance, name_overlap, name_in_source, cand_log_degree, source_log_degree) with 100 random negatives per training source; GraphSAGE is 2-layer mean-aggregator (128-d, bilinear decoder) over the provided graph with provided embeddings plus log degree as input, trained on the training-split graphs only. The self-supervised row predicts 20% held-out edges per step with a full softmax over all nodes and never sees labels; the other rows add a linear decoder term on the observed-edge bit (and, in the last row, the 9 pair features) and are trained on the training labels, from scratch or from the self-supervised weights. The self-supervised row scores pairs that are mostly visible edges of the message-passing graph, which it was never trained to do; in a sanity check on 6 training graphs with a separate 10% of edges held out, it recovers the held-out edges with MRR 0.26 (random about 0.005).

## Split by repository

train 5001 pairs, test 2361 pairs, hard 814

LightGBM gain importance: name_in_source=0.34, bm25=0.17, distance=0.14, same_class=0.10, cosine=0.08, same_module=0.05, source_log_degree=0.05, cand_log_degree=0.04, name_overlap=0.02

**all test pairs**

| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |
|---|---|---|---|---|---|---|
| provided adjacency | 0.347 | 0.168 | 0.592 | 0.358 | 0.169 | 0.648 |
| provided embeddings (cosine) | 0.153 | 0.080 | 0.306 | 0.141 | 0.064 | 0.310 |
| lexical (BM25) | 0.280 | 0.187 | 0.465 | 0.246 | 0.155 | 0.433 |
| callproof static calls | 0.701 | 0.601 | 0.841 | 0.665 | 0.524 | 0.870 |
| pair ranker: logistic regression | 0.664 | 0.544 | 0.853 | 0.632 | 0.499 | 0.868 |
| pair ranker: LightGBM | 0.601 | 0.463 | 0.822 | 0.586 | 0.436 | 0.844 |
| GraphSAGE self-supervised on edges | 0.034 | 0.000 | 0.091 | 0.028 | 0.001 | 0.064 |
| GraphSAGE self-supervised + fine-tuned on labels (decoder sees edge bit) | 0.374 | 0.215 | 0.594 | 0.358 | 0.169 | 0.649 |
| GraphSAGE trained on labels only (decoder sees edge bit) | 0.369 | 0.199 | 0.626 | 0.360 | 0.174 | 0.664 |
| GraphSAGE self-supervised + fine-tuned, decoder also gets pair features | 0.672 | 0.549 | 0.864 | 0.643 | 0.505 | 0.870 |

**hard test pairs**

| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |
|---|---|---|---|---|---|---|
| provided adjacency | 0.008 | 0.000 | 0.010 | 0.007 | 0.000 | 0.007 |
| provided embeddings (cosine) | 0.091 | 0.051 | 0.164 | 0.076 | 0.041 | 0.138 |
| lexical (BM25) | 0.142 | 0.104 | 0.213 | 0.141 | 0.101 | 0.210 |
| callproof static calls | 0.450 | 0.326 | 0.623 | 0.413 | 0.267 | 0.625 |
| pair ranker: logistic regression | 0.390 | 0.262 | 0.646 | 0.389 | 0.246 | 0.662 |
| pair ranker: LightGBM | 0.388 | 0.260 | 0.628 | 0.393 | 0.258 | 0.646 |
| GraphSAGE self-supervised on edges | 0.020 | 0.002 | 0.016 | 0.021 | 0.002 | 0.020 |
| GraphSAGE self-supervised + fine-tuned on labels (decoder sees edge bit) | 0.013 | 0.000 | 0.011 | 0.011 | 0.000 | 0.010 |
| GraphSAGE trained on labels only (decoder sees edge bit) | 0.026 | 0.000 | 0.077 | 0.021 | 0.000 | 0.058 |
| GraphSAGE self-supervised + fine-tuned, decoder also gets pair features | 0.394 | 0.263 | 0.644 | 0.384 | 0.246 | 0.652 |

## Random-commit split

train 5667 pairs, test 1695 pairs, hard 1261

LightGBM gain importance: name_in_source=0.35, distance=0.28, bm25=0.12, same_class=0.07, cosine=0.06, source_log_degree=0.04, cand_log_degree=0.03, same_module=0.03, name_overlap=0.02

**all test pairs**

| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |
|---|---|---|---|---|---|---|
| provided adjacency | 0.139 | 0.055 | 0.254 | 0.126 | 0.044 | 0.254 |
| provided embeddings (cosine) | 0.090 | 0.044 | 0.186 | 0.072 | 0.032 | 0.149 |
| lexical (BM25) | 0.227 | 0.143 | 0.385 | 0.149 | 0.086 | 0.269 |
| callproof static calls | 0.625 | 0.476 | 0.863 | 0.520 | 0.330 | 0.854 |
| pair ranker: logistic regression | 0.547 | 0.420 | 0.782 | 0.465 | 0.322 | 0.743 |
| pair ranker: LightGBM | 0.712 | 0.598 | 0.908 | 0.642 | 0.504 | 0.885 |
| GraphSAGE self-supervised on edges | 0.021 | 0.004 | 0.044 | 0.017 | 0.002 | 0.031 |
| GraphSAGE self-supervised + fine-tuned on labels (decoder sees edge bit) | 0.257 | 0.193 | 0.384 | 0.231 | 0.170 | 0.344 |
| GraphSAGE trained on labels only (decoder sees edge bit) | 0.248 | 0.182 | 0.372 | 0.224 | 0.164 | 0.335 |
| GraphSAGE self-supervised + fine-tuned, decoder also gets pair features | 0.602 | 0.483 | 0.810 | 0.518 | 0.378 | 0.761 |

**hard test pairs**

| method | macro MRR | macro H@1 | macro H@10 | pooled MRR | pooled H@1 | pooled H@10 |
|---|---|---|---|---|---|---|
| provided adjacency | 0.003 | 0.000 | 0.002 | 0.003 | 0.000 | 0.001 |
| provided embeddings (cosine) | 0.065 | 0.032 | 0.130 | 0.049 | 0.023 | 0.094 |
| lexical (BM25) | 0.184 | 0.118 | 0.303 | 0.121 | 0.072 | 0.210 |
| callproof static calls | 0.532 | 0.363 | 0.808 | 0.430 | 0.225 | 0.805 |
| pair ranker: logistic regression | 0.476 | 0.359 | 0.702 | 0.409 | 0.271 | 0.665 |
| pair ranker: LightGBM | 0.636 | 0.504 | 0.877 | 0.569 | 0.417 | 0.845 |
| GraphSAGE self-supervised on edges | 0.009 | 0.002 | 0.011 | 0.007 | 0.001 | 0.007 |
| GraphSAGE self-supervised + fine-tuned on labels (decoder sees edge bit) | 0.067 | 0.029 | 0.169 | 0.045 | 0.014 | 0.122 |
| GraphSAGE trained on labels only (decoder sees edge bit) | 0.057 | 0.016 | 0.155 | 0.038 | 0.008 | 0.111 |
| GraphSAGE self-supervised + fine-tuned, decoder also gets pair features | 0.514 | 0.395 | 0.735 | 0.445 | 0.310 | 0.684 |

