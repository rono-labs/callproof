# Extra analyses: precision proxy, normalised reachability, model-free walker

## 1. Precision proxy (static out-neighbours of executed definitions that also ran)

Bias: a real call that the F2P tests simply do not exercise counts as a miss, so every share is a lower bound on
precision; the provided graph is symmetric, so its out-neighbours are all neighbours (callers included).
Callproof uses `calls` edges only. Intervals are task-bootstrap 95%.

| graph | tasks | edges from executed nodes | mean out-degree | pooled | macro |
|---|---|---|---|---|---|
| provided (all neighbours) | 114 | 19081 | 2.9 | 32.2% [28.7%, 36.0%] | 28.7% [25.5%, 32.0%] |
| callproof (calls edges) | 114 | 15491 | 2.2 | 61.9% [59.7%, 63.7%] | 57.1% [53.8%, 60.3%] |

## 2. Reachability of gold modified definitions within k undirected hops

Tasks are those whose start nodes exist in every variant, so all rows share one task set.

Start: F2P test present at base (33 tasks, 303 gold definitions).

| variant | k | pooled | macro [95% CI] | mean nodes in ball | gold reached per 1,000 nodes |
|---|---|---|---|---|---|
| provided | 1 | 3.0% | 14.1% [3.2%, 26.3%] | 15 | 18.56 |
| provided | 2 | 8.6% | 25.0% [11.5%, 39.5%] | 113 | 6.95 |
| provided | 3 | 15.2% | 31.3% [17.6%, 46.1%] | 201 | 6.94 |
| provided | 4 | 28.1% | 41.8% [26.8%, 57.2%] | 483 | 5.33 |
| callproof calls | 1 | 4.3% | 17.6% [6.5%, 30.7%] | 20 | 19.79 |
| callproof calls | 2 | 8.3% | 22.1% [9.9%, 36.2%] | 197 | 3.85 |
| callproof calls | 3 | 25.4% | 39.4% [24.8%, 54.7%] | 373 | 6.25 |
| callproof calls | 4 | 41.3% | 47.9% [32.6%, 63.5%] | 632 | 5.99 |
| callproof calls+contains+inherits | 1 | 4.3% | 17.6% [6.5%, 30.9%] | 22 | 17.74 |
| callproof calls+contains+inherits | 2 | 13.5% | 30.3% [16.4%, 45.4%] | 235 | 5.29 |
| callproof calls+contains+inherits | 3 | 37.3% | 49.0% [34.0%, 63.8%] | 1005 | 3.41 |
| callproof calls+contains+inherits | 4 | 56.1% | 58.8% [42.4%, 74.2%] | 2001 | 2.57 |
| callproof all edges | 1 | 4.3% | 17.6% [6.5%, 31.3%] | 23 | 17.26 |
| callproof all edges | 2 | 22.1% | 34.7% [20.7%, 50.0%] | 343 | 5.92 |
| callproof all edges | 3 | 47.9% | 55.7% [40.1%, 71.2%] | 1367 | 3.21 |
| callproof all edges | 4 | 78.2% | 66.3% [50.8%, 80.9%] | 2435 | 2.95 |
| callproof calls, directed | 1 | 4.3% | 17.6% [6.3%, 30.7%] | 20 | 19.91 |
| callproof calls, directed | 2 | 5.6% | 21.3% [9.1%, 36.0%] | 27 | 18.87 |
| callproof calls, directed | 3 | 7.9% | 25.4% [12.5%, 39.9%] | 36 | 20.27 |
| callproof calls, directed | 4 | 9.6% | 27.0% [13.8%, 41.2%] | 43 | 20.21 |

Start: executed test code present at base (38 tasks, 308 gold definitions).

| variant | k | pooled | macro [95% CI] | mean nodes in ball | gold reached per 1,000 nodes |
|---|---|---|---|---|---|
| provided | 1 | 2.9% | 12.2% [2.8%, 23.4%] | 17 | 14.06 |
| provided | 2 | 8.4% | 21.7% [10.1%, 35.1%] | 168 | 4.08 |
| provided | 3 | 15.9% | 35.1% [21.6%, 49.3%] | 259 | 4.99 |
| provided | 4 | 29.2% | 49.4% [34.9%, 63.8%] | 617 | 3.84 |
| callproof calls | 1 | 4.2% | 15.3% [5.6%, 26.8%] | 23 | 14.61 |
| callproof calls | 2 | 8.4% | 19.6% [8.4%, 32.4%] | 250 | 2.74 |
| callproof calls | 3 | 26.0% | 40.1% [26.4%, 54.4%] | 486 | 4.33 |
| callproof calls | 4 | 42.5% | 52.8% [38.3%, 67.6%] | 796 | 4.33 |
| callproof calls+contains+inherits | 1 | 4.2% | 15.3% [5.6%, 26.9%] | 25 | 13.47 |
| callproof calls+contains+inherits | 2 | 13.3% | 26.5% [14.3%, 40.0%] | 290 | 3.72 |
| callproof calls+contains+inherits | 3 | 37.3% | 50.9% [36.4%, 64.7%] | 1070 | 2.83 |
| callproof calls+contains+inherits | 4 | 57.8% | 69.7% [56.0%, 83.0%] | 2136 | 2.19 |
| callproof all edges | 1 | 4.2% | 15.3% [5.4%, 26.9%] | 26 | 13.03 |
| callproof all edges | 2 | 21.8% | 30.4% [17.7%, 43.7%] | 525 | 3.36 |
| callproof all edges | 3 | 48.1% | 56.7% [42.1%, 71.1%] | 1642 | 2.37 |
| callproof all edges | 4 | 79.9% | 74.6% [61.4%, 86.7%] | 2566 | 2.52 |
| callproof calls, directed | 1 | 4.2% | 15.3% [5.5%, 27.0%] | 20 | 16.73 |
| callproof calls, directed | 2 | 5.8% | 18.8% [8.1%, 31.6%] | 31 | 15.35 |
| callproof calls, directed | 3 | 8.1% | 22.4% [10.5%, 35.4%] | 41 | 16.10 |
| callproof calls, directed | 4 | 9.4% | 23.5% [11.9%, 36.6%] | 50 | 15.21 |

## 3. Model-free walker (119 tasks with a modified gold function)

Start: top-3 BM25 function hits for the issue text (tests and docs_src excluded; one seed list from the provided
graph's index, shared by all variants). Expansion: breadth-first through `CodeGraph.get_code_neighbors` (merged,
alphabetical, 50-neighbour and 5,000-character caps, no edge types), one call per expanded node. Hit: any gold
modified definition among the seed or listed nodes. Budget 0 is the seeds alone. Listed = distinct nodes seen.

| variant | budget | tasks hit [95% CI] | mean listed | median listed | tasks whose gold has a node |
|---|---|---|---|---|---|
| provided | 0 | 35.3% [26.9%, 43.7%] | 3 | 3 | 96.6% |
| provided | 5 | 45.4% [36.1%, 54.6%] | 36 | 29 | 96.6% |
| provided | 10 | 53.8% [45.4%, 62.2%] | 76 | 75 | 96.6% |
| provided | 20 | 54.6% [45.4%, 63.9%] | 107 | 84 | 96.6% |
| callproof all edges | 0 | 35.3% [26.9%, 43.7%] | 3 | 3 | 100.0% |
| callproof all edges | 5 | 49.6% [40.3%, 58.8%] | 76 | 81 | 100.0% |
| callproof all edges | 10 | 53.8% [45.4%, 63.0%] | 118 | 123 | 100.0% |
| callproof all edges | 20 | 58.0% [48.7%, 66.4%] | 185 | 175 | 100.0% |
| callproof calls+contains+inherits | 0 | 35.3% [26.9%, 43.7%] | 3 | 3 | 100.0% |
| callproof calls+contains+inherits | 5 | 48.7% [39.5%, 58.0%] | 61 | 61 | 100.0% |
| callproof calls+contains+inherits | 10 | 53.8% [44.5%, 63.0%] | 103 | 108 | 100.0% |
| callproof calls+contains+inherits | 20 | 60.5% [51.3%, 68.9%] | 160 | 140 | 100.0% |
