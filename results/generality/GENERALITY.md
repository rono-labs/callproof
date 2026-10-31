# Generality: edge recall on six repositories outside the competition

## Repositories

| repo | tag | commit | tests traced | traced edges (non-test) | callproof graph (nodes/edges) | baseline graph (nodes/edges) | build s (callproof / baseline) | pytest exit |
|---|---|---|---|---|---|---|---|---|
| click | 8.5.0 | `8b19813f2bfc` | 300 | 236 | 1975/6870 | 1573/2770 | 1.25 / 0.64 | 0 |
| attrs | 26.1.0 | `7bfc49e9b22d` | 299 | 162 | 1796/5184 | 1365/2786 | 1.02 / 0.60 | 0 |
| more-itertools | v11.2.1 | `4aeb4338ae14` | 300 | 86 | 1343/2669 | 1322/2028 | 1.15 / 0.48 | 0 |
| packaging | 26.3 | `929fd4b1410a` | 300 | 372 | 2338/8654 | 2258/9036 | 2.15 / 1.26 | 0 |
| itsdangerous | 2.2.0 | `096c8d42545d` | 297 | 87 | 151/466 | 136/414 | 0.09 / 0.06 | 0 |
| h11 | v0.16.0 | `1c5b07581f05` | 78 | 143 | 295/1716 | 251/1578 | 0.38 / 0.28 | 0 |

## Pooled recall (95% bootstrap CI over tests, resampled within each repository)

| call kind | traced edges | baseline: both ends nodes | baseline: linked either way | baseline: correct direction | callproof: both ends nodes | callproof: linked either way | callproof: correct direction |
|---|---|---|---|---|---|---|---|
| all | 1086 | 99.9% (1085/1086) | 31.1% (338/1086) [29.2, 32.4] | 31.1% (338/1086) [29.2, 32.4] | 100.0% (1086/1086) | 84.5% (918/1086) [82.5, 85.0] | 84.2% (914/1086) [82.1, 84.5] |
| self_method | 238 | 100.0% (238/238) | 0.0% (0/238) [0.0, 0.0] | 0.0% (0/238) [0.0, 0.0] | 100.0% (238/238) | 94.5% (225/238) [93.7, 95.4] | 94.5% (225/238) [93.7, 95.4] |
| constructor | 127 | 100.0% (127/127) | 0.0% (0/127) [0.0, 0.0] | 0.0% (0/127) [0.0, 0.0] | 100.0% (127/127) | 89.8% (114/127) [87.0, 91.9] | 89.8% (114/127) [87.0, 91.9] |
| module_function | 384 | 100.0% (384/384) | 87.5% (336/384) [85.6, 88.6] | 87.5% (336/384) [85.6, 88.6] | 100.0% (384/384) | 89.3% (343/384) [87.5, 90.4] | 89.1% (342/384) [87.2, 90.0] |
| cross_module | 456 | 100.0% (456/456) | 26.1% (119/456) [20.2, 28.1] | 26.1% (119/456) [20.2, 28.1] | 100.0% (456/456) | 78.3% (357/456) [75.7, 79.0] | 77.6% (354/456) [75.0, 78.2] |
| async | 0 | n/a | n/a | n/a | n/a | n/a | n/a |
| via_test | 1838 | 99.8% (1834/1838) | 9.8% (180/1838) [8.5, 10.8] | 9.8% (180/1838) [8.5, 10.8] | 100.0% (1838/1838) | 77.2% (1419/1838) [76.2, 79.6] | 74.0% (1361/1838) [73.0, 76.3] |
| nested | 112 | 99.1% (111/112) | 22.3% (25/112) [18.3, 25.3] | 22.3% (25/112) [18.3, 25.3] | 100.0% (112/112) | 64.3% (72/112) [59.3, 67.5] | 60.7% (68/112) [55.4, 64.3] |

### click

| call kind | traced edges | baseline either / forward | callproof either / forward |
|---|---|---|---|
| all | 236 | 29.2% (69/236) / 29.2% (69/236) | 89.4% (211/236) / 88.6% (209/236) |
| self_method | 76 | 0.0% (0/76) / 0.0% (0/76) | 98.7% (75/76) / 98.7% (75/76) |
| constructor | 22 | 0.0% (0/22) / 0.0% (0/22) | 90.9% (20/22) / 90.9% (20/22) |
| module_function | 77 | 88.3% (68/77) / 88.3% (68/77) | 90.9% (70/77) / 90.9% (70/77) |
| cross_module | 61 | 49.2% (30/61) / 49.2% (30/61) | 78.7% (48/61) / 77.0% (47/61) |
| async | 0 | n/a / n/a | n/a / n/a |
| via_test | 130 | 1.5% (2/130) / 1.5% (2/130) | 63.1% (82/130) / 62.3% (81/130) |
| nested | 30 | 30.0% (9/30) / 30.0% (9/30) | 53.3% (16/30) / 46.7% (14/30) |

### attrs

| call kind | traced edges | baseline either / forward | callproof either / forward |
|---|---|---|---|
| all | 162 | 40.1% (65/162) / 40.1% (65/162) | 78.4% (127/162) / 78.4% (127/162) |
| self_method | 23 | 0.0% (0/23) / 0.0% (0/23) | 73.9% (17/23) / 73.9% (17/23) |
| constructor | 18 | 0.0% (0/18) / 0.0% (0/18) | 88.9% (16/18) / 88.9% (16/18) |
| module_function | 84 | 77.4% (65/84) / 77.4% (65/84) | 79.8% (67/84) / 79.8% (67/84) |
| cross_module | 38 | 31.6% (12/38) / 31.6% (12/38) | 63.2% (24/38) / 63.2% (24/38) |
| async | 0 | n/a / n/a | n/a / n/a |
| via_test | 535 | 20.6% (110/535) / 20.6% (110/535) | 78.9% (422/535) / 78.1% (418/535) |
| nested | 52 | 19.2% (10/52) / 19.2% (10/52) | 65.4% (34/52) / 65.4% (34/52) |

### more-itertools

| call kind | traced edges | baseline either / forward | callproof either / forward |
|---|---|---|---|
| all | 86 | 55.8% (48/86) / 55.8% (48/86) | 77.9% (67/86) / 76.7% (66/86) |
| self_method | 14 | 0.0% (0/14) / 0.0% (0/14) | 85.7% (12/14) / 85.7% (12/14) |
| constructor | 7 | 0.0% (0/7) / 0.0% (0/7) | 42.9% (3/7) / 42.9% (3/7) |
| module_function | 58 | 82.8% (48/58) / 82.8% (48/58) | 86.2% (50/58) / 84.5% (49/58) |
| cross_module | 24 | 54.2% (13/24) / 54.2% (13/24) | 62.5% (15/24) / 58.3% (14/24) |
| async | 0 | n/a / n/a | n/a / n/a |
| via_test | 560 | 0.7% (4/560) / 0.7% (4/560) | 81.8% (458/560) / 72.7% (407/560) |
| nested | 11 | 45.5% (5/11) / 45.5% (5/11) | 54.5% (6/11) / 45.5% (5/11) |

### packaging

| call kind | traced edges | baseline either / forward | callproof either / forward |
|---|---|---|---|
| all | 372 | 26.6% (99/372) / 26.6% (99/372) | 84.4% (314/372) / 84.1% (313/372) |
| self_method | 67 | 0.0% (0/67) / 0.0% (0/67) | 98.5% (66/67) / 98.5% (66/67) |
| constructor | 28 | 0.0% (0/28) / 0.0% (0/28) | 92.9% (26/28) / 92.9% (26/28) |
| module_function | 102 | 96.1% (98/102) / 96.1% (98/102) | 97.1% (99/102) / 97.1% (99/102) |
| cross_module | 191 | 12.0% (23/191) / 12.0% (23/191) | 81.7% (156/191) / 81.2% (155/191) |
| async | 0 | n/a / n/a | n/a / n/a |
| via_test | 99 | 5.1% (5/99) / 5.1% (5/99) | 71.7% (71/99) / 71.7% (71/99) |
| nested | 19 | 5.3% (1/19) / 5.3% (1/19) | 84.2% (16/19) / 78.9% (15/19) |

### itsdangerous

| call kind | traced edges | baseline either / forward | callproof either / forward |
|---|---|---|---|
| all | 87 | 31.0% (27/87) / 31.0% (27/87) | 92.0% (80/87) / 92.0% (80/87) |
| self_method | 28 | 0.0% (0/28) / 0.0% (0/28) | 96.4% (27/28) / 96.4% (27/28) |
| constructor | 11 | 0.0% (0/11) / 0.0% (0/11) | 100.0% (11/11) / 100.0% (11/11) |
| module_function | 28 | 96.4% (27/28) / 96.4% (27/28) | 96.4% (27/28) / 96.4% (27/28) |
| cross_module | 51 | 45.1% (23/51) / 45.1% (23/51) | 88.2% (45/51) / 88.2% (45/51) |
| async | 0 | n/a / n/a | n/a / n/a |
| via_test | 139 | 7.2% (10/139) / 7.2% (10/139) | 42.4% (59/139) / 41.7% (58/139) |
| nested | 0 | n/a / n/a | n/a / n/a |

### h11

| call kind | traced edges | baseline either / forward | callproof either / forward |
|---|---|---|---|
| all | 143 | 21.0% (30/143) / 21.0% (30/143) | 83.2% (119/143) / 83.2% (119/143) |
| self_method | 30 | 0.0% (0/30) / 0.0% (0/30) | 93.3% (28/30) / 93.3% (28/30) |
| constructor | 41 | 0.0% (0/41) / 0.0% (0/41) | 92.7% (38/41) / 92.7% (38/41) |
| module_function | 35 | 85.7% (30/35) / 85.7% (30/35) | 85.7% (30/35) / 85.7% (30/35) |
| cross_module | 91 | 19.8% (18/91) / 19.8% (18/91) | 75.8% (69/91) / 75.8% (69/91) |
| async | 0 | n/a / n/a | n/a / n/a |
| via_test | 375 | 13.1% (49/375) / 13.1% (49/375) | 87.2% (327/375) / 86.9% (326/375) |
| nested | 0 | n/a / n/a | n/a / n/a |

## Method notes

- Traced edges: `callproof.trace` under `pytest` on Python 3.13 (call phase only), unique caller/callee pairs over all sampled tests, nested defs folded into their enclosing definition. `all` excludes edges with a test-code endpoint (`via_test`, reported as its own row); self-edges are dropped. Categories overlap.
- Suites larger than 300 tests are sampled at evenly spaced positions in collection order.
- Metrics are those of `callproof.audit` (both ends are nodes; linked in either direction; linked in the traced direction), computed per edge so tests can be resampled.
- Baseline: a reimplementation (`project/scripts/provided_style.py`) of the provided graphs' documented rules, not their actual builder: harness node ids, undirected `calls` edges for containment, inheritance, instantiation and plain-name calls, no async definitions, no module nodes, no `self.method` or attribute calls. Plain names resolve by same module, then `from` imports, then a unique repository-wide name.
- Bootstrap: 1000 resamples of tests within each repository; each resample's edge set is the union of its tests' edges; the interval is the basic (reflected) bootstrap interval around the full-sample recall, which corrects the downward bias of resampled unions. Call-kind categories exclude `via_test` edges, except the `via_test` row itself.
