import numpy as np

from callproof.linkpred import aggregate, class_prefix, module_of, split_by_commit, tied_metrics, tokenize


def test_tokenize_and_ids():
    assert tokenize("getHTTPResponse_v2") == ["get", "http", "response", "v", "2"]
    assert module_of("pkg.mod.Klass.method") == "pkg.mod"
    assert class_prefix("pkg.mod.Klass.method") == "pkg.mod.Klass"


def test_tied_metrics_expected_rank():
    exact = tied_metrics(5.0, np.array([1.0, 2.0, 9.0]))
    assert exact["mrr"] == 0.5 and exact["hits@1"] == 0.0 and exact["hits@10"] == 1.0
    tied = tied_metrics(1.0, np.array([1.0, 1.0, 1.0]))
    assert tied["hits@1"] == 0.25 and np.isclose(tied["mrr"], (1 + 1 / 2 + 1 / 3 + 1 / 4) / 4)


def test_aggregate_macro_differs_from_pooled():
    hit, miss = {"mrr": 1.0, "hits@1": 1.0, "hits@10": 1.0}, {"mrr": 0.0, "hits@1": 0.0, "hits@10": 0.0}
    rows = [("c", "a", hit), ("c", "a", hit), ("c", "b", miss)]
    out = aggregate(rows)
    assert np.isclose(out["pooled_mrr"], 2 / 3) and np.isclose(out["macro_mrr"], 0.5)


def test_commit_split_is_deterministic():
    assert split_by_commit("fastapi_abc") == split_by_commit("fastapi_abc")
