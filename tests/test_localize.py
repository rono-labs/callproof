import networkx as nx

from callproof.agent import parse_answer
from callproof.localize_eval import score
from callproof.retrieve import BM25, tokenize
from callproof.tools import Repo, ToolBox, resolve_node_name


def test_resolve_node_name_precedence():
    graph = nx.MultiDiGraph()
    graph.add_nodes_from(["pkg.mod.Class.method", "pkg.other.method", "pkg.mod.Class"])
    assert resolve_node_name("pkg.mod.Class", graph) == "pkg.mod.Class"
    assert resolve_node_name("Class.method", graph) == "pkg.mod.Class.method"
    assert resolve_node_name("repo;method", graph) == "pkg.other.method"
    assert resolve_node_name("CLASS.METHOD", graph) == "pkg.mod.Class.method"
    assert resolve_node_name("lass.meth", graph) == "pkg.mod.Class.method"
    assert resolve_node_name("missing", graph) is None


def test_parse_answer_json_and_fallback():
    assert parse_answer('Sure: ["a.py::f", "b.py::C.m"]') == ["a.py::f", "b.py::C.m"]
    assert parse_answer("try a/b.py::C.m then c.py::g") == ["a/b.py::C.m", "c.py::g"]


def test_score_levels():
    gold = [{"path": "a.py", "qualname": "C.m", "kind": "modified"}, {"path": "n.py", "qualname": "new", "kind": "added"}]
    flags = score(["b.py::x", "a.py::C.m"], gold)
    assert flags["function"] == {1: False, 5: True}
    assert flags["file"] == {1: False, 5: True}


def test_bm25_prefers_matching_document():
    assert tokenize("get_RequestHandler") == ["get", "request", "handler"]
    ranking = BM25([tokenize("parse header value"), tokenize("render template page")]).scores(tokenize("header parse"))
    assert ranking[0] > ranking[1]


def test_toolbox_flags_invalid_calls(tmp_path):
    (tmp_path / "m.py").write_text("def f():\n    pass\n")
    box = ToolBox(Repo(tmp_path), None, "search")
    assert box.call("read_file", '{"filepath": "m.py"}')[1]
    assert not box.call("read_file", "{}")[1]
    assert not box.call("get_code_neighbors", {"node": "f"})[1]
    assert Repo(tmp_path).locate("m.f") == ("m.py", "f")
