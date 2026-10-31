"""Reading and writing code graphs in the competition's JSON format.

The format is NetworkX node-link data of a directed multigraph::

    {"directed": true, "multigraph": true, "graph": {"repo_name": "<repo>_<commit>"},
     "nodes": [{"id": ..., "name": ..., "text": ...}, ...],
     "edges": [{"source": ..., "target": ..., "type": "calls", "key": 0}, ...]}

The harness reads ``id``, ``name`` and ``text`` from nodes and ``type`` from edges and ignores any
other attribute, so graphs written here carry extra node attributes (``kind``, ``path``,
``qualname``, ``lineno``, ``end_lineno``) and extra edge attributes (``resolution``) without
breaking the harness tools.
"""

from __future__ import annotations

import json
from pathlib import Path

import networkx as nx

NODE_FIELDS = ("id", "name", "text")
EDGE_FIELDS = ("source", "target", "type", "key")


def load_graph(path: Path) -> nx.MultiDiGraph:
    with open(path, encoding="utf-8") as handle:
        data = json.load(handle)
    edges = "edges" if "edges" in data else "links"
    return nx.node_link_graph(data, directed=True, multigraph=True, edges=edges)


def to_node_link(graph: nx.MultiDiGraph) -> dict:
    data = nx.node_link_data(graph, edges="edges")
    for node in data["nodes"]:
        node.setdefault("name", node["id"])
        node.setdefault("text", "")
    return data


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_node_link(graph), handle)
