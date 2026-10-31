import json
import textwrap
from pathlib import Path

import networkx as nx
import pytest

from callproof.build import CALLS, CONTAINS, DECORATED_BY, IMPORTS, INHERITS, REFERENCES, UNIQUE_NAME, build_graph
from callproof.graphio import load_graph, save_graph
from callproof.naming import RepoIndex

FILES = {
    "src/shop/__init__.py": """
        from .models import Item as Item
        from .service import Service
    """,
    "src/shop/models.py": """
        from typing import Iterator


        class Base:
            def __init__(self, name):
                self.name = name

            def describe(self):
                return self.label()

            def label(self):
                return self.name


        class Item(Base):
            def label(self):
                return super().label().upper()

            @property
            def size(self) -> int:
                return len(self.name)


        class Price:
            def __call__(self, item):
                return item.size


        class Basket:
            def __init__(self, items: list[Item]) -> None:
                self.items = items

            def __len__(self):
                return len(self.items)

            def __iter__(self) -> Iterator[Item]:
                return iter(self.items)

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def labels(self):
                with self:
                    return [item.label() for item in self], len(self)

            def first(self):
                return self.items[0].describe()
    """,
    "src/shop/service.py": """
        import functools

        from shop import Item
        from shop.models import Price as Pricing


        def logged(func):
            @functools.wraps(func)
            def wrapper(*args):
                return func(*args)

            return wrapper


        class Service:
            def __init__(self) -> None:
                self.pricing = Pricing()

            @logged
            def create(self, name: str) -> Item:
                item = Item(name)
                self.check(item)
                return item

            def check(self, item: Item):
                total = self.pricing(item)
                return item.describe(), total

            async def fetch(self, name):
                async def inner():
                    return self.create(name)

                return await inner()

            def plain(self):
                class Local:
                    pass

                return Local

            def after_local_class(self):
                return self.plain()


        def handler(callback=Service):
            return sorted([], key=helper)


        def helper(value):
            return value.unique_operation()


        class Other:
            def unique_operation(self):
                return 1
    """,
    "tests/__init__.py": "",
    "tests/test_service.py": """
        from shop import Service


        def test_create():
            service = Service()
            assert service.create("a").size == 1
    """,
}


@pytest.fixture(scope="module")
def repo(tmp_path_factory) -> Path:
    root = tmp_path_factory.mktemp("repo")
    for path, text in FILES.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(text).lstrip())
    return root


@pytest.fixture(scope="module")
def graph(repo) -> nx.MultiDiGraph:
    return build_graph(repo, "shop_0000")


def edges(graph: nx.MultiDiGraph, kind: str) -> set[tuple[str, str]]:
    return {(u, v) for u, v, data in graph.edges(data=True) if data["type"] == kind}


def test_nodes_cover_modules_and_all_definitions(graph):
    kinds = nx.get_node_attributes(graph, "kind")
    assert kinds["shop"] == "module"
    assert kinds["shop.models"] == "module"
    assert kinds["shop.service.Service.fetch"] == "async_method"
    assert kinds["shop.service.Service.fetch.inner"] == "async_function"
    assert kinds["shop.service.logged.wrapper"] == "function"
    assert kinds["shop.service.Service.plain.Local"] == "class"
    assert graph.nodes["shop.service.Service.create"]["text"].startswith("@logged\ndef create(")


def test_calls_resolve_through_self_super_and_mro(graph):
    calls = edges(graph, CALLS)
    assert ("shop.models.Base.describe", "shop.models.Base.label") in calls
    assert ("shop.models.Item.label", "shop.models.Base.label") in calls
    assert ("shop.service.Service.create", "shop.service.Service.check") in calls
    assert ("shop.service.Service.after_local_class", "shop.service.Service.plain") in calls
    assert ("shop.service.Service.fetch.inner", "shop.service.Service.create") in calls


def test_constructor_calls_link_class_and_inherited_init(graph):
    calls = edges(graph, CALLS)
    assert ("shop.service.Service.create", "shop.models.Item") in calls
    assert ("shop.service.Service.create", "shop.models.Base.__init__") in calls
    assert ("shop.service.Service.__init__", "shop.models.Price") in calls


def test_typed_attributes_parameters_and_properties(graph):
    calls = edges(graph, CALLS)
    assert ("shop.service.Service.check", "shop.models.Price.__call__") in calls
    assert ("shop.service.Service.check", "shop.models.Base.describe") in calls
    assert ("tests.test_service.test_create", "shop.models.Item.size") in calls


def test_reexports_and_aliases(graph):
    imports = edges(graph, IMPORTS)
    assert ("shop", "shop.models.Item") in imports
    assert ("shop.service", "shop.models.Item") in imports
    assert ("shop.service", "shop.models.Price") in imports
    assert ("tests.test_service", "shop.service.Service") in imports


def test_structure_edges(graph):
    assert ("shop.models.Item", "shop.models.Base") in edges(graph, INHERITS)
    assert ("shop.service", "shop.service.Service") in edges(graph, CONTAINS)
    assert ("shop.service.Service", "shop.service.Service.create") in edges(graph, CONTAINS)
    assert ("shop.service.Service.create", "shop.service.logged") in edges(graph, DECORATED_BY)
    assert ("shop.service.Service", "shop.service.logged") in edges(graph, CALLS)
    assert ("shop.service.handler", "shop.service.helper") in edges(graph, REFERENCES)
    assert ("shop.service", "shop.service.Service") in edges(graph, REFERENCES)


def test_protocols_iteration_and_overrides(graph):
    resolutions = {(u, v): d["resolution"] for u, v, d in graph.edges(data=True) if d["type"] == CALLS}
    labels = "shop.models.Basket.labels"
    assert resolutions[(labels, "shop.models.Basket.__enter__")] == "protocol"
    assert resolutions[(labels, "shop.models.Basket.__exit__")] == "protocol"
    assert resolutions[(labels, "shop.models.Basket.__iter__")] == "protocol"
    assert resolutions[(labels, "shop.models.Basket.__len__")] == "protocol"
    assert resolutions[(labels, "shop.models.Item.label")] == "static"
    assert resolutions[("shop.models.Base.describe", "shop.models.Item.label")] == "override"
    assert ("shop.models.Basket.first", "shop.models.Base.describe") in resolutions


def test_unique_name_fallback_is_flagged(graph):
    resolutions = {(u, v): d["resolution"] for u, v, d in graph.edges(data=True) if d["type"] == CALLS}
    assert resolutions[("shop.service.helper", "shop.service.Other.unique_operation")] == UNIQUE_NAME
    assert resolutions[("shop.models.Base.describe", "shop.models.Base.label")] == "static"


def test_external_calls_are_not_guessed(graph):
    assert not any(v.endswith(".wraps") for _, v in edges(graph, CALLS))


def test_round_trip_keeps_harness_schema(graph, tmp_path):
    path = tmp_path / "graph.json"
    save_graph(graph, path)
    data = json.loads(path.read_text())
    assert data["directed"] and data["multigraph"]
    assert data["graph"] == {"repo_name": "shop_0000"}
    assert {"id", "name", "text"} <= set(data["nodes"][0])
    assert {"source", "target", "type", "key"} <= set(data["edges"][0])
    assert nx.utils.graphs_equal(load_graph(path), graph)


def test_naming_conventions(repo):
    index = RepoIndex.from_checkout(repo)
    assert index.module_of("src/shop/models.py") == "shop.models"
    assert index.harness_id("src/shop/models.py", "Item.label") == "shop.models.Item.label"
    assert index.harness_id("src/shop/service.py", "logged.<locals>.wrapper") == "shop.service.wrapper"
    assert index.harness_id("src/shop/service.py", "Service.plain.<locals>.Local") == "shop.service.Local"
    assert index.harness_id("src/shop/service.py", "Service.after_local_class") == "shop.service.after_local_class"
    assert index.harness_id("src/shop/service.py", "Service.fetch") is None
    assert index.node_id("src/shop/service.py", "Service.fetch.<locals>.inner") == "shop.service.Service.fetch.inner"
    assert index.node_id("src/shop/__init__.py", "<module>") == "shop"
    assert index.harness_id("src/shop/new.py", "Thing.method") == "shop.new.Thing.method"
    assert [d.qualname for d in index.from_harness_id("shop.service.wrapper")] == ["logged.<locals>.wrapper"]
