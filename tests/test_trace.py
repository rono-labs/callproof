import json
import subprocess
import sys
import textwrap

from callproof.trace import collapse_qualname, is_test_path


def test_collapse_nested_and_synthetic():
    assert collapse_qualname("f.<locals>.g") == ("f", True)
    assert collapse_qualname("C.m.<locals>.<lambda>") == ("C.m", True)
    assert collapse_qualname("<listcomp>") == ("<module>", True)
    assert collapse_qualname("C.m") == ("C.m", False)


def test_is_test_path():
    assert is_test_path("tests/helpers.py")
    assert is_test_path("pkg/test_x.py")
    assert not is_test_path("pkg/core.py")


def test_plugin_records_edges(tmp_path):
    (tmp_path / "lib.py").write_text(textwrap.dedent("""
        def leaf():
            return [x for x in range(2)]

        def outer():
            def inner():
                return leaf()
            return inner()
    """))
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_lib.py").write_text("import lib\n\ndef test_it():\n    lib.outer()\n")
    out = tmp_path / "edges.json"
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "callproof.trace", "--callproof-root", str(tmp_path),
         "--callproof-out", str(out), "-p", "no:cacheprovider", str(tests)],
        cwd=tmp_path, check=True, capture_output=True,
        env={"PYTHONPATH": f"{tmp_path}:{__import__('callproof').__path__[0]}/.."},
    )
    data = json.loads(out.read_text())
    edges = {tuple(e[:4]) for e in data["edges"]}
    assert ("tests/test_lib.py", "test_it", "lib.py", "outer") in edges
    assert ("lib.py", "outer", "lib.py", "leaf") in edges
    assert ("lib.py", "outer", "lib.py", "outer") not in edges
    assert ["lib.py", "outer", True] in data["executed"]
