import difflib

from callproof.patch import extract_gold_functions, parse_diff

BASE = '''import os

CONST = 1


class A:
    x = 1

    def f(self):
        return 1

    def g(self):
        def inner():
            return 2

        return inner()


def top():
    return 3
'''

HEAD = '''import os

CONST = 2


class A:
    x = 1

    def f(self):
        return 10

    def g(self):
        def inner():
            return 20

        return inner()

    def h(self):
        return 4


def top():
    return 3


def fresh():
    return 5
'''

DIFF = "".join(
    difflib.unified_diff(BASE.splitlines(True), HEAD.splitlines(True), "a/pkg/mod.py", "b/pkg/mod.py", n=1)
) + "--- a/README.md\n+++ b/README.md\n@@ -1 +1,2 @@\n hi\n+there\n"


def run(diff=DIFF, base=BASE, head=HEAD):
    funcs, non_py = extract_gold_functions(diff, lambda p: base, lambda p: head)
    return {(f.qualname, f.kind) for f in funcs}, non_py


def test_parse_diff_numbers_lines():
    edits = parse_diff(DIFF)
    assert [e.path for e in edits] == ["pkg/mod.py", "README.md"]
    assert edits[0].removed == [3, 10, 14]
    assert [n for n, _ in edits[0].added][:3] == [3, 10, 14]


def test_modified_and_added_split():
    found, non_py = run()
    assert found == {
        ("<module>", "modified"),
        ("A.f", "modified"),
        ("A.g.<locals>.inner", "modified"),
        ("A.h", "added"),
        ("fresh", "added"),
    }
    assert non_py == ["README.md"]


def test_collapsed_name_for_nested():
    funcs, _ = extract_gold_functions(DIFF, lambda p: BASE, lambda p: HEAD)
    nested = next(f for f in funcs if f.qualname.endswith("inner"))
    assert nested.collapsed_qualname == "A.g"


def test_new_file_is_all_added():
    diff = "--- /dev/null\n+++ b/new.py\n@@ -0,0 +1,3 @@\n+def a():\n+    return 1\n+X = 1\n"
    funcs, _ = extract_gold_functions(diff, lambda p: None, lambda p: "def a():\n    return 1\nX = 1\n")
    assert {(f.qualname, f.kind) for f in funcs} == {("a", "added"), ("<module>", "added")}


def test_deleted_function_is_modified_at_base():
    diff = "--- a/m.py\n+++ b/m.py\n@@ -1,5 +1,2 @@\n def keep():\n     return 1\n-\n-def gone():\n-    return 2\n"
    base = "def keep():\n    return 1\n\ndef gone():\n    return 2\n"
    funcs, _ = extract_gold_functions(diff, lambda p: base, lambda p: "def keep():\n    return 1\n")
    assert {(f.qualname, f.kind) for f in funcs} == {("gone", "modified")}


def test_comment_and_blank_lines_ignored():
    diff = "--- a/m.py\n+++ b/m.py\n@@ -1,2 +1,4 @@\n def f():\n     return 1\n+\n+# note\n"
    funcs, _ = extract_gold_functions(diff, lambda p: "def f():\n    return 1\n", lambda p: "def f():\n    return 1\n\n# note\n")
    assert funcs == []
