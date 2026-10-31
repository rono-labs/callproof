"""Build per-task ground truth: gold functions, fail-to-pass tests and dynamic call edges."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from callproof import sandbox  # noqa: E402
from callproof.patch import extract_gold_functions, parse_diff  # noqa: E402

DATA = Path(".cache/compdata/unz/root/.cache/kagglehub/competitions/gemma-4-developer-agent")
SUMMARY_FIELDS = ["instance_id", "repo", "status", "n_f2p", "n_gold_funcs", "n_added", "n_dyn_edges", "runtime_s"]


class TaskFailure(Exception):
    def __init__(self, status: str, note: str) -> None:
        super().__init__(note)
        self.status = status


def git(root: Path, *args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=root, input=stdin, capture_output=True, text=True, errors="replace")


def apply_patch(root: Path, patch: str, reverse: bool = False) -> bool:
    patch = patch if patch.endswith("\n") else patch + "\n"
    flags = ["apply", "--whitespace=nowarn", "--recount"] + (["-R"] if reverse else [])
    if git(root, *flags, "-", stdin=patch).returncode == 0:
        return True
    fallback = ["patch", "-p1", "-s", "-f", "--no-backup-if-mismatch"] + (["-R"] if reverse else [])
    return subprocess.run(fallback, cwd=root, input=patch, capture_output=True, text=True).returncode == 0


def test_targets(test_patch: str) -> list[str]:
    paths = []
    for edit in parse_diff(test_patch):
        name = Path(edit.path).name
        if edit.new_path and name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")):
            paths.append(edit.new_path)
    return sorted(set(paths))


class Runner:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.wheels_dir = args.data / "wheels"
        self.index = sandbox.index_wheels(self.wheels_dir)
        self.setup = sandbox.load_scorer_setup(args.data / "sandbox" / "setup.py")
        self.log_dir = args.logs
        self.host = sandbox.host_distributions(args.host_packages)

    def outcomes(self, venv, root, env, targets, tag, scratch, trace=False, timeout=None) -> tuple[dict[str, str], dict | None, sandbox.PytestRun]:
        out = scratch / f"{tag}.outcomes.json"
        edges = scratch / f"{tag}.edges.json"
        extra = ["-p", "callproof.outcomes", "--outcomes-out", str(out)]
        if trace:
            extra += ["-p", "callproof.trace", "--callproof-root", str(root), "--callproof-out", str(edges)]
        run = sandbox.run_pytest(venv, root, targets, env, extra, timeout or self.args.timeout)
        outcomes = json.loads(out.read_text()) if out.exists() else {}
        return outcomes, (json.loads(edges.read_text()) if trace and edges.exists() else None), run

    def isolate_base(self, venv, root, env, nodes: list[str], scratch: Path) -> dict[str, str]:
        """Run each node alone at base so one hanging test cannot hide the outcomes of the others."""
        results: dict[str, str] = {}
        for index, node in enumerate(nodes):
            outcome, _, run = self.outcomes(venv, root, env, [node], f"isolated{index}", scratch, timeout=self.args.node_timeout)
            results[node] = "timeout" if run.timed_out else outcome.get(node, "missing")
        return results

    def run_task(self, task: dict) -> dict:
        started = time.time()
        iid = task["instance_id"]
        record = {
            "instance_id": iid, "repo": task["repo"], "status": "error", "f2p_tests": [], "base_failed": [],
            "gold_passed": [], "gold_functions": [], "non_py_files": [], "dyn_edges": [], "dyn_edges_base": [],
            "executed": [], "executed_base": [], "notes": [],
        }
        work = self.args.work / iid
        try:
            self.execute(task, work, record)
        except TaskFailure as failure:
            record["status"] = failure.status
            record["notes"].append(str(failure))
        except Exception as exc:
            record["status"] = "error"
            record["notes"].append(f"{type(exc).__name__}: {exc}")
            (self.log_dir / f"{iid}.trace.log").write_text(traceback.format_exc())
        finally:
            if not self.args.keep:
                shutil.rmtree(work, ignore_errors=True)
        record["runtime_s"] = round(time.time() - started, 1)
        (self.args.out / f"{iid}.json").write_text(json.dumps(record))
        return record

    def execute(self, task: dict, work: Path, record: dict) -> None:
        iid = task["instance_id"]
        shutil.rmtree(work, ignore_errors=True)
        root = work / "repo"
        root.mkdir(parents=True)
        scratch = work / "scratch"
        scratch.mkdir()
        subprocess.run(["tar", "-xzf", str(self.args.data / "snapshots" / f"{iid}.tgz"), "-C", str(root)], check=True)
        if git(root, "rev-parse", "HEAD").stdout.strip() != task["base_commit"]:
            git(root, "checkout", "-q", "-f", task["base_commit"])

        def read_base(path: str) -> str | None:
            shown = git(root, "show", f"HEAD:{path}")
            return shown.stdout if shown.returncode == 0 else None

        def read_head(path: str) -> str | None:
            file = root / path
            return file.read_text("utf-8", "replace") if file.is_file() else None

        wheels, missing = sandbox.resolve(root, self.wheels_dir, self.index, task["repo"].split("/")[1])

        def build_env(wanted: list[str]) -> Path:
            extras = sandbox.supplements(wanted, wheels, self.host)
            shimmed = sandbox.stand_ins(wanted, extras)
            venv = sandbox.ensure_env(wheels, extras, shimmed, self.wheels_dir, self.args.data / "docker", self.args.envs)
            record["notes"].append(
                f"env {venv.name}; host packages: {','.join(d.name for d in extras) or 'none'}; "
                f"stand-ins: {','.join(shimmed) or 'none'}"
            )
            return venv

        venv = build_env(missing)

        self.setup.setup_git_exclude(root)
        if not apply_patch(root, task["test_patch"]):
            raise TaskFailure("test_patch_failed", "test_patch did not apply")
        targets = test_targets(task["test_patch"])
        if not targets:
            raise TaskFailure("no_test_files", "test_patch touches no test_*.py files")
        self.setup.configure_workspace_pytest_ini(root)
        self.setup.configure_workspace_conftest(root)
        env = sandbox.pytest_env(root, PROJECT, scratch)

        base_out, _, base_run = self.outcomes(venv, root, env, targets, "base", scratch)
        wanted_more = sandbox.providers(base_run.missing_modules, self.host)
        if wanted_more:
            venv = build_env([*missing, *wanted_more])
            base_out, _, base_run = self.outcomes(venv, root, env, targets, "base", scratch)
        record["base_failed"] = sorted(n for n, o in base_out.items() if o != "passed")

        funcs_base_reader = read_base
        if not apply_patch(root, task["patch"]):
            raise TaskFailure("gold_patch_failed", "gold patch did not apply")
        gold_funcs, non_py = extract_gold_functions(task["patch"], funcs_base_reader, read_head)
        record["gold_functions"] = [
            {k: v for k, v in asdict(f).items()} for f in gold_funcs
        ]
        record["non_py_files"] = non_py

        gold_out, _, gold_run = self.outcomes(venv, root, env, targets, "gold", scratch)
        record["gold_passed"] = sorted(n for n, o in gold_out.items() if o == "passed")
        if gold_run.timed_out:
            record["notes"].append("gold run timed out")
        if base_run.timed_out:
            if not apply_patch(root, task["patch"], reverse=True):
                raise TaskFailure("revert_failed", "could not revert gold patch to isolate base tests")
            base_out = self.isolate_base(venv, root, env, record["gold_passed"], scratch)
            if not apply_patch(root, task["patch"]):
                raise TaskFailure("revert_failed", "could not re-apply gold patch")
            record["base_failed"] = sorted(n for n, o in base_out.items() if o != "passed")
            record["notes"].append("base run timed out; base outcomes recomputed per test")
        f2p = [n for n in record["gold_passed"] if base_out.get(n) != "passed"]
        record["f2p_tests"] = f2p
        if not f2p:
            record["status"] = "no_f2p"
            record["notes"].append(f"gold run: {len(gold_out)} nodes, rc={gold_run.returncode}; missing modules: {','.join(gold_run.missing_modules) or 'none'}; tail: {gold_run.tail[-300:]}")
            return
        _, traced, trace_run = self.outcomes(venv, root, env, f2p, "gold_trace", scratch, trace=True)
        if traced is None:
            raise TaskFailure("trace_failed", f"tracer produced no output: {trace_run.tail[-300:]}")
        record["dyn_edges"], record["executed"] = traced["edges"], traced["executed"]
        record["status"] = "ok"

        if base_run.timed_out:
            record["notes"].append("base run timed out; base trace skipped")
        elif apply_patch(root, task["patch"], reverse=True):
            _, traced_base, _ = self.outcomes(venv, root, env, f2p, "base_trace", scratch, trace=True)
            if traced_base:
                record["dyn_edges_base"], record["executed_base"] = traced_base["edges"], traced_base["executed"]
        else:
            record["notes"].append("could not revert gold patch; base trace skipped")


def summarize(out: Path) -> None:
    rows = []
    for file in sorted(out.glob("*.json")):
        r = json.loads(file.read_text())
        funcs = r["gold_functions"]
        rows.append({
            "instance_id": r["instance_id"], "repo": r["repo"], "status": r["status"], "n_f2p": len(r["f2p_tests"]),
            "n_gold_funcs": len(funcs), "n_added": sum(f["kind"] == "added" for f in funcs),
            "n_dyn_edges": len(r["dyn_edges"]), "runtime_s": r.get("runtime_s", ""),
        })
    with open(out / "summary.csv", "w", newline="") as handle:
        writer = csv.DictWriter(handle, SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all", help="'all', a comma list of instance ids, or a file of ids")
    parser.add_argument("--jobs", type=int, default=12)
    parser.add_argument("--out", type=Path, default=Path("results/groundtruth"))
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--work", type=Path, default=Path(".cache/work/groundtruth"))
    parser.add_argument("--envs", type=Path, default=Path(".cache/work/envs"))
    parser.add_argument("--logs", type=Path, default=Path("logs/groundtruth"))
    parser.add_argument("--host-packages", type=Path, nargs="*", default=[Path(subprocess.check_output(["uv", "cache", "dir"], text=True).strip()) / "archive-v0"])
    parser.add_argument("--timeout", type=int, default=900, help="seconds per pytest invocation")
    parser.add_argument("--node-timeout", type=int, default=20, help="seconds per isolated test when re-running a hung base")
    parser.add_argument("--keep", action="store_true", help="keep extracted checkouts")
    parser.add_argument("--redo", action="store_true", help="recompute tasks that already have output")
    args = parser.parse_args()
    for name in ("out", "data", "work", "envs", "logs"):
        setattr(args, name, getattr(args, name).resolve())
    args.host_packages = [p.resolve() for p in args.host_packages]
    for directory in (args.out, args.work, args.envs, args.logs):
        directory.mkdir(parents=True, exist_ok=True)
    tasks = [json.loads(line) for line in (args.data / "tasks.jsonl").read_text().splitlines() if line.strip()]
    if args.tasks != "all":
        spec = Path(args.tasks)
        wanted = set(spec.read_text().split()) if spec.is_file() else set(args.tasks.split(","))
        tasks = [t for t in tasks if t["instance_id"] in wanted]
    if not args.redo:
        tasks = [t for t in tasks if not (args.out / f"{t['instance_id']}.json").exists()]
    runner = Runner(args)

    def work(task: dict) -> None:
        record = runner.run_task(task)
        print(f"{record['instance_id']} {record['status']} f2p={len(record['f2p_tests'])} "
              f"edges={len(record['dyn_edges'])} {record['runtime_s']}s", flush=True)

    with ThreadPoolExecutor(args.jobs) as pool:
        list(pool.map(work, tasks))
    summarize(args.out)


if __name__ == "__main__":
    main()
