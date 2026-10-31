"""Offline Python environments built from the competition wheel mirror, and pytest runners."""

from __future__ import annotations

import fcntl
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
import re
import shutil
import subprocess
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path

from packaging.markers import Marker
from packaging.requirements import InvalidRequirement, Requirement
from packaging.specifiers import SpecifierSet
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

PYTHON = "3.13"
BASELINE = ("pytest",)
BUILD_BACKENDS = {"hatchling", "pdm-backend", "poetry-core", "flit-core", "setuptools", "wheel", "editables"}


@dataclass(frozen=True)
class Wheel:
    name: str
    version: Version
    filename: str

    def metadata_requires(self, wheels_dir: Path) -> list[Requirement]:
        with zipfile.ZipFile(wheels_dir / self.filename) as archive:
            meta = next(n for n in archive.namelist() if n.endswith(".dist-info/METADATA"))
            text = archive.read(meta).decode("utf-8", "replace")
        requirements = []
        for line in text.splitlines():
            if line.startswith("Requires-Dist:"):
                try:
                    requirements.append(Requirement(line.split(":", 1)[1].strip()))
                except InvalidRequirement:
                    continue
        return requirements


def _compatible(tags: str) -> bool:
    pyver, abi, _ = tags.split("-")
    if pyver.startswith("cp"):
        return pyver == "cp313" and abi in ("cp313", "abi3")
    return pyver.startswith(("py3", "py2.py3"))


def index_wheels(wheels_dir: Path) -> dict[str, list[Wheel]]:
    index: dict[str, list[Wheel]] = {}
    for path in wheels_dir.glob("*.whl"):
        parts = path.stem.split("-")
        if len(parts) < 5 or not _compatible("-".join(parts[-3:])):
            continue
        try:
            version = Version(parts[1])
        except InvalidVersion:
            continue
        name = canonicalize_name(parts[0])
        index.setdefault(name, []).append(Wheel(name, version, path.name))
    for wheels in index.values():
        wheels.sort(key=lambda w: w.version, reverse=True)
    return index


def _flatten_groups(groups: dict) -> list[str]:
    return [item for items in groups.values() for item in items if isinstance(item, str)]


def declared_requirements(root: Path) -> tuple[list[Requirement], list[Requirement]]:
    """Return (hard, optional) requirements declared by the checkout."""
    hard: list[str] = []
    optional: list[str] = []
    pyproject = root / "pyproject.toml"
    if pyproject.exists():
        try:
            data = tomllib.loads(pyproject.read_text("utf-8", "replace"))
        except tomllib.TOMLDecodeError:
            data = {}
        project = data.get("project", {})
        hard += project.get("dependencies", [])
        optional += _flatten_groups(project.get("optional-dependencies", {}))
        optional += _flatten_groups(data.get("dependency-groups", {}))
        poetry = data.get("tool", {}).get("poetry", {})
        hard += [name for name in poetry.get("dependencies", {}) if name.lower() != "python"]
        for group in poetry.get("group", {}).values():
            optional += list(group.get("dependencies", {}))
    for pattern in ("requirements*.txt", "test-requirements*.txt"):
        for file in root.glob(f"**/{pattern}"):
            if ".git" in file.parts or "docker" in file.parts:
                continue
            for line in file.read_text("utf-8", "replace").splitlines():
                line = line.split("#")[0].strip()
                if line and not line.startswith("-"):
                    optional.append(line)

    def parse(items: list[str]) -> list[Requirement]:
        parsed = []
        for item in items:
            try:
                parsed.append(Requirement(item))
            except InvalidRequirement:
                continue
        return parsed

    return parse(hard), parse(optional)


def resolve(root: Path, wheels_dir: Path, index: dict[str, list[Wheel]], project: str = "") -> tuple[list[Wheel], list[str]]:
    """Pick one wheel per distribution (with transitive dependencies) that the mirror can supply."""
    hard, optional = declared_requirements(root)
    own = canonicalize_name(_project_name(root) or project)
    specs: dict[str, SpecifierSet] = {}
    for requirement in hard:
        _merge(specs, requirement)
    for requirement in optional:
        trial = dict(specs)
        _merge(trial, requirement)
        name = canonicalize_name(requirement.name)
        if name in index and any(w.version in trial[name] for w in index[name]):
            specs = trial
        elif name not in specs:
            specs[name] = SpecifierSet()
    for own_wheel in index.get(own, [])[:1]:
        for requirement in own_wheel.metadata_requires(wheels_dir):
            _merge(specs, requirement)
    for name in BASELINE:
        specs[name] = SpecifierSet()
    chosen: dict[str, Wheel] = {}
    missing: set[str] = set()
    queue = [name for name in specs if name != own]
    while queue:
        name = queue.pop()
        if name in chosen or name == own:
            continue
        candidates = index.get(name, [])
        spec = specs.get(name, SpecifierSet())
        allowed = [w for w in candidates if spec.contains(w.version, prereleases=True)]
        pool = allowed or candidates
        pick = next((w for w in pool if _installable(w, wheels_dir, index)), pool[0] if pool else None)
        if pick is None:
            missing.add(name)
            continue
        chosen[name] = pick
        for requirement in pick.metadata_requires(wheels_dir):
            if requirement.marker and not _marker_ok(requirement.marker):
                continue
            dep = canonicalize_name(requirement.name)
            specs[dep] = specs.get(dep, SpecifierSet()) & requirement.specifier
            queue.append(dep)
    missing -= BUILD_BACKENDS
    return sorted(chosen.values(), key=lambda w: w.name), sorted(missing)


def _installable(wheel: Wheel, wheels_dir: Path, index: dict[str, list[Wheel]]) -> bool:
    return all(
        canonicalize_name(r.name) in index
        for r in wheel.metadata_requires(wheels_dir)
        if not r.marker or _marker_ok(r.marker)
    )


def _marker_ok(marker: Marker) -> bool:
    try:
        return marker.evaluate({"extra": "", "python_version": PYTHON, "python_full_version": f"{PYTHON}.5"})
    except Exception:
        return False


def _merge(specs: dict[str, SpecifierSet], requirement: Requirement) -> None:
    if requirement.marker and not _marker_ok(requirement.marker):
        return
    name = canonicalize_name(requirement.name)
    specs[name] = specs.get(name, SpecifierSet()) & requirement.specifier


def _project_name(root: Path) -> str:
    try:
        data = tomllib.loads((root / "pyproject.toml").read_text("utf-8", "replace"))
    except (OSError, tomllib.TOMLDecodeError):
        return ""
    return data.get("project", {}).get("name") or data.get("tool", {}).get("poetry", {}).get("name", "")


def host_distributions(search_roots: list[Path]) -> dict[str, metadata.Distribution]:
    """Index distributions found under `search_roots` (site-packages or uv archive dirs), newest version wins."""
    found: dict[str, metadata.Distribution] = {}
    paths = [str(p) for root in search_roots for p in ([root] if root.name != "archive-v0" else root.iterdir())]
    for dist in metadata.distributions(path=paths):
        name = canonicalize_name(dist.metadata["Name"] or "")
        if name and (name not in found or Version(dist.version) > Version(found[name].version)):
            found[name] = dist
    return found


def providers(modules: tuple[str, ...], host: dict[str, metadata.Distribution]) -> list[str]:
    """Host distribution names that ship the given top-level import names."""
    wanted = {m.split(".")[0] for m in modules}
    names = []
    for name, dist in host.items():
        tops = {Path(str(f)).parts[0].removesuffix(".py") for f in dist.files or [] if f.suffix == ".py"}
        if wanted & tops:
            names.append(name)
    return names


def supplements(
    missing: list[str], chosen: list[Wheel], host: dict[str, metadata.Distribution]
) -> list[metadata.Distribution]:
    """Pure-Python host distributions standing in for packages the mirror lacks."""
    have = {w.name for w in chosen}
    found: dict[str, metadata.Distribution] = {}
    queue = list(missing)
    while queue:
        name = canonicalize_name(queue.pop())
        dist = host.get(name)
        if dist is None or name in have or name in found or name.startswith("pytest") or name in BUILD_BACKENDS:
            continue
        files = dist.files or []
        if any(f.suffix in (".so", ".pyd") or str(f).startswith("..") for f in files):
            continue
        found[name] = dist
        for line in dist.requires or []:
            requirement = Requirement(line)
            if not requirement.marker or _marker_ok(requirement.marker):
                queue.append(requirement.name)
    return sorted(found.values(), key=lambda d: canonicalize_name(d.name))


SHIM_PACKAGES = Path(__file__).parent / "shims"


def stand_ins(missing: list[str], extras: list[metadata.Distribution]) -> list[str]:
    """Names of bundled minimal stand-ins for test-only packages that neither mirror nor host provides."""
    covered = {canonicalize_name(d.name) for d in extras}
    available = {canonicalize_name(p.name): p.name for p in SHIM_PACKAGES.iterdir() if p.is_dir()}
    return sorted(available[n] for n in map(canonicalize_name, missing) if n in available and n not in covered)


def ensure_env(
    wheels: list[Wheel],
    extras: list[metadata.Distribution],
    shimmed: list[str],
    wheels_dir: Path,
    shims_dir: Path,
    envs_dir: Path,
) -> Path:
    """Create (or reuse) a venv holding exactly `wheels` plus `extras`; returns the venv directory."""
    ident = [w.filename for w in wheels] + [f"{d.name}=={d.version}" for d in extras] + [f"shim:{n}" for n in shimmed]
    key = hashlib.sha256("\n".join(ident).encode()).hexdigest()[:12]
    venv = envs_dir / key
    envs_dir.mkdir(parents=True, exist_ok=True)
    with open(envs_dir / f"{key}.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if (venv / ".ready").exists():
            return venv
        subprocess.run(["uv", "venv", "--python", PYTHON, str(venv)], check=True, capture_output=True)
        python = venv / "bin" / "python"
        files = [str(wheels_dir / w.filename) for w in wheels]
        subprocess.run(
            ["uv", "pip", "install", "--python", str(python), "--no-index", "--no-deps", *files],
            check=True,
            capture_output=True,
        )
        site = next((venv / "lib").glob("python3*/site-packages"))
        for dist in extras:
            for file in dist.files or []:
                target = site / file
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(dist.locate_file(file), target)
        for name in shimmed:
            shutil.copytree(SHIM_PACKAGES / name, site / name)
        for shim in ("imp.py", "telnetlib.py"):
            (site / shim).write_bytes((shims_dir / shim).read_bytes())
        (venv / "wheels.json").write_text(json.dumps(ident, indent=1))
        (venv / ".ready").touch()
    return venv


def load_scorer_setup(setup_py: Path):
    spec = importlib.util.spec_from_file_location("scorer_setup", setup_py)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def pytest_env(root: Path, project_dir: Path, scratch: Path) -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in {"PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME"}}
    env.update(
        PYTHONPATH=os.pathsep.join([str(root), str(root / "src"), str(project_dir)]),
        PYTHONDONTWRITEBYTECODE="1",
        TMPDIR=str(scratch),
        TEST_TMPDIR=str(scratch),
        PYTHONHASHSEED="0",
    )
    return env


@dataclass
class PytestRun:
    returncode: int | None
    timed_out: bool
    tail: str
    missing_modules: tuple[str, ...] = ()


def run_pytest(venv: Path, root: Path, targets: list[str], env: dict[str, str], extra: list[str], timeout: int) -> PytestRun:
    command = [str(venv / "bin" / "python"), "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=line", "--continue-on-collection-errors", *extra, *targets]
    try:
        done = subprocess.run(command, cwd=root, env=env, capture_output=True, timeout=timeout, text=True, errors="replace")
    except subprocess.TimeoutExpired as exc:
        out = (exc.stdout or b"")
        return PytestRun(None, True, (out if isinstance(out, str) else out.decode("utf-8", "replace"))[-600:])
    output = done.stdout + done.stderr
    missing = tuple(sorted(set(re.findall(r"No module named '([\w.]+)'", output))))
    return PytestRun(done.returncode, False, output[-600:], missing)

