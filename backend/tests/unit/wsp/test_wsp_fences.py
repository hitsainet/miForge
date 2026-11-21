"""The WSP import fences (FTDD §6.1), enforced by AST over the real source tree.

Each fence is an **allow-list** where the layer is small enough to have one (``wsp.weight_ops``:
pure
numpy, D5), and a deny-list otherwise. Every walk asserts it found modules, because a walk over an
empty
or mis-rooted glob asserts nothing (suite memory ``source-scrape-guards-fail-open``); the
planted-import
controls (C3, C9b) prove each fence can go red.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[3] / "src"

#: wsp.weight_ops may import only these top-level names (and, inside miForge, only these modules).
WEIGHT_OPS_ALLOWED_TOP = frozenset(
    {
        "__future__",
        "collections",
        "dataclasses",
        "fnmatch",
        "hashlib",
        "io",
        "itertools",
        "json",
        "math",
        "os",
        "pathlib",
        "re",
        "struct",
        "sys",
        "typing",
        "zipfile",
        "numpy",
        "rfc8785",
    }
)
WEIGHT_OPS_ALLOWED_INTERNAL = ("core.errors", "core.atomic_dir", "wsp.weight_ops", "wsp.schemas")


def imported_modules(path: Path) -> list[tuple[int, str]]:
    """Every absolute module name ``path`` imports, with its line (relative imports resolved)."""
    package = ".".join(path.relative_to(SRC).with_suffix("").parts[:-1])
    return imports_in(path.read_text(encoding="utf-8"), package, str(path))


def imports_in(source: str, package: str, filename: str = "<planted>") -> list[tuple[int, str]]:
    """:func:`imported_modules` of a source text as if it lived in ``package`` (the planted-
    violation controls feed their sources through the same walk as the tree)."""
    tree = ast.parse(source, filename=filename)
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = package.split(".")[: len(package.split(".")) - (node.level - 1)]
                module = ".".join([*base, node.module] if node.module else base)
            else:
                module = node.module or ""
            found.append((node.lineno, module))
            # ``from x import y`` may import a submodule y; record it too.
            found.extend((node.lineno, f"{module}.{alias.name}") for alias in node.names)
        elif isinstance(node, ast.Call) and getattr(node.func, "id", None) == "__import__":
            found.append((node.lineno, "__import__"))
        elif isinstance(node, ast.Call) and getattr(node.func, "attr", None) == "import_module":
            found.append((node.lineno, "importlib.import_module"))
    return found


def modules_under(package: str) -> list[Path]:
    root = SRC.joinpath(*package.split("."))
    files = sorted(root.rglob("*.py"))
    assert files, f"no modules found under {root}: the fence would assert nothing"
    return files


def test_weight_ops_imports_no_io() -> None:
    """FR-016.3: wsp/weight_ops imports no LLM, HTTP, database or wsp_worker module (D5)."""
    files = modules_under("wsp.weight_ops")
    assert {"hashing.py", "evaluator.py", "inspect.py"} <= {f.name for f in files}, files
    bad: list[str] = []
    for path in files:
        for line, module in imported_modules(path):
            top = module.split(".")[0]
            if module in ("__import__", "importlib.import_module"):
                bad.append(f"{path.name}:{line} dynamic import")
            elif top in WEIGHT_OPS_ALLOWED_TOP or module.startswith(WEIGHT_OPS_ALLOWED_INTERNAL):
                continue
            else:
                bad.append(f"{path.name}:{line} imports {module}")
    assert not bad, bad


def _imports_under(package: str) -> list[tuple[Path, int, str]]:
    return [
        (path, line, module)
        for path in modules_under(package)
        for line, module in imported_modules(path)
    ]


WORKER_FORBIDDEN = ("wsp", "api", "db", "models", "services", "training", "workers", "clients")
GATEWAY_FORBIDDEN = ("db", "models", "services", "training")


def test_worker_imports_nothing_trusted() -> None:
    """FTDD §6.1: the untrusted worker imports nothing from the backend's packages."""
    bad = [
        f"{p.name}:{line} imports {m}"
        for p, line, m in _imports_under("wsp_worker")
        if m.split(".")[0] in WORKER_FORBIDDEN
    ]
    assert not bad, bad


def test_only_the_adapter_imports_weightspace() -> None:
    """D4, D6, PF-25: only wsp_worker/adapters/ imports weightspace (or another WSP library)."""
    bad = []
    for path in sorted(SRC.rglob("*.py")):
        if "wsp_worker/adapters" in path.as_posix():
            continue
        for line, module in imported_modules(path):
            if module.split(".")[0] in ("weightspace", "wsplab", "dspy", "dspy_wsp"):
                bad.append(f"{path.relative_to(SRC)}:{line} imports {module}")
    assert not bad, bad


def test_gateway_imports_no_database_or_training() -> None:
    gateway = SRC / "wsp_gateway"
    if not gateway.is_dir():
        pytest.skip("wsp_gateway/ lands in FTASKS 6.5; this fence then applies with no edit here")
    bad = [
        f"{p.name}:{line} imports {m}"
        for p, line, m in _imports_under("wsp_gateway")
        if m.split(".")[0] in GATEWAY_FORBIDDEN
    ]
    assert not bad, bad


def test_wsp_imports_no_midataworks_client() -> None:
    """FTDD §3.6 / FTASKS 28.3: WSP reads dataset versions only through miWorkbench's bundles."""
    bad = []
    for package in ("wsp", "wsp_worker", "wsp_gateway"):
        if not SRC.joinpath(package).is_dir():
            continue
        bad.extend(
            f"{p.name}:{line} imports {m}"
            for p, line, m in _imports_under(package)
            if m.startswith("clients.midataworks")
        )
    assert not bad, bad


# ------------------------------------------------- feature 018: the engine's fences (FTASKS 7.4)

ENGINE = "wsp_worker.miwsp_engine"
#: FR-018.16: no WSP library and no vendor LLM SDK in the engine.
ENGINE_FORBIDDEN = ("openai", "anthropic", "weightspace", "wsplab", "dspy", "dspy_wsp")
#: FR-018.16 / FTDD §3: the standard library, numpy, httpx2 and the worker's own modules only.
ENGINE_ALLOWED_TOP = frozenset(sys.stdlib_module_names) | {"numpy", "httpx2", "wsp_worker"}


def engine_sdk_violations(imports: list[tuple[str, int, str]]) -> list[str]:
    return [
        f"{p}:{line} imports {m}" for p, line, m in imports if m.split(".")[0] in ENGINE_FORBIDDEN
    ]


def engine_allowlist_violations(imports: list[tuple[str, int, str]]) -> list[str]:
    bad = []
    for p, line, m in imports:
        if m in ("__import__", "importlib.import_module"):
            bad.append(f"{p}:{line} dynamic import")
        elif m.split(".")[0] not in ENGINE_ALLOWED_TOP:
            bad.append(f"{p}:{line} imports {m}")
    return bad


def engine_importer_violations(imports: list[tuple[str, int, str]]) -> list[str]:
    """Only ``wsp_worker`` modules may import the engine."""
    return [
        f"{p}:{line} imports {m}"
        for p, line, m in imports
        if (m == ENGINE or m.startswith(ENGINE + ".") or m.split(".")[0] == "miwsp_engine")
        and not str(p).startswith("wsp_worker/")
    ]


def _engine_imports() -> list[tuple[str, int, str]]:
    files = modules_under(ENGINE)
    names = {f.name for f in files}
    assert {"session.py", "client.py", "kernel.py", "kernel_main.py", "store.py"} <= names, names
    return [
        (f.relative_to(SRC).as_posix(), line, m) for f in files for line, m in imported_modules(f)
    ]


def _planted(source: str, package: str = ENGINE) -> list[tuple[str, int, str]]:
    path = package.replace(".", "/") + "/planted.py"
    return [(path, line, m) for line, m in imports_in(source, package)]


def test_the_engine_imports_no_wsp_library_or_vendor_sdk() -> None:
    assert engine_sdk_violations(_engine_imports()) == []


@pytest.mark.parametrize(
    "source",
    [
        "import openai",
        "from anthropic import Anthropic",
        "import weightspace.llm",
        "from dspy_wsp import x",
        "import wsplab",
    ],
)
def test_the_sdk_fence_goes_red_on_a_planted_import(source: str) -> None:
    assert engine_sdk_violations(_planted(source))


def test_the_engine_imports_only_the_stdlib_numpy_httpx2_and_the_worker() -> None:
    imports = _engine_imports()
    assert {m.split(".")[0] for _p, _l, m in imports} >= {"numpy", "httpx2", "wsp_worker"}
    assert engine_allowlist_violations(imports) == []


@pytest.mark.parametrize(
    "source",
    [
        "import requests",
        "import torch",
        "from wsp import runs",
        "__import__('os')",
        "import importlib\nimportlib.import_module('x')",
        "from ...api import deps",
    ],
)
def test_the_allowlist_fence_goes_red_on_a_planted_import(source: str) -> None:
    assert engine_allowlist_violations(_planted(source))


def test_only_wsp_worker_imports_the_engine() -> None:
    imports = [
        (f.relative_to(SRC).as_posix(), line, m)
        for f in sorted(SRC.rglob("*.py"))
        for line, m in imported_modules(f)
    ]
    importers = {p for p, _l, m in imports if m.startswith(ENGINE)}
    assert "wsp_worker/__main__.py" in importers  # the dispatch (FR-018.2) is seen by the walk
    assert engine_importer_violations(imports) == []


@pytest.mark.parametrize(
    ("source", "package"),
    [
        ("from wsp_worker import miwsp_engine", "wsp"),
        ("from wsp_worker.miwsp_engine import session", "api.v1"),
        ("import miwsp_engine", "services"),
    ],
)
def test_the_importer_fence_goes_red_on_a_planted_import(source: str, package: str) -> None:
    assert engine_importer_violations(_planted(source, package))


# --------------------------------------------- feature 018: the generator's reference stays apart


#: The shared generator's parity reference must not come from the code it checks.
REFERENCE_FORBIDDEN = ("wsp.weight_ops.evaluator", "wsp.bundles", "wsp_worker")


def _reference_violations(found: list[tuple[int, str]]) -> list[str]:
    return [
        f"{line}: {module}"
        for line, module in found
        if module.startswith(REFERENCE_FORBIDDEN)
        or module in ("__import__", "importlib.import_module")
    ]


def test_the_shared_bundle_generator_never_imports_the_evaluator() -> None:
    """018 FTASKS 9.4: ``wsp.fixture_bundle`` (moved out of the test support so the protocol test
    and the tests share one generator) computes its parity scores with its own pure-Python
    reference. If it imported the evaluator, the import's parity check would compare the
    evaluator with itself and pass by construction."""
    path = SRC / "wsp" / "fixture_bundle.py"
    found = imported_modules(path)
    assert any(module == "numpy" for _line, module in found), "the walk read nothing"
    assert _reference_violations(found) == []


@pytest.mark.parametrize(
    "planted",
    [
        "from wsp.weight_ops.evaluator import parity\n",
        "from wsp.weight_ops import evaluator\n",
        "import wsp.bundles\n",
        "from .weight_ops.evaluator import row_scores\n",
    ],
)
def test_the_reference_fence_names_a_planted_import(planted: str) -> None:
    assert _reference_violations(imports_in(planted, "wsp")), planted
