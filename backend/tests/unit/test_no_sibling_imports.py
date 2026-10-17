"""No module under ``backend/src/`` may import a sibling application  (task 1.5).

ADR-001. The siblings are checked out side by side on this machine - ``/home/x-sean/app/
miSuite/miLLM``, ``miStudio``, ``miDataworks`` - so ``import millm`` *resolves locally* and a
coupling that should be impossible instead works on the developer's workstation and fails
in the container. miStudio already enforces the mirror of this rule and its guard forbids
importing ``miforge``.

The shape is copied from miStudio's ``test_mcp_tools_stay_on_http.py`` (see
``docs/REUSE.md``). What matters here is the **self-test**: a scanner that silently stops
recognising one import form is a guard that passes forever. Foundation task 10.2 requires
this guard to be seen red twice - once with a real sibling import added, and once with the
scanner deliberately blinded so its self-test fails.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC = REPO_ROOT / "backend" / "src"

#: The sibling applications. Each is a real importable package on this machine.
FORBIDDEN_ROOTS = frozenset({"millm", "mistudio", "midataworks"})

#: Deliberately empty, and it must stay that way. An allowlist is where a guard goes to
#: die: the first entry is always "just this one", and ADR-001 admits no exception -
#: there is nothing to import, because a sibling is reached over HTTP or not at all.
ALLOWLIST: frozenset[str] = frozenset()

#: Functions that import by *name*, which an `ast.Import` walk alone does not see.
DYNAMIC_IMPORTERS = frozenset({"import_module", "__import__"})


def _root_of(dotted: str) -> str:
    return dotted.split(".", 1)[0]


def find_sibling_imports(source: str, filename: str = "<string>") -> list[tuple[int, str]]:
    """Return ``(lineno, offending_name)`` for every sibling import in ``source``.

    Handles all five static forms and the two dynamic ones. Kept as a plain function over
    a string so the self-test can exercise it directly on synthetic source - a scanner
    that can only be run over the real tree cannot be proven to see anything.
    """
    try:
        tree = ast.parse(source, filename=filename)
    except SyntaxError as exc:  # a file that will not parse is its own failure
        raise AssertionError(f"{filename} does not parse: {exc}") from exc

    hits: list[tuple[int, str]] = []

    for node in ast.walk(tree):
        # `import millm`, `import millm.services`, `import millm as m`
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _root_of(alias.name) in FORBIDDEN_ROOTS:
                    hits.append((node.lineno, alias.name))

        # `from millm import x`, `from millm.services import y`
        elif isinstance(node, ast.ImportFrom):
            # `level > 0` is a relative import (`from . import x`) and has no sibling root.
            if node.level == 0 and node.module and _root_of(node.module) in FORBIDDEN_ROOTS:
                hits.append((node.lineno, node.module))

        # `importlib.import_module("millm")`, `__import__("millm.services")`
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            if name in DYNAMIC_IMPORTERS:
                for arg in node.args:
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and _root_of(arg.value) in FORBIDDEN_ROOTS
                    ):
                        hits.append((node.lineno, arg.value))

    return hits


def python_files() -> list[Path]:
    return sorted(p for p in SRC.rglob("*.py"))


def test_source_tree_is_not_empty() -> None:
    """A guard that scans nothing is green. See the schema-sync test for the same rule."""
    assert python_files(), f"no Python files found under {SRC}; the guard would be vacuous"


@pytest.mark.parametrize("path", python_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_module_does_not_import_a_sibling(path: Path) -> None:
    relative = str(path.relative_to(SRC))
    if relative in ALLOWLIST:  # pragma: no cover - the allowlist is empty
        pytest.fail(
            f"{relative} is in ALLOWLIST, but ADR-001 admits no exception: a sibling is "
            f"reached over HTTP or not at all. Remove the entry and the import."
        )
    hits = find_sibling_imports(path.read_text(encoding="utf-8"), relative)
    assert not hits, "\n".join(
        f"{relative}:{line} imports {name!r} - ADR-001 forbids it. Call the sibling over "
        f"HTTP through its single client module in backend/src/clients/."
        for line, name in hits
    )


# --------------------------------------------------------------------------------------
# The self-test. Task 1.5 requires it, and task 10.2 requires it to be SEEN failing.
# Each case is one import form. If a future edit makes the scanner stop recognising a
# form, the corresponding case here goes red and names the form that was lost.
# --------------------------------------------------------------------------------------

SELF_TEST_CASES: list[tuple[str, str]] = [
    ("plain import", "import millm\n"),
    ("dotted import", "import millm.services.sae_service\n"),
    ("aliased import", "import millm as engine\n"),
    ("from import", "from millm import attach\n"),
    ("dotted from import", "from mistudio.probes import load\n"),
    ("importlib by name", "import importlib\nm = importlib.import_module('midataworks')\n"),
    ("dunder import by name", "m = __import__('millm')\n"),
    ("import inside a function", "def go():\n    import millm\n    return millm\n"),
    ("import inside a try", "try:\n    import millm\nexcept ImportError:\n    millm = None\n"),
]


@pytest.mark.parametrize(("label", "source"), SELF_TEST_CASES, ids=[c[0] for c in SELF_TEST_CASES])
def test_scanner_sees_every_import_form(label: str, source: str) -> None:
    """The scanner must catch this form. If it does not, the guard above is decorative."""
    hits = find_sibling_imports(source, f"<self-test {label}>")
    assert hits, (
        f"the scanner did NOT see a sibling import in the {label!r} form:\n{source}\n"
        f"Every form it cannot see is a way for ADR-001 to be violated with a green suite."
    )


@pytest.mark.parametrize(
    "source",
    [
        "from . import sibling\n",
        "from .clients import millm_client\n",
        "import millmfoo\n",  # a package whose name merely starts with a forbidden root
        "from millmfoo import thing\n",
        "MILLM_BASE_URL = 'http://millm:8000'\n",  # a URL is how a sibling SHOULD be reached
        "import importlib\nm = importlib.import_module(name)\n",  # non-constant: cannot judge
    ],
    ids=["relative", "relative-client", "prefix-only", "prefix-only-from", "url", "dynamic-name"],
)
def test_scanner_does_not_cry_wolf(source: str) -> None:
    """False positives matter too: a guard that flags ``millmfoo`` or a base URL gets
    switched off, and then it protects nothing. ``import_module(name)`` with a non-constant
    argument is deliberately NOT flagged - it cannot be judged statically, and guessing
    would be the fabricated value ADR-018 forbids."""
    assert not find_sibling_imports(source, "<negative>")
