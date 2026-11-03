"""No sibling's name in ``providers/``, and a scanner that can prove it  (FR-004.3, AC-004.3).

R-02.49 exists because *"a test path wired to one sibling by name is what made the whole
document depend on it."* miLLM, vLLM, Ollama and the OpenAI API are **configuration values,
not code paths**, so no module under ``providers/`` may carry a sibling-specific branch, a
sibling-specific endpoint path, or a sibling's name in a conditional or a default.

**It is the test, not the intention, that keeps the property** - and a guard that cannot see
the thing it forbids is worse than none, because it reads as protection. So the scanner
carries a **self-test** proving it catches each form it is meant to catch, modelled on
foundation 1.5's import guard and on miStudio's stay-on-HTTP guard, which already names
``miforge``.

The allowlist starts empty and stays empty. A row for miLLM is a row of configuration.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import ClassVar

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
PROVIDERS = SRC / "providers"

#: The sibling names a provider module may not contain. ``miforge`` is deliberately absent -
#: this is our own package - and the list is the suite's three siblings, which is also
#: ADR-001's import-guard list.
SIBLINGS = ("millm", "mistudio", "midataworks")

#: Deliberately empty, and a test asserts it stays that way. An entry here would be a
#: sibling-specific code path with a note saying it was allowed.
ALLOWLIST: frozenset[str] = frozenset()


def scan_text(source: str) -> list[str]:
    """Every sibling-shaped occurrence in ``source``, as ``"<line>:<form>"``.

    Three forms, because they are three different ways the same dependency gets written and
    a guard that saw only one would read as protection against all three:

    * a sibling's name in a **string** - a hardcoded endpoint, a default URL, a model name;
    * a sibling's name in an **identifier** - ``settings.millm_api_url``, ``MILLM_BASE_URL``;
    * a sibling's name in a **comparison**, which is the conditional branch form.

    Comments are deliberately **not** scanned: the module docstrings here explain *why*
    miLLM must be a row and not a name, and a guard that forbade saying so would make the
    reasoning unwritable.
    """
    hits: list[str] = []
    tree = ast.parse(source)
    pattern = re.compile("|".join(SIBLINGS), re.IGNORECASE)
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            # A docstring is an Expr whose value is the constant; those are documentation.
            if pattern.search(node.value) and not _is_docstring(tree, node):
                hits.append(f"{node.lineno}:string")
        elif isinstance(node, ast.Name) and pattern.search(node.id):
            hits.append(f"{node.lineno}:name")
        elif isinstance(node, ast.Attribute) and pattern.search(node.attr):
            hits.append(f"{node.lineno}:attribute")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom) and node.module:
                names.append(node.module)
            if any(pattern.search(name) for name in names):
                hits.append(f"{node.lineno}:import")
    return sorted(set(hits))


def _is_docstring(tree: ast.AST, target: ast.Constant) -> bool:
    """Whether ``target`` is a module, class or function docstring."""
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and body[0].value is target
            ):
                return True
    return False


def _modules() -> list[Path]:
    return sorted(path for path in PROVIDERS.glob("*.py") if path.name != "__init__.py")


class TestTheScannerSeesEachFormItForbids:
    """**The self-test.** Without it the guard below is a hope with a green tick."""

    def test_a_name_in_a_conditional(self) -> None:
        assert scan_text('def f(kind):\n    if kind == "millm":\n        return 1\n')

    def test_a_hardcoded_endpoint_path(self) -> None:
        assert scan_text('url = "http://millm:8000/api/weights/attach"\n')

    def test_a_sibling_shaped_default(self) -> None:
        assert scan_text("base_url = settings.millm_api_url or 'x'\n")

    def test_a_sibling_import(self) -> None:
        assert scan_text("from millm.services import sae_service\n")

    def test_a_sibling_named_variable(self) -> None:
        assert scan_text("MIDATAWORKS_BASE = 'http://dw'\n")

    def test_it_does_not_fire_on_a_docstring_explaining_the_rule(self) -> None:
        """The reasoning must stay writable.

        Every module in this package explains *why* miLLM is a row and not a name; a guard
        that forbade the explanation would make the rule unstateable in the place it applies.
        """
        assert scan_text('"""miLLM is one row beside vLLM, never a named dependency."""\n') == []

    def test_it_does_not_fire_on_clean_code(self) -> None:
        """The other half of a self-test: it must also be able to say *no*.

        A scanner that returned a hit for everything would pass every assertion above and
        fail the real one below for the wrong reason.
        """
        assert scan_text("base_url = row.base_url\nmodel = row.model\n") == []


class TestNoProviderModuleNamesASibling:
    @pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
    def test_the_module_is_clean(self, path: Path) -> None:
        hits = scan_text(path.read_text(encoding="utf-8"))
        assert hits == [], (
            f"{path.name} names a sibling at {hits}. R-02.49: miLLM is a configured "
            f"base_url, not a code path - *a test path wired to one sibling by name is what "
            f"made the whole document depend on it.*"
        )

    def test_the_allowlist_is_empty(self) -> None:
        assert ALLOWLIST == frozenset(), (
            "an allowlist entry is a sibling-specific code path with a note saying it was "
            "allowed. There is no such row; there is configuration."
        )


class TestThePackageHoldsExactlyTheRatifiedSet:
    """The ownership split of foundation task 3.8, asserted as a file list.

    *This replaces an assertion that could no longer be true.* F3's
    ``test_f3_ships_exactly_one_module_into_providers`` asserts the package holds **exactly**
    ``base.py`` and ``openai_compatible.py``, with the docstring *"F4 owns the registry, the
    in-process path, provider identity and the settings"* - a sentence its own assertion
    makes unsatisfiable the moment F4 lands. The invariant F3 meant is kept here, widened by
    exactly F4's share and no more: **F3 contributes one HTTP module and nothing else, and
    nothing beyond this set appears without a reviewer.**
    """

    EXPECTED: ClassVar[list[str]] = [
        "base.py",  # the foundation's: the interface (task 3.8)
        "fakes.py",  # F4: the no-GPU handle and provider (ADR-023)
        "in_process.py",  # F4: ADR-009's default implementation
        "model_handle.py",  # F4: the port F1's acquisition satisfies (FR-004.11)
        "openai_compatible.py",  # F3's: reused, never forked
        "registry.py",  # F4: the ONE resolution path
        "settings.py",  # F4: rows from live settings
    ]

    def test_the_file_set_is_exactly_the_ratified_one(self) -> None:
        found = sorted(path.name for path in _modules())
        assert found == self.EXPECTED, (
            f"providers/ holds {found}. The ratified split (foundation 3.8) is: the "
            f"interface is the foundation's, openai_compatible.py is F3's, and the registry, "
            f"the provider identity and the in-process path are F4's. A module beyond this "
            f"set is a second answer to 'which provider produced this score'."
        )

    def test_there_is_exactly_one_http_implementation(self) -> None:
        """F4 reuses F3's module. **No subclass, no wrapper, no copy** (FR-004.1).

        Asserted by looking for a second ``httpx`` client in the package rather than by
        counting files, because the fork that matters is a *class*, not a filename.
        """
        offenders: list[str] = []
        for path in _modules():
            if path.name == "openai_compatible.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [alias.name for alias in node.names]
                    if isinstance(node, ast.ImportFrom) and node.module:
                        names.append(node.module)
                    if any(name.split(".")[0] == "httpx" for name in names):
                        offenders.append(f"{path.name}:{node.lineno}")
        assert offenders == [], (
            f"a second HTTP client lives at {offenders}. Two HTTP providers are two sets of "
            f"timeout semantics and two answers to capabilities()."
        )

    def test_nothing_subclasses_f3s_provider(self) -> None:
        offenders: list[str] = []
        for path in _modules():
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    bases = {ast.unparse(base) for base in node.bases}
                    if any("OpenAICompatible" in base for base in bases):
                        offenders.append(f"{path.name}:{node.lineno} {node.name}")
        assert offenders == [], f"F3's HTTP provider is subclassed at {offenders}; reuse it"


class TestNoHardCodedGpuSite:
    """ADR-008: **by UUID, resolved at submit from live ``nvidia-smi -L``.**

    An index inverted once on this cluster and a copied bus address unbound the root NVMe.
    F4 adds no entry to the ratchet's allowlist, so the assertion is simply that it has
    nothing to add.
    """

    @pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
    def test_no_device_index_or_bus_address(self, path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        offenders = [
            match.group(0)
            for match in re.finditer(r"cuda:\d|0000:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d", source)
        ]
        assert offenders == [], f"{path.name} hard-codes a GPU site: {offenders}"
