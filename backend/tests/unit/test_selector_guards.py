"""Static guards for F3  (FR-003.13, AC-003.9; ADR-001, ADR-010, ADR-017, ADR-018).

Four things kept true by scanning the source rather than by remembering them, **each with a
self-test proving the scanner sees the form it forbids** - the shape foundation 1.5
establishes, because *a scanner that silently stops recognising one form is a guard that
passes forever.*

1. **Nothing under ``services/selectors/`` imports ``training/``, ``packs/``, ``workers/``
   or ``clients/``.** That is what keeps *"a selector is a pure function of a completion"*
   true rather than aspirational, and it is why F3 depends on nothing from F1. ``clients/``
   is on the list on purpose: depending on F10's concrete client instead of the port is how
   the port stops being a port.
2. **No second miDataworks call site in this feature** (ADR-010). ``F10 is not ready yet so
   I will just call it directly`` is how the seam stops being findable.
3. **Nothing in this package constructs a provider.** F3 resolves one from F4's registry;
   a second construction path here would be the import-by-module-path resolution this
   feature's own registry rule exists to forbid, one directory over.
4. **No verdict type is defined outside the contract.** One exported enum, three members,
   no synonyms - because a gate task once offered ``undecided``/``pending`` as if either
   would do, and a consumer reading two names for one value invented a fourth vocabulary.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
PACKAGE = SRC / "services" / "selectors"
GRADERS = SRC / "services" / "graders"

#: Packages a selector may not reach into. ``training`` is the headline one; ``clients`` is
#: the subtle one.
FORBIDDEN_PACKAGES = frozenset({"training", "packs", "workers", "clients"})


def _modules() -> list[Path]:
    return sorted([*PACKAGE.rglob("*.py"), *GRADERS.rglob("*.py")])


def find_forbidden_imports(source: str, filename: str = "<string>") -> list[tuple[int, str]]:
    """``(lineno, name)`` for every import of a forbidden package.

    A plain function over a string so the self-test can exercise it on synthetic source: a
    scanner that can only be run over the real tree cannot be proven to see anything.
    """
    tree = ast.parse(source, filename=filename)
    hits: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in FORBIDDEN_PACKAGES:
                    hits.append((node.lineno, alias.name))
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and node.module.split(".")[0] in FORBIDDEN_PACKAGES:
                hits.append((node.lineno, node.module))
        elif isinstance(node, ast.Call):
            func = node.func
            name = (
                func.attr
                if isinstance(func, ast.Attribute)
                else func.id if isinstance(func, ast.Name) else None
            )
            if name in {"import_module", "__import__"}:
                for arg in node.args:
                    if (
                        isinstance(arg, ast.Constant)
                        and isinstance(arg.value, str)
                        and arg.value.split(".")[0] in FORBIDDEN_PACKAGES
                    ):
                        hits.append((node.lineno, arg.value))
    return hits


def test_the_scan_is_not_vacuous() -> None:
    """A guard that scans nothing is green."""
    assert _modules(), f"no modules found under {PACKAGE}; the guards would be vacuous"


class TestNoImportOfTheForbiddenPackages:
    @pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
    def test_module_does_not_reach_into_another_layer(self, path: Path) -> None:
        hits = find_forbidden_imports(path.read_text(encoding="utf-8"), path.name)
        assert not hits, "\n".join(
            f"{path.name}:{line} imports {name!r}. A selector is a pure function of a "
            f"completion (FR-003.13); reach a sibling's queue through the port and a "
            f"provider through F4's registry."
            for line, name in hits
        )

    @pytest.mark.parametrize(
        ("label", "source"),
        [
            ("plain import", "import training\n"),
            ("dotted import", "import training.hotupdate.heartbeat\n"),
            ("aliased import", "import workers as w\n"),
            ("from import", "from clients import midataworks\n"),
            ("dotted from import", "from packs.registry import load\n"),
            ("inside a function", "def go():\n    import training\n"),
            ("inside a try", "try:\n    import clients\nexcept ImportError:\n    pass\n"),
            ("importlib by name", "import importlib\nm = importlib.import_module('workers')\n"),
            ("dunder import", "m = __import__('training')\n"),
        ],
        ids=lambda v: v if isinstance(v, str) and " " in v else "",
    )
    def test_the_scanner_sees_every_import_form(self, label: str, source: str) -> None:
        """Every form it cannot see is a way for FR-003.13 to be violated with a green
        suite."""
        assert find_forbidden_imports(
            source, f"<self-test {label}>"
        ), f"the scanner did NOT see a forbidden import in the {label!r} form:\n{source}"

    @pytest.mark.parametrize(
        "source",
        [
            "from . import registry\n",
            "from .plugins import rule\n",
            "import trainingwheels\n",
            "from packsmith import x\n",
            "from core.errors import refuse\n",
            "from providers.base import InferenceProvider\n",
            "import importlib\nm = importlib.import_module(name)\n",
        ],
        ids=[
            "relative",
            "relative-sub",
            "prefix-only",
            "prefix-only-from",
            "core",
            "providers",
            "dynamic-name",
        ],
    )
    def test_the_scanner_does_not_cry_wolf(self, source: str) -> None:
        """False positives matter too: a guard that flags ``trainingwheels`` or a legitimate
        ``core`` import gets switched off, and then it protects nothing.
        ``import_module(name)`` with a non-constant argument is deliberately NOT flagged -
        it cannot be judged statically, and guessing would be the fabricated value ADR-018
        forbids."""
        assert not find_forbidden_imports(source, "<negative>")


class TestNoSecondMiDataworksCallSite:
    """AC-003.9. ADR-010: one outbound call site per sibling, and it is F10's."""

    @pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
    def test_no_module_imports_httpx(self, path: Path) -> None:
        hits = [
            node.lineno
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
            if (isinstance(node, ast.Import) and any(a.name == "httpx" for a in node.names))
            or (isinstance(node, ast.ImportFrom) and node.module == "httpx")
        ]
        assert not hits, (
            f"{path.name}:{hits} imports httpx. F3's only HTTP is the judge, and that goes "
            f"through F4's provider registry; the review queue goes through F10's single "
            f"outbound call site (ADR-010)."
        )

    @pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
    def test_no_module_names_a_midataworks_route(self, path: Path) -> None:
        # Docstrings and comments are stripped first: this module's own docstring CITES
        # `midataworks_base_url` while explaining the split-module blindness that made the
        # base-URL guard miss three settings classes. A citation is not a call, and a checker
        # that fires on documentation gets switched off. See `tests/support/source_text`.
        from tests.support.source_text import code_only

        source = code_only(path.read_text(encoding="utf-8"))
        for fragment in ("/api/v1/review-queues", "X-Dataworks-Agent", "midataworks_base_url"):
            assert fragment not in source, (
                f"{path.name} names {fragment!r}. The route shapes are recorded in the "
                f"FPRD as the sibling's to change, not miForge's to call."
            )


class TestNothingConstructsAProvider:
    """FTASKS 4.1a: resolve from F4's registry; construct none here."""

    @pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
    def test_no_module_instantiates_the_http_provider(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "OpenAICompatibleProvider"
        ]
        assert not hits, (
            f"{path.name}:{hits} constructs a provider. F3 ships the implementation into "
            f"providers/ and RESOLVES one from F4's registry; a construction path here "
            f"would be the import-by-module-path resolution the registry rule forbids."
        )

    @pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
    def test_no_module_imports_the_http_provider_implementation(self, path: Path) -> None:
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source)
        hits = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "providers.openai_compatible"
        ]
        assert not hits, (
            f"{path.name}:{hits} imports providers.openai_compatible. The package depends "
            f"on the INTERFACE (providers.base) and on the resolver port, never on one "
            f"implementation."
        )

    def test_f3s_module_is_where_the_split_puts_it(self) -> None:
        """F3 ships the HTTP implementation; the interface is the foundation's.

        **Loosened 2026-10-14, and the original fired on correct work.** This asserted
        ``providers/`` holds *exactly* ``["base.py", "openai_compatible.py"]`` - while its
        own docstring said *"F4 owns the registry, the in-process path, provider identity
        and the settings"* and its own failure message said *"the rest is F4's"*. So the
        assertion forbade precisely what the test's own prose anticipated, and it went red
        the moment F4 added `registry.py`, `in_process.py`, `model_handle.py`, `settings.py`
        and `fakes.py` - all of them its own, all of them expected.

        A ratchet that fires on correct work gets loosened rather than respected, so the
        useful thing is to assert the invariant instead of the inventory. **The invariant is
        already enforced by the two tests above**: no module under ``services/selectors/``
        constructs a provider, and none imports ``providers.openai_compatible`` - the
        package depends on the *interface* and on the resolver port, never on one
        implementation. Those bite whatever else lands in the directory.

        What is left for this test is the half the others cannot see: that the split put
        F3's module where the split says, and that the interface it depends on is there.
        """
        providers = {
            path.name for path in (SRC / "providers").glob("*.py") if path.name != "__init__.py"
        }
        assert "openai_compatible.py" in providers, (
            "F3's HTTP implementation is missing from providers/. R-02.11 makes the "
            "in-process provider unusable as a judge, so F3 cannot wait on F4 for it."
        )
        assert "base.py" in providers, "the foundation's interface is missing"


class TestOneVerdictTypeInTheWholeTree:
    """Foundation 3.1a's spelling, kept from being re-invented."""

    def test_only_the_contract_defines_a_verdict_enum(self) -> None:
        offenders: list[str] = []
        for path in sorted(SRC.rglob("*.py")):
            if path == SRC / "schemas" / "selector.py":
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ClassDef):
                    continue
                bases = {ast.unparse(base) for base in node.bases}
                looks_like_enum = any("Enum" in base for base in bases)
                if looks_like_enum and "verdict" in node.name.lower():
                    offenders.append(f"{path.relative_to(SRC)}:{node.lineno} {node.name}")
        assert offenders == [], (
            f"a second verdict type is defined at {offenders}. One exported enum, three "
            f"members, no synonyms: a consumer reading two names for one value invented a "
            f"fourth vocabulary once already."
        )

    def test_the_member_set_is_exactly_the_three(self) -> None:
        from schemas.selector import VerdictValue

        assert {member.value for member in VerdictValue} == {"accept", "reject", "undecided"}

    def test_this_feature_imports_the_contract_type_rather_than_aliasing_it(self) -> None:
        from schemas.selector import VerdictValue
        from services.selectors.types import VerdictRecord

        assert (
            VerdictRecord.model_fields["verdict"].annotation.model_fields["value"].annotation
            is VerdictValue
        )

    def test_admission_is_decided_in_one_place_only(self) -> None:
        """``is_admitted`` is the only route to admitted, so *"no path from undecided to
        admitted"* is enforced once rather than trusted four times.

        Scanned for the shape that routes around it: a comparison against the ACCEPT member
        outside the contract module and outside a test.
        """
        offenders: list[str] = []
        for path in _modules():
            source = path.read_text(encoding="utf-8")
            for lineno, line in enumerate(source.splitlines(), start=1):
                stripped = line.strip()
                if stripped.startswith("#"):
                    continue
                if "VerdictValue.ACCEPT" in stripped and ("==" in stripped or " is " in stripped):
                    offenders.append(f"{path.name}:{lineno} {stripped}")
        assert offenders == [], (
            f"a comparison against VerdictValue.ACCEPT at {offenders} routes around "
            f"is_admitted(), which is the one place admission is decided."
        )


class TestNoGetattrDefaultThatAlwaysFires:
    """ADR-018, and the shape to watch in this feature is a defaulted judge threshold."""

    @pytest.mark.parametrize("path", _modules(), ids=lambda p: p.name)
    def test_no_three_argument_getattr_with_a_constant_default(self, path: Path) -> None:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        hits: list[str] = []
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "getattr"
                and len(node.args) == 3
            ):
                hits.append(f"{node.lineno}: {ast.unparse(node)}")
        # Optional plugin CAPABILITIES looked up by name (a method or None), not values with a
        # default. `release_batch` joined 2026-10-16 with the judge prefetch (tracker 4.28).
        allowed_prefixes = ("prepare", "prefetch", "grade", "release_batch")
        hits = [
            hit
            for hit in hits
            if not any(f"'{prefix}'" in hit or f'"{prefix}"' in hit for prefix in allowed_prefixes)
        ]
        assert hits == [], (
            f"{path.name} has a getattr default: {hits}. A default that always fires is a "
            f"fabricated value, and in this feature the shape to watch is "
            f"getattr(config, 'threshold', 0.5) - a verdict policy chosen by whoever wrote "
            f"the line."
        )
