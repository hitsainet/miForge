"""One outbound call site, and miForge writes no dataset-version manifest  (FR-010.7, .20).

Two structural tests, both by **AST**. ADR-016 #1: *a capability is not shipped until a test
fails when its wiring is removed* - and the mirror of it, a prohibition is not enforced until
a test fails when it is broken.

**Walking the tree, not the text.** miDataworks' own single-call-site test says why: *"so a
comment or docstring naming a function neither satisfies nor trips a check."* This file's own
docstring mentions ``httpx`` and ``midataworks-dataset-version.json`` repeatedly, and a
text-matching version of either test would be unable to tell that from a real use.

**The negative controls for these two live in ``test_midataworks_mutations.py``** and are run
rather than assumed: plant a second ``httpx`` import outside the client, plant a write to the
manifest filename, require a red each time.

**Why a second call site is worth a structural test at all.** A seam with two call sites has
two sets of headers, two retry policies and two opinions about what "unreachable" means - and
the second one is always the one that forgets the agent header. That is not hypothetical: the
hard dependency this project removed grew to thirteen required rows one convenience at a time.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

#: The only modules that may reach miDataworks over HTTP. ``midataworks.py`` is the wrapper;
#: ``midataworks_review.py`` is F3's port implemented **through** that wrapper and imports no
#: transport of its own, which this file also asserts.
ALLOWED_HTTPX_MODULES: frozenset[str] = frozenset({"clients/midataworks.py"})

#: The instance filename miDataworks writes. **No ``-v1``** - that is the schema's name.
MANIFEST_INSTANCE = "midataworks-dataset-version.json"


#: Modules that may legitimately *name* the instance filename: the validator declares it as
#: a constant, the ingest path looks for it, the index walks for it. Naming it is not
#: writing it - the test below is about writes.
def modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


#: F10's own modules in the shared ``clients/`` directory. **Named, not prefix-matched**:
#: ADR-017 puts one module per optional sibling under ``clients/``, so the directory
#: belongs to no single feature and a prefix rule would refuse the next sibling's client.
F10_CLIENT_MODULES: tuple[str, ...] = (
    "clients/availability.py",
    "clients/capabilities.py",
    "clients/midataworks.py",
    "clients/midataworks_errors.py",
    "clients/midataworks_review.py",
)

#: F10's own service packages. These *are* whole-package rules: no other feature writes in
#: them, so a new module in either is F10's and is covered automatically.
F10_PACKAGE_PREFIXES: tuple[str, ...] = ("services/parts/", "services/review/")


def f10_modules() -> list[Path]:
    """Every F10 module, with each named path proved to exist.

    The existence check is the part that matters: a renamed module would silently leave
    :data:`F10_CLIENT_MODULES` and the guards over it would narrow without failing - the
    construction-green failure this project keeps finding, including in its own review
    harness.
    """
    found: list[Path] = []
    for name in F10_CLIENT_MODULES:
        path = SRC / name
        assert path.is_file(), (
            f"{name} is listed as an F10 module and does not exist. Update "
            f"F10_CLIENT_MODULES, or the guards over it have quietly narrowed."
        )
        found.append(path)
    found.extend(path for path in modules() if relative(path).startswith(F10_PACKAGE_PREFIXES))
    return sorted(set(found))


def relative(path: Path) -> str:
    return str(path.relative_to(SRC))


def imported_names(tree: ast.AST, *, runtime_only: bool = False) -> set[str]:
    """Top-level module names this tree imports, in every form.

    ``import x``, ``import x.y``, ``from x import y`` and ``from x.y import z`` all count.
    A guard that only knew one form is a guard the next author routes around by accident -
    which is why ``test_the_scanner_sees_every_import_form`` below is a self-test rather
    than a comment.

    Args:
        runtime_only: Skip imports inside ``if TYPE_CHECKING:``. **A typing-only import
            cannot issue a request**, so it is not an outbound call site - and forbidding
            one would force a module that merely *annotates* a transport to drop its types,
            which trades a real guarantee (MyPy strict over the seam) for a cosmetic one.
    """
    found: set[str] = set()
    skip: set[int] = set()
    if runtime_only:
        for node in ast.walk(tree):
            if isinstance(node, ast.If) and _is_type_checking_test(node.test):
                for child in node.body:
                    for inner in ast.walk(child):
                        skip.add(id(inner))
    for node in ast.walk(tree):
        if id(node) in skip:
            continue
        if isinstance(node, ast.Import):
            for alias in node.names:
                found.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.add(node.module.split(".")[0])
    return found


def _is_type_checking_test(test: ast.expr) -> bool:
    if isinstance(test, ast.Name):
        return test.id == "TYPE_CHECKING"
    return isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"


def _code_only(source: str) -> str:
    """``source`` with every docstring and whole-line comment removed.

    **Promoted to ``tests/support/source_text.code_only`` on 2026-10-16** and kept here as a
    thin alias so this file's callers are untouched. It was retuned here on 2026-10-14 after
    F4's ``clients/activations.py`` was swept into the offender list for one line of prose
    naming ``clients/midataworks.py`` - *a citation is not an operation*. Two more guards hit
    the identical defect on 2026-10-16, so the implementation moved rather than being copied a
    fourth time; the full history is in that module's docstring.
    """
    from tests.support.source_text import code_only

    return code_only(source)


def mentions_midataworks(path: Path) -> bool:
    """Whether a module's **code** has anything to do with the miDataworks seam.

    Only a **filter**: the assertion below is still made on the syntax tree. ADR-010 allows
    one outbound call site *per sibling*, so a module using ``httpx`` for something else -
    ``providers/openai_compatible.py`` reaching an OpenAI-compatible endpoint (ADR-009), or
    ``clients/activations.py`` reaching R-02.19's activation source - is not in scope and
    must not be swept up. Scoping this rule to the whole tree would make F10's test fail on
    another feature's legitimate HTTP client, which is how a guard gets loosened to make an
    unrelated feature green.

    Docstrings and comments are excluded - see :func:`_code_only`.
    """
    text = _code_only(path.read_text("utf-8")).lower()
    return "midataworks" in text or "dataworks-agent" in text


class TestOneOutboundCallSite:
    def test_only_the_client_reaches_midataworks_over_http(self):
        """No module that mentions miDataworks imports ``httpx`` at runtime but the wrapper."""
        offenders = [
            relative(path)
            for path in modules()
            if mentions_midataworks(path)
            and "httpx" in imported_names(ast.parse(path.read_text("utf-8")), runtime_only=True)
            and relative(path) not in ALLOWED_HTTPX_MODULES
        ]
        assert offenders == [], (
            f"{offenders} import httpx and mention miDataworks. ADR-010 allows exactly one "
            f"outbound call site per sibling, and it is {sorted(ALLOWED_HTTPX_MODULES)}. A "
            f"second site has its own headers and its own idea of what 'unreachable' means, "
            f"and it is always the one that forgets the agent header."
        )

    def test_no_f10_module_but_the_client_imports_httpx(self):
        """The same rule stated over this feature's own packages, so a failure points here.

        Broader than the test above - it does not need the module to *mention* miDataworks -
        and narrower in scope, which is the combination that keeps the guard tight without
        making F10's test fail on another feature's legitimate HTTP client.

        *Scope corrected 2026-10-14.* This selected every module under ``clients/``, and
        **``clients/`` is not F10's package**: ADR-017 puts *one module per optional
        sibling* there, so the directory is shared by construction and the next sibling's
        client must import ``httpx`` to exist at all. As written, the test refused F4's
        ``clients/activations.py`` - R-02.19's single call site, built to exactly the shape
        ADR-010 prescribes - and would refuse every sibling client after it. F10's own
        modules are named instead, and :func:`f10_modules` asserts each one resolves so the
        list cannot go stale into vacuity.
        """
        mine = f10_modules()
        offenders = [
            relative(path)
            for path in mine
            if "httpx" in imported_names(ast.parse(path.read_text("utf-8")), runtime_only=True)
            and relative(path) not in ALLOWED_HTTPX_MODULES
        ]
        assert offenders == [], offenders

    def test_the_allowed_module_actually_exists_and_uses_it(self):
        """Otherwise both tests above pass vacuously - which is the construction-green
        failure this project keeps finding, including in its own review harness."""
        for allowed in ALLOWED_HTTPX_MODULES:
            path = SRC / allowed
            assert path.is_file(), f"{allowed} does not exist; the allowlist is stale"
            assert "httpx" in imported_names(
                ast.parse(path.read_text("utf-8")), runtime_only=True
            ), (
                f"{allowed} is allowed to import httpx and does not, so these tests prove "
                f"nothing about the modules they exclude"
            )

    def test_the_typing_only_exemption_is_not_a_hole(self):
        """The exemption must cover **annotations**, not calls.

        ``midataworks_review.py`` imports ``httpx`` under ``TYPE_CHECKING`` for one
        annotation. If that module ever grew a runtime import, this is the assertion that
        would not notice - so the companion assertion is on the *calls* it makes
        (``test_the_review_port_implementation_goes_through_the_wrapper``), and this one
        pins that the exemption is scoped to typing-only blocks rather than to the module.
        """
        source = (SRC / "clients" / "midataworks_review.py").read_text("utf-8")
        tree = ast.parse(source)
        assert "httpx" in imported_names(tree), "the fixture for this test moved"
        assert "httpx" not in imported_names(tree, runtime_only=True)

    def test_the_review_port_implementation_goes_through_the_wrapper(self):
        """F3's port is implemented at the single call site rather than beside it.

        It may *type-annotate* against ``httpx`` for a transport it passes through, which is
        why the check is on the **call**: it must construct a ``MiDataworksClient`` and must
        not construct a transport or a client of its own.
        """
        tree = ast.parse((SRC / "clients" / "midataworks_review.py").read_text("utf-8"))
        constructed = {
            node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
        }
        assert "MiDataworksClient" in constructed
        assert "Client" not in constructed and "AsyncClient" not in constructed


class TestMiForgeWritesNoDatasetVersionManifest:
    """FR-010.20. **miForge is a consumer of this contract.**

    The mirror image of what miDataworks pins in its own repository (*"no module in ``src/``
    may write either path: the script is the only writer"*). A manifest written here would be
    miForge claiming to have produced a miDataworks version - provenance as a lie, which is
    the one thing the whole contract exists to prevent.
    """

    def _writes_to_manifest(self, tree: ast.AST) -> list[int]:
        """Lines where a write call's target mentions the instance filename.

        Looks for the *combination*: a call to a writing function whose arguments or whose
        receiver chain mention the filename. Naming the constant is fine; writing to it is
        not.
        """
        writers = {"write_bytes", "write_text", "write_canonical", "open", "dump", "save"}
        found: list[int] = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            if name not in writers:
                continue
            source = ast.dump(node)
            if MANIFEST_INSTANCE in source or "midataworks-dataset-version" in source:
                found.append(node.lineno)
        return found

    def test_no_module_writes_the_instance_manifest(self):
        offenders = {
            relative(path): lines
            for path in modules()
            if (lines := self._writes_to_manifest(ast.parse(path.read_text("utf-8"))))
        }
        assert offenders == {}, (
            f"{offenders} writes a {MANIFEST_INSTANCE}. miForge consumes this contract and "
            f"does not produce it (FR-010.20): miDataworks is normative for it "
            f"(BRD-03 R-03.47), and a manifest written here would claim provenance miForge "
            f"cannot have."
        )

    def test_no_module_constructs_a_dataset_version_document(self):
        """The other half: a dict literal with ``kind: "midataworks.dataset-version"``.

        Writing the file is the obvious form; building the document and handing it to
        something else to write is the form that would pass the test above.
        """
        offenders: dict[str, list[int]] = {}
        for path in modules():
            tree = ast.parse(path.read_text("utf-8"))
            lines = [
                node.lineno
                for node in ast.walk(tree)
                if isinstance(node, ast.Dict)
                and any(
                    isinstance(key, ast.Constant) and key.value == "kind"
                    for key in node.keys
                    if key is not None
                )
                and any(
                    isinstance(value, ast.Constant) and value.value == "midataworks.dataset-version"
                    for value in node.values
                )
            ]
            if lines:
                offenders[relative(path)] = lines
        assert offenders == {}, offenders


class TestNoSiblingImport:
    def test_nothing_imports_the_sibling_packages(self):
        """ADR-001, restated for the feature most likely to be tempted.

        The siblings are checked out side by side on this machine, so an absolute
        ``import midataworks`` **resolves locally and passes every manual review**. The
        foundation's guard covers the whole tree; this asserts it for this feature's own
        modules, so a failure points at the right place.
        """
        banned = {"millm", "mistudio", "midataworks"}
        mine = [
            path
            for path in modules()
            if relative(path).startswith(("clients/", "services/parts/", "services/review/"))
        ]
        assert mine, "no F10 modules found; this test would pass vacuously"
        offenders = {
            relative(path): sorted(imported_names(ast.parse(path.read_text("utf-8"))) & banned)
            for path in mine
            if imported_names(ast.parse(path.read_text("utf-8"))) & banned
        }
        assert offenders == {}, offenders


@pytest.mark.parametrize(
    "form", ["import midataworks", "import midataworks.api", "from midataworks import x"]
)
def test_the_scanner_sees_every_import_form(form: str, tmp_path: Path):
    """The guard's own self-test. **A scanner that cannot see one form is a guard with a
    hole**, and the hole is invisible until someone writes code through it."""
    module = tmp_path / "probe.py"
    module.write_text(form + "\n", encoding="utf-8")
    assert "midataworks" in imported_names(ast.parse(module.read_text("utf-8")))
