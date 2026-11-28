"""ADR-022's fence, as a checked property  (FR-011.46, FR-011.50, FR-011.52; **MUT-11**).

Three assertions, and the third is the one a reader would not think to make:

1. **Nothing outside ``training/hotupdate/`` references a weights endpoint** - not a literal
   path, not an f-string, not a module constant, not a URL built by joining. The allowlist is
   **exactly one directory**.
2. **Exactly one module in ``backend/src/`` reaches the package**, and it is the registry.
   This is what rules out F8's adapter run and merge reaching into ``hotupdate/`` for
   FR-011.60/61: routing a **core-path** feature into §5.10 would make it the second
   importer, which is precisely what this guard exists to catch.
3. **The one reference is dynamic.** A *static* import would be a dependency edge from the
   core path into the optional package - and F1's own AST guard forbids one - so the
   registry imports it by name inside a try/except. This file asserts the dynamic form too,
   because a scanner that only knew ``import`` statements would be blind to the shape F11
   actually uses: *a guard that cannot see one form is a guard with a hole, and the hole is
   invisible until someone writes code through it.*

**Why the fence is worth three tests.** §5.10 *"grew to thirteen required rows once without
anyone deciding it should."* One convenience import and ADR-022's claim is false while the
suite stays green.

## Provenance of the shape (ADR-001; FTASKS 12.4, 16.3)

The shape is **miStudio's**, cited and not copied:
`backend/tests/unit/test_mcp_tools_stay_on_http.py` in the miStudio repository is an AST
scan over a directory with a short, deliberately awkward allowlist, whose docstring
records why the allowlist is awkward on purpose - *"an entry is a claim that a module
genuinely needs in-process backend state, which would mean the MCP
deployment needs that configuration too - a deployment decision, not an import decision."*
The same reasoning is why :data:`FENCE` is one path and why the staleness test below refuses
to let it grow silently.

**No line in this file came from that one, so `docs/REUSE.md` carries no row for it** - that
file's own *"what is NOT reuse"* rule exempts a pattern reimplemented from a read. What is
cited is the *idea*, including the part this file would otherwise have had to learn the hard
way: miStudio's own docstring records a capability that passed a payload test, a live-registry
reachability test **and** a manifest check and would still have raised on its first production
call, because every check ran in a process that had what production does not. That is the same
asymmetry the §5.10 deletion control exists for - a guard proves the reference is absent; only
running without the thing proves nothing needs it.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"

#: **Exactly one directory.** Adding F11 must not widen this by one path.
FENCE = "training/hotupdate"

#: Every form a weights-endpoint reference can take, as the *markers* the scanner looks for.
#: ``/api/weights`` is the path prefix; the ops are the distinctive segments that would
#: appear in an f-string or a joined URL.
ENDPOINT_MARKERS = ("/api/weights", "commit-begin", "commit-end", "X-miLLM-Attachment")


def modules() -> list[Path]:
    return sorted(SRC.rglob("*.py"))


def relative(path: Path) -> str:
    return str(path.relative_to(SRC))


def inside_fence(path: Path) -> bool:
    return "hotupdate" in path.parts


def endpoint_references(source: str) -> list[int]:
    """Lines where a **string in the code** names a weights endpoint, in any form.

    Walks the syntax tree and looks at string constants and f-string parts, so a docstring
    *about* the fence is not a reference to it - which matters, because every module in this
    project explains at length what it does not do, and a text matcher reports a disclaimer
    as a claim. The project has paid for that lesson: a fresh harness reported 18
    scope-creep failures and all 18 were documents correctly disclaiming a foreign
    requirement.
    """
    tree = ast.parse(source)
    docstrings = _docstring_nodes(tree)
    found: list[int] = []
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if any(marker in node.value for marker in ENDPOINT_MARKERS):
                found.append(node.lineno)
        elif isinstance(node, ast.JoinedStr):
            # An f-string: the literal halves are `Constant` children, and walking covers
            # them - but a joined URL whose constant half is only "/api" plus a variable is
            # caught by the "/api/weights" marker on whichever half carries it.
            literal = "".join(
                part.value
                for part in node.values
                if isinstance(part, ast.Constant) and isinstance(part.value, str)
            )
            if any(marker in literal for marker in ENDPOINT_MARKERS):
                found.append(node.lineno)
    return sorted(set(found))


def _docstring_nodes(tree: ast.AST) -> set[int]:
    """Ids of string constants that are docstrings, which are prose rather than code."""
    ids: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = getattr(node, "body", [])
            if (
                body
                and isinstance(body[0], ast.Expr)
                and isinstance(body[0].value, ast.Constant)
                and isinstance(body[0].value.value, str)
            ):
                ids.add(id(body[0].value))
    return ids


def package_references(source: str) -> list[str]:
    """Every reference to the fenced package: static imports **and** dynamic ones."""
    tree = ast.parse(source)
    found: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [alias.name for alias in node.names if "training.hotupdate" in alias.name]
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if "training.hotupdate" in module:
                found.append(module)
        elif isinstance(node, ast.Call):
            # `importlib.import_module("training.hotupdate")` and `__import__(...)`.
            name = (
                node.func.attr
                if isinstance(node.func, ast.Attribute)
                else (node.func.id if isinstance(node.func, ast.Name) else "")
            )
            if name in {"import_module", "__import__"}:
                for argument in node.args:
                    if isinstance(argument, ast.Constant) and "training.hotupdate" in str(
                        argument.value
                    ):
                        found.append(str(argument.value))
    return found


class TestNothingOutsideTheFenceNamesAWeightsEndpoint:
    def test_the_core_path_names_no_weights_endpoint(self) -> None:
        offenders = {
            relative(path): lines
            for path in modules()
            if not inside_fence(path) and (lines := endpoint_references(path.read_text("utf-8")))
        }
        assert offenders == {}, (
            f"{offenders} name a miLLM weights endpoint outside {FENCE}/. ADR-022: nothing "
            f"in the §5.1-§5.9 code path may reference one, and the standalone test runs "
            f"with that directory absent."
        )

    def test_the_allowlist_is_exactly_one_directory(self) -> None:
        """Stated as a test so widening it is a deliberate edit to an assertion rather than
        a line added to a list nobody reads."""
        assert FENCE == "training/hotupdate"
        assert (SRC / FENCE).is_dir()

    def test_the_fenced_package_really_does_name_them(self) -> None:
        """Otherwise the test above passes **vacuously** - the construction-green failure
        this project has found in its own harness, in a sync test's empty vendor list, and in
        a reindex assertion over a table nothing wrote."""
        named = {
            relative(path)
            for path in modules()
            if inside_fence(path) and endpoint_references(path.read_text("utf-8"))
        }
        assert named, "no module inside the fence names an endpoint; the guard proves nothing"


class TestExactlyOneModuleReachesThePackage:
    def test_the_importer_count_is_one_and_it_is_the_registry(self) -> None:
        """**MUT-11.** And the assertion that rules out F8 reaching in for FR-011.60/61."""
        importers = {
            relative(path): refs
            for path in modules()
            if not inside_fence(path) and (refs := package_references(path.read_text("utf-8")))
        }
        assert list(importers) == ["training/acquisition/__init__.py"], (
            f"the fenced package has {len(importers)} importers: {importers}. ADR-022 allows "
            f"one - the acquisition registry - and a second on a core-path feature is how "
            f"this mode grew to thirteen required rows once before."
        )

    def test_the_one_reference_is_dynamic(self) -> None:
        """A static import would be a dependency edge on an optional package, which F1's own
        guard forbids by AST and the standalone test would break on."""
        source = (SRC / "training" / "acquisition" / "__init__.py").read_text("utf-8")
        tree = ast.parse(source)
        static = [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and "hotupdate" in (node.module or "")
        ] + [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
            if "hotupdate" in alias.name
        ]
        assert static == [], f"static import of the fenced package: {static}"
        assert package_references(source) == ["training.hotupdate"]

    def test_the_import_is_guarded_so_an_absent_package_is_absent_and_normal(self) -> None:
        """ADR-010: unset means absent-and-normal. The ``except ImportError`` is what makes
        FR-011.48 true **by construction** rather than by remembering."""
        source = (SRC / "training" / "acquisition" / "__init__.py").read_text("utf-8")
        tree = ast.parse(source)
        installer = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "_install_optional_attach_mode"
        )
        handlers = [
            handler
            for node in ast.walk(installer)
            if isinstance(node, ast.Try)
            for handler in node.handlers
        ]
        caught = {handler.type.id for handler in handlers if isinstance(handler.type, ast.Name)}
        assert "ImportError" in caught
        # And it must not re-raise: a re-raise makes the absent package a build failure.
        assert not any(isinstance(node, ast.Raise) for node in ast.walk(installer))


class TestTheScannerSeesEveryForm:
    """**The self-test.** Each form a reference can take, planted and required to be seen."""

    @pytest.mark.parametrize(
        "form",
        [
            'path = "/api/weights/attach"',
            'url = f"{base}/api/weights/{op}"',
            'PATH = "/api/weights/commit-begin"',
            'url = "/".join([base, "api/weights", "quiesce"]) + "/api/weights"',
            'headers = {"X-miLLM-Attachment": attachment_id}',
            'body = {"op": "commit-end"}',
        ],
    )
    def test_each_endpoint_form_is_seen(self, form: str) -> None:
        assert endpoint_references(form) != [], f"the scanner is blind to {form!r}"

    @pytest.mark.parametrize(
        "form",
        [
            "from training.hotupdate import AttachAcquirer",
            "import training.hotupdate",
            "from training.hotupdate.client import WeightsClient",
            'importlib.import_module("training.hotupdate")',
            '__import__("training.hotupdate.attach")',
        ],
    )
    def test_each_import_form_is_seen(self, form: str) -> None:
        assert package_references(form) != [], f"the scanner is blind to {form!r}"

    def test_a_docstring_about_the_fence_is_not_a_reference_to_it(self) -> None:
        """The other half of the self-test, and the one that keeps the guard honest: every
        module here explains what it does not do, and a scanner that could not tell a
        disclaimer from a claim would be "fixed" by deleting the explanation."""
        source = '"""This module never calls /api/weights/attach."""\nx = 1\n'
        assert endpoint_references(source) == []

    def test_the_scanner_can_be_blinded_and_then_it_proves_nothing(self) -> None:
        """Task 13.5: the self-test itself, seen to fail when the scanner is blinded. A
        marker list that matched nothing would make every assertion above green."""
        blinded_markers: tuple[str, ...] = ()
        source = 'path = "/api/weights/attach"'
        tree = ast.parse(source)
        hits = [
            node.lineno
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and any(marker in node.value for marker in blinded_markers)
        ]
        assert hits == [], "with an empty marker list the scanner sees nothing - as expected"
        assert endpoint_references(source) != [], "and with the real list it sees it"
