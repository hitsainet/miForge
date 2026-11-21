"""The static guards for F2's tree, each with a self-test  (AC-002.23; ADR-001, 010, 022).

**Every scanner here is AST-based and every one carries a self-test that plants the
violation and requires it to be caught.** A string scan reports a module's own prose as the
violation - three of F1's structural tests did exactly that and were rewritten - and a
scanner that has never been seen catching anything is not known to work. That is the same
rule as foundation task 10.2's six controls, applied to this feature's four guards.

What is asserted:

* **no ``httpx``** - generation goes through ``providers/`` and miDataworks goes through
  F10's single call site (ADR-010);
* **no import of ``training/hotupdate/``** - §5.10 is fenced and the standalone test runs
  with that package absent (ADR-022);
* **no sibling import** - ``millm``, ``mistudio``, ``midataworks`` resolve locally because
  the repositories sit side by side (ADR-001);
* **no hard-coded GPU site** - the card is pinned by UUID at submit (ADR-008);
* **no Celery task in this feature** - in-process assembly is called synchronously by the
  loop in the ``train`` worker, and the placement is the caller's (F6 AMB-1);
* **no manifest write** - F2 contributes one value to a manifest and proposes no field;
* **``is_admitted`` is the only route to admitted** - the contract's own rule.
"""

from __future__ import annotations

import ast
from pathlib import Path

SRC = Path(__file__).resolve().parents[2] / "src"

#: Every module this feature owns. Listed by directory rather than by name so a module
#: added later is covered without anyone remembering to add it here.
F2_PATHS = (
    SRC / "training" / "signals",
    SRC / "training" / "objectives",
)
F2_FILES = (
    SRC / "services" / "signal_source_service.py",
    SRC / "api" / "v1" / "endpoints" / "signal_sources.py",
    SRC / "api" / "v1" / "endpoints" / "signal_batches.py",
)


def f2_modules() -> list[Path]:
    found = [path for directory in F2_PATHS for path in directory.rglob("*.py")]
    found.extend(F2_FILES)
    assert len(found) > 12, f"the guard is scanning {len(found)} files; it should be the feature"
    return found


def imported_names(tree: ast.AST) -> set[str]:
    """Every module name an AST imports, in **either** form.

    ``import x.y``, ``from x.y import z`` and ``from . import z`` all land here, because a
    guard that only sees one form is a guard with a documented way around it.
    """
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def offenders(predicate) -> list[str]:  # type: ignore[no-untyped-def]
    return [
        str(path.relative_to(SRC))
        for path in f2_modules()
        if predicate(ast.parse(path.read_text("utf-8")))
    ]


class TestNoHttpClientInThisFeature:
    def test_no_module_imports_httpx(self) -> None:
        """ADR-010: one outbound call site per service, and miDataworks' is F10's."""
        assert (
            offenders(
                lambda tree: any(
                    name == "httpx" or name.startswith("httpx.") for name in imported_names(tree)
                )
            )
            == []
        )

    def test_the_scanner_sees_each_import_form(self) -> None:
        for source in ("import httpx", "import httpx as h", "from httpx import AsyncClient"):
            tree = ast.parse(source)
            assert any(
                name == "httpx" or name.startswith("httpx.") for name in imported_names(tree)
            ), source


class TestTheHotUpdateModeIsFenced:
    def test_no_module_imports_training_hotupdate(self) -> None:
        """ADR-022: the standalone test runs with that package absent, so a core-path import
        of it fails the suite rather than the mode."""
        assert (
            offenders(lambda tree: any("hotupdate" in name for name in imported_names(tree))) == []
        )

    def test_the_scanner_sees_the_import(self) -> None:
        tree = ast.parse("from training.hotupdate import heartbeat")
        assert any("hotupdate" in name for name in imported_names(tree))


class TestNoSiblingImport:
    def test_no_module_imports_a_sibling_package(self) -> None:
        """The siblings are checked out side by side, so an absolute import resolves."""
        siblings = ("millm", "mistudio", "midataworks")
        assert (
            offenders(
                lambda tree: any(name.split(".")[0] in siblings for name in imported_names(tree))
            )
            == []
        )

    def test_the_scanner_sees_each_sibling(self) -> None:
        for source in ("import millm", "from mistudio import x", "import midataworks.api"):
            tree = ast.parse(source)
            assert any(
                name.split(".")[0] in ("millm", "mistudio", "midataworks")
                for name in imported_names(tree)
            ), source


class TestNoHardCodedGpuSite:
    #: A device index, a bus address or a probe. ADR-008: pin by **UUID resolved at submit
    #: from live ``nvidia-smi -L``** - indices inverted once on this machine, and a copied
    #: bus address unbound the root NVMe.
    FORBIDDEN = ("cuda:0", "cuda:1", "nvidia-smi", "/dev/nvidia", "0000:")

    def _constants(self, tree: ast.AST) -> set[str]:
        """Every string constant **except a docstring**.

        Docstrings are excluded because this feature's modules *explain* the rule they
        follow - "no ``nvidia-smi`` call, no index, no bus address" - and a scanner that
        read that prose as the violation would report the comment that exists to prevent it.
        That is the exact trap three of F1's structural tests fell into before they were
        rewritten with ``ast``, and the self-test below plants a real assignment rather than
        a sentence.
        """
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef)
            and node.body
            and isinstance(node.body[0], ast.Expr)
            and isinstance(node.body[0].value, ast.Constant)
        }
        return {
            node.value
            for node in ast.walk(tree)
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        }

    def test_no_module_names_a_device_index_or_a_bus_address(self) -> None:
        assert (
            offenders(
                lambda tree: any(
                    any(bad in value for bad in self.FORBIDDEN) for value in self._constants(tree)
                )
            )
            == []
        )

    def test_the_scanner_sees_a_planted_site(self) -> None:
        tree = ast.parse('DEVICE = "cuda:0"')
        assert any(any(bad in value for bad in self.FORBIDDEN) for value in self._constants(tree))


class TestThisFeatureDeclaresNoCeleryTask:
    """F6's AMB-1, structurally.

    In-process assembly is called **synchronously by the loop, in the ``train`` worker**:
    ADR-009's default provider generates from parameters that exist only there, and a
    ``generate`` worker would need a second copy of the parameter set on the leased card -
    against ADR-008's one-run-per-card lease and BRD §6's budget. A task declared here would
    make the placement F2's decision rather than its caller's, and would hand a
    ``--pool=solo -c 1`` worker a cross-queue ``.get()``, which deadlocks rather than queues.

    F3 states the same rule for the same reason one directory over.
    """

    def _decorated(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if not isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                continue
            for decorator in node.decorator_list:
                target = decorator.func if isinstance(decorator, ast.Call) else decorator
                name = getattr(target, "attr", None) or getattr(target, "id", None)
                if name in {"task", "shared_task"}:
                    return True
        return False

    def test_no_module_declares_a_task(self) -> None:
        assert offenders(self._decorated) == []

    def test_no_module_imports_the_celery_app(self) -> None:
        assert (
            offenders(lambda tree: any("celery" in name.lower() for name in imported_names(tree)))
            == []
        )

    def test_the_scanner_sees_both_decorator_forms(self) -> None:
        for source in (
            "@shared_task\ndef f(): ...",
            "@app.task(name='x')\ndef f(): ...",
        ):
            assert self._decorated(ast.parse(source)), source


class TestNoManifestIsWrittenHere:
    """AC-002.17 / FR-002.29. F2 contributes **one** value to a checkpoint manifest - the
    required ``signal_source`` - and proposes no field.

    *Adding a manifest field "to record what it was trained on" is the exact shape of this
    project's recurring defect, and it was caught once already in R-02.35's own drafting.*
    Part references live in the batch record and the run log.
    """

    def test_no_module_writes_a_manifest(self) -> None:
        def writes(tree: ast.AST) -> bool:
            constants = TestNoHardCodedGpuSite()._constants(tree)
            if any(value.endswith("manifest.json") for value in constants):
                return True
            return any(
                name.endswith("write_canonical") or "checkpoints" in name
                for name in imported_names(tree)
            )

        assert offenders(writes) == []

    def test_the_scanner_sees_a_planted_write(self) -> None:
        tree = ast.parse('from core.canonical import write_canonical\nP = "manifest.json"')
        constants = TestNoHardCodedGpuSite()._constants(tree)
        assert any(value.endswith("manifest.json") for value in constants)
        assert any(name.endswith("write_canonical") for name in imported_names(tree))


class TestAdmissionGoesThroughTheContract:
    """``schemas.selector.is_admitted`` is **the only place admission is decided**.

    The contract says so in terms: *"every consumer calls this rather than comparing to a
    value itself, so that 'no path from undecided to admitted' is enforced in one place that
    can be tested."* A comparison written here would be a second gate that a later edit
    could loosen without touching the contract.
    """

    def _compares_to_accept(self, tree: ast.AST) -> bool:
        for node in ast.walk(tree):
            if not isinstance(node, ast.Compare):
                continue
            for side in [node.left, *node.comparators]:
                if isinstance(side, ast.Attribute) and side.attr == "ACCEPT":
                    return True
                if isinstance(side, ast.Constant) and side.value == "accept":
                    return True
        return False

    def test_no_module_compares_a_verdict_to_accept(self) -> None:
        assert offenders(self._compares_to_accept) == []

    def test_the_partition_calls_is_admitted(self) -> None:
        """The positive half: the one gate is not merely unduplicated, it is **used**."""
        from training.signals import selection

        source = Path(selection.__file__).read_text("utf-8")
        tree = ast.parse(source)
        called = {
            node.func.id
            for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert "is_admitted" in called

    def test_the_scanner_sees_both_comparison_forms(self) -> None:
        for source in (
            "x = verdict.value is VerdictValue.ACCEPT",
            'x = verdict.value == "accept"',
        ):
            assert self._compares_to_accept(ast.parse(source)), source


class TestTheFeatureScansItself:
    def test_the_file_list_covers_every_module_this_feature_owns(self) -> None:
        """A guard that scans the wrong directory is a guard that passes for free."""
        scanned = {str(path.relative_to(SRC)) for path in f2_modules()}
        for expected in (
            "training/signals/selection.py",
            "training/signals/batch_record.py",
            "training/objectives/sft.py",
            "services/signal_source_service.py",
            "api/v1/endpoints/signal_sources.py",
        ):
            assert expected in scanned
