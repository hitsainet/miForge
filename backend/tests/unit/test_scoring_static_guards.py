"""Structural guards over F4's packages  (FR-004.3, 20; ADR-001, ADR-008, ADR-022; M-8, M-10).

Each of these is a property that a careful reader could verify today and that would go wrong
silently tomorrow. A guard that cannot see what it forbids is worse than none, so the two
scanners here carry self-tests.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[2] / "src"
SCORING = SRC / "services" / "scoring"


def modules() -> list[Path]:
    return sorted(path for path in SCORING.glob("*.py") if path.name != "__init__.py")


def imports_of(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.append(node.module)
    return names


class TestNoSiblingImportAnywhere:
    """ADR-001. The siblings are checked out side by side on this workstation, so an
    absolute import **resolves locally** and would pass every test here while being
    unbuildable anywhere else."""

    @pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
    def test_no_module_imports_a_sibling(self, path: Path) -> None:
        offenders = [
            name
            for name in imports_of(path)
            if name.split(".")[0] in ("millm", "mistudio", "midataworks")
        ]
        assert offenders == [], f"{path.name} imports {offenders}"


class TestF4NeverCallsMiDataworks:
    """FR-004.12's other half: **F10 owns the single outbound call site** (its FR-010.1).

    F4 reads imported part rows and F10's **validated** view of a manifest. It never reaches
    the service, which is why there is nothing here to mock - *mock a sibling's HTTP surface,
    never its code; there is nothing to import.*
    """

    @pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
    def test_no_module_imports_a_client(self, path: Path) -> None:
        offenders = [name for name in imports_of(path) if name.split(".")[0] == "clients"]
        assert offenders == [], (
            f"{path.name} reaches a sibling client at {offenders}. F10's "
            f"`clients/midataworks.py` is the one outbound call site; F4 reads rows."
        )

    def test_f4_reads_f10s_validator_rather_than_parsing_a_manifest(self) -> None:
        """One validator, F10's. A second parser here would be a second contract."""
        source = (SCORING / "entrypoint.py").read_text(encoding="utf-8")
        assert "from services.parts.contract_validator import" in source

    def test_no_module_declares_a_second_dataset_schema_validator(self) -> None:
        offenders = [
            path.name
            for path in modules()
            if "Draft202012Validator" in path.read_text(encoding="utf-8")
        ]
        assert offenders == [], f"{offenders} validate a contract F10 already validates"


class TestNothingOnTheCorePathReferencesAnActivationSource:
    """FR-004.35, ADR-022. R-02.19 is **opt-in throughout**.

    The capability is resolved at runtime from configuration, never imported by the scoring
    core, so the standalone harness runs with the module **absent from the tree** - which is
    what makes AC20 a fact rather than an assertion.
    """

    @pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
    def test_no_module_imports_an_activation_client(self, path: Path) -> None:
        offenders = [name for name in imports_of(path) if "activation" in name]
        assert offenders == [], f"{path.name} imports {offenders} on the core path"

    def test_no_module_names_a_sibling_weights_endpoint(self) -> None:
        """§5.10 is fenced: *a ``/api/weights/...`` path anywhere else is the mode leaking
        into the product.*"""
        for path in modules():
            assert "/api/weights/" not in path.read_text(encoding="utf-8"), path.name

    def test_activations_is_not_a_declared_provider_capability(self) -> None:
        """R-02.19's own amended text, and the implementation of DEF-10's closure.

        *A declared-but-unused capability slot is how the next reader concludes the provider
        is the route* - which is exactly what DEF-10 reported.
        """
        from providers import registry

        assert "activations" not in registry.KNOWN_CAPABILITIES
        for row in registry.list_rows():
            assert "activations" not in row.capabilities


class TestNoHardCodedGpuSite:
    """ADR-008: **by UUID**, resolved at submit from live ``nvidia-smi -L``.

    An index inverted once on this cluster and a copied bus address unbound the root NVMe.
    F4 adds **no** entry to the ratchet's allowlist.
    """

    @pytest.mark.parametrize("path", modules(), ids=lambda p: p.name)
    def test_no_device_index_or_bus_address(self, path: Path) -> None:
        import re

        source = path.read_text(encoding="utf-8")
        offenders = re.findall(r"cuda:\d|0000:[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d", source)
        assert offenders == [], f"{path.name} hard-codes a GPU site: {offenders}"

    def test_the_handle_reports_a_uuid_not_an_index(self) -> None:
        from providers.fakes import FakeModelHandle

        assert FakeModelHandle().device_uuid() == "GPU-fake-0000"


class TestHeldOutDoubleBindingIsRefusedAtSubmission:
    """E-16, M-8. **Training on the test set raises the test score for the wrong reason**,
    and the rise is indistinguishable from success on every surface this feature produces.

    F10's ``reader.splits_for`` keeps marked splits out of training material at **import**;
    this is the **composition-time** half, because a part can be imported cleanly and then
    bound twice.
    """

    def test_an_overlap_refuses_and_names_the_part(self) -> None:
        from core.errors import Refusal
        from services.scoring import entrypoint

        with pytest.raises(Refusal) as caught:
            entrypoint.check_submission(
                test_part_versions=["test_set/general@1"],
                retention_part_versions=["retention_set/general@1"],
                signal_part_versions=["corpus/general@2", "test_set/general@1"],
                retention_set_present=True,
                retention_floor=0.72,
            )
        assert caught.value.code == "held_out_set_used_for_training"
        assert caught.value.details["parts"] == ["test_set/general@1"]

    def test_a_retention_part_bound_as_a_signal_also_refuses(self) -> None:
        from core.errors import Refusal
        from services.scoring import entrypoint

        with pytest.raises(Refusal):
            entrypoint.check_submission(
                test_part_versions=[],
                retention_part_versions=["retention_set/general@1"],
                signal_part_versions=["retention_set/general@1"],
                retention_set_present=True,
                retention_floor=0.72,
            )

    def test_no_overlap_passes(self) -> None:
        from services.scoring import entrypoint

        entrypoint.check_submission(
            test_part_versions=["test_set/general@1"],
            retention_part_versions=["retention_set/general@1"],
            signal_part_versions=["corpus/general@2"],
            retention_set_present=True,
            retention_floor=0.72,
        )

    def test_a_missing_retention_set_refuses_before_the_run_starts(self) -> None:
        """E-12, R-02.48. *A run that cannot proceed says so before it starts.*"""
        from core.errors import Refusal
        from services.scoring import entrypoint

        with pytest.raises(Refusal) as caught:
            entrypoint.check_submission(
                test_part_versions=[],
                retention_part_versions=[],
                signal_part_versions=[],
                retention_set_present=False,
                retention_floor=None,
            )
        assert caught.value.code == "retention_set_missing"

    def test_a_retention_set_with_no_floor_refuses(self) -> None:
        from core.errors import Refusal
        from services.scoring import entrypoint

        with pytest.raises(Refusal) as caught:
            entrypoint.check_submission(
                test_part_versions=[],
                retention_part_versions=["retention_set/general@1"],
                signal_part_versions=[],
                retention_set_present=True,
                retention_floor=None,
            )
        assert caught.value.code == "retention_floor_missing"


def _operates_on(path: Path, table: str) -> bool:
    """Whether a migration **operates on** ``table``, as opposed to merely mentioning it.

    **Retuned 2026-10-14, because the substring form flagged correct work.** This read
    ``if "mf_scores" in path.read_text(...)``, and migration ``0016`` went red for a
    *comment* citing F4's precedent — *"`mf_scores.run_id` is a plain column for exactly this
    reason"* — written to explain why F6's chain table carries no cross-metadata foreign key.
    The guard's intent is right and load-bearing (two revisions touching one table is how an
    Alembic chain forks into multiple heads), but a citation is not an operation, and a
    checker that fires on documentation gets switched off.

    That is the fifth time in this project a fresh checker disagreed with careful work and
    **the work was right** — four rounds on the review harness, now this. So the comparison
    is made against the code: the module docstring and whole-line comments are removed before
    searching, which is exactly the distinction between describing a table and touching one.
    """
    import ast

    source = path.read_text(encoding="utf-8")
    try:
        docstring = ast.get_docstring(ast.parse(source))
    except SyntaxError:  # pragma: no cover - a migration that does not parse fails elsewhere
        docstring = None
    if docstring:
        source = source.replace(docstring, "")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("#"))
    return table in code


def _creates(path: Path, table: str) -> bool:
    """Whether this revision's ``upgrade`` **creates** ``table``, by AST not by substring.

    Mentioning a table and creating it are different facts, and conflating them made the
    guard below fail the moment a later revision legitimately altered a column type
    (``0018``, which widens three ``mf_scores`` JSON columns to JSONB on PostgreSQL). The
    fork this guards against is **two revisions creating one table**; an alter is the
    normal way a schema moves forward and must not read as a fork.
    """
    import ast

    try:
        tree = ast.parse(path.read_text(encoding="utf-8"))
    except SyntaxError:  # pragma: no cover - a migration that does not parse fails elsewhere
        return False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        name = func.attr if isinstance(func, ast.Attribute) else getattr(func, "id", None)
        if name != "create_table" or not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and first.value == table:
            return True
    return False


class TestTheTablesAreNotOnTheBaseMetadata:
    """Foundation 4.2's ratchet asserts ``Base.metadata`` holds **exactly** the base-schema
    tables, with the message *"a table only one feature reads belongs in that feature's own
    migration"*. F10's four tables are the worked example and F4 follows it."""

    def test_f4s_tables_are_on_their_own_metadata(self) -> None:
        from db.base import Base
        from services.scoring.tables import SCORE_ITEMS, SCORES, metadata

        assert set(metadata.tables) == {"mf_scores", "mf_score_items"}
        assert "mf_scores" not in Base.metadata.tables
        assert SCORES.metadata is metadata and SCORE_ITEMS.metadata is metadata

    def test_they_carry_the_mf_prefix(self) -> None:
        from services.scoring.tables import metadata

        assert all(name.startswith("mf_") for name in metadata.tables)

    def test_they_carry_the_shared_naming_convention(self) -> None:
        """Unnamed CHECK and UNIQUE constraints cannot be dropped by name, which makes a
        migration reversible on PostgreSQL and irreversible on the SQLite the unit tests run
        against."""
        from db.base import NAMING_CONVENTION
        from services.scoring.tables import metadata

        assert metadata.naming_convention == NAMING_CONVENTION

    def test_the_scoring_migration_exists_and_is_the_coordinators(self) -> None:
        """The revision graph is **shared state** and the coordinator owns it.

        This asserted the *absence* of a revision, because two revisions chaining off one
        parent make ``alembic upgrade head`` fail for everyone on someone else's clean
        clone - so F4 handed the DDL over instead of writing one. **Flipped 2026-10-14
        when the coordinator applied it as ``0013_scoring``**, which is what the original
        said to do: *"delete it then, the tables come from the migration like everyone
        else's."*

        It now asserts the opposite and the stronger thing: the tables **are** in a
        migration, there is **exactly one** such revision, and it chains off the head
        rather than the base. A second one would reintroduce the multiple-heads failure the
        original guarded against.
        """
        versions = SRC.parent / "alembic" / "versions"
        mentioning = sorted(
            path.name for path in versions.glob("*.py") if _operates_on(path, "mf_scores")
        )
        assert mentioning, (
            "no revision mentions mf_scores at all, so the positive control for the "
            "assertion below is absent and it would hold vacuously"
        )
        owning = sorted(path.name for path in versions.glob("*.py") if _creates(path, "mf_scores"))
        assert len(owning) == 1, (
            f"expected exactly one revision CREATING mf_scores, found {owning} (revisions "
            f"mentioning it: {mentioning}). Two revisions creating one table is how a chain "
            f"forks; a later revision altering it is normal and is not a fork."
        )
        body = (versions / owning[0]).read_text(encoding="utf-8")
        assert (
            'down_revision: str | None = "0012_save_req"' in body
        ), "the scoring revision must chain off the then-head, not off the base"

    def test_the_effective_ceiling_is_f3s_when_f3s_is_lower(self) -> None:
        from services.scoring.settings import ScoringSettings, effective_max_prompts
        from services.selectors.settings import get_selector_settings

        f3_bound = get_selector_settings().evaluate_max_completions
        assert effective_max_prompts(ScoringSettings(scoring_max_prompts=2000)) == f3_bound

    def test_it_is_f4s_when_f4s_is_lower(self) -> None:
        from services.scoring.settings import ScoringSettings, effective_max_prompts

        assert effective_max_prompts(ScoringSettings(scoring_max_prompts=10)) == 10

    def test_a_set_above_the_ceiling_refuses_BEFORE_generating(self) -> None:
        """R-02.48's rule applied to the pass: *says so before it starts, not partway
        through* - and here "partway through" meant after paying for every completion."""
        from core.errors import Refusal
        from services.scoring import sets
        from services.scoring.settings import effective_max_prompts

        ceiling = effective_max_prompts()
        reference = sets.SetReference(rendered="test_set/big@1", structured=None)
        with pytest.raises(Refusal) as caught:
            sets.resolve(
                "test",
                reference,
                [{"prompt": f"p{i}"} for i in range(ceiling + 1)],
                max_prompts=ceiling,
            )
        assert caught.value.code == "evaluation_set_too_large"
        assert caught.value.details["limit"] == ceiling
        assert "F3's synchronous evaluation bound" in caught.value.message

    def test_a_seeded_sample_within_the_ceiling_is_the_documented_way_through(self) -> None:
        """The refusal's own advice, asserted so it is actually true."""
        from services.scoring import sets
        from services.scoring.settings import effective_max_prompts

        ceiling = effective_max_prompts()
        reference = sets.SetReference(rendered="test_set/big@1", structured=None)
        resolved = sets.resolve(
            "test",
            reference,
            [{"prompt": f"p{i}", "group": f"g{i % 4}"} for i in range(ceiling * 3)],
            sample=sets.SampleConfig(size=ceiling, seed=11),
            max_prompts=ceiling,
        )
        assert resolved.n == ceiling
        assert resolved.population == ceiling * 3


class TestTheDDLCompilesForBOTHEngines:
    """**The blind spot ``support/db.py`` names, closed by compiling rather than reading.**

    The suite runs on SQLite by default and ``MIFORGE_TEST_DATABASE_URL`` switches it to
    PostgreSQL: *"the two are not equivalent and that is a known limitation, recorded rather
    than hidden - a PostgreSQL-only type or a JSONB operator would pass here and fail there."*

    These tables first carried ``server_default=sa.text("0")`` on their booleans and
    ``CHECK (breached = 0)``. Both are valid SQLite and **both are rejected by PostgreSQL**,
    so every test passed and the migration would have failed on the engine production uses -
    on the coordinator's clean clone, which is the worst moment. Found by compiling the DDL
    for the dialect, which is also what the coordinator needs handed over.
    """

    @pytest.mark.parametrize("dialect_name", ["postgresql", "sqlite"])
    def test_both_tables_compile(self, dialect_name: str) -> None:
        from sqlalchemy.dialects import postgresql, sqlite
        from sqlalchemy.schema import CreateIndex, CreateTable

        from services.scoring.tables import SCORE_ITEMS, SCORES

        dialect = {"postgresql": postgresql.dialect(), "sqlite": sqlite.dialect()}[dialect_name]
        for table in (SCORES, SCORE_ITEMS):
            assert str(CreateTable(table).compile(dialect=dialect))
            for index in table.indexes:
                assert str(CreateIndex(index).compile(dialect=dialect))

    def test_no_boolean_default_renders_as_an_integer_on_postgresql(self) -> None:
        """``BOOLEAN DEFAULT 0`` is the exact form PostgreSQL refuses."""
        from sqlalchemy.dialects import postgresql
        from sqlalchemy.schema import CreateTable

        from services.scoring.tables import SCORE_ITEMS, SCORES

        for table in (SCORES, SCORE_ITEMS):
            sql = str(CreateTable(table).compile(dialect=postgresql.dialect()))
            assert "BOOLEAN DEFAULT 0" not in sql
            assert "BOOLEAN DEFAULT 1" not in sql

    def test_no_check_constraint_compares_a_boolean_to_an_integer(self) -> None:
        from services.scoring.tables import SCORE_ITEMS, SCORES

        for table in (SCORES, SCORE_ITEMS):
            for constraint in table.constraints:
                text = str(getattr(constraint, "sqltext", ""))
                assert "breached = 0" not in text
                assert "reproducible = 1" not in text
