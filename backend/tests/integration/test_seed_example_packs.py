"""The example-pack seeder  (task 4.6, 4.7; FR-007.16, R-02.39).

Four properties, and three of them are about what the seeder **does not** do.

1. **It seeds two packs and marks both.** Read back through the real reader and the real
   registry listing, not from the script's own return value: a seeder asserted against its
   own report is a fixture agreeing with itself.
2. **It is not a migration.** Asserted mechanically - a migrated database holds **zero**
   pack rows, and no revision in the Alembic graph mentions an example pack - rather than in
   prose. That is the whole of FR-007.16's *"deletable"*: a deleted example must stay
   deleted, and a seeder that ran inside ``alembic upgrade`` would put it back on the next
   deploy whether or not anyone asked.
3. **It fabricates no parts.** The examples hold contract references into miDataworks and
   nothing writes a row into F10's part index. An example that resolved because this script
   invented its data would be a pack an operator could start a run from, and that run would
   train on fiction while every number it produced looked real.
4. **It is idempotent, and it repairs the one thing a downgrade loses.** Re-running creates
   nothing and rewrites no file; and with the pack on disk but ``is_example`` cleared - which
   is exactly the state migration ``0015``'s downgrade leaves behind, since neither
   ``pack.json`` nor an export can carry the flag - a re-run sets it again and *says so*.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest
import sqlalchemy as sa

from packs import registry, store
from packs.settings import reset_pack_settings
from packs.slots import SLOT_ORDER, SlotKind
from tests.support.db import REPO_ROOT, migrated_engine, session_for


def _load_seeder():
    """Import the script **by path**, the way an operator runs it.

    ``backend/scripts/`` is not a package and is not on ``pythonpath`` (``pyproject.toml``
    puts only ``backend/src`` there), so a plain import would not find it. Loading the real
    file matters more than the convenience: the thing under test is the script that ships,
    including its ``sys.path`` shim, and a copy of its data in this file would be a second
    declaration of the two example packs.
    """
    path = REPO_ROOT / "backend" / "scripts" / "seed_example_packs.py"
    spec = importlib.util.spec_from_file_location("seed_example_packs", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


seeder = _load_seeder()


@pytest.fixture
def packs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    # Unset deliberately: the examples hold contract references, so an installation with no
    # miDataworks is the case they must behave honestly in (R-02.50, FR-007.31).
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_pack_settings()
    yield tmp_path / "packs"
    reset_pack_settings()


@pytest.fixture
def engine(tmp_path: Path, packs_root: Path):
    engine = migrated_engine(tmp_path / "db")
    yield engine
    engine.dispose()


@pytest.fixture
def session(engine):
    with session_for(engine) as session:
        yield session


class TestItSeedsTwoMarkedExamples:
    def test_it_creates_both_and_marks_them(self, session, packs_root: Path) -> None:
        report = seeder.seed(session)
        assert sorted(report.created) == ["ex-carburetor-rebuild@1", "ex-financial-supervision@1"]
        assert report.flag_restored == []
        assert report.already_present == []

        # Read back through the real registry listing, not the report.
        rows = {row.pack_id: row for row in registry.listing(session)}
        assert set(rows) == {"ex-financial-supervision", "ex-carburetor-rebuild"}
        for row in rows.values():
            assert (
                row.is_example is True
            ), "an unmarked example is indistinguishable from a real pack"
            assert row.derived_from is None

    def test_each_pack_is_on_disk_and_reads_back_through_the_real_reader(
        self, session, packs_root: Path
    ) -> None:
        seeder.seed(session)
        for example in seeder.EXAMPLE_PACKS:
            path = packs_root / example.name / example.version / "pack.json"
            assert (
                path.is_file()
            ), f"{example.name} has no pack.json; ADR-004 makes the file the truth"
            # Through `store.read`, which validates the document and verifies its digest.
            definition = store.read(example.name, example.version)
            assert definition.name == example.name
            assert definition.domain == example.domain
            # And the document on disk is the canonical bytes the digest covers.
            document = json.loads(path.read_text())
            assert document["definition_sha256"] == definition.document()["definition_sha256"]

    def test_both_fill_six_of_seven_slots_and_differ_in_exactly_one(self) -> None:
        """The pedagogical claim in the script's docstring, asserted.

        The pair exists to show the two signal shapes - a corpus against preference pairs -
        and a pair that differed in nothing would be the same example printed twice.
        """
        filled = {
            example.name: {slot for slot in SLOT_ORDER if example.slots.get(slot) is not None}
            for example in seeder.EXAMPLE_PACKS
        }
        assert len(filled) == 2
        for slots in filled.values():
            assert len(slots) == 6
        left, right = filled.values()
        assert left ^ right == {SlotKind.CORPUS, SlotKind.PREFERENCE_PAIRS}

    def test_neither_is_invalid_for_a_reason_the_author_controls(self, session) -> None:
        """No missing required slot and no missing floor - the two faults a pack can have.

        What remains is a resolution state, which is about this *installation* and not about
        the pack: with miDataworks unset the registry says "Needs a service", and
        :mod:`packs.resolver`'s most important line is that nothing is wrong with the pack.
        An operator told their example pack is broken will go and edit a pack that is fine.
        """
        seeder.seed(session)
        for row in registry.listing(session):
            assert row.validity.missing_slots == ()
            assert row.validity.floor_missing is False
            assert row.parts_filled == 6
            assert row.parts_total == 7
            # Unresolvable, so the count is unknown and must never be rendered as 0.
            assert row.prompt_count is None


class TestItIsNotAMigration:
    """FR-007.16's *deletable* - which a migration would quietly undo."""

    def test_a_migrated_database_holds_no_packs(self, engine) -> None:
        with engine.connect() as conn:
            assert (
                conn.execute(sa.select(sa.func.count()).select_from(sa.table("mf_packs"))).scalar()
                == 0
            )

    def test_no_revision_in_the_graph_seeds_an_example(self) -> None:
        versions = REPO_ROOT / "backend" / "alembic" / "versions"
        offenders = [
            path.name
            for path in sorted(versions.glob("*.py"))
            for text in [path.read_text()]
            if "ex-financial-supervision" in text
            or "ex-carburetor-rebuild" in text
            or "EXAMPLE_PACKS" in text
            or "seed_example_packs" in text
        ]
        assert offenders == [], (
            f"{offenders} seeds or imports the example packs from inside the Alembic graph. "
            f"A deleted example would then return on the next deploy, which is the opposite "
            f"of FR-007.16's 'deletable with nothing depending on them'."
        )

    def test_an_example_can_be_deleted_and_stays_deleted(self, session, packs_root: Path) -> None:
        seeder.seed(session)
        registry.delete(session, "ex-carburetor-rebuild", "1")

        assert not (packs_root / "ex-carburetor-rebuild" / "1" / "pack.json").exists()
        assert {row.pack_id for row in registry.listing(session)} == {"ex-financial-supervision"}

        # A reindex reads the files, so it must not bring the deleted one back either.
        registry.rebuild_index(session)
        assert {row.pack_id for row in registry.listing(session)} == {"ex-financial-supervision"}
        # And the surviving example keeps its flag across the rebuild, which is the db-only
        # column `rebuild_index` carries across its own delete.
        assert registry.listing(session)[0].is_example is True


class TestItFabricatesNothing:
    def test_it_writes_no_part_into_f10s_index(self, session) -> None:
        seeder.seed(session)
        count = session.execute(
            sa.select(sa.func.count()).select_from(sa.table("mf_contract_parts"))
        ).scalar()
        assert count == 0, (
            "the seeder invented parts. The parts are miDataworks' (BRD-03 R-03.48) and an "
            "example that resolved because this script wrote its data would be a pack an "
            "operator could start a real run from."
        )

    def test_every_slot_is_a_contract_reference_and_none_is_external(self) -> None:
        for example in seeder.EXAMPLE_PACKS:
            for slot in SLOT_ORDER:
                filled = example.slots.get(slot)
                if filled is None:
                    continue
                assert filled.ref.source == "midataworks"

    def test_the_seeded_document_carries_a_fixed_timestamp_not_the_clock(
        self, session, packs_root: Path
    ) -> None:
        """``created_at`` is fixed, not ``now()``.

        Two installations that seed the same example must hold the same document, or an
        exported example is not comparable with the one it came from - and a wall-clock stamp
        would make every seed a different document describing the same pack.

        **Asserted against a literal, and the first version of this test was vacuous.** It
        compared the written document's digest with a freshly built one and passed with
        ``SEEDED_AT = datetime.now(tz=UTC)`` - both sides read the same module constant in
        the same process, so they agreed by construction and the mutation survived. The
        literal below is the only form that can tell a fixed instant from a clock reading.
        One line to update if the seeded instant is ever deliberately changed, which is the
        right cost for the only assertion that bites.
        """
        seeder.seed(session)
        document = json.loads(
            (packs_root / "ex-financial-supervision" / "1" / "pack.json").read_text()
        )
        assert document["created_at"] == "2026-10-14T00:00:00Z"
        for example in seeder.EXAMPLE_PACKS:
            assert example.definition().created_at == seeder.SEEDED_AT
            assert (
                json.loads((packs_root / example.name / example.version / "pack.json").read_text())[
                    "created_at"
                ]
                == "2026-10-14T00:00:00Z"
            )


class TestItIsIdempotentAndRepairsTheFlag:
    def test_a_second_run_creates_nothing_and_rewrites_no_file(
        self, session, packs_root: Path
    ) -> None:
        seeder.seed(session)
        path = packs_root / "ex-financial-supervision" / "1" / "pack.json"
        before = (path.read_bytes(), path.stat().st_mtime_ns)

        report = seeder.seed(session)
        assert report.created == []
        assert report.flag_restored == []
        assert sorted(report.already_present) == [
            "ex-carburetor-rebuild@1",
            "ex-financial-supervision@1",
        ]
        # Not rewritten: a committed version's file is never rewritten (FR-007.2), and the
        # seeder must not route around that refusal.
        assert (path.read_bytes(), path.stat().st_mtime_ns) == before
        assert len(registry.listing(session)) == 2

    def test_it_restores_the_flag_a_downgrade_dropped(self, session, packs_root: Path) -> None:
        """Migration ``0015``'s downgrade, simulated at the one place it shows.

        That downgrade drops ``is_example``, and re-upgrading brings the column back as
        ``false`` for every row. Neither ``pack.json`` nor an export carries the flag - the
        document's top-level object is closed - so re-running this script is the remedy, and
        it must be able to tell *"repaired"* from *"nothing to do"*.
        """
        seeder.seed(session)
        registry.mark_example(session, "ex-financial-supervision", "1", is_example=False)

        report = seeder.seed(session)
        assert report.created == []
        assert report.flag_restored == ["ex-financial-supervision@1"]
        assert report.already_present == ["ex-carburetor-rebuild@1"]
        rows = {row.pack_id: row for row in registry.listing(session)}
        assert rows["ex-financial-supervision"].is_example is True

    def test_a_dry_run_writes_nothing(self, session, packs_root: Path) -> None:
        report = seeder.seed(session, dry_run=True)
        assert sorted(report.created) == ["ex-carburetor-rebuild@1", "ex-financial-supervision@1"]
        assert registry.listing(session) == []
        assert not (packs_root / "ex-financial-supervision").exists()

    def test_the_report_names_each_pack_and_what_happened_to_it(self, session) -> None:
        """A seeder that printed "done" would be the silent half of ADR-013."""
        lines = seeder.seed(session).lines()
        assert len(lines) == 2
        assert all("created" in line and "is_example" in line for line in lines)
        assert any("ex-financial-supervision@1" in line for line in lines)

        # And the dry run says what it *would* do, in a tense an operator can read: a prefix
        # glued onto a past-tense line produces "would created".
        reset = seeder.SeedReport(created=["ex-a@1"])
        assert reset.lines(dry_run=True) == ["would create  ex-a@1  (and mark is_example)"]


class TestMarkExampleIsTheOnlyWriter:
    def test_it_refuses_a_pack_that_has_no_index_row(self, session) -> None:
        with pytest.raises(Exception) as caught:
            registry.mark_example(session, "nothing-here", "1")
        assert "pack_not_indexed" in str(caught.value) or "no index row" in str(caught.value)

    def test_marking_twice_reports_no_change_the_second_time(self, session) -> None:
        seeder.seed(session)
        assert registry.mark_example(session, "ex-carburetor-rebuild", "1") is False
        assert (
            registry.mark_example(session, "ex-carburetor-rebuild", "1", is_example=False) is True
        )
