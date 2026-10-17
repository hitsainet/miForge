"""The production ``reindex`` rebuilds the pack index  (ADR-016 #1, ADR-004; foundation 4.4).

**Why this file exists.** F7 built ``packs.registry.rebuild_index`` - it drops ``mf_packs``
and ``mf_pack_parts`` and rebuilds both from each committed version's ``pack.json`` - and
reported, accurately, that *"declaring a mechanism is not wiring it: the capability is tested,
the wiring is owed"*, because ``services/reindex.py`` is not a feature agent's file. By this
project's own rule that means the pack rebuild was **not shipped**: a capability is not
shipped until a test fails when its wiring is removed.

It was the **fourth** rebuilder in that module written without a production caller. F1's
``rebuild_step_index``, F2's ``rebuild_batch_index`` and F4's ``reindex_scores`` were the
first three, each reachable only from its own tests, each declining to edit the shared file -
which was right, and is why they accumulated. All four are now called.

**This test is behavioural, not a grep, and the distinction is the whole point.**
``test_scoring_reindex.py`` pins its three siblings with an AST check that the name appears in
a ``Call`` node, and this project's own rule says that form *"proves a call exists, not that
anything depends on it"* - it would pass against a call passing the wrong session, or one
inside a branch that never executes. So the assertions here seed packs on disk, **empty the
index**, run the production :func:`services.reindex.reindex`, and require the rows back.
Delete the call in ``reindex.py`` and these go red; that was run as a negative control.

**And it pins the reversal.** Packs were the one artifact with no on-disk manifest, which is
why the 2026-10-13 review ruled that *"a pack assertion must not be added to the
reindex-equivalence test without first deciding whether packs get a disk mirror."* The gate
contract decided it - ``pack-definition-v1.json`` reads *"THIS IS AN ON-DISK FILE,
'pack.json'"* - so the condition is met and the assertion belongs here. If packs ever stop
being written to disk, these fail and say so rather than leaving a stale prohibition in a
docstring nobody rereads.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa

from models.pack import Pack, PackPart
from packs.registry import PACK_INDEX_TABLES
from packs.settings import reset_pack_settings
from services import reindex as reindex_module
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for


@pytest.fixture
def packs_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
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


@pytest.fixture
def seeded(session, packs_root: Path):
    """Two committed pack versions, written through the **real** writer.

    Through ``registry.create`` rather than by inserting rows or writing a literal document:
    a fixture whose fields agree by construction is how a suite stays green over a real bug,
    and here the thing under test is precisely whether the file on disk and the row agree.
    """
    fx.commit(session, fx.definition(name="notary-md", version="1", slots=fx.valid_slots()))
    fx.commit(session, fx.definition(name="notary-md", version="2", slots=fx.valid_slots()))
    session.commit()
    return packs_root


def _counts(session) -> tuple[int, int]:
    packs = session.execute(sa.select(sa.func.count()).select_from(Pack.__table__)).scalar_one()
    parts = session.execute(sa.select(sa.func.count()).select_from(PackPart.__table__)).scalar_one()
    return packs, parts


def _rows(session) -> list[tuple]:
    """Every indexed pack row, ordered, as comparable tuples.

    Compared as whole rows rather than counts: a rebuild that produced the right *number* of
    rows with the wrong contents would pass a count assertion, and that is the failure a
    reindex is least able to notice about itself.

    ``definition_sha256`` is included deliberately and is the strongest column here - it is
    the digest of the ``pack.json`` the row was built from, so comparing it proves the
    rebuilt row describes the same *bytes* and not merely the same identity. ``valid`` comes
    too, because a rebuild that restored every pack as invalid would list them all and let
    none start a run.
    """
    return [
        tuple(row)
        for row in session.execute(
            sa.select(
                Pack.id, Pack.name, Pack.version, Pack.valid, Pack.definition_sha256
            ).order_by(Pack.id)
        ).all()
    ]


class TestThePackTablesAreNowIndexTables:
    def test_they_are_classified_as_index_not_record(self) -> None:
        """The classification is the premise of everything else in this file.

        While they sat in ``RECORD_TABLES`` the equivalence test was *required* to skip them,
        and the skip was correct. Asserted rather than remembered, because the reversal is
        recent and a later reader will find the old prohibition in git history.
        """
        assert set(PACK_INDEX_TABLES) <= set(reindex_module.INDEX_TABLES)
        assert not set(PACK_INDEX_TABLES) & set(reindex_module.RECORD_TABLES)

    def test_drop_index_actually_empties_them(self, session, seeded) -> None:
        """**Without this, every assertion below is vacuous.**

        If ``drop_index`` missed the pack tables, the rows would still be there when
        ``reindex`` ran and "identical after a rebuild" would be satisfied by rows the
        rebuild never had to produce - construction-green, in the one test whose job is to
        prove the index is reconstructible.
        """
        assert _counts(session) > (0, 0), "the fixture seeded nothing, so nothing is tested"
        reindex_module.drop_index(session)
        assert _counts(session) == (0, 0), (
            "drop_index left pack rows behind. Every equivalence assertion over packs is "
            "then vacuous: it would compare rows that were never removed."
        )


class TestTheProductionReindexRebuildsPacks:
    def test_the_rows_come_back_after_a_drop(self, session, seeded) -> None:
        """**The wiring assertion.** Red when the call in ``reindex.py`` is deleted.

        Not "was rebuild_index called" - the rows are read back, so a call passing the wrong
        session or sitting in a dead branch fails here too.
        """
        before = _rows(session)
        assert len(before) == 2

        reindex_module.drop_index(session)
        assert _rows(session) == []

        report = reindex_module.reindex(session)
        session.flush()

        assert _rows(session) == before, (
            "the production reindex did not rebuild the pack index from pack.json. If this "
            "is empty, services/reindex.py no longer calls packs.registry.rebuild_index and "
            "losing the database again loses every pack - which is the whole claim ADR-004 "
            "makes and the reason pack.json exists."
        )
        assert report.packs == 2, (
            f"the report says {report.packs} packs rebuilt, not 2. A reindex that rebuilds "
            f"silently is a reindex whose operator cannot tell it ran."
        )

    def test_the_part_rows_come_back_too(self, session, seeded) -> None:
        """``mf_pack_parts`` is the child table and the one carrying the composition.

        A rebuild that restored ``mf_packs`` alone would give a registry listing with every
        pack present and every pack empty - which looks like data loss rather than a broken
        rebuild, and would be diagnosed as such.
        """
        _, parts_before = _counts(session)
        assert parts_before > 0

        reindex_module.drop_index(session)
        reindex_module.reindex(session)
        session.flush()

        assert _counts(session)[1] == parts_before

    def test_a_rebuild_is_idempotent(self, session, seeded) -> None:
        """Twice in a row must not duplicate. ``rebuild_index`` deletes before it inserts,
        so this pins that rather than trusting it - a reindex an operator is afraid to run
        twice is one they will not run when they need it."""
        reindex_module.reindex(session)
        session.flush()
        once = _rows(session)
        reindex_module.reindex(session)
        session.flush()
        assert _rows(session) == once

    def test_a_pack_deleted_from_disk_leaves_the_index(self, session, seeded) -> None:
        """The disk is the truth, in both directions.

        A rebuild that only ever *added* would keep a pack the store no longer holds, and the
        index would then assert the existence of something nobody can open. ``pack.json`` is
        removed rather than the row, because the file is what the rebuild reads.
        """
        removed = seeded / "notary-md" / "2"
        assert (removed / "pack.json").exists()
        for child in removed.iterdir():
            child.unlink()
        removed.rmdir()

        reindex_module.reindex(session)
        session.flush()

        assert [row[2] for row in _rows(session)] == ["1"], (
            "the deleted version is still indexed. The filesystem is authoritative, so a "
            "rebuild must drop what the store no longer carries, not merely add what it does."
        )


class TestAnAbsentPackStoreIsSkippedNotDropped:
    """**The case that cost three wrong fixes, so each one is pinned.**

    An absent pack store must leave the index **untouched**: not rebuilt, and above all not
    dropped. ``rebuild_index`` deletes before it inserts, so running it against a store that
    is merely not mounted yet would clear every pack row on the grounds that no file backs
    them - and from inside this function "the directory was deleted" and "the volume is not
    attached" are the same observation. One of them makes the drop precisely the data loss
    that the 2026-10-13 review's finding 10 existed to prevent.
    """

    def test_an_empty_store_rebuilds_to_nothing_without_refusing(self, session, packs_root):
        """No packs is an ordinary state, not an error - miForge runs with none.

        Distinguished from a *failure* to read the store, which is reported. A reindex that
        refused on an empty store would make ADR-004's recovery path unusable on a fresh
        installation, which is when it is most likely to be run.
        """
        report = reindex_module.reindex(session)
        assert report.packs == 0
        assert not [
            entry for entry in report.refused if entry[0] == "mf_packs"
        ], "an empty or absent store was reported as a failure"

    def test_an_absent_store_does_not_clear_existing_rows(
        self, session, seeded, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """Point the configuration at a directory that does not exist, then reindex.

        The rows must still be there afterwards. This is the assertion that stops a later
        simplification - "just always call rebuild_index, it handles an empty store" - from
        turning a missing mount into an emptied registry.
        """
        before = _rows(session)
        assert len(before) == 2

        monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "not-mounted"))
        reset_pack_settings()
        try:
            report = reindex_module.reindex(session)
            session.flush()
            assert _rows(session) == before, (
                "an absent pack store cleared the pack index. A reindex must not read a "
                "missing or unmounted directory as evidence that no pack exists - that is "
                "the data loss finding 10 was written about."
            )
            assert report.packs == 0
        finally:
            reset_pack_settings()

    def test_the_reindex_still_reports_the_artifacts_it_did_rebuild(
        self, session, seeded, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """A pack-side problem must not abort the whole operation.

        The first version of this wiring called ``session.rollback()`` on a filesystem error,
        which discarded every checkpoint and adapter row already flushed - so the reindex
        reported success having indexed nothing. A reindex is run when something is already
        wrong, so each subsystem has to be able to fail by itself. Asserted through the
        report rather than the log, because a report is what a caller reads.
        """
        monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "not-mounted"))
        reset_pack_settings()
        try:
            report = reindex_module.reindex(session)
            # No exception, and a usable report - the point is that it returned at all.
            assert report.packs == 0
            assert isinstance(report.checkpoints, int)
            assert isinstance(report.adapters, int)
        finally:
            reset_pack_settings()


class TestTheDatabaseOnlyColumnsSurviveARebuild:
    """**`0015`'s three columns cannot ride in `pack.json`, so a rebuild would erase them.**

    The operator added `is_example`, `derived_from_pack_id` and `addressed_through_pack_id`
    on 2026-10-14 because each is a capability a feature document promises that the pack
    artifact **cannot express** - `pack-definition-v1.json`'s top-level object and its
    `PackSlot` are both closed to unknown keys. That is exactly why they are columns.

    And it is exactly why they are a hazard: `reindex` rebuilds `mf_packs` **from the files
    alone**, so the obvious implementation would seed an example pack, run a reindex, and
    silently lose the Example flag and the whole derivation history. Nothing would report it;
    the registry would simply stop saying which packs were examples.

    So `rebuild_index` captures them before the delete and restores them after, and these are
    the assertions that keep it doing so. They make these three **the one piece of pack state
    the filesystem is not authoritative for**, which is a real exception to ADR-004 and is
    stated here rather than left to be discovered.
    """

    def test_is_example_and_derivation_survive_the_production_reindex(
        self, session, seeded
    ) -> None:
        """Set them, reindex through the production entry point, read them back."""
        from models.pack import Pack

        original = session.get(Pack, "notary-md@1")
        assert original is not None, "the fixture's pack is not indexed, so nothing is tested"
        original.is_example = True
        original.derived_from_pack_id = "notary-md@2"
        session.flush()

        reindex_module.reindex(session)
        session.flush()

        after = session.get(Pack, "notary-md@1")
        assert after is not None, "the rebuild dropped the pack entirely"
        assert after.is_example is True, (
            "the reindex erased is_example. pack.json cannot carry it, so rebuild_index must "
            "carry it across the delete - otherwise seeding an example pack and running a "
            "reindex silently un-marks it and the Example pill disappears for good."
        )
        assert (
            after.derived_from_pack_id == "notary-md@2"
        ), "the reindex erased the derivation record, which exists nowhere else"

    def test_the_part_level_column_survives_too(self, session, seeded) -> None:
        """`addressed_through_pack_id` is on the child table and is just as unrecoverable."""
        import sqlalchemy as sa

        from models.pack import PackPart

        part = session.execute(
            sa.select(PackPart).where(PackPart.pack_id == "notary-md@1").limit(1)
        ).scalar_one()
        kind = part.part_kind
        part.addressed_through_pack_id = "shared-prompts@1"
        session.flush()

        reindex_module.reindex(session)
        session.flush()

        after = session.execute(
            sa.select(PackPart).where(PackPart.pack_id == "notary-md@1", PackPart.part_kind == kind)
        ).scalar_one()
        assert after.addressed_through_pack_id == "shared-prompts@1"

    def test_a_carried_value_for_a_pack_the_store_no_longer_holds_is_dropped(
        self, session, seeded
    ) -> None:
        """**The other direction, and it is the one that could resurrect a deleted pack.**

        A naive "restore everything I captured" would re-insert a row for a pack whose
        `pack.json` is gone, making this function the one place a deleted pack comes back.
        The filesystem is authoritative for *which packs exist*; the three columns are
        carried only for packs the rebuild actually produced.
        """
        from models.pack import Pack

        row = session.get(Pack, "notary-md@2")
        assert row is not None
        row.is_example = True
        session.flush()

        removed = seeded / "notary-md" / "2"
        for child in removed.iterdir():
            child.unlink()
        removed.rmdir()

        reindex_module.reindex(session)
        session.flush()

        assert session.get(Pack, "notary-md@2") is None, (
            "a pack whose pack.json was deleted came back from the carried-column map. The "
            "filesystem decides which packs exist; the columns only decorate the ones it "
            "still holds."
        )
        assert session.get(Pack, "notary-md@1") is not None, "the surviving pack was lost"
