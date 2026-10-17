"""``reindex`` is a first-class command  (task 4.3; ADR-004).

It walks the artifact root and rebuilds every index row **from manifests alone**. These
tests cover what it does with what it finds: conforming artifacts, nonconforming ones, and
things that are not artifacts at all.

**The refuse-versus-record split is the design decision under test.** An artifact with no
usable identity is refused - there is nothing to index and nothing to guess (ADR-018). An
artifact that is identifiable but nonconforming is **recorded with its anomaly logged**,
because refusing it would make a checkpoint that exists on disk invisible, and a silently
missing artifact is worse than an honestly flawed row.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from core.canonical import write_canonical
from models.domain_state import DomainState
from services.reindex import (
    INDEX_TABLES,
    RECORD_TABLES,
    list_adapters,
    list_checkpoints,
    reindex,
)
from tests.support import db as support


@pytest.fixture()
def engine(tmp_path: Path):
    return support.migrated_engine(tmp_path / "db")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    path = tmp_path / "checkpoints"
    path.mkdir()
    return path


def test_reindex_rebuilds_rows_from_manifests_alone(engine, root: Path) -> None:
    support.write_checkpoint(root, support.checkpoint_manifest(step=100))
    support.write_checkpoint(
        root,
        support.checkpoint_manifest(step=200, parent="LFM2-1.2B:run-1:step-100"),
    )
    support.write_adapter(root, support.adapter_manifest())

    with support.session_for(engine) as session:
        report = reindex(session, root=root)
        assert (report.checkpoints, report.adapters) == (2, 1)
        assert report.refused == []
        assert report.anomalies == []

        names = [row["name"] for row in list_checkpoints(session)]
        assert names == ["LFM2-1.2B:run-1:step-100", "LFM2-1.2B:run-1:step-200"]
        assert [row["name"] for row in list_adapters(session)] == ["LFM2-1.2B:run-2:adapter-50"]


def test_reindex_is_idempotent_and_replaces_rather_than_merges(engine, root: Path) -> None:
    """Destructive by design: a merge would let a deleted checkpoint's row live forever.

    The index would then slowly stop being a picture of the disk, which is the failure
    ADR-004 is written against.
    """
    directory = support.write_checkpoint(root, support.checkpoint_manifest(step=100))
    with support.session_for(engine) as session:
        reindex(session, root=root)
        assert len(list_checkpoints(session)) == 1

    for leftover in directory.iterdir():
        leftover.unlink()
    directory.rmdir()

    with support.session_for(engine) as session:
        report = reindex(session, root=root)
        assert report.total == 0
        assert list_checkpoints(session) == [], "a row survived the artifact's deletion"


def test_a_manifest_with_no_identity_is_refused_not_guessed_at(engine, root: Path) -> None:
    """A name must not be synthesised from the directory it happens to sit in."""
    manifest = support.checkpoint_manifest(step=100)
    del manifest["name"]
    write_canonical(root / "run-1" / "step-000100" / "manifest.json", manifest)

    with support.session_for(engine) as session:
        report = reindex(session, root=root)
        assert report.total == 0
        assert [code for _, code, _ in report.refused] == ["manifest_has_no_identity"]


def test_unparseable_json_is_refused_by_name(engine, root: Path) -> None:
    (root / "run-1" / "step-000100").mkdir(parents=True)
    (root / "run-1" / "step-000100" / "manifest.json").write_text("{ not json", encoding="utf-8")

    with support.session_for(engine) as session:
        report = reindex(session, root=root)
        assert [code for _, code, _ in report.refused] == ["manifest_not_json"]


def test_a_foreign_manifest_json_is_not_indexed_as_a_checkpoint(engine, root: Path) -> None:
    """``manifest.json`` is a common filename. The ``kind`` const is what identifies ours."""
    write_canonical(root / "something-else" / "manifest.json", {"kind": "other.thing"})

    with support.session_for(engine) as session:
        report = reindex(session, root=root)
        assert report.total == 0
        assert [code for _, code, _ in report.refused] == ["manifest_wrong_kind"]


def test_a_checkpoint_missing_its_required_domain_is_indexed_as_absent_and_logged(
    engine, root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Recorded, not refused - and the omission is named in the log and in the report.

    The row carries ``absent``: an omission, which is a different fact from the explicit
    null a from-scratch run writes.
    """
    manifest = support.checkpoint_manifest(step=100)
    del manifest["domain"]
    write_canonical(root / "run-1" / "step-000100" / "manifest.json", manifest)

    with (
        caplog.at_level(logging.WARNING, logger="miforge.reindex"),
        support.session_for(engine) as session,
    ):
        report = reindex(session, root=root)
        rows = list_checkpoints(session)

    assert report.checkpoints == 1
    assert rows[0]["domain_state"] == DomainState.ABSENT.value
    assert rows[0]["domain"] is None
    assert rows[0]["domain_display"] == "not recorded"
    assert ("LFM2-1.2B:run-1:step-100", "checkpoint_manifest_missing_required_domain") in (
        report.anomalies
    )
    anomalies = [r for r in caplog.records if r.message == "reindex_manifest_nonconforming"]
    assert anomalies, "a nonconforming manifest was indexed without saying so"


def test_a_from_scratch_run_records_an_explicit_null_without_an_anomaly(engine, root: Path) -> None:
    """A general pretraining run biases toward nothing, and saying so is conforming."""
    support.write_checkpoint(root, support.checkpoint_manifest(step=100, domain=None))
    with support.session_for(engine) as session:
        report = reindex(session, root=root)
        rows = list_checkpoints(session)
    assert report.anomalies == []
    assert rows[0]["domain_state"] == DomainState.ASSERTED_NULL.value
    assert rows[0]["domain_display"] == "no domain"


def test_an_adapter_is_indexed_without_a_domain_and_never_borrows_its_bases(
    engine, root: Path
) -> None:
    """The specific inference task 4.2a forbids, asserted end to end.

    The base on disk has ``domain='support'``; the adapter's row must still say *not
    recorded in this artifact kind*.
    """
    support.write_checkpoint(root, support.checkpoint_manifest(step=100, domain="support"))
    support.write_adapter(root, support.adapter_manifest())

    with support.session_for(engine) as session:
        reindex(session, root=root)
        checkpoint = list_checkpoints(session)[0]
        adapter = list_adapters(session)[0]

    assert checkpoint["domain"] == "support"
    assert adapter["domain"] is None
    assert adapter["domain_state"] == DomainState.NOT_IN_KIND.value
    assert adapter["domain_display"] == "not recorded in this artifact kind"


def test_an_adapter_without_a_base_is_refused(engine, root: Path) -> None:
    """An adapter applied to the wrong base is undefined behaviour, not a worse answer."""
    manifest = support.adapter_manifest()
    manifest["base"] = {"name": ""}
    write_canonical(root / "run-2" / "adapter-000050" / "adapter_manifest.json", manifest)

    with support.session_for(engine) as session:
        report = reindex(session, root=root)
        assert report.adapters == 0
        assert [code for _, code, _ in report.refused] == ["adapter_manifest_has_no_base"]


def test_a_score_that_omits_provider_round_trips_as_omitted(engine, root: Path) -> None:
    """``provider`` is nullable AND not ``required``, so three states again - and only a
    test can see the difference.

    The schema-sync test compares schema *files* and cannot notice a payload that leaves
    the key out; a shredded column would have turned the omission into a null on the way
    in. The JSON round trip keeps them apart.
    """
    support.write_checkpoint(
        root,
        support.checkpoint_manifest(step=100, test=support.score()),
    )
    support.write_checkpoint(
        root,
        support.checkpoint_manifest(step=200, test=support.score(provider=None)),
    )
    support.write_checkpoint(
        root,
        support.checkpoint_manifest(step=300, test=support.score(provider="in-process")),
    )

    with support.session_for(engine) as session:
        reindex(session, root=root)
        rows = {row["step"]: row["test"] for row in list_checkpoints(session)}

    assert "provider" not in rows[100], "an omitted provider was filled in"
    assert rows[200]["provider"] is None and "provider" in rows[200]
    assert rows[300]["provider"] == "in-process"


def test_reindex_touches_only_the_index_tables(engine, root: Path) -> None:
    """The record tables are not rebuildable and must survive a reindex untouched.

    **The pack registry was the sharp case and is no longer one.** This test carried a
    tripwire - ``assert "mf_packs" in RECORD_TABLES`` with the message *"packs became an
    index table; they have no disk mirror, so a reindex would be data loss (the 2026-10-13
    review's finding 10)"* - and it **fired on 2026-10-14**, which is exactly what it was
    for. The condition it was guarding has since been met: ``pack-definition-v1.json`` makes
    ``pack.json`` an on-disk file, F7 writes it atomically, and
    ``packs.registry.rebuild_index`` reconstructs both pack tables from those files alone.
    So the tripwire is replaced by its opposite rather than deleted, and ``mf_runs`` takes
    over as the example of a table nothing on disk can rebuild.
    """
    from models import Pack, Run

    support.write_checkpoint(root, support.checkpoint_manifest(step=100))
    with support.session_for(engine) as session:
        session.add(Pack(id="p1", name="support", version="1", valid=True))
        session.add(Run(id="run-1", status="running", run_type="change"))
        session.flush()

    with support.session_for(engine) as session:
        reindex(session, root=root)
        # The RUN row survives: `mf_runs` holds a status no artifact carries, so it stays a
        # record table and a reindex must not touch it.
        assert session.query(Run).count() == 1, (
            "a reindex cleared mf_runs, which no artifact can rebuild - a run's status, "
            "cancellation request and failure code exist nowhere else"
        )
        # The PACK row survives TOO, and for a different and more interesting reason: this
        # fixture configures no pack store, so there is no directory to rebuild from and
        # `reindex` skips the pack index **entirely** rather than dropping it.
        #
        # That is the deliberate behaviour and it is the one worth pinning here.
        # `rebuild_index` deletes before it inserts, so running it against an absent store
        # would clear every pack row on the grounds that no file backs them - and "the
        # directory is gone" is indistinguishable from "the volume is not mounted yet". One
        # of those makes the drop exactly the data loss the 2026-10-13 prohibition existed to
        # prevent. So an absent store means hands off.
        #
        # The other direction - a store that IS present and no longer carries a pack, where
        # the row must go - is asserted in
        # `test_pack_reindex_wiring.py::test_a_pack_deleted_from_disk_leaves_the_index`.
        assert session.query(Pack).count() == 1, (
            "an absent pack store cleared the pack index. A reindex must not treat an "
            "unmounted or missing store as evidence that no pack exists"
        )

    # Assert the INVARIANT, not a frozen list. This read
    # `set(INDEX_TABLES) == {"mf_checkpoints", "mf_adapters"}` and broke the moment F10
    # legitimately added its part tables to the index - a ratchet that fires on correct
    # work gets loosened rather than respected. What actually matters is that no RECORD
    # table is ever in the index set, because `reindex` clears what it owns.
    assert {"mf_checkpoints", "mf_adapters"} <= set(INDEX_TABLES), "the core pair must be indexed"
    assert set(INDEX_TABLES).isdisjoint(RECORD_TABLES), (
        f"a record table is in the index set and `reindex` would clear it: "
        f"{sorted(set(INDEX_TABLES) & set(RECORD_TABLES))}"
    )
    assert Run.__tablename__ in RECORD_TABLES, (
        "mf_runs became an index table. Nothing on disk carries a run's status, so a "
        "reindex that cleared it would lose state no artifact can reproduce - which is "
        "what the same assertion used to say about mf_packs, before pack.json existed."
    )
    assert Pack.__tablename__ in INDEX_TABLES, (
        "mf_packs left the index set. It belongs there since pack.json became normative "
        "(the gate contract); if packs have stopped being written to disk then the "
        "2026-10-13 review's finding 10 is live again and this must go back to RECORD_TABLES "
        "together with the call in services/reindex.py and the pack wiring test."
    )


def test_reindex_finds_artifacts_at_any_depth(engine, root: Path) -> None:
    """No requirement fixes the layout below the root, so none is assumed.

    Assuming ``<root>/<run>/<step>/`` would make ``reindex`` quietly find nothing on any
    arrangement nobody wrote down - the going-quietly-dark failure ADR-013 names.
    """
    support.write_checkpoint(
        root, support.checkpoint_manifest(step=100), relative="a/b/c/d/step-000100"
    )
    with support.session_for(engine) as session:
        report = reindex(session, root=root)
        assert report.checkpoints == 1
        assert list_checkpoints(session)[0]["path"] == "a/b/c/d/step-000100"


def test_reindex_on_a_missing_root_indexes_nothing_rather_than_raising(engine, tmp_path) -> None:
    """A fresh install has no checkpoint root yet; that is not an error."""
    with support.session_for(engine) as session:
        report = reindex(session, root=tmp_path / "does-not-exist")
        assert report.total == 0


# ------------------------------------------------------------------------------------
# Four tests added 2026-10-14 because a mutation control SURVIVED without them. Each
# line below was load-bearing and uncaught: a surviving mutation is a test finding, not
# a code finding, and these are the regressions that pin it.
# ------------------------------------------------------------------------------------


def test_a_schema_violation_that_is_not_about_domain_is_still_recorded_as_an_anomaly(
    engine, root: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """``schema_violations`` is the general nonconformance check and nothing tested it.

    Every earlier anomaly assertion came from the *domain* classifier, so turning this
    function into ``return []`` left the suite green - the whole of schema validation was
    unprotected. Here the manifest's ``weights_sha256`` is not 64 hex characters: the
    identity is intact so the row is written, and the violation must be reported.
    """
    manifest = support.checkpoint_manifest(step=100)
    manifest["weights_sha256"] = "not-a-digest"
    write_canonical(root / "run-1" / "step-000100" / "manifest.json", manifest)

    with (
        caplog.at_level(logging.WARNING, logger="miforge.reindex"),
        support.session_for(engine) as session,
    ):
        report = reindex(session, root=root)

    assert report.checkpoints == 1
    anomalies = [anomaly for _, anomaly in report.anomalies]
    assert any("weights_sha256" in anomaly for anomaly in anomalies), anomalies
    assert [r for r in caplog.records if r.message == "reindex_manifest_nonconforming"]


def test_the_indexed_manifest_digest_is_the_one_the_writer_produced(engine, root: Path) -> None:
    """Two independent paths to one digest, required to agree.

    ``write_canonical`` hashes the bytes it writes; ``reindex`` reads the file, parses it
    and canonicalises it again. Nothing asserted they met, so replacing the digest with a
    constant left the suite green - and the digest is what ``MergedAdapter.manifest_sha256``
    and ``AdapterBase.manifest_sha256`` point at, so a constant would make every adapter
    resolve to the same base.
    """
    manifest = support.checkpoint_manifest(step=100)
    digest = write_canonical(root / "run-1" / "step-000100" / "manifest.json", manifest)
    other = support.checkpoint_manifest(step=200)
    other_digest = write_canonical(root / "run-1" / "step-000200" / "manifest.json", other)

    with support.session_for(engine) as session:
        reindex(session, root=root)
        rows = {row["step"]: row["manifest_sha256"] for row in list_checkpoints(session)}

    assert rows[100] == digest
    assert rows[200] == other_digest
    assert digest != other_digest


def test_a_manifest_with_no_created_at_is_refused_rather_than_dated_from_the_file(
    engine, root: Path
) -> None:
    """The file's mtime is the **copy's** time, not the save's.

    A checkpoint ``scp``'d to this machine would be dated to the copy and its position in
    every chronological view would be wrong - plausibly, which is the worst kind. Nothing
    tested the refusal, so removing it left the suite green.
    """
    manifest = support.checkpoint_manifest(step=100)
    del manifest["created_at"]
    write_canonical(root / "run-1" / "step-000100" / "manifest.json", manifest)

    with support.session_for(engine) as session:
        report = reindex(session, root=root)

    assert report.checkpoints == 0
    assert [code for _, code, _ in report.refused] == ["manifest_timestamp_missing"]


def test_the_listing_is_ordered_by_name_not_by_insertion(engine, root: Path) -> None:
    """The ordering task 4.4's comparison depends on, and nothing pinned it.

    ``reindex`` inserts in **path** order, so a listing without ``ORDER BY`` comes back in
    insertion order - which happens to match name order in every other fixture here. These
    two artifacts are written to directories that sort the *opposite* way from their names,
    so dropping the ordering goes red. Without this, task 4.4's equality would pass or fail
    on PostgreSQL's physical row order rather than on whether the rebuild was faithful.
    """
    support.write_checkpoint(
        root,
        support.checkpoint_manifest(base_model="zzz-model", step=1),
        relative="aaa-directory/step-000001",
    )
    support.write_checkpoint(
        root,
        support.checkpoint_manifest(base_model="aaa-model", step=2),
        relative="zzz-directory/step-000002",
    )

    with support.session_for(engine) as session:
        reindex(session, root=root)
        names = [row["name"] for row in list_checkpoints(session)]

    assert names == sorted(names), names
    assert names[0].startswith("aaa-model"), names
