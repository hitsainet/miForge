"""Drop the part index, rebuild from manifests, get identical answers  (AC-010.16; ADR-004).

ADR-016's fourth structural test, for F10's half of the index. **The filesystem is the source
of truth; the database is a rebuildable index** - and that is a claim which is only true while
something checks it.

**Group B is out of scope, and this file says so in words.** ``mf_review_bindings`` and
``mf_review_candidate_refs`` hold a miDataworks queue id and an ``external_id`` ->
``item_id`` map; **no local file can reconstruct either**, because the authority for a
decision is miDataworks. Writing the exclusion down is what stops a later agent "fixing" this
test by inventing a rebuild for them - which would quietly make miForge the authority for
somebody else's decisions.

## Not wired into ``services.reindex.reindex`` yet, and that is a test rather than a note

``services/reindex.py`` is the foundation's and walks the *checkpoint* root for
``manifest.json`` / ``adapter_manifest.json``. F10's rebuild is a separate function over the
parts root, and nothing calls it from the global command. ADR-016 #1 says *declaring a
mechanism is not wiring it*, so the gap is pinned by a **strict xfail** below rather than by a
sentence in a report: when the one-line wiring lands, that test XPASSes and the suite goes red
until the marker is removed.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from services.parts import acquire, index
from services.parts.tables import PART_INDEX_TABLES, REVIEW_STATE_TABLES
from tests.support import db as support
from tests.support import part_fixtures as fx


@pytest.fixture()
def engine(tmp_path: Path):
    return support.migrated_engine(tmp_path / "db")


@pytest.fixture()
def seeded(engine, tmp_path: Path) -> Path:
    """Three parts of different shapes, imported through the real path.

    Different shapes on purpose: a rows part with a reference, a rows part **without** one
    (E-21), and a config part with no splits at all. A single-shape fixture would pass while
    the rebuild dropped whichever shape it did not cover.
    """
    parts_root = tmp_path / "parts"
    parts_root.mkdir()

    cases = [
        fx.miforge_set_manifest(reference=("prompt-set", "support-tickets", "3")),
        fx.miforge_set_manifest(miforge_set_kind="test_set", reference=None),
        fx.config_manifest(kind="selector"),
    ]
    with support.session_for(engine) as session:
        for number, (manifest, bodies) in enumerate(cases):
            directory = fx.write_export(tmp_path / f"export-{number}", manifest, bodies)
            acquire.from_dir(session, directory, parts_root=parts_root)
    return parts_root


def test_the_fixture_covers_every_shape_before_anything_is_compared(engine, seeded):
    """Otherwise the equivalence below is a statement about one shape dressed as three."""
    with support.session_for(engine) as session:
        parts = index.list_parts(session)
    assert len(parts) == 3
    assert {part["content_kind"] for part in parts} == {"rows", "config"}
    assert any(part["contract_reference"] is None for part in parts), "no E-21 case"
    assert any(part["splits"] for part in parts) and any(not part["splits"] for part in parts)


def test_dropping_and_reindexing_produces_identical_responses(engine, seeded):
    """AC-010.16. Compared through the **API's own read seam**, not by counting rows: a row
    count can match while a field a caller reads has gone missing."""
    with support.session_for(engine) as session:
        before = index.parts_snapshot(session)

    with support.session_for(engine) as session:
        index.delete_part_index(session)
        assert index.parts_snapshot(session)["parts"] == []

    with support.session_for(engine) as session:
        report = index.reindex_parts(session, root=seeded)
        assert report.parts == 3
        assert report.refused == []

    with support.session_for(engine) as session:
        after = index.parts_snapshot(session)

    assert after == before


def test_a_dropped_index_that_is_not_reindexed_differs(engine, seeded):
    """The negative control for the test above, inline. **An equivalence test that cannot
    tell a rebuilt index from an empty one is green against any rebuild at all** - including
    none - which is the construction-green failure this project keeps finding."""
    with support.session_for(engine) as session:
        before = index.parts_snapshot(session)
    with support.session_for(engine) as session:
        index.delete_part_index(session)
    with support.session_for(engine) as session:
        assert index.parts_snapshot(session) != before


def test_the_rebuild_reads_manifests_and_not_previous_rows(engine, seeded):
    """**From the manifest only.** A rebuild that read the old row would be comparing the
    index with itself, and the equivalence assertion above would mean nothing."""
    manifest_path = next(seeded.rglob("midataworks-dataset-version.json"))
    document = manifest_path.read_text("utf-8").replace(
        '"dataset": "humor-format-balanced"', '"dataset": "renamed-upstream"'
    )
    manifest_path.write_text(document, encoding="utf-8")

    with support.session_for(engine) as session:
        index.reindex_parts(session, root=seeded)
        datasets = {part["dataset"] for part in index.list_parts(session)}
    assert "renamed-upstream" in datasets, "the rebuild did not read the manifest it was given"


def test_a_part_whose_manifest_stopped_conforming_is_refused_and_named(engine, seeded):
    """Not carried forward from its old row. **Carrying it forward is how an index outlives
    the contract** - the row would keep asserting fields the manifest no longer has."""
    manifest_path = next(seeded.rglob("midataworks-dataset-version.json"))
    manifest_path.write_text('{"kind": "not-a-dataset-version"}', encoding="utf-8")

    with support.session_for(engine) as session:
        report = index.reindex_parts(session, root=seeded)
    assert report.parts == 2
    assert len(report.refused) == 1
    assert report.refused[0][1] == "dataset_version_invalid"


def test_the_source_of_an_external_part_survives_a_rebuild(engine, tmp_path):
    """``source`` is **not a manifest field**, so a rebuild with nothing beside the manifest
    would have to guess - and the cheap guess ("it has a contract reference, so it came from
    miDataworks") is exactly the promotion R-02.40 forbids. Hence the sidecar."""
    parts_root = tmp_path / "parts"
    parts_root.mkdir()
    manifest, bodies = fx.miforge_set_manifest(reference=None)
    directory = fx.write_export(tmp_path / "export", manifest, bodies)
    provenance = {
        "source": "https://example.invalid/corpus.jsonl",
        "content_sha256": "e" * 64,
        "retrieved_at": "2026-10-14T00:00:00Z",
    }
    with support.session_for(engine) as session:
        acquire.from_dir(
            session,
            directory,
            parts_root=parts_root,
            source="external_files",
            external_provenance=provenance,
        )
        before = index.parts_snapshot(session)

    with support.session_for(engine) as session:
        index.reindex_parts(session, root=parts_root)
        after = index.parts_snapshot(session)

    assert after == before
    assert after["parts"][0]["source"] == "external_files"
    assert after["parts"][0]["external_provenance"] == provenance


class TestGroupBIsOutOfScopeAndThisSaysSoInWords:
    """``mf_review_bindings`` and ``mf_review_candidate_refs`` are **durable run state**.

    They are not an index and must never be rebuilt: there is no local file that can
    reconstruct a miDataworks queue id or an ``external_id`` -> ``item_id`` mapping, because
    **the authority for a decision is miDataworks**. Losing them loses the ability to read
    decisions back - which is a real loss, and the honest response is to say so rather than
    to invent a rebuild that would make miForge the authority for somebody else's decisions.
    """

    def test_the_two_groups_are_declared_separately(self):
        assert PART_INDEX_TABLES == ("mf_contract_parts", "mf_part_splits")
        assert REVIEW_STATE_TABLES == ("mf_review_bindings", "mf_review_candidate_refs")
        assert not set(PART_INDEX_TABLES) & set(REVIEW_STATE_TABLES)

    def test_the_rebuild_touches_only_group_a(self, engine, seeded):
        """Asserted by reading the module's source for the tables it writes, because a
        rebuild that *happened* to leave Group B alone on an empty database would pass a
        behavioural check while still being wired to clear them."""
        import ast
        from pathlib import Path as _Path

        source = (
            _Path(__file__).resolve().parents[2] / "src" / "services" / "parts" / "index.py"
        ).read_text("utf-8")
        tree = ast.parse(source)
        referenced = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
        assert "REVIEW_BINDINGS" not in referenced
        assert "REVIEW_CANDIDATE_REFS" not in referenced
        assert "CONTRACT_PARTS" in referenced

    def test_rebuilding_does_not_empty_a_review_binding(self, engine, seeded):
        """The behavioural half: a binding written before a rebuild is still there after."""
        import sqlalchemy as sa

        from services.parts.tables import REVIEW_BINDINGS
        from services.review import binding as binding_service

        with support.session_for(engine) as session:
            binding_service.record_binding(
                session,
                run_id="run-1",
                queue_id="rq0001",
                question="Is this completion good?",
                label_set=["good", "bad"],
            )
        with support.session_for(engine) as session:
            index.reindex_parts(session, root=seeded)
        with support.session_for(engine) as session:
            count = session.execute(
                sa.select(sa.func.count()).select_from(REVIEW_BINDINGS)
            ).scalar_one()
        assert count == 1, "a rebuild emptied durable run state"


class TestTheManifestWinsOverTheIndex:
    def test_a_divergent_row_is_corrected_from_the_manifest_and_logged(
        self, engine, seeded, caplog
    ):
        """Foundation 4.5's rule applied to parts. **Logged rather than silently corrected**
        because a divergence means something wrote the index without going through the
        manifest, or edited a manifest in place - both are bugs, and a silent self-heal is
        how they stay unfound."""
        import logging

        import sqlalchemy as sa

        from services.parts.tables import CONTRACT_PARTS

        with support.session_for(engine) as session:
            part_id = index.list_parts(session)[0]["id"]
            session.execute(
                sa.update(CONTRACT_PARTS)
                .where(CONTRACT_PARTS.c.id == part_id)
                .values(row_count=999_999)
            )

        with (
            caplog.at_level(logging.WARNING, logger="miforge.parts.index"),
            support.session_for(engine) as session,
        ):
            payload = index.resolve_part(session, part_id)

        assert payload["row_count"] != 999_999
        # Read off the RECORD's structured fields, not `caplog.text`: ADR-021 puts the
        # detail in `extra`, which never reaches the formatted message - so a text search
        # would pass against a log line that named no field at all.
        divergences = [
            record for record in caplog.records if record.message == index.DIVERGENCE_EVENT
        ]
        assert len(divergences) == 1
        assert divergences[0].divergent_fields == ["row_count"]
        assert divergences[0].resolution == "manifest_wins"
        assert divergences[0].detail["row_count"]["index"] == 999_999

        with support.session_for(engine) as session:
            assert (
                index.get_part(session, part_id)["row_count"] == payload["row_count"]
            ), "the row was not corrected, so the same divergence is reported forever"

    def test_an_indexed_part_whose_manifest_is_gone_refuses_rather_than_being_served(
        self, engine, seeded
    ):
        """**The index silently outliving the artifact is the exact inversion of ADR-004.**"""
        from core.errors import Refusal

        with support.session_for(engine) as session:
            part = index.list_parts(session)[0]
        (Path(part["local_path"]) / "midataworks-dataset-version.json").unlink()

        with support.session_for(engine) as session, pytest.raises(Refusal) as caught:
            index.resolve_part(session, part["id"])
        assert caught.value.code == "manifest_missing_for_indexed_part"


# The strict xfail that guarded this is removed: it fired the moment the wiring landed
# in services/reindex.reindex, which is what a strict xfail is for. It asserts the
# CALL, not a grep - a grep proves reindex_parts exists, not that anything calls it.
def test_the_global_reindex_command_rebuilds_the_part_index():
    """ADR-016 #1: *declaring a mechanism is not wiring it.*

    A grep would show ``reindex_parts`` exists; that proves a function exists, not that
    anything depends on it. This asserts the **call**, which is the only form that bites.
    """
    import ast
    from pathlib import Path as _Path

    source = (_Path(__file__).resolve().parents[2] / "src" / "services" / "reindex.py").read_text(
        "utf-8"
    )
    called = {
        node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
    }
    assert "reindex_parts" in called
