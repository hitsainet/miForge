"""Reindex equivalence - ADR-016's fourth structural test  (task 4.4).

**This is what makes ADR-004's central claim true rather than asserted.** The claim is
that the filesystem is the source of truth and the database is a rebuildable index; the
only way to know is to drop the index, rebuild it from manifests alone, and require the
reads to answer identically.

**The seam, and what the API-level test will have to call.** The endpoints do not exist
yet, so the equivalence is asserted at the service layer:
:func:`services.reindex.index_snapshot` - which is ``list_checkpoints`` plus
``list_adapters`` plus ``checkpoint_tree``. When ``/api/v1`` lands, the API-level test
asserts the same equality over the HTTP bodies, and those bodies must be built from these
same functions. **An endpoint that assembles its own body from the ORM rows is outside
this guarantee** and would pass this test while breaking the thing it protects.

**Two traps this file is written to avoid.**

1. *Construction-green.* Comparing two snapshots proves nothing if both are empty, or if
   the drop was a no-op. So the fixture's coverage is asserted first, and the drop is
   asserted to have actually emptied the index.
2. *A subset comparison.* ``index_snapshot`` enumerates the seam in one place, so adding
   a query without adding it here is not possible by forgetting.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from models.domain_state import DomainState
from services.reindex import drop_index, index_snapshot, reindex
from tests.support import db as support


@pytest.fixture()
def engine(tmp_path: Path):
    return support.migrated_engine(tmp_path / "db")


@pytest.fixture()
def seeded_root(tmp_path: Path) -> Path:
    """An artifact root covering every shape the index has to carry.

    Deliberately not one checkpoint: single-case coverage is one of the named reasons a
    suite stays green over a real bug (ADR-016). This root holds a root checkpoint, a
    child, a from-scratch run asserting a null domain, a nonconforming one missing the key
    altogether, a merged checkpoint with its second ancestor, and an adapter.
    """
    root = tmp_path / "checkpoints"
    root.mkdir()

    support.write_checkpoint(
        root,
        support.checkpoint_manifest(
            step=100, domain="support", test=support.score(provider="in-process")
        ),
    )
    support.write_checkpoint(
        root,
        support.checkpoint_manifest(
            step=200,
            domain="support",
            parent="LFM2-1.2B:run-1:step-100",
            test=support.score(),
            retention=support.score(set_name="general", value=0.74, floor=0.70),
        ),
    )
    support.write_checkpoint(
        root,
        support.checkpoint_manifest(run_id="run-scratch", step=10, domain=None),
    )

    # Missing the required `domain` - written through `write_canonical` after deleting the
    # key, because the fixture builder validates and would refuse it.
    from core.canonical import write_canonical

    nonconforming = support.checkpoint_manifest(run_id="run-old", step=5)
    del nonconforming["domain"]
    write_canonical(root / "run-old" / "step-000005" / "manifest.json", nonconforming)

    support.write_checkpoint(
        root,
        support.checkpoint_manifest(
            run_id="run-merged",
            step=300,
            parent="LFM2-1.2B:run-1:step-200",
            merged_adapter={
                "name": "LFM2-1.2B:run-2:adapter-50",
                "step": 50,
                "manifest_sha256": "c" * 64,
            },
        ),
    )
    support.write_adapter(root, support.adapter_manifest())
    return root


def test_the_fixture_covers_every_shape_before_anything_is_compared(
    engine, seeded_root: Path
) -> None:
    """An equivalence test over a thin fixture is a green with no content."""
    with support.session_for(engine) as session:
        reindex(session, root=seeded_root)
        snapshot = index_snapshot(session, root=seeded_root)

    assert len(snapshot["checkpoints"]) == 5
    assert len(snapshot["adapters"]) == 1
    states = {row["domain_state"] for row in snapshot["checkpoints"]}
    assert states == {
        DomainState.RECORDED.value,
        DomainState.ASSERTED_NULL.value,
        DomainState.ABSENT.value,
    }, states
    assert {row["domain_state"] for row in snapshot["adapters"]} == {DomainState.NOT_IN_KIND.value}
    edge_kinds = {edge["edge"] for edge in snapshot["tree"]["edges"]}
    assert edge_kinds == {"parent", "merged_adapter", "base"}, edge_kinds
    # The merged checkpoint has TWO ancestors; a tree that showed one would look plausible.
    merged_edges = [
        e for e in snapshot["tree"]["edges"] if e["child"].startswith("LFM2-1.2B:run-merged")
    ]
    assert len(merged_edges) == 2, merged_edges


def test_dropping_and_reindexing_produces_identical_responses(engine, seeded_root: Path) -> None:
    """**The structural test.** Seed, snapshot, drop, reindex, compare."""
    with support.session_for(engine) as session:
        reindex(session, root=seeded_root)
        before = index_snapshot(session, root=seeded_root)

    with support.session_for(engine) as session:
        drop_index(session)
        # Assert the drop actually dropped. Without this the comparison below would pass
        # against a `drop_index` that did nothing - the construction-green failure.
        emptied = index_snapshot(session, root=seeded_root)
        assert emptied["checkpoints"] == [] and emptied["adapters"] == []
        # Nodes and edges, named rather than compared against a whole-payload literal: the
        # forest grew `page` and `index_stale` when `checkpoint_tree` was widened for
        # R-02.29 on 2026-10-14, and a literal of the whole response would have had to be
        # re-typed for every later field. What this assertion is for is that the drop
        # actually dropped.
        assert emptied["tree"]["nodes"] == [] and emptied["tree"]["edges"] == []
        assert emptied["tree"]["page"]["total"] == 0

    with support.session_for(engine) as session:
        reindex(session, root=seeded_root)
        after = index_snapshot(session, root=seeded_root)

    assert after == before


def test_a_dropped_index_that_is_not_reindexed_differs(engine, seeded_root: Path) -> None:
    """Task 10.2's control, as a permanent test: the equality above has to be falsifiable.

    If an empty index compared equal to a populated one, the test above would be green no
    matter what ``reindex`` did.
    """
    with support.session_for(engine) as session:
        reindex(session, root=seeded_root)
        before = index_snapshot(session, root=seeded_root)

    with support.session_for(engine) as session:
        drop_index(session)
        after = index_snapshot(session, root=seeded_root)

    assert after != before


def test_the_snapshot_enumerates_the_whole_read_seam(engine, seeded_root: Path) -> None:
    """Adding a query to the seam without adding it here must not be possible by omission.

    ``index_snapshot``'s keys are compared against the module's public read functions, so
    a new one that nobody wired into the snapshot fails here rather than quietly sitting
    outside the equivalence guarantee.
    """
    import services.reindex as module

    read_functions = {
        name
        for name in dir(module)
        if name.startswith(("list_", "checkpoint_tree"))
        and callable(getattr(module, name))
        and not name.startswith("_")
    }
    with support.session_for(engine) as session:
        reindex(session, root=seeded_root)
        snapshot = index_snapshot(session, root=seeded_root)

    covered = {"list_checkpoints", "list_adapters", "checkpoint_tree"}
    assert read_functions == covered, (
        f"the read seam changed: {read_functions ^ covered}. Add it to index_snapshot "
        f"or it sits outside task 4.4's equivalence guarantee."
    )
    assert set(snapshot) == {"checkpoints", "adapters", "tree"}
