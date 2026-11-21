"""``compose()`` over every forest shape  (task 3.8).

Fixtures come from **F5's and F8's real writers**, are indexed by ``reindex`` **from the
manifests on disk**, and the index rows are then fed straight into the pure composer. A
hand-built node dict agrees with the composer's expectations by construction and proves
nothing (ADR-016), which is why nothing in this file writes an ORM row or a manifest
literal.

## The two unverified-edge cases are two tests, deliberately

``MergedAdapter.manifest_sha256`` became ``required`` on 2026-10-13 and is **also
nullable**, so there are three states and not two: a hash, the key present and ``null``
(**conforming** - the edge is legitimately unverified, E-12), and the key absent (a missing
required key, nonconforming - E-12a). **A single "no hash" test would pass against an
implementation that conflated them**, which is the conflation FR-009.17 rule 3 forbids for
``domain``. They are different artifacts and they get different tests.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from services.reindex import list_adapters, list_checkpoints, reindex
from services.tree import compose
from tests.support import db as support
from tests.support import tree as forest


@pytest.fixture()
def engine(tmp_path: Path):  # type: ignore[no-untyped-def]
    return support.migrated_engine(tmp_path / "db")


@pytest.fixture()
def root(tmp_path: Path) -> Path:
    path = tmp_path / "checkpoints"
    path.mkdir()
    return path


def rows(engine, root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], set[str]]:  # type: ignore[no-untyped-def]
    """Index ``root`` from its manifests and return the read seam's own payloads."""
    with support.session_for(engine) as session:
        reindex(session, root=root)
        session.commit()
        checkpoints = list_checkpoints(session)
        adapters = list_adapters(session)
    names = {row["name"] for row in checkpoints} | {row["name"] for row in adapters}
    return checkpoints, adapters, names


def forest_of(engine, root: Path, *, known_run_ids: set[str] | None = None, **kwargs: Any):  # type: ignore[no-untyped-def]
    checkpoints, adapters, names = rows(engine, root)
    return compose(
        checkpoints,
        adapters,
        (
            known_run_ids
            if known_run_ids is not None
            else {row["run_id"] for row in checkpoints} | {row["run_id"] for row in adapters}
        ),
        names,
        **kwargs,
    )


def node(response, name: str):  # type: ignore[no-untyped-def]
    return next(n for n in response.nodes if n.name == name)


# ============================================================================ the shapes


class TestARootAndAChain:
    def test_a_root_asserts_root_and_a_child_resolves_ok(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        saved = forest.chain(root, steps=(100, 200))
        response = forest_of(engine, root)
        assert node(response, saved[0].name).parent.state == "root"
        assert node(response, saved[1].name).parent.target == saved[0].name
        assert node(response, saved[1].name).parent.state == "ok"

    def test_a_branch_gives_one_parent_two_children(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        base = forest.root_checkpoint(root, run_id="run-a", step=100)
        first = forest.branch(root, parent=base.name, run_id="run-b", step=200)
        second = forest.branch(root, parent=base.name, run_id="run-c", step=300)
        response = forest_of(engine, root)
        assert node(response, first.name).parent.target == base.name
        assert node(response, second.name).parent.target == base.name


class TestTruncatedIsNotRoot:
    """**M9.** F5 FR-005.39 / its E-11: *"one means 'the root', the other 'the record stops
    here', and conflating them would make a partial tree look complete."*"""

    def test_the_two_shapes_produce_two_different_states(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        rooted = forest.root_checkpoint(root, run_id="run-rooted", step=10)
        stopped = forest.unresolvable_parent_checkpoint(root)
        response = forest_of(engine, root)
        assert node(response, rooted.name).parent.state == "root"
        assert node(response, stopped.name).parent.state == "truncated"

    def test_a_truncated_edge_NAMES_the_parent_it_could_not_find(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        stopped = forest.unresolvable_parent_checkpoint(root, parent="LFM2-1.2B:run-gone:step-7")
        response = forest_of(engine, root)
        assert node(response, stopped.name).parent.target == "LFM2-1.2B:run-gone:step-7"


class TestAnAdapterJoinsOnItsBase:
    def test_an_adapter_is_a_child_of_the_base_its_own_manifest_names(  # type: ignore[no-untyped-def]
        self, engine, root: Path
    ) -> None:
        """**M8.** F8 FR-008.34: *"not of the run's current head."*"""
        base = forest.root_checkpoint(root, run_id="run-base", step=40)
        # A later, higher checkpoint in a DIFFERENT run, so joining on anything but the
        # manifest's own base would place the adapter somewhere visibly wrong.
        forest.root_checkpoint(root, run_id="run-adapter", step=99)
        published = forest.adapter(root, base=base, run_id="run-adapter", step=50)
        response = forest_of(engine, root)
        adapter_node = node(response, published["name"])
        assert adapter_node.base is not None
        assert (adapter_node.base.target, adapter_node.base.state) == (base.name, "ok")
        # And it has no `parent` edge at all: the adapter manifest has no `parent` field,
        # and emitting one would assert a field R-02.29 says it does not have.
        assert adapter_node.parent is None

    def test_an_adapter_carries_its_bases_manifest_hash(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        base = forest.root_checkpoint(root, run_id="run-base", step=40)
        published = forest.adapter(root, base=base)
        response = forest_of(engine, root)
        assert (
            node(response, published["name"]).base_manifest_sha256
            == published["base"]["manifest_sha256"]
        )

    def test_a_hub_based_adapter_is_UNPARENTED_and_names_the_repo_it_wants(  # type: ignore[no-untyped-def]
        self, engine, root: Path
    ) -> None:
        """A base that is a Hub repo id is not a checkpoint, so it resolves to nothing -
        and that is ``unparented``, not ``truncated`` (FTDD section 4.3's edge 2)."""
        published = forest.adapter(root, base=None)
        response = forest_of(engine, root)
        adapter_node = node(response, published["name"])
        assert adapter_node.base is not None
        assert adapter_node.base.state == "unparented"
        assert adapter_node.base.target == forest.HUB_REPO

    def test_an_adapters_domain_is_NOT_IN_KIND_and_never_its_bases(  # type: ignore[no-untyped-def]
        self, engine, root: Path
    ) -> None:
        """``adapter-manifest-v1.json`` declares no ``domain`` and closes its objects, so a
        conforming adapter manifest cannot carry one. Reading the base's across would be a
        fabricated value (ADR-018) and FPRD F-11 reports R-02.29's field list as
        unsatisfiable for this kind."""
        base = forest.root_checkpoint(root, run_id="run-base", step=40, domain="support")
        published = forest.adapter(root, base=base)
        response = forest_of(engine, root)
        adapter_node = node(response, published["name"])
        assert adapter_node.domain is None
        assert adapter_node.domain_state == "not_in_kind"
        assert adapter_node.domain_display == "not recorded in this artifact kind"
        assert node(response, base.name).domain == "support"


class TestAMergedCheckpointHasBOTHEdges:
    """**M7**, the single most important shape in the feature.

    ``merged_adapter`` was added to ``checkpoint-manifest-v1.json`` because *"a merged
    checkpoint has two ancestors, and ``parent`` carries only the base"*. An
    implementation that drops one undoes that.
    """

    def test_both_edges_are_set(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        base = forest.root_checkpoint(root, run_id="run-base", step=40)
        published = forest.adapter(root, base=base)
        merged = forest.merged_checkpoint(root, base=base, adapter_document=published)
        response = forest_of(engine, root)
        merged_node = node(response, merged.name)
        assert merged_node.parent is not None and merged_node.parent.target == base.name
        assert merged_node.parent.state == "ok"
        assert merged_node.merged is not None and merged_node.merged.target == published["name"]
        assert merged_node.merged.state == "ok"

    def test_the_flat_edge_projection_carries_both_too(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        base = forest.root_checkpoint(root, run_id="run-base", step=40)
        published = forest.adapter(root, base=base)
        merged = forest.merged_checkpoint(root, base=base, adapter_document=published)
        response = forest_of(engine, root)
        edges = {(e.child, e.parent, e.edge) for e in response.edges}
        assert (merged.name, base.name, "parent") in edges
        assert (merged.name, published["name"], "merged_adapter") in edges

    def test_an_unresolvable_merged_adapter_is_unresolved_and_the_parent_edge_SURVIVES(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """The two edges are independent: a missing adapter must not cost the base edge."""
        base = forest.root_checkpoint(root, run_id="run-base", step=40)
        published = forest.adapter(root, base=base, run_id="run-ad", step=50)
        merged = forest.merged_checkpoint(root, base=base, adapter_document=published)
        # Index everything except the adapter directory, by removing it before the scan.
        import shutil

        shutil.rmtree(root / "run-ad")
        response = forest_of(engine, root)
        merged_node = node(response, merged.name)
        assert merged_node.merged is not None and merged_node.merged.state == "unresolved"
        assert merged_node.merged.target == published["name"]
        assert merged_node.parent is not None and merged_node.parent.state == "ok"


class TestTheTwoUnverifiedEdgeCasesAreDifferentArtifacts:
    """E-12 and E-12a. See the module docstring - two tests, never one."""

    def test_a_hash_makes_the_edge_VERIFIED(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        base = forest.root_checkpoint(root, run_id="run-base", step=40)
        published = forest.adapter(root, base=base)
        merged = forest.merged_checkpoint(root, base=base, adapter_document=published)
        response = forest_of(engine, root)
        assert node(response, merged.name).merged_verification == "verified"

    def test_a_present_NULL_hash_is_CONFORMING_and_deliberately_unverified(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        base = forest.root_checkpoint(root, run_id="run-base", step=40)
        published = forest.adapter(root, base=base)
        merged = forest.merged_checkpoint(
            root, base=base, adapter_document=published, manifest_sha256=None
        )
        response = forest_of(engine, root)
        merged_node = node(response, merged.name)
        assert merged_node.merged_verification == "unverified_null"
        # And the edge still resolves: conforming-but-unverified is not a broken edge.
        assert merged_node.merged is not None and merged_node.merged.state == "ok"

    def test_an_ABSENT_hash_key_is_NONCONFORMING_and_a_different_answer(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        base = forest.root_checkpoint(root, run_id="run-base", step=40)
        published = forest.adapter(root, base=base)
        merged = forest.merged_checkpoint(
            root, base=base, adapter_document=published, manifest_sha256=forest.KEY_ABSENT
        )
        response = forest_of(engine, root)
        merged_node = node(response, merged.name)
        assert merged_node.merged_verification == "key_absent"
        assert merged_node.merged_verification != "unverified_null"
        assert merged_node.merged is not None and merged_node.merged.state == "ok"


class TestDomainCarriesFourStatesAndCollapsesNone:
    """**M6**, which pins the BRD's 2026-10-13 nullable-but-required decision."""

    def test_a_recorded_domain_an_asserted_null_and_an_ABSENT_key_are_three_answers(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        recorded = forest.root_checkpoint(root, run_id="run-named", step=10, domain="support")
        asserted = forest.root_checkpoint(root, run_id="run-scratch", step=20, domain=None)
        absent = forest.checkpoint_without_domain_key(root)
        response = forest_of(engine, root)

        assert node(response, recorded.name).domain_state == "recorded"
        assert node(response, asserted.name).domain_state == "asserted_null"
        assert node(response, absent.name).domain_state == "absent"
        # The values are all None for two of the three, which is exactly why the STATE is
        # what the view reads: `domain` alone cannot tell them apart.
        assert node(response, asserted.name).domain is None
        assert node(response, absent.name).domain is None
        assert (
            node(response, asserted.name).domain_state != node(response, absent.name).domain_state
        )

    def test_each_state_has_its_own_sentence(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        """One rendering, from ``models.domain_state.render_domain``, so four states do not
        acquire one presentation per screen. ``asserted_null`` is "no domain" and ``absent``
        is **not** - a blank reads as "no domain", which is what the other state means."""
        asserted = forest.root_checkpoint(root, run_id="run-scratch", step=20, domain=None)
        absent = forest.checkpoint_without_domain_key(root)
        response = forest_of(engine, root)
        assert node(response, asserted.name).domain_display == "no domain"
        assert node(response, absent.name).domain_display == "not recorded"


class TestAnUnindexedRunIsStillANode:
    """**M10.** ``reindex`` indexes a valid manifest whose ``run_id`` has no run row (F5's
    OQ-4) and FR-009.28 renders it; a checkpoint silently absent from the tree is strictly
    worse than one whose run is unknown."""

    def test_the_node_is_present_and_marked_rather_than_dropped(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        indexed = forest.root_checkpoint(root, run_id="run-known", step=10)
        orphan = forest.root_checkpoint(root, run_id="run-copied-in", step=20)
        response = forest_of(engine, root, known_run_ids={"run-known"})
        assert {n.name for n in response.nodes} == {indexed.name, orphan.name}
        assert node(response, indexed.name).run_indexed is True
        assert node(response, orphan.name).run_indexed is False

    def test_the_page_total_counts_it(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        forest.root_checkpoint(root, run_id="run-copied-in", step=20)
        response = forest_of(engine, root, known_run_ids=set())
        assert response.page.total == 1


class TestMalformedEdgesAreNamedNotFollowed:
    """FR-009.30: no infinite recursion, no silent drop."""

    def test_a_self_parent_is_broken_and_the_node_is_still_there(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        published = forest.self_parented_checkpoint(root)
        response = forest_of(engine, root)
        assert published.name in {n.name for n in response.nodes}
        assert node(response, published.name).parent.state == "broken"

    def test_a_LONGER_cycle_is_caught_by_the_visited_set_and_both_nodes_survive(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """``a -> b -> a``. Neither edge is a self-edge, so ``resolve_parent`` answers ``ok``
        for both (``test_tree_edges.py`` asserts that) - the cycle is a property of the walk
        and ``compose``'s visited set is what catches it."""
        a, b = forest.cycle_of_two(root)
        response = forest_of(engine, root)
        assert {a.name, b.name} <= {n.name for n in response.nodes}
        assert node(response, a.name).parent.state == "broken"
        assert node(response, b.name).parent.state == "broken"
        # The edge still names the ancestor, so the row can say which walk broke.
        assert node(response, a.name).parent.target == b.name

    def test_a_self_based_adapter_is_broken_and_still_visible(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        """Constructed by composing directly, because F8's writer derives the adapter's
        name from its base and cannot produce one that names itself - the artifact is
        reachable only through a hand-edited manifest, so the composer is where the rule is
        provable."""
        name = "LFM2-1.2B:run-x:adapter-5"
        response = compose(
            [],
            [
                {
                    "name": name,
                    "run_id": "run-x",
                    "step": 5,
                    "domain": None,
                    "domain_state": "not_in_kind",
                    "domain_display": "not recorded in this artifact kind",
                    "base_name": name,
                    "base_manifest_sha256": None,
                    "manifest_sha256": "a" * 64,
                    "test": None,
                    "path": "run-x/adapter-000005",
                }
            ],
            {"run-x"},
            {name},
        )
        assert [n.name for n in response.nodes] == [name]
        assert response.nodes[0].base is not None
        assert response.nodes[0].base.state == "broken"


class TestTheFourthFactTheTestScore:
    def test_the_score_rides_VERBATIM_with_its_sample_floor_and_provider(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        """R-02.29's fourth fact. Nothing is defaulted: ``provider`` is nullable *and* not
        ``required``, so only passing the record through preserves whether it was there."""
        from tests.support.checkpoints import score

        published = forest.root_checkpoint(
            root,
            run_id="run-scored",
            step=10,
            test=score(value=0.77, n=250, floor=0.7, provider="in-process:lfm2-1.2b"),
        )
        response = forest_of(engine, root)
        assert node(response, published.name).test == {
            "set": "contract-clauses-test",
            "score": 0.77,
            "n": 250,
            "floor": 0.7,
            "provider": "in-process:lfm2-1.2b",
        }

    def test_an_unscored_artifact_reports_the_score_absent_rather_than_zero(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        published = forest.root_checkpoint(root, run_id="run-unscored", step=10)
        response = forest_of(engine, root)
        assert node(response, published.name).test is None


class TestWhatTheResponseRefusesToCarry:
    def test_there_is_no_DEPTH_field_anywhere(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        """FR-009.26, and the precondition for mutation M17: taking depth from a
        transmitted field requires *adding* one."""
        forest.chain(root, steps=(100, 200))
        response = forest_of(engine, root)
        payload = response.model_dump(mode="json")
        assert "depth" not in payload
        assert all("depth" not in node_payload for node_payload in payload["nodes"])

    def test_there_is_no_PACK_field_anywhere(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        """FPRD F-10: the checkpoint manifest has no pack field (R-02.35), and a join
        through ``run_id`` to invent one would break both FR-009.28 and reindex
        equivalence."""
        forest.root_checkpoint(root, run_id="run-a", step=10)
        payload = forest_of(engine, root).model_dump(mode="json")
        assert all(
            not any("pack" in key for key in node_payload) for node_payload in payload["nodes"]
        )


class TestPaginationIsFlatAndNotLoadedIsNotTruncated:
    def test_a_parent_outside_the_page_is_not_loaded(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        saved = forest.chain(root, run_id="run-p", steps=(100, 200, 300))
        checkpoints, adapters, names = rows(engine, root)
        # Names sort as step-100 < step-200 < step-300, so a page of one holds the root and
        # a page starting at offset 1 holds a child whose parent is outside it.
        response = compose(checkpoints, adapters, {"run-p"}, names, limit=1, offset=1)
        assert [n.name for n in response.nodes] == [saved[1].name]
        assert response.nodes[0].parent.state == "not_loaded"
        assert response.nodes[0].parent.state != "truncated"
        assert response.page.total == 3

    def test_the_page_block_reports_what_was_asked_and_the_full_total(
        self, engine, root: Path
    ) -> None:  # type: ignore[no-untyped-def]
        forest.chain(root, run_id="run-p", steps=(100, 200, 300))
        checkpoints, adapters, names = rows(engine, root)
        response = compose(checkpoints, adapters, {"run-p"}, names, limit=2, offset=0)
        assert (response.page.limit, response.page.offset, response.page.total) == (2, 0, 3)


class TestTheIndexStaleProjectionCannotDisagreeWithTheNodes:
    def test_the_response_list_is_exactly_the_flagged_nodes(self, engine, root: Path) -> None:  # type: ignore[no-untyped-def]
        """Two renderings of one fact. Derived in the same pass so they cannot drift, and
        asserted so that a later hand-maintained list would be red."""
        saved = forest.chain(root, run_id="run-p", steps=(100, 200))
        checkpoints, adapters, names = rows(engine, root)
        response = compose(
            checkpoints, adapters, {"run-p"}, names, index_stale=frozenset({saved[1].name})
        )
        assert response.index_stale == [saved[1].name]
        assert [n.name for n in response.nodes if n.index_stale] == response.index_stale
