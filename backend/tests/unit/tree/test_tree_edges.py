"""One case per rule in FTDD section 4.3  (task 3.7).

Every rule in that table, plus the two distinctions it exists to protect - ``root`` vs
``truncated`` (M9) and ``not_loaded`` vs ``truncated`` - plus the three-valued merged
verification (E-12, E-12a) and the longer cycle the visited set catches.

The edge functions are pure, so these are literals rather than fixtures: there is no
writer, no schema and no index between the input and the answer, which is the one place in
this feature where a literal cannot drift away from anything.
"""

from __future__ import annotations

import pytest

from schemas.tree import EDGE_STATES
from services.tree_edges import (
    merged_verification,
    resolve_base,
    resolve_merged,
    resolve_parent,
)

CHILD = "LFM2-1.2B:run-a:step-200"
PARENT = "LFM2-1.2B:run-a:step-100"
ADAPTER = "LFM2-1.2B:run-b:adapter-50"


class TestEdgeOneTheCheckpointsParent:
    def test_a_null_parent_is_ROOT_and_not_truncated(self) -> None:
        """**M9.** One means *this is a root*; the other means *the record stops here*.

        Conflating them "would make a partial tree look complete" (F5 FR-005.39 / its
        E-11). If this assertion can be satisfied by returning ``truncated``, F5's own fix
        is unpinned.
        """
        edge = resolve_parent(CHILD, None, frozenset(), frozenset())
        assert edge.state == "root"
        assert edge.target is None
        assert edge.state != "truncated"

    def test_a_resolvable_parent_is_ok(self) -> None:
        edge = resolve_parent(CHILD, PARENT, frozenset({CHILD, PARENT}), frozenset({CHILD, PARENT}))
        assert (edge.target, edge.state) == (PARENT, "ok")

    def test_an_unresolvable_parent_is_TRUNCATED_and_names_what_it_wanted(self) -> None:
        edge = resolve_parent(CHILD, PARENT, frozenset({CHILD}), frozenset({CHILD}))
        assert (edge.target, edge.state) == (PARENT, "truncated")

    def test_a_parent_outside_the_page_is_NOT_LOADED_and_not_truncated(self) -> None:
        """A paging artefact must not read as a broken lineage (FTDD section 5.2)."""
        edge = resolve_parent(CHILD, PARENT, frozenset({CHILD}), frozenset({CHILD, PARENT}))
        assert (edge.target, edge.state) == (PARENT, "not_loaded")
        assert edge.state != "truncated"

    def test_a_self_parent_is_BROKEN_and_the_node_still_names_it(self) -> None:
        edge = resolve_parent(CHILD, CHILD, frozenset({CHILD}), frozenset({CHILD}))
        assert (edge.target, edge.state) == (CHILD, "broken")

    def test_a_self_parent_is_broken_even_when_it_would_otherwise_resolve(self) -> None:
        """The self-check runs before the lookup, so a node present in its own page does
        not resolve to itself as ``ok``."""
        edge = resolve_parent(CHILD, CHILD, frozenset({CHILD, PARENT}), frozenset({CHILD, PARENT}))
        assert edge.state == "broken"


class TestEdgeTwoTheAdaptersBase:
    def test_resolve_base_TAKES_NO_RUN_ID(self) -> None:
        """**M8**, made a signature fact rather than a convention.

        F8 FR-008.34 places an adapter under *the base its own manifest names, not the
        run's current head*. The function cannot see a ``run_id``, so joining on the head
        requires editing the signature - visible in review as well as red here.
        """
        import inspect

        assert "run_id" not in inspect.signature(resolve_base).parameters

    def test_a_resolvable_base_is_ok(self) -> None:
        edge = resolve_base(ADAPTER, PARENT, frozenset({ADAPTER, PARENT}), frozenset())
        assert (edge.target, edge.state) == (PARENT, "ok")

    def test_an_unresolvable_base_is_UNPARENTED_and_names_the_base_it_wants(self) -> None:
        """FTDD section 4.3's edge 2: ``unparented``, **not** ``truncated``.

        An adapter with no reachable base is not a chain that stops early - it is an
        artifact that cannot be placed at all.
        """
        edge = resolve_base(ADAPTER, PARENT, frozenset({ADAPTER}), frozenset({ADAPTER}))
        assert (edge.target, edge.state) == (PARENT, "unparented")
        assert edge.state != "truncated"

    def test_a_base_outside_the_page_is_NOT_LOADED(self) -> None:
        edge = resolve_base(ADAPTER, PARENT, frozenset({ADAPTER}), frozenset({ADAPTER, PARENT}))
        assert (edge.target, edge.state) == (PARENT, "not_loaded")

    @pytest.mark.parametrize("base", [None, ""])
    def test_an_adapter_with_no_base_at_all_is_unparented_not_root(self, base: str | None) -> None:
        edge = resolve_base(ADAPTER, base, frozenset({ADAPTER}), frozenset({ADAPTER}))
        assert (edge.target, edge.state) == (None, "unparented")

    def test_a_self_based_adapter_is_broken(self) -> None:
        edge = resolve_base(ADAPTER, ADAPTER, frozenset({ADAPTER}), frozenset({ADAPTER}))
        assert (edge.target, edge.state) == (ADAPTER, "broken")


class TestEdgeThreeTheMergedAdapter:
    def test_a_resolvable_merged_adapter_is_ok(self) -> None:
        edge = resolve_merged(
            CHILD, {"name": ADAPTER, "step": 50}, frozenset({CHILD, ADAPTER}), frozenset()
        )
        assert (edge.target, edge.state) == (ADAPTER, "ok")

    def test_an_unresolvable_merged_adapter_is_UNRESOLVED_and_names_it(self) -> None:
        edge = resolve_merged(
            CHILD, {"name": ADAPTER, "step": 50}, frozenset({CHILD}), frozenset({CHILD})
        )
        assert (edge.target, edge.state) == (ADAPTER, "unresolved")

    def test_a_merged_adapter_outside_the_page_is_NOT_LOADED(self) -> None:
        edge = resolve_merged(
            CHILD, {"name": ADAPTER, "step": 50}, frozenset({CHILD}), frozenset({CHILD, ADAPTER})
        )
        assert edge.state == "not_loaded"

    def test_a_checkpoint_naming_ITSELF_as_its_merged_adapter_is_broken(self) -> None:
        edge = resolve_merged(CHILD, {"name": CHILD, "step": 1}, frozenset({CHILD}), frozenset())
        assert (edge.target, edge.state) == (CHILD, "broken")

    @pytest.mark.parametrize("block", [{}, {"name": None}, {"name": ""}, {"name": 7}])
    def test_a_merged_block_with_no_usable_name_is_unresolved_naming_nothing(
        self, block: dict[str, object]
    ) -> None:
        """Refuse to guess rather than coerce: ``str(7)`` would fabricate an adapter name."""
        edge = resolve_merged(CHILD, block, frozenset({CHILD}), frozenset({CHILD}))
        assert (edge.target, edge.state) == (None, "unresolved")


class TestMergedVerificationIsThreeValued:
    """E-12 and E-12a. ``manifest_sha256`` is ``required`` **and** nullable, so there are
    three answers and a boolean has two.

    The middle case is the one several documents collapsed into the third: a key present
    and ``null`` is a **conforming** manifest whose edge is deliberately unverified, and
    reporting it as a missing required key is the exact conflation FR-009.17 rule 3 forbids
    for ``domain``.
    """

    def test_a_hash_is_verified(self) -> None:
        assert merged_verification({"name": ADAPTER, "manifest_sha256": "a" * 64}) == "verified"

    def test_a_present_NULL_is_conforming_and_unverified(self) -> None:
        assert merged_verification({"name": ADAPTER, "manifest_sha256": None}) == "unverified_null"

    def test_an_ABSENT_key_is_nonconforming_and_is_a_DIFFERENT_answer(self) -> None:
        assert merged_verification({"name": ADAPTER}) == "key_absent"

    def test_the_three_answers_are_all_distinct(self) -> None:
        """The assertion that makes the two above more than a pair of spellings."""
        answers = {
            merged_verification({"manifest_sha256": "a" * 64}),
            merged_verification({"manifest_sha256": None}),
            merged_verification({}),
        }
        assert len(answers) == 3


class TestTheStateVocabularyIsClosed:
    def test_every_state_an_edge_function_can_return_is_in_EDGE_STATES(self) -> None:
        """So a new state cannot be introduced without appearing in the exported union the
        frontend's ``satisfies Record<EdgeState, ...>`` map is checked against."""
        produced = {
            resolve_parent(CHILD, None, frozenset(), frozenset()).state,
            resolve_parent(CHILD, CHILD, frozenset(), frozenset()).state,
            resolve_parent(CHILD, PARENT, frozenset({PARENT}), frozenset({PARENT})).state,
            resolve_parent(CHILD, PARENT, frozenset(), frozenset({PARENT})).state,
            resolve_parent(CHILD, PARENT, frozenset(), frozenset()).state,
            resolve_base(ADAPTER, PARENT, frozenset(), frozenset()).state,
            resolve_merged(CHILD, {"name": ADAPTER}, frozenset(), frozenset()).state,
        }
        # Equality, not a subset: every value in the exported union is reachable from one
        # of the three resolutions, so the union carries no state nothing can produce and
        # no state is produced that the union does not name.
        assert produced == set(EDGE_STATES)


class TestALongerCycleIsNOTAnEdgeLevelFact:
    def test_resolve_parent_reports_ok_for_both_halves_of_a_two_cycle(self) -> None:
        """Which is why the visited set in ``compose`` is load-bearing and not a nicety.

        ``a -> b -> a``: neither edge is a self-edge, both targets resolve, so
        :func:`resolve_parent` answers ``ok`` twice - correctly, because a cycle is a
        property of the walk and not of either edge. ``test_tree_compose.py`` asserts the
        node is marked ``broken`` and **stays visible** (FR-009.30); without this test it
        would look as though the edge resolver were supposed to catch it.
        """
        a, b = "LFM2-1.2B:run-c:step-10", "LFM2-1.2B:run-c:step-20"
        page = frozenset({a, b})
        assert resolve_parent(a, b, page, page).state == "ok"
        assert resolve_parent(b, a, page, page).state == "ok"
