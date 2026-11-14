"""The lifecycle policy, decided without touching a disk  (OQ-4).

The operator's decision: **keep the last N saves, every Mth save, and unconditionally every
checkpoint that is a parent in the lineage tree.** Each clause gets its own class, and the
lineage clause gets the one whose failure is a correctness failure rather than a disk one.

**Why these are unit tests and not integration tests.** ``plan`` is pure, so every case
below is a value in and a value out - which is what *decide, then act* buys. The
corresponding integration file proves the decision is carried out and that a lineage walk
still resolves afterwards; this file proves the decision itself, including the cases a
filesystem fixture makes awkward (a parent that is not under the root, a manifest naming
itself).
"""

from __future__ import annotations

import pytest

from core.errors import Refusal
from models.domain_state import ArtifactKind
from services.checkpoints.lifecycle import (
    DELETE_NOT_RETAINED,
    KEEP_EVERY_MTH,
    KEEP_KIND_NEVER_PRUNED,
    KEEP_LATEST,
    KEEP_LINEAGE_ANCESTOR,
    KEEP_REASON_PRECEDENCE,
    KEEP_RUN_ACTIVE,
    KEEP_WITHIN_LAST_N,
    ArtifactRef,
    LifecyclePolicy,
    ancestor_names,
    plan,
)

BASE = "LiquidAI/LFM2-1.2B"


def checkpoint(step: int, *, run_id: str = "run-a", parent: str | None = None) -> ArtifactRef:
    """A checkpoint ref in F5's **real** rendering - padded path, unpadded name.

    Spelled through the real renderers rather than by hand so a fixture cannot drift into a
    plausible-looking format the rest of the system does not produce. *That is how a
    refusal's test ends up passing while testing a different refusal.*
    """
    from services.checkpoints.paths import qualified_name, step_dir_name

    return ArtifactRef(
        name=qualified_name(BASE, run_id, step),
        kind=ArtifactKind.CHECKPOINT,
        run_id=run_id,
        step=step,
        relative_path=f"{run_id}/{step_dir_name(step)}",
        parent=parent,
    )


def adapter(step: int, *, run_id: str = "run-b", base_name: str | None = None) -> ArtifactRef:
    from training.runtypes.adapters.paths import adapter_name

    return ArtifactRef(
        name=adapter_name(BASE, run_id, step),
        kind=ArtifactKind.ADAPTER,
        run_id=run_id,
        step=step,
        relative_path=f"{run_id}/adapters/step-{step:06d}",
        base_name=base_name,
    )


def decisions(decided) -> dict[str, str]:
    """``name -> action``, for asserting the whole plan at once rather than one member."""
    return {d.name: d.action for d in (*decided.keep, *decided.delete)}


def reasons_for(decided, name: str) -> tuple[str, ...]:
    for d in (*decided.keep, *decided.delete):
        if d.name == name:
            return d.reasons
    raise AssertionError(f"{name} is in neither half of the plan")


class TestTheLastNSaves:
    def test_the_newest_n_survive_and_the_rest_do_not(self) -> None:
        """``keep_every_m`` is set past the end of the run to isolate N - **except for the
        first save**, which is index 0 and so is always an Mth save however large M is.
        That is deliberate (the oldest save is the likeliest branch point) and it is
        asserted on its own below; here it is named so this test reads as exact rather than
        as a near miss."""
        refs = [checkpoint(s) for s in range(100, 1100, 100)]
        decided = plan(refs, LifecyclePolicy(keep_last_n=3, keep_every_m=1000))
        assert sorted(d.step for d in decided.keep) == [100, 800, 900, 1000]
        assert reasons_for(decided, checkpoint(100).name) == (KEEP_EVERY_MTH,)
        assert sorted(d.step for d in decided.delete) == [200, 300, 400, 500, 600, 700]

    def test_n_is_ordinal_over_the_runs_actual_saves_not_arithmetic_on_the_step(self) -> None:
        """A continue starts at an arbitrary step and a repair save lands off the grid.

        ``step % something == 0`` would keep a different fraction of every run and none at
        all of an irregular one. These steps are deliberately not multiples of each other.
        """
        refs = [checkpoint(s) for s in (7, 93, 94, 1001, 1002)]
        decided = plan(refs, LifecyclePolicy(keep_last_n=2, keep_every_m=1000))
        assert sorted(d.step for d in decided.keep) == [7, 1001, 1002]  # 7 is index 0
        assert sorted(d.step for d in decided.delete) == [93, 94]

    def test_a_run_with_fewer_saves_than_n_loses_nothing(self) -> None:
        refs = [checkpoint(s) for s in (100, 200)]
        decided = plan(refs, LifecyclePolicy(keep_last_n=10, keep_every_m=10))
        assert decided.delete == ()


class TestEveryMthSave:
    def test_every_mth_counted_from_the_oldest(self) -> None:
        refs = [checkpoint(s) for s in range(100, 1100, 100)]
        decided = plan(refs, LifecyclePolicy(keep_last_n=1, keep_every_m=4))
        # indices 0, 4, 8 -> steps 100, 500, 900; plus the last one.
        assert sorted(d.step for d in decided.keep) == [100, 500, 900, 1000]

    def test_the_first_save_of_a_run_is_always_among_them(self) -> None:
        """Index 0, which is the save a branch most often points at. Counting from the
        newest instead would make the oldest keeper depend on how many saves have happened
        so far - so the set would churn as a run proceeds."""
        for count in range(1, 12):
            refs = [checkpoint(100 * i) for i in range(1, count + 1)]
            decided = plan(refs, LifecyclePolicy(keep_last_n=1, keep_every_m=3))
            assert 100 in {d.step for d in decided.keep}, f"with {count} saves"

    def test_m_of_one_keeps_everything(self) -> None:
        refs = [checkpoint(s) for s in range(100, 1100, 100)]
        decided = plan(refs, LifecyclePolicy(keep_last_n=1, keep_every_m=1))
        assert decided.delete == ()


class TestTheLineageClause:
    """**The correctness clause.** Everything in this class is about a walk, not a volume.

    R-02.23 lets a run branch from any checkpoint and R-02.29 walks the tree by ``parent``.
    A pruned ancestor does not corrupt anything - :mod:`services.checkpoints.lineage`
    honestly reports ``truncated`` - and that is precisely the harm: a subtree is detached
    from its history and the record says only that it stops there.
    """

    def test_an_ancestor_outside_n_and_m_is_kept_anyway(self) -> None:
        """**The control this feature is judged on.**

        Step 200 of ``run-a`` is not among the last N and is not an Mth save, so both space
        clauses would delete it - and ``run-b`` branched from it. Removing the ancestry
        check from ``plan`` must turn this red.
        """
        branch_point = checkpoint(200).name
        refs = [
            *[checkpoint(s) for s in range(100, 600, 100)],
            checkpoint(50, run_id="run-b", parent=branch_point),
        ]
        decided = plan(refs, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))

        assert decisions(decided)[branch_point] == "keep"
        # The reason is the lineage one AND NOTHING ELSE. That exactness is what makes the
        # assertion a control: if N or M were also keeping it, the test would pass with the
        # ancestry check deleted and would be proving nothing.
        assert reasons_for(decided, branch_point) == (KEEP_LINEAGE_ANCESTOR,)
        # And its unreferenced neighbours still go, so the clause is not simply keeping
        # everything - which is the other way this test could pass vacuously.
        assert decisions(decided)[checkpoint(300).name] == "delete"
        assert decisions(decided)[checkpoint(400).name] == "delete"

    def test_an_adapters_base_protects_that_checkpoint_too(self) -> None:
        """Edge 2. **The one a reader forgets**, and the reason this policy reads adapter
        manifests at all: an adapter names its base, so an adapter creates ancestry."""
        base = checkpoint(200).name
        refs = [
            *[checkpoint(s) for s in range(100, 600, 100)],
            adapter(50, base_name=base),
        ]
        decided = plan(refs, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        assert reasons_for(decided, base) == (KEEP_LINEAGE_ANCESTOR,)

    def test_a_merged_checkpoint_protects_the_adapter_it_baked_in(self) -> None:
        """Edge 3. The adapter is also kept by kind, so the assertion is on the ancestor
        set and on the **precedence** - an operator reading the log must be told the
        load-bearing reason, not the incidental one."""
        baked = adapter(50).name
        merged = ArtifactRef(
            name=f"{BASE}:run-c:step-10",
            kind=ArtifactKind.CHECKPOINT,
            run_id="run-c",
            step=10,
            relative_path="run-c/step-000010",
            merged_adapter_name=baked,
        )
        decided = plan([adapter(50), merged], LifecyclePolicy())
        assert baked in decided.ancestors
        assert reasons_for(decided, baked) == (KEEP_LINEAGE_ANCESTOR, KEEP_KIND_NEVER_PRUNED)

    def test_ancestry_is_computed_over_the_whole_set_before_anything_goes(self) -> None:
        """*Unconditionally*, says the decision. A fixpoint that re-asked "is anything
        still pointing at this" after each delete would walk a chain from its tail and take
        all of it, one defensible step at a time."""
        refs = [
            checkpoint(100),
            checkpoint(200),
            checkpoint(300, parent=checkpoint(200).name),
            checkpoint(400),
            checkpoint(500),
        ]
        decided = plan(refs, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        # 300 is the ONLY thing naming 200, and 300 itself is on its way out. 200 stays
        # regardless: the ancestor set was taken once, over the set as it stands.
        assert decisions(decided)[checkpoint(300).name] == "delete"
        assert reasons_for(decided, checkpoint(200).name) == (KEEP_LINEAGE_ANCESTOR,)

    def test_a_self_parent_does_not_make_an_artifact_immortal(self) -> None:
        """``resolve_parent``'s ``broken`` state, reachable through a copied-in or
        hand-edited manifest. Counting a self-edge as ancestry would let any artifact
        exempt itself from the policy by naming itself."""
        self_named = checkpoint(200)
        rogue = ArtifactRef(
            name=self_named.name,
            kind=ArtifactKind.CHECKPOINT,
            run_id="run-a",
            step=200,
            relative_path=self_named.relative_path,
            parent=self_named.name,
        )
        refs = [checkpoint(100), rogue, checkpoint(300), checkpoint(400)]
        decided = plan(refs, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        assert rogue.name not in decided.ancestors
        assert decisions(decided)[rogue.name] == "delete"

    def test_an_ancestor_outside_the_root_is_still_never_invented(self) -> None:
        """A parent name that resolves to nothing under this root is still a name, and
        ``ancestor_names`` reports it. Nothing here deletes it because nothing here is it -
        the assertion is that the set is built from the edge rather than from a lookup."""
        names = ancestor_names([checkpoint(10, run_id="run-z", parent="elsewhere:run-q:step-9")])
        assert names == {"elsewhere:run-q:step-9"}


class TestTheThingsThatAreNeverTouched:
    def test_latest_is_never_deleted(self) -> None:
        """Deleting it dangles the pointer and leaves a continue with no default step."""
        refs = [checkpoint(s) for s in range(100, 600, 100)]
        # Step 300: index 2, so neither N nor M keeps it. A ``latest`` that is also the
        # newest save would make this test pass with the clause deleted.
        pointed_at = checkpoint(300).name
        decided = plan(
            refs, LifecyclePolicy(keep_last_n=1, keep_every_m=1000), latest_names={pointed_at}
        )
        assert reasons_for(decided, pointed_at) == (KEEP_LATEST,)

    def test_nothing_of_a_non_terminal_run_is_touched(self) -> None:
        """*Never delete while a run is actively writing.* Coarse on purpose: a save
        publishes with one ``rename``, so a sweep that checked a finer flag would still
        race it."""
        refs = [checkpoint(s) for s in range(100, 600, 100)]
        decided = plan(
            refs, LifecyclePolicy(keep_last_n=1, keep_every_m=1000), active_run_ids={"run-a"}
        )
        assert decided.delete == ()
        assert all(KEEP_RUN_ACTIVE in d.reasons for d in decided.keep)

    def test_another_runs_activity_protects_nothing_here(self) -> None:
        """The converse, so the predicate is on the artifact's own run rather than on
        whether *any* run is active."""
        refs = [checkpoint(s) for s in range(100, 600, 100)]
        decided = plan(
            refs, LifecyclePolicy(keep_last_n=1, keep_every_m=1000), active_run_ids={"run-other"}
        )
        # Steps 200, 300, 400: the newest is kept by N and the oldest by M (index 0).
        assert sorted(d.step for d in decided.delete) == [200, 300, 400]

    def test_adapters_are_never_deleted(self) -> None:
        """A scope statement, not an oversight: an adapter is megabytes, is deliberately
        self-describing so it travels without its run log, and is edge 3's target."""
        refs = [adapter(s) for s in range(10, 200, 10)]
        decided = plan(refs, LifecyclePolicy(keep_last_n=1, keep_every_m=1000))
        assert decided.delete == ()
        assert all(KEEP_KIND_NEVER_PRUNED in d.reasons for d in decided.keep)


class TestTheReasonIsDerivedByPrecedence:
    def test_the_primary_reason_is_the_strongest_not_the_first_appended(self) -> None:
        """**M-19b, from F6's mutation sweep.** Its fired-condition array was written in
        registry order, which put ``test_threshold`` before ``retention_floor``, so a
        reader taking element zero reported *"the run reached its target"* for a run that
        had fallen below its floor. A reason derived from array order is derived from
        nothing."""
        latest_and_ancestor = checkpoint(500)
        refs = [
            *[checkpoint(s) for s in range(100, 600, 100)],
            checkpoint(10, run_id="run-b", parent=latest_and_ancestor.name),
        ]
        decided = plan(
            refs,
            LifecyclePolicy(keep_last_n=5, keep_every_m=1),
            latest_names={latest_and_ancestor.name},
            active_run_ids={"run-a"},
        )
        got = reasons_for(decided, latest_and_ancestor.name)
        assert got == (
            KEEP_LINEAGE_ANCESTOR,
            KEEP_LATEST,
            KEEP_RUN_ACTIVE,
            KEEP_WITHIN_LAST_N,
            KEEP_EVERY_MTH,
        )
        primary = next(d.reason for d in decided.keep if d.name == latest_and_ancestor.name)
        assert primary == KEEP_LINEAGE_ANCESTOR

    def test_every_reason_the_plan_can_emit_is_in_the_precedence_table(self) -> None:
        """Otherwise a new reason sorts to nowhere and the primary silently becomes the
        next one down."""
        refs = [
            *[checkpoint(s) for s in range(100, 600, 100)],
            adapter(10, base_name=checkpoint(100).name),
        ]
        decided = plan(
            refs,
            LifecyclePolicy(),
            latest_names={checkpoint(500).name},
            active_run_ids={"run-b"},
        )
        emitted = {r for d in decided.keep for r in d.reasons}
        assert emitted <= set(KEEP_REASON_PRECEDENCE)
        assert emitted == {
            KEEP_LINEAGE_ANCESTOR,
            KEEP_LATEST,
            KEEP_RUN_ACTIVE,
            KEEP_KIND_NEVER_PRUNED,
            KEEP_WITHIN_LAST_N,
            KEEP_EVERY_MTH,
        }, "the fixture must exercise all six or the subset assertion above is vacuous"


class TestTheRunsAreIndependent:
    def test_n_and_m_apply_per_run_not_across_the_root(self) -> None:
        """Otherwise one busy run's hundred saves would consume the whole keep budget and
        a quiet run would lose everything but its ``latest``."""
        refs = [
            *[checkpoint(s, run_id="busy") for s in range(100, 2100, 100)],
            *[checkpoint(s, run_id="quiet") for s in (100, 200)],
        ]
        decided = plan(refs, LifecyclePolicy(keep_last_n=2, keep_every_m=1000))
        kept = {(d.run_id, d.step) for d in decided.keep}
        assert ("quiet", 100) in kept and ("quiet", 200) in kept
        assert ("busy", 1900) in kept and ("busy", 2000) in kept
        assert ("busy", 200) not in kept and ("busy", 1800) not in kept


class TestAnImpossiblePolicyIsRefused:
    @pytest.mark.parametrize("last_n,every_m", [(0, 10), (-1, 10), (10, 0), (10, -3)])
    def test_refused_by_name_rather_than_clamped(self, last_n: int, every_m: int) -> None:
        """ADR-013: refuse with a reason rather than guess. A clamp would turn a
        configuration mistake into a deletion policy nobody chose - and ``latest`` would
        still be protected, so the damage would be bounded and invisible, which is worse."""
        with pytest.raises(Refusal) as caught:
            plan([checkpoint(100)], LifecyclePolicy(keep_last_n=last_n, keep_every_m=every_m))
        assert caught.value.code == "checkpoint_lifecycle_policy_invalid"


class TestThePlanIsReportable:
    def test_it_carries_the_keeps_in_full_with_their_reasons(self) -> None:
        """*Log what was kept and why, not only what was deleted.* A plan that reported
        only its deletions cannot answer "why is this still filling my volume"."""
        refs = [checkpoint(s) for s in range(100, 600, 100)]
        payload = plan(refs, LifecyclePolicy(keep_last_n=2, keep_every_m=1000)).as_dict()
        assert payload["kept"] == 3 and payload["to_delete"] == 2
        assert payload["kept_by_reason"] == {KEEP_WITHIN_LAST_N: 2, KEEP_EVERY_MTH: 1}
        assert {d["reason"] for d in payload["delete"]} == {DELETE_NOT_RETAINED}
        assert payload["policy"] == {"keep_last_n": 2, "keep_every_m": 1000}

    def test_an_empty_root_is_an_empty_plan_rather_than_a_failure(self) -> None:
        decided = plan([], LifecyclePolicy())
        assert decided.keep == () and decided.delete == ()
