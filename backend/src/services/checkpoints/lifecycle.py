"""The checkpoint lifecycle policy - **pure, and it decides nothing it cannot explain**  (OQ-4).

The operator's decision, 2026-10-16: **keep the last N saves, every Mth save, and
unconditionally every checkpoint that is a parent in the lineage tree. Delete the rest.**

**Called "lifecycle", never "retention", and the name is load-bearing.** ``retention`` is
already taken in this codebase and means something else entirely: R-02.17's held-out
retention *set* and the ``retention`` ``ScoreRecord`` on every manifest and both index
tables. A module named ``retention.py`` beside ``Checkpoint.retention`` would have two
readings on its first line, and this project has paid for a conflated vocabulary three
times (the ``step-`` prefix, three signal vocabularies, ``full`` versus ``change``).

------------------------------------------------------------------------------------
THE LINEAGE CLAUSE IS A CORRECTNESS RULE THAT COSTS SPACE, NOT A SPACE RULE
------------------------------------------------------------------------------------

R-02.23 lets a run **branch** from any checkpoint and R-02.29 walks the tree by ``parent``.
Delete a node something else descends from and every walk through it stops: F5's
:mod:`services.checkpoints.lineage` reports ``truncated``, honestly, and an entire subtree
is detached from its history. So the ancestry check is not a heuristic to be tuned
alongside N and M - it **overrides** them, and it is first in the precedence order below.

**Three ancestor edges, and all three are checked**, because an adapter creates ancestry
too:

* a checkpoint's ``parent`` -> a checkpoint (edge 1);
* an adapter's ``base.name`` -> a checkpoint (edge 2) - *this is the one a reader forgets,
  and it is why an adapter manifest is read by this policy at all*;
* a checkpoint's ``merged_adapter.name`` -> an adapter (edge 3).

The vocabulary is :mod:`services.tree_edges`', deliberately, so the policy protects exactly
what the tree traverses rather than a second opinion about it.

**Ancestry is computed once over the whole set, before anything is deleted, and is never
recomputed as deletions proceed.** The operator's word is *unconditionally*. A fixpoint
that re-asked "is anything still pointing at this?" after each delete would walk a chain
from its tail and take the whole thing, one defensible step at a time - the failure mode
that makes an eventual "where did my lineage go" unanswerable.

*Note what this means in production, because it is easy to mis-predict from the test
fixtures.* ``training/runtypes/change.py`` passes ``parent=ctx.record.parent`` - **the
acquisition record's parent, which is constant for a whole run**. So a run's saves do not
form a chain of a hundred parents; they all name the one checkpoint the run continued or
branched from. The parent set over a real root is the set of *branch points*, which is why
~20-30 checkpoints survive rather than ~99. A synthetic linear chain (``write_chain`` in
the test support, which exists to exercise the walk) is all-parents and this policy
correctly keeps all of it.

**A self-edge never protects.** A manifest naming itself as its own ``parent`` is
:func:`services.tree_edges.resolve_parent`'s ``broken`` state, reachable through a
copied-in or hand-edited manifest. Counting it as ancestry would let any artifact make
itself immortal by naming itself.

------------------------------------------------------------------------------------
WHAT THIS MODULE IS NOT
------------------------------------------------------------------------------------

It touches no filesystem, opens no session and reads no clock. Decide, then act: the
decision is a value a test can assert in full, and :mod:`services.checkpoints.pruning`
is the only thing that destroys anything. The three facts it cannot derive -
``latest_names``, ``active_run_ids`` and the policy - arrive as arguments.

**Adapters are never deleted by this policy, and that is a scope statement rather than an
oversight.** The space problem is checkpoints (480-960 GB for one 10,000-step run at
``save_every=100``); an adapter is megabytes, is deliberately self-describing so it can
travel without its run log, and is the target of edge 3. Pruning them would buy nothing
measurable and risk a ``merged_adapter`` edge. They participate here as a *reason to keep a
checkpoint*, which is the whole of their role.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence, Set
from dataclasses import dataclass, field
from typing import Any

from core.errors import refuse
from models.domain_state import ArtifactKind

__all__ = [
    "KEEP_REASON_PRECEDENCE",
    "ArtifactRef",
    "Decision",
    "LifecyclePlan",
    "LifecyclePolicy",
    "ancestor_names",
    "plan",
]

#: Keep because something descends from it. **First, and it overrides N and M.**
KEEP_LINEAGE_ANCESTOR = "lineage_ancestor"
#: Keep because ``latest`` resolves to it. Deleting it leaves a run's pointer dangling and
#: a continue with no default step (FR-005.2's ``resolve_latest``).
KEEP_LATEST = "latest_pointer"
#: Keep because the owning run is not terminal - it is, or may at any moment be, writing.
KEEP_RUN_ACTIVE = "run_not_terminal"
#: Keep because this kind is never pruned. Adapters; see the module docstring.
KEEP_KIND_NEVER_PRUNED = "kind_never_pruned"
#: Keep because it is one of the last N saves of its run.
KEEP_WITHIN_LAST_N = "within_last_n"
#: Keep because it is every Mth save of its run, counted from the oldest.
KEEP_EVERY_MTH = "every_mth_save"

#: **Precedence, and it is read by index into this tuple rather than by the order the
#: reasons happen to be appended in.** M-19b, from F6's sweep: the boundary block's fired
#: conditions were written in registry order, so a reader taking the first element reported
#: *"the run reached its target"* for a run that had fallen below its floor. A reason
#: derived from array order is a reason derived from nothing.
KEEP_REASON_PRECEDENCE: tuple[str, ...] = (
    KEEP_LINEAGE_ANCESTOR,
    KEEP_LATEST,
    KEEP_RUN_ACTIVE,
    KEEP_KIND_NEVER_PRUNED,
    KEEP_WITHIN_LAST_N,
    KEEP_EVERY_MTH,
)

#: The one delete reason. Named rather than left as an empty string so a log line says
#: *why* in the same field either way.
DELETE_NOT_RETAINED = "outside_last_n_and_every_mth"


@dataclass(frozen=True)
class ArtifactRef:
    """One artifact, as its **manifest** describes it (ADR-004).

    Built by :func:`services.checkpoints.pruning.collect` from
    :func:`services.reindex.scan_artifact`, never from an index row: the index has no
    authority over a correctness answer, and which checkpoints are ancestors is one.
    """

    name: str
    kind: ArtifactKind
    run_id: str
    step: int
    #: Relative to the artifact root, as the index stores it. Carried so a decision can be
    #: logged and acted on without a second path computation.
    relative_path: str
    #: Edge 1, for a checkpoint: the checkpoint this one descends from, or ``None``
    #: **asserting** it is a root.
    parent: str | None = None
    #: Edge 2, for an adapter: the base its own manifest names.
    base_name: str | None = None
    #: Edge 3, for a merged checkpoint: the adapter baked into it.
    merged_adapter_name: str | None = None

    @property
    def is_checkpoint(self) -> bool:
        return self.kind is ArtifactKind.CHECKPOINT


@dataclass(frozen=True)
class LifecyclePolicy:
    """N and M. **Configuration with defaults, never literals** - they come from settings.

    The defaults keep roughly twenty checkpoints out of a hundred saves: the last ten, plus
    every tenth counted from the oldest, plus every branch point. That is the sizing in
    OQ-4 - an 800Gi volume against 480-960 GB of checkpoints for one 10,000-step run.
    """

    keep_last_n: int = 10
    keep_every_m: int = 10

    def validated(self) -> LifecyclePolicy:
        """Refuse a policy that cannot mean anything, rather than coerce it (ADR-013).

        ``keep_last_n = 0`` reads as *"keep no recent saves"* and ``keep_every_m = 0`` as a
        modulus by zero. Both are configuration mistakes with a one-line fix, and both are
        the kind of value that a clamp would silently turn into a deletion policy nobody
        chose. ``latest`` would still be protected, so the damage would be bounded and
        invisible, which is worse.
        """
        if self.keep_last_n < 1:
            raise refuse(
                "checkpoint_lifecycle_policy_invalid",
                f"keep_last_n must be at least 1; got {self.keep_last_n}. A policy that "
                f"keeps none of a run's recent saves is not a retention setting, so it is "
                f"refused rather than clamped.",
                keep_last_n=self.keep_last_n,
            )
        if self.keep_every_m < 1:
            raise refuse(
                "checkpoint_lifecycle_policy_invalid",
                f"keep_every_m must be at least 1; got {self.keep_every_m}. 1 means 'keep "
                f"every save'; there is no value meaning 'keep none'.",
                keep_every_m=self.keep_every_m,
            )
        return self


@dataclass(frozen=True)
class Decision:
    """What is to happen to one artifact, and **every reason that applied**.

    ``reason`` is the primary one by :data:`KEEP_REASON_PRECEDENCE`; ``reasons`` is all of
    them. Both, because an operator asking *"why is this still here"* wants the strongest
    answer and an operator asking *"what would change if I lowered N"* wants to know
    whether anything else was also holding it.
    """

    name: str
    kind: ArtifactKind
    run_id: str
    step: int
    relative_path: str
    action: str  # "keep" | "delete"
    reason: str
    reasons: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "run_id": self.run_id,
            "step": self.step,
            "path": self.relative_path,
            "action": self.action,
            "reason": self.reason,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class LifecyclePlan:
    """The whole decision, assertable without touching a disk.

    ``keep`` is carried in full rather than summarised. *"Log what was kept and why"* is
    half of what makes this auditable: a plan that reports only its deletions cannot answer
    the question asked after an incident, which is always one of *"why did it delete that"*
    or *"why is this still filling my volume"* - and a list of deletions answers only the
    first.
    """

    keep: tuple[Decision, ...] = ()
    delete: tuple[Decision, ...] = ()
    #: The ancestor set, so a caller can log it and a test can assert it directly.
    ancestors: frozenset[str] = field(default_factory=frozenset)
    policy: LifecyclePolicy = field(default_factory=LifecyclePolicy)

    @property
    def kept_by_reason(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for decision in self.keep:
            counts[decision.reason] = counts.get(decision.reason, 0) + 1
        return counts

    def as_dict(self) -> dict[str, Any]:
        return {
            "keep": [d.as_dict() for d in self.keep],
            "delete": [d.as_dict() for d in self.delete],
            "kept": len(self.keep),
            "to_delete": len(self.delete),
            "kept_by_reason": self.kept_by_reason,
            "ancestors": sorted(self.ancestors),
            "policy": {
                "keep_last_n": self.policy.keep_last_n,
                "keep_every_m": self.policy.keep_every_m,
            },
        }


def ancestor_names(artifacts: Iterable[ArtifactRef]) -> frozenset[str]:
    """Every name some **other** artifact names as an ancestor, over all three edges.

    A self-edge is excluded; see the module docstring. The target need not exist in the
    set: a name pointing outside the root is still a name this policy will not delete,
    because there is nothing here that could be it.
    """
    found: set[str] = set()
    for artifact in artifacts:
        for target in (artifact.parent, artifact.base_name, artifact.merged_adapter_name):
            if target and target != artifact.name:
                found.add(target)
    return frozenset(found)


def _ordinal_keepers(
    steps: Sequence[int], *, keep_last_n: int, keep_every_m: int
) -> dict[int, tuple[str, ...]]:
    """Which of one run's steps N and M keep, **and which of the two rules kept each**.

    One function returning both, rather than a set plus a re-derivation at the call site.
    A second computation of "was this one of the last N" would be a second answer to the
    same question, and the one the log reports is not the one that decided - ADR-005's rule
    about hashing, applied to a predicate. It is also how a reason silently stops matching
    its action: the action would stay right and the explanation would drift.

    **Both rules are ordinal, counted over the run's actual saves**, not arithmetic on the
    step number. ``save_every`` is configuration and a continue starts at an arbitrary
    step, so ``step % (m * save_every) == 0`` would keep a different fraction of every run
    and none at all of a run whose saves are irregular - a repair save, a branch's step 0.

    Counted **from the oldest**, so index 0 - which is the save a branch most often points
    at - is always among the keepers.
    """
    ascending = sorted(steps)
    recent = set(ascending[-keep_last_n:])
    keepers: dict[int, tuple[str, ...]] = {}
    for index, step in enumerate(ascending):
        reasons: list[str] = []
        if step in recent:
            reasons.append(KEEP_WITHIN_LAST_N)
        if index % keep_every_m == 0:
            reasons.append(KEEP_EVERY_MTH)
        if reasons:
            keepers[step] = tuple(reasons)
    return keepers


def plan(
    artifacts: Iterable[ArtifactRef],
    policy: LifecyclePolicy | None = None,
    *,
    latest_names: Set[str] = frozenset(),
    active_run_ids: Set[str] = frozenset(),
) -> LifecyclePlan:
    """Decide what to delete. **Pure**: no disk, no session, no clock.

    Args:
        artifacts: every artifact under the root, from its manifest.
        policy: N and M. Validated here, so a bad configuration refuses at the decision
            rather than part-way through a deletion.
        latest_names: the artifact names ``latest`` resolves to, one per run at most.
            Names rather than steps: a name is what the pointer ultimately identifies and
            what every edge is spelled in.
        active_run_ids: runs whose ``mf_runs`` row is **not** terminal. Their artifacts are
            untouchable: the run may publish, swap ``latest`` or branch at any moment, and
            a sweep racing a save is the one window in which "decide then act" is not
            enough on its own.

    Returns:
        The plan. Nothing is deleted and nothing is logged here.

    Raises:
        Refusal: ``checkpoint_lifecycle_policy_invalid`` for a policy that cannot mean
            anything.
    """
    settled = (policy or LifecyclePolicy()).validated()
    all_artifacts = list(artifacts)
    ancestors = ancestor_names(all_artifacts)

    by_run: dict[str, list[ArtifactRef]] = {}
    for artifact in all_artifacts:
        if artifact.is_checkpoint:
            by_run.setdefault(artifact.run_id, []).append(artifact)

    retained_steps: dict[str, dict[int, tuple[str, ...]]] = {
        run_id: _ordinal_keepers(
            [a.step for a in refs],
            keep_last_n=settled.keep_last_n,
            keep_every_m=settled.keep_every_m,
        )
        for run_id, refs in by_run.items()
    }

    keep: list[Decision] = []
    delete: list[Decision] = []

    for artifact in sorted(all_artifacts, key=lambda a: (a.run_id, a.kind.value, a.step)):
        reasons: list[str] = []
        # ---- THE CHECKS ARE ORDERED CHEAPEST-FIRST AND THAT IS *NOT* THE REPORTING ORDER.
        #
        # Deliberate, and it is the only thing that makes the precedence table below do any
        # work. Written with the appends in precedence order first, `ordered = tuple(reasons)`
        # produced an identical answer - so mutation **M6** survived the whole suite: the
        # table looked load-bearing and was decorative, and a later edit that inserted one
        # check in the wrong place would have silently started reporting the wrong primary
        # reason with nothing going red. That is F6's M-19b in this module, found by running
        # the mutation rather than by reading the loop.
        #
        # So: do not "tidy" these into precedence order. Aligning them re-opens the hole.
        if artifact.is_checkpoint:
            reasons.extend(retained_steps.get(artifact.run_id, {}).get(artifact.step, ()))
        if not artifact.is_checkpoint:
            reasons.append(KEEP_KIND_NEVER_PRUNED)
        if artifact.run_id in active_run_ids:
            reasons.append(KEEP_RUN_ACTIVE)
        if artifact.name in latest_names:
            reasons.append(KEEP_LATEST)
        # The correctness clause, and it is unconditional - last to be asked, first to be
        # reported.
        if artifact.name in ancestors:
            reasons.append(KEEP_LINEAGE_ANCESTOR)

        ordered = tuple(r for r in KEEP_REASON_PRECEDENCE if r in reasons)
        common: dict[str, Any] = {
            "name": artifact.name,
            "kind": artifact.kind,
            "run_id": artifact.run_id,
            "step": artifact.step,
            "relative_path": artifact.relative_path,
        }
        if ordered:
            keep.append(Decision(action="keep", reason=ordered[0], reasons=ordered, **common))
        else:
            delete.append(
                Decision(
                    action="delete",
                    reason=DELETE_NOT_RETAINED,
                    reasons=(DELETE_NOT_RETAINED,),
                    **common,
                )
            )

    return LifecyclePlan(
        keep=tuple(keep), delete=tuple(delete), ancestors=ancestors, policy=settled
    )
