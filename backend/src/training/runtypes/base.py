"""The ``RunType`` seam: three registrations and one loop  (R-02.41, R-02.43).

**BRD-02 section 5.9's first sentence is the whole design**: *"Three run types share one
loop. Section 5.4's step, 5.5's tests, 5.6's save and 5.7's loop apply to all three; what
differs is what is being changed."* So a run type is a **registry entry, not a
conditional** - and an ``if run_type ==`` anywhere outside this package is a defect in F8
rather than a shortcut (FR-008.1, FR-008.2).

## What may differ, exhaustively

Three methods, and nothing else:

* :meth:`RunType.preflight` - the refusals that are specific to this run type.
* :meth:`RunType.prepare` - **what is trainable**. This is where the substance lives: a
  ``change`` or ``create`` run trains every parameter, an ``adapter`` run trains only the
  adapter's.
* :meth:`RunType.save` - **which artifact kind** is written. Not *how*: both writers go
  through ``core.atomic_dir`` and are atomic by the same rule (ADR-015).

Everything else is one object all three execute, and :data:`STEP` is that statement made
mechanical: every run type carries the **same function object**, so
``test_runtype_registry.py`` can assert identity rather than trusting a comment. R-02.41's
entire content is *"step 0 is random instead of pretrained"* - which F1's acquisition
already did - so a large ``create.py`` would mean the loop had been duplicated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from training.acquisition.base import AcquiredModel, AcquisitionRecord, AcquisitionSpec
from training.run_config import RunConfig
from training.step import run_one

#: **THE** step function. Bound here, once, and carried by all three run types so that
#: "three run types share one loop" is a testable identity rather than a claim. F6's loop
#: calls ``run_type.step``; it does not import ``run_one`` a second time.
#:
#: Each run type declares it as ``step = staticmethod(STEP)``, and the ``staticmethod`` is
#: **not** decoration. A bare function assigned to a class attribute is a descriptor, so
#: ``instance.step`` would be a *bound method* and the step would receive the run type as
#: its ``model`` argument. Three bound methods are also three distinct objects, which is
#: why ``test_runtype_registry.py`` asserts identity rather than equality - that assertion
#: caught exactly this.
STEP = run_one

#: The three names, in the order a form should offer them. ``change`` is F1's default.
RUN_TYPE_NAMES: tuple[str, ...] = ("change", "create", "adapter")


@dataclass(frozen=True)
class TrainablePlan:
    """What the optimizer is built over, and what the step forwards through.

    ``trainable`` is a list rather than the module itself because that is the difference
    between the run types: an adapter run's optimizer covers the adapter's parameters
    **alone** (FR-008.19), and handing the step a module would leave "which parameters"
    to be rediscovered by whoever built the optimizer.
    """

    trainable: list[Any]
    module: Any
    #: AC14's *before* value, taken in :meth:`RunType.prepare` and compared after the run.
    #: ``None`` for ``change`` and ``create``, which are *supposed* to change the base.
    #: Taken in ``prepare`` rather than later because later means hashing a base something
    #: has already had a chance to touch.
    base_digest: str | None = None


@dataclass(frozen=True, kw_only=True)
class SaveContext:
    """Everything either writer needs, named once instead of per call site.

    A **superset**, deliberately: the two artifact kinds need overlapping but unequal sets,
    and the alternative - two context types plus a branch at the call site to pick one - is
    the conditional this package exists to remove. A run type reads the fields its artifact
    kind requires and **refuses by name** when one is missing (ADR-018); it never defaults
    one, because a fabricated value in the artifact meant to outlive everything is worse
    than a refusal.

    ``domain`` and ``signal_source`` have **no defaults**, matching F5's ``ManifestFields``
    for the same reason: ``domain=None`` has to be passed deliberately, since *null is an
    assertion and an absent key is an omission*, and there is no value to default
    ``signal_source`` to that would not be a fabricated provenance claim.
    """

    run_id: str
    step: int
    session: Any
    #: What is saved. For ``change``/``create`` the model; for ``adapter`` the wrapped
    #: model, whose ``save_pretrained`` writes the adapter and **not** the base.
    module: Any
    domain: str | None
    signal_source: str
    #: F1's record. ``base_model`` and ``parent`` are **derived from it** by the run type
    #: using F1's own lineage functions, never supplied loose - which is what makes M10
    #: (write ``base_model`` as the model's name, or ``parent`` non-null) a one-line
    #: mutation with a test to catch it.
    record: AcquisitionRecord
    #: F1's optimizer identity block, same shape in both manifests so consumers share code.
    optimizer: Mapping[str, Any]

    tokenizer: Any = None
    test: Mapping[str, Any] | None = None
    retention: Mapping[str, Any] | None = None
    selector_hash: str | None = None
    grader_hash: str | None = None
    merged_adapter: Mapping[str, Any] | None = None

    # ------------------------------------------------------------- adapter-only values
    #: The run's adapter configuration, as validated at submit. The manifest's ``adapter``
    #: block and the ``LoraConfig`` handed to ``peft`` come from **this one object**
    #: (task 9.2): a writer deriving the manifest's numbers independently of what it gave
    #: ``save_pretrained`` produces a directory that validates against both schemas, loads
    #: under plain ``peft``, and is **refused by miLLM**, whose loader cross-checks them.
    adapter: Any = None
    #: ``base.manifest_sha256`` for an adapter whose base **was** a saved checkpoint.
    base_manifest_sha256: str | None = None
    #: Contract references from F7's producer. Never constructed here (FR-008.32).
    parts: Sequence[Mapping[str, Any]] | None = None

    root: Path | None = None
    emit: Any = None
    heartbeat: Any = None
    effects: list[str] = field(default_factory=list)
    #: **Where this save point's artifact landed**, appended by whichever run type wrote it.
    #:
    #: One list rather than a return shape per artifact kind, because the loop has to record
    #: what was published and the two writers return different things:
    #: :func:`services.checkpoints.service.save` returns a ``SaveOutcome`` and
    #: :func:`training.runtypes.adapters.save.save` returns the manifest **as published**.
    #: A loop that reached for ``.directory`` on whichever came back would be the
    #: ``if run_type ==`` this package exists to remove, one duck-typed step away; a loop
    #: that re-derived the path from ``adapter_paths``/``checkpoint_dir`` would be a second
    #: spelling of a path the writer already knows.
    artifacts: list[str] = field(default_factory=list)


@runtime_checkable
class RunType(Protocol):
    """One answer to *"what is being changed"*.

    ``step`` is an attribute rather than a method so that the registry can be asked whether
    all three carry the same object. A method would make ``a.step is b.step`` false for
    three identical bodies, and the assertion that defends section 5.9's first sentence
    would become untestable.
    """

    name: str
    step: Any

    def preflight(
        self,
        spec: AcquisitionSpec,
        cfg: RunConfig,
        *,
        signal_source: str | None = None,
        adapter: Any = None,
    ) -> None:
        """Refuse what this run type cannot do, **before the lease and before any
        allocation**. Cheapest refusals first: shape before filesystem, filesystem before
        card (ADR-023).

        ``signal_source`` and ``adapter`` are keyword-only and optional on **all three**
        rather than on the one that reads each. ``AcquisitionSpec`` and ``RunConfig`` are
        both frozen and closed and neither carries a signal source or an adapter block, so
        a run type that needed one would otherwise have no way to see it - and three
        different ``preflight`` arities would reintroduce at the call site the conditional
        this package exists to remove. The field these should properly arrive in is F6's
        run definition (R-02.53); see the feature report.
        """
        ...

    def prepare(self, acquired: AcquiredModel, *, adapter: Any = None) -> TrainablePlan:
        """What is trainable, and the base digest AC14 compares against."""
        ...

    def save(self, ctx: SaveContext) -> Any:
        """Write this run type's artifact kind. Atomic either way (ADR-015)."""
        ...


def all_parameters(acquired: AcquiredModel) -> TrainablePlan:
    """Every parameter, trainable. Shared by ``change`` and ``create``.

    **One function for both, and that is the point.** R-02.41 is about step 0 being random,
    which F1's acquisition already handled; the trainable set is identical, so writing it
    twice would invite the two to drift and would make a reader think the difference was
    here. ``test_runtype_registry.py`` asserts the two run types return the same set.
    """
    return TrainablePlan(
        trainable=list(acquired.module.parameters()),
        module=acquired.module,
        base_digest=None,
    )
