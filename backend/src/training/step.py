"""One step: a batch, a forward and backward, an update  (R-02.12; FR-001.17 - FR-001.20).

R-02.12 ends *"Nothing else is in the step."* That sentence is the specification, and a
reviewer should be able to read :func:`run_one` and see three statements.

## No barrier, and that is the whole dividend of the reframe

No quiesce, no ``commit-begin``/``commit-end`` bracket, no resume, no watchdog, no ``torn``
state. **The parameters are miForge's own**, so there is no second party to coordinate with
and nothing to protect. The bracket and everything built on it are BRD-02 section 5.10's
(R-02.52, R-02.30, R-02.31) and F11's; a test asserts their absence here rather than
trusting this paragraph.

## THE LOSS SEAM - F1 computes no loss, and must not

This is the defect two internally-consistent document chains disagreed about and neither
could see alone. R-02.12's stage 2 appears in all four of F1's documents as *"forward and
backward"*, with no mention of F2's objectives registry; F2 states the dispatch twice and
truthfully adds *"no F1 FR changes"*. R-02.4, R-02.6 and R-02.7 each name the loss their
signal source trains with, and **those losses are F2's** (``training/objectives/``, its
FR-002.3).

So stage 2 is **one registry lookup on the batch's declared objective** and nothing else:

    objectives.get(batch.objective).loss(model, batch)

**A hard-coded cross-entropy here would become the surviving definition of F2's contract.**
It is worse than a missing feature, because a DPO or GRPO batch would train with the wrong
loss while the loss still *falls* and the batch record still says ``dpo`` - the record and
the computation disagree, silently. Hence :class:`ObjectiveRegistry` below: F1 declares the
seam's shape and calls it, F2 fills it, and F1 names no objective and implements no loss.
Until F2 lands the lookup resolves against a **fixture** objective registered in a test,
never against a placeholder in this module.

## The cancellation check is at the top, and only at the top

ADR-007: a step either completes and is recorded, or has not started. A check between the
forward and the update would produce exactly the half-applied step that rule forbids, and
R-02.31 is why it matters - *an interrupted step is never retried*, because re-applying a
non-idempotent AdamW update to a half-written parameter set compounds the damage.

## Gradient accumulation is the caller's, not this function's

R-02.12 defines *one* step as one update. Accumulation changes how many micro-batches feed
one update, so folding it in here would make "step" mean two things and R-02.14's per-step
record ambiguous. The loop (F6) and the source (F2) own it; ``run_one`` stays the unit that
applies exactly one update.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol, cast, runtime_checkable

from core.errors import refuse


@runtime_checkable
class Batch(Protocol):
    """What a signal source hands the step. **F2 owns the sources and this shape's filling.**

    Two members, for two different readers:

    * ``objective`` - the name stage 2 dispatches on. **Not a field of
      ``batch-record-v1.json``**: that schema carries ``source`` (one of four signal
      sources) and no objective, while ``step-record-v1.json``'s step block *requires*
      ``objective``. The objective therefore travels on the in-process batch, and the step
      record is where it is written down.
    * ``record`` - the batch's provenance, conforming to ``batch-record-v1.json``. F1 reads
      ``batch_sha256`` from it for R-02.14 and **defines none of its fields**.
    """

    @property
    def objective(self) -> str: ...

    @property
    def record(self) -> dict[str, Any]: ...


@runtime_checkable
class Objective(Protocol):
    """One training objective. **Every implementation is F2's** (``training/objectives/``)."""

    @property
    def name(self) -> str: ...

    def loss(self, model: Any, batch: Any) -> Any:
        """Return the scalar loss tensor for ``batch``. ``backward()`` is the step's."""
        ...


@runtime_checkable
class ObjectiveRegistry(Protocol):
    """Name -> objective. **F2 implements this**; F1 declares it and looks up through it.

    Declared here rather than in ``training/objectives/`` because F1 is a wave earlier than
    F2 and must be able to *call* the seam before anything fills it - and because a
    ``Protocol`` in the caller is what lets F2 own the registry without F1 owning a
    placeholder. ``get`` raises rather than returning ``None``: an unknown objective is a
    refusal, and an ``Optional`` return invites the ``or default`` that would reintroduce a
    hard-coded loss.
    """

    def get(self, name: str) -> Objective: ...


@runtime_checkable
class BatchSource(Protocol):
    """Where a batch comes from. **F2's** (R-02.4 - R-02.8)."""

    def next(self) -> Batch: ...


@runtime_checkable
class Cancellation(Protocol):
    """A stop request, read at a step boundary. The foundation's ``CancellationToken``."""

    def check(self) -> None: ...


@dataclass(frozen=True)
class StepOutcome:
    """What one completed step produced, for the step record (R-02.14).

    ``grad_norm`` is ``float | None`` and a **non-finite value is kept, not dropped**
    (E-12): a missing number is indistinguishable from a healthy one, so a diverged run
    must be able to say it diverged. ``finite`` is the flag the loop reads to decide the
    step failed, which keeps the number and the verdict as two separate facts.
    """

    loss: float
    objective: str
    grad_norm: float | None
    wall_time_ms: float
    batch_sha256: str | None
    #: False when the loss or the gradient norm is not a finite number.
    finite: bool = True
    #: **§5.10 only** (R-02.14, R-02.52). The quiesce hold and the commit duration of this
    #: step's bracket, in milliseconds, set by F11's stepper and by nothing else. ``None`` on
    #: every standalone step, and that ``None`` is the assertion that no bracket was held -
    #: not a missing measurement. Carried on the outcome rather than read from F11's timings
    #: object by the writer, so the writer stays ignorant of the mode (ADR-022).
    quiesce_hold_ms: float | None = None
    commit_duration_ms: float | None = None


def default_objective_registry() -> ObjectiveRegistry:
    """F2's registry, imported lazily, **refused by name when absent**.

    Lazy because ``training/objectives/`` is F2's package and does not exist in F1's wave;
    refused rather than defaulted because a default is a second definition of F2's contract.
    ADR-018: *a value that is quietly substituted is a fabricated value* - and this one
    would be substituted for the whole purpose of the step.
    """
    try:
        from training import objectives
    except ImportError as exc:
        raise refuse(
            "objective_registry_unavailable",
            "No objective registry is available, so the step has no loss to compute. "
            "miForge will not "
            "substitute a cross-entropy, because the batch declares which objective it is "
            "for and a substituted loss would disagree with the record silently.",
            detail=str(exc),
        ) from exc
    # The module itself satisfies `get(name)`. Cast at the seam rather than typing the
    # import as the Protocol: the module is F2's and MyPy cannot see it from here, so the
    # cast is where the trust in F2's contract is made explicit and reviewable.
    return cast(ObjectiveRegistry, objectives)


def resolve_objective(batch: Any, registry: ObjectiveRegistry) -> Objective:
    """The objective ``batch`` declares. Refuses on a batch that declares none.

    ``getattr`` with no default, then an explicit check: ADR-018 forbids *a default that
    always fires*, and ``getattr(batch, "objective", "sft")`` is exactly that - it would
    train every batch with one loss the moment an attribute name changed.
    """
    name = getattr(batch, "objective", None)
    if not isinstance(name, str) or not name:
        raise refuse(
            "batch_objective_missing",
            "The batch declares no objective, so the step cannot tell which loss to "
            "compute. Each signal source names the loss it trains "
            "with, and the batch carries that name. Refusing rather than "
            "guessing one.",
        )
    return registry.get(name)


def run_one(
    model: Any,
    optimizer: Any,
    batch_source: BatchSource,
    *,
    cancel: Cancellation | None = None,
    objectives: ObjectiveRegistry | None = None,
) -> StepOutcome:
    """One step: assemble a batch, forward and backward, apply the update (R-02.12).

    Three stages and nothing else. The ``cancel`` check is **before** stage 1, never
    between stages.

    Args:
        model: the parameters miForge holds. Not mocked in this feature's tests - a mocked
            model makes the three stages agree by construction, which is how a suite stays
            green over a real bug (ADR-016).
        optimizer: built by :func:`training.optimizer.build`.
        batch_source: F2's source. The call is **in-process and synchronous, inside the
            step, on the ``train`` queue** - ADR-009's default provider generates from the
            parameters this worker holds, and a ``generate`` worker would need a second
            copy of them on the same leased card.
        cancel: checked once, at the boundary. ``None`` in a test that is not exercising
            cancellation; the loop always supplies one.
        objectives: F2's registry. ``None`` resolves it lazily and refuses by name if it is
            not there.

    Returns:
        The completed step's :class:`StepOutcome`.
    """
    # ---- boundary: before stage 1, never inside -------------------------- FR-001.20
    if cancel is not None:
        cancel.check()

    started = time.monotonic()

    # ---- 1. assemble a batch --------------------------------------------- FR-001.19
    batch = batch_source.next()

    # ---- 2. forward and backward: ONE registry lookup -------------------- FR-001.17
    registry = objectives if objectives is not None else default_objective_registry()
    objective = resolve_objective(batch, registry)
    loss = objective.loss(model, batch)
    loss.backward()

    # ---- 3. apply the update --------------------------------------------- FR-001.17
    grad_norm = _apply_update(optimizer, model)

    wall_time_ms = (time.monotonic() - started) * 1000.0
    loss_value = float(loss.detach().item()) if hasattr(loss, "detach") else float(loss)

    return StepOutcome(
        loss=loss_value,
        objective=objective.name,
        grad_norm=grad_norm,
        wall_time_ms=wall_time_ms,
        batch_sha256=_batch_sha256(batch),
        finite=_is_finite(loss_value) and (grad_norm is None or _is_finite(grad_norm)),
    )


def _apply_update(optimizer: Any, model: Any) -> float | None:
    """Apply one update and return the gradient norm the optimizer consumed.

    **The norm is computed where the update is applied**, because it is a property of the
    gradients the optimizer is about to consume - computing it anywhere else would report a
    different set of gradients than the ones that moved the weights.

    A non-finite norm is **returned as it is** (E-12). Clipping or dropping it would hide a
    diverged run behind a healthy-looking record.
    """
    grad_norm = _gradient_norm(model)
    optimizer.step()
    # `set_to_none=True` where it is supported: zeroed gradient tensors stay allocated,
    # which on a card budgeted to the megabyte is the difference between fitting and not.
    try:
        optimizer.zero_grad(set_to_none=True)
    except TypeError:
        optimizer.zero_grad()
    return grad_norm


def _gradient_norm(model: Any) -> float | None:
    """The global L2 norm of the model's gradients, or ``None`` when it cannot be taken.

    ``None`` rather than ``0.0``: zero is a measurement - a model whose gradients all
    vanished - and absence is not. Reporting absence as zero would make a broken
    instrumentation path look like a pathological but real training signal.
    """
    try:
        import torch
    except ImportError:
        return None

    norms = [
        p.grad.detach().float().norm(2)
        for p in model.parameters()
        if getattr(p, "grad", None) is not None
    ]
    if not norms:
        return None
    return float(torch.norm(torch.stack(norms), 2).item())


def _batch_sha256(batch: Any) -> str | None:
    """``batch_sha256`` from the batch's provenance record (R-02.14's batch source hash).

    Read from the record F2 writes, never recomputed here: recomputing would be a second
    definition of what a batch's content hash covers, and ADR-005's rule applies -
    two canonicalisers are two answers.
    """
    record = getattr(batch, "record", None)
    if isinstance(record, dict):
        value = record.get("batch_sha256")
        return value if isinstance(value, str) else None
    return None


def _is_finite(value: float) -> bool:
    """Whether a float is a usable number. NaN is not equal to itself, which is the test."""
    return value == value and value not in (float("inf"), float("-inf"))
