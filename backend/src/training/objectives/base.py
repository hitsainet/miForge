"""The objective seam F1's step dispatches through  (FR-002.3; R-02.4, R-02.6, R-02.7).

R-02.4, R-02.6 and R-02.7 each name the loss their signal source trains with, and those are
F2's requirements - so **F2 owns the loss implementations and the batch names the one it is
for.** F1's stage 2 is one registry lookup and nothing else:

    objectives.get(batch.objective).loss(model, batch)

**A hard-coded cross-entropy in the step would have become the surviving definition of this
contract**, and it would be worse than a missing feature: a DPO or GRPO batch would train
with the wrong loss while the loss still *falls* and the batch record still says ``dpo``.
The record and the computation would disagree, silently. That defect was found only because
two internally-consistent document chains were read against each other.

## ``get`` raises; it never returns ``None``

An ``Optional`` return invites ``objectives.get(name) or default``, and the only default
available is the loss nobody chose. F1 declares the same thing from its side
(``training/step.py``'s ``ObjectiveRegistry``): *"an unknown objective is a refusal."*

## A registry, not a match statement

A registration can be deleted and a test can see it; a branch cannot (ADR-016 #1). The
control is in the test: delete one ``register(...)`` line and the suite must go red.

## ``torch`` is imported lazily, at one seam per module

The import lives in :func:`torch_module` so that every refusal, every record and the whole
API remain importable on a machine with no ``torch`` - ADR-023's *"a developer without a GPU
can run everything except a training step"* is a property of the code, not a hope. It is
also the seam a test substitutes a recording double at, which is how the masking policy and
the parameter-group assertion are checked without a card.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

from core.errors import refuse


@runtime_checkable
class Objective(Protocol):
    """One training objective. Satisfies ``training.step.Objective``."""

    @property
    def name(self) -> str: ...

    def loss(self, model: Any, batch: Any) -> Any:
        """The scalar loss tensor for ``batch``. ``backward()`` is the **step's**."""
        ...


_OBJECTIVES: dict[str, Objective] = {}


def register(objective: Objective) -> None:
    """Register one objective under its own declared name. **Raises on a duplicate.**

    Not last-one-wins: two implementations claiming one name would make which loss trains a
    run depend on import order, and the batch record would name the objective either way.
    """
    name = objective.name
    if name in _OBJECTIVES:
        raise refuse(
            "objective_already_registered",
            f"An objective named {name!r} is already registered. Refusing rather than "
            f"replacing it: last-one-wins would make which loss trains a run depend on "
            f"import order, and the record would say the same thing either way.",
            name=name,
            registered=sorted(_OBJECTIVES),
        )
    _OBJECTIVES[name] = objective


def get(name: str) -> Objective:
    """The objective ``name``. **Raises; never returns ``None`` and never defaults.**

    Raises:
        Refusal: ``objective_unknown``, listing the registered names.
    """
    objective = _OBJECTIVES.get(name)
    if objective is None:
        raise refuse(
            "objective_unknown",
            f"No objective named {name!r}. Registered: {sorted(_OBJECTIVES)}. Refusing "
            f"rather than falling back to cross-entropy: a substituted loss trains a model "
            f"the batch record does not describe, and the loss still falls.",
            name=name,
            registered=sorted(_OBJECTIVES),
        )
    return objective


def names() -> tuple[str, ...]:
    """Every registered objective name, sorted."""
    return tuple(sorted(_OBJECTIVES))


def registered() -> Mapping[str, Objective]:
    """A copy of the registry, for a listing. Not a mutable view."""
    return dict(_OBJECTIVES)


def _reset_for_tests() -> None:
    """Empty the registry. **Only a test calls this**, and the reachability control does."""
    _OBJECTIVES.clear()


def torch_module() -> Any:
    """``torch``, imported lazily, **refused by name when absent**.

    The one seam. ADR-023: everything except a training step must run without a card, and
    the CPU wheels are a core dependency - but a developer or a CI job without them still
    has to be able to import this package, resolve an objective, and see every refusal.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a declared dependency
        raise refuse(
            "torch_unavailable",
            "PyTorch is not importable, so no loss can be computed. Every other path in "
            "this feature - the readers, the refusals, the records, the whole API - runs "
            "without it (ADR-023); only the step needs it.",
            detail=str(exc),
        ) from exc
    return torch


def require_examples(batch: Any, kind: type, objective: str) -> tuple[Any, ...]:
    """The batch's material, refusing material of the wrong shape.

    A batch names its objective and carries the material that objective consumes. If the
    two disagree, something assembled a batch for one loss and labelled it another - and
    **the loss would still compute something**, which is the failure mode this check exists
    to make loud.
    """
    examples = tuple(getattr(batch, "examples", ()) or ())
    if not examples:
        raise refuse(
            "batch_empty",
            f"A {objective!r} batch arrived with no examples. A zero-example step trains on "
            f"nothing while every log line looks healthy, so it is refused rather than run.",
            objective=objective,
        )
    wrong = [type(example).__name__ for example in examples if not isinstance(example, kind)]
    if wrong:
        raise refuse(
            "batch_material_mismatch",
            f"A {objective!r} batch carries {sorted(set(wrong))} where it needs "
            f"{kind.__name__}. The batch names the objective that consumes it, so a "
            f"mismatch means one of the two is wrong - and the loss would still produce a "
            f"number.",
            objective=objective,
            expected=kind.__name__,
            found=sorted(set(wrong)),
        )
    return examples
