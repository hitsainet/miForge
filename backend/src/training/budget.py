"""The budget guard **seam**  (FR-001.8; ADR-008, R-02.42).

A ``Protocol`` and a pass-through. **The estimator is not here** - R-02.42 is F8's
requirement and :class:`training.budget_guard.ComputedBudgetGuard` implements this
``Protocol`` from its own module, bound in :mod:`services.acquisition`. F8 opens no file in
this feature: this one keeps the declaration and the pass-through and nothing else.

## Why a seam at all, when F1 is the thing that allocates

ADR-008: *"R-02.42's budget check runs before any allocation."* Acquisition **is** the first
allocation, so it is the point after which a budget check is worthless - and F1 lands two
waves before F8 in PPRD section 6's finishing order. Shipping the call site with F1 means
F8 lands an implementation rather than a call site; shipping only the estimator would mean
F8 also has to find the right place to call it, which is how a guard ends up running after
the allocation it was meant to precede.

## A guard that cannot estimate must not permit

The spec carries **exactly one** of a parameter count or an architecture config. Both is
ambiguous and refused; **neither is refused too, not passed** (F8's FR-008.10, recorded here
so the two sides cannot drift). The asymmetry is the whole point: the failing case is
obvious and gets tested, while "I had nothing to work with, so I allowed it" is a guard
that is green against every test of its refusing path and silently useless in production.

:class:`PassThroughGuard` is the one deliberate exception, and **it says so in the log**. It
declines to estimate *by design and loudly*, which is a different fact from being handed a
spec it cannot read.
"""

from __future__ import annotations

import logging
from typing import Protocol, runtime_checkable

from core.errors import refuse
from training.acquisition.base import AcquisitionSpec
from training.run_config import RunConfig

log = logging.getLogger(__name__)


@runtime_checkable
class BudgetGuard(Protocol):
    """Decide whether a run fits before anything is allocated (R-02.42).

    **Do not change this signature.** F8 adopted it verbatim rather than proposing one, so
    the ``Protocol`` is not edited and the call site is not moved; the only change at this
    seam when F8's estimator lands is the binding in :mod:`services.acquisition`.
    """

    def check(self, spec: AcquisitionSpec, cfg: RunConfig) -> None:
        """Return on a run that fits; raise ``budget_exceeded`` on one that does not.

        Returning ``None`` rather than a verdict object is deliberate: a boolean return is a
        value a caller can forget to read, and the one thing this guard must never do is be
        called and ignored.
        """
        ...


def require_exactly_one_input(spec: AcquisitionSpec) -> None:
    """Refuse a spec the guard cannot estimate from (F8 FR-008.10).

    Shared by every real implementation, so the rule has one definition rather than one per
    guard. Both inputs present is ambiguous - the two would disagree and nothing says which
    to trust. Neither present is **refused**, because a guard that cannot estimate must not
    permit the run it exists to stop.
    """
    has_count = spec.param_count is not None
    has_config = spec.architecture_config is not None

    if has_count and has_config:
        raise refuse(
            "budget_spec_ambiguous",
            "The budget spec carries both a parameter count and an architecture config. "
            "They are two independent estimates of the same thing and nothing says which "
            "to trust, so this is refused rather than resolved by precedence. A load "
            "supplies the count; a create supplies the config.",
            param_count=spec.param_count,
        )
    if not has_count and not has_config:
        raise refuse(
            "budget_spec_incomplete",
            "The budget request has neither a parameter count nor an architecture config, "
            "so the run's GPU memory cannot be estimated. A run that cannot be estimated is "
            "refused rather than started unmeasured.",
            mode=spec.mode.value,
        )


class PassThroughGuard:
    """Permits every run, and **says so loudly** (FR-001.8's shipped default until F8).

    **A silent pass-through is indistinguishable from a working guard**, which is exactly
    the shape of defect this project's review discipline exists to catch - so this one logs
    at ``WARNING`` on every call, naming the requirement that is not yet implemented. It
    must be obvious in a run log that no estimate was made.

    **Keep this class after F8 lands.** F8's ordering test uses it as a spy, so deleting it
    breaks another feature's test, and its log line documents the window in which runs were
    admitted unmeasured.
    """

    #: Named so a caller can identify which guard is bound without ``isinstance``.
    name = "pass-through"

    def __init__(self) -> None:
        self.calls: list[tuple[AcquisitionSpec, RunConfig]] = []

    def check(self, spec: AcquisitionSpec, cfg: RunConfig) -> None:
        """Permit, and record that no estimate was made.

        **Does not call** :func:`require_exactly_one_input`. That is the deliberate
        exception named in the module docstring: this guard declines to estimate by design,
        which is not the same as being handed a spec it cannot read, and refusing here would
        make a pass-through stricter than the estimator that replaces it.
        """
        self.calls.append((spec, cfg))
        log.warning(
            "no memory estimate was made for this run: the budget guard is a pass-through",
            extra={
                "code": "budget_guard_pass_through",
                "requirement": "R-02.42",
                "acquisition_mode": spec.mode.value,
                "param_count": spec.param_count,
                "effective_batch_size": cfg.effective_batch_size,
                "optimizer_state_precision": cfg.optimizer_state_precision,
            },
        )
