"""Selectors and graders — **F3**  (R-02.9, R-02.10, R-02.11).

The plugin surface that decides which model outputs are good. A **selector** takes a
completion and answers ``accept``, ``reject`` or ``undecided``; a **grader** takes one and
answers a scalar. Both sit behind the one contract in :mod:`schemas.selector`.

Every signal source in miForge that is not a pre-built corpus needs something to say *this
completion is worth training on and that one is not*. Self-generation (R-02.5) is a closed
loop without it: the model produces text and the text goes back into the model, so whatever
decides which text survives **is** the domain. **The quality of this one surface sets the
ceiling on every recursive run in the product.**

## THE REGISTRATION BLOCK BELOW IS THE WIRING, AND IT IS THE POINT

Five explicit :func:`services.selectors.registry.register` calls, in one place, run as an
**import side effect of this package**. That is deliberate, and a directory scan was
rejected:

* a scan makes *"is it registered?"* **untestable**, and an unregistered plugin **silent**;
* which is the exact failure this project's review discipline was written after - a sibling
  shipped **sixteen fully-implemented, unit-tested, documented MCP tools that were never
  registered with the server.** Every one of their tests passed by importing the module
  directly, so the suite was green and the documentation said ✅ while the feature did not
  exist for any caller. Two full review rounds had read that code.

So: **deleting one ``register(...)`` line below must make the reachability test red.** That
red was observed, not assumed - see ``test_selector_registry_reachability.py``.

:func:`ensure_registered` exists because an import side effect that nobody triggers is not a
side effect. Anything resolving a selector calls it first, and it is idempotent.
"""

from __future__ import annotations

from services.selectors import registry
from services.selectors.plugins import composite, human_review, judge_model, reference_match, rule
from services.selectors.types import Capability, Requirement, SelectorForm

_BOTH = frozenset({SelectorForm.SELECTOR, SelectorForm.GRADER})
_SELECTOR_ONLY = frozenset({SelectorForm.SELECTOR})

#: The five types R-02.9 names, each with what it decides and what it needs. **The single
#: declaration of the surface**: ``GET /api/v1/selectors`` serves this, and a type absent
#: here does not exist for any caller however well its module is tested.
_REGISTRATIONS = (
    (
        "rule",
        rule.factory,
        Capability(
            kind="rule",
            # SELECTOR ONLY. FTASKS 6.1 puts a grader form on `rule` *"only where its
            # configuration defines a scoring form"*, and the contract gives this type a
            # single opaque `expression` and no scoring form at all. Declaring a grader
            # form anyway would be a capability with nothing behind it - and it would make
            # the health record treat a rule's 1.0/0.0 admission flag as a graded scalar,
            # so every all-accept batch would raise `grader_scores_have_no_spread` beside
            # `selection_accepts_everything`: two names for one fact, one of them wrong.
            forms=_SELECTOR_ONLY,
            requires=Requirement.NONE,
            summary=(
                "regex, length bounds, required terms, forbidden terms and structure - "
                "pure, in-process, no model and no network"
            ),
        ),
    ),
    (
        "judge_model",
        judge_model.factory,
        Capability(
            kind="judge_model",
            forms=_BOTH,
            requires=Requirement.GENERATION_PROVIDER,
            summary=(
                "a second model scores the completion against a prompt template, over any "
                "OpenAI-compatible endpoint - never the model being trained (R-02.11)"
            ),
        ),
    ),
    (
        "reference_match",
        reference_match.factory,
        Capability(
            kind="reference_match",
            forms=_BOTH,
            requires=Requirement.NONE,
            summary=(
                "exact or fuzzy similarity to a reference answer; the mode and the metric "
                "are recorded, because a verdict is not comparable across them"
            ),
        ),
    ),
    (
        "human_review",
        human_review.factory,
        Capability(
            kind="human_review",
            forms=_SELECTOR_ONLY,
            requires=Requirement.MIDATAWORKS,
            summary=(
                "a person adjudicates in-run, through miDataworks' review queue; returns "
                "undecided until a decision exists, and a timeout never admits"
            ),
        ),
    ),
    (
        "composite",
        composite.factory,
        Capability(
            kind="composite",
            forms=_SELECTOR_ONLY,
            requires=Requirement.NONE,
            summary=(
                "all, any or majority over other configured selectors; undecided "
                "propagates, and a decidable composite never waits on a pending human"
            ),
        ),
    ),
)


def ensure_registered() -> None:
    """Register the five built-in types, idempotently.

    Idempotent rather than raising on a second call, while
    :func:`services.selectors.registry.register` itself **does** raise on a duplicate name.
    The two are not in tension: a repeated *import* is normal and must not fail, while a
    second plugin claiming one name is a real conflict whose resolution would otherwise
    depend on import order.
    """
    existing = set(registry.registered_kinds())
    for kind, factory, capability in _REGISTRATIONS:
        if kind in existing:
            continue
        registry.register(kind, factory, capability)  # type: ignore[arg-type]


ensure_registered()
