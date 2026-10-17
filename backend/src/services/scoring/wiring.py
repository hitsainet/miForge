"""The wiring, and the gaps it closes  (ADR-016 #1; R-02.11, R-02.36, R-02.49).

**Two ports, and only one of them is bound here.** The provider resolver is bound below;
F3's review-queue port is bound by :func:`clients.midataworks_review.wire`, and this module
only *reaches* it - see "The review queue" at the foot of this docstring, because a
``clients`` import in this package is forbidden and rightly so.

**F3's judge-model selector was unreachable in production, and this module is why it is not.**

F3 resolves its judge through ``services.selectors.provider_port.ProviderResolver``, bound by
:func:`services.selectors.provider_port.bind_provider_resolver`. Before this module existed
that binding was made **only in tests**: every judge test passed, the type was listed, the
plugin was fully implemented and documented - and in a running miForge no resolver was ever
bound, so ``current_provider_resolver()`` returned ``None`` and the judge refused
``judge_provider_not_configured`` **whatever the operator configured.**

That is ADR-016 #1 exactly, and the shape the project's whole review discipline was written
after: *a sibling shipped sixteen fully-implemented, unit-tested, documented MCP tools that
were never registered with the server. Every one of those tests passed by importing the
module directly, so the suite was green and the documentation said ✅ while the feature did
not exist for any caller.* A grep would have found ``bind_provider_resolver`` and concluded
the call existed - which is why the rule's stronger form is the one that applies: **delete
the line below, run the suite, require a red.**

``backend/tests/unit/test_scoring_provider_resolver_reachability.py`` is that control, and it
asserts three things rather than one: that a resolver is bound **after a production import**,
that it is **this registry's** and not an arbitrary object, and that an HTTP row configured
in the environment is **actually resolvable through it** - because a binding that resolves
nothing is a registry that exists for no caller, one layer down.

## Why the binding lives here and not in the app factory

``api/app.py`` builds the HTTP process only. The judge is needed in the ``train`` worker and
in the ``generate`` worker too, and a binding made in the app factory would be absent from
both - an API-only wiring is the same defect in a smaller blast radius. An import side effect
of ``services.scoring`` reaches every process that scores, which is every process that needs
it, and :func:`wire` is idempotent because *an import side effect that nobody triggers is not
a side effect* while a repeated import is normal and must not fail.

## What it does NOT do

It does not configure a judge. With no ``PROVIDER_<NAME>_BASE_URL`` set there is no
judge-eligible row, the resolver reports ``available("judge") is False``, and F3 names
``judge_provider_not_configured`` - **which is the standalone path and is normal, not
broken** (AC20). Binding a resolver does not make a judge appear; it makes one reachable when
the operator configures it.

## The review queue - the same defect, and the half of it this module can reach

``services.selectors.plugins.human_review.bind_review_queue`` was called in eleven test
files and **nowhere in ``backend/src/``**, so ``current_review_queue()`` was always ``None``
and ``HumanReviewSelector.prepare`` **always** raised ``midataworks_not_configured`` -
*naming an absent service that was present*, with miDataworks configured and answering.

The binding itself is :func:`clients.midataworks_review.wire`'s, as an import side effect of
that module. **It cannot be made here:** ``test_scoring_static_guards.TestF4NeverCallsMiDataworks``
and ``test_F4_no_scoring_module_reaches_a_sibling_or_an_http_client`` both forbid a
``clients`` import under ``services/scoring/``, and the fence is right - F4 reads rows and
resolves providers and has no outbound call site of its own. Observed red against the first
attempt, which had the binding on the line below the resolver's.

What :func:`wire` does instead is import F10's :mod:`services.review`, which *is* permitted
to reach ``clients`` and is where the import trigger belongs. That makes the whole review
path reachable from every process - the API through ``api/v1/endpoints/scoring.py`` and
``runs_review.py``, the workers through ``services.scoring.tasks`` in ``TASK_MODULES`` - as
soon as :mod:`services.review` imports the client. **That last line is owed and is one
line**; :func:`clients.midataworks_review.wire`'s docstring carries it verbatim, and
``test_review_queue_wiring.TestTheImportTriggerIsOwed`` is the strict-xfail tripwire that
goes red when it lands.
"""

from __future__ import annotations

import logging

log = logging.getLogger("miforge.scoring")

_wired = False


def wire() -> None:
    """Register the provider kinds and bind F4's registry into F3's resolver port.

    Idempotent. Raises nothing: a wiring failure at import would take down every process that
    imports this package, and the failure mode that matters here is the opposite one - being
    silently absent - which the reachability test catches.
    """
    global _wired
    if _wired:
        return

    from providers import registry

    # F10'S REVIEW PACKAGE, REACHED NOT CALLED. Importing `services.review` is what gives
    # every scoring process F3's review-queue port, because that package is permitted to
    # import `clients` and this one is not (see the docstring). It is an import for its
    # side effect; this module calls nothing in it and must not, since F4 has no review
    # path of its own. The remaining line is owed inside `services/review/__init__.py`.
    from services import review  # noqa: F401
    from services.selectors import provider_port

    registry.ensure_registered()
    # THE LINE. Delete it and `test_scoring_provider_resolver_reachability.py` must go red:
    # the judge-model selector becomes unreachable for every caller while every one of its
    # own tests still passes, because they bind a stub resolver themselves.
    provider_port.bind_provider_resolver(registry.judge_resolver())
    _wired = True
    log.debug(
        "provider registry wired into the selector resolver port",
        extra={"code": "provider_resolver_bound"},
    )


def unwire() -> None:
    """Unbind, for a test that needs the unconfigured world.

    The unbind path exists and is used: *a test that configured a provider and could not
    configure it away would make the standalone case (AC20) unassertable.*
    """
    global _wired
    from services.selectors import provider_port

    provider_port.bind_provider_resolver(None)
    _wired = False
