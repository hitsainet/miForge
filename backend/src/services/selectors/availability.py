"""Is a requirement satisfied right now?  (FR-003.12; ADR-010, ADR-013).

One function, so the listing endpoint and the plugins give the **same** answer. Two places
computing availability is two places to disagree, and the symptom would be a type the
listing calls available refusing on the first call.

**An unconfigured optional service is reported by name** (ADR-010): the code is
``<service>_not_configured``, it names the service, and the capability is reported
unavailable rather than substituted. Never substitute a different behaviour for a missing
service - on the judge path the only substitute available is the model being trained, which
R-02.11 forbids because a model grading itself is not a measurement.
"""

from __future__ import annotations

from services.selectors.provider_port import current_provider_resolver
from services.selectors.settings import SelectorSettings
from services.selectors.types import Requirement

#: Refusal code for an absent generation endpoint. Spelled with the ``_not_configured``
#: suffix ADR-010 fixes, which the app's ``status_for`` maps to 503 by suffix - so a new
#: optional service does not need an edit there to be reported as "not configured" rather
#: than "your request was bad".
JUDGE_PROVIDER_NOT_CONFIGURED = "judge_provider_not_configured"
MIDATAWORKS_NOT_CONFIGURED = "midataworks_not_configured"


def generation_provider_available(settings: SelectorSettings) -> bool:
    """A judge needs a bound resolver **and** a configured endpoint and model.

    Both halves, because either alone is useless: a resolver with no endpoint has nothing
    to resolve, and an endpoint with no resolver cannot be reached without F3 constructing
    a provider, which it does not do (:mod:`services.selectors.provider_port`).
    """
    if current_provider_resolver() is None:
        return False
    return bool(settings.judge_provider_base_url) and bool(settings.judge_provider_model)


def midataworks_available() -> bool:
    """Whether miDataworks is reachable **according to the port**.

    Asked of the port rather than read off the settings, and the distinction is not
    cosmetic: ADR-010 gives each sibling **one** outbound call site, which is F10's client,
    and the base URL is **F10's** setting. If this module read that setting directly there
    would be two places deciding whether miDataworks is present - and the symptom would be
    a listing that calls the type available while the port refuses on the first call.

    With no port bound the answer is ``False``, which is the standalone path and is
    **normal** rather than broken (AC20).
    """
    from services.selectors.plugins.human_review import current_review_queue

    port = current_review_queue()
    return port is not None and port.available()


def availability_of(
    requirement: Requirement, settings: SelectorSettings
) -> tuple[bool, str | None]:
    """``(available, unavailable_reason)``. The reason names the service, or is ``None``.

    ``None`` exactly when available: a reason beside an available type would be a second
    answer to the same question.
    """
    if requirement is Requirement.NONE:
        return True, None
    if requirement is Requirement.GENERATION_PROVIDER:
        if generation_provider_available(settings):
            return True, None
        return False, JUDGE_PROVIDER_NOT_CONFIGURED
    if requirement is Requirement.MIDATAWORKS:
        if midataworks_available():
            return True, None
        return False, MIDATAWORKS_NOT_CONFIGURED
    # No `else: return True`. A requirement this function does not know is not available;
    # claiming availability for an unhandled case is the fabricated value ADR-018 forbids.
    raise AssertionError(f"unhandled requirement {requirement!r}")
