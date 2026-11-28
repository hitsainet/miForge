"""The one answer to "is miDataworks available?"  (F10 task 2.3; FR-010.3, FR-010.5, R-02.50).

**Unset means absent and normal.** Not a misconfiguration, not a warning, not an error log.
miForge was the suite's only application with a hard runtime dependency on a sibling, and
the thing it depended on did not exist; R-02.50 is what stops that growing back, and this
module is the first place it could. So the default posture is inverted from the obvious
one: ``MIDATAWORKS_BASE_URL`` is unset by default, a miForge install talks to nothing, and
every capability that needs the sibling is reported unavailable **by name**.

**Never substitute a different behaviour for a missing service** (ADR-010, ADR-018). There
is no fallback here: miForge does not read a local directory "instead", does not synthesise
a review decision, does not accept candidates by default and does not carry on with a
partial part. R-02.40's external-file path is a *different, separately chosen* source that
the operator reaches deliberately, and a part taken from files is recorded with its own
provenance and **never** a contract reference.

**The guard returns the URL.** A guard that only raises invites its caller to re-read
settings, and then the setting's name appears in two places. One call, one answer.

**Where the refusal's shape comes from.** ``core.errors.not_configured`` fixes the
``<service>_not_configured`` spelling and its ``service`` / ``capabilities_unavailable``
details; FR-010.3 additionally requires ``sibling``, ``setting``, ``action`` and
``capability``, which is the shape miDataworks' own ``require_mistudio`` produces in the
other direction - *including* its standalone next-step sentence, which is the part most
people would drop and the part that makes the refusal useful rather than merely correct.
This module emits the **union**, so a handler written against either shape works.
"""

from __future__ import annotations

from typing import Any

from clients import capabilities
from core.config import get_settings
from core.errors import Refusal, not_configured
from core.runtime_settings import effective_settings

#: The sibling's name as a human writes it, kept in one place so a log line, a refusal and
#: a health row cannot disagree about the capitalisation.
SIBLING = "miDataworks"
SETTING = "MIDATAWORKS_BASE_URL"

#: The sentence that tells the operator what to do instead. R-02.40's path, named.
STANDALONE_NEXT_STEP = (
    "Everything else in miForge works without miDataworks: supply parts from your own "
    "files and they are recorded with their own provenance."
)


def base_url() -> str | None:
    """The configured base URL, or ``None`` when miDataworks is absent.

    Read through the settings object every call rather than cached. Settings are already
    cached by ``pydantic-settings``; caching *availability* here would make
    :func:`integration_status` report a stale answer, and being stale is the one thing a
    status report must not be.
    """
    # Runtime-writable (F12, D5): the override, when one is set, is the configured URL.
    return effective_settings().midataworks_base_url


def is_configured() -> bool:
    """Whether an operator has configured miDataworks. ``False`` is **normal**."""
    return base_url() is not None


def agent_identity() -> str:
    """The ``X-Dataworks-Agent`` value. Validated at settings load, not here.

    miDataworks refuses an identity outside ``^agent:[a-z0-9][a-z0-9-]{0,47}$`` with
    ``400 INVALID_AGENT_IDENTITY``; ``core.config`` checks the pattern when settings are
    built so a malformed value fails at startup rather than as a rejected call an hour
    into a run.
    """
    return get_settings().midataworks_agent


def not_configured_refusal(action: str, capability: str) -> Refusal:
    """The ``midataworks_not_configured`` refusal, built once and here only.

    Args:
        action: What the caller was trying to do, in the operator's words. It goes into
            the message verbatim, so write it as a sentence fragment: *"importing a part
            over HTTP"*.
        capability: A key of :data:`clients.capabilities.CAPABILITIES`.

    Returns:
        The refusal, carrying ``sibling``, ``setting``, ``action`` and ``capability`` in
        ``details`` beside the ``service`` / ``capabilities_unavailable`` pair that
        :func:`core.errors.not_configured` fixes for every optional sibling.
    """
    # Raises `capability_unknown` on a typo. Deliberately BEFORE the refusal is built: a
    # guard that reported a capability nothing gates would be worse than no report.
    description = capabilities.describe(capability)

    base = not_configured("midataworks", description)
    details: dict[str, Any] = dict(base.details)
    details.update(
        {
            "sibling": SIBLING,
            "setting": SETTING,
            "action": action,
            "capability": capability,
        }
    )
    return Refusal(
        base.code,
        f"{action} needs {SIBLING}, and no {SIBLING} URL is configured ({SETTING}). "
        f"Set it in the deployment configuration to use it. {STANDALONE_NEXT_STEP}",
        details,
    )


def require_midataworks(action: str, capability: str) -> str:
    """The configured miDataworks base URL, or refuse by name.

    **Call this in the service, never only in the route.** A guard in the route is
    bypassed by the worker calling the service directly, which is exactly how a capability
    comes back unguarded (F10 task 6.6).

    Args:
        action: What the caller was trying to do, in the operator's words.
        capability: The registry key from :mod:`clients.capabilities`.

    Returns:
        The configured base URL, with any trailing slash removed so callers can join
        paths without producing ``//api/v1``.

    Raises:
        Refusal: ``capability_unknown`` for an unregistered key, configured or not; and
            ``midataworks_not_configured`` naming the setting, the action and the
            capability, plus what to do instead.
    """
    # The key is validated FIRST, on both paths. Until 2026-10-17 it was checked only
    # inside the not-configured refusal, so a typo'd capability was silently accepted
    # whenever miDataworks WAS configured - surfacing exactly when the service was down and
    # never when it was up (suite tracker 1.6, found from miDataworks).
    capabilities.describe(capability)
    url = base_url()
    if url is None:
        raise not_configured_refusal(action, capability)
    return url.rstrip("/")


def integration_status(*, reachable: bool | None = None) -> dict[str, Any]:
    """The miDataworks entry health and ``/midataworks/status`` both serve (FR-010.4).

    Args:
        reachable: ``True``/``False`` when a probe was actually made, ``None`` when none
            was. **``None`` is not ``False``**: "not configured" and "configured and
            down" are different states with different fixes, and a payload that reported
            both as ``false`` would hide a misconfiguration behind a network message -
            the single most tempting simplification in this feature (FR-010.6).

    Returns:
        A dict carrying ``configured``, ``setting``, ``agent``, ``reachable``,
        ``capabilities`` and ``capabilities_unavailable``. Built by **looping the
        registry**, not by an ``if`` per capability, so a capability cannot be added
        without appearing here.
    """
    configured = is_configured()
    unavailable = [] if configured else capabilities.all_keys()
    return {
        "configured": configured,
        "setting": SETTING,
        "code": None if configured else not_configured("midataworks").code,
        # The agent identity is configuration, not a secret (ADR-020), and an operator
        # debugging an ORIGIN_APP_MISMATCH needs to see what miForge is sending.
        "agent": agent_identity(),
        "reachable": reachable if configured else None,
        "capabilities": {key: capabilities.describe(key) for key in capabilities.all_keys()},
        "capabilities_unavailable": unavailable,
    }
