"""R-02.19's optional activation source  (F4 task 8.6; FR-004.33-36, AC-004.28).

**Off by default, and off is normal** (R-02.50, ADR-010). ``ACTIVATION_SOURCE_BASE_URL``
is unset in a default install, nothing on the scoring path reaches for it, and a test or
retention pass completes with the capability absent. That is the whole shape of an opt-in
sibling integration, and it is why R-02.19 reads *"This is how miStudio work plugs in, **if
wanted**."*

## Why this module exists at all, and why it is in ``clients/``

R-02.19 was written as a §5.5 requirement naming **miLLM's** probe endpoints, while
ADR-022 forbids anything in §5.1-§5.9's code path from referencing a sibling endpoint.
Implemented literally it would have failed the project's own standalone guard. Its
third-pass amendment routes it here instead: *"an optional sibling integration under
R-02.50 and ADR-010 - reached through that service's single outbound call site in
``clients/``, absent and normal when its base URL is unset, and refused by name."*

So: **one module, one outbound call site** (ADR-010, ADR-017), beside
``clients/midataworks.py`` and built to the same shape. ``services/scoring/`` does not
import it; the capability is reached by :func:`resolve`, from configuration, at runtime.

## Three refusal codes, and the distinction between them is the requirement

``not_configured`` and ``unreachable`` are **different codes** because they are different
states with different fixes - F10's FR-010.6 makes the same distinction from the other
side, and collapsing them hides a misconfiguration behind a network message:

* :data:`NOT_CONFIGURED` (E-20) - nobody asked for this. Set the base URL to enable it.
* :data:`UNREACHABLE` (E-21) - somebody asked for it and the service did not answer.
  **The test and retention scores still record**; only the activation statistics are lost.
* :data:`REFUSED` - the service answered, with a 4xx. It is reachable and it said no,
  which is a third thing again and must not be reported as a network failure.

## What this module will not do

**It does not know a sibling's URL path, and it does not guess one.** R-02.19 says *"a
sibling's probe endpoints"* - deliberately, since the third-pass amendment removed the
endpoint names - and no requirement in BRD-02 names a route. So :meth:`Probe.statistics`
takes the route from its caller and :data:`ROUTE_UNSPECIFIED` refuses an empty one rather
than defaulting to a path. A default route here would be a sibling endpoint invented in
this repository and asserted as that sibling's contract, which is the defect class this
project has recorded eleven times.

**It writes nothing to a manifest** (FR-004.34). Neither
``checkpoint-manifest-v1.json`` nor ``adapter-manifest-v1.json`` has an activation, SAE or
probe field, so the statistics belong in the run log and **no field is invented** for
them. This module returns the payload and records nothing itself.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from typing import Any

import httpx

from core.errors import Refusal, not_configured, refuse

log = logging.getLogger("miforge.clients.activations")

#: The service's name, in the ``<service>_not_configured`` spelling ADR-010 fixes. Kept in
#: one place so a log line, a refusal and a status row cannot disagree about it.
SERVICE = "activation_source"
SETTING = "ACTIVATION_SOURCE_BASE_URL"

#: The three codes, as constants rather than literals at each raise site. Two spellings of
#: one code is how a caller's ``match`` loses a branch, and this project has paid for a
#: renamed enum value five times.
NOT_CONFIGURED = f"{SERVICE}_not_configured"
UNREACHABLE = f"{SERVICE}_unreachable"
REFUSED = f"{SERVICE}_refused"
ROUTE_UNSPECIFIED = f"{SERVICE}_route_unspecified"
CAPABILITY_UNKNOWN = "activation_capability_unknown"

#: Capability key -> what it is, in the operator's words. One entry, and it is the whole of
#: R-02.19. Looped rather than ``if``-ed in :func:`integration_status`, so a capability
#: cannot be added without appearing there.
CAPABILITIES: dict[str, str] = {
    "activation_statistics": (
        "Per-step feature or cluster activation statistics from a sibling's probe "
        "endpoints with an SAE attached (R-02.19)"
    ),
}

#: What the operator should do instead. Named, because a refusal that does not say what is
#: still possible reads as a breakage (ADR-013, ADR-018).
STANDALONE_NEXT_STEP = (
    "Every test and retention score is unaffected: activation statistics are additional, "
    "never a substitute, and a run records its scores with the capability absent."
)

DEFAULT_TIMEOUT_SECONDS = 10.0


# ====================================================================== configuration


def base_url() -> str | None:
    """The configured base URL, or ``None`` when no activation source is configured.

    Read through F4's settings object every call rather than cached here. The value is
    declared **once**, on :class:`services.scoring.settings.ScoringSettings`, whose own
    comment says it is reached *"through a single ``clients/`` call site"* - this one. A
    second declaration in this module would be a second source of truth for one
    environment variable, which is the split that cost this project ``packs_root`` twice
    in one day.

    **A blank value is unset**, and that coercion is applied here because this is the one
    reader. ``core.config.Settings`` does it with a ``field_validator`` for
    ``MILLM_BASE_URL`` and ``MIDATAWORKS_BASE_URL``, *"because ``.env.example``'s
    deliberately-blank lines would otherwise configure every sibling as the empty URL, and
    the standalone path would never be taken on a developer's machine."*
    ``ScoringSettings.activation_source_base_url`` **now carries that validator too**, so
    both layers coerce and neither is load-bearing alone: the settings one stops
    ``ACTIVATION_SOURCE_BASE_URL=`` ever becoming the empty URL (which would have reported
    the source as configured and failed later as a connection error to ``""`` instead of a
    named refusal at the boundary), and this one is the reader refusing to be fooled by a
    value assigned past validation. *This read "carries **no such validator** … Owed in
    that module too" - true when written, and the debt it recorded is paid.*
    """
    from services.scoring.settings import get_scoring_settings

    configured = get_scoring_settings().activation_source_base_url
    if configured is None or not configured.strip():
        return None
    return configured


def _reject_credentials_in(url: str) -> str:
    """Refuse ``http://user:token@host`` rather than mask it.

    **A base URL is a location**; identity travels separately on all three of this
    system's surfaces. The same hole was found and closed on ``millm_base_url``, where a
    credential in the URL *"defeated every field-by-field mask"* and reached any log line
    or traceback - masking hides the symptom and leaves the credential live on the wire.

    **DEFENCE IN DEPTH, not the primary refusal.**
    :class:`services.scoring.settings.ScoringSettings` now carries the validator
    ``core.config``'s two sibling URLs always had, and it runs first - so a credentialled
    ``ACTIVATION_SOURCE_BASE_URL`` never becomes settings and this branch is unreachable
    from any ordinary path. *This docstring read "``ScoringSettings`` has no equivalent
    validator, so the refusal lives at this boundary until it does" - true when written,
    false once the validator landed, and the kind of line a reader trusts instead of
    checking.*

    It is **kept rather than deleted** only because two routes still reach it, both
    asserted in ``test_activations_client.py``: ``ScoringSettings.model_construct``, which
    skips validation by design, and plain in-place assignment, because that class does not
    set ``validate_assignment`` and a settings singleton is assignable from anywhere. That
    test also asserts the bypass is real, so if the config changes this guard's
    justification fails loudly instead of leaving an unreachable guard reading as
    protection while protecting nothing - which is why F8's duplicate code check was
    deleted rather than kept "for safety".

    The message carries the **host only**, never the rejected value: a validation error is
    an output, and reproducing the credential while refusing it is how the first fix for
    that leak leaked again.
    """
    import urllib.parse

    parsed = urllib.parse.urlsplit(url)
    if parsed.username or parsed.password:
        raise refuse(
            f"{SERVICE}_url_carries_credentials",
            f"{SETTING} contains a username or password. A base URL is a location, not an "
            f"identity: remove the credential from the URL and supply it the way the "
            f"service expects. Host: {parsed.hostname!r}.",
            service=SERVICE,
            setting=SETTING,
        )
    return url


def is_configured() -> bool:
    """Whether an operator has configured an activation source. ``False`` is **normal**."""
    return base_url() is not None


def describe(capability: str) -> str:
    """What ``capability`` is for.

    Raises:
        Refusal: :data:`CAPABILITY_UNKNOWN` on an unregistered key. Refused rather than
            defaulted: a typo'd capability string would otherwise make a status response
            report on a capability nothing gates - a fabricated value, and silent
            (ADR-018).
    """
    try:
        return CAPABILITIES[capability]
    except KeyError:
        raise refuse(
            CAPABILITY_UNKNOWN,
            f"{capability!r} is not a registered activation-source capability. Add it to "
            f"clients/activations.py so a status response and the availability guard "
            f"agree about it. Known: {sorted(CAPABILITIES)}.",
            capability=capability,
            known=sorted(CAPABILITIES),
        ) from None


def all_keys() -> list[str]:
    """Every capability key, sorted. What a status response reports as unavailable."""
    return sorted(CAPABILITIES)


def not_configured_refusal(action: str, capability: str) -> Refusal:
    """The :data:`NOT_CONFIGURED` refusal (E-20), built once and here only.

    Args:
        action: What the caller was trying to do, in the operator's words, as a sentence
            fragment: *"recording activation statistics for step 40"*.
        capability: A key of :data:`CAPABILITIES`.

    Returns:
        The refusal, carrying ``service`` and ``capabilities_unavailable`` from
        :func:`core.errors.not_configured` plus ``setting``, ``action`` and ``capability``
        - the union F10's ``clients/availability.py`` emits, so a handler written against
        either sibling's shape works against both.
    """
    # Validated BEFORE the refusal is built. A refusal naming a capability nothing gates
    # would be worse than no refusal at all.
    description = describe(capability)

    base = not_configured(SERVICE, description)
    details: dict[str, Any] = dict(base.details)
    details.update({"setting": SETTING, "action": action, "capability": capability})
    return Refusal(
        base.code,
        f"{action} needs an activation source, and none is configured ({SETTING}). "
        f"Set it in the deployment configuration to use it. {STANDALONE_NEXT_STEP}",
        details,
    )


def require_activation_source(action: str, capability: str) -> str:
    """The configured base URL, or refuse by name.

    **Call this in the service, never only in the route or the worker.** A guard that runs
    in one caller is bypassed by the next one, which is exactly how a capability comes back
    unguarded.

    Returns:
        The base URL with any trailing slash removed, so a caller can join a route without
        producing ``//probe``.

    Raises:
        Refusal: :data:`NOT_CONFIGURED`, naming the setting, the action and the capability.
    """
    url = base_url()
    if url is None:
        raise not_configured_refusal(action, capability)
    return _reject_credentials_in(url).rstrip("/")


def integration_status(*, reachable: bool | None = None) -> dict[str, Any]:
    """This integration's status row, in the shape F10's miDataworks row already uses.

    Args:
        reachable: ``True``/``False`` when a probe was actually made, ``None`` when none
            was. **``None`` is not ``False``** (E-21, F10's FR-010.6): "not configured"
            and "configured and down" are different states with different fixes, and a row
            reporting both as ``false`` hides a misconfiguration behind a network message.

    Returns:
        ``configured``, ``setting``, ``code``, ``reachable``, ``capabilities`` and
        ``capabilities_unavailable``.
    """
    configured = is_configured()
    return {
        "configured": configured,
        "setting": SETTING,
        "code": None if configured else NOT_CONFIGURED,
        "reachable": reachable if configured else None,
        "capabilities": {key: describe(key) for key in all_keys()},
        "capabilities_unavailable": [] if configured else all_keys(),
    }


# ====================================================================== the one call site


class Probe:
    """Every call miForge makes to an activation source. **The only outbound call here.**

    Constructed per unit of work rather than held as a module global, so the availability
    guard runs in the constructor: an unconfigured install refuses at the boundary instead
    of building an object with a ``None`` base URL that fails later somewhere else.
    """

    def __init__(
        self,
        *,
        action: str,
        capability: str = "activation_statistics",
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Open a probe client, refusing by name when no activation source is configured.

        Args:
            action: What the caller is doing, in the operator's words. Carried into every
                refusal, so a message says *"recording activation statistics could not
                reach the activation source"* rather than *"request failed"*.
            capability: The :data:`CAPABILITIES` key this work needs.
            transport: An ``httpx`` transport, for tests. Production passes ``None``.
            timeout: Per-request timeout in seconds.

        Raises:
            Refusal: :data:`NOT_CONFIGURED` when ``ACTIVATION_SOURCE_BASE_URL`` is unset.
        """
        self.action = action
        self.capability = capability
        self.base_url = require_activation_source(action, capability)
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={"Accept": "application/json"},
            transport=transport,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> Probe:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def statistics(self, route: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Ask the activation source for one step's statistics.

        Args:
            route: The sibling's probe route, **supplied by the caller from
                configuration**. There is no default and there must not be one: R-02.19
                says *"a sibling's probe endpoints"* and names none, so a constant here
                would be an invented contract asserted as somebody else's. See the module
                docstring.
            payload: The request body. Passed through unexamined - this module does not
                know a sibling's schema either.

        Returns:
            The decoded JSON body, as it came back. **Not reshaped, not defaulted and not
            written anywhere**: FR-004.34 puts these statistics in the run log and no
            manifest field exists for them, so this module returns and records nothing.

        Raises:
            Refusal: :data:`ROUTE_UNSPECIFIED` on an empty route; :data:`UNREACHABLE` on a
                transport error or a 5xx; :data:`REFUSED` on any other 4xx.
        """
        if not route or not route.strip():
            raise refuse(
                ROUTE_UNSPECIFIED,
                f"{self.action} needs the activation source's probe route, and none was "
                f"given. miForge does not default it: no requirement names a route for a "
                f"sibling's probe endpoints, and a constant here would be an invented "
                f"contract. Supply it from configuration.",
                service=SERVICE,
                action=self.action,
                capability=self.capability,
            )

        path = route if route.startswith("/") else f"/{route}"
        try:
            response = self._client.post(path, json=dict(payload))
        except httpx.HTTPError as exc:
            raise self._unreachable(path, f"{type(exc).__name__}: {exc}") from exc

        if response.status_code >= 500:
            raise self._unreachable(path, f"HTTP {response.status_code}")

        if response.status_code >= 400:
            log.warning(
                "activation_source_refused",
                extra={
                    "code": REFUSED,
                    "status": response.status_code,
                    "action": self.action,
                },
            )
            raise refuse(
                REFUSED,
                f"the activation source refused {self.action} with HTTP "
                f"{response.status_code}. It is reachable and it said no, which is not a "
                f"network problem: read its response. {STANDALONE_NEXT_STEP}",
                service=SERVICE,
                action=self.action,
                capability=self.capability,
                status=response.status_code,
                body=response.text[:300],
            )

        try:
            body = response.json()
        except ValueError as exc:
            # A 2xx that is not JSON is not a refusal from the service and not a transport
            # failure either; it is an unusable answer. Named rather than guessed at.
            raise refuse(
                REFUSED,
                f"the activation source answered {self.action} with HTTP "
                f"{response.status_code} and a body that is not JSON. {STANDALONE_NEXT_STEP}",
                service=SERVICE,
                action=self.action,
                capability=self.capability,
                status=response.status_code,
                body=response.text[:300],
            ) from exc

        if not isinstance(body, dict):
            raise refuse(
                REFUSED,
                f"the activation source answered {self.action} with a JSON "
                f"{type(body).__name__} rather than an object. {STANDALONE_NEXT_STEP}",
                service=SERVICE,
                action=self.action,
                capability=self.capability,
                status=response.status_code,
            )
        return body

    def _unreachable(self, path: str, detail: str) -> Refusal:
        """:data:`UNREACHABLE` (E-21) - **a different code from not-configured.**"""
        log.warning(
            "activation_source_unreachable",
            extra={"code": UNREACHABLE, "action": self.action, "detail": detail},
        )
        return refuse(
            UNREACHABLE,
            f"{self.action} could not reach the activation source at "
            f"{self.base_url}{path} ({detail}). It is configured ({SETTING}) and it did "
            f"not answer - which is not the same as not being configured, and the fix is "
            f"different. {STANDALONE_NEXT_STEP}",
            service=SERVICE,
            action=self.action,
            capability=self.capability,
            setting=SETTING,
            detail=detail,
        )


# ====================================================================== runtime resolution


def resolve(
    action: str,
    capability: str = "activation_statistics",
    *,
    transport: httpx.BaseTransport | None = None,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
) -> Probe | None:
    """The activation source, or ``None`` when there is none. **``None`` is normal.**

    This is the runtime resolution F4 task 8.6 means by *"resolved at runtime from
    configuration"*: ``services/scoring/`` imports nothing from this module, and a caller
    that wants the capability asks here. A caller that gets ``None`` records its test and
    retention scores exactly as it would have anyway - the statistics are additional, and
    absent is not an error (R-02.50).

    **Returns ``None`` rather than raising**, deliberately, and that is the one place in
    this module where absence is not a refusal: a caller doing optional extra work needs a
    branch, not an exception, or every call site grows a ``try`` that swallows the
    difference between *unset* and *down*. The refusing form is
    :func:`require_activation_source`, for a caller acting on an explicit operator request
    that must be answered by name.

    Raises:
        Refusal: :data:`CAPABILITY_UNKNOWN` on an unregistered capability - checked even
            on the absent path, so a typo is caught on a default install rather than first
            appearing when somebody configures the service.
    """
    describe(capability)
    if not is_configured():
        return None
    return Probe(action=action, capability=capability, transport=transport, timeout=timeout)
