"""The error envelope, and the refusal helpers  (task 2.3; ADR-013, ADR-018, ADR-021).

Every failure leaves this process as ``{"error": {"code", "message", "details"}}``.
Success bodies are plain JSON - the envelope is not wrapped around success, because a
uniform wrapper makes every client unwrap twice and tells it nothing.

**The rule this module exists to enforce: refuse with a reason rather than guess.**
ADR-018 puts it sharply - *a ``getattr`` default that always fires is a fabricated value,
and that is worse than a refusal because it is silent.* Half of this system's behaviour is
refusing correctly: an unconfigured sibling, an oversized run, an unrecognised adapter
scaling, a non-conforming manifest. So refusal is a first-class return, not an exception
path bolted on afterwards.

**Fail closed where correctness is at stake.** A visible refusal is recoverable; a
silently wrong answer is not. That asymmetry is why a corrupted adapter must not load and
why an unknown ``scaling`` value is refused instead of assumed.
"""

from __future__ import annotations

from typing import Any


class Refusal(Exception):  # noqa: N818 - the project-wide name; renaming touches every module
    """A deliberate, named refusal - the normal way this system says no.

    Carries a stable ``code`` because the code is what another application matches on.
    ADR-010 fixes the spelling for an absent optional service as
    ``<service>_not_configured``, and eleven cross-feature refusal codes are spelled
    identically across the feature documents; that agreement is load-bearing, so a new
    code belongs in the feature's own documents before it appears here.
    """

    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None) -> None:
        if not code:
            raise ValueError("a Refusal must name a code; an unnamed refusal is unactionable")
        if not message:
            raise ValueError(f"Refusal {code!r} must carry a message saying what to do next")
        self.code = code
        self.message = message
        self.details: dict[str, Any] = details or {}
        super().__init__(f"{code}: {message}")

    def envelope(self) -> dict[str, Any]:
        """The wire form. Always these three keys, always in this shape."""
        return {"error": {"code": self.code, "message": self.message, "details": self.details}}


def refuse(code: str, message: str, **details: Any) -> Refusal:
    """Build a :class:`Refusal`. ``raise refuse(...)`` reads as the sentence it is.

    The message says **what to do next**, not how the reader should feel (ADR-021). "Set
    MIDATAWORKS_BASE_URL to use contract parts" is useful; "Sorry, something went wrong"
    is not.
    """
    return Refusal(code, message, details or None)


def not_configured(service: str, *capabilities: str) -> Refusal:
    """An optional sibling is absent - which is **normal**, not an error (ADR-010).

    Names the service and the capabilities that needed it, so the caller learns what it
    cannot do rather than merely that something failed. Never substitute a different
    behaviour for a missing service: a score taken from a fallback provider, recorded as
    though it came from the configured one, is the fabricated value ADR-018 forbids.
    """
    return Refusal(
        f"{service}_not_configured",
        f"{service} is not configured; set its base URL to enable "
        f"{', '.join(capabilities) if capabilities else 'this capability'}.",
        {"service": service, "capabilities_unavailable": list(capabilities)},
    )


def require(value: Any, *, code: str, message: str, **details: Any) -> Any:
    """Return ``value`` when it is present, else refuse by name.

    Exists so that the tempting one-liner is the *correct* one. The shape this replaces -
    ``getattr(obj, "field", some_default)`` - returns the default on a typo in the field
    name, forever, with no signal; this raises.
    """
    if value is None:
        raise Refusal(code, message, details or None)
    return value
