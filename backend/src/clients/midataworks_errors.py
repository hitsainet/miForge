"""The typed refusals for the miDataworks seam  (F10 task 2.4; FR-010.6).

**Nothing `httpx` leaves the client.** Every outcome of an outbound call arrives at a
caller as one of these, so no service, worker or route needs to know the transport. A bare
``httpx.ConnectError`` escaping would reach the application's last-resort handler as a
``500 internal_error``, which turns a refusal the operator can act on into one they cannot.

**The codes split on who must act**, because a code whose audience is ambiguous gets
handled by neither:

======================================  =========================================  ======
Condition                               Code                                       Who
======================================  =========================================  ======
base URL unset                          ``midataworks_not_configured``             config
connect/read error, or 5xx              ``midataworks_unreachable``                env
``401``/``403``                         ``midataworks_unauthorized``               deploy
other ``4xx``                           ``midataworks_refused`` (+ their code)     either
======================================  =========================================  ======

``not_configured`` and ``unreachable`` are deliberately far apart: different audience,
different status, different retry policy. **Collapsing them is the single most tempting
simplification in this feature** and the one that hides a misconfiguration behind a
network message for as long as nobody looks.

Each class is a :class:`core.errors.Refusal`, so the application's one exception handler
renders it in ADR-013's envelope with no per-route work.
"""

from __future__ import annotations

from typing import Any

from core.errors import Refusal

CODE_UNREACHABLE = "midataworks_unreachable"
CODE_UNAUTHORIZED = "midataworks_unauthorized"
CODE_REFUSED = "midataworks_refused"


class MiDataworksUnreachable(Refusal):
    """Configured, but the request did not complete. **Retried, then refused.**

    A fault, not a decision. The distinction from ``not_configured`` is the whole of
    FR-010.6: this one is worth retrying and worth paging about; that one is normal.
    """

    def __init__(self, action: str, detail: str, *, attempts: int, url: str) -> None:
        super().__init__(
            CODE_UNREACHABLE,
            f"{action} could not reach miDataworks at {url} after {attempts} "
            f"attempt(s): {detail}. miDataworks is configured, so this is a fault rather "
            f"than an absence - check that it is running and reachable on the LAN.",
            {"action": action, "url": url, "attempts": attempts, "detail": detail},
        )


class MiDataworksUnauthorized(Refusal):
    """``401``/``403``. **Never retried** - a rejected credential does not become valid
    by repetition (ADR-020).

    Worth knowing on this seam specifically: miDataworks' REST surface has **no bearer
    gate** today, and identity is the ``X-Dataworks-Agent`` header. So a 401/403 here most
    likely means either that header was malformed or that miDataworks has grown a gate
    (the F10 FPRD's OQ-7). Either way it is a deployment fact, and the fix is a
    configuration change rather than another attempt.
    """

    def __init__(self, action: str, status: int, *, url: str, agent: str, body: str = "") -> None:
        super().__init__(
            CODE_UNAUTHORIZED,
            f"miDataworks rejected {action} with HTTP {status}. miForge sends no bearer "
            f"token to miDataworks by design - identity on its REST surface is "
            f"X-Dataworks-Agent, which miForge sent as {agent!r}. Check that value, and "
            f"whether miDataworks has introduced an authentication gate. Not retried: a "
            f"rejected credential does not become valid by repetition.",
            {
                "action": action,
                "status": status,
                "url": url,
                "agent": agent,
                "body": body[:500],
                "retried": False,
            },
        )


class MiDataworksRefused(Refusal):
    """Any other ``4xx``: miDataworks said no, and said why in its own vocabulary.

    Their code is carried in ``details['midataworks_code']`` rather than translated. A
    translation table would be a second description of someone else's error vocabulary -
    the restatement that made two BRDs' field lists disagree - and the operator-facing
    fix for ``ORIGIN_APP_MISMATCH`` is miDataworks', not miForge's.
    """

    def __init__(
        self,
        action: str,
        status: int,
        *,
        url: str,
        midataworks_code: str | None,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            CODE_REFUSED,
            f"miDataworks refused {action} with HTTP {status}"
            f"{f' ({midataworks_code})' if midataworks_code else ''}: {message}",
            {
                "action": action,
                "status": status,
                "url": url,
                "midataworks_code": midataworks_code,
                "midataworks_message": message,
                "midataworks_details": details or {},
            },
        )
