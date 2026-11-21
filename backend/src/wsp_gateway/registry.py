"""The gateway's session registry (FTDD §5.4, §5.5; P50, P61, R1B-M7, R2A-4).

It maps a session token's sha256 to the run, its one endpoint, its sensitivity, caps and expiry.
It is rebuilt from the backend on start (``GET …/gateway/sessions``, with spend), and a token it has
not seen is resolved through ``GET …/sessions/by-token/{sha256}`` (a session minted after the
gateway started; there is no registration). A token the backend does not resolve is cached as
unknown for 5 s. Spend here is only a hint for the output cap: the backend's reserve is the bound.

A session the gateway ENDED (a steered editor, a miServer session error, a failed settle) is kept
ended for the life of the process: every later request on its token is refused.
"""

from __future__ import annotations

import datetime as dt
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Any

import httpx

from wsp_gateway.meter import PromptMeter

PREFIX = "/wsp-internal/v1/gateway"
NEGATIVE_CACHE_S = 5.0


class RegistryUnavailable(Exception):  # noqa: N818 - a refusal, like core.errors.Refusal
    """The internal application could not be reached or answered an error."""


def _positive(value: Any) -> int | None:
    """A served window from the backend: a positive integer, else ``None`` (never trusted as
    anything else)."""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None
    return value


def _seeded(evidence: Any) -> PromptMeter:
    """A session's prompt meter, recovered from the evidence its served calls persisted."""
    meter = PromptMeter()
    meter.seed(evidence)
    return meter


@dataclass
class GatewaySession:
    run_id: str
    token_sha256: str
    endpoint_id: str
    sensitivity: str
    cap_usd: Decimal | None
    cap_tokens: int | None
    expires_at: dt.datetime | None
    spend_usd: Decimal = Decimal("0")
    spend_tokens: int = 0
    edit_target_model: str | None = None
    ended: str | None = None
    served_models: list[str] = field(default_factory=list)
    steering: list[str] = field(default_factory=list)
    #: The window miServer served at push (2026-11-21), or ``None``.
    served_context_tokens: int | None = None
    #: The session's prompt-token evidence (fix 2026-11-21): the upstream's own counts, persisted
    #: with each served call's settle and recovered from the backend's session entry, so a
    #: restarted gateway estimates as before.
    prompt: PromptMeter = field(default_factory=PromptMeter, compare=False, repr=False)

    @classmethod
    def of(cls, entry: dict[str, Any]) -> GatewaySession:
        caps = entry.get("caps") or {}
        spend = entry.get("spend") or {}
        expires = entry.get("expires_at")
        return cls(
            run_id=str(entry["run_id"]),
            token_sha256=str(entry["token_sha256"]),
            endpoint_id=str(entry["endpoint_id"]),
            sensitivity=str(entry.get("sensitivity") or "unknown"),
            cap_usd=None if caps.get("cost_usd") is None else Decimal(str(caps["cost_usd"])),
            cap_tokens=None if caps.get("tokens") is None else int(caps["tokens"]),
            expires_at=None if not expires else dt.datetime.fromisoformat(str(expires)),
            spend_usd=Decimal(str(spend.get("usd", 0))),
            spend_tokens=int(spend.get("tokens", 0)),
            edit_target_model=entry.get("edit_target_model"),
            served_context_tokens=_positive(entry.get("served_context_tokens")),
            prompt=_seeded(entry.get("prompt_evidence")),
        )

    def expired(self, now: dt.datetime) -> bool:
        if self.expires_at is None:
            return True
        expires = self.expires_at
        if expires.tzinfo is None:
            expires = expires.replace(tzinfo=dt.UTC)
        return expires <= now

    def remaining_usd(self) -> Decimal | None:
        return None if self.cap_usd is None else self.cap_usd - self.spend_usd

    def remaining_tokens(self) -> int | None:
        return None if self.cap_tokens is None else self.cap_tokens - self.spend_tokens


class Registry:
    def __init__(
        self, internal: httpx.Client, *, monotonic: Callable[[], float] = time.monotonic
    ) -> None:
        self._internal = internal
        self._monotonic = monotonic
        self._sessions: dict[str, GatewaySession] = {}
        self._negative: dict[str, float] = {}
        self._lock = threading.Lock()

    def _get(self, path: str) -> httpx.Response:
        try:
            return self._internal.get(PREFIX + path)
        except httpx.HTTPError as exc:
            raise RegistryUnavailable(path) from exc

    def rebuild(self) -> int:
        """Replace the registry with the backend's active sessions, spend included (R1B-M7)."""
        response = self._get("/sessions")
        if response.status_code != 200:
            raise RegistryUnavailable("/sessions")
        sessions = {
            s.token_sha256: s for s in (GatewaySession.of(e) for e in response.json()["sessions"])
        }
        with self._lock:
            ended = {k: v.ended for k, v in self._sessions.items() if v.ended}
            for sha, reason in ended.items():
                if sha in sessions:
                    sessions[sha].ended = reason
            self._sessions = sessions
            self._negative.clear()
        return len(sessions)

    def resolve(self, token_sha256: str) -> GatewaySession | None:
        with self._lock:
            known = self._sessions.get(token_sha256)
            if known is not None:
                return known
            missed = self._negative.get(token_sha256)
            if missed is not None and self._monotonic() - missed < NEGATIVE_CACHE_S:
                return None
        response = self._get(f"/sessions/by-token/{token_sha256}")
        if response.status_code == 404:
            with self._lock:
                self._negative[token_sha256] = self._monotonic()
            return None
        if response.status_code != 200:
            raise RegistryUnavailable("/sessions/by-token")
        session = GatewaySession.of(response.json())
        with self._lock:
            return self._sessions.setdefault(token_sha256, session)

    def refresh_expiry(self, session: GatewaySession) -> GatewaySession | None:
        """Re-read a session whose cached expiry has passed: the backend re-dates a token when
        its push completes (fix 2026-11-21), which may be after this registry first resolved
        it. The session object is updated in place (an ``ended`` mark on it stands); ``None``
        when the backend no longer holds it live."""
        response = self._get(f"/sessions/by-token/{session.token_sha256}")
        if response.status_code == 404:
            return None
        if response.status_code != 200:
            raise RegistryUnavailable("/sessions/by-token")
        fresh = GatewaySession.of(response.json())
        with self._lock:
            session.expires_at = fresh.expires_at
        return session

    def forget(self, token_sha256: str) -> None:
        with self._lock:
            self._sessions.pop(token_sha256, None)

    def end(self, session: GatewaySession, reason: str) -> None:
        with self._lock:
            if session.ended is None:
                session.ended = reason

    def endpoint(self, endpoint_id: str) -> dict[str, Any]:
        """The endpoint row as the backend holds it NOW (a changed row must stop opening)."""
        response = self._get(f"/endpoints/{endpoint_id}")
        if response.status_code != 200:
            raise RegistryUnavailable("/endpoints")
        data: dict[str, Any] = response.json()
        return data

    def __len__(self) -> int:
        return len(self._sessions)
