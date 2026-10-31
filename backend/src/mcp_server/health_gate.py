# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/mcp_server/health_gate.py @ a055a38
# Mode: adapt (docs/REUSE.md). Kept: TTL cache, single flight, never-blocking `snapshot`,
# `invalidate`, `public_reason`, `gated_call`. Changed: miForge's health body - the backend
# answers 503 with a JSON body when a dependency is degraded, so "the backend is up" is "it
# answered JSON with a dependencies block", and postgres/redis read `status == "ok"`.
"""Per-dependency availability for the MCP tools (FR-012.4).

Tools stay registered through outages: an MCP client caches its tool list, and an agent must
tell "the backend is down" from "this tool does not exist". A tool whose dependency is down
returns ``{"unavailable": "<dependency>", "reason": "..."}`` and issues no request.

One probe, ``GET {api_url}/api/health``, answers every dependency; cached 10 s.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

PROBE_TIMEOUT_S = 3.0
HEALTH_PATH = "/api/health"

#: The dependencies a tool may require.
DEPENDENCIES: tuple[str, ...] = ("backend", "postgres", "redis")


def _retrieve_exception(task: asyncio.Task[Any]) -> None:
    """Consume a background task's result without re-raising a cancellation."""
    if task.cancelled():
        return
    task.exception()


class HealthGate:
    """TTL-cached availability of the backend and the dependencies it reports."""

    def __init__(self, api_url: str, ttl_s: float = 10.0) -> None:
        self._url = api_url.rstrip("/")
        self._ttl = ttl_s
        # (checked_at_monotonic, backend_available, reason, body)
        self._cache: tuple[float, bool, str, dict[str, Any]] | None = None
        self._http: httpx.AsyncClient | None = None
        self._lock: asyncio.Lock | None = None
        self._refreshing: asyncio.Task[Any] | None = None
        #: Probe count, read by the single-flight test.
        self.probes = 0

    def _client(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(timeout=PROBE_TIMEOUT_S, follow_redirects=False)
        return self._http

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    def _fresh(self) -> bool:
        return self._cache is not None and time.monotonic() - self._cache[0] < self._ttl

    async def _refresh(self) -> tuple[float, bool, str, dict[str, Any]]:
        if self._fresh():
            assert self._cache is not None
            return self._cache
        if self._lock is None:
            self._lock = asyncio.Lock()
        async with self._lock:
            if self._fresh():
                assert self._cache is not None
                return self._cache
            available, reason, body = await self._probe()
            self._cache = (time.monotonic(), available, reason, body)
            return self._cache

    async def check(self, dependency: str) -> tuple[bool, str]:
        """``(available, reason)`` for one dependency."""
        _, backend_ok, reason, body = await self._refresh()
        return self._judge(dependency, backend_ok, reason, body)

    @staticmethod
    def _judge(
        dependency: str, backend_ok: bool, reason: str, body: dict[str, Any]
    ) -> tuple[bool, str]:
        if dependency not in DEPENDENCIES:
            return False, f"unknown dependency '{dependency}'"
        if not backend_ok:
            return False, reason
        if dependency == "backend":
            return True, "ok"
        deps = body.get("dependencies") if isinstance(body, dict) else None
        entry = deps.get(dependency) if isinstance(deps, dict) else None
        if not isinstance(entry, dict):
            return False, f"{dependency} not reported by the backend"
        if entry.get("status") == "ok":
            return True, "ok"
        return False, f"{dependency} down: {entry.get('detail') or 'no reason given'}"

    async def unavailable(self, *dependencies: str) -> dict[str, str] | None:
        """The structured refusal for the first dependency that is down, else ``None``."""
        for dependency in dependencies:
            ok, reason = await self.check(dependency)
            if not ok:
                return {"unavailable": dependency, "reason": reason}
        return None

    def snapshot(self, dependency: str) -> tuple[bool | None, str]:
        """Last known state WITHOUT probing; ``(None, "not probed yet")`` before the first."""
        pending = self._refreshing
        if not self._fresh() and (pending is None or pending.done()):
            try:
                task = asyncio.get_running_loop().create_task(self._refresh())
                task.add_done_callback(_retrieve_exception)
                self._refreshing = task
            except RuntimeError:
                pass  # no running loop; the next check() probes
        if self._cache is None:
            return None, "not probed yet"
        _, backend_ok, reason, body = self._cache
        return self._judge(dependency, backend_ok, reason, body)

    def invalidate(self) -> None:
        self._cache = None

    @staticmethod
    def public_reason(reason: str) -> str:
        """A coarse category safe for the UNAUTHENTICATED ``/health`` route."""
        if reason in ("ok", "not probed yet"):
            return reason
        if reason.startswith("HTTP "):
            return "error response"
        if " down: " in reason:
            return "down"
        if reason.endswith("not reported by the backend"):
            return "not reported"
        if reason.startswith("timed out"):
            return "timed out"
        if reason.startswith("unknown dependency"):
            return "unknown dependency"
        return "unreachable"

    async def _probe(self) -> tuple[bool, str, dict[str, Any]]:
        self.probes += 1
        url = f"{self._url}{HEALTH_PATH}"
        try:
            response = await self._client().get(url)
        except httpx.TimeoutException:
            return False, f"timed out after {PROBE_TIMEOUT_S}s ({url})", {}
        except httpx.HTTPError as exc:
            return False, f"{type(exc).__name__}: {exc} ({url})", {}
        except Exception as exc:
            return False, f"probe failed, {type(exc).__name__}: {exc} ({url})", {}
        try:
            body = response.json()
        except ValueError:
            return False, f"HTTP {response.status_code} from {url} was not JSON", {}
        # The backend answers 503 WITH its JSON body when a dependency is degraded; that is the
        # backend up and reporting, not the backend down. Anything else is not the API.
        if not isinstance(body, dict) or "dependencies" not in body:
            return False, f"HTTP {response.status_code} from {url} is not miForge's health", {}
        return True, "ok", body


async def gated_call(
    gate: HealthGate, dependencies: tuple[str, ...], call: Callable[[], Awaitable[Any]]
) -> Any:
    """Run ``call`` only when every dependency is up; otherwise return the structured refusal.

    A connection failure DURING the call invalidates the cache and becomes the same shape, so
    the next call probes rather than trusting a stale "ok".
    """
    from .client import ForgeError

    down = await gate.unavailable(*dependencies)
    if down is not None:
        return down
    try:
        return await call()
    except ForgeError as exc:
        if exc.code == "BACKEND_UNREACHABLE":
            gate.invalidate()
            return {"unavailable": "backend", "reason": exc.message}
        raise
