# Origin: miDataworks (Onegaishimas/miDataworks) backend/src/mcp_server/client.py @ a055a38
# Mode: adapt (docs/REUSE.md). Kept: one async client as the only path from a tool to the
# backend; the agent header on the client AND re-asserted per request with no `headers`
# argument; the envelope unwrapped into a ToolError; a 2xx non-JSON body is an error; the
# timeout warning. Changed: X-Miforge-Agent; no multipart (no miForge route takes a file).
"""The only path from an MCP tool to the miForge backend (FR-012.2, FR-012.22).

Guarantees:

- **Every request carries ``X-Miforge-Agent``.** Set on the client's default headers AND
  re-asserted inside :meth:`ForgeClient.request`, which takes no ``headers`` argument, so no
  tool can send a request the REST layer would take for the operator's. The approval gate keys
  on this header; a request without it would run ungated.
- **A 2xx body that is not JSON is an error,** ``NON_JSON_RESPONSE``: a misrouted ingress
  answers 200 with an HTML page, which must not reach an agent as an empty success.
- **An error envelope surfaces its code, message and details** as a :class:`ForgeError`.
- **A read timeout says the request may already have been applied.**

A ``202`` pending-approval body is returned unchanged: it is the tool's result.
"""

from __future__ import annotations

import json
import logging
from typing import Any

import httpx
from mcp.server.mcpserver.exceptions import ToolError

logger = logging.getLogger(__name__)

#: Must equal ``AGENT_HEADER`` in ``backend/src/api/agent_origin.py``; compared by AST in
#: ``tests/unit/mcp/test_client.py`` (this package must not import the backend).
AGENT_HEADER = "X-Miforge-Agent"


class ForgeError(ToolError):
    """The backend refused, failed, or could not be reached. Carries the envelope verbatim.

    A ``ToolError`` so the agent reads it: mcp 2.x shows a tool's message to the client only for
    a ``ToolError`` and answers any other exception with a bare "Error executing tool".
    """

    def __init__(
        self, status: int, code: str, message: str, details: dict[str, Any] | None = None
    ) -> None:
        self.status = status
        self.code = code
        self.message = message
        self.details = details or {}
        suffix = f" {json.dumps(self.details, sort_keys=True, default=str)}" if self.details else ""
        super().__init__(f"{code} ({status}): {message}{suffix}")


def _clean(params: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in params.items() if v is not None}


class ForgeClient:
    """Thin async wrapper over ``{base_url}/api/v1`` (and ``/api/health``)."""

    def __init__(self, base_url: str, identity: str, timeout: float = 60.0) -> None:
        self._identity = identity
        self._root = base_url.rstrip("/")
        self._http = httpx.AsyncClient(
            base_url=f"{self._root}/api/v1",
            timeout=timeout,
            headers={AGENT_HEADER: identity},
        )

    @property
    def identity(self) -> str:
        return self._identity

    async def close(self) -> None:
        await self._http.aclose()

    async def request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        json_body: Any = None,
    ) -> Any:
        """Send one request. ``path`` is relative to ``/api/v1``; one starting ``/api/`` is
        sent to the backend root (``/api/health``)."""
        url = f"{self._root}{path}" if path.startswith("/api/") else path
        # Re-asserted per call: the default header could be dropped by a refactor of the
        # constructor; this line cannot be bypassed by any argument, because there is none.
        headers = {AGENT_HEADER: self._identity}
        try:
            response = await self._http.request(
                method,
                url,
                params=params or None,
                json=json_body,
                headers=headers,
            )
        except httpx.ReadTimeout as exc:
            raise ForgeError(
                0,
                "BACKEND_TIMEOUT",
                f"The backend did not answer {method} {path} in time. The request may already "
                "have been applied: check its state before retrying.",
            ) from exc
        except httpx.HTTPError as exc:
            raise ForgeError(
                0,
                "BACKEND_UNREACHABLE",
                f"The miForge backend could not be reached ({type(exc).__name__}).",
            ) from exc

        content_type = response.headers.get("content-type", "")
        is_json = "json" in content_type
        if response.status_code >= 400:
            if is_json:
                try:
                    body = response.json()
                except ValueError:
                    body = None
                error = body.get("error") if isinstance(body, dict) else None
                if isinstance(error, dict):
                    details = error.get("details")
                    raise ForgeError(
                        response.status_code,
                        str(error.get("code") or "ERROR"),
                        str(error.get("message") or ""),
                        details if isinstance(details, dict) else None,
                    )
            raise ForgeError(
                response.status_code,
                "NON_JSON_ERROR",
                f"The backend answered {response.status_code} without an error envelope; the "
                "request may have reached a proxy rather than the API.",
            )

        if not response.content:
            return {"status_code": response.status_code}
        if not is_json:
            raise ForgeError(
                response.status_code,
                "NON_JSON_RESPONSE",
                f"{method} {path} answered {response.status_code} with {content_type or 'no'} "
                "content, which did not come from the API (misrouted ingress or proxy page). "
                "This is not an empty result.",
            )
        try:
            return response.json()
        except ValueError as exc:
            raise ForgeError(
                response.status_code,
                "NON_JSON_RESPONSE",
                f"{method} {path} answered with a body that is not valid JSON.",
            ) from exc

    async def get(self, path: str, **params: Any) -> Any:
        return await self.request("GET", path, params=_clean(params))

    async def post(self, path: str, json_body: Any = None, **params: Any) -> Any:
        return await self.request("POST", path, params=_clean(params), json_body=json_body)

    async def put(self, path: str, json_body: Any) -> Any:
        return await self.request("PUT", path, json_body=json_body)

    async def delete(self, path: str, **params: Any) -> Any:
        return await self.request("DELETE", path, params=_clean(params))
