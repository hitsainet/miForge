"""miWorkbench, as miForge's WSP provider reads it (feature 016; P41, P54).

Only what phase 1 needs: the bundle's LIVE sensitivity at push time
(``GET /api/v1/wsp/bundles/{bundle_id}/sensitivity``, which recomputes the most sensitive value
from miDataworks and the marks). Unset, unreachable or malformed reads as ``unknown`` - the
caller treats that as sensitive, so an outage can only refuse an external session, never admit
one. The bundle import and the scorer join this module with FTASKS 7.5 and 13.x.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx

log = logging.getLogger(__name__)

SENSITIVITY_VALUES = ("sensitive", "not_sensitive", "unknown")
TIMEOUT = httpx.Timeout(10.0, connect=5.0)


def read_bundle_sensitivity(bundle_id: str, *, base_url: str | None = None) -> str:
    """The live value, or ``unknown``. Review OR3-016-5 (2026-11-16): every ``unknown`` logs WHY
    (``wsp_sensitivity_unread``: the URL unset, unreachable, an HTTP status, or a malformed answer),
    so the operator can tell an outage from a sensitive bundle. The value stays fail-closed."""
    from wsp.settings import get_wsp_settings

    url = base_url if base_url is not None else get_wsp_settings().miworkbench_api_url
    if not url:
        return _unread(bundle_id, "MIWORKBENCH_API_URL is unset")
    path = f"/api/v1/wsp/bundles/{quote(bundle_id, safe='')}/sensitivity"
    try:
        with httpx.Client(base_url=url.rstrip("/"), timeout=TIMEOUT, trust_env=False) as client:
            response = client.get(path)
    except httpx.HTTPError as exc:
        return _unread(bundle_id, f"miWorkbench is unreachable ({type(exc).__name__})")
    if response.status_code != 200:
        return _unread(bundle_id, f"miWorkbench answered HTTP {response.status_code}")
    try:
        document = response.json()
    except ValueError:
        return _unread(bundle_id, "miWorkbench's answer is not JSON")
    # A list, a string or a number is malformed too: only an object's ``value`` is read.
    value = document.get("value") if isinstance(document, dict) else None
    if value not in SENSITIVITY_VALUES:
        return _unread(bundle_id, "miWorkbench's answer has no known value")
    return str(value)


def _unread(bundle_id: str, cause: str) -> str:
    log.warning(
        "bundle sensitivity read as unknown: %s",
        cause,
        extra={"code": "wsp_sensitivity_unread", "bundle_id": bundle_id, "cause": cause},
    )
    return "unknown"


# ----------------------------------------------------------------- the bundle half (FTASKS 7.5)
#
# miForge PULLS a bundle (P29): the manifest, then each ``files[]`` entry by name, from its OWN
# configured ``MIWORKBENCH_API_URL``. No request carries a URL from a caller, and no request
# carries ``X-Miforge-Agent`` (P31): the client is built with no default headers but httpx's own.

BUNDLE_TIMEOUT = httpx.Timeout(120.0, connect=5.0)
CHUNK = 1 << 20


class FetchFailed(Exception):  # noqa: N818 - a refusal, like core.errors.Refusal
    """``wsp_bundle_fetch_failed``: an error answer, or a digest that disagrees (names the file)."""

    def __init__(self, message: str, name: str | None = None) -> None:
        self.name = name
        super().__init__(message)


def bundle_client(base_url: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        timeout=BUNDLE_TIMEOUT,
        trust_env=False,
        follow_redirects=False,
        transport=transport,
    )


def fetch_bundle_manifest(client: httpx.Client, bundle_id: str) -> dict[str, Any]:
    try:
        response = client.get(f"/api/v1/wsp/bundles/{quote(bundle_id, safe='')}")
    except httpx.HTTPError as exc:
        raise FetchFailed("miWorkbench did not answer the manifest request") from exc
    if response.status_code != 200:
        raise FetchFailed(f"miWorkbench answered {response.status_code} for the manifest")
    try:
        document = response.json()
    except ValueError as exc:
        raise FetchFailed("the manifest is not JSON") from exc
    if not isinstance(document, dict):
        raise FetchFailed("the manifest is not a JSON object")
    return document


def fetch_bundle_file(
    client: httpx.Client,
    bundle_id: str,
    name: str,
    *,
    expected_sha256: str,
    expected_bytes: int,
    dest: Path,
) -> None:
    """Stream one file to ``dest``; its ``X-Content-SHA256`` must equal the manifest's digest AND
    the digest of the bytes received, and its size the manifest's (FR-016.1)."""
    import hashlib
    import hmac

    path = f"/api/v1/wsp/bundles/{quote(bundle_id, safe='')}/files/{quote(name, safe='')}"
    digest = hashlib.sha256()
    size = 0
    try:
        with client.stream("GET", path) as response:
            if response.status_code != 200:
                raise FetchFailed(f"miWorkbench answered {response.status_code} for {name}", name)
            header = response.headers.get("X-Content-SHA256", "").lower()
            if not hmac.compare_digest(header, expected_sha256):
                raise FetchFailed(f"{name}: X-Content-SHA256 disagrees with the manifest", name)
            with dest.open("xb") as out:
                for chunk in response.iter_bytes(CHUNK):
                    size += len(chunk)
                    if size > expected_bytes:
                        raise FetchFailed(f"{name}: more bytes than the manifest declares", name)
                    digest.update(chunk)
                    out.write(chunk)
    except httpx.HTTPError as exc:
        raise FetchFailed(f"miWorkbench did not answer for {name}", name) from exc
    if size != expected_bytes:
        raise FetchFailed(f"{name}: {size} bytes, the manifest declares {expected_bytes}", name)
    if not hmac.compare_digest(digest.hexdigest(), header):
        raise FetchFailed(f"{name}: the bytes disagree with X-Content-SHA256", name)


# ----------------------------------------------------------------- the scorer (FTASKS 13.5)
#
# ``POST /api/v1/wsp/score`` (FR-016.26, P29): miWorkbench scores the heads miForge proposes on
# held-out rows miForge never sees. ONE call per evaluation, to miForge's own configured base URL
# (no URL from a caller), with no ``X-Miforge-Agent`` (the scorer refuses agents; P31). The
# request is validated against the vendored contract before it is sent and the answer after it
# arrives; a refusal is dispatched on the BODY's code, so a ``scorer_busy`` body under 409 still
# reads as busy (P64, R3B-9).

SCORING_SCHEMA = "miworkbench-wsp-scoring-v1.json"
SCORE_PATH = "/api/v1/wsp/score"
SCORE_TIMEOUT = httpx.Timeout(600.0, connect=5.0)


class ScoreRefused(Exception):  # noqa: N818 - a refusal, like core.errors.Refusal
    """The scorer did not give a valid answer: ``code`` is miWorkbench's refusal code, or
    ``scorer_unreachable`` / ``scorer_contract_invalid`` for a transport or shape failure."""

    def __init__(
        self, code: str, message: str, status: int | None = None, body: Any = None
    ) -> None:
        super().__init__(message)
        self.code, self.status, self.body = code, status, body


def _scoring_validator(part: str) -> Any:
    import json

    from jsonschema import Draft202012Validator

    from wsp.schemas import schema_dir

    document = json.loads((schema_dir() / SCORING_SCHEMA).read_text(encoding="utf-8"))
    return Draft202012Validator(document[part])


def _shape_errors(part: str, document: Any) -> list[str]:
    return sorted(e.message for e in _scoring_validator(part).iter_errors(document))[:5]


def refusal_of(body: Any) -> dict[str, Any] | None:
    """The refusal object inside an error answer: ``{"detail": {...}}`` (miWorkbench's real shape,
    FastAPI's ``HTTPException`` with a dict ``detail``), ``{"error": {...}}`` (miForge's own
    envelope) or a bare ``{"code": ...}``. Found 2026-11-19 (016-P2): the client read only the
    last two, so every real refusal - ``scorer_busy`` and the budget's 409 included - read as
    ``scorer_unreachable``; ``contract/test_scoring_metric_guard_cross_repo.py`` now feeds it the
    bodies miWorkbench's own handler writes."""
    if not isinstance(body, dict):
        return None
    for key in ("detail", "error"):
        inner = body.get(key)
        if isinstance(inner, dict) and isinstance(inner.get("code"), str):
            return inner
    return body if isinstance(body.get("code"), str) else None


def scorer_client(base_url: str, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """The scorer's client: miForge's OWN configured base URL, no default headers (so never
    ``X-Miforge-Agent``), no redirects, no proxy from the environment."""
    return httpx.Client(
        base_url=base_url.rstrip("/"),
        timeout=SCORE_TIMEOUT,
        trust_env=False,
        follow_redirects=False,
        transport=transport,
    )


def score(client: httpx.Client, request: dict[str, Any]) -> dict[str, Any]:
    """One scoring call; the validated answer, or :class:`ScoreRefused`."""
    errors = _shape_errors("request", request)
    if errors:
        raise ScoreRefused("scorer_contract_invalid", f"the request breaks the contract: {errors}")
    try:
        response = client.post(SCORE_PATH, json=request, timeout=SCORE_TIMEOUT)
    except httpx.HTTPError as exc:
        raise ScoreRefused("scorer_unreachable", f"{type(exc).__name__}") from exc
    try:
        body = response.json()
    except ValueError:
        body = None
    if response.status_code != 200:
        error = refusal_of(body)
        code = error.get("code") if error is not None else None
        raise ScoreRefused(
            str(code or "scorer_unreachable"),
            f"the scorer answered {response.status_code}",
            status=response.status_code,
            body=body,
        )
    if not isinstance(body, dict) or _shape_errors("response", body):
        raise ScoreRefused("scorer_contract_invalid", "the answer breaks the contract", 200, body)
    return body
