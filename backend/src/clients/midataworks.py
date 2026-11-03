"""The one outbound HTTP wrapper for miDataworks  (F10 tasks 2.4, 8.x; FR-010.1, .6, .8).

**Etiquette lives here and nowhere else.** An AST test
(``backend/tests/unit/test_midataworks_one_call_site.py``) fails if ``httpx`` is imported
anywhere else under ``backend/src/``. That is not tidiness: *a seam with two call sites has
two sets of headers, two retry policies and two opinions about what "unreachable" means*,
and the second one is always the one that forgets the agent header. miDataworks' own
``clients/endpoint_caller.py`` concentrates its outbound etiquette the same way and guards
it with the same kind of test; this module copies the arrangement, not the contents.

- ``X-Dataworks-Agent: agent:miforge`` on **every** request. Read-back does not require it
  (verified in their route: ``read_back_review_decisions`` takes no ``actor``); it is sent
  anyway, because one call site means one set of headers and no per-route special cases.
- **No ``Authorization`` header, ever.** miDataworks' REST layer has no bearer gate -
  verified by reading ``core/agent_origin.py`` and its endpoint modules. The shared MCP
  bearer token belongs to its MCP server and the ``trainer`` token (R-02.32) is miLLM's.
  Three surfaces, three credentials, none shared. :data:`FORBIDDEN_HEADERS` makes that a
  checked property rather than a habit.

## The status mapping, and every row is a test

``backend/tests/unit/test_midataworks_status_mapping.py`` has one case per row. A mapping
expressed only as a chain of ``if``s cannot be audited against the design.

===============================  ====================================================
Response                         Outcome
===============================  ====================================================
2xx                              the parsed body
5xx, connect or read error       wait ``Retry-After`` (else 1, 2, 4 ... capped), retry
                                 up to ``retry_max``, then ``MiDataworksUnreachable``
401 / 403                        ``MiDataworksUnauthorized`` - **never retried**
other 4xx                        ``MiDataworksRefused``, carrying miDataworks' own code
===============================  ====================================================

## Waiting is cooperative

``on_wait(seconds, reason)`` is called before every backoff and **may raise**. This matters
more than it looks: every worker is ``--pool=solo -c 1`` (ADR-006), so
``revoke(terminate=True)`` has no pool child to signal and a busy solo worker never reads
the control queue. A cancelled run must be able to interrupt a backoff rather than sleep
through it, and the only way it can is if something it controls is consulted first.
``sleep`` is injected for the same reason in reverse: a test that really sleeps through a
backoff is a test nobody runs.

## Sync, not async

``httpx.Client``, because every caller in this codebase is synchronous - the Celery solo
worker and FastAPI's sync routes. An async client would need a bridge at each call site,
and a bridge at each call site is the second opinion this module exists to prevent.
"""

from __future__ import annotations

import contextlib
import logging
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import httpx

from clients.availability import agent_identity, require_midataworks
from clients.midataworks_errors import (
    MiDataworksRefused,
    MiDataworksUnauthorized,
    MiDataworksUnreachable,
)

log = logging.getLogger("miforge.clients.midataworks")

#: The header that carries miForge's identity. Its value is validated against
#: miDataworks' own pattern at settings load (``core.config``).
AGENT_HEADER = "X-Dataworks-Agent"

#: Headers this client must never send. Asserted by a test, because "we do not send a
#: bearer token" is a property worth pinning: the credential that would be reached for is
#: ``MILLM_TRAINER_TOKEN``, which belongs to a different service, and sending a sibling's
#: credential to the wrong sibling is a leak, not a misconfiguration.
FORBIDDEN_HEADERS: frozenset[str] = frozenset({"authorization", "proxy-authorization"})

#: Per-request timeout, seconds. A constant rather than a setting **for now**: promoting
#: it is a one-line addition to ``core/config.py``, which F10 does not own (see the
#: feature report). Until then it is a named constant and a constructor argument, never an
#: ``os.environ`` read - one reader for configuration (ADR-020).
DEFAULT_TIMEOUT_SECONDS = 10.0

#: Attempts after the first, for 5xx and transport errors only. Small on purpose: a long
#: retry ladder inside a step loop turns a sibling outage into a run that looks hung.
DEFAULT_RETRY_MAX = 3

#: The ceiling on a single backoff wait, seconds, so an honoured ``Retry-After: 3600``
#: cannot park a run for an hour inside one call.
MAX_BACKOFF_SECONDS = 30.0

#: miDataworks' own limits, read from its implementation rather than guessed:
#: ``CANDIDATE_BATCH_MAX = 500`` in ``services/calibration/constants.py`` and
#: ``review_candidate_max_bytes`` (default 65,536) in ``core/config.py``.
#: **miForge splits to fit before sending**; it does not send a batch to be refused.
CANDIDATE_BATCH_MAX = 500
CANDIDATE_MAX_BYTES_DEFAULT = 65_536

WaitHook = Callable[[float, str], None]
SleepFn = Callable[[float], None]


@dataclass(frozen=True, slots=True)
class CandidateAccepted:
    """One candidate as miDataworks acknowledged it.

    ``created`` is ``False`` for a repeated ``external_id`` - the idempotence that makes a
    re-post after a crash safe. It is carried rather than discarded because "we posted
    500 and 500 already existed" and "we posted 500 new ones" are different facts about a
    resumed run.
    """

    external_id: str
    item_id: str
    created: bool


@dataclass(frozen=True, slots=True)
class ExternalDecision:
    """One decision as miDataworks recorded it. **Everything here is untrusted data.**

    ``reason``, ``decided_by`` and ``override_label`` are stored as data and rendered as
    text. None of them is interpreted as an instruction, used to build a path or a command,
    or interpolated into a prompt instruction block (FR-010.29).

    ``decision`` is **miDataworks' vocabulary as read** - ``accept``, ``override``,
    ``flag`` or ``reject`` - never miForge's verdict value. The single conversion lives in
    :mod:`services.review.verdicts`.
    """

    decision_id: str
    external_id: str
    item_id: str
    decision: str
    override_label: str | None
    reason: str
    decided_by: str
    decided_by_origin: str
    created_at: str


class MiDataworksClient:
    """Every call miForge makes to miDataworks. **The only module that may use httpx.**

    Constructed per unit of work rather than held as a module global: the availability
    guard runs in the constructor, so an unconfigured install refuses at the boundary
    instead of building an object with a ``None`` base URL that fails later somewhere else.
    """

    def __init__(
        self,
        *,
        action: str,
        capability: str,
        transport: httpx.BaseTransport | None = None,
        timeout: float = DEFAULT_TIMEOUT_SECONDS,
        retry_max: int = DEFAULT_RETRY_MAX,
        sleep: SleepFn = time.sleep,
        on_wait: WaitHook | None = None,
    ) -> None:
        """Open a client, refusing by name when miDataworks is not configured.

        Args:
            action: What the caller is doing, in the operator's words. Carried into every
                refusal this client raises, so a message says *"importing a part over
                HTTP could not reach miDataworks"* rather than *"request failed"*.
            capability: The :mod:`clients.capabilities` key this work needs.
            transport: An ``httpx`` transport, for tests. Production passes ``None``.
            timeout: Per-request timeout in seconds.
            retry_max: Attempts after the first, for 5xx and transport errors only.
            sleep: Injected so a backoff test uses a fake clock, not real time.
            on_wait: Called as ``on_wait(seconds, reason)`` before each backoff; **may
                raise** to abandon the call. This is how a cancelled run interrupts a
                wait under the solo pool (ADR-006).

        Raises:
            Refusal: ``midataworks_not_configured`` when ``MIDATAWORKS_BASE_URL`` is unset.
        """
        self.action = action
        self.base_url = require_midataworks(action, capability)
        self.agent = agent_identity()
        self._retry_max = max(0, retry_max)
        self._sleep = sleep
        self._on_wait = on_wait
        self._client = httpx.Client(
            base_url=self.base_url,
            timeout=timeout,
            headers={AGENT_HEADER: self.agent, "Accept": "application/json"},
            transport=transport,
        )

    # ------------------------------------------------------------------ lifecycle

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> MiDataworksClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ transport

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, Any] | None = None,
        json_body: Any | None = None,
    ) -> httpx.Response:
        """One call, with the retry policy of the table above. Returns a 2xx response.

        Raises:
            MiDataworksUnauthorized: on 401/403, with no retry.
            MiDataworksRefused: on any other 4xx.
            MiDataworksUnreachable: after ``retry_max`` attempts at a 5xx or a transport
                error.
        """
        url = f"{self.base_url}{path}"
        attempts = 0
        last_detail = "no attempt was made"

        while attempts <= self._retry_max:
            attempts += 1
            try:
                response = self._client.request(method, path, params=params, json=json_body)
            except httpx.HTTPError as exc:
                # A transport error: no response, so nothing to classify. Retryable.
                last_detail = f"{type(exc).__name__}: {exc}"
                if attempts > self._retry_max:
                    break
                self._wait(attempts, None, last_detail)
                continue

            if response.status_code < 400:
                return response

            if self._is_credential_rejection(response):
                # NOT retried. The one row of the table that is a policy rather than a
                # mechanism (ADR-020).
                log.error(
                    "midataworks_unauthorized",
                    extra={
                        "code": "midataworks_unauthorized",
                        "status": response.status_code,
                        "action": self.action,
                        "agent": self.agent,
                    },
                )
                raise MiDataworksUnauthorized(
                    self.action,
                    response.status_code,
                    url=url,
                    agent=self.agent,
                    body=response.text,
                )

            if response.status_code >= 500:
                last_detail = f"HTTP {response.status_code}"
                if attempts > self._retry_max:
                    break
                self._wait(attempts, response.headers.get("Retry-After"), last_detail)
                continue

            raise self._refused(response, url)

        log.error(
            "midataworks_unreachable",
            extra={
                "code": "midataworks_unreachable",
                "action": self.action,
                "attempts": attempts,
                "detail": last_detail,
            },
        )
        raise MiDataworksUnreachable(self.action, last_detail, attempts=attempts, url=url)

    @staticmethod
    def _is_credential_rejection(response: httpx.Response) -> bool:
        """Whether this is a rejected **credential** rather than a declined **action**.

        **A 403 is not automatically an authentication failure on this surface, and
        treating it as one is a real defect this project caught by writing the test.**
        miDataworks answers ``403 ORIGIN_APP_MISMATCH`` when miForge posts candidates to a
        queue another application created (*"This queue belongs to ...; ... cannot post to
        it"*) and ``403 agent_forbidden`` for a route agents may read but not change. Those
        are miDataworks declining an action, and the operator-facing fix - create the queue
        from miForge, or ask the operator to make the change - has nothing to do with a
        credential. Reporting them as ``midataworks_unauthorized`` sends the reader looking
        for a token that **does not exist on this surface at all** (FR-010.8).

        So: a 401 is always a credential rejection, and a 403 is one only when miDataworks
        names no code of its own. When it names one, **their code is more specific than the
        status** and is carried through as ``midataworks_refused``.
        """
        if response.status_code == 401:
            return True
        if response.status_code != 403:
            return False
        try:
            payload = response.json()
        except ValueError:
            return True
        error = payload.get("error") if isinstance(payload, Mapping) else None
        named = isinstance(error, Mapping) and error.get("code")
        return not named

    def _refused(self, response: httpx.Response, url: str) -> MiDataworksRefused:
        """Turn a 4xx into a refusal carrying miDataworks' own code.

        Their envelope is ``{"error": {"code", "message", "details"}}`` - the same shape
        miForge uses - but a body that is not JSON, or not that shape, must still produce
        a usable refusal rather than an exception while reporting an exception.
        """
        code: str | None = None
        message = response.text[:500]
        details: dict[str, Any] = {}
        try:
            payload = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, Mapping):
            error = payload.get("error")
            if isinstance(error, Mapping):
                code = str(error.get("code")) if error.get("code") is not None else None
                message = str(error.get("message", message))
                raw_details = error.get("details")
                if isinstance(raw_details, Mapping):
                    details = dict(raw_details)
            elif payload.get("detail") is not None:
                message = str(payload["detail"])
        log.warning(
            "midataworks_refused",
            extra={
                "code": "midataworks_refused",
                "status": response.status_code,
                "midataworks_code": code,
                "action": self.action,
            },
        )
        return MiDataworksRefused(
            self.action,
            response.status_code,
            url=url,
            midataworks_code=code,
            message=message,
            details=details,
        )

    def _wait(self, attempt: int, retry_after: str | None, reason: str) -> None:
        """Back off before the next attempt, cooperatively.

        ``Retry-After`` is honoured when miDataworks sends one, because it knows more than
        an exponential guess does - but capped, so a server asking for an hour cannot park
        a step loop for an hour.
        """
        seconds = min(2.0 ** (attempt - 1), MAX_BACKOFF_SECONDS)
        if retry_after:
            # An HTTP-date `Retry-After` falls through: the exponential value is already
            # correct-ish, and a date parser here would be a second opinion about time.
            with contextlib.suppress(ValueError):
                seconds = min(float(retry_after), MAX_BACKOFF_SECONDS)
        if self._on_wait is not None:
            # May raise. That is the point - see the module docstring.
            self._on_wait(seconds, reason)
        log.warning(
            "midataworks_retry",
            extra={
                "code": "midataworks_retry",
                "action": self.action,
                "attempt": attempt,
                "wait_seconds": seconds,
                "detail": reason,
            },
        )
        self._sleep(seconds)

    # ------------------------------------------------------------------ review (R-02.36)

    def create_external_queue(
        self, *, question: str, label_set: Sequence[str], external_ref: Mapping[str, Any]
    ) -> dict[str, Any]:
        """Create the run's external review queue.

        The agent header is what sets the queue's ``origin_app``, and the poster's app must
        match it - so **miForge must create the queue it posts to.** An operator-created
        queue carries the operator's origin and miForge could not post to it
        (``ORIGIN_APP_MISMATCH``), which is why FR-010.21 has miForge create it rather than
        offering the operator the choice.

        ``label_set`` must hold 2-64 entries (their ``Field(min_length=2, max_length=64)``);
        miForge does not pre-validate it here, because their refusal names the real rule and
        a second copy of it would be a second answer.
        """
        response = self._request(
            "POST",
            "/api/v1/review-queues",
            json_body={
                "kind": "external",
                "question": question,
                "label_set": list(label_set),
                "external_ref": dict(external_ref),
            },
        )
        body = response.json()
        return dict(body) if isinstance(body, Mapping) else {}

    def get_queue(self, queue_id: str) -> dict[str, Any]:
        """The queue's record, including its ``items`` and ``decided`` counts."""
        body = self._request("GET", f"/api/v1/review-queues/{queue_id}").json()
        return dict(body) if isinstance(body, Mapping) else {}

    def close_queue(self, queue_id: str, *, reason: str | None = None) -> dict[str, Any]:
        """Close a review queue miForge created (miDataworks D2; suite tracker 4.1).

        ``POST /api/v1/review-queues/{id}/close`` with ``{"reason": ...}`` or ``{}``. The
        answer is the queue with ``state: "closed"`` and ``closed_at`` / ``closed_by`` /
        ``closed_by_origin`` / ``close_reason``. **Idempotent on their side**: a second close
        answers 200 with the first close's details. Items and decisions are kept and the
        decisions stay readable; new candidates and decisions are refused ``409
        QUEUE_CLOSED``.

        Raises:
            MiDataworksRefused: ``404 QUEUE_NOT_FOUND``; ``403 QUEUE_NOT_YOURS`` (an agent
                closes only queues it created - miForge creates its own, so this means the
                queue is someone else's); ``409 AUDIT_IN_PROGRESS``.
        """
        body: dict[str, Any] = {} if reason is None else {"reason": reason}
        response = self._request(
            "POST", f"/api/v1/review-queues/{queue_id}/close", json_body=body
        ).json()
        return dict(response) if isinstance(response, Mapping) else {}

    def post_candidates(
        self, queue_id: str, candidates: Sequence[Mapping[str, Any]]
    ) -> list[CandidateAccepted]:
        """Post one batch of candidates. **At most** :data:`CANDIDATE_BATCH_MAX`.

        Batching is the caller's job (:mod:`services.review.binding`) because the caller is
        the only one that knows which candidates are new. This method refuses an oversized
        batch locally rather than sending it: a 413 round trip costs the whole payload's
        upload for an answer miForge already had.
        """
        if not candidates:
            return []
        if len(candidates) > CANDIDATE_BATCH_MAX:
            from core.errors import refuse

            raise refuse(
                "review_batch_too_large",
                f"A miDataworks review batch holds at most {CANDIDATE_BATCH_MAX} "
                f"candidates and this one holds {len(candidates)}. Split it before "
                f"sending rather than sending it to be refused.",
                count=len(candidates),
                max=CANDIDATE_BATCH_MAX,
            )
        body = self._request(
            "POST",
            f"/api/v1/review-queues/{queue_id}/candidates",
            json_body={"candidates": [dict(c) for c in candidates]},
        ).json()
        items = body.get("items", []) if isinstance(body, Mapping) else []
        return [
            CandidateAccepted(
                external_id=str(item["external_id"]),
                item_id=str(item["item_id"]),
                created=bool(item.get("created", True)),
            )
            for item in items
        ]

    def read_decisions(
        self,
        queue_id: str,
        *,
        since: datetime | None = None,
        external_ids: Sequence[str] | None = None,
    ) -> list[ExternalDecision]:
        """Decisions on an external queue, by watermark and/or by candidate.

        **Both arguments, not one.** ``since`` catches the common case cheaply; the explicit
        ids catch a decision whose ``created_at`` precedes a candidate miForge posted late.
        Doing only one of the two loses decisions, and a lost decision is a candidate that
        stays pending forever while a human believes they answered it.
        """
        params: dict[str, Any] = {}
        if since is not None:
            params["since"] = since.isoformat()
        if external_ids:
            params["external_ids"] = list(external_ids)
        body = self._request(
            "GET", f"/api/v1/review-queues/{queue_id}/decisions", params=params
        ).json()
        items = body.get("items", []) if isinstance(body, Mapping) else []
        return [
            ExternalDecision(
                decision_id=str(item["decision_id"]),
                external_id=str(item["external_id"]),
                item_id=str(item["item_id"]),
                decision=str(item["decision"]),
                override_label=item.get("override_label"),
                reason=str(item.get("reason") or ""),
                decided_by=str(item.get("decided_by") or ""),
                decided_by_origin=str(item.get("decided_by_origin") or ""),
                created_at=str(item["created_at"]),
            )
            for item in items
        ]

    # ------------------------------------------------------------------ parts (R-02.37)

    def list_exports(
        self, *, version_id: str | None = None, limit: int = 50
    ) -> list[dict[str, Any]]:
        """The exports miDataworks offers. FR-010.19's listing, **read-only**.

        Never cached as truth: the authoritative record of a part miForge *has* is the
        manifest in the part's own directory. Two answers to "what parts exist" means the
        stale one is the one the UI shows.
        """
        params: dict[str, Any] = {"limit": limit}
        if version_id:
            params["version_id"] = version_id
        body = self._request("GET", "/api/v1/exports", params=params).json()
        items = body.get("items", []) if isinstance(body, Mapping) else []
        return [dict(item) for item in items if isinstance(item, Mapping)]

    def get_export(self, export_id: str) -> dict[str, Any]:
        """One export's record, including its ``files`` and ``manifest_sha256``."""
        body = self._request("GET", f"/api/v1/exports/{export_id}").json()
        return dict(body) if isinstance(body, Mapping) else {}

    def fetch_export_file(self, export_id: str, name: str) -> bytes:
        """One file of an export, as bytes.

        Bytes rather than a parsed object even for the manifest: FR-010.14 requires the
        manifest be hashed **as received**, and re-serialising would digest miForge's
        serialisation rather than what miDataworks sent.
        """
        return self._request("GET", f"/api/v1/exports/{export_id}/files/{name}").content

    # ------------------------------------------------------------------ liveness

    def ping(self) -> bool:
        """Whether miDataworks answers. Used only for health's ``reachable`` line.

        ``/api/health``, not ``/api/v1/health``: miDataworks mounts its health route at
        ``/api`` outside the version prefix, for the same reason miForge does - a liveness
        probe that moves when the API version moves breaks a cluster on a day nobody was
        changing the cluster. Verified in their ``api/v1/endpoints/health.py``, whose own
        docstring says *"Mounted at ``/api``, not ``/api/v1``"*.

        Never raises: health reports a state, and a probe that threw would make an
        unconfigured-or-down sibling able to fail the health route itself - which is the
        one thing a health route must not do.
        """
        try:
            self._request("GET", "/api/health")
        except Exception:
            return False
        return True
