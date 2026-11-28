"""A fake miDataworks, written independently of anything in that repository  (ADR-016).

**Why "independently" is a requirement and not a nicety.** A fake built from a double in
miDataworks' own tests would encode its author's reading of its own API on **both** sides of
every test here, so a shared misunderstanding passes. miDataworks took exactly this care in
its own standalone test, using a fake *"written independently of the miLLM fake"*. So this
file was written from their **route signatures and response models** - the endpoint
decorators in ``api/v1/endpoints/review.py`` and ``exports.py``, the Pydantic models in
``schemas/review.py`` and ``schemas/publishing.py``, and the refusal codes raised in
``services/review/external_service.py`` - and from nothing else.

**It implements the refusals, not just the happy path.** The happy path is the easy half and
the half that was going to work anyway; every expensive bug on this seam is in a refusal:

===========================  ===================================================  ======
Condition                    Code                                                 Status
===========================  ===================================================  ======
no / malformed agent header   ``ORIGIN_APP_REQUIRED`` / ``INVALID_AGENT_IDENTITY``  400
posting to another app's      ``ORIGIN_APP_MISMATCH``                              403
  queue
candidates to a non-external  ``QUEUE_KIND_INVALID``                               422
  queue
batch over 500, or a          ``PAYLOAD_TOO_LARGE``                                413
  candidate over the ceiling
unknown queue                 ``queue_not_found``                                  404
===========================  ===================================================  ======

**It is an ``httpx.MockTransport``**, not a running server: no network, no port, no event
loop. The F10 FTDD specifies "an ASGI app mounted over httpx's transport"; a mock transport
is the same arrangement with one fewer moving part, and the independence requirement - which
is the part that actually protects the tests - is unchanged.

**Idempotence is modelled, because it is load-bearing.** A repeated ``external_id`` returns
the existing item with ``created: False``, which is what makes a re-post after a crash safe.
A fake that created a duplicate would make miForge's crash-and-resume test pass for the
wrong reason.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qs

import httpx

#: Their limits, as their implementation states them: ``CANDIDATE_BATCH_MAX = 500`` in
#: ``services/calibration/constants.py``, and ``review_candidate_max_bytes`` (default
#: 65,536) in ``core/config.py``.
BATCH_MAX = 500
CANDIDATE_MAX_BYTES = 65_536

AGENT_HEADER = "x-dataworks-agent"
#: Their pattern, from ``core/agent_origin.py``.
AGENT_PATTERN = r"^agent:[a-z0-9][a-z0-9-]{0,47}$"


def _error(code: str, message: str, status: int, **details: Any) -> httpx.Response:
    """Their error envelope, which happens to be the same shape as miForge's."""
    return httpx.Response(
        status,
        json={"error": {"code": code, "message": message, "details": details}},
    )


@dataclass
class FakeQueue:
    id: str
    kind: str
    question: str
    label_set: list[str]
    origin_app: str | None
    external_ref: dict[str, Any]
    items: dict[str, str] = field(default_factory=dict)
    decisions: list[dict[str, Any]] = field(default_factory=list)
    #: miDataworks' D2 close transition (its e2e/outstanding 989038f): ``open`` -> ``closed``.
    created_by: str = "operator"
    state: str = "open"
    closed_at: str | None = None
    closed_by: str | None = None
    closed_by_origin: str | None = None
    close_reason: str | None = None
    #: Their audit queues refuse a close while the audit is in progress (409).
    audit_in_progress: bool = False


@dataclass
class FakeMiDataworks:
    """A fake with a record of what it was asked.

    ``calls`` exists because a reachability or dry-run assertion must check the **call
    count**: *"was not called with" passes against a call sending the wrong arguments*, and
    "the client is never used during a dry-run" is only provable against a counter.
    """

    queues: dict[str, FakeQueue] = field(default_factory=dict)
    exports: dict[str, dict[str, Any]] = field(default_factory=dict)
    export_files: dict[tuple[str, str], bytes] = field(default_factory=dict)
    calls: list[tuple[str, str]] = field(default_factory=list)
    #: Statuses to return before behaving normally, e.g. ``[503, 503]`` to exercise retries.
    fail_with: list[int] = field(default_factory=list)
    #: ``Retry-After`` to send with a 5xx, so the backoff path is exercised with a value
    #: rather than only with the exponential default.
    retry_after: str | None = None
    #: When set, every request answers this status - for the 401-no-retry case.
    always_fail: int | None = None
    next_id: int = 1

    # ------------------------------------------------------------------ wiring

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def call_count(self, method: str | None = None, path: str | None = None) -> int:
        """How many calls matched. The assertion a dry-run guard needs is ``== 0``."""
        return sum(
            1
            for m, p in self.calls
            if (method is None or m == method) and (path is None or p == path)
        )

    # ------------------------------------------------------------------ the router

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        method = request.method
        self.calls.append((method, path))

        if self.always_fail is not None:
            return _error("fake_always_fail", "forced failure", self.always_fail)
        if self.fail_with:
            status = self.fail_with.pop(0)
            headers = {"Retry-After": self.retry_after} if self.retry_after else {}
            return httpx.Response(
                status,
                json={"error": {"code": "fake_transient", "message": "try again", "details": {}}},
                headers=headers,
            )

        agent = request.headers.get(AGENT_HEADER)
        if agent is not None:
            import re

            if not re.fullmatch(AGENT_PATTERN, agent):
                # Their middleware refuses a malformed header on EVERY request, and an
                # empty header is malformed rather than absent.
                return _error(
                    "INVALID_AGENT_IDENTITY",
                    "X-Dataworks-Agent must look like agent:<name>.",
                    400,
                )

        if path == "/api/health":
            return httpx.Response(200, json={"status": "ok"})
        if path == "/api/v1/review-queues" and method == "POST":
            return self._create_queue(request, agent)
        if path == "/api/v1/review-queues" and method == "GET":
            wanted = parse_qs(request.url.query.decode()).get("state", [None])[0]
            rows = [
                self._queue_out(q)
                for q in self.queues.values()
                if wanted is None or q.state == wanted
            ]
            return httpx.Response(200, json={"items": rows, "total": len(rows)})
        if path == "/api/v1/exports" and method == "GET":
            return httpx.Response(
                200,
                json={
                    "items": list(self.exports.values()),
                    "total": len(self.exports),
                    "page": 1,
                    "limit": 50,
                },
            )

        parts = [segment for segment in path.split("/") if segment]
        if len(parts) >= 4 and parts[:3] == ["api", "v1", "review-queues"]:
            queue_id = parts[3]
            queue = self.queues.get(queue_id)
            if queue is None:
                return _error("QUEUE_NOT_FOUND", f"No review queue {queue_id}.", 404)
            if len(parts) == 5 and parts[4] == "close" and method == "POST":
                return self._close(request, queue, agent)
            if len(parts) == 4 and method == "GET":
                return httpx.Response(200, json=self._queue_out(queue))
            if len(parts) == 5 and parts[4] == "candidates" and method == "POST":
                return self._post_candidates(request, queue, agent)
            if len(parts) == 5 and parts[4] == "decisions" and method == "GET":
                return self._read_back(request, queue)

        if len(parts) >= 4 and parts[:3] == ["api", "v1", "exports"]:
            export_id = parts[3]
            if len(parts) == 4 and method == "GET":
                record = self.exports.get(export_id)
                if record is None:
                    return _error("export_not_found", f"No export {export_id}.", 404)
                return httpx.Response(200, json=record)
            if len(parts) == 6 and parts[4] == "files":
                body = self.export_files.get((export_id, parts[5]))
                if body is None:
                    return _error("export_file_not_found", f"No file {parts[5]}.", 404)
                return httpx.Response(200, content=body)

        return _error("http_404", f"No route {method} {path}.", 404)

    # ------------------------------------------------------------------ review

    def _create_queue(self, request: httpx.Request, agent: str | None) -> httpx.Response:
        body = json.loads(request.content or b"{}")
        if body.get("kind") != "external":
            return _error("QUEUE_KIND_INVALID", "Candidates go to an external queue only.", 422)
        labels = body.get("label_set") or []
        if not 2 <= len(labels) <= 64:
            return _error("DECISION_INVALID", "label_set holds 2 to 64 entries.", 422)
        queue_id = f"rq{self.next_id:04d}"
        self.next_id += 1
        # `origin_app` comes from the CREATING caller's agent header - which is why an
        # operator-created queue is one miForge cannot post to, and why miForge must create
        # its own.
        self.queues[queue_id] = FakeQueue(
            id=queue_id,
            kind="external",
            question=str(body.get("question", "")),
            label_set=[str(label) for label in labels],
            origin_app=None if agent is None else agent.removeprefix("agent:"),
            external_ref=dict(body.get("external_ref") or {}),
            created_by=agent or "operator",
        )
        return httpx.Response(201, json=self._queue_out(self.queues[queue_id]))

    def _queue_out(self, queue: FakeQueue) -> dict[str, Any]:
        return {
            "id": queue.id,
            "kind": queue.kind,
            "question": queue.question,
            "label_set": queue.label_set,
            "origin_app": queue.origin_app,
            "external_ref": queue.external_ref,
            "state": queue.state,
            "closed_at": queue.closed_at,
            "closed_by": queue.closed_by,
            "closed_by_origin": queue.closed_by_origin,
            "close_reason": queue.close_reason,
            "created_by": queue.created_by,
            "items": len(queue.items),
            "decided": len({d["external_id"] for d in queue.decisions}),
        }

    def _post_candidates(
        self, request: httpx.Request, queue: FakeQueue, agent: str | None
    ) -> httpx.Response:
        app = None if agent is None else agent.removeprefix("agent:")
        if app is None:
            return _error(
                "ORIGIN_APP_REQUIRED",
                "External candidates come from an application, which identifies itself "
                "with X-Dataworks-Agent.",
                400,
            )
        if queue.kind != "external":
            return _error("QUEUE_KIND_INVALID", "Candidates go to an external queue only.", 422)
        if queue.state == "closed":
            return _error("QUEUE_CLOSED", "This queue is closed and takes no candidates.", 409)
        if queue.origin_app != app:
            return _error(
                "ORIGIN_APP_MISMATCH",
                f"This queue belongs to {queue.origin_app}; {app} cannot post to it.",
                403,
            )

        body = json.loads(request.content or b"{}")
        candidates = body.get("candidates") or []
        if len(candidates) > BATCH_MAX:
            return _error(
                "PAYLOAD_TOO_LARGE",
                f"A batch holds at most {BATCH_MAX} candidates; split it.",
                413,
                count=len(candidates),
                max=BATCH_MAX,
            )
        for candidate in candidates:
            size = len(json.dumps(candidate, sort_keys=True, separators=(",", ":")).encode())
            if size > CANDIDATE_MAX_BYTES:
                return _error(
                    "PAYLOAD_TOO_LARGE",
                    f"Candidate {candidate.get('external_id')!r} is {size} bytes.",
                    413,
                    external_id=candidate.get("external_id"),
                    bytes=size,
                    max=CANDIDATE_MAX_BYTES,
                )

        out = []
        for candidate in candidates:
            external_id = str(candidate["external_id"])
            existing = queue.items.get(external_id)
            if existing is not None:
                # Idempotent: the existing item, not a duplicate.
                out.append({"external_id": external_id, "item_id": existing, "created": False})
                continue
            item_id = f"ri{self.next_id:04d}"
            self.next_id += 1
            queue.items[external_id] = item_id
            out.append({"external_id": external_id, "item_id": item_id, "created": True})
        return httpx.Response(201, json={"items": out})

    def _close(self, request: httpx.Request, queue: FakeQueue, agent: str | None) -> httpx.Response:
        """``POST /review-queues/{id}/close`` exactly as miDataworks D2 ships it.

        Order of rules, theirs (``queue_service.close_queue``): an agent closes only a queue
        it created (403 ``QUEUE_NOT_YOURS``); a closed queue is answered as it stands (200,
        the first close's who/when/why kept); an audit in progress is 409
        ``AUDIT_IN_PROGRESS``. Body ``{}``, ``{"reason": ...}`` or none.
        """
        who = agent or "operator"
        origin = "agent" if agent else "operator"
        if origin == "agent" and queue.created_by != who:
            return _error(
                "QUEUE_NOT_YOURS",
                f"{who} may close only queues it created; this one was created by "
                f"{queue.created_by}.",
                403,
                queue_id=queue.id,
                created_by=queue.created_by,
            )
        if queue.state == "closed":
            return httpx.Response(200, json=self._queue_out(queue))
        if queue.audit_in_progress:
            return _error("AUDIT_IN_PROGRESS", "This audit is in progress.", 409, queue_id=queue.id)
        body = json.loads(request.content) if request.content else None
        reason = body.get("reason") if isinstance(body, dict) else None
        queue.state = "closed"
        queue.closed_at = datetime.now(tz=UTC).isoformat()
        queue.closed_by = who
        queue.closed_by_origin = origin
        queue.close_reason = reason
        return httpx.Response(200, json=self._queue_out(queue))

    def _read_back(self, request: httpx.Request, queue: FakeQueue) -> httpx.Response:
        query = parse_qs(request.url.query.decode())
        wanted = query.get("external_ids")
        since = query.get("since", [None])[0]
        rows = list(queue.decisions)
        if wanted:
            rows = [row for row in rows if row["external_id"] in set(wanted)]
        if since:
            # Strictly greater than, as their `created_at > since` is - a watermark that
            # included its own boundary would re-read the last decision forever.
            rows = [row for row in rows if row["created_at"] > since]
        return httpx.Response(200, json={"items": rows})

    # ------------------------------------------------------------------ test helpers

    def decide(
        self,
        queue_id: str,
        external_id: str,
        decision: str,
        *,
        label: str | None = None,
        reason: str = "because",
        at: datetime | None = None,
        decided_by: str = "operator",
    ) -> str:
        """Record a decision as a person would, returning its id.

        Deliberately permissive about the vocabulary so a test can post a kind miForge's
        mapping does not know - the only way to exercise the *refuse rather than guess*
        path on a real read-back shape.
        """
        queue = self.queues[queue_id]
        if queue.state == "closed":
            raise ValueError("QUEUE_CLOSED: a closed queue takes no decisions (miDataworks D2)")
        decision_id = f"rd{self.next_id:04d}"
        self.next_id += 1
        queue.decisions.append(
            {
                "decision_id": decision_id,
                "external_id": external_id,
                "item_id": queue.items.get(external_id, "ri-unknown"),
                "decision": decision,
                "override_label": label,
                "reason": reason,
                "decided_by": decided_by,
                "decided_by_origin": "operator",
                "created_at": (at or datetime.now(tz=UTC)).isoformat(),
            }
        )
        return decision_id

    def add_export(
        self,
        export_id: str,
        *,
        files: dict[str, bytes],
        target: str = "miforge_set",
        miforge_set_kind: str | None = "prompt_set",
        status: str = "complete",
    ) -> None:
        """Register an export and its files, in their ``ExportOut`` shape."""
        self.exports[export_id] = {
            "id": export_id,
            "job_id": f"job-{export_id}",
            "target": target,
            "version_id": "7b5d1d0e-0000-4000-8000-000000000001",
            "config_version_id": None,
            "params": {"miforge_set_kind": miforge_set_kind},
            "trl_version": None,
            "status": status,
            "files": [{"path": name, "role": "data"} for name in files],
            "manifest_sha256": None,
            "error": None,
            "started_by": "operator",
            "started_by_origin": "operator",
            "created_at": "2026-10-14T00:00:00Z",
            "completed_at": "2026-10-14T00:00:01Z",
        }
        for name, body in files.items():
            self.export_files[(export_id, name)] = body
