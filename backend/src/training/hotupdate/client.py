"""The ONE outbound call site for miLLM's weights surface  (ADR-010, ADR-022; FR-011.21-23).

**No other module in miForge may call a weights endpoint.** The credential, both headers,
the per-op timeout, the response validation and the "never retry a refusal" rule are
applied in exactly one place, so there is one thing to mutate in review and one thing a
test must assert. ``tests/unit/test_hotupdate_fence.py`` asserts that count rather than
trusting it, with a self-test proving its scanner sees each form a reference can take.

**It lives here and not in ``clients/``.** ADR-017 puts one module per optional sibling
under ``clients/``; **ADR-022 is binding above it** and requires all of §5.10's code under
``training/hotupdate/``, which is also what this feature's four documents specify and what
the fence guard's one-directory allowlist can express. A ``clients/millm.py`` would put a
module that names a weights endpoint on the core path, and would trip F10's own
single-call-site test, which asserts that nothing under ``clients/`` but its own wrapper
imports ``httpx``. Recorded here because the instruction to create ``clients/millm.py`` was
reasonable and the fence is the reason it is not followed.

## The endpoint set, enumerated and diffed against BRD-01

:data:`OPS` is **every** op this client can send, and each one cites the requirement that
defines it. Enumerating them and diffing against the set BRD-01 §4 actually defines is the
check that caught the missing load call (*"a section title promising a capability is not a
requirement defining one"*) and the missing ``heartbeat``.
``tests/unit/test_hotupdate_client_headers.py``
re-runs that diff, so a future op added here without a citation is red.

## Timeouts, and why ``commit-end``'s is the longest

A late ``commit-end`` is **accepted** by miLLM (BRD-01 R-01.23a) and returns the set to
``clean``; abandoning the bracket leaves it ``torn`` and serving refused. So giving up on
``commit-end`` early is strictly worse than waiting, and that asymmetry is configuration
rather than a constant.

## Retry policy: one line, and it is a prohibition

A **refusal is never retried** - no backoff, no jitter, no "maybe transient" branch
(R-02.34, FR-011.35). Only a connection-level failure *before any request was accepted*
may be retried, once, and it is logged distinctly from a refusal so the two are never
confused in a post-mortem.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import httpx

from core.errors import Refusal, refuse
from training.hotupdate import identity, refusals
from training.hotupdate.settings import HotUpdateSettings

log = logging.getLogger(__name__)

#: Every op, its method and path, and the BRD-01 requirement that defines it.
#: **Cited, never restated**: no response field list appears anywhere in miForge.
OPS: dict[str, tuple[str, str, str]] = {
    "attach": ("POST", "/api/weights/attach", "R-01.1"),
    "detach": ("POST", "/api/weights/detach", "R-01.3"),
    "quiesce": ("POST", "/api/weights/quiesce", "R-01.6"),
    "resume": ("POST", "/api/weights/resume", "R-01.7"),
    "commit-begin": ("POST", "/api/weights/commit-begin", "R-01.21"),
    "commit-end": ("POST", "/api/weights/commit-end", "R-01.21"),
    "save": ("POST", "/api/weights/save", "R-01.10"),
    "load": ("POST", "/api/weights/load", "R-01.25"),
    "lineage": ("GET", "/api/weights/lineage", "R-01.16"),
    "status": ("GET", "/api/weights/status", "R-01.20"),
    "heartbeat": ("POST", "/api/weights/heartbeat", "R-01.30"),
}


def parameter_count(record: dict[str, Any] | None) -> int | None:
    """``num_parameters`` from a catalogue record, when it is a positive int."""
    value = (record or {}).get("num_parameters")
    return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None


#: Ops that write or verify a whole checkpoint before answering, and so get the long timeout.
CHECKPOINT_OPS: frozenset[str] = frozenset({"save", "load"})

#: Ops BRD-01 defines that this client deliberately **does not** send, and why. Kept as
#: data rather than as prose so a test can assert the absence (FR-011.34, AC-011.10).
DELIBERATELY_ABSENT: dict[str, str] = {
    "clear-stale": (
        "Clearing a stale attachment is OPERATOR-ONLY. miForge holds 'trainer', so the call "
        "is unreachable by protocol - and miForge's own policy is the same, so the code "
        "does not contain it at all."
    ),
    "instantiate": (
        "Instantiating through miLLM is deferred upstream: its only consumer was miForge's "
        "from-scratch run routed through miLLM, and standalone miForge initialises its own "
        "weights."
    ),
    "adapter/load": "Loading an adapter is serving-side; miForge produces adapters and never "
    "serves them.",
    "adapter/unload": "Unloading an adapter is serving-side, as above.",
}


@dataclass
class CallRecord:
    """One call, as made. **Payload and headers both**, because the assertion needs both.

    ADR-016: *"was called" passes against a call sending the wrong arguments.* A client log
    that recorded only the op would make every header test vacuous.
    """

    op: str
    method: str
    path: str
    headers: dict[str, str]
    body: dict[str, Any] | None


@dataclass
class WeightsClient:
    """Every call miForge makes to miLLM's weights surface, and the only module that does.

    Args:
        settings: §5.10's live settings.
        attachment_id: the run's attachment id, or ``None`` before an attach. **Run state,
            never a module global** (FR-011.5): miLLM returns it once and no read path
            returns it again, so a worker restart that lost it has lost the ability to
            commit. :meth:`remember_attachment` is how it arrives, and the caller persists
            it before any other call is made.
        transport: an ``httpx`` transport, for tests. Production passes ``None`` and the
            fake is reached by pointing the **configured base URL** at a transport - never
            by patching ``httpx``, which would bypass the headers and the refusal mapping
            that are the point of this module.
    """

    settings: HotUpdateSettings
    attachment_id: str | None = None
    transport: httpx.BaseTransport | None = None
    #: Every call made, in order. Not telemetry: the stage-order assertion for the commit
    #: bracket is made on this sequence, and an inversion is only visible as a sequence.
    calls: list[CallRecord] = field(default_factory=list)
    _client: httpx.Client | None = field(default=None, repr=False)

    # ------------------------------------------------------------------ lifecycle

    def __post_init__(self) -> None:
        if self.settings.base_url is None:
            raise refuse(
                "millm_not_configured",
                "miLLM is not configured, so no weights call can be made. Set "
                "MILLM_BASE_URL to use the optional hot-update mode; every "
                "other acquisition mode and run type is unaffected.",
                capabilities_unavailable=list(_CAPABILITIES),
            )
        # ONE connection for the attachment's lifetime (TID §13 hint 3). Four calls per step
        # at a fresh connection each is latency added to *held serving time* for no reason.
        self._client = httpx.Client(
            base_url=self.settings.base_url.rstrip("/"),
            transport=self.transport,
            timeout=httpx.Timeout(self.settings.request_timeout_seconds),
        )

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    def __enter__(self) -> WeightsClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def remember_attachment(self, attachment_id: str) -> None:
        """Hold the id for the attachment's lifetime. The caller persists it as run state."""
        self.attachment_id = attachment_id

    # ------------------------------------------------------------------ the one call

    def call(self, op: str, body: dict[str, Any] | None = None) -> dict[str, Any]:
        """Make one weights call. **No per-endpoint branches.**

        Raises:
            Refusal: ``weights_op_forbidden`` for :data:`identity.FORBIDDEN_OP` - a
                programming error, raised before anything is sent;
                ``weights_identity_rejected`` on ``401``/``403`` (**the run stops, no
                retry**); whatever :func:`refusals.interpret` names for any other non-2xx;
                ``weights_surface_unreachable`` for a transport failure or a body that is
                not readable as JSON.
        """
        if op == identity.FORBIDDEN_OP:
            # FR-011.34's innermost layer. A programming error rather than a refusal of the
            # operator's request: nothing in miForge may ask for this, so the right failure
            # is the one that stops a developer, not one that reads as a configuration
            # problem. MUT-7 adds such a call and this is one of the two things that catch
            # it; the other is the source scan, which catches it even if this line is gone.
            raise refuse(
                "weights_op_forbidden",
                f"miForge must never call {op!r}: clearing another process's attachment is "
                f"an operator action, and it is how two trainers end up writing the same "
                f"tensors. {DELIBERATELY_ABSENT[op]}",
                op=op,
            )
        if op not in OPS:
            raise refuse(
                "weights_op_unknown",
                f"{op!r} is not an op on miLLM's weights surface. "
                f"Known: {sorted(OPS)}. Refusing rather than guessing a path - a guessed "
                f"endpoint is a request against a contract nobody wrote.",
                op=op,
                known=sorted(OPS),
                deliberately_absent=sorted(DELIBERATELY_ABSENT),
            )

        method, path, requirement = OPS[op]
        headers = identity.headers_for(
            op,
            token=self.settings.trainer_token,
            label=self.settings.audit_label,
            attachment_id=self.attachment_id,
        )
        self.calls.append(
            CallRecord(op=op, method=method, path=path, headers=dict(headers), body=body)
        )
        if op == "commit-end":
            timeout = self.settings.commit_end_timeout_seconds
        elif op in CHECKPOINT_OPS:
            timeout = self.settings.checkpoint_timeout_seconds
        else:
            timeout = self.settings.request_timeout_seconds
        assert self._client is not None  # set in __post_init__, cleared only by close()
        try:
            response = self._client.request(
                method, path, json=body, headers=headers, timeout=timeout
            )
        except httpx.HTTPError as exc:
            raise refuse(
                "weights_surface_unreachable",
                f"The {op!r} call to miLLM ({method} {path}) did not "
                f"complete: {type(exc).__name__}. Reported rather than retried: at this "
                f"point it is unknown whether miLLM accepted the request, and a blind retry "
                f"of a weights call can double a write. Check that the serving process is "
                f"up and holds an attachable model.",
                op=op,
                requirement=requirement,
                error=type(exc).__name__,
            ) from exc

        return self._interpret(op, response)

    # ------------------------------------------------------------------ the boundary

    def resident_model_record(self, model_id: int) -> dict[str, Any] | None:
        """The resident model's catalogue record (``GET /api/models/{id}``), or ``None``.

        Not a weights op (it is not on the attachment's surface and carries no identity).
        Pre-flight reads two facts from it: ``num_parameters`` for the budget guard, and
        ``quantization`` for R-01.5's attachability (see :func:`resident_model_size`).
        """
        assert self._client is not None
        try:
            response = self._client.get(
                f"/api/models/{int(model_id)}", timeout=self.settings.request_timeout_seconds
            )
        except httpx.HTTPError:
            return None
        body = self._json_or_none(response)
        if response.status_code >= 400 or body is None:
            return None
        data = body.get("data")
        return data if isinstance(data, dict) else None

    def resident_model_size(self, model_id: int) -> int | None:
        """``num_parameters`` of the resident model, from miLLM's ungated catalogue read
        (filled while the model is loaded), or ``None``. The budget guard needs the size an
        attach pre-flight otherwise lacks."""
        return parameter_count(self.resident_model_record(model_id))

    def _interpret(self, op: str, response: httpx.Response) -> dict[str, Any]:
        """Validate at the boundary. **No partially read object, no ``getattr`` default.**"""
        body = self._json_or_none(response)
        if response.status_code in (401, 403):
            upstream_code = refusals._code_of(body)
            # 403 NOT_ATTACHMENT_HOLDER is a *holder* refusal, not a credential refusal:
            # R-01.31 uses the same status for a valid token that is not the holder. Mapping
            # it to "fix your credential" would send the operator to the wrong file.
            if upstream_code != "NOT_ATTACHMENT_HOLDER":
                raise identity.identity_rejected(
                    op,
                    response.status_code,
                    label=self.settings.audit_label,
                    upstream={"code": upstream_code},
                )
        if response.status_code >= 400:
            # The holder is read ONLY for the already-attached row, which is the one
            # R-02.33 says must name who holds it. Reading it on every refusal would add a
            # call to every failure path - which also makes a zero-call-count assertion
            # (AC-011.18) fail for a reason that has nothing to do with what it tests.
            holder = (
                self._holder_if_known(op) if refusals._code_of(body) == "ALREADY_ATTACHED" else None
            )
            raise refusals.interpret(response.status_code, body, op=op, holder=holder)
        if body is None:
            raise refuse(
                "weights_surface_unreachable",
                f"miLLM answered the {op!r} call with HTTP {response.status_code} and a "
                f"body that is not a JSON object. Refusing rather than reading what can be "
                f"parsed of it: a half-read response produces a fabricated value, which is "
                f"worse than a refusal because it is silent.",
                op=op,
                status=response.status_code,
            )
        return unwrap_envelope(op, body)

    @staticmethod
    def _json_or_none(response: httpx.Response) -> dict[str, Any] | None:
        try:
            parsed = response.json()
        except ValueError:
            return None
        return parsed if isinstance(parsed, dict) else None

    def _holder_if_known(self, op: str) -> str | None:
        """The holding identity for an already-attached refusal, from a status read.

        **Read, not cached** (FR-011.36's rule applied here): it is another process's state.
        A failure to read it returns ``None`` rather than raising, because the refusal being
        reported is the important one - losing it to a secondary failure would replace a
        precise diagnosis with a vaguer one.
        """
        if op == "status":
            return None
        try:
            status = self.call("status")
        except Refusal:
            return None
        from training.hotupdate import state

        return state.holder_of(status)


def unwrap_envelope(op: str, body: dict[str, Any]) -> dict[str, Any]:
    """The payload of a successful weights call, out of miLLM's management envelope.

    **Every** ``/api/weights/*`` route answers ``{"success": true, "data": {...}, "error":
    null}`` (miLLM ``millm/api/routes/management/weights.py``: ``ApiResponse.ok(...)`` on all
    of them; ``manual/docs/api/weights.md``: *"answer in the management envelope"*). Until
    2026-10-17 this client returned the envelope itself, so every field read one level too
    high - ``weights_state`` on a status read, ``attachment_id`` on attach - and the fake
    surface agreed by construction, because it never wrapped anything. Found on the GPU
    node before the first live call (F11 14.3).

    Refuses a body that is not the envelope rather than reading it as the payload: a
    top-level read of an unwrapped body is exactly the silent misreading this closes.
    """
    if "success" not in body or "data" not in body:
        raise refuse(
            "weights_surface_unreachable",
            f"miLLM answered the {op!r} call with a body that is not its management "
            f"envelope ({{success, data, error}}). Refusing rather than reading fields at "
            f"the top level: that is how every field of every weights call was misread.",
            op=op,
            keys=sorted(body)[:10],
        )
    data = body["data"]
    if body["success"] is not True or not isinstance(data, dict):
        raise refuse(
            "weights_surface_unreachable",
            f"miLLM answered the {op!r} call with a 2xx envelope that is not a success "
            f"carrying an object (success={body['success']!r}, data is "
            f"{type(data).__name__}). Refusing rather than guessing what it meant.",
            op=op,
        )
    return data


_CAPABILITIES = (
    "attach to a serving process's resident tensors",
    "update weights in place while they are being served",
)
