"""A fake for miLLM's weights surface, built from BRD-01's requirement text.

**There is nothing to record and nothing to import.** miLLM has no weight
attach/detach/quiesce/resume/commit/save/load code - BRD-01 is specced and unimplemented,
and its only ``attach`` is SAE attach, which is unrelated. ADR-016 is explicit: *mock a
sibling's HTTP surface, never its code.* So this file **is** the contract as F11
understands it, and **every route cites the requirement it implements**: a route with no
citation is an invention, and F11 would then be built against it.

It is an ``httpx.MockTransport``, not a running server - no port, no event loop - reached by
pointing the **configured base URL** at it, which exercises the real client including its
headers, its refusal mapping and its no-retry rule. Patching ``httpx`` would bypass all
three.

## It must be able to misbehave, and that is the point

*A fake that only succeeds makes every refusal path in F11 agree with its test by
construction*, which is exactly how a suite stays green over a real bug (ADR-016). This one
can:

1. **refuse** every code BRD-01 defines - ``409 ALREADY_ATTACHED`` (R-01.2),
   ``400 NOT_ATTACHABLE`` (R-01.5), ``409 WEIGHTS_NOT_CLEAN`` (R-01.7, R-01.11),
   ``503 WEIGHTS_TORN`` (R-01.22), ``403 NOT_ATTACHMENT_HOLDER`` and ``409 NOT_ATTACHED``
   (R-01.31), ``401``/``403`` on the credential (R-01.17), and
   ``409 ATTACHMENT_NOT_STALE`` (R-01.29);
2. **stall** - past the drain timeout and, separately, past the commit watchdog;
3. accept a **late ``commit-end``** and report the overrun (R-01.23a);
4. **restart with its barrier state reset to ``clean``** while its resident identity stays
   at the last load - the only way to test FR-011.37, and the AMB-J case;
5. hold a **stale** attachment, and refuse ``clear-stale`` to a ``trainer`` role (R-01.29).

Time is **controlled, never slept.** ``trip_commit_watchdog()`` makes the watchdog expire
and ``stall_next(op)`` makes one call time out. A test that really waits out a watchdog is a
test nobody runs.

## The fifteen endpoints, enumerated

BRD-01 §4 defines fifteen: attach, instantiate, detach, quiesce, resume, commit-begin,
commit-end, save, load, lineage, adapter/load, adapter/unload, clear-stale, status and -
since R-01.30, added 2026-10-13 - **heartbeat**. All fifteen are routed here, including the
four F11 never calls, **so that a call F11 should not make reaches a route that records it**
rather than a 404 that looks like a transport problem.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import httpx

ATTACHMENT_HEADER = "x-millm-attachment"

#: R-01.5: the layout must be plain FP16 or BF16; attach is refused on a quantized model.
ATTACHABLE_DTYPES = ("float16", "bfloat16")


def _error(code: str, message: str, status: int, **details: Any) -> httpx.Response:
    """BRD-01 names codes and statuses, not an envelope. This is miForge's own shape, which
    the client reads defensively precisely because the real one is not fixed anywhere."""
    return httpx.Response(
        status, json={"error": {"code": code, "message": message, "details": details}}
    )


@dataclass
class FakeWeightsSurface:
    """miLLM's weights surface, as BRD-01 specifies it, with the five misbehaviours.

    ``calls`` records ``(op, headers, body)`` for **every** request, because the assertions
    this fake exists for are about the payload and the **count**: *"was called" passes
    against a call sending the wrong arguments*, and a zero-call assertion (AC-011.18) is
    only provable against a counter.
    """

    # ------------------------------------------------------------------ resident state
    resident_model: str = "LFM2-1.2B"
    manifest_sha256: str | None = "a" * 64
    weights_state: str = "clean"
    steps_since_load: int = 0
    dtype: str = "bfloat16"
    #: R-01.30's two values. **Both**, because FR-011.16 reads the hint rather than a local
    #: constant, and a fake that omitted them could not test that at all.
    staleness_window_seconds: float = 60.0
    #: ⚠ MUST NOT EQUAL `heartbeat.FALLBACK_INTERVAL_SECONDS`, and it did (both 10.0).
    #: While they matched, no test could distinguish "read miLLM's hint" from "fell through
    #: to the local constant" — the two produce the same number — so FR-011.16 was
    #: unobservable *by construction*, which is the fixtures-agree-by-construction failure
    #: ADR-016 names. 5.0 is also miLLM's own FTDD 031 §5.3 example (`interval_hint_s: 5`),
    #: so the fake is more faithful as well as more discriminating.
    #: `test_hotupdate_heartbeat_is_wired.py` asserts the two differ before relying on it.
    suggested_interval_seconds: float = 5.0
    #: miLLM's `pause_unavailable` on the status body: `"cbm"`, `"concurrency"` or None.
    #: Default None = the barrier CAN pause, which is the normal deployment.
    pause_unavailable: str | None = None
    #: Omit the key entirely, modelling an older miLLM that does not report it. Separate
    #: from `pause_unavailable = None` ON PURPOSE: absent is "unknown" and null is "it can
    #: pause", and a fake that could not express both would make the distinction the reader
    #: draws untestable — which is the fixture-shaped vacuity this suite keeps finding.
    omit_pause_unavailable: bool = False

    # ------------------------------------------------------------------ attachment
    attachment_id: str | None = None
    attached_by: str | None = None
    #: An attachment held by someone else, for R-01.2's ``409 ALREADY_ATTACHED``.
    foreign_holder: str | None = None
    stale: bool = False

    # ------------------------------------------------------------------ identity
    #: R-01.17: a bearer token carrying a role and an audit-log label.
    valid_token: str = "trainer-token"
    #: ``401`` when the token is absent or wrong; ``403`` when the role is insufficient.
    role: str = "trainer"

    # ------------------------------------------------------------------ misbehaviours
    #: Ops that should time out, consumed one at a time.
    stalls: list[str] = field(default_factory=list)
    #: ``{op: (status, code)}`` - refuse that op once.
    refuse_once: dict[str, tuple[int, str]] = field(default_factory=dict)
    #: ``{op: (status, code)}`` - refuse that op every time.
    refuse_always: dict[str, tuple[int, str]] = field(default_factory=dict)
    commit_open: bool = False
    commit_overran: bool = False
    held_requests: int = 3
    quiesced: bool = False
    #: Set when the watchdog has expired on an open commit (R-01.22): the state is ``torn``
    #: and miLLM **does not resume serving**.
    watchdog_expired: bool = False
    #: miLLM's commit watchdog (BRD-01 R-01.22), **in real time**, when set: a commit left
    #: open longer than this is declared ``torn`` on the next request, exactly as miLLM's
    #: watchdog does (200 ms since miLLM ``db44fea``). ``None`` keeps the manual
    #: :meth:`trip_commit_watchdog` as the only trigger, which every older test relies on.
    commit_watchdog_ms: float | None = None
    commit_begun_at: float | None = None

    # ------------------------------------------------------------------ observation
    calls: list[tuple[str, dict[str, str], Any]] = field(default_factory=list)
    last_headers: dict[str, str] = field(default_factory=dict)
    #: Completions served while the set was not ``clean``. **Must stay empty** (BRD AC9,
    #: R-01.22: miLLM never serves a torn parameter set).
    served_while_not_clean: list[str] = field(default_factory=list)
    saves: list[dict[str, Any]] = field(default_factory=list)
    loads: list[dict[str, Any]] = field(default_factory=list)

    # ------------------------------------------------------------------ wiring

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def call_count(self, op: str | None = None) -> int:
        """How many calls matched. The assertion AC-011.18 needs is ``== 0``."""
        return sum(1 for recorded, _, _ in self.calls if op is None or recorded == op)

    def ops(self) -> list[str]:
        """The op sequence. **The stage-order assertion is made on this** - an inversion is
        only visible as a sequence, since every call succeeds in the wrong order too."""
        return [op for op, _, _ in self.calls]

    def headers_of(self, op: str) -> list[dict[str, str]]:
        return [headers for recorded, headers, _ in self.calls if recorded == op]

    def bodies_of(self, op: str) -> list[Any]:
        return [body for recorded, _, body in self.calls if recorded == op]

    # ------------------------------------------------------------------ misbehaviour 2/3

    def stall_next(self, op: str) -> None:
        """Make the next ``op`` time out. **Deterministic, not slow.**"""
        self.stalls.append(op)

    def trip_commit_watchdog(self) -> None:
        """R-01.22: the commit watchdog expires. The set is ``torn`` and serving is refused.

        miLLM *"marks the state torn, logs it with the attaching identity and the elapsed
        time, and does not resume serving"*, and **never self-resumes out of it**. So this
        does not clear ``commit_open``: a late ``commit-end`` from the same attachment is
        still accepted (R-01.23a), which is misbehaviour 3.
        """
        self.watchdog_expired = True
        self.weights_state = "torn"

    def restart(self) -> None:
        """Misbehaviour 4: the serving process restarts (FR-011.37, FPRD AMB-J).

        **The barrier state does not survive** - it lives in process memory and a restart
        begins ``clean`` - while the resident identity goes back to the last **load**, not
        the last step. That combination is what makes ``clean`` a lie here, and is the only
        reason F11's state read also reads an identity. Note what else a restart loses: the
        attachment, which is why the id is unrecoverable afterwards.
        """
        self.weights_state = "clean"
        self.commit_open = False
        self.watchdog_expired = False
        self.quiesced = False
        self.attachment_id = None
        self.attached_by = None
        self.steps_since_load = 0

    def make_stale(self, holder: str = "trainer:nightly-forge") -> None:
        """Misbehaviour 5: an attachment held by someone else and marked stale (R-01.4)."""
        self.foreign_holder = holder
        self.stale = True

    def serve_completion(self) -> str:
        """What ``/v1/*`` would answer right now (R-01.22, BRD AC9, AC-011.1).

        Records a violation rather than raising, so a test asserts on the *record*: a fake
        that raised would make "was a completion served from a non-clean set" a question
        about this file's control flow instead of about F11's behaviour.
        """
        if self.weights_state != "clean" or self.quiesced:
            self.served_while_not_clean.append(self.weights_state)
            return ""
        return f"served by {self.resident_model}"

    # ------------------------------------------------------------------ the router

    def handle(self, request: httpx.Request) -> httpx.Response:
        op = request.url.path.removeprefix("/api/weights/")
        body = self._body(request)
        headers = {key.lower(): value for key, value in request.headers.items()}
        self.calls.append((op, headers, body))
        #: The current request's headers, for the two routes that **read** the attachment
        #: header without requiring it (R-01.31: ``load`` and ``instantiate``).
        self.last_headers = headers

        if (
            self.commit_watchdog_ms is not None
            and self.commit_open
            and not self.watchdog_expired
            and self.commit_begun_at is not None
            and (time.monotonic() - self.commit_begun_at) * 1000.0 > self.commit_watchdog_ms
        ):
            self.trip_commit_watchdog()

        if op in self.stalls:
            self.stalls.remove(op)
            raise httpx.ReadTimeout("the fake stalled past the timeout", request=request)

        gate = self._identity_gate(op, headers)
        if gate is not None:
            return gate
        holder = self._holder_gate(op, headers)
        if holder is not None:
            return holder

        forced = self.refuse_always.get(op) or self.refuse_once.pop(op, None)
        if forced is not None:
            status, code = forced
            return _error(code, f"the fake refused {op} with {code}", status)

        route = getattr(self, f"_op_{op.replace('-', '_').replace('/', '_')}", None)
        if route is None:
            return _error("UNKNOWN_OP", f"{op} is not a weights endpoint", 404)
        return route(body)  # type: ignore[no-any-return]

    # ------------------------------------------------------------------ identity (R-01.17)

    #: R-01.17's gated set, as that requirement lists it. ``lineage`` and ``status`` are
    #: **not** in it - they are §4.4 and §4.6 - which is how a restarting trainer can
    #: discover a torn set it did not create before it has any credential in hand.
    GATED = frozenset(
        {
            "attach",
            "detach",
            "heartbeat",
            "quiesce",
            "commit-begin",
            "commit-end",
            "resume",
            "save",
            "load",
            "instantiate",
            "adapter/load",
            "adapter/unload",
            "clear-stale",
        }
    )
    #: R-01.31's holder-only set: **exactly seven**.
    HOLDER_ONLY = frozenset(
        {"detach", "heartbeat", "quiesce", "resume", "commit-begin", "commit-end", "save"}
    )

    def _identity_gate(self, op: str, headers: dict[str, str]) -> httpx.Response | None:
        if op not in self.GATED:
            return None
        authorization = headers.get("authorization", "")
        if authorization != f"Bearer {self.valid_token}":
            # R-01.17. **This is the one that matters for the heartbeat**: a heartbeat
            # refused here is indistinguishable from a dead trainer unless the trainer
            # treats it as an identity failure rather than a transient one.
            return _error("UNAUTHORIZED", f"{op} requires a trainer or operator identity", 401)
        if op == "clear-stale" and self.role != "operator":
            # R-01.29 is OPERATOR-ONLY, which is what makes R-02.34's prohibition protocol
            # rather than policy: miForge's `trainer` identity cannot clear another
            # process's attachment even if its code tried.
            return _error("FORBIDDEN", "clear-stale requires an operator identity", 403)
        return None

    def _holder_gate(self, op: str, headers: dict[str, str]) -> httpx.Response | None:
        """R-01.31: the seven holder-only calls, and their two refusals."""
        if op not in self.HOLDER_ONLY:
            return None
        if self.attachment_id is None:
            return _error("NOT_ATTACHED", f"{op} needs an attachment; there is none", 409)
        if headers.get(ATTACHMENT_HEADER) != self.attachment_id:
            return _error(
                "NOT_ATTACHMENT_HOLDER",
                f"{op} is restricted to the holding attachment (R-01.31)",
                403,
            )
        return None

    # ------------------------------------------------------------------ 4.1 attach/detach

    def _op_attach(self, body: Any) -> httpx.Response:
        """**R-01.1**: a handle per parameter, the resolved name, the manifest hash or null.

        Also the attachment id, which **R-01.31** says is returned once and by no read path
        again, and the producer build identity miLLM's Feature 031 FR-31.1.4 adds - which
        **no requirement in either BRD obliges** (FPRD DEF-F) and F11 checks anyway.
        """
        if self.foreign_holder is not None:
            return _error(
                "ALREADY_ATTACHED",
                "another process holds the attachment (R-01.2)",
                409,
                holder=self.foreign_holder,
                stale=self.stale,
            )
        if self.dtype not in ATTACHABLE_DTYPES:
            return _error(
                "NOT_ATTACHABLE",
                f"the resident layout is {self.dtype}; attach needs plain FP16 or BF16 "
                f"(R-01.5)",
                400,
                dtype=self.dtype,
            )
        self.attachment_id = "attachment-0001"
        self.attached_by = "trainer"
        return httpx.Response(
            200,
            json={
                "attachment_id": self.attachment_id,
                # ⚠ THE FIELD NAMES BELOW ARE miLLM's, READ FROM ITS COMMITTED FTDD 031
                # §5.3 - not restated from BRD-02's prose. That restatement is how this
                # fake came to agree with F11's reader on four names the upstream does not
                # use, and three of the four failed silently against a real response:
                # `build` (so FR-011.7's build check was a no-op), `suggested_interval_
                # seconds` (so FR-011.16's hint silently became a local constant) and a
                # top-level `manifest_sha256` beside a STRING `resident_model` (so the
                # identity cross-check compared a stringified dict and a None). Every F11
                # test passed throughout, because the fake and the reader were wrong
                # together. If you change a name here, change it because miLLM's document
                # says so, and cite the line.
                "model": {
                    "model_id": 7,
                    "name": self.resident_model,
                    # `null` until miLLM's Feature 32 supplies it - their FTDD says so.
                    "manifest_hash": self.manifest_sha256,
                },
                "steps_since_load": self.steps_since_load,
                "weights_state": self.weights_state,
                "heartbeat": {
                    "timeout_s": self.staleness_window_seconds,
                    "interval_hint_s": self.suggested_interval_seconds,
                },
                "device_uuid": "GPU-fake-0000",
                "producer": {"torch": "2.4.1", "cuda": "12.1", "driver": "580.178"},
                "config": {"model_type": "llama", "hidden_size": 8, "num_hidden_layers": 1},
                "parameters": [
                    {
                        "name": "model.embed_tokens.weight",
                        "module_path": "model.embed_tokens",
                        "shape": [16, 8],
                        "dtype": self.dtype,
                        "device": "cuda:0",
                        "storage_key": "storage-a",
                        "handle": {"fake": "handle-a"},
                    },
                    {
                        # TIED to the one above: same storage, and a rebuild must produce
                        # ONE allocation. Two copies of one tensor means the trained model
                        # differs from the served one in exactly this tensor, and every
                        # other assertion in F11 passes.
                        "name": "lm_head.weight",
                        "module_path": "lm_head",
                        "shape": [16, 8],
                        "dtype": self.dtype,
                        "device": "cuda:0",
                        "storage_key": "storage-a",
                        "handle": {"fake": "handle-a"},
                    },
                ],
            },
        )

    def _op_detach(self, body: Any) -> httpx.Response:
        """**R-01.3**: the attachment is released; serving continues on the current values.

        **``torn`` survives this and the open bracket does not** (R-01.23). The state stays
        whatever it is - *"the tensors are still mixed"*, and the next process to attach
        inherits the truth - while ``commit_open`` is cleared, because after a detach no
        process holds an open commit: ``commit-end`` is holder-only and there is no holder.
        That distinction is what makes a recovery ``load`` permitted here (R-01.25 refuses
        one only while **another process** holds an open commit) - and that load is the only
        recovery available when the trainer is dead.
        """
        self.attachment_id = None
        self.attached_by = None
        self.commit_open = False
        return httpx.Response(200, json={"detached": True, "weights_state": self.weights_state})

    def _op_clear_stale(self, body: Any) -> httpx.Response:
        """**R-01.29**, operator-only. Refuses an attachment that is not stale.

        Routed although **F11 never calls it**: a call F11 should not make must reach a
        route that *records* it, so the absence is observable in this fake's ``calls`` as
        well as in the source scan.
        """
        if not self.stale:
            return _error("ATTACHMENT_NOT_STALE", "that attachment is not stale", 409)
        self.foreign_holder = None
        self.stale = False
        return httpx.Response(200, json={"cleared": True})

    def _op_instantiate(self, body: Any) -> httpx.Response:
        """**R-01.26 - DEFERRED upstream.** Routed so a call to it is recorded, never made."""
        return httpx.Response(200, json={"resident_model": "random-init", "manifest_sha256": None})

    # ------------------------------------------------------------------ 4.2 the barrier

    def _op_quiesce(self, body: Any) -> httpx.Response:
        """**R-01.6**: drains and holds. Refused unless the state is ``clean``.

        The response carries the **held-request count and the state** - and *no hold
        duration*, which is FPRD DEF-B: R-02.14 asks for the hold *"as reported by miLLM"*
        and nothing on this surface reports one. F11 measures it locally. A timestamp and a
        deadline are included because miLLM's Feature 031 adds them, and a deadline is a
        limit rather than a hold.
        """
        if self.weights_state != "clean":
            return _error("WEIGHTS_NOT_CLEAN", f"the set is {self.weights_state}", 409)
        self.quiesced = True
        return httpx.Response(
            200,
            json={
                "held_requests": self.held_requests,
                "weights_state": self.weights_state,
                "quiesced_at": "2026-10-14T00:00:00Z",
                "hold_deadline": "2026-10-14T00:00:05Z",
            },
        )

    def _op_resume(self, body: Any) -> httpx.Response:
        """**R-01.7**: releases held requests. Refused unless ``clean``."""
        if self.weights_state != "clean":
            return _error("WEIGHTS_NOT_CLEAN", f"the set is {self.weights_state}", 409)
        self.quiesced = False
        return httpx.Response(200, json={"resumed": True, "weights_state": self.weights_state})

    def _op_commit_begin(self, body: Any) -> httpx.Response:
        """**R-01.21**: accepted only while quiesced and only from the holder.

        It moves the state to ``writing`` and **suspends R-01.8's hold timeout** - which is
        the whole reason the bracket exists, and the reason a write that starts before this
        call can be cut in half by a self-resume.
        """
        if not self.quiesced:
            return _error("WEIGHTS_NOT_CLEAN", "commit-begin needs a quiesced set", 409)
        if self.weights_state != "clean":
            return _error("WEIGHTS_NOT_CLEAN", f"the set is {self.weights_state}", 409)
        self.commit_open = True
        self.commit_begun_at = time.monotonic()
        self.weights_state = "writing"
        return httpx.Response(
            200, json={"weights_state": "writing", "hold_timeout_suspended": True}
        )

    def _op_commit_end(self, body: Any) -> httpx.Response:
        """**R-01.21**, and **R-01.23a** for the late one.

        A ``commit-end`` after the watchdog expired is **accepted** and the overrun is
        recorded: that is one of only two ways out of ``torn``, and the only one available
        while the trainer is alive.
        """
        if not self.commit_open:
            return _error("WEIGHTS_NOT_CLEAN", "no commit is open", 409)
        self.commit_open = False
        if self.watchdog_expired:
            self.commit_overran = True
            self.watchdog_expired = False
        self.weights_state = "clean"
        self.steps_since_load += 1
        return httpx.Response(
            200,
            json={
                "weights_state": "clean",
                "steps_since_load": self.steps_since_load,
                "overran": self.commit_overran,
            },
        )

    # ------------------------------------------------------------------ 4.3/4.4 save, load

    #: The manifest fields **miLLM writes itself** and refuses from a caller. Spelled here
    #: as a literal **on purpose**, unlike production's :func:`training.hotupdate.save
    #: .serving_written_fields`, which derives them from the schema: a fake that derived the
    #: set the same way production does could not disagree with it, and the disagreement is
    #: the whole thing being tested. *A test that reimplements what it tests cannot fail.*
    CALLER_MUST_NOT_SEND = frozenset({"base_model", "created_at", "name", "weights_sha256"})

    #: Whether a successful save renames the resident model to the checkpoint it wrote.
    #: **True by default, because that is what miLLM does** (its own API manual: *"the
    #: resident model is renamed to the checkpoint"*), and because a fake that did not
    #: rename could not express the defect this models - a run whose second save is refused
    #: `weights_identity_changed` for a serving restart that never happened. Settable so a
    #: test can prove the production path is correct under BOTH readings, since **published
    #: BRD-01 does not state the rename** and miForge must not depend on either answer.
    renames_resident_on_save: bool = True

    def _op_save(self, body: Any) -> httpx.Response:
        """**R-01.10 - R-01.13**. Refused unless ``clean`` (R-01.11).

        *"Checkpointing a writing or torn parameter set would persist a model that never
        existed at any step"* - which is why F11 reads the state **locally** before asking,
        rather than relying on this refusal.

        Returns ``201``, and validates the caller's field ownership the way miLLM does
        (``400 MANIFEST_FIELD_NOT_CALLERS``). Both were added 2026-10-14 with F11's save
        orchestrator: before it, nothing sent this endpoint a body at all, so the fake
        accepted any shape and a wrong one would have passed.
        """
        if self.weights_state != "clean":
            return _error(
                "WEIGHTS_NOT_CLEAN", f"save refused: the set is {self.weights_state}", 409
            )
        payload = body if isinstance(body, dict) else {}
        manifest = payload.get("manifest")
        if not isinstance(manifest, dict):
            return _error(
                "INVALID_MANIFEST",
                "the body carries the caller's manifest fields under 'manifest'",
                400,
            )
        if trespassing := sorted(self.CALLER_MUST_NOT_SEND & set(manifest)):
            return _error(
                "MANIFEST_FIELD_NOT_CALLERS",
                f"these fields are written by miLLM, not the caller: {trespassing}",
                400,
            )
        self.saves.append(payload)
        step = int(manifest.get("step", self.steps_since_load))
        run_id = str(manifest.get("run_id", "run"))
        name = f"{self.resident_model}:{run_id}:step-{step}"
        if self.renames_resident_on_save:
            # miLLM renames the resident model to the checkpoint, so `status`'s
            # `resident_model.name` - which is F11's `ResidentIdentity.model` - changes.
            self.resident_model = name
        return httpx.Response(
            201,
            json={
                "name": name,
                "path": f"{run_id}/step-{step:06d}",
                "weights_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
                "latest": f"step-{step:06d}",
                "identity_set": self.renames_resident_on_save,
            },
        )

    def _op_load(self, body: Any) -> httpx.Response:
        """**R-01.25**: verifies the hash before touching a tensor; permitted while ``torn``.

        Matching architecture writes **in place**, so an attachment's handles stay valid -
        which is the case R-02.30(b)'s rollback uses, and the only reason the verification
        is re-run afterwards rather than the attachment being rebuilt.
        """
        self.loads.append(body if isinstance(body, dict) else {})
        if self.commit_open and self.last_headers.get(ATTACHMENT_HEADER) != self.attachment_id:
            # R-01.25 refuses a load *"while ANOTHER PROCESS holds an open commit"* - not
            # while the caller's own commit is open. The holder recovering its own
            # interrupted commit is R-01.23b, the only recovery that works when the trainer
            # is dead, so conflating the two would make this fake refuse the primary
            # recovery path. R-01.31 is why it is distinguishable: ``load`` reads the
            # attachment header when present in order to recognise the holder.
            return _error("WEIGHTS_NOT_CLEAN", "another process holds an open commit", 409)
        self.weights_state = "clean"
        self.watchdog_expired = False
        self.steps_since_load = 0
        if isinstance(body, dict) and body.get("manifest_sha256"):
            self.manifest_sha256 = str(body["manifest_sha256"])
        return httpx.Response(
            200, json={"weights_state": "clean", "in_place": True, "steps_since_load": 0}
        )

    def _op_lineage(self, body: Any) -> httpx.Response:
        """**R-01.16**: the parent chain back to ``base_model``, read from manifests."""
        return httpx.Response(200, json={"chain": [self.resident_model]})

    # ------------------------------------------------------------------ 4.4b adapters

    def _op_adapter_load(self, body: Any) -> httpx.Response:
        """**R-01.27**. Routed, never called by F11: miForge produces adapters, miLLM serves
        them. It runs inside the barrier but **uses no commit bracket** and cannot tear the
        set, because it does not write base tensors."""
        return httpx.Response(200, json={"adapter": "loaded"})

    def _op_adapter_unload(self, body: Any) -> httpx.Response:
        """**R-01.27**, as above."""
        return httpx.Response(200, json={"adapter": "unloaded"})

    # ------------------------------------------------------------------ 4.5/4.6 identity

    def _op_heartbeat(self, body: Any) -> httpx.Response:
        """**R-01.30**: proves the attached process is alive, holder-only (R-01.31).

        Returns **the staleness window and a suggested interval** - the hint FR-011.16 reads
        instead of a local constant. *A fake that omitted them could not test FR-011.16 at
        all.* A heartbeat from a stale-but-uncleared attachment **revives** it.
        """
        if self.stale and self.attachment_id is not None:
            self.stale = False
        return httpx.Response(
            200,
            json={
                # miLLM FTDD 031 §5.2 gives `{attachment_state, timeout_s, weights_state}`
                # and its as-built amendment adds `interval_hint_s`. The two names that
                # were here, `staleness_window_seconds` and `suggested_interval_seconds`,
                # are F11's own and appear in no version of the contract.
                "timeout_s": self.staleness_window_seconds,
                "interval_hint_s": self.suggested_interval_seconds,
                "alive": True,
            },
        )

    def _op_status(self, body: Any) -> httpx.Response:
        """**R-01.20**: the state, the holder, the resident name, steps-since-load.

        *"This endpoint is how a restarting trainer discovers a torn set it did not
        create"*, which is why it is **not** in R-01.17's gated list and why this fake does
        not gate it either.
        """
        return httpx.Response(
            200,
            json={
                "attached": self.attachment_id is not None or self.foreign_holder is not None,
                "attaching_identity": self.foreign_holder or self.attached_by,
                "stale": self.stale,
                "quiesced": self.quiesced,
                "weights_state": self.weights_state,
                # An OBJECT, per miLLM FPRD FR-31.16.1: `resident_model (model_id, name,
                # manifest_hash)`. It was a bare string here, with the hash beside it at
                # the top level, which is not a shape the upstream ever sends.
                "resident_model": {
                    "model_id": 7,
                    "name": self.resident_model,
                    "manifest_hash": self.manifest_sha256,
                },
                "steps_since_load": self.steps_since_load,
                "commit_open": self.commit_open,
                **(
                    {}
                    if self.omit_pause_unavailable
                    else {"pause_unavailable": self.pause_unavailable}
                ),
                "commit_overran": self.commit_overran,
            },
        )

    # ------------------------------------------------------------------ internals

    @staticmethod
    def _body(request: httpx.Request) -> Any:
        raw = request.content
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None
