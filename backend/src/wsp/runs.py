"""WSP runs: preparing a start, storing it, moving it between states (feature 016; FTDD §5.3, §7).

* ``prepare_start`` runs **the same code** for ``POST /wsp/runs/validate``, the approval gate's
  validate hook and ``POST /wsp/runs`` itself (FR-012.11's rule): the provider, the bundle, the
  endpoint row, the caller's queue, then ``wsp.freeze.freeze``.
* ``endpoint_locality`` derives an endpoint's locality from the STORED row, never a body field
  (P52, R1B-M12). It is ``external`` when either the derivation from ``base_url`` or the stored
  column says so: the stricter reading wins.
* ``endpoint_fingerprint`` is the RFC 8785 sha256 of every routing and price field plus the sha256
  of the key ciphertext (R3A-24). The 8-hex display fingerprint is never part of it.
* ``transition`` is the conditional update ``WHERE status = :from`` (the claim pattern of
  ``services/approval_service.py``): ``True`` only for the one caller that moved the row.
* ``run_document`` renders a row as ``miforge.wsp-run/v1``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import hashlib
import re
import secrets
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sqlalchemy as sa
from sqlalchemy.orm import Session

from core.errors import Refusal, refuse
from wsp.endpoints import derive_locality
from wsp.freeze import Frozen, approval_action, freeze
from wsp.models import (
    ACTION_PROTOCOLS,
    DEFAULT_ACTION_PROTOCOL,
    DEFAULT_OUTPUT_CAP,
    DEFAULT_PAGING,
    DEFAULT_THINKING,
    STOP_REASONS,
    TERMINAL_STATUSES,
    WspBundle,
    WspEndpoint,
    WspRun,
)
from wsp.providers import registry
from wsp.result_archive import (
    STORED_MEMBERS,
    _invalid,
    check_record,
    check_transcript,
    json_object,
    read_result_archive,
)
from wsp.settings import WSPSettings, get_wsp_settings, not_configured_message
from wsp.weight_ops.hashing import jcs_sha256

RUN_KIND = "miforge.wsp-run/v1"
#: Statuses a cancel moves to ``cancelled`` (FTDD §7). ``pushing`` is not one: the driver owns a
#: run while it pushes, and moves it out before releasing its lock.
CANCELLABLE = ("queued", "running")


def utcnow() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


def new_run_id() -> str:
    """``wsp_`` and 12 hex digits: 16 characters, the column's width."""
    return f"wsp_{secrets.token_hex(6)}"


# ----------------------------------------------------------------------------- endpoints


def endpoint_locality(row: WspEndpoint) -> str:
    """``external`` when the base URL derives external OR the stored column says so."""
    derived = derive_locality(row.base_url)
    return "external" if "external" in (derived, row.locality) else "local"


def _price(value: Any) -> str | None:
    return None if value is None else format(value, "f")


def endpoint_fingerprint(row: WspEndpoint) -> str:
    """RFC 8785 sha256 of ``{id, kind, base_url, model, locality, serving, output_cap_field,
    price_*}`` plus the sha256 of the key ciphertext (R3A-24; FTDD §5.3).

    Feature 018 (F18-D8): ``action_protocol`` and ``context_window_tokens`` join it ONLY when away
    from their defaults (``tool_calls``, null), so an endpoint at its defaults hashes exactly as it
    did before 018 (no queued run or pending approval fails at the deploy), while the operator's
    PATCH fails a queued or approved run ``wsp_endpoint_changed``. ``thinking`` and
    ``thinking_tokens`` (2026-11-21) follow the same rule (defaults ``off``, null)."""
    ciphertext = row.key_ciphertext
    fields: dict[str, Any] = {
        "id": row.id,
        "kind": row.kind,
        "base_url": row.base_url,
        "model": row.model,
        "locality": row.locality,
        "serving": row.serving,
        "output_cap_field": row.output_cap_field,
        "price_in_per_mtok": _price(row.price_in_per_mtok),
        "price_out_per_mtok": _price(row.price_out_per_mtok),
        "price_cache_write_per_mtok": _price(row.price_cache_write_per_mtok),
        "price_cache_read_per_mtok": _price(row.price_cache_read_per_mtok),
        "key_ciphertext_sha256": (
            None if ciphertext is None else hashlib.sha256(bytes(ciphertext)).hexdigest()
        ),
    }
    protocol = row.action_protocol or DEFAULT_ACTION_PROTOCOL
    if protocol != DEFAULT_ACTION_PROTOCOL:
        fields["action_protocol"] = protocol
    if row.context_window_tokens is not None:
        fields["context_window_tokens"] = int(row.context_window_tokens)
    thinking = row.thinking or DEFAULT_THINKING
    if thinking != DEFAULT_THINKING:
        fields["thinking"] = thinking
    if row.thinking_tokens is not None:
        fields["thinking_tokens"] = int(row.thinking_tokens)
    output_cap = row.output_cap or DEFAULT_OUTPUT_CAP
    if output_cap != DEFAULT_OUTPUT_CAP:
        fields["output_cap"] = output_cap
    paging = row.paging or DEFAULT_PAGING
    if paging != DEFAULT_PAGING:
        fields["paging"] = paging
    return jcs_sha256(fields)


def live_endpoint(session: Session, endpoint_id: Any) -> WspEndpoint | None:
    """The stored, not-removed endpoint row, or ``None``."""
    if not isinstance(endpoint_id, str) or not endpoint_id:
        return None
    row = session.get(WspEndpoint, endpoint_id)
    return None if row is None or row.removed_at is not None else row


# ----------------------------------------------------------------------------- the start


def require_configured(settings: WSPSettings) -> None:
    """R3 review fix (OR3-016-2), 2026-11-16: refuse ``wsp_not_configured`` (503) when
    ``WSP_WORKER_URL`` or ``WSP_GATEWAY_URL`` is unset. Without it a run started on an enabled
    provider was stored ``queued`` and never moved, and nothing said why; now the start (and its
    validate, the approval gate's hook, a replay or a rerun) refuses and names the setting."""
    missing = settings.missing_urls()
    if missing:
        raise refuse("wsp_not_configured", not_configured_message(missing), missing=missing)


@dataclass(frozen=True)
class Prepared:
    frozen: Frozen
    bundle: WspBundle
    endpoint: WspEndpoint
    provider_id: str
    provider_version: str
    caller: str

    @property
    def fingerprint(self) -> str:
        return endpoint_fingerprint(self.endpoint)


def prepare_start(
    session: Session,
    body: dict[str, Any],
    caller: str,
    settings: WSPSettings | None = None,
) -> Prepared:
    """Validate and freeze a start body. Refuses with the route's own codes; writes nothing."""
    settings = settings or get_wsp_settings()
    if not isinstance(body, dict):
        raise refuse("validation_failed", "The start body is a JSON object.")
    provider = body.get("provider")
    if provider is not None and not isinstance(provider, dict):
        raise refuse("validation_failed", 'provider is an object: {"id": <provider id>}.')
    provider_id = (provider or {}).get("id") or settings.wsp_default_provider
    if not isinstance(provider_id, str):
        raise refuse("validation_failed", "provider.id is a string.")
    descriptor = registry.get(provider_id, settings)
    require_configured(settings)
    registry.require_operation(descriptor, str(body.get("kind")), str(body.get("mode")))
    ref = body.get("bundle") or {}
    if not isinstance(ref, dict):
        raise refuse("validation_failed", "bundle is an object: {owner, bundle_id, bundle_sha256}.")
    bundle = session.scalar(
        sa.select(WspBundle).where(
            WspBundle.owner == ref.get("owner"), WspBundle.bundle_id == ref.get("bundle_id")
        )
    )
    if bundle is None:
        raise refuse("wsp_bundle_not_found", "No such bundle; import it first.")
    row = live_endpoint(session, body.get("endpoint_id"))
    # ``kind``, ``serving`` and the thinking fields are for ``freeze``'s resolution of the
    # thinking in effect (2026-11-21); the frozen ``request.endpoint`` keeps only id, locality
    # and model.
    endpoint = (
        None
        if row is None
        else {
            "id": row.id,
            "locality": endpoint_locality(row),
            "model": row.model,
            "kind": row.kind,
            "serving": row.serving,
            "thinking": row.thinking or DEFAULT_THINKING,
            "thinking_tokens": row.thinking_tokens,
            "output_cap": row.output_cap or DEFAULT_OUTPUT_CAP,
            "paging": row.paging or DEFAULT_PAGING,
        }
    )
    queued = session.scalar(
        sa.select(sa.func.count())
        .select_from(WspRun)
        .where(WspRun.caller == caller, WspRun.status == "queued")
    )
    frozen = freeze(
        body,
        bundle=bundle.manifest,
        bundle_imported=bundle.import_status == "imported",
        endpoint=endpoint,
        provider_version=descriptor.version,
        queued_for_caller=int(queued or 0),
        settings=settings,
    )
    assert row is not None  # freeze refuses wsp_endpoint_unknown without one
    return Prepared(
        frozen=frozen,
        bundle=bundle,
        endpoint=row,
        provider_id=descriptor.id,
        provider_version=descriptor.version,
        caller=caller,
    )


def create_run(session: Session, prepared: Prepared, *, approval_id: str | None) -> WspRun:
    """Store a ``queued`` session run (FR-016.19). The driver pushes it; nothing runs here."""
    frozen = prepared.frozen
    row = WspRun(
        run_id=new_run_id(),
        operation=frozen.request["kind"],
        style=frozen.request["mode"],
        status="queued",
        request=frozen.request,
        request_sha256=frozen.request_sha256,
        bundle_sha256=prepared.bundle.bundle_sha256,
        endpoint_id=prepared.endpoint.id,
        endpoint_locality=frozen.locality,
        sensitivity_effective=frozen.sensitivity_effective,
        provider_id=prepared.provider_id,
        provider_version=prepared.provider_version,
        approval_id=approval_id,
        endpoint_fingerprint=prepared.fingerprint,
        caller=prepared.caller,
        caller_ref=frozen.caller_ref,
        usage={},
        created_at=utcnow(),
    )
    session.add(row)
    session.flush()
    return row


#: Styles whose runs reach the gateway: a token is minted for these only (R3A-13). A replay and a
#: program rerun get neither a token nor a gateway in their push; a CHECK and the gateway's own
#: style predicate guard the same rule below this one (C21c, C21d, C23a).
GATEWAY_STYLES = ("session", "program")
REPLAYABLE_STYLES = ("session", "program")
SUPPORT_KINDS = ("practice", "practice_train")


def takes_gateway(style: str) -> bool:
    return style in GATEWAY_STYLES


def _queued_for(session: Session, caller: str, settings: WSPSettings) -> None:
    """R1B-M23: replays and reruns count toward the caller's queue like any run."""
    queued = session.scalar(
        sa.select(sa.func.count())
        .select_from(WspRun)
        .where(WspRun.caller == caller, WspRun.status == "queued")
    )
    if int(queued or 0) >= settings.wsp_max_queued_per_caller:
        raise refuse(
            "wsp_queue_limit_reached",
            f"This caller already has {int(queued or 0)} queued WSP runs (the limit is "
            f"{settings.wsp_max_queued_per_caller}). Wait for one of them to start, then file "
            "this request again: the limit is checked when an approved request executes too, and "
            "an approval refused here is not retried.",
        )


def _derived_run(
    session: Session,
    source: WspRun,
    *,
    style: str,
    request: dict[str, Any],
    caller: str,
    **values: Any,
) -> WspRun:
    from wsp.schemas import validate_frozen_request

    validate_frozen_request(request)
    row = WspRun(
        run_id=new_run_id(),
        operation=source.operation,
        style=style,
        status="queued",
        request=request,
        request_sha256=jcs_sha256(request),
        bundle_sha256=source.bundle_sha256,
        endpoint_id=None,
        endpoint_locality=None,
        sensitivity_effective=source.sensitivity_effective,
        provider_id=source.provider_id,
        provider_version=source.provider_version,
        approval_id=None,
        endpoint_fingerprint=None,
        caller=caller,
        caller_ref=source.caller_ref,
        usage={},
        created_at=utcnow(),
        **values,
    )
    session.add(row)
    session.flush()
    return row


def create_replay(
    session: Session, of_run_id: str, caller: str, settings: WSPSettings | None = None
) -> WspRun:
    """Queue a replay of a completed session or program run (FR-016.21; ungated, PF-13).

    The frozen request is the original's with ``mode: replay``, no endpoint and ``of_run_id``; the
    driver pushes the original transcript as a file (checked against ``transcript_sha256``) and no
    gateway, and mints no token."""
    settings = settings or get_wsp_settings()
    require_configured(settings)
    original = get_run(session, of_run_id)
    if (
        original.style not in REPLAYABLE_STYLES
        or original.status != "completed"
        or original.transcript_sha256 is None
    ):
        raise refuse(
            "validation_failed",
            "Only a completed session or program run with a transcript can be replayed.",
            run_id=of_run_id,
            status=original.status,
            style=original.style,
        )
    _queued_for(session, caller, settings)
    request = {**dict(original.request), "mode": "replay", "endpoint": None}
    request["of_run_id"] = original.run_id
    return _derived_run(
        session, original, style="replay", request=request, caller=caller, replay_of=original.run_id
    )


def create_rerun(
    session: Session,
    program_sha256: str,
    support_kind: str,
    caller: str,
    settings: WSPSettings | None = None,
) -> WspRun:
    """Queue a rerun of a stored program on a named support set (FR-016.23; P21; ungated, PF-13).

    It rebuilds from the base: no LLM, no endpoint, no token, ``gateway: null``."""
    from wsp.models import WspProgram

    settings = settings or get_wsp_settings()
    require_configured(settings)
    program = session.get(WspProgram, program_sha256)
    if program is None:
        raise refuse("wsp_program_not_found", "No stored program with that sha256.")
    if support_kind not in SUPPORT_KINDS:
        raise refuse("validation_failed", f"support_set.kind is one of {list(SUPPORT_KINDS)}.")
    first = get_run(session, program.first_run_id)
    _queued_for(session, caller, settings)
    request = {**dict(first.request), "mode": "program_rerun", "endpoint": None}
    request["program_sha256"] = program_sha256
    request["support_set"] = {
        "kind": support_kind,
        "digest": jcs_sha256({"bundle_sha256": first.bundle_sha256, "kind": support_kind}),
    }
    return _derived_run(
        session,
        first,
        style="program_rerun",
        request=request,
        caller=caller,
        program_sha256=program_sha256,
    )


def program_reruns(session: Session, program_sha256: str | None) -> list[str]:
    """Every ``program_rerun`` of this program, oldest first (P19)."""
    if program_sha256 is None:
        return []
    rows = session.scalars(
        sa.select(WspRun.run_id)
        .where(WspRun.style == "program_rerun", WspRun.program_sha256 == program_sha256)
        .order_by(WspRun.created_at, WspRun.run_id)
    ).all()
    return list(rows)


def transition(
    session: Session, run_id: str, frm: str | tuple[str, ...], to: str, **values: Any
) -> bool:
    """Move ``run_id`` from ``frm`` to ``to`` with one conditional UPDATE; ``True`` if it moved."""
    sources = (frm,) if isinstance(frm, str) else tuple(frm)
    if to in TERMINAL_STATUSES:
        values.setdefault("finished_at", utcnow())
    result = session.execute(
        sa.update(WspRun)
        .where(WspRun.run_id == run_id, WspRun.status.in_(sources))
        .values(status=to, **values)
        .execution_options(synchronize_session=False)
    )
    moved = int(getattr(result, "rowcount", 0) or 0) == 1
    if moved and to in TERMINAL_STATUSES:
        # FTDD §7: on every terminal transition, open reservations settle at worst case.
        from wsp.gateway_ledger import close_open_at_worst

        close_open_at_worst(session, run_id)
        store_metered_usage(session, run_id)
    return moved


def store_metered_usage(session: Session, run_id: str) -> None:
    """The run's ``usage`` from the gateway's records, on EVERY terminal transition (fix
    2026-11-21): a run failed by the driver (``time_budget``, ``worker_lost``), cancelled or
    refused at intake kept the ``{}`` it was created with, so it reported ``requests: 0`` while
    the gateway had metered every call it served (run ``wsp_cb092cedad97``: 18)."""
    current = session.scalar(sa.select(WspRun.usage).where(WspRun.run_id == run_id))
    usage = {**dict(current or {}), **metered_usage(session, run_id)}
    session.execute(
        sa.update(WspRun)
        .where(WspRun.run_id == run_id)
        .values(usage=usage)
        .execution_options(synchronize_session=False)
    )


def get_run(session: Session, run_id: str) -> WspRun:
    row = session.get(WspRun, run_id)
    if row is None or row.deleted_at is not None:
        raise refuse("wsp_run_not_found", f"No WSP run with id {run_id!r}.", run_id=run_id)
    return row


def cancel(session: Session, run_id: str) -> WspRun:
    """``queued`` or ``running`` → ``cancelled`` (FTDD §7). A running run's worker is left to the
    driver, which treats its boot as an orphan: never pushed again, its result discarded unread
    and acked once ``finished`` (the supervisor has no kill route, by design)."""
    row = get_run(session, run_id)
    if row.status not in CANCELLABLE or not transition(
        session, run_id, CANCELLABLE, "cancelled", stop_reason="cancelled"
    ):
        session.refresh(row)
        raise refuse(
            "wsp_run_not_cancellable",
            f"A run that is {row.status} cannot be cancelled.",
            run_id=run_id,
            status=row.status,
        )
    session.refresh(row)
    return row


# ----------------------------------------------------------------------------- the document


def _iso(value: dt.datetime | None) -> str | None:
    if value is None:
        return None
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.UTC)
    return value.astimezone(dt.UTC).isoformat().replace("+00:00", "Z")


def run_document(
    row: WspRun,
    bundle_manifest: dict[str, Any] | None = None,
    *,
    reruns: list[str] | None = None,
    candidate: Any = None,
    chain: list[str] | None = None,
    verdict: dict[str, Any] | None = None,
    run_sha256: str | None = None,
) -> dict[str, Any]:
    """``miforge.wsp-run/v1`` for one row (FTDD §3.3). ``candidate`` is the run's
    ``WspCandidate`` row (its lineage parent and id), ``chain`` the patch sha256s from the bundle
    to it, ``verdict`` its stored verdict document and ``run_sha256`` that verdict's
    (``wsp.evaluate.document_of`` gathers them; FR-016.33a)."""
    request = dict(row.request)
    sensitivity = (bundle_manifest or {}).get("sensitivity") or {}
    usage = dict(row.usage or {})
    parent = request.get("parent") or {"kind": "bundle", "ref": row.bundle_sha256}
    if candidate is not None:
        parent = {"kind": candidate.parent_kind, "ref": candidate.parent_ref}
    return {
        "kind": RUN_KIND,
        "run_id": row.run_id,
        "status": row.status,
        "created_at": _iso(row.created_at),
        "request": request,
        "request_sha256": row.request_sha256,
        "caller_ref": row.caller_ref,
        "approval": (
            None
            if row.approval_id is None
            else {"action": _approval_action(row), "approval_id": row.approval_id}
        ),
        "boot_id": row.boot_id,
        "sensitivity": {
            "value": request.get("sensitivity_declared", "unknown"),
            "effective": row.sensitivity_effective,
            "source": sensitivity.get("source", "none"),
        },
        "usage": {
            "metered_by": "wsp_gateway",
            "requests": int(usage.get("requests", 0)),
            "input_tokens": int(usage.get("input_tokens", 0)),
            "output_tokens": int(usage.get("output_tokens", 0)),
            "cost_usd": float(usage.get("cost_usd", 0.0)),
            "self_reported": usage.get("self_reported"),
        },
        "forward_passes_used": row.forward_passes_used,
        "stop_reason": row.stop_reason,
        "outcome": row.outcome,
        "failure": row.failure,
        "content_sha256": row.content_sha256,
        "transcript_sha256": row.transcript_sha256,
        "result_sha256": row.result_sha256,
        "patch_sha256": row.patch_sha256,
        "program_sha256": row.program_sha256,
        "program_reruns": list(reruns or []),
        "replay": (
            None
            if row.style != "replay"
            else {
                "of_run_id": row.replay_of,
                "reproduced": bool(row.replay_reproduced),
                "patch_sha256": row.replay_patch_sha256,
                "content_sha256": row.content_sha256,
            }
        ),
        "candidate_id": None if candidate is None else candidate.candidate_id,
        "verdict": verdict,
        "run_sha256": run_sha256,
        "lineage": {
            "bundle_sha256": row.bundle_sha256,
            "chain": list(chain or []),
            "parent": {"kind": parent["kind"], "ref": parent["ref"]},
            "provider": row.provider_id,
            "provider_version": row.provider_version,
        },
    }


def _approval_action(row: WspRun) -> str:
    """The action this run's approval was decided under: the one the frozen request implies."""
    return approval_action(row.operation, row.endpoint_locality or "local") or "wsp_session"


def run_dir(settings: WSPSettings, run_id: str) -> Path:
    """Where the intake stores one run's result: ``<WSP_BUNDLE_ROOT>/runs/<run_id>/``."""
    if not run_id.startswith("wsp_") or not run_id[4:].isalnum():
        raise refuse("wsp_run_not_found", f"No WSP run with id {run_id!r}.", run_id=run_id)
    return Path(settings.wsp_bundle_root) / "runs" / run_id


# ----------------------------------------------------------------------------- the intake
@dataclass(frozen=True)
class Intake:
    status: str
    stop_reason: str | None
    outcome: str | None
    failure: dict[str, Any] | None
    transcript_sha256: str | None
    result_sha256: str
    patch_sha256: str | None = None
    content_sha256: str | None = None
    program_sha256: str | None = None
    replay_reproduced: bool | None = None
    replay_patch_sha256: str | None = None
    #: Feature 018 (FR-018.11a): ``result.json``'s ``forward_used`` and ``engine`` block.
    counters: dict[str, Any] = dataclasses.field(default_factory=dict)

    def columns(self) -> dict[str, Any]:
        """The run columns this intake sets on its terminal transition."""
        values: dict[str, Any] = {
            **self.counters,
            "stop_reason": self.stop_reason,
            "outcome": self.outcome,
            "failure": self.failure,
            "transcript_sha256": self.transcript_sha256,
            "patch_sha256": self.patch_sha256,
            "content_sha256": self.content_sha256,
            "replay_reproduced": self.replay_reproduced,
            "replay_patch_sha256": self.replay_patch_sha256,
        }
        if self.program_sha256 is not None:
            values["program_sha256"] = self.program_sha256
        return values


#: The ``engine`` block ``miwsp_engine`` writes in ``result.json``: EXACTLY these four keys (018-P1
#: deviation D-4: ``kernel_restarts`` is the fourth). A provider that writes no block (weightspace,
#: a program rerun) leaves the four columns null.
ENGINE_KEYS = ("action_protocol", "handoffs", "refused_turns", "kernel_restarts")
ENGINE_COUNTERS = ("handoffs", "refused_turns", "kernel_restarts")
#: Keys the block MAY carry beside those four (2026-11-21): ``reasoning_turns``, the replies that
#: carried reasoning. Optional, so an engine from before it still writes a valid block.
ENGINE_OPTIONAL = (
    "reasoning_turns",
    "repeated_turns",
    # Paging (2026-11-21, migration 0034).
    "pages_out",
    "recalls",
    "ledger_rows",
    "peak_prompt_tokens",
)
#: An INTEGER column's ceiling: a counter above it is malformed, never truncated.
MAX_COUNTER = 2**31 - 1


def _counter(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= MAX_COUNTER:
        raise _invalid(name, f"{name} is not a non-negative integer")
    return value


def result_counters(result: dict[str, Any]) -> dict[str, Any]:
    """The run columns from ``result.json`` (FR-018.11a): ``forward_used`` (both providers) into
    ``forward_passes_used``, and the ``engine`` block into its four columns. Absent means null; a
    present value of any other shape refuses the result ``wsp_result_invalid`` (FTASKS 8.6),
    naming the check, so a child cannot write a counter the run row cannot hold."""
    counters: dict[str, Any] = {
        "forward_passes_used": None,
        **dict.fromkeys(ENGINE_KEYS),
        **dict.fromkeys(ENGINE_OPTIONAL),
    }
    if result.get("forward_used") is not None:
        counters["forward_passes_used"] = _counter(result["forward_used"], "forward_used")
    if "engine" not in result:
        return counters
    engine = result["engine"]
    if (
        not isinstance(engine, dict)
        or not set(ENGINE_KEYS) <= set(engine)
        or not set(engine) <= set(ENGINE_KEYS) | set(ENGINE_OPTIONAL)
    ):
        raise _invalid(
            "engine",
            f"the engine block is an object with the keys {', '.join(ENGINE_KEYS)} and "
            f"optionally {', '.join(ENGINE_OPTIONAL)}",
        )
    if engine["action_protocol"] not in ACTION_PROTOCOLS:
        raise _invalid("engine", "engine.action_protocol is not tool_calls or structured")
    counters["action_protocol"] = engine["action_protocol"]
    for name in ENGINE_COUNTERS:
        counters[name] = _counter(engine[name], f"engine.{name}")
    for name in ENGINE_OPTIONAL:
        if engine.get(name) is not None:
            counters[name] = _counter(engine[name], f"engine.{name}")
    return counters


#: Failure codes a worker's ``result.json`` may name to FAIL its own run (program mode,
#: FR-016.23), recorded with that code as given. Any other ``failure`` (or ``weightspace``'s
#: ``error`` text) is recorded as the run's cause (:func:`reported_failure`, a plain code or
#: :data:`UNNAMED_CAUSE`), and the stop decides the status (:data:`BUDGET_STOPS`).
WORKER_FAILURE_CODES = ("wsp_program_invalid", "wsp_program_nondeterministic")
#: Fix 2026-11-21 (live run ``wsp_c3e954d3f21f``): a session that stopped with a cause is a
#: FAILED run, whatever code the cause names (an upstream 400 read ``completed / no_change``),
#: except a stop at a budget: the budget was spent, the session ended as its budget says, and its
#: partial result is kept (FR-018.11), so it completes. A cause can only fail a run, never
#: promote one: a failed run has no candidate.
BUDGET_STOPS = ("token_budget", "cost_budget", "time_budget")
#: The coordinator's ruling (2026-11-21): a gateway refusal that ran the editor out of context is
#: budget-like, so the run completes and keeps its partial result (a valid patch is a candidate);
#: every other gateway refusal (a field refused, auth, an upstream 4xx) fails the run.
BUDGET_LIKE_CAUSES = ("editor_context_exceeded",)

#: The gateway's own refusal codes (``wsp_gateway.app.STATUS``; a test pins the two equal). A
#: session that stopped ``error`` on one of them was refused by the gateway, so its stop is
#: ``gateway_refused`` (FR-016's AC), or the budget it names (:func:`labelled_stop`).
GATEWAY_CODES = frozenset(
    {
        "wsp_gateway_token_invalid",
        "wsp_session_token_invalid",
        "wsp_session_not_active",
        "wsp_sensitive_data_external_endpoint",
        "wsp_endpoint_url_refused",
        "wsp_endpoint_key_binding_failed",
        "wsp_request_field_refused",
        "wsp_streaming_unsupported",
        "wsp_budget_exhausted",
        "wsp_daily_spend_cap_reached",
        "wsp_request_in_flight",
        "wsp_reservation_closed",
        "wsp_editor_is_edit_target",
        "wsp_gateway_ledger_unavailable",
        "wsp_gateway_upstream_failed",
        "editor_steered",
        "editor_context_exceeded",
        "editor_model_not_resident",
        "editor_model_loading",
    }
)
#: A cause whose code is not a plain code (or that names none) is recorded under this one.
UNNAMED_CAUSE = "worker_session_failed"
_CODE = re.compile(r"[a-z][a-z0-9_]{0,63}")
_WORDS = re.compile(r"\b[a-z][a-z0-9]*(?:_[a-z0-9]+)+\b")


def reported_failure(result: dict[str, Any], stop_reason: str | None) -> dict[str, str] | None:
    """The cause a worker reported for its stop, as ``{code, message}`` (fix 2026-11-20).

    - ``miwsp_engine`` writes ``failure: {code, message}``: kept, its code if it is a plain code
      (``[a-z][a-z0-9_]{0,63}``), else :data:`UNNAMED_CAUSE`;
    - ``weightspace`` writes ``error: "<Type>: <text>"``: the message is that text, and the code is
      the first gateway code it names (the gateway's messages end ``(code: …)``), else
      :data:`UNNAMED_CAUSE`;
    - a stop of ``error`` with neither still gets a cause, saying none was reported.

    Untrusted data, like the rest of ``result.json``: bounded (64, 2000), shown as text."""
    failure = result.get("failure")
    error = result.get("error")
    if isinstance(failure, dict) and isinstance(failure.get("code"), str):
        code = str(failure["code"])
        message = str(failure.get("message", ""))
        if _CODE.fullmatch(code) is None:
            code, message = UNNAMED_CAUSE, f"{code[:64]}: {message}"
        return {"code": code, "message": message[:2000]}
    if isinstance(error, str) and error.strip():
        named = next((w for w in _WORDS.findall(error) if w in GATEWAY_CODES), None)
        return {"code": named or UNNAMED_CAUSE, "message": error[:2000]}
    if stop_reason == "error":
        return {
            "code": UNNAMED_CAUSE,
            "message": "The session stopped on an error and its result names no cause.",
        }
    return None


def labelled_stop(stop_reason: str, cause: dict[str, str] | None) -> str:
    """An ``error`` stop the gateway caused, labelled as what it was: the budget it spent
    (``token_budget`` or ``cost_budget``, from ``wsp_budget_exhausted``'s message), else
    ``gateway_refused``. Every other stop is kept."""
    if stop_reason != "error" or cause is None or cause["code"] not in GATEWAY_CODES:
        return stop_reason
    if cause["code"] == "wsp_budget_exhausted":
        text = cause["message"].lower()
        if "cost cap" in text:
            return "cost_budget"
        if "token budget" in text:
            return "token_budget"
    return "gateway_refused"


def intake_result(session: Session, run: WspRun, data: bytes, settings: WSPSettings) -> Intake:
    """Check the archive, store the transcript and logs as data, meter the usage, and record the
    run's terminal state. Raises ``wsp_result_invalid`` (naming the check) on any failure."""
    result_sha256 = hashlib.sha256(data).hexdigest()
    members = read_result_archive(data)
    record = check_record(members, run.run_id, run.boot_id)
    deadline_hit = bool(record.get("deadline_hit"))
    exit_code = record.get("exit_code")
    # Review OR2-016-3: a child killed at its deadline or ending abnormally may leave its files
    # half-written (RLIMIT_FSIZE, a full /out, a crash). Such a run fails by the supervisor's
    # record, never with a candidate, so its result.json and patch are not read, and its
    # transcript and logs are stored as written (the evidence of what the editor did) without
    # parsing; the transcript route serves a line that is not a JSON object as ``{"raw": ...}``.
    abnormal = deadline_hit or exit_code != 0
    if abnormal:
        # R3 review fix (OR3-016-5), 2026-11-16: an abnormal run's stop reason comes from the
        # supervisor's record only (time_budget or error below), never from what the child
        # wrote, so a killed run never reads "submitted". Its result.json is stored, not read.
        result: dict[str, Any] = {}
        stop_reason = None
    else:
        result = (
            json_object(members["result.json"], "result_json") if "result.json" in members else {}
        )
        stop_reason = result.get("stop_reason")
        if stop_reason is not None and stop_reason not in STOP_REASONS:
            raise _invalid("stop_reason", "the stop reason is not one this run can end with")
        if "transcript.jsonl" in members:
            check_transcript(members["transcript.jsonl"])
    # FR-018.11a / FTASKS 8.6: read before anything is verified or stored, so a malformed
    # counter refuses the result with nothing written. An abnormal run's result is not read.
    counters = result_counters(result)
    patch = None
    has_patch = "patch/patch.json" in members or "patch/edit.npz" in members
    if has_patch and abnormal and run.request.get("kind") == "inspect":
        raise refuse(
            "wsp_inspect_changed_weights", "An inspect session changed weights; it is refused."
        )
    verified: Verified | None = None
    if has_patch and not abnormal:
        verified = verify_result_patch(session, run, members, settings)
        patch = verified.manifest
    directory = run_dir(settings, run.run_id)
    directory.mkdir(parents=True, exist_ok=True)
    for name in STORED_MEMBERS:
        if name in members:
            target = directory / name.replace("/", "_") if "/" in name else directory / name
            target.write_bytes(members[name])
    transcript = members.get("transcript.jsonl")
    transcript_sha = None if transcript is None else hashlib.sha256(transcript).hexdigest()
    reported = result.get("failure") if isinstance(result.get("failure"), dict) else None
    if deadline_hit:
        status, stop_reason, outcome = "failed", "time_budget", "failed"
        failure: dict[str, Any] | None = {
            "code": "time_budget",
            "message": "The session ran past its time budget.",
        }
    elif exit_code != 0:
        status, outcome = "failed", "failed"
        stop_reason = stop_reason or "error"
        failure = {
            "code": "worker_session_failed",
            "message": f"The session exited with code {exit_code}.",
        }
    elif reported is not None and reported.get("code") in WORKER_FAILURE_CODES:
        status, outcome, stop_reason = "failed", "failed", "error"
        failure = {
            "code": str(reported["code"]),
            "message": str(reported.get("message", ""))[:2000],
        }
    else:
        # FR-016.30: an edit run (create, repair; a session, a program or its rerun) that left
        # no patch changed nothing, so it is ``no_change``; ``report`` is an inspect's (and a
        # replay's, which is never a candidate).
        edits = run.request.get("kind") in ("create", "repair") and run.style != "replay"
        status, outcome = "completed", "no_change" if edits else "report"
        # Fix 2026-11-20: the cause a worker reports is the run's, so the run API shows why it
        # stopped (it lived only in result.json); a gateway's refusal is labelled as one.
        failure = reported_failure(result, stop_reason)
        stop_reason = labelled_stop(stop_reason or "answered", failure)
        budget_like = failure is not None and failure["code"] in BUDGET_LIKE_CAUSES
        if failure is not None and stop_reason not in BUDGET_STOPS and not budget_like:
            status, outcome = "failed", "failed"
    done = Intake(
        status,
        stop_reason,
        outcome,
        failure,
        transcript_sha,
        result_sha256,
        counters=counters,
    )
    if verified is not None and patch is not None and status == "completed":
        if run.style == "replay":
            # FR-016.21: never a candidate; reproduced compares content_sha256 (R1B-A1).
            done = _replaced(
                done, content=patch["content_sha256"], replay_patch=patch["patch_sha256"]
            )
        elif qualify(run, verified, settings) is not None:
            # FR-016.30 (P22, PF-20): no candidate row; the patch stays beside the run as evidence.
            done = _replaced(done, outcome="no_change", content=patch["content_sha256"])
        else:
            sha = record_candidate(session, run, verified, members, directory)
            done = _replaced(done, outcome="candidate", patch=sha, content=patch["content_sha256"])
    if run.style == "program" and status == "completed" and "program.py" in members:
        stored = store_program(session, run, members["program.py"], directory, settings)
        done = _replaced(done, program=stored)
    if run.style == "replay":
        done = _replaced(done, reproduced=status == "completed" and _reproduces(session, run, done))
    return done


def _replaced(done: Intake, **changes: Any) -> Intake:
    names = {
        "outcome": "outcome",
        "patch": "patch_sha256",
        "content": "content_sha256",
        "program": "program_sha256",
        "reproduced": "replay_reproduced",
        "replay_patch": "replay_patch_sha256",
    }
    return dataclasses.replace(done, **{names[k]: v for k, v in changes.items()})


def _reproduces(session: Session, run: WspRun, done: Intake) -> bool:
    """``content_sha256`` equal to the original's (both null: neither changed a weight)."""
    original = session.get(WspRun, run.replay_of) if run.replay_of else None
    return original is not None and original.content_sha256 == done.content_sha256


def store_program(
    session: Session, run: WspRun, source: bytes, directory: Path, settings: WSPSettings
) -> str:
    """The submitted program, keyed by its sha256 (FR-016.23): stored beside the run and in
    ``mf_wsp_programs`` (the first run that submitted it), never executed here."""
    from wsp.models import WspProgram

    sha = hashlib.sha256(source).hexdigest()
    (directory / "program.py").write_bytes(source)
    target = program_path(settings, sha)
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        target.write_bytes(source)
    if session.get(WspProgram, sha) is None:
        session.add(
            WspProgram(
                program_sha256=sha,
                first_run_id=run.run_id,
                bytes=len(source),
                created_at=dt.datetime.now(dt.UTC),
            )
        )
        session.flush()
    return sha


def program_path(settings: WSPSettings, program_sha256: str) -> Path:
    """``<WSP_BUNDLE_ROOT>/programs/<sha256>.py``: what a rerun pushes."""
    if len(program_sha256) != 64 or not all(c in "0123456789abcdef" for c in program_sha256):
        raise refuse("wsp_program_not_found", "No stored program with that sha256.")
    return Path(settings.wsp_bundle_root) / "programs" / f"{program_sha256}.py"


def refuse_non_editable(bundle_manifest: dict[str, Any], patch_manifest: dict[str, Any]) -> None:
    """Refuse a patch that changes a tensor the bundle does not mark ``editable`` (review
    OR1-016-2; FTDD §3.1 ``TensorEntry.editable``).

    The worker reads the flag too, but the worker is untrusted (FR-016.19), and program mode runs
    LLM-written code there, so the backend enforces it wherever a patch is verified: the intake,
    ``POST /wsp/patches/verify`` and ``/stack``. A tensor the bundle does not list at all is not
    editable either. "Changed" is ``check_limits``' reading: changed entries, or an append.
    """
    editable = {t["name"] for t in bundle_manifest.get("tensors", []) if t.get("editable") is True}
    refused = [
        t["name"]
        for t in patch_manifest.get("tensors", [])
        if (int(t.get("changed_entries", 0)) != 0 or t.get("append") is not None)
        and t["name"] not in editable
    ]
    if refused:
        raise refuse(
            "wsp_change_limit_exceeded",
            "; ".join(f"tensor {n!r} is not editable in the bundle" for n in refused),
            kind="frozen_tensor",
            tensors=refused,
        )


@dataclass(frozen=True)
class Verified:
    """A worker's patch, re-applied to the stored parent: its manifest, the bundle it belongs to,
    the parent's tensors and the rebuilt ones."""

    manifest: dict[str, Any]
    bundle: WspBundle
    base: dict[str, Any]
    edited: dict[str, Any]


def verify_result_patch(
    session: Session, run: WspRun, members: dict[str, bytes], settings: WSPSettings
) -> Verified:
    """Re-apply the worker's patch to the STORED parent tensors (FR-016.19; P63, R2A-15).

    Nothing the worker says about its patch is trusted: ``apply_patch`` re-checks every base hash,
    the archive's digest, every rebuilt tensor's hash and every changed-entry count (C19a); the
    parent is the frozen request's, never the patch's own claim (C19d); the change limits are
    checked again here. An inspect that changed weights is refused (FR-016.32).
    """
    import tempfile

    from wsp.bundles import store_dir
    from wsp.weight_ops.limits import check_limits
    from wsp.weight_ops.patch import ARRAYS_FILE, PATCH_JSON, apply_patch
    from wsp.weight_ops.st_io import read_safetensors

    if run.request.get("kind") == "inspect":
        raise refuse(
            "wsp_inspect_changed_weights", "An inspect session changed weights; it is refused."
        )
    if "patch/patch.json" not in members or "patch/edit.npz" not in members:
        raise _invalid("patch_incomplete", "a patch needs both patch.json and edit.npz")
    bundle = session.get(WspBundle, run.bundle_sha256)
    if bundle is None or bundle.import_status != "imported":
        raise _invalid("patch_unverified", "the run's bundle is not stored")
    heads = next(f["name"] for f in bundle.manifest["files"] if f["role"] == "heads")
    tensors = read_safetensors(store_dir(settings, bundle.bundle_sha256) / heads)
    with tempfile.TemporaryDirectory(prefix="wsp-intake-") as tmp:
        directory = Path(tmp)
        (directory / PATCH_JSON).write_bytes(members["patch/patch.json"])
        (directory / ARRAYS_FILE).write_bytes(members["patch/edit.npz"])
        try:
            manifest, rebuilt = apply_patch(directory, tensors)
        except Refusal as refusal:
            raise _invalid(
                "patch_unverified",
                f"the patch does not re-apply: {refusal.code}: {refusal.message[:300]}",
            ) from refusal
    parent = run.request.get("parent") or {}
    if (
        manifest["bundle_base_sha256"] != run.bundle_sha256
        or manifest["parent"].get("kind") != parent.get("kind", "bundle")
        or manifest["parent"].get("ref") != parent.get("ref", run.bundle_sha256)
    ):
        raise _invalid("patch_parent", "the patch names another parent than the frozen request")
    refuse_non_editable(bundle.manifest, manifest)
    violations = check_limits(manifest, run.request.get("change_limits") or {})
    if violations:
        raise refuse(
            "wsp_change_limit_exceeded",
            "; ".join(v.detail for v in violations),
        )
    return Verified(manifest, bundle, tensors, rebuilt)


def new_candidate_id() -> str:
    """``wspc_`` and 12 hex digits, the run schema's pattern (17 characters, the column's width).
    Review 016-P2 (2026-11-19): phase 1 minted ``wsc_`` + 13 hex, which the run schema refuses
    and which is miWorkbench's score-call prefix."""
    return f"wspc_{secrets.token_hex(6)}"


def practice_auroc(
    bundle: WspBundle, tensors: dict[str, Any], settings: WSPSettings
) -> float | None:
    """The practice-VALIDATION AUROC of ``tensors`` under the bundle's function (FR-016.2's
    evaluator), or ``None`` when the validation rows hold one class."""
    import numpy as np

    from wsp.bundles import store_dir
    from wsp.verdict import auroc
    from wsp.weight_ops.evaluator import score_rows
    from wsp.weight_ops.practice import load_practice

    directory = store_dir(settings, bundle.bundle_sha256)
    paths = {f["role"]: directory / f["name"] for f in bundle.manifest["files"]}
    practice = load_practice(paths, bundle.manifest["practice"])
    rows = np.flatnonzero(np.asarray(practice.validation_mask, dtype=bool))
    scores = score_rows(
        bundle.manifest["function"], tensors, practice.activations, practice.offsets, rows
    )
    return auroc(scores, np.asarray(practice.labels, dtype=np.int64)[rows])


def qualify(run: WspRun, verified: Verified, settings: WSPSettings) -> str | None:
    """PF-20 (FR-016.30): why this patch is ``no_change``, or ``None`` for a candidate.

    ``no_change`` when the re-applied patch changes zero entries, or when the candidate's
    practice-validation AUROC is BELOW its parent's (the parent of a ``create`` session without
    ``keep_weights`` is the bundle with its editable tensors zeroed, ``lineage.starts_zeroed``).
    A validation set with one class has no AUROC, so it cannot disqualify."""
    from wsp.lineage import starts_zeroed, zeroed

    if int(verified.manifest["changed_entries_total"]) == 0:
        return "the patch changes no entry"
    parent = verified.base
    if starts_zeroed(run.request):
        parent = zeroed(verified.bundle.manifest, verified.base)
    before = practice_auroc(verified.bundle, parent, settings)
    after = practice_auroc(verified.bundle, verified.edited, settings)
    if before is not None and after is not None and after < before:
        return f"practice-validation AUROC {after:.6f} is below the parent's {before:.6f}"
    return None


def record_candidate(
    session: Session,
    run: WspRun,
    verified: Verified,
    members: dict[str, bytes],
    directory: Path,
) -> str:
    """Store the verified patch beside the run and append its candidate row (never updated)."""
    from wsp.lineage import lineage_parent
    from wsp.models import WspCandidate

    manifest = verified.manifest
    parent = lineage_parent(run.request, verified.bundle.manifest, verified.base)
    (directory / "patch_patch.json").write_bytes(members["patch/patch.json"])
    (directory / "patch_edit.npz").write_bytes(members["patch/edit.npz"])
    session.add(
        WspCandidate(
            candidate_id=new_candidate_id(),
            run_id=run.run_id,
            parent_kind=parent["kind"],
            parent_ref=parent["ref"],
            bundle_base_sha256=manifest["bundle_base_sha256"],
            patch_sha256=manifest["patch_sha256"],
            changed_entries=int(manifest["changed_entries_total"]),
            tensors={"tensors": manifest["tensors"]},
            created_at=dt.datetime.now(dt.UTC),
        )
    )
    session.flush()
    return str(manifest["patch_sha256"])


def metered_usage(session: Session, run_id: str) -> dict[str, Any]:
    """The gateway's records, the only usage a run reports (FR-016.19 step 3). A call the gateway
    refused before spending anything (``outcome: refused``, live progress 2026-11-20) is not a
    request: it is on the run's call timeline, never in these counts."""
    from wsp.models import WspUsage

    not_refused = sa.or_(WspUsage.outcome.is_(None), WspUsage.outcome != "refused")

    row = session.execute(
        sa.select(
            sa.func.count(WspUsage.id),
            sa.func.coalesce(sa.func.sum(WspUsage.input_tokens), 0),
            sa.func.coalesce(sa.func.sum(WspUsage.output_tokens), 0),
            sa.func.coalesce(sa.func.sum(WspUsage.cost_usd), 0),
        ).where(WspUsage.run_id == run_id, not_refused)
    ).one()
    return {
        "requests": int(row[0]),
        "input_tokens": int(row[1]),
        "output_tokens": int(row[2]),
        "cost_usd": float(row[3]),
    }
