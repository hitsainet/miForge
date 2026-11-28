"""The protocol-test pass: its rule, its record on the endpoint, and whether it is current (feature
019 P2; FR-019.11, FR-019.12; F19-D3, D8, D9, D23, D25, D26, D27; TID §3.6).

- **A run is a protocol test** when ``protocol_test_endpoint_id`` is set, which only the
  protocol-test route does (``runs.create_run`` with ``Prepared.protocol_test``). A ``caller_ref``
  that starts ``protocol_test:`` marks nothing.
- **A pass** (:func:`is_pass`, F19-D9, F19-D26) is a test run that ended ``completed`` with zero
  protocol-refused turns and at least one call the gateway SERVED. Anything else is a fail,
  including a cancel (of a queued test too: F19-D23) and ``worker_lost``.
- **The record** (:func:`record`) is written by ``runs.transition``'s terminal branch, the one
  place every run's end passes, so the latest test's result replaces the one before only when that
  test ends; while a re-test is queued or running, the earlier result stands.
- **What was tested:** the base URL and the model (sealed: the endpoint's own, unchanging) and the
  action protocol the run was PUSHED with (``mf_wsp_runs.action_protocol``, written at the push).
- **Current** (:func:`pass_state`) is computed on read: a pass whose three tested values all equal
  the endpoint's now. A protocol PATCH clears it; nothing is rewritten.
- **Required** (F19-D8): only a generic OpenAI-compatible endpoint whose protocol is
  ``structured`` needs a current pass; miServer keeps 018's rule.
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Any, Literal

import sqlalchemy as sa
from sqlalchemy.orm import Session

from wsp.models import DEFAULT_ACTION_PROTOCOL, WspEndpoint, WspRun, WspUsage


@dataclass(frozen=True)
class PassState:
    result: Literal["passed", "failed"] | None
    run_id: str | None
    at: dt.datetime | None
    tested: dict[str, str | None]
    current: bool
    cleared: bool
    required: bool


def required(endpoint: WspEndpoint) -> bool:
    """F19-D8: a pass is needed by a generic OpenAI-compatible ``structured`` endpoint only."""
    return (
        endpoint.kind == "openai_compatible"
        and endpoint.serving == "generic"
        and (endpoint.action_protocol or DEFAULT_ACTION_PROTOCOL) == "structured"
    )


def pass_state(endpoint: WspEndpoint) -> PassState:
    """The endpoint's latest test and whether its pass still holds for the endpoint as it is."""
    tested = {
        "base_url": endpoint.protocol_tested_base_url,
        "model": endpoint.protocol_tested_model,
        "action_protocol": endpoint.protocol_tested_action_protocol,
    }
    now = {
        "base_url": endpoint.base_url,
        "model": endpoint.model,
        "action_protocol": endpoint.action_protocol or DEFAULT_ACTION_PROTOCOL,
    }
    result = endpoint.protocol_test_result
    passed = result == "passed"
    same = tested == now
    return PassState(
        result="passed" if passed else ("failed" if result == "failed" else None),
        run_id=endpoint.protocol_test_run_id,
        at=endpoint.protocol_test_at,
        tested=tested,
        current=passed and same,
        cleared=passed and not same,
        required=required(endpoint),
    )


def served_calls(session: Session, run_id: str) -> int:
    return int(
        session.scalar(
            sa.select(sa.func.count())
            .select_from(WspUsage)
            .where(WspUsage.run_id == run_id, WspUsage.outcome == "served")
        )
        or 0
    )


def is_pass(session: Session, status: str, refused_turns: int | None, run_id: str) -> bool:
    """F19-D9, F19-D26: completed, zero refused turns, at least one served call."""
    return (
        status == "completed" and (refused_turns or 0) == 0 and served_calls(session, run_id) >= 1
    )


def record(session: Session, run_id: str) -> None:
    """Write a protocol test's result on its endpoint, at the run's end (FR-019.12). Reads the run
    by column, fresh from the database (``transition`` updates without loading). A run that is no
    protocol test writes nothing."""
    row = session.execute(
        sa.select(
            WspRun.protocol_test_endpoint_id,
            WspRun.status,
            WspRun.refused_turns,
            WspRun.action_protocol,
            WspRun.finished_at,
        ).where(WspRun.run_id == run_id)
    ).one_or_none()
    if row is None or row.protocol_test_endpoint_id is None:
        return
    endpoint_id = row.protocol_test_endpoint_id
    _lock_endpoint(session, endpoint_id)
    endpoint = session.scalar(
        sa.select(WspEndpoint).where(WspEndpoint.id == endpoint_id).with_for_update()
    )
    if endpoint is None:
        return
    passed = is_pass(session, row.status, row.refused_turns, run_id)
    values: dict[str, Any] = {
        "protocol_test_result": "passed" if passed else "failed",
        "protocol_test_run_id": run_id,
        "protocol_test_at": row.finished_at or dt.datetime.now(dt.UTC),
        "protocol_tested_base_url": endpoint.base_url,
        "protocol_tested_model": endpoint.model,
        # The protocol the run was PUSHED with; a test that never reached its push (a cancelled
        # queued test, F19-D23) tested the endpoint's protocol as it stood.
        "protocol_tested_action_protocol": row.action_protocol
        or endpoint.action_protocol
        or DEFAULT_ACTION_PROTOCOL,
    }
    for name, value in values.items():
        setattr(endpoint, name, value)
    session.flush()


def _lock_endpoint(session: Session, endpoint_id: str) -> None:
    """SQLite ignores ``FOR UPDATE``: a no-op write takes its write lock (``gateway_ledger``)."""
    if session.get_bind().dialect.name == "sqlite":
        session.execute(
            sa.update(WspEndpoint)
            .where(WspEndpoint.id == endpoint_id)
            .values(id=WspEndpoint.id)
            .execution_options(synchronize_session=False)
        )


def admitted(endpoint: WspEndpoint) -> bool:
    """Whether the gateway admits ``response_format`` for this endpoint (FR-019.13): miServer's
    OpenAI kind always (018), a generic OpenAI-compatible endpoint only with a current pass."""
    if endpoint.kind != "openai_compatible":
        return False
    return endpoint.serving == "miserver" or pass_state(endpoint).current
