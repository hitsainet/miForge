"""An error stop's cause reaches the run (fix 2026-11-20, defect 2).

Production, 2026-11-20: four repair runs read ``completed``, ``stop_reason: error``, ``outcome:
no_change``, ``failure: null`` in ``GET /api/v1/wsp/runs/{id}``, while the cause sat only in
``result.json`` (``weightspace``: an ``error`` string; ``miwsp_engine``: ``failure: {code:
wsp_budget_exhausted, ...}``). Each case here runs the real driver and intake (``dispatch.drive``
over a fake worker answering a real result archive), then reads the run through the production
app's routes, and checks the served document against ``miforge-wsp-run-v1``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from tests.integration.wsp.test_dispatch import World, _finish, _j, build_world
from wsp import runs, schemas
from wsp.settings import get_wsp_settings

#: What weightspace wrote on production run wsp_bbf37e82f8ba (before the gateway's messages
#: carried their code).
PRODUCTION_ERROR = "ConflictError: Error code: 409 - The editor session ended; it is not retried."


@pytest.fixture()
def world(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[World]:
    built = build_world(tmp_path, monkeypatch)
    yield built
    built.engine.dispose()
    get_wsp_settings.cache_clear()


def _client(world: World, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    from api.app import create_app
    from api.deps import session_dependency
    from db import session as db_session
    from tests.support.jobs import session_for

    monkeypatch.setattr(db_session, "_engine", world.engine)
    monkeypatch.setattr(db_session, "_session_factory", None)
    app = create_app()

    def one_session() -> Iterator[Any]:
        with session_for(world.engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    return TestClient(app)


def _served(
    world: World,
    monkeypatch: pytest.MonkeyPatch,
    result: dict[str, Any],
    status: str = "failed",
) -> tuple[dict[str, Any], dict[str, Any]]:
    """The run document and its list row, as the routes serve them after the intake."""
    run_id, _worker, actions = _finish(world, result)
    assert actions[0] == f"collected:{run_id}:{status}", actions
    client = _client(world, monkeypatch)
    document = client.get(f"/api/v1/wsp/runs/{run_id}")
    assert document.status_code == 200, document.text
    schemas.validate_run(document.json())
    (row,) = [r for r in client.get("/api/v1/wsp/runs").json()["runs"] if r["run_id"] == run_id]
    return document.json(), row


CASES = [
    pytest.param(
        {
            "stop_reason": "error",
            "error": PRODUCTION_ERROR[:-1] + ". (code: editor_context_exceeded)",
        },
        "gateway_refused",
        "editor_context_exceeded",
        id="weightspace-gateway-refusal",
    ),
    pytest.param(
        {"stop_reason": "error", "error": PRODUCTION_ERROR},
        "error",
        runs.UNNAMED_CAUSE,
        id="weightspace-production-text",
    ),
    pytest.param(
        {
            "stop_reason": "error",
            "failure": {
                "code": "wsp_budget_exhausted",
                "message": "The session's token budget is spent. (code: wsp_budget_exhausted)",
            },
        },
        "token_budget",
        "wsp_budget_exhausted",
        id="engine-token-budget",
    ),
    pytest.param(
        {
            "stop_reason": "error",
            "failure": {
                "code": "wsp_budget_exhausted",
                "message": "The session's cost cap would be exceeded. (code: wsp_budget_exhausted)",
            },
        },
        "cost_budget",
        "wsp_budget_exhausted",
        id="engine-cost-budget",
    ),
    pytest.param(
        {
            "stop_reason": "error",
            "failure": {"code": "worker_kernel_isolation_failed", "message": "PR_SET_DUMPABLE"},
        },
        "error",
        "worker_kernel_isolation_failed",
        id="engine-own-code",
    ),
]


@pytest.mark.parametrize(("result", "stop", "code"), CASES)
def test_the_run_api_serves_the_reported_cause(
    world: World, monkeypatch: pytest.MonkeyPatch, result: dict[str, Any], stop: str, code: str
) -> None:
    budget = stop in runs.BUDGET_STOPS or code in runs.BUDGET_LIKE_CAUSES
    document, row = _served(world, monkeypatch, result, "completed" if budget else "failed")
    message = result.get("error") or result["failure"]["message"]
    assert document["failure"] == {"code": code, "message": message}
    assert document["stop_reason"] == stop
    # Fix 2026-11-21 (live run wsp_c3e954d3f21f): a session that stopped with a cause is FAILED;
    # one a budget ended completes, its partial result kept (an edit with no patch: no_change).
    # This line pinned ("completed", "no_change") for every case until then: it pinned the defect.
    assert (document["status"], document["outcome"]) == (
        ("completed", "no_change") if budget else ("failed", "failed")
    )
    assert row["failure"] == document["failure"] and row["stop_reason"] == stop


def test_the_intake_writes_the_cause_on_the_row(world: World) -> None:
    run_id, _worker, _actions = _finish(world, {"stop_reason": "error", "error": PRODUCTION_ERROR})
    run = world.run(run_id)
    assert _j(run["failure"]) == {"code": runs.UNNAMED_CAUSE, "message": PRODUCTION_ERROR}


def test_a_hostile_code_is_kept_as_text_under_a_plain_code(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    document, _row = _served(
        world,
        monkeypatch,
        {"stop_reason": "error", "failure": {"code": "<b>x</b>" * 20, "message": "m" * 5000}},
    )
    failure = document["failure"]
    assert failure["code"] == runs.UNNAMED_CAUSE
    assert failure["message"].startswith(("<b>x</b>" * 8)[:64] + ": ")
    assert len(failure["message"]) == 2000


def test_an_error_stop_with_no_cause_still_names_one(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    document, _row = _served(world, monkeypatch, {"stop_reason": "error"})
    assert document["failure"]["code"] == runs.UNNAMED_CAUSE
    assert document["stop_reason"] == "error"


def test_a_clean_stop_has_no_cause(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    document, row = _served(
        world, monkeypatch, {"stop_reason": "answered", "error": None}, "completed"
    )
    assert document["failure"] is None and row["failure"] is None
    assert document["stop_reason"] == "answered"


def test_the_runs_gateway_codes_are_the_gateways() -> None:
    """One list, two copies: the intake's must equal the gateway's own refusal table."""
    from wsp_gateway.app import STATUS

    assert set(STATUS) == runs.GATEWAY_CODES


# ------------------------------------- fix 2026-11-21: a cause fails the run (wsp_c3e954d3f21f)


def test_only_running_out_of_context_is_budget_like() -> None:
    """The split, pinned: one gateway code completes, every other one fails."""
    assert runs.BUDGET_LIKE_CAUSES == ("editor_context_exceeded",)
    assert set(runs.BUDGET_LIKE_CAUSES) <= runs.GATEWAY_CODES


#: Every code a worker can report as its cause, and then some: the gateway's own refusal codes,
#: the engine's own (its exceptions' ``code`` attributes and the ones it raises by name), the
#: adapter's text form, and codes nothing produces today. None may read ``completed``, unless
#: the gateway's code was a budget the session spent (then the partial result is kept).
ENGINE_CODES = (
    "wsp_gateway_unreachable",
    "editor_reply_invalid",
    "editor_context_window_too_small",
    "wsp_endpoint_protocol_unsupported",
    "wsp_endpoint_thinking_unsupported",
    "worker_kernel_isolation_failed",
    "worker_session_failed",
    "worker_patch_failed",
    "worker_spec_invalid",
    "http_400",
    "http_502",
    "invalid_request_error",
    "a_code_nothing_produces_yet",
)


def _engine_code_attributes() -> set[str]:
    """The ``code`` class attribute of every exception the engine's session catches by type."""
    from wsp_worker.miwsp_engine import context, protocol

    found = {
        protocol.ProtocolUnsupportedError.code,
        protocol.ThinkingUnsupportedError.code,
        context.ThinkingDoesNotFitError.code,
    }
    assert len(found) == 3
    return found


def _cases() -> list[Any]:
    out: list[Any] = []
    codes = sorted(set(ENGINE_CODES) | runs.GATEWAY_CODES | _engine_code_attributes())
    for code in codes:
        out.append(
            pytest.param(
                {"stop_reason": "error", "failure": {"code": code, "message": f"x (code: {code})"}},
                # The coordinator's ruling: running out of context is budget-like.
                "completed" if code in runs.BUDGET_LIKE_CAUSES else "failed",
                id=f"engine-{code}",
            )
        )
    # The budget the gateway names (``labelled_stop``) completes: the partial result is kept. A
    # budget refusal that names neither budget is a gateway refusal, and fails.
    for which, stop in (("token budget is spent", "token"), ("cost cap would be exceeded", "cost")):
        out.append(
            pytest.param(
                {
                    "stop_reason": "error",
                    "failure": {
                        "code": "wsp_budget_exhausted",
                        "message": f"The session's {which}. (code: wsp_budget_exhausted)",
                    },
                },
                "completed",
                id=f"budget-{stop}",
            )
        )
    out += [
        pytest.param(
            {"stop_reason": "error", "error": "BadRequestError: Error code: 400 - bad request"},
            "failed",
            id="weightspace-upstream-400",
        ),
        pytest.param({"stop_reason": "error"}, "failed", id="error-with-no-cause"),
        pytest.param(
            {
                "stop_reason": "submitted",
                "failure": {"code": "worker_patch_failed", "message": "m"},
            },
            "failed",
            id="a-cause-beside-a-submit",
        ),
    ]
    return out


@pytest.mark.parametrize(("result", "status"), _cases())
def test_a_session_that_stopped_with_a_cause_is_failed_unless_a_budget_ended_it(
    world: World, result: dict[str, Any], status: str
) -> None:
    run_id, _worker, actions = _finish(world, result)
    assert actions[0] == f"collected:{run_id}:{status}", actions
    row = world.run(run_id)
    assert row["status"] == status
    assert row["outcome"] == ("failed" if status == "failed" else "no_change")
    assert _j(row["failure"]) is not None
    if status == "completed":
        assert row["stop_reason"] in runs.BUDGET_STOPS or (
            _j(row["failure"])["code"] in runs.BUDGET_LIKE_CAUSES
        )


def test_a_clean_stop_still_completes(world: World) -> None:
    """The control: no cause, no failure."""
    run_id, _worker, actions = _finish(world, {"stop_reason": "submitted", "answer": "done"})
    assert actions[0] == f"collected:{run_id}:completed"
