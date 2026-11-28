"""``api.v1.endpoints.wsp_endpoints``'s routes answer as the contract says (021 FR-021.13).

Registration, change, removal and the list run on ``test_endpoints_api``'s world (the gateway's
sealing key set, so a keyed hosted endpoint registers for real). The protocol test runs on the
start gate's world (``test_protocol_test``'s ``gw``), whose stored endpoints are one local and one
external: a local test runs at once, an agent's external one waits for ``wsp_session_external``.
"""

from __future__ import annotations

from tests.integration.wsp.test_endpoint_proposals import PROPOSAL
from tests.integration.wsp.test_endpoints_api import ANTHROPIC, MISERVER, Env
from tests.integration.wsp.test_endpoints_api import env as env
from tests.integration.wsp.test_protocol_test import gw as gw
from tests.integration.wsp.test_wsp_approvals import World as GateWorld
from tests.integration.wsp.test_wsp_approvals import world as gate_world  # noqa: F401
from tests.support.answer_cases import answered, assert_covered, case
from tests.support.approval_world import AGENT

ROUTER = "api.v1.endpoints.wsp_endpoints"


def _registered(env: Env) -> None:
    for body in (ANTHROPIC, MISERVER):
        answered("POST /api/v1/wsp/endpoints", env.client.post("/api/v1/wsp/endpoints", json=body))


@case(ROUTER, "POST /api/v1/wsp/endpoints")
def test_register_endpoint(env: Env) -> None:
    """Gated: an agent's local keyless proposal waits for ``wsp_endpoint_register``; the
    operator's keyed hosted registration and local miServer one each answer the row."""
    answered(
        "POST /api/v1/wsp/endpoints",
        env.client.post("/api/v1/wsp/endpoints", json=PROPOSAL, headers=AGENT),
        pending=True,
    )
    _registered(env)


@case(ROUTER, "GET /api/v1/wsp/endpoints")
def test_list_endpoints(env: Env) -> None:
    _registered(env)
    listed = answered("GET /api/v1/wsp/endpoints", env.client.get("/api/v1/wsp/endpoints"))
    assert len(listed["endpoints"]) == 2


@case(ROUTER, "GET /api/v1/wsp/endpoints")
def test_list_endpoints_after_a_run(gw: GateWorld) -> None:
    """The gate world's rows after a protocol test queued a run on one (its latest run read
    beside the registration read). A world of its own: both worlds' engines live at
    ``tmp_path/db``."""
    answered(
        "POST /api/v1/wsp/endpoints/{endpoint_id}/protocol-test",
        gw.client.post("/api/v1/wsp/endpoints/ep-local/protocol-test", json={}),
    )
    listed = answered("GET /api/v1/wsp/endpoints", gw.client.get("/api/v1/wsp/endpoints"))
    assert len(listed["endpoints"]) == 2


@case(ROUTER, "PATCH /api/v1/wsp/endpoints/{endpoint_id}")
def test_update_endpoint(env: Env) -> None:
    _registered(env)
    changed = answered(
        "PATCH /api/v1/wsp/endpoints/{endpoint_id}",
        env.client.patch(
            "/api/v1/wsp/endpoints/local-ed",
            json={"action_protocol": "structured", "context_window_tokens": 16384},
        ),
    )
    assert changed["action_protocol"] == "structured"


@case(ROUTER, "DELETE /api/v1/wsp/endpoints/{endpoint_id}")
def test_remove_endpoint(env: Env) -> None:
    _registered(env)
    answered(
        "DELETE /api/v1/wsp/endpoints/{endpoint_id}",
        env.client.delete("/api/v1/wsp/endpoints/claude-ed"),
    )


@case(ROUTER, "POST /api/v1/wsp/endpoints/{endpoint_id}/protocol-test")
def test_protocol_test_run(gw: GateWorld) -> None:
    """Gated: an agent's test of the external endpoint waits for ``wsp_session_external``; the
    operator's test of the local one queues its run."""
    key = "POST /api/v1/wsp/endpoints/{endpoint_id}/protocol-test"
    answered(
        key,
        gw.client.post("/api/v1/wsp/endpoints/ep-ext/protocol-test", json={}, headers=AGENT),
        pending=True,
    )
    queued = answered(
        key,
        gw.client.post(
            "/api/v1/wsp/endpoints/ep-local/protocol-test", json={"caller_ref": "answers"}
        ),
    )
    assert queued["status"] == "queued"


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
