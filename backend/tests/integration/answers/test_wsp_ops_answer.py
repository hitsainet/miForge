"""``api.v1.endpoints.wsp_ops``'s routes answer as the contract says (021 FR-021.13).

The weight operations run over a bundle imported for real (``test_weight_ops_api``'s ``api``
world); the run validation runs on the start gate's world (``test_wsp_approvals``), whose stored
bundle and endpoints are what ``POST /wsp/runs`` freezes against.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from tests.integration.wsp.test_bundle_import import Env
from tests.integration.wsp.test_weight_ops_api import _edit, _patch, _wire
from tests.integration.wsp.test_weight_ops_api import api as api
from tests.integration.wsp.test_wsp_approvals import World as GateWorld
from tests.integration.wsp.test_wsp_approvals import world as gate_world  # noqa: F401
from tests.support.answer_cases import answered, assert_covered, case

ROUTER = "api.v1.endpoints.wsp_ops"


@case(ROUTER, "GET /api/v1/wsp/providers")
def test_list_providers(api: tuple[Env, TestClient]) -> None:
    _, client = api
    listed = answered("GET /api/v1/wsp/providers", client.get("/api/v1/wsp/providers"))
    assert listed["providers"]


@pytest.fixture()
def gate(request: pytest.FixtureRequest) -> GateWorld:
    """``test_wsp_approvals``' start-gate world (taken by name: its import is not shadowed)."""
    world: GateWorld = request.getfixturevalue("gate_world")
    return world


@case(ROUTER, "POST /api/v1/wsp/runs/validate")
def test_validate_run(gate: GateWorld) -> None:
    body = {**gate.body, "endpoint_id": "ep-local", "kind": "repair"}
    answered(
        "POST /api/v1/wsp/runs/validate", gate.client.post("/api/v1/wsp/runs/validate", json=body)
    )


@case(ROUTER, "POST /api/v1/wsp/weight-ops/inspect")
def test_inspect_weights(api: tuple[Env, TestClient]) -> None:
    env, client = api
    body = {"bundle_sha256": env.manifest["bundle_sha256"], "cosine_with": ["mean"]}
    answered(
        "POST /api/v1/wsp/weight-ops/inspect",
        client.post("/api/v1/wsp/weight-ops/inspect", json=body),
    )


@case(ROUTER, "POST /api/v1/wsp/patches/verify")
def test_verify_patch(api: tuple[Env, TestClient]) -> None:
    env, client = api
    sha = env.manifest["bundle_sha256"]
    built = _patch(env.bundle.tensors, _edit(env.bundle.tensors, 0, 0.5), sha)
    answered(
        "POST /api/v1/wsp/patches/verify",
        client.post(
            "/api/v1/wsp/patches/verify", json={"bundle_sha256": sha, "patch": _wire(built)}
        ),
    )


@case(ROUTER, "POST /api/v1/wsp/patches/stack")
def test_stack_patches(api: tuple[Env, TestClient]) -> None:
    """A chain of two composes into one patch (the shape with ``patch`` and both digests)."""
    env, client = api
    sha = env.manifest["bundle_sha256"]
    base = env.bundle.tensors
    one = _edit(base, 0, 0.5)
    first = _patch(base, one, sha)
    parent = {"kind": "candidate", "ref": "cand-1", "patch_sha256": first.manifest["patch_sha256"]}
    second = _patch(one, _edit(one, 1, -0.25), sha, parent)
    stacked = answered(
        "POST /api/v1/wsp/patches/stack",
        client.post(
            "/api/v1/wsp/patches/stack",
            json={"bundle_sha256": sha, "patches": [_wire(first), _wire(second)]},
        ),
    )
    assert stacked["no_change"] is False


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
