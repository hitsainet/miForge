"""F8's reachability control: the live route table and the live registry  (ADR-016 #1).

*A capability is not shipped until a test FAILS when its wiring is removed.* **M12's
target**, and the rule's own origin is the reason this file does not import an endpoint
function: a sibling shipped sixteen fully-implemented, unit-tested, documented MCP tools
that were never registered with its server, and every one of those tests passed by importing
the module directly.

So each assertion below is made against the **mounted application** - its route table, its
generated OpenAPI document, and an actual request through the transport - or against the
**live run-type registry** reached through ``resolve``.

**The control to run** (recorded in the feature report): delete
``api_v1_router.include_router(runtypes.run_types_router)`` from ``api/v1/router.py``, run
this file, require a red, restore. Then the same for each of the other two routers, and for
each ``register(...)`` line in ``training/runtypes/__init__.py``.
"""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from api import health as health_module
from api.app import create_app
from core.config import reset_settings
from training.runtypes import known, resolve
from training.runtypes.base import RUN_TYPE_NAMES

#: ``(method, path)`` for F8's five routes, as the **mounted** app renders them - converter
#: included, because the converter is behaviour: without ``:path`` the adapter routes would
#: 404 every adapter whose base is a Hub repo id.
F8_ROUTES: frozenset[tuple[str, str]] = frozenset(
    {
        ("GET", "/api/v1/run-types"),
        ("POST", "/api/v1/budget/estimate"),
        ("GET", "/api/v1/adapters"),
        ("GET", "/api/v1/adapters/{name:path}"),
        ("POST", "/api/v1/adapters/{name:path}/merge"),
    }
)


@pytest.fixture
def app() -> Any:
    reset_settings()
    return create_app()


@pytest.fixture
def client(app: Any, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    for probe in ("postgres", "redis"):
        monkeypatch.setitem(
            health_module.DEPENDENCY_PROBES, probe, lambda: {"status": "ok", "detail": None}
        )
    return TestClient(app, raise_server_exceptions=False)


def mounted(app: Any) -> set[tuple[str, str]]:
    return {
        (method, route.path)
        for route in app.routes
        for method in (getattr(route, "methods", set()) or set())
        if method != "HEAD"
    }


class TestEveryF8RouteIsOnTheMountedApp:
    @pytest.mark.parametrize(("method", "path"), sorted(F8_ROUTES))
    def test_the_route_is_mounted(self, app: Any, method: str, path: str) -> None:
        assert (method, path) in mounted(app), (
            f"{method} {path} is not mounted. The handler would still be implemented, "
            f"unit-tested and documented, and would exist for no caller."
        )

    def test_every_f8_route_appears_in_the_generated_openapi_document(self, app: Any) -> None:
        """A route absent from OpenAPI is absent for every generated client too."""
        documented = {
            (method.upper(), path)
            for path, operations in app.openapi()["paths"].items()
            for method in operations
        }
        for method, path in F8_ROUTES:
            # OpenAPI renders the path without the converter.
            assert (method, path.replace(":path", "")) in documented, (method, path)


class TestTheRoutesActuallyANSWER:
    """Registered is not the same as routable."""

    def test_the_run_type_list_answers_over_the_transport(self, client: TestClient) -> None:
        response = client.get("/api/v1/run-types")
        assert response.status_code == 200
        body = response.json()
        assert body["count"] == len(RUN_TYPE_NAMES)
        assert {entry["name"] for entry in body["run_types"]} == set(RUN_TYPE_NAMES)

    def test_the_list_is_served_from_the_LIVE_registry(self, client: TestClient) -> None:
        """Not from a literal in the endpoint. Asserted by comparing the response with the
        registry's own contents, so a hand-written list would disagree."""
        served = {entry["name"] for entry in client.get("/api/v1/run-types").json()["run_types"]}
        assert served == set(known())

    def test_each_entry_says_what_the_run_type_CHANGES(self, client: TestClient) -> None:
        """What F8 owes F9: the three values and their labels (FPRD section 4)."""
        for entry in client.get("/api/v1/run-types").json()["run_types"]:
            assert entry["changes"], entry
            assert entry["stored_as"]

    def test_the_budget_preview_answers_and_carries_the_breakdown(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/budget/estimate",
            json={
                "run_type": "create",
                "config": {"learning_rate": 0.0001, "warmup_steps": 10, "batch_size": 1},
                "architecture_config": {
                    "model_type": "llama",
                    "hidden_size": 2048,
                    "intermediate_size": 8192,
                    "num_hidden_layers": 16,
                    "num_attention_heads": 32,
                    "vocab_size": 65536,
                },
            },
        )
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["estimate"]["total_gib_high"] > 0
        assert body["estimate"]["gradients_gib"] > 0

    def test_an_unknown_run_type_is_a_NAMED_refusal_not_a_500(self, client: TestClient) -> None:
        response = client.post(
            "/api/v1/budget/estimate",
            json={
                "run_type": "lora",
                "config": {"learning_rate": 0.0001, "warmup_steps": 10, "batch_size": 1},
                "param_count": 1000,
            },
        )
        assert response.status_code < 500, response.text
        assert response.json()["error"]["code"] == "run_type_unknown"

    def test_an_unrouted_adapter_path_is_a_different_404_from_an_unmounted_one(
        self, client: TestClient
    ) -> None:
        """The path converter matches anything, so a 404 here must come from the handler's
        own refusal rather than from the router - which is what proves the route is live."""
        response = client.get("/api/v1/adapters/not-a-qualified-name")
        assert response.status_code != 404 or response.json().get("error")

    def test_the_merge_route_is_routable_under_a_name_containing_slashes(
        self, client: TestClient
    ) -> None:
        """The converter's whole reason: a Hub repo id contains a slash. Without ``:path``
        this request would be a router 404 with no error envelope."""
        response = client.post(
            "/api/v1/adapters/LiquidAI/LFM2.5-1.2B-Instruct:run-7:adapter-40/merge"
        )
        assert response.status_code != 404 or response.json().get("error"), response.text
        assert "error" in response.json() or response.status_code == 200


class TestTheLiveRegistryIsWhatIsReached:
    @pytest.mark.parametrize("name", RUN_TYPE_NAMES)
    def test_each_run_type_resolves_through_the_registry(self, name: str) -> None:
        assert resolve(name).name == name

    def test_the_registry_is_populated_at_IMPORT_of_the_package(self) -> None:
        """Not by a call the application might forget to make. Re-imported fresh so the
        assertion is about import, not about this session's earlier work."""
        import importlib

        module = importlib.reload(importlib.import_module("training.runtypes"))
        assert set(module.known()) == set(RUN_TYPE_NAMES)

    def test_removing_a_registration_would_be_VISIBLE_here(self) -> None:
        """The control's shape, asserted as a property of the test rather than run here:
        the registry is compared **two-way**, so a missing registration is red and so is an
        undeclared fourth one."""
        assert set(known()) == set(RUN_TYPE_NAMES)
        assert len(known()) == 3


class TestNoF8RouteWasAddedWithoutBeingDeclared:
    def test_the_projects_route_manifest_covers_all_five(self) -> None:
        """``test_reachability.py``'s ``EXPECTED_ROUTES`` is compared in both directions, so
        an F8 route missing from it makes **that** file red. This asserts the two lists
        agree, which is what stops the manifest becoming a partial list that reads as
        exhaustive."""
        from tests.unit.test_reachability import EXPECTED_ROUTES

        assert F8_ROUTES <= EXPECTED_ROUTES, sorted(F8_ROUTES - EXPECTED_ROUTES)
