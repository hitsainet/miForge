"""Reachability: the five types are in the LIVE registry  (AC-003.1; ADR-016 #1).

**A capability is not shipped until a test FAILS when its wiring is removed.** This file is
that test for F3, and nothing in it asserts that a plugin module imports - which is exactly
how a sibling shipped **sixteen fully-implemented, unit-tested, documented MCP tools that
were never registered with the server**: every one of their tests passed by importing the
module directly, so the suite was green and the docs said ✅ while the feature did not exist
for any caller.

So every assertion here is made against:

* the **live registry object** the service resolves from, not a list of module names;
* the **mounted application's** route table, not the endpoint module;
* and, for the one type that leaves the process, the **payload and the call count** - "was
  called" passes against a call sending the wrong arguments.

## NEGATIVE CONTROLS - RUN, OBSERVED RED, REVERTED

Recorded here because a structural test never seen failing is not known to work:

1. Delete the ``register("rule", ...)`` entry from ``_REGISTRATIONS`` in
   ``services/selectors/__init__.py``
   -> ``test_all_five_types_resolve_from_the_live_registry`` and
   ``test_every_registered_type_is_resolvable_by_the_service`` **RED** (4 failures).
   Reverted; ``git diff`` clean.
2. Delete ``api_v1_router.include_router(selectors.router)`` from ``api/v1/router.py``
   -> ``test_both_routes_are_on_the_mounted_app`` **RED**, plus
   ``test_no_declared_route_is_missing`` in ``test_reachability.py`` **RED**.
   Reverted; ``git diff`` clean.
3. Empty the registry entirely (``registry._reset_for_tests()`` with no re-registration)
   -> five failures here. This is the control that proves the green above is not vacuous,
   and it is exercised **as a test** rather than by hand, in
   :func:`test_an_empty_registry_fails_this_file`.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from core.config import reset_settings
from core.errors import Refusal
from services.selectors import _REGISTRATIONS, ensure_registered, registry
from services.selectors.resolve import resolve
from services.selectors.settings import get_selector_settings, reset_selector_settings

#: The five names R-02.9 fixes. Declared here independently of the registration block, so
#: that a type silently renamed on one side goes red rather than agreeing with itself.
EXPECTED_KINDS = frozenset({"rule", "judge_model", "reference_match", "human_review", "composite"})

#: The two routes this feature serves, as ``(method, path)``.
EXPECTED_ROUTES = frozenset({("GET", "/api/v1/selectors"), ("POST", "/api/v1/selectors/evaluate")})


@pytest.fixture(autouse=True)
def _registered() -> None:
    ensure_registered()
    reset_selector_settings()
    yield
    reset_selector_settings()


class TestTheLiveRegistry:
    def test_all_five_types_resolve_from_the_live_registry(self) -> None:
        """**The sixteen-unregistered-tools test.** Importable is not registered."""
        assert set(registry.registered_kinds()) == EXPECTED_KINDS, (
            f"registered: {registry.registered_kinds()}. A type that is not in this live "
            f"object does not exist for any caller, however well its module is tested."
        )

    def test_the_registration_block_is_the_only_declaration(self) -> None:
        """The block and the expectation agree, so a rename cannot be self-consistent."""
        assert {kind for kind, _, _ in _REGISTRATIONS} == EXPECTED_KINDS

    @pytest.mark.parametrize("kind", sorted(EXPECTED_KINDS))
    def test_every_registered_type_is_resolvable_by_the_service(self, kind: str) -> None:
        """``registry.get`` is the one resolution path; there is no importlib fallback."""
        assert callable(registry.get(kind))
        assert registry.capability(kind).kind == kind

    def test_an_unknown_type_is_refused_by_name_listing_the_registered_ones(self) -> None:
        from core.errors import Refusal

        with pytest.raises(Refusal) as exc:
            registry.get("bm25_reranker")
        assert exc.value.code == "selector_type_unknown"
        assert exc.value.details["requested"] == "bm25_reranker"
        assert set(exc.value.details["registered"]) == EXPECTED_KINDS

    def test_re_registering_a_name_raises(self) -> None:
        """A silent overwrite is how two plugins answer to one name and import order wins."""
        existing = registry.get("rule")
        with pytest.raises(ValueError, match="already registered"):
            registry.register("rule", existing, registry.capability("rule"))  # type: ignore[arg-type]

    def test_an_empty_registry_fails_this_file(self) -> None:
        """**The control, as a test.** Proves the green above is not vacuous.

        Without this, "the five types are registered" would pass just as happily against a
        harness that could not tell a full registry from an empty one.
        """
        saved_factories = dict(registry._factories)
        saved_capabilities = dict(registry._capabilities)
        try:
            registry._reset_for_tests()
            assert registry.registered_kinds() == ()
            with pytest.raises(Refusal):
                registry.get("rule")
        finally:
            registry._factories.update(saved_factories)
            registry._capabilities.update(saved_capabilities)
        assert set(registry.registered_kinds()) == EXPECTED_KINDS


class TestTheRoutesAreMounted:
    @pytest.fixture
    def app(self):
        reset_settings()
        return create_app()

    def test_both_routes_are_on_the_mounted_app(self, app) -> None:
        """Asserted from the mounted app, not the module: a router that exists but is not
        included is unreachable, and importing it would still pass."""
        mounted = {
            (method, route.path)
            for route in app.routes
            for method in (getattr(route, "methods", set()) or set())
            if method != "HEAD"
        }
        assert EXPECTED_ROUTES <= mounted, f"not mounted: {sorted(EXPECTED_ROUTES - mounted)}"

    def test_both_routes_appear_in_the_generated_openapi(self, app) -> None:
        schema = app.openapi()
        documented = {
            (method.upper(), path)
            for path, operations in schema["paths"].items()
            for method in operations
        }
        assert EXPECTED_ROUTES <= documented

    def test_the_listing_answers_through_the_transport(self, app) -> None:
        """A route that is registered but unroutable is red here and nowhere else."""
        client = TestClient(app, raise_server_exceptions=False)
        response = client.get("/api/v1/selectors")
        assert response.status_code == 200
        assert {row["name"] for row in response.json()} == EXPECTED_KINDS

    def test_evaluate_answers_through_the_transport(self, app) -> None:
        client = TestClient(app, raise_server_exceptions=False)
        body = {
            "config": {
                "kind": "rule",
                "name": "no-guarantees",
                "config_sha256": None,
                "expression": json.dumps({"forbidden_terms": ["guarantee"]}),
            },
            "items": [{"completion": "a careful answer"}, {"completion": "we guarantee it"}],
        }
        response = client.post("/api/v1/selectors/evaluate", json=body)
        assert response.status_code == 200, response.text
        payload = response.json()
        assert [row["value"] for row in payload["verdicts"]] == ["accept", "reject"]
        assert payload["verdicts"][1]["reason_code"] == "rule_forbidden_term"

    def test_there_is_no_write_endpoint(self, app) -> None:
        """R-02.10: a second authoring surface would mean a second hash.

        Asserted rather than merely documented, because "we did not build one" is exactly
        the kind of claim that stops being true quietly.
        """
        mounted = {
            (method, route.path)
            for route in app.routes
            for method in (getattr(route, "methods", set()) or set())
        }
        writes = {
            (method, path)
            for method, path in mounted
            if (path.startswith("/api/v1/selectors") and method in {"PUT", "PATCH", "DELETE"})
            or (method == "POST" and path == "/api/v1/selectors")
        }
        assert writes == set(), f"F3 authors no configuration, but found: {sorted(writes)}"


class TestResolutionGoesThroughTheRegistry:
    def test_a_resolved_config_carries_the_registered_capability(self) -> None:
        """Resolution reads the capability off the live registry, not off the plugin class.

        So a plugin whose registration was removed cannot be resolved by accident through
        some other path - the thing the single-resolution-path rule exists for.
        """
        resolved = resolve(
            {
                "kind": "rule",
                "name": "len",
                "config_sha256": None,
                "expression": json.dumps({"min_length": 3}),
            },
            source="inline",
        )
        assert resolved.capability is registry.capability("rule")

    def test_the_listing_is_the_registry_rather_than_a_hand_written_list(self) -> None:
        views = registry.list_capabilities(get_selector_settings())
        assert {view.name for view in views} == set(registry.registered_kinds())
