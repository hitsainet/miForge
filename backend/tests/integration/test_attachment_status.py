"""``GET /api/v1/attachment`` - the chip's live read of miLLM, through the live app.

The chip used to render the open run's record, which a run that failed while attached left
at ``attached: true`` (production, 2026-10-18: ``f11w2-r8`` failed at 11:36 UTC and the chip
still read "Attached ... step-900 ... Weights clean" the next day while miLLM held nothing).
These tests drive the real route, the real registry and the real ``AttachAcquirer`` against
the fake weights surface - the HTTP boundary is the only double - and assert the status call
is made once per request, with no identity header, and that what it says is what is served.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.route_table import mounted_routes
from core.config import reset_settings
from tests.support.fake_weights_surface import FakeWeightsSurface
from tests.support.fixtures_hotupdate import BASE_URL, configure
from training import acquisition
from training.acquisition.base import AcquisitionMode


@pytest.fixture
def app() -> Iterator[Any]:
    reset_settings()
    yield create_app()
    reset_settings()


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> FakeWeightsSurface:
    configure(monkeypatch)
    surface = FakeWeightsSurface()
    acquirer = acquisition._REGISTRY[AcquisitionMode.ATTACH]
    monkeypatch.setattr(acquirer, "_transport", surface.transport())
    return surface


def get(app: Any) -> dict[str, Any]:
    response = TestClient(app).get("/api/v1/attachment")
    assert response.status_code == 200
    body: dict[str, Any] = response.json()
    return body


class TestTheChipReadsMiLLMNow:
    def test_the_route_is_in_the_live_registry(self, app: Any) -> None:
        paths = {(m, r.path) for r in mounted_routes(app) for m in getattr(r, "methods", ())}
        assert ("GET", "/api/v1/attachment") in paths

    def test_nothing_attached_reads_not_attached(self, app: Any, fake: FakeWeightsSurface) -> None:
        body = get(app)
        assert body["configured"] is True
        assert body["reachable"] is True
        assert body["attached"] is False
        assert body["holder"] is None
        assert body["serving_base_url"] == BASE_URL
        assert body["resident_model"] == fake.resident_model
        assert body["observed_at"]
        # One status read per request, unauthenticated: status is not identity-gated.
        assert fake.ops() == ["status"]
        assert "authorization" not in fake.calls[0][1]

    def test_a_held_attachment_names_its_holder(self, app: Any, fake: FakeWeightsSurface) -> None:
        fake.foreign_holder = "trainer:nightly-forge"
        body = get(app)
        assert body["attached"] is True
        assert body["holder"] == "trainer:nightly-forge"
        assert body["stale"] is False
        assert body["weights_state"] == "clean"

    def test_each_request_reads_again(self, app: Any, fake: FakeWeightsSurface) -> None:
        fake.foreign_holder = "trainer:nightly-forge"
        assert get(app)["attached"] is True
        fake.foreign_holder = None
        assert get(app)["attached"] is False
        assert fake.call_count("status") == 2

    def test_an_unreachable_miLLM_is_named_not_reported_as_attached(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configure(monkeypatch)

        def down(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        acquirer = acquisition._REGISTRY[AcquisitionMode.ATTACH]
        monkeypatch.setattr(acquirer, "_transport", httpx.MockTransport(down))
        body = get(app)
        assert body["configured"] is True
        assert body["reachable"] is False
        assert body["reason"] == "weights_surface_unreachable"
        assert "attached" not in body

    def test_with_no_miLLM_configured_the_answer_is_not_configured(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configure(monkeypatch, MILLM_BASE_URL="")
        assert get(app) == {"configured": False}

    def test_without_the_hot_update_package_the_answer_is_not_configured(
        self, app: Any, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        configure(monkeypatch)
        acquisition.unavailable(AcquisitionMode.ATTACH, acquisition.ATTACH_UNAVAILABLE)
        try:
            assert get(app) == {"configured": False}
        finally:
            acquisition._install_optional_attach_mode()
