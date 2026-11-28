"""``api.v1.endpoints.attachment``'s route answers as the contract says (021 FR-021.13).

The real route, registry and ``AttachAcquirer``, against the fake miServer weights surface
(the HTTP transport is the only double, as in ``test_attachment_status.py``). The surface
reports an attachment held by another trainer, the answer with every field filled.
"""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from tests.support.answer_cases import answered, assert_covered, case

ROUTER = "api.v1.endpoints.attachment"


@pytest.fixture
def client(monkeypatch: pytest.MonkeyPatch) -> Iterator[TestClient]:
    from api.app import create_app
    from core.config import reset_settings
    from tests.support.fake_weights_surface import FakeWeightsSurface
    from tests.support.fixtures_hotupdate import configure
    from training import acquisition
    from training.acquisition.base import AcquisitionMode

    configure(monkeypatch)
    surface = FakeWeightsSurface()
    surface.foreign_holder = "trainer:nightly-forge"
    monkeypatch.setattr(
        acquisition._REGISTRY[AcquisitionMode.ATTACH], "_transport", surface.transport()
    )
    yield TestClient(create_app(), raise_server_exceptions=False)
    reset_settings()


@case(ROUTER, "GET /api/v1/attachment")
def test_attachment_status(client: TestClient) -> None:
    body = answered("GET /api/v1/attachment", client.get("/api/v1/attachment"))
    assert body["attached"] is True


def test_every_route_of_the_router_has_a_case() -> None:
    assert_covered(ROUTER)
