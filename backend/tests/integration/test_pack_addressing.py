"""FR-007.25's storage half: a slot's part addressed THROUGH another pack, recorded and read.

Operator decision (Q3, 2026-10-16): build the writer ``mf_pack_parts.addressed_through_pack_id``
never had. Asserted through the mounted app (``POST /api/v1/packs`` -> the row -> the
lineage route), so removing the call in the service or the field in the route goes red.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from models.pack import PackPart
from packs import registry, store
from packs.settings import reset_pack_settings
from tests.support import pack_fixtures as fx
from tests.support.db import migrated_engine, session_for


@pytest.fixture
def engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):  # type: ignore[no-untyped-def]
    monkeypatch.setenv("PACKS_ROOT", str(tmp_path / "packs"))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_pack_settings()
    engine = migrated_engine(tmp_path / "db")
    yield engine
    engine.dispose()
    reset_pack_settings()


@pytest.fixture
def client(engine):  # type: ignore[no-untyped-def]
    app = create_app()

    def override():  # type: ignore[no-untyped-def]
        with session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = override
    with TestClient(app, raise_server_exceptions=False) as test_client:
        yield test_client


def _create(client: TestClient, name: str, **extra):  # type: ignore[no-untyped-def]
    return client.post(
        "/api/v1/packs", json={"name": name, "version": "1", "slots": fx.valid_slots(), **extra}
    )


def _through(engine, pack_id: str) -> dict[str, str | None]:  # type: ignore[no-untyped-def]
    with session_for(engine) as session:
        rows = session.execute(
            sa.select(PackPart.part_kind, PackPart.addressed_through_pack_id).where(
                PackPart.pack_id == pack_id
            )
        ).all()
    return {str(kind): through for kind, through in rows}


class TestTheWriter:
    def test_a_slot_addressed_through_another_pack_is_stored(self, engine, client) -> None:
        assert _create(client, "general").status_code == 201
        response = _create(client, "notary-md", addressed_through={"retention_set": "general@1"})
        assert response.status_code == 201, response.text
        assert _through(engine, "notary-md@1") == {
            "prompt-set": None,
            "test-set": None,
            "retention-set": "general@1",
        }

    def test_the_lineage_renders_the_pack_it_is_addressed_through(self, client) -> None:
        _create(client, "general")
        _create(client, "notary-md", addressed_through={"retention_set": "general@1"})
        slots = {
            row["slot"]: row
            for row in client.get("/api/v1/packs/notary-md/versions/1/lineage").json()["slots"]
        }
        assert slots["retention_set"]["addressed_through"] == "general"
        assert slots["retention_set"]["display_form"].startswith("general/")
        assert slots["prompt_set"]["addressed_through"] == "notary-md"

    def test_a_rebuild_keeps_it(self, engine, client) -> None:
        _create(client, "general")
        _create(client, "notary-md", addressed_through={"retention_set": "general@1"})
        with session_for(engine) as session:
            registry.rebuild_index(session)
        assert _through(engine, "notary-md@1")["retention-set"] == "general@1"


class TestItRefusesBeforeWritingAnything:
    @pytest.mark.parametrize(
        ("through", "code"),
        [
            ({"retention_set": "notary-md@1"}, "pack_addressed_through_self"),
            ({"retention_set": "nowhere@9"}, "pack_addressed_through_unknown"),
            ({"corpus": "general@1"}, "pack_addressed_through_empty_slot"),
            ({"not_a_slot": "general@1"}, "pack_slot_unknown"),
        ],
    )
    def test_a_bad_addressing_refuses_and_leaves_no_pack(
        self, client, through, code
    ) -> None:  # type: ignore[no-untyped-def]
        _create(client, "general")
        response = _create(client, "notary-md", addressed_through=through)
        assert response.status_code >= 400
        assert response.json()["error"]["code"] == code
        assert [d.name for d in store.iter_versions()] == ["general"]
