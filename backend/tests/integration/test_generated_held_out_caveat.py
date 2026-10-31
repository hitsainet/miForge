"""A version carrying miDataworks' ``generated_held_out`` caveat imports, and the caveat is
visible at the point of use  (ENH-006 D-10; miForge pickup task 1.6; FR-010.17; R-02.37).

miDataworks 017 widened the closed enum ``$defs/Caveat/properties/code`` of
``midataworks.dataset-version/v1`` with ``generated_held_out``: generated rows may sit in a
split held out by genre or generator, and the version says so. miForge validates every
imported version against its **vendored** copy of that schema, so until the re-vendor
(``0568796``) such a version was refused at import.

Everything goes through the **mounted** app (``create_app()``) over a database built by
``alembic upgrade head``: the import route with ``wait`` and ``GET /api/v1/parts/{id}``.

Negative control (recorded in ``0xcc/tasks/DEBT_TASKS|ENH-006_Pickup_2026-10.md`` 1.6): with
the previous schema bytes (``git show 0568796~1:docs/schemas/midataworks-dataset-version-v1.json``)
restored, the import is refused and this test is red.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from api.app import create_app
from api.deps import session_dependency
from core.config import reset_settings
from tests.support import db as support
from tests.support import part_fixtures as fx

#: The caveat as miDataworks 017 emits it: the held-out split, its directing columns and
#: values (ENH-006 PADR amendment). ``detail`` is open in the schema, so it must come back
#: whole - a reader that kept only ``code`` would lose what makes the caveat actionable.
GENERATED_HELD_OUT: dict[str, Any] = {
    "code": "generated_held_out",
    "severity": "note",
    "message": (
        "Split 'test' is held out by genre and generator and contains generated rows; "
        "disjointness from the training splits is recorded in the held-out proof."
    ),
    "detail": {
        "split": "test",
        "group_columns": ["genre", "generator"],
        "held_out_values": {"genre": ["legal"], "generator": ["model-b"]},
    },
}


@pytest.fixture()
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[TestClient, Path]]:
    monkeypatch.setenv("CHECKPOINT_ROOT", str(tmp_path / "checkpoints"))
    monkeypatch.setenv("PARTS_ROOT", str(tmp_path / "parts"))
    monkeypatch.setenv("TRAINER_STATE_ROOT", str(tmp_path / "trainer-state"))
    monkeypatch.delenv("MIDATAWORKS_BASE_URL", raising=False)
    reset_settings()
    engine = support.migrated_engine(tmp_path / "db")
    app = create_app()

    def one_session():  # type: ignore[no-untyped-def]
        with support.session_for(engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    (tmp_path / "parts").mkdir()
    try:
        yield TestClient(app, raise_server_exceptions=False), tmp_path
    finally:
        app.dependency_overrides.clear()
        reset_settings()


def test_a_generated_held_out_caveat_imports_and_is_visible_at_the_point_of_use(rig) -> None:
    client, tmp_path = rig
    manifest, bodies = fx.miforge_set_manifest()
    manifest["caveats"] = [*manifest["caveats"], GENERATED_HELD_OUT]
    directory = fx.write_export(tmp_path / "export", manifest, bodies)

    imported = client.post(
        "/api/v1/parts/import", json={"source": "dir", "locator": str(directory), "wait": True}
    )

    # Imports: before the re-vendor the strict validator refused the unknown code here.
    assert imported.status_code == 200, imported.text
    answer = imported.json()
    # The import's own answer is the first point of use - carried verbatim, every field.
    assert GENERATED_HELD_OUT in answer["caveats"]
    assert answer["caveats"] == manifest["caveats"]

    # And on the part itself, which is what a run definition and the UI read.
    part = client.get(f"/api/v1/parts/{answer['part_id']}")
    assert part.status_code == 200, part.text
    assert GENERATED_HELD_OUT in part.json()["caveats"]
    assert part.json()["caveats"] == manifest["caveats"]
