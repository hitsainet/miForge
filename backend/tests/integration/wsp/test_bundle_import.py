"""Bundle import, against a recording fake miWorkbench (feature 016; FTASKS 7.4, 7.5; FR-016.1's AC;
P29, P30, P31, P54, R1B-X7, R1B-X9).

``wsp.bundles.request_import`` and ``import_bundle`` (the task's body) on a migrated database, with
the fake's transport; the route test goes through ``create_app()`` and records the queued task.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient
from sqlalchemy.orm import sessionmaker

from core.errors import Refusal
from tests.integration.wsp.wsp_db import engine_at_head
from tests.support.fake_miworkbench import BASE, FakeMiWorkbench
from tests.support.wsp_fixtures import FixtureBundle, enable_wsp, make_bundle, seal
from wsp import bundles
from wsp.settings import WSPSettings, get_wsp_settings
from wsp.weight_ops.hashing import file_sha256


class Env:
    def __init__(self, tmp_path: Path) -> None:
        self.bundle = make_bundle(tmp_path / "src", rule="max")
        self.fake = FakeMiWorkbench(self.bundle)
        self.engine = engine_at_head(tmp_path)
        self.factory = sessionmaker(bind=self.engine, expire_on_commit=False)
        self.settings = WSPSettings(
            miworkbench_api_url=BASE, wsp_bundle_root=tmp_path / "wsp", wsp_providers_available=""
        )

    @property
    def manifest(self) -> dict[str, Any]:
        return self.fake.manifest if self.fake.manifest is not None else self.bundle.manifest

    def body(self) -> dict[str, Any]:
        m = self.manifest
        return {
            "owner": "miworkbench",
            "bundle_id": m["bundle_id"],
            "bundle_sha256": m["bundle_sha256"],
        }

    def run(self) -> tuple[str, dict[str, Any]]:
        with self.factory() as session:
            row, _queued = bundles.request_import(
                session, self.body(), received_by="operator", settings=self.settings
            )
            session.commit()
            outcome = bundles.import_bundle(
                session, row.bundle_sha256, settings=self.settings, transport=self.fake.transport()
            )
            session.commit()
            return outcome, self.row()

    def row(self) -> dict[str, Any]:
        with self.engine.connect() as conn:
            got = conn.execute(sa.text("SELECT * FROM mf_wsp_bundles")).mappings().one()
            return dict(got)

    def stored(self) -> Path:
        return bundles.store_dir(self.settings, self.manifest["bundle_sha256"])


@pytest.fixture()
def env(tmp_path: Path) -> Iterator[Env]:
    e = Env(tmp_path)
    yield e
    e.engine.dispose()


def _failure(row: dict[str, Any]) -> dict[str, Any]:
    value = row["import_failure"]
    return json.loads(value) if isinstance(value, str) else value


def test_a_bundle_is_imported_and_importing_again_is_a_no_op(env: Env) -> None:
    outcome, row = env.run()
    assert outcome == "imported" and row["import_status"] == "imported", _failure(row)
    assert row["model"] == "fixture/model" and row["sensitivity_value"] == "not_sensitive"
    stored = sorted(p.name for p in env.stored().iterdir())
    assert stored == sorted(e["name"] for e in env.manifest["files"])  # no manifest file there
    for entry in env.manifest["files"]:
        assert file_sha256(env.stored() / entry["name"]) == entry["sha256"]
    # exactly 1 + len(files) GETs, all to the configured base, none with the agent header (P31)
    assert len(env.fake.requests) == 1 + len(env.manifest["files"])
    assert all(r.method == "GET" and str(r.url).startswith(BASE + "/") for r in env.fake.requests)
    assert not any("x-miforge-agent" in {k.lower() for k in r.headers} for r in env.fake.requests)
    with env.factory() as session:
        again, queued = bundles.request_import(
            session, env.body(), received_by="operator", settings=env.settings
        )
        assert queued is False and again.import_status == "imported"


def _stored_nothing(env: Env) -> None:
    assert not env.stored().exists()
    leftovers = list((Path(env.settings.wsp_bundle_root) / "bundles").glob("*"))
    assert leftovers == [], leftovers


def test_a_header_disagreeing_with_the_manifest_is_refused_naming_the_file(env: Env) -> None:
    env.fake.header_override["heads.safetensors"] = "0" * 64
    outcome, row = env.run()
    assert outcome == "failed:wsp_bundle_fetch_failed"
    assert "heads.safetensors" in _failure(row)["message"]
    _stored_nothing(env)


def test_bytes_disagreeing_with_the_header_are_refused(env: Env) -> None:
    """The control C1a: skip the header-versus-bytes comparison and this goes red."""
    data = bytearray((env.bundle.root / "practice_labels.npy").read_bytes())
    data[-1] ^= 1
    env.fake.body_override["practice_labels.npy"] = bytes(data)
    outcome, row = env.run()
    assert outcome == "failed:wsp_bundle_fetch_failed"
    assert "practice_labels.npy" in _failure(row)["message"]
    _stored_nothing(env)


def _reseal_file(env: Env, name: str) -> None:
    """Make the manifest agree with a changed file (so only the deeper check can catch it)."""
    manifest = json.loads(json.dumps(env.bundle.manifest))
    path = env.bundle.root / name
    for entry in manifest["files"]:
        if entry["name"] == name:
            entry["sha256"], entry["bytes"] = file_sha256(path), path.stat().st_size
    env.fake.manifest = seal(manifest)


def test_one_flipped_tensor_byte_is_a_hash_mismatch(env: Env) -> None:
    """The control C1b: skip the tensor re-hash and this goes red (parity may still pass)."""
    path = env.bundle.root / "heads.safetensors"
    data = bytearray(path.read_bytes())
    header = int.from_bytes(data[:8], "little")
    data[8 + header] ^= 1  # the first byte of the first tensor's data
    path.write_bytes(bytes(data))
    _reseal_file(env, "heads.safetensors")
    outcome, _row = env.run()
    assert outcome in ("failed:wsp_bundle_hash_mismatch",)
    _stored_nothing(env)


def test_a_pt_tensor_file_is_refused(env: Env) -> None:
    manifest = json.loads(json.dumps(env.bundle.manifest))
    source = env.bundle.root / "heads.safetensors"
    (env.bundle.root / "heads.pt").write_bytes(source.read_bytes())
    for entry in manifest["files"]:
        if entry["role"] == "heads":
            entry["name"] = "heads.pt"
    env.fake.manifest = seal(manifest)
    outcome, _row = env.run()
    assert outcome == "failed:wsp_bundle_format_refused"
    assert env.fake.requests == env.fake.requests[:1]  # refused before any file was fetched


class _Marker:
    """Unpickling this runs ``os.mkdir(path)``: observable proof that a loader unpickled."""

    def __init__(self, path: str) -> None:
        self.path = path

    def __reduce__(self) -> tuple[Any, tuple[str]]:
        import os

        return (os.mkdir, (self.path,))


def test_a_pickled_npy_is_refused_without_being_unpickled(env: Env, tmp_path: Path) -> None:
    """Refused, and NEVER unpickled: a check after an ``allow_pickle=True`` load would come too
    late, since unpickling untrusted bytes already runs their code (control C1-pickle)."""
    marker = tmp_path / "unpickled"
    path = env.bundle.root / "practice_labels.npy"
    np.save(path, np.array([_Marker(str(marker))] * 24, dtype=object), allow_pickle=True)
    _reseal_file(env, "practice_labels.npy")
    outcome, _row = env.run()
    assert outcome == "failed:wsp_bundle_format_refused"
    assert not marker.exists(), "the importer unpickled the file"
    _stored_nothing(env)


def test_bytes_agreeing_with_their_header_but_not_the_manifest_are_refused(env: Env) -> None:
    """The header is checked against the MANIFEST at the fetch, before any deeper check could
    name it otherwise (control C1a-header)."""
    import hashlib

    name = "practice_labels.npy"
    data = bytearray((env.bundle.root / name).read_bytes())
    data[-1] ^= 1
    env.fake.body_override[name] = bytes(data)
    env.fake.header_override[name] = hashlib.sha256(bytes(data)).hexdigest()
    outcome, row = env.run()
    assert outcome == "failed:wsp_bundle_fetch_failed"
    assert name in _failure(row)["message"]
    _stored_nothing(env)


def test_an_unknown_manifest_key_is_refused(env: Env) -> None:
    manifest = json.loads(json.dumps(env.bundle.manifest))
    manifest["practice"]["extra"] = 1
    env.fake.manifest = seal(manifest)
    outcome, _row = env.run()
    assert outcome == "failed:wsp_bundle_invalid"


def test_a_permuted_weight_fails_parity(env: Env) -> None:
    """FR-016.2's pilot control, end to end through the import."""
    from tests.support.wsp_fixtures import write_safetensors
    from wsp.weight_ops.hashing import tensor_sha256
    from wsp.weight_ops.tensor import from_array

    tensors = dict(env.bundle.tensors)
    weight = tensors["weight"].values64().astype(np.float32)[::-1].copy()
    tensors["weight"] = from_array(weight)
    write_safetensors(env.bundle.root / "heads.safetensors", tensors)
    manifest = json.loads(json.dumps(env.bundle.manifest))
    for t in manifest["tensors"]:
        if t["name"] == "weight":
            t["sha256"] = tensor_sha256(tensors["weight"].bits)
    env.fake.manifest = manifest
    _reseal_file(env, "heads.safetensors")
    env.fake.manifest["tensors"] = manifest["tensors"]
    env.fake.manifest = seal(env.fake.manifest)
    outcome, _row = env.run()
    assert outcome == "failed:wsp_bundle_parity_failed"


def test_the_bundle_digest_is_rfc8785_with_the_1e5_float(env: Env) -> None:
    """R1B-X9: the manifest holds 1e-05 (the parity tolerance); RFC 8785 and json.dumps differ on
    it, so an importer hashing with ``canonical.py`` refuses every real bundle (control C1c)."""
    import rfc8785

    body = {k: v for k, v in env.bundle.manifest.items() if k != "bundle_sha256"}
    assert body["function"]["parity"]["tolerance"] == 1e-05
    jcs = rfc8785.dumps(body)
    assert b"1e-5" in jcs or b"0.00001" in jcs
    assert jcs != json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    assert hashlib.sha256(jcs).hexdigest() == env.bundle.manifest["bundle_sha256"]
    assert env.run()[0] == "imported"


def test_a_start_on_a_bundle_still_importing_is_refused(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from wsp import runs

    enable_wsp(monkeypatch)
    get_wsp_settings.cache_clear()

    with env.factory() as session:
        bundles.request_import(session, env.body(), received_by="operator", settings=env.settings)
        session.commit()
        body = {
            "bundle": env.body(),
            "mode": "session",
            "kind": "repair",
            "task": "t",
            "budgets": {"tokens": 1000, "time_s": 60},
            "primary_sets": ["eval"],
            "sensitivity": "not_sensitive",
        }
        with pytest.raises(Refusal) as refused:
            runs.prepare_start(session, body, "operator")
    get_wsp_settings.cache_clear()
    assert refused.value.code == "wsp_bundle_not_imported"


def test_the_request_shape_is_closed(env: Env) -> None:
    with env.factory() as session:
        for bad in (
            {**env.body(), "url": "http://evil"},
            {**env.body(), "owner": "someone"},
            # 018 (P2-D7): dispatch trusts a ``miforge``-owned row's stored sensitivity, so no
            # import may ever create one; this pins the precondition that shortcut relies on.
            {**env.body(), "owner": "miforge"},
            {**env.body(), "bundle_sha256": "XYZ"},
        ):
            with pytest.raises(Refusal) as refused:
                bundles.request_import(session, bad, received_by="operator", settings=env.settings)
            assert refused.value.code == "wsp_bundle_invalid"
        unset = env.settings.model_copy(update={"miworkbench_api_url": ""})
        with pytest.raises(Refusal) as refused:
            bundles.request_import(session, env.body(), received_by="operator", settings=unset)
        assert refused.value.code == "miworkbench_not_configured"


def test_the_route_answers_202_and_queues_the_import_once(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    from api.app import create_app
    from api.deps import session_dependency
    from tests.support.jobs import session_for
    from wsp import tasks

    monkeypatch.setenv("MIWORKBENCH_API_URL", BASE)
    monkeypatch.setenv("WSP_BUNDLE_ROOT", str(env.settings.wsp_bundle_root))
    get_wsp_settings.cache_clear()
    queued: list[str] = []
    monkeypatch.setattr(tasks.import_bundle_task, "delay", lambda sha: queued.append(sha))
    app = create_app()

    def one_session() -> Iterator[Any]:
        with session_for(env.engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    client = TestClient(app)
    try:
        first = client.post("/api/v1/wsp/bundles/import", json=env.body())
        assert first.status_code == 202 and first.json()["import_status"] == "importing"
        assert queued == [env.manifest["bundle_sha256"]]
        again = client.post("/api/v1/wsp/bundles/import", json=env.body())
        assert again.status_code == 202 and again.json()["queued"] is False
        assert queued == [env.manifest["bundle_sha256"]]
        got = client.get(f"/api/v1/wsp/bundles/{env.manifest['bundle_sha256']}")
        assert got.status_code == 200 and got.json()["manifest"] is None
        assert client.get("/api/v1/wsp/bundles/" + "0" * 64).status_code == 404
    finally:
        get_wsp_settings.cache_clear()


def test_the_importer_task_is_the_registered_one() -> None:
    from workers.celery_app import celery_app
    from wsp import tasks

    celery_app.loader.import_default_modules()
    registered = celery_app.tasks["miforge.wsp.import_bundle"]
    assert (
        registered.run.__module__ == "wsp.tasks"
        and registered.name == tasks.import_bundle_task.name
    )
    assert isinstance(FixtureBundle, type)


def test_an_npz_stored_as_a_practice_npy_is_refused(env: Env, tmp_path: Path) -> None:
    """Review OR1-016-7: an ``.npz`` under a ``.npy`` name loads as an ``NpzFile`` (no pickle
    runs), which used to escape the refusal path and leave the row ``importing`` forever."""
    marker = tmp_path / "unpickled"
    path = env.bundle.root / "practice_labels.npy"
    with path.open("wb") as handle:
        np.savez(handle, labels=np.array([_Marker(str(marker))] * 24, dtype=object))
    _reseal_file(env, "practice_labels.npy")
    outcome, row = env.run()
    assert outcome == "failed:wsp_bundle_format_refused"
    assert row["import_status"] == "failed"
    assert not marker.exists()
    _stored_nothing(env)


def test_two_dimensional_practice_positions_are_refused(env: Env) -> None:
    path = env.bundle.root / "practice_positions.npy"
    positions = np.load(path)
    np.save(path, np.stack([positions, positions]))
    _reseal_file(env, "practice_positions.npy")
    outcome, row = env.run()
    assert outcome == "failed:wsp_bundle_invalid"
    assert row["import_status"] == "failed"
    # The named check, not the catch-all's "could not be checked (TypeError)".
    assert "1-D integer arrays" in _failure(row)["message"]


def test_an_unexpected_error_ends_the_import_failed_not_importing(
    env: Env, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The safety net: any other exception a hostile file provokes marks the row ``failed``, so
    a re-request is not a silent no-op on a row stuck ``importing``."""
    from wsp import bundles

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise TypeError("a shape nobody checked")

    monkeypatch.setattr(bundles, "check_files", boom)
    outcome, row = env.run()
    assert outcome == "failed:wsp_bundle_invalid"
    assert row["import_status"] == "failed"
    assert _failure(row)["code"] == "wsp_bundle_invalid"
