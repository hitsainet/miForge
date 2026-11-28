"""The weight-ops and patch routes through the mounted app (feature 016; FTASKS 7.2; FR-016.3,
FR-016.4, FR-016.5; D5).

A fixture bundle is imported for real (``wsp.bundles.import_bundle`` against the fake miWorkbench),
then ``create_app()`` serves ``POST /wsp/weight-ops/inspect``, ``/wsp/patches/verify`` and
``/wsp/patches/stack`` over the stored files.
"""

from __future__ import annotations

import base64
import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient

from tests.integration.wsp.test_bundle_import import Env
from wsp.settings import get_wsp_settings
from wsp.weight_ops.patch import BuiltPatch, build_patch, manifest_text
from wsp.weight_ops.tensor import Tensor, from_array


@pytest.fixture()
def api(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[tuple[Env, TestClient]]:
    from api.app import create_app
    from api.deps import session_dependency
    from tests.support.jobs import session_for

    env = Env(tmp_path)
    assert env.run()[0] == "imported"
    monkeypatch.setenv("WSP_BUNDLE_ROOT", str(env.settings.wsp_bundle_root))
    get_wsp_settings.cache_clear()
    app = create_app()

    def one_session() -> Iterator[Any]:
        with session_for(env.engine) as session:
            yield session

    app.dependency_overrides[session_dependency] = one_session
    yield env, TestClient(app, raise_server_exceptions=False)
    get_wsp_settings.cache_clear()
    env.engine.dispose()


def _edit(tensors: dict[str, Tensor], index: int, delta: float) -> dict[str, Tensor]:
    w = tensors["weight"].values64().astype(np.float32)
    w[index] += np.float32(delta)
    return {**tensors, "weight": from_array(w)}


def _patch(
    base: dict[str, Tensor], edited: dict[str, Tensor], sha: str, parent: Any = None
) -> BuiltPatch:
    built = build_patch(
        base,
        {"weight": edited["weight"]},
        bundle_base_sha256=sha,
        parent=parent or {"kind": "bundle", "ref": sha, "patch_sha256": None},
        parent_patch_sha256=None if parent is None else parent["patch_sha256"],
        created_by={"provider": "miwsp_engine", "provider_version": "0.1.0", "run_id": "r"},
        created_at="2026-11-16T05:00:00Z",
    )
    assert built is not None
    return built


def _wire(built: BuiltPatch) -> dict[str, str]:
    return {
        "patch_json": manifest_text(built.manifest),
        "edit_npz_base64": base64.b64encode(built.arrays).decode(),
    }


def test_inspect_is_canonical_and_byte_identical(api: tuple[Env, TestClient]) -> None:
    env, client = api
    body = {"bundle_sha256": env.manifest["bundle_sha256"], "cosine_with": ["mean"]}
    first, second = (client.post("/api/v1/wsp/weight-ops/inspect", json=body) for _ in range(2))
    assert first.status_code == 200, first.text
    assert first.content == second.content
    assert first.headers["X-Content-SHA256"] == hashlib.sha256(first.content).hexdigest()
    report = first.json()
    assert [t["name"] for t in report["tensors"]] == sorted(env.bundle.tensors)
    assert "mean" in report["probe"]["cosine_with"]


def test_a_bundle_not_imported_or_unknown_is_refused(api: tuple[Env, TestClient]) -> None:
    env, client = api
    unknown = client.post("/api/v1/wsp/weight-ops/inspect", json={"bundle_sha256": "0" * 64})
    assert unknown.status_code == 404 and unknown.json()["error"]["code"] == "wsp_bundle_not_found"
    with env.engine.begin() as conn:
        import sqlalchemy as sa

        conn.execute(sa.text("UPDATE mf_wsp_bundles SET import_status = 'importing'"))
    pending = client.post(
        "/api/v1/wsp/weight-ops/inspect", json={"bundle_sha256": env.manifest["bundle_sha256"]}
    )
    assert (
        pending.status_code == 409 and pending.json()["error"]["code"] == "wsp_bundle_not_imported"
    )


def test_verify_applies_the_patch_to_the_stored_bundle(api: tuple[Env, TestClient]) -> None:
    env, client = api
    sha = env.manifest["bundle_sha256"]
    built = _patch(env.bundle.tensors, _edit(env.bundle.tensors, 0, 0.5), sha)
    r = client.post(
        "/api/v1/wsp/patches/verify", json={"bundle_sha256": sha, "patch": _wire(built)}
    )
    assert r.status_code == 200, r.text
    assert r.json()["content_sha256"] == built.manifest["content_sha256"]
    assert r.json()["tensors"][0]["edited_sha256"] == built.manifest["tensors"][0]["edited_sha256"]
    other = _patch(env.bundle.tensors, _edit(env.bundle.tensors, 0, 0.5), "c" * 64)
    wrong = client.post(
        "/api/v1/wsp/patches/verify", json={"bundle_sha256": sha, "patch": _wire(other)}
    )
    assert wrong.json()["error"]["code"] == "wsp_patch_base_mismatch"
    bad = client.post(
        "/api/v1/wsp/patches/verify",
        json={"bundle_sha256": sha, "patch": {"patch_json": "{}", "edit_npz_base64": "!!"}},
    )
    assert bad.status_code == 422 and bad.json()["error"]["code"] == "wsp_patch_malformed"


def test_stack_composes_a_chain_and_refuses_siblings(api: tuple[Env, TestClient]) -> None:
    env, client = api
    sha = env.manifest["bundle_sha256"]
    base = env.bundle.tensors
    one = _edit(base, 0, 0.5)
    first = _patch(base, one, sha)
    parent = {"kind": "candidate", "ref": "cand-1", "patch_sha256": first.manifest["patch_sha256"]}
    two = _edit(one, 1, -0.25)
    second = _patch(one, two, sha, parent)
    r = client.post(
        "/api/v1/wsp/patches/stack",
        json={"bundle_sha256": sha, "patches": [_wire(first), _wire(second)]},
    )
    assert r.status_code == 200, r.text
    stacked = r.json()
    verify = client.post(
        "/api/v1/wsp/patches/verify", json={"bundle_sha256": sha, "patch": stacked["patch"]}
    )
    assert verify.json()["tensors"][0]["edited_sha256"] == two["weight"].sha256
    sibling = _patch(base, _edit(base, 2, 1.0), sha)
    refused = client.post(
        "/api/v1/wsp/patches/stack",
        json={"bundle_sha256": sha, "patches": [_wire(first), _wire(sibling)]},
    )
    assert refused.json()["error"]["code"] == "wsp_patch_not_chained"


def _mean_patch(base: dict[str, Tensor], sha: str) -> BuiltPatch:
    m = base["mean"].values64().astype(np.float32)
    m.reshape(-1)[0] += np.float32(0.5)
    built = build_patch(
        base,
        {"mean": from_array(m)},
        bundle_base_sha256=sha,
        parent={"kind": "bundle", "ref": sha, "patch_sha256": None},
        parent_patch_sha256=None,
        created_by={"provider": "miwsp_engine", "provider_version": "0.1.0", "run_id": "r"},
        created_at="2026-11-16T05:00:00Z",
    )
    assert built is not None
    return built


def test_a_patch_to_a_non_editable_tensor_is_refused_by_verify_and_stack(
    api: tuple[Env, TestClient],
) -> None:
    """Review OR1-016-2: ``mean`` is not editable in the bundle, so neither route reports a patch
    that changes it as verified or composable, though it re-applies bit-exactly."""
    env, client = api
    sha = env.manifest["bundle_sha256"]
    assert {t["name"]: t["editable"] for t in env.manifest["tensors"]}["mean"] is False
    built = _mean_patch(env.bundle.tensors, sha)
    verify = client.post(
        "/api/v1/wsp/patches/verify", json={"bundle_sha256": sha, "patch": _wire(built)}
    )
    assert verify.status_code == 422, verify.text
    assert verify.json()["error"]["code"] == "wsp_change_limit_exceeded"
    assert "'mean' is not editable" in verify.json()["error"]["message"]
    stack = client.post(
        "/api/v1/wsp/patches/stack", json={"bundle_sha256": sha, "patches": [_wire(built)]}
    )
    assert stack.status_code == 422, stack.text
    assert stack.json()["error"]["code"] == "wsp_change_limit_exceeded"
