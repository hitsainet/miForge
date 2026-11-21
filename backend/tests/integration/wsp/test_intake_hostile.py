"""A hostile result through the REAL driver and intake (feature 016; FR-016.19; review OR3-016-1).

R3 review fix (OR3-016-1), 2026-11-16. The worker is untrusted, and program mode runs LLM-written
code there, so every malformed ``patch/edit.npz`` is refused cleanly at intake: the run is failed
``wsp_result_invalid`` with its reason, the result is acked at once (the worker is freed), nothing
is a candidate, no pickle is loaded. Before the fix, a bad CRC-32, a corrupt deflate stream, the
encrypted flag or deflate64 raised out of ``dispatch.drive()``: the run stayed ``running`` and was
later failed ``worker_lost`` when the worker's pull timeout restarted it.

Each case keeps ``patch.json``'s digests consistent with the hostile archive (they are not
secrets), so only the archive reader can refuse it. Adopted from review-016-r3's probe
``test_r3_probe_hostile_npz.py``.
"""

from __future__ import annotations

import base64
import hashlib
import io
import json
import struct
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import sqlalchemy as sa
from fastapi.testclient import TestClient

from tests.integration.wsp.test_bundle_import import Env
from tests.integration.wsp.test_dispatch import World, _j, archive
from tests.integration.wsp.test_dispatch import _running as _pushed
from tests.integration.wsp.test_dispatch import world as world  # the fixture
from tests.integration.wsp.test_runs_lifecycle import _patch_members, _store
from tests.integration.wsp.test_weight_ops_api import _edit, _patch
from tests.integration.wsp.test_weight_ops_api import api as api  # the fixture
from wsp import runs
from wsp.weight_ops.patch import content_sha256, manifest_text, patch_sha256

Members = dict[str, bytes]


def _zip(members: Members, method: int = zipfile.ZIP_STORED) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", method) as z:
        for name, data in members.items():
            z.writestr(name, data)
    return buffer.getvalue()


def _pickle_npy(marker: Path) -> bytes:
    class Boom:
        def __reduce__(self) -> tuple[Any, ...]:
            import os

            return (os.mkdir, (str(marker),))

    buffer = io.BytesIO()
    np.save(buffer, np.array([Boom()], dtype=object), allow_pickle=True)
    return buffer.getvalue()


def _first(members: Members) -> str:
    return next(iter(members))


def case_pickle(m: Members, marker: Path) -> bytes:
    return _zip({**m, _first(m): _pickle_npy(marker)})


def case_unicode_dtype(m: Members, marker: Path) -> bytes:
    buffer = io.BytesIO()
    np.save(buffer, np.array(["a", "b"]))
    return _zip({**m, _first(m): buffer.getvalue()})


def case_encrypted_flag(m: Members, marker: Path) -> bytes:
    data = bytearray(_zip(m))
    for signature, offset in ((b"PK\x03\x04", 6), (b"PK\x01\x02", 8)):
        i = 0
        while (j := data.find(signature, i)) >= 0:
            data[j + offset] |= 1  # the general-purpose "encrypted" bit
            i = j + 4
    return bytes(data)


def case_bad_crc(m: Members, marker: Path) -> bytes:
    data = bytearray(_zip(m))
    j = data.find(b"PK\x01\x02")
    data[j + 16] ^= 0xFF  # the central directory's CRC-32
    k = data.find(b"PK\x03\x04")
    data[k + 14] ^= 0xFF  # the local header's CRC-32
    return bytes(data)


def case_deflate64(m: Members, marker: Path) -> bytes:
    data = bytearray(_zip(m))
    for signature, offset in ((b"PK\x03\x04", 8), (b"PK\x01\x02", 10)):
        i = 0
        while (j := data.find(signature, i)) >= 0:
            data[j + offset : j + offset + 2] = struct.pack("<H", 9)  # compression method 9
            i = j + 4
    return bytes(data)


def case_bad_deflate(m: Members, marker: Path) -> bytes:
    data = bytearray(_zip(m, zipfile.ZIP_DEFLATED))
    k = data.find(b"PK\x03\x04")
    n, e = struct.unpack("<HH", data[k + 26 : k + 30])
    start = k + 30 + n + e
    for q in range(start, start + 8):
        data[q] ^= 0xA5
    return bytes(data)


def case_huge_shape(m: Members, marker: Path) -> bytes:
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (4611686018427387904, 4), }"
    header = header + " " * (63 - (len(header) + 10) % 64) + "\n"
    raw = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + b"\0" * 16
    return _zip({**m, _first(m): raw})


def case_huge_alloc(m: Members, marker: Path) -> bytes:
    """A 100-byte member whose header declares 64 GiB: refused before numpy allocates it."""
    header = "{'descr': '<f4', 'fortran_order': False, 'shape': (17179869184,), }"
    header = header + " " * (63 - (len(header) + 10) % 64) + "\n"
    raw = b"\x93NUMPY\x01\x00" + struct.pack("<H", len(header)) + header.encode() + b"\0" * 16
    return _zip({**m, _first(m): raw})


def case_truncated_npy(m: Members, marker: Path) -> bytes:
    name = _first(m)
    return _zip({**m, name: m[name][:20]})


def case_not_a_zip(m: Members, marker: Path) -> bytes:
    return b"PK\x05\x06" + b"\0" * 10 + b"junk"


CASES: dict[str, Callable[[Members, Path], bytes]] = {
    f.__name__[5:]: f
    for f in (
        case_pickle,
        case_unicode_dtype,
        case_encrypted_flag,
        case_bad_crc,
        case_deflate64,
        case_bad_deflate,
        case_huge_shape,
        case_huge_alloc,
        case_truncated_npy,
        case_not_a_zip,
    )
}


def _hostile(members: Members, case: str, marker: Path) -> Members:
    """``members`` with ``patch/edit.npz`` replaced and every digest recomputed to match."""
    source = zipfile.ZipFile(io.BytesIO(members["patch/edit.npz"]))
    inner = {i.filename: source.read(i) for i in source.infolist()}
    hostile = CASES[case](inner, marker)
    doc = json.loads(members["patch/patch.json"])
    doc["arrays_sha256"] = hashlib.sha256(hostile).hexdigest()
    doc["content_sha256"] = content_sha256(doc)
    doc["patch_sha256"] = patch_sha256(doc)
    return {**members, "patch/patch.json": manifest_text(doc).encode(), "patch/edit.npz": hostile}


def _candidates(world: World) -> int:
    with world.engine.connect() as conn:
        return int(conn.execute(sa.text("SELECT COUNT(*) FROM mf_wsp_candidates")).scalar_one())


@pytest.mark.parametrize("case", sorted(CASES))
def test_a_hostile_edit_npz_fails_the_run_cleanly_and_frees_the_worker(
    world: World, tmp_path: Path, case: str
) -> None:
    marker = tmp_path / "pickle-ran"
    manifest = _store(world, tmp_path)
    members = _hostile(_patch_members(manifest, tmp_path), case, marker)
    run_id, worker = _pushed(world)
    worker.state = "finished"
    worker.data = archive(worker.boot, run_id, **members)
    actions = world.drive(worker)  # must not raise: one pass ends the run
    run = world.run(run_id)
    failure = _j(run["failure"])
    assert run["status"] == "failed" and failure["code"] == "wsp_result_invalid", run
    assert run["outcome"] == "failed" and run["patch_sha256"] is None, run
    assert "the patch does not re-apply: wsp_patch_malformed" in failure["message"], failure
    assert "Start the run again" in failure["message"], failure  # OR3-016-5: what to do
    assert _candidates(world) == 0 and not marker.exists()
    # Freed at once: the result is acked in the same pass, exactly once.
    assert [c for c in worker.calls if c[0] == "ack"] == [("ack", worker.boot, run_id)]
    assert run["acked_at"] is not None
    assert actions[-2:] == [f"intake_refused:{run_id}", f"acked:{run_id}"], actions


def test_a_bad_crc_is_never_followed_to_worker_lost(world: World, tmp_path: Path) -> None:
    """The review's end-to-end reading: before the fix three beats each raised and the run ended
    ``worker_lost``. Now the first beat ends it, and later beats leave it as it was."""
    manifest = _store(world, tmp_path)
    members = _hostile(_patch_members(manifest, tmp_path), "bad_crc", tmp_path / "m")
    run_id, worker = _pushed(world)
    worker.state = "finished"
    worker.data = archive(worker.boot, run_id, **members)
    world.drive(worker)
    first = world.run(run_id)
    worker.state = "idle"
    world.drive(worker)
    run = world.run(run_id)
    assert first["status"] == run["status"] == "failed"
    assert _j(run["failure"])["code"] == "wsp_result_invalid", run


def test_an_unforeseen_intake_exception_fails_the_run_and_undoes_its_writes(
    world: World, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The intake's last line of defence: an exception nobody named (here, after the real intake
    recorded a candidate) fails the run ``wsp_result_invalid`` naming its type, acks, and the
    savepoint undoes the candidate row."""
    manifest = _store(world, tmp_path)
    members = _patch_members(manifest, tmp_path)
    real = runs.intake_result

    def explodes(*args: Any, **kwargs: Any) -> Any:
        real(*args, **kwargs)
        raise KeyError("unforeseen")

    monkeypatch.setattr(runs, "intake_result", explodes)
    run_id, worker = _pushed(world)
    worker.state = "finished"
    worker.data = archive(worker.boot, run_id, **members)
    world.drive(worker)
    run = world.run(run_id)
    failure = _j(run["failure"])
    assert run["status"] == "failed" and failure["code"] == "wsp_result_invalid", run
    assert "KeyError" in failure["message"] and "wsp_intake_error" in failure["message"]
    assert _candidates(world) == 0 and run["patch_sha256"] is None
    assert [c[0] for c in worker.calls].count("ack") == 1


@pytest.mark.parametrize("case", ["bad_crc", "bad_deflate", "encrypted_flag", "deflate64"])
def test_verify_refuses_a_hostile_archive_instead_of_a_500(
    api: tuple[Env, TestClient], case: str, tmp_path: Path
) -> None:
    """``POST /wsp/patches/verify`` reads the same archive through the same reader."""
    env, client = api
    sha = env.manifest["bundle_sha256"]
    built = _patch(env.bundle.tensors, _edit(env.bundle.tensors, 0, 0.5), sha)
    members = _hostile(
        {
            "patch/patch.json": manifest_text(built.manifest).encode(),
            "patch/edit.npz": built.arrays,
        },
        case,
        tmp_path / "m",
    )
    r = client.post(
        "/api/v1/wsp/patches/verify",
        json={
            "bundle_sha256": sha,
            "patch": {
                "patch_json": members["patch/patch.json"].decode(),
                "edit_npz_base64": base64.b64encode(members["patch/edit.npz"]).decode(),
            },
        },
    )
    assert r.status_code == 422, r.text
    assert r.json()["error"]["code"] == "wsp_patch_malformed"


def test_a_declared_size_is_checked_before_numpy_allocates(tmp_path: Path) -> None:
    """The header's shape times its item size must equal the data the member holds; the message
    names both, so the refusal comes from the size check, not from numpy's MemoryError."""
    from core.errors import Refusal
    from wsp.weight_ops import patch

    manifest = {"tensors": [{"form": "dense", "dtype": "float32", "edited_shape": [4]}]}
    huge = case_huge_alloc({"t0_bits.npy": b""}, tmp_path / "m")
    with pytest.raises(Refusal) as refused:
        patch.read_archive(huge, manifest)
    assert refused.value.code == "wsp_patch_malformed"
    assert "declares 68719476736 bytes of array data but holds 16" in refused.value.message


def test_any_numpy_failure_is_malformed(monkeypatch: pytest.MonkeyPatch) -> None:
    """The backstop around ``read_array``: an exception of any type is ``wsp_patch_malformed``."""
    from core.errors import Refusal
    from wsp.weight_ops import patch

    buffer = io.BytesIO()
    np.save(buffer, np.zeros(4, dtype="<u2"))
    archive_bytes = _zip({"t0_bits.npy": buffer.getvalue()})

    def broken(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("numpy broke")

    monkeypatch.setattr(np.lib.format, "read_array", broken)
    with pytest.raises(Refusal) as refused:
        patch.read_archive(
            archive_bytes, {"tensors": [{"form": "dense", "dtype": "float16", "edited_shape": [4]}]}
        )
    assert refused.value.code == "wsp_patch_malformed" and "RuntimeError" in refused.value.message
