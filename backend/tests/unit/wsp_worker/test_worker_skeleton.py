"""The worker package's skeleton (FTASKS 8.2): ``bundle.py``, ``patch_writer.py``, ``safe_out.py``.

These run in the backend's interpreter, which holds numpy, safetensors and the backend's own
``wsp.weight_ops``, so the worker's writer and reader are compared with the backend's authority
byte for byte: the worker may not import the backend (the fence), so this test is what keeps the
two twins equal.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import pytest
import rfc8785

from tests.support import wsp_fixtures
from wsp.weight_ops import patch as backend_patch
from wsp.weight_ops.tensor import Tensor, from_values
from wsp_worker import bundle, patch_writer, safe_out
from wsp_worker.bundle import BundleError, WorkerTensor

PARENT = {"kind": "bundle", "ref": "c" * 64, "patch_sha256": None}
CREATED_BY = {"provider": "miwsp_engine", "provider_version": "0.1.0", "run_id": "wsp_0123456789ab"}
CREATED_AT = "2026-11-16T05:00:00Z"


def _w(t: Tensor) -> WorkerTensor:
    return WorkerTensor(t.dtype, t.bits.copy())


# ----------------------------------------------------------------------------- RFC 8785


FLOATS = [
    0.0,
    -0.0,
    1.0,
    -1.5,
    1e-5,
    1e-6,
    1e-7,
    1.5e-7,
    123.456,
    1e20,
    1e21,
    1.2345e16,
    5e-324,
    1.7976931348623157e308,
    0.1,
    2.5e-3,
    100.0,
    333333333.3333333,
    4.35e-310,
]


@pytest.mark.parametrize("x", FLOATS)
def test_jcs_numbers_match_the_rfc8785_package(x: float) -> None:
    assert patch_writer.jcs_bytes(x) == bytes(rfc8785.dumps(x))


def test_jcs_matches_the_package_on_random_floats_and_a_document() -> None:
    rng = np.random.default_rng(7)
    values = np.concatenate(
        [rng.normal(size=500) * 10.0 ** rng.integers(-30, 30, 500), rng.uniform(-1, 1, 500)]
    )
    for x in values.tolist():
        assert patch_writer.jcs_bytes(x) == bytes(rfc8785.dumps(x)), x
    doc = {"z": [1, 2.5, None, True, 'é\n"x"'], "a": {"b": -0.0001, "ä": 1e-9}, "é": 0}
    assert patch_writer.jcs_bytes(doc) == bytes(rfc8785.dumps(doc))


def test_jcs_refuses_what_rfc8785_cannot_hold() -> None:
    for bad in (math.nan, math.inf, 2**53):
        with pytest.raises(ValueError):
            patch_writer.jcs_bytes(bad)


# ----------------------------------------------------------------------------- the patch


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "float64"])
@pytest.mark.parametrize("changed", [1, 40])
def test_the_worker_patch_equals_the_backends_byte_for_byte(
    dtype: str, changed: int, tmp_path: Path
) -> None:
    rng = np.random.default_rng(3)
    base = {
        "weight": from_values(dtype, rng.normal(size=(8, 8)).astype(np.float32)),
        "bias": from_values("float64", np.array([0.25])),
    }
    values = base["weight"].values64().reshape(-1).copy()
    values[:changed] += 0.5
    edited = {
        "weight": from_values(dtype, values.reshape(8, 8).astype(np.float32)),
        "bias": base["bias"],
    }
    kwargs = {
        "bundle_base_sha256": "a" * 64,
        "parent": PARENT,
        "parent_patch_sha256": None,
        "created_by": CREATED_BY,
        "created_at": CREATED_AT,
    }
    theirs = backend_patch.build_patch(base, edited, **kwargs)  # type: ignore[arg-type]
    ours = patch_writer.build_patch(
        {k: _w(v) for k, v in base.items()}, {k: _w(v) for k, v in edited.items()}, **kwargs
    )  # type: ignore[arg-type]
    assert theirs is not None and ours is not None
    assert ours[0] == theirs.manifest
    assert ours[1] == theirs.arrays
    assert ours[0]["tensors"][0]["form"] == ("sparse" if changed == 1 else "dense")
    out = patch_writer.write_patch(tmp_path / "patch", *ours)
    _manifest, rebuilt = backend_patch.apply_patch(out, base)
    assert rebuilt["weight"].same_bits(edited["weight"])


def test_no_change_writes_no_patch() -> None:
    t = _w(from_values("float32", np.ones((2, 2), np.float32)))
    assert (
        patch_writer.build_patch(
            {"weight": t},
            {"weight": t},
            bundle_base_sha256="a" * 64,
            parent=PARENT,
            parent_patch_sha256=None,
            created_by=CREATED_BY,
            created_at=CREATED_AT,
        )
        is None
    )


def test_bf16_round_trip_matches_the_backend() -> None:
    from wsp.weight_ops.tensor import float32_to_bf16_bits

    values = np.random.default_rng(1).normal(size=1000).astype(np.float32)
    values[:3] = [np.nan, np.inf, -0.0]
    assert np.array_equal(patch_writer.float32_to_bf16_bits(values), float32_to_bf16_bits(values))


# ----------------------------------------------------------------------------- the bundle


def _pushed(fx: wsp_fixtures.FixtureBundle) -> list[dict[str, object]]:
    return [
        {"name": f["name"], "role": f["role"], "sha256": f["sha256"], "bytes": f["bytes"]}
        for f in fx.manifest["files"]
    ]


@pytest.mark.parametrize("weight_dtype", ["float32", "float64"])
def test_a_fixture_bundle_loads_with_every_hash_checked(weight_dtype: str, tmp_path: Path) -> None:
    fx = wsp_fixtures.make_bundle(tmp_path, weight_dtype=weight_dtype)
    b = bundle.load(tmp_path, fx.manifest, _pushed(fx))
    assert {n: t.sha256 for n, t in b.tensors.items()} == {
        t["name"]: t["sha256"] for t in fx.manifest["tensors"]
    }
    assert b.tensors["weight"].dtype == weight_dtype
    assert isinstance(b.practice.activations, np.memmap)
    assert not b.practice.activations.flags.writeable
    assert b.editable("weight") and not b.editable("mean")


def test_a_tampered_tensor_is_refused(tmp_path: Path) -> None:
    fx = wsp_fixtures.make_bundle(tmp_path)
    manifest = json.loads(json.dumps(fx.manifest))
    next(t for t in manifest["tensors"] if t["name"] == "weight")["sha256"] = "0" * 64
    with pytest.raises(BundleError):
        bundle.load(tmp_path, manifest, _pushed(fx))


def test_bfloat16_heads_are_read_as_their_bits(tmp_path: Path) -> None:
    """safetensors' numpy loader has no bf16; the worker's own reader returns the uint16 bits."""
    from wsp.weight_ops.st_io import write_safetensors

    tensors = {
        "weight": from_values("bfloat16", np.linspace(-2, 2, 12, dtype=np.float32)),
        "bias": from_values("float64", np.array([0.5])),
    }
    write_safetensors(tmp_path / "heads.safetensors", tensors)
    got = bundle.read_heads(tmp_path / "heads.safetensors")
    assert {n: (t.dtype, t.sha256) for n, t in got.items()} == {
        n: (t.dtype, t.sha256) for n, t in tensors.items()
    }
    assert np.array_equal(got["weight"].values(), tensors["weight"].values64().astype(np.float32))


@pytest.mark.parametrize("suffix", [".pt", ".pth", ".bin", ".ckpt", ".pkl"])
def test_a_pickled_format_is_refused_by_name(suffix: str, tmp_path: Path) -> None:
    path = tmp_path / f"heads{suffix}"
    path.write_bytes(b"\x80\x04not read")
    with pytest.raises(BundleError) as caught:
        bundle.read_heads(path)
    assert caught.value.code == "wsp_bundle_format_refused"


# ----------------------------------------------------------------------------- the safe walk


def test_the_safe_walk_keeps_regular_files_and_records_the_rest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(safe_out, "FILE_MAX_BYTES", 100)
    (tmp_path / "result.json").write_text("{}")
    (tmp_path / "patch").mkdir()
    (tmp_path / "patch" / "patch.json").write_text("{}")
    os.symlink("/etc/hostname", tmp_path / "transcript.jsonl")
    os.symlink("/etc", tmp_path / "linkdir")
    os.mkfifo(tmp_path / "fifo")
    (tmp_path / "a").write_text("x")
    os.link(tmp_path / "a", tmp_path / "b")
    (tmp_path / "big").write_bytes(b"x" * 101)
    (tmp_path / "supervisor.json").write_text("{}")
    walk = safe_out.walk(str(tmp_path))
    assert set(walk.files) == {"result.json", "patch/patch.json"}
    reasons = " | ".join(walk.unsafe)
    for needle in (
        "transcript.jsonl: a symlink",
        "linkdir: a symlink",
        "fifo: a fifo",
        "a: a hard link",
        "b: a hard link",
        "big: 101 bytes",
        "supervisor.json",
    ):
        assert needle in reasons, reasons


class _Marker:
    """Unpickling this runs ``os.mkdir(path)``: observable proof that a loader unpickled."""

    def __init__(self, path: str) -> None:
        self.path = path

    def __reduce__(self) -> tuple[object, tuple[str]]:
        import os

        return (os.mkdir, (self.path,))


@pytest.mark.parametrize("form", ["object-npy", "npz-as-npy"])
def test_a_pickled_practice_file_is_refused_without_being_unpickled(
    tmp_path: Path, form: str
) -> None:
    """Review OR1-016-6/-7: the worker's own loader. An object array (which needs pickle) and
    an ``.npz`` stored under the ``.npy`` name are refused ``wsp_bundle_format_refused``, and the
    marker proves nothing was unpickled. The pushed entries match the manifest, so the refusal is
    the loader's, not the push check's."""
    fx = wsp_fixtures.make_bundle(tmp_path / "b")
    marker = tmp_path / "unpickled"
    payload = np.array([_Marker(str(marker))] * 24, dtype=object)
    path = tmp_path / "b" / "practice_labels.npy"
    if form == "object-npy":
        np.save(path, payload, allow_pickle=True)
    else:
        with path.open("wb") as handle:
            np.savez(handle, labels=payload)
    with pytest.raises(BundleError) as caught:
        bundle.load(tmp_path / "b", fx.manifest, _pushed(fx))
    assert caught.value.code == "wsp_bundle_format_refused"
    assert not marker.exists(), "the worker unpickled the file"
