"""FR-016.4's refusals. Every tampered patch is **re-sealed** (arrays_sha256, content_sha256 and
patch_sha256 recomputed), so the refusal under test is the only thing that can fire: a test that let
a
stale digest refuse first would pass for the wrong reason.
"""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from core.errors import Refusal
from tests.support.wsp_fixtures import patch_case
from wsp.weight_ops.hashing import bytes_sha256
from wsp.weight_ops.patch import (
    apply_patch,
    build_patch,
    content_sha256,
    manifest_text,
    npz_bytes,
    patch_sha256,
    write_patch,
)
from wsp.weight_ops.tensor import Tensor


def _written(tmp_path: Path, dtype: str = "float32", form: str = "sparse"):
    base, edited, kwargs = patch_case(dtype, form, seed=21)
    built = build_patch(base, edited, **kwargs)
    write_patch(tmp_path / "p", built)
    return tmp_path / "p", base


def _arrays(patch_dir: Path) -> dict[str, np.ndarray]:
    with np.load(patch_dir / "edit.npz", allow_pickle=False) as data:
        return {k: data[k] for k in data.files}


def reseal(
    patch_dir: Path,
    *,
    arrays: dict[str, np.ndarray] | None = None,
    raw: bytes | None = None,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> None:
    manifest = json.loads((patch_dir / "patch.json").read_text())
    if arrays is not None:
        raw = npz_bytes(arrays)
    if raw is not None:
        (patch_dir / "edit.npz").write_bytes(raw)
        manifest["arrays_sha256"] = bytes_sha256(raw)
    if mutate is not None:
        mutate(manifest)
    manifest["content_sha256"] = content_sha256(manifest)
    manifest["patch_sha256"] = patch_sha256(manifest)
    (patch_dir / "patch.json").write_text(manifest_text(manifest))


def _refused(patch_dir: Path, base: dict[str, Tensor]) -> Refusal:
    with pytest.raises(Refusal) as err:
        apply_patch(patch_dir, base)
    return err.value


def test_the_untampered_patch_applies(tmp_path) -> None:
    """The control for every test below: the same patch, unmodified, is accepted."""
    patch_dir, base = _written(tmp_path)
    reseal(patch_dir)
    apply_patch(patch_dir, base)


def test_a_base_with_one_changed_bit_is_refused(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)
    bits = np.array(base["weight"].bits, copy=True)
    bits[3] ^= 1
    base = dict(base, weight=Tensor("float32", bits))
    assert _refused(patch_dir, base).code == "wsp_patch_base_mismatch"


def test_a_base_missing_the_tensor_is_refused(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)
    assert _refused(patch_dir, {"bias": base["bias"]}).code == "wsp_patch_base_mismatch"


def test_an_out_of_range_index_is_refused(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)
    arrays = _arrays(patch_dir)
    arrays["t0_idx"] = arrays["t0_idx"].copy()
    arrays["t0_idx"][-1] = base["weight"].size  # one past the end
    reseal(patch_dir, arrays=arrays)
    err = _refused(patch_dir, base)
    assert err.code == "wsp_patch_malformed" and "outside" in err.message


def test_a_negative_index_is_refused(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)
    arrays = _arrays(patch_dir)
    arrays["t0_idx"] = arrays["t0_idx"].copy()
    arrays["t0_idx"][0] = -1
    reseal(patch_dir, arrays=arrays)
    assert "outside" in _refused(patch_dir, base).message


def test_values_of_the_wrong_dtype_are_refused(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)
    arrays = _arrays(patch_dir)
    arrays["t0_val"] = arrays["t0_val"].astype(np.uint16)
    reseal(patch_dir, arrays=arrays)
    err = _refused(patch_dir, base)
    assert err.code == "wsp_patch_malformed" and "bits" in err.message


def test_a_zip_bomb_is_refused_before_decompression(tmp_path) -> None:
    """A member declaring more bytes than its tensors can need; it compresses to almost nothing."""
    patch_dir, base = _written(tmp_path)
    arrays = _arrays(patch_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("t0_idx.npy", b"\0" * (64 << 20))
        body = io.BytesIO()
        np.lib.format.write_array(body, arrays["t0_val"])
        archive.writestr("t0_val.npy", body.getvalue())
    raw = buffer.getvalue()
    assert len(raw) < 1 << 20
    reseal(patch_dir, raw=raw)
    err = _refused(patch_dir, base)
    assert err.code == "wsp_patch_malformed" and "uncompressed" in err.message


@pytest.mark.parametrize(
    "name", ["../t0_idx.npy", "/abs.npy", "t0_idx.npy/../x.npy", "t0_evil.npy"]
)
def test_an_unsafe_or_unknown_member_name_is_refused(name: str, tmp_path) -> None:
    patch_dir, base = _written(tmp_path)
    arrays = _arrays(patch_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        for key, arr in arrays.items():
            body = io.BytesIO()
            np.lib.format.write_array(body, arr)
            archive.writestr(f"{key}.npy", body.getvalue())
        archive.writestr(name, b"x")
    reseal(patch_dir, raw=buffer.getvalue())
    err = _refused(patch_dir, base)
    assert err.code == "wsp_patch_malformed" and "unexpected member" in err.message


def test_a_pickled_member_is_refused(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)
    arrays = _arrays(patch_dir)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        body = io.BytesIO()
        np.lib.format.write_array(body, arrays["t0_idx"])
        archive.writestr("t0_idx.npy", body.getvalue())
        body = io.BytesIO()
        np.lib.format.write_array(body, np.array([object()], dtype=object), allow_pickle=True)
        archive.writestr("t0_val.npy", body.getvalue())
    reseal(patch_dir, raw=buffer.getvalue())
    err = _refused(patch_dir, base)
    assert err.code == "wsp_patch_malformed" and "plain numpy" in err.message


def test_an_archive_that_does_not_hash_to_arrays_sha256_is_refused(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)
    raw = bytearray((patch_dir / "edit.npz").read_bytes())
    raw[-30] ^= 0xFF
    (patch_dir / "edit.npz").write_bytes(bytes(raw))
    assert "arrays_sha256" in _refused(patch_dir, base).message


def test_a_tampered_manifest_digest_is_refused(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)
    manifest = json.loads((patch_dir / "patch.json").read_text())
    manifest["tensors"][0]["l2_delta"] = 99.0  # not resealed
    (patch_dir / "patch.json").write_text(manifest_text(manifest))
    assert "content_sha256" in _refused(patch_dir, base).message


def test_a_wrong_edited_hash_is_refused(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)

    def wrong(m: dict[str, Any]) -> None:
        m["tensors"][0]["edited_sha256"] = "c" * 64

    reseal(patch_dir, mutate=wrong)
    assert "edited_sha256" in _refused(patch_dir, base).message


def test_a_schema_invalid_manifest_is_refused(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)

    def invent(m: dict[str, Any]) -> None:
        m["invented"] = True

    reseal(patch_dir, mutate=invent)
    assert _refused(patch_dir, base).code == "wsp_patch_malformed"


def test_an_append_is_refused_in_phase_1(tmp_path) -> None:
    patch_dir, base = _written(tmp_path)

    def append(m: dict[str, Any]) -> None:
        m["tensors"][0]["append"] = {"axis": 0, "count": 1}

    reseal(patch_dir, mutate=append)
    assert "append" in _refused(patch_dir, base).message
