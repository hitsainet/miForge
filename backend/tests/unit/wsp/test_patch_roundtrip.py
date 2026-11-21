"""FR-016.4: create a patch, apply it to a fresh base copy, get the edit back bit-exactly."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tests.support.wsp_fixtures import DTYPES, FORMS, patch_case
from wsp.schemas import errors as schema_errors
from wsp.weight_ops.hashing import file_sha256, jcs_sha256
from wsp.weight_ops.patch import apply_patch, build_patch, write_patch
from wsp.weight_ops.tensor import Tensor


def _fresh(tensors: dict[str, Tensor]) -> dict[str, Tensor]:
    return {k: Tensor(t.dtype, np.array(t.bits, copy=True)) for k, t in tensors.items()}


@pytest.mark.parametrize("form", FORMS)
@pytest.mark.parametrize("dtype", DTYPES)
def test_round_trip_is_bit_exact(dtype: str, form: str, tmp_path) -> None:
    base, edited, kwargs = patch_case(dtype, form, seed=11)
    built = build_patch(base, edited, **kwargs)
    assert built is not None
    (name,) = edited
    entry = built.manifest["tensors"][0]
    assert entry["form"] == form, entry
    write_patch(tmp_path / "p", built)
    manifest, out = apply_patch(tmp_path / "p", _fresh(base))
    assert np.array_equal(out[name].bits, edited[name].bits)
    assert out[name].dtype == dtype
    assert out["bias"].same_bits(base["bias"])  # an untouched tensor passes through
    assert entry["edited_sha256"] == edited[name].sha256
    assert entry["base_sha256"] == base[name].sha256
    assert manifest == built.manifest


def test_the_written_manifest_conforms_and_its_digests_recompute(tmp_path) -> None:
    base, edited, kwargs = patch_case("float32", "sparse")
    built = build_patch(base, edited, **kwargs)
    write_patch(tmp_path / "p", built)
    manifest = json.loads((tmp_path / "p" / "patch.json").read_text())
    assert schema_errors("miforge-wsp-patch-v1.json", manifest) == []
    assert manifest["arrays_sha256"] == file_sha256(tmp_path / "p" / "edit.npz")
    assert manifest["patch_sha256"] == jcs_sha256(
        {k: v for k, v in manifest.items() if k != "patch_sha256"}
    )
    assert manifest["content_sha256"] == jcs_sha256(
        {k: manifest[k] for k in ("bundle_base_sha256", "parent", "tensors")}
    )


def test_content_sha256_ignores_who_and_when() -> None:
    base, edited, kwargs = patch_case("float32", "dense")
    first = build_patch(base, edited, **kwargs)
    kwargs2 = dict(
        kwargs,
        created_at="2027-01-01T00:00:00Z",
        created_by={
            "provider": "weightspace",
            "provider_version": "0.3.0",
            "run_id": "wsp_ffffffffffff",
        },
    )
    second = build_patch(base, edited, **kwargs2)
    assert first.manifest["content_sha256"] == second.manifest["content_sha256"]
    assert first.manifest["patch_sha256"] != second.manifest["patch_sha256"]


def test_no_change_writes_no_patch() -> None:
    base, _, kwargs = patch_case("float32", "sparse")
    assert build_patch(base, {"weight": base["weight"]}, **kwargs) is None


def test_the_archive_is_deterministic() -> None:
    base, edited, kwargs = patch_case("bfloat16", "lowrank", seed=3)
    assert build_patch(base, edited, **kwargs).arrays == build_patch(base, edited, **kwargs).arrays


def test_lowrank_applies_one_term_at_a_time() -> None:
    """R1B-X4: ``(base + t1) + t2``, never ``base + (t1 + t2)``.

    A fused ``a @ b`` rounds differently.

    With ``t1 = 1e16`` and ``t2 = -1e16`` on a base of 1.0, the stepwise sum loses the 1.0 (float64
    has
    53 bits) and gives 0; the fused product cancels first and gives 1.0.
    """
    from wsp.weight_ops.patch import lowrank_reconstruct
    from wsp.weight_ops.tensor import from_values

    base = from_values("float32", np.array([[1.0]], dtype=np.float32))
    a = np.array([[1e8, 1e8]])
    b = np.array([[1e8], [-1e8]])
    assert lowrank_reconstruct(base, a, b).values64()[0, 0] == 0.0
    assert (base.values64() + a @ b)[0, 0] == 1.0
