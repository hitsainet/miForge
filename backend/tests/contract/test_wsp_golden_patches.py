"""The committed golden patches (FTASKS 2.11a; PF-55).

Each re-applies bit-exactly, and a regenerated copy
is byte-identical, so the files miWorkbench vendors for its round trip are exactly what patch.py
writes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.support.wsp_golden import GOLDEN, generate
from wsp.weight_ops.hashing import file_sha256
from wsp.weight_ops.patch import apply_patch
from wsp.weight_ops.st_io import dump_safetensors, read_safetensors

GOLDEN_DIR = Path(__file__).resolve().parents[3] / "docs" / "schemas" / "examples" / "wsp-patch"
FILES = ("base.safetensors", "edit.npz", "expected.json", "patch.json")


def test_every_form_has_a_golden() -> None:
    assert sorted(p.name for p in GOLDEN_DIR.iterdir() if p.is_dir()) == sorted(GOLDEN)
    for form in GOLDEN:
        assert sorted(p.name for p in (GOLDEN_DIR / form).iterdir()) == sorted(FILES), form


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_each_golden_reapplies_bit_exactly(name: str) -> None:
    form = GOLDEN[name][0]
    folder = GOLDEN_DIR / name
    base = read_safetensors(folder / "base.safetensors")
    expected = json.loads((folder / "expected.json").read_text())
    manifest, edited = apply_patch(folder, base)
    assert manifest["tensors"][0]["form"] == form
    for tensor, digest in expected["edited_sha256"].items():
        assert edited[tensor].sha256 == digest
    import hashlib

    assert (
        hashlib.sha256(dump_safetensors(edited)).hexdigest()
        == expected["edited_safetensors_sha256"]
    )


def test_a_regenerated_copy_is_byte_identical(tmp_path) -> None:
    generate(tmp_path)
    for form in GOLDEN:
        for name in FILES:
            assert file_sha256(tmp_path / form / name) == file_sha256(GOLDEN_DIR / form / name), (
                form,
                name,
            )


@pytest.mark.parametrize(
    "name", sorted(n for n, (form, _, _) in GOLDEN.items() if form == "lowrank")
)
def test_each_lowrank_golden_has_its_stated_rank(name: str) -> None:
    """miWorkbench 044's NEEDS 7: a rank-2 golden, so fusing the terms cannot pass as their sum."""
    import numpy as np

    rank = GOLDEN[name][2]
    with np.load(GOLDEN_DIR / name / "edit.npz") as npz:
        a, b = npz["t0_a"], npz["t0_b"]
    assert a.shape[1] == rank and b.shape[0] == rank, (a.shape, b.shape)
    assert {GOLDEN[n][2] for n in GOLDEN if GOLDEN[n][0] == "lowrank"} >= {1, 2}


def test_rank2_term_order_is_observable() -> None:
    """Adding the two terms in order in float64 is NOT bit-equal to a fused float32 product for
    this golden, so a reader that fuses them fails the reapply test above."""
    import numpy as np

    folder = GOLDEN_DIR / "lowrank_rank2"
    base = read_safetensors(folder / "base.safetensors")["W_dec"].values64()
    with np.load(folder / "edit.npz") as npz:
        a, b = npz["t0_a"], npz["t0_b"]
    in_order = base.copy()
    for k in range(a.shape[1]):
        in_order = in_order + np.outer(a[:, k], b[k])
    fused = (base.astype(np.float32) + (a.astype(np.float32) @ b.astype(np.float32))).astype(
        np.float32
    )
    assert not np.array_equal(in_order.astype(np.float32).view(np.uint32), fused.view(np.uint32))


def test_cancelling_terms_make_a_fused_reader_visibly_wrong() -> None:
    """miWorkbench 044's NEEDS 8: for ``lowrank_cancel`` the in-order float64 sum (what
    ``lowrank_reconstruct`` does, and what the golden's ``edited_sha256`` records) is far from a
    fused ``base + a @ b`` in float64 AND in float32, so a fused reader fails the reapply test."""
    import numpy as np

    folder = GOLDEN_DIR / "lowrank_cancel"
    base = read_safetensors(folder / "base.safetensors")["W_dec"].values64()
    with np.load(folder / "edit.npz") as npz:
        a, b = npz["t0_a"], npz["t0_b"]
    in_order = base.copy()
    for k in range(a.shape[1]):
        in_order = in_order + np.outer(a[:, k], b[k])
    for fused in (
        base + a @ b,
        base.astype(np.float32) + a.astype(np.float32) @ b.astype(np.float32),
    ):
        gap = np.abs(in_order.astype(np.float64) - np.asarray(fused, dtype=np.float64))
        assert float(np.median(gap)) > 0.1, float(np.median(gap))
    _manifest, edited = apply_patch(folder, read_safetensors(folder / "base.safetensors"))
    assert np.allclose(edited["W_dec"].values64().reshape(-1)[1:-1], 0.0)  # the base was lost
