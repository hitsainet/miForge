"""The engine's tensor store (feature 018; FTASKS 2.2, 2.4; FR-018.3, FR-018.4).

Run in the backend's interpreter: ``store.py`` imports numpy and the worker's own modules only.
The patch round trip goes through the BACKEND's ``apply_patch`` and ``build_patch``, so the store
and the patch writer are checked against the code that will read their output.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp.weight_ops.patch import apply_patch
from wsp_worker import bundle as bundle_mod
from wsp_worker import patch_writer
from wsp_worker.bundle import Bundle, Practice, WorkerTensor
from wsp_worker.miwsp_engine.store import Accepted, Refused, TensorStore

NO_LIMITS: dict[str, Any] = {
    "max_changed_entries": None,
    "allowed_tensors": None,
    "frozen_tensors": [],
}


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "b", rule="mean")


def _bundle(fx: FixtureBundle) -> Bundle:
    files = [{k: e[k] for k in ("name", "role", "sha256")} for e in fx.manifest["files"]]
    return bundle_mod.load(fx.root, fx.manifest, files)


def _small(tensors: dict[str, WorkerTensor]) -> Bundle:
    """A bundle of ``tensors`` only, every one editable (no practice data needed)."""
    manifest = {"tensors": [{"name": n, "editable": True} for n in tensors]}
    return Bundle(manifest, tensors, Practice(np.zeros((0, 1)), {}))


def _patch(store: TensorStore, bundle: Bundle, tmp: Path) -> Path | None:
    built = patch_writer.build_patch(
        bundle.tensors,
        store.edited(),
        bundle_base_sha256=bundle.manifest["bundle_sha256"],
        parent={"kind": "bundle", "ref": bundle.manifest["bundle_sha256"], "patch_sha256": None},
        parent_patch_sha256=None,
        created_by={"provider": "miwsp_engine", "provider_version": "0.1.0", "run_id": "r"},
        created_at="2026-11-18T00:00:00Z",
    )
    if built is None:
        return None
    patch_writer.write_patch(tmp / "patch", built[0], built[1])
    return tmp / "patch"


# ----------------------------------------------------------------------------- 2.2


def test_a_create_edit_diff_and_patch_round_trip_is_bit_exact(
    fx: FixtureBundle, tmp_path: Path
) -> None:
    b = _bundle(fx)
    store = TensorStore(b, NO_LIMITS, zero_start=True)
    assert all(np.all(v == 0) for v in store.values.values())  # create's zero start
    assert store.diff() == "no weights changed"  # against the START (zeros), not the bundle
    weight = np.arange(store.values["weight"].size, dtype=np.float32).reshape(
        store.values["weight"].shape
    )
    assert isinstance(store.write("weight", weight), Accepted)
    assert "weight" in store.diff() and "bias" not in store.diff()
    patch = _patch(store, b, tmp_path)
    assert patch is not None
    _manifest, rebuilt = apply_patch(patch, fx.tensors)
    assert np.array_equal(rebuilt["weight"].values64().astype(np.float32), weight)
    assert np.all(rebuilt["bias"].values64() == 0.0)  # zeroed, not written
    assert rebuilt["mean"].sha256 == fx.tensors["mean"].sha256  # not editable: never held
    assert "mean" not in store.values


def test_an_untouched_nan_is_no_diff_and_no_change() -> None:
    values = np.array([1.0, np.nan, 3.0], dtype=np.float32)
    bits = values.view(np.uint32).copy()
    b = _small({"t": WorkerTensor("float32", bits)})
    store = TensorStore(b, NO_LIMITS, zero_start=False)
    assert store.changes() == {"t": 0}
    assert store.diff() == "no weights changed"
    assert isinstance(store.write("t", 5.0, (0,)), Accepted)
    assert store.changes() == {"t": 1}  # the NaN entry is still not a change


def test_bf16_writes_below_one_ulp_are_not_changes() -> None:
    base = patch_writer.float32_to_bf16_bits(np.array([1.0, 2.0, -0.5], dtype=np.float32))
    b = _small({"h": WorkerTensor("bfloat16", base)})
    store = TensorStore(b, {**NO_LIMITS, "max_changed_entries": 0}, zero_start=False)
    assert store.values["h"].dtype == np.float32  # bf16 upcast for the editor
    tiny = np.float32(1.0) + np.float32(2.0**-12)  # well below bf16's ulp at 1.0 (2**-7)
    assert isinstance(store.write("h", tiny, (0,)), Accepted)  # cap 0, and still accepted
    assert store.changes() == {"h": 0} and store.diff() == "no weights changed"
    assert isinstance(store.write("h", 1.5, (0,)), Refused)  # a real change breaks the cap


# ----------------------------------------------------------------------------- 2.4


def test_a_write_to_a_frozen_tensor_is_refused_and_the_store_is_unchanged(
    fx: FixtureBundle,
) -> None:
    store = TensorStore(_bundle(fx), {**NO_LIMITS, "frozen_tensors": ["wei*"]}, zero_start=False)
    before = store.sha256()
    refused = store.write("weight", 9.0, (0,))
    assert isinstance(refused, Refused) and "frozen" in refused.text
    assert store.sha256() == before
    assert isinstance(store.write("bias", 0.25), Accepted)  # another tensor is still editable


def test_a_write_past_the_cap_is_refused_and_the_store_is_unchanged(fx: FixtureBundle) -> None:
    store = TensorStore(_bundle(fx), {**NO_LIMITS, "max_changed_entries": 2}, zero_start=False)
    assert isinstance(store.write("weight", 5.0, (slice(0, 2),)), Accepted)
    before = store.sha256()
    refused = store.write("weight", 7.0, (2,))
    assert isinstance(refused, Refused) and "the limit is 2" in refused.text
    assert store.sha256() == before
    assert isinstance(store.write("weight", 7.0, (1,)), Accepted)  # rewriting a changed entry


def test_a_tensor_outside_allowed_tensors_is_refused(fx: FixtureBundle) -> None:
    store = TensorStore(_bundle(fx), {**NO_LIMITS, "allowed_tensors": ["bias"]}, zero_start=False)
    before = store.sha256()
    assert isinstance(store.write("weight", 1.0, (0,)), Refused)
    assert store.sha256() == before


def test_a_non_finite_write_is_refused(fx: FixtureBundle) -> None:
    store = TensorStore(_bundle(fx), NO_LIMITS, zero_start=False)
    before = store.sha256()
    refused = store.write("weight", np.inf, (0,))
    assert isinstance(refused, Refused) and "non-finite" in refused.text
    assert store.sha256() == before


def test_a_create_that_starts_in_violation_can_move_towards_compliance(fx: FixtureBundle) -> None:
    """Zeroing every editable tensor may already break the cap; a write that does not make it
    worse is accepted, one that adds changes is refused, and submit-time violations remain."""
    b = _bundle(fx)
    limits = {**NO_LIMITS, "max_changed_entries": 1}
    store = TensorStore(b, limits, zero_start=True)
    assert store.violations()  # zero start: every non-zero bundle entry is a change
    original = b.tensors["weight"].values()
    assert isinstance(store.write("weight", original), Accepted)  # fewer changes: accepted
    before = store.sha256()
    assert isinstance(store.write("weight", 3.0, (0,)), Refused)  # more changes: refused
    assert store.sha256() == before


def test_checkpoint_restore_and_revert_work_against_the_start(fx: FixtureBundle) -> None:
    store = TensorStore(_bundle(fx), NO_LIMITS, zero_start=False)
    start = store.sha256()
    store.write("weight", 2.0, (0,))
    store.checkpoint("one")
    edited = store.sha256()
    store.write("weight", 3.0, (1,))
    assert isinstance(store.restore("one"), Accepted) and store.sha256() == edited
    assert isinstance(store.restore("missing"), Refused)
    assert isinstance(store.revert("weight"), Accepted) and store.sha256() == start
