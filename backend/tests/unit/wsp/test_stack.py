"""FR-016.5: chained patches stack bit-equal.

Siblings on one tensor are refused; chain: false patches
against the bundle."""

from __future__ import annotations

import numpy as np
import pytest

from core.errors import Refusal
from tests.support.wsp_fixtures import HEX_PARENT, patch_case
from wsp.weight_ops.patch import apply_patch, build_patch, write_patch
from wsp.weight_ops.stack import resolve_parent, stack
from wsp.weight_ops.tensor import from_values

WHO = {"provider": "weightspace", "provider_version": "0.3.0", "run_id": "wsp_0123456789ab"}
WHEN = "2026-11-16T05:00:00Z"


def _two(tmp_path, *, chained: bool):
    base, edited1, kwargs = patch_case("float32", "sparse", seed=31)
    p1 = build_patch(base, edited1, **kwargs)
    write_patch(tmp_path / "p1", p1)
    after1 = {**base, **edited1}
    v = after1["weight"].values64().copy()
    v[5] += 2.0
    v[7] -= 1.0
    edited2 = {"weight": from_values("float32", v.astype(np.float32))}
    if chained:
        parent, link = resolve_parent(
            chain=True,
            request_parent={"kind": "candidate", "ref": "wspc_000000000001"},
            bundle_sha256=HEX_PARENT,
            candidate_patch_sha256=p1.manifest["patch_sha256"],
        )
        p2 = build_patch(
            after1,
            edited2,
            bundle_base_sha256=HEX_PARENT,
            parent=parent,
            parent_patch_sha256=link,
            created_by=WHO,
            created_at=WHEN,
        )
    else:
        sibling = {
            "weight": from_values("float32", (base["weight"].values64() + 0.25).astype(np.float32))
        }
        p2 = build_patch(base, sibling, **kwargs)
    write_patch(tmp_path / "p2", p2)
    return base, tmp_path / "p1", tmp_path / "p2"


def test_a_chain_of_two_stacks_bit_equal_to_applying_both(tmp_path) -> None:
    base, p1, p2 = _two(tmp_path, chained=True)
    _, step1 = apply_patch(p1, base)
    _, step2 = apply_patch(p2, step1)
    stacked = stack(base, [p1, p2], created_by=WHO, created_at=WHEN)
    write_patch(tmp_path / "s", stacked)
    _, once = apply_patch(tmp_path / "s", base)
    assert np.array_equal(once["weight"].bits, step2["weight"].bits)
    assert stacked.manifest["parent"]["kind"] == "bundle"


def test_two_sibling_patches_on_one_tensor_are_refused(tmp_path) -> None:
    base, p1, p2 = _two(tmp_path, chained=False)
    with pytest.raises(Refusal) as err:
        stack(base, [p1, p2], created_by=WHO, created_at=WHEN)
    assert err.value.code == "wsp_patch_not_chained"
    assert err.value.details["tensors"] == ["weight"]


def test_chain_false_against_a_candidate_patches_against_the_bundle() -> None:
    parent, link = resolve_parent(
        chain=False,
        request_parent={"kind": "candidate", "ref": "wspc_000000000001"},
        bundle_sha256=HEX_PARENT,
        candidate_patch_sha256="d" * 64,
    )
    assert parent == {"kind": "bundle", "ref": HEX_PARENT, "patch_sha256": None} and link is None


def test_chain_true_links_to_the_candidate() -> None:
    parent, link = resolve_parent(
        chain=True,
        request_parent={"kind": "candidate", "ref": "wspc_000000000001"},
        bundle_sha256=HEX_PARENT,
        candidate_patch_sha256="d" * 64,
    )
    assert parent["kind"] == "candidate" and link == "d" * 64
