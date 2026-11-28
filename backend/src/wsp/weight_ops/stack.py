"""Chained patches and their composition (FR-016.5, WR-5, P21; [wl] ``patch/stack.py#L16-40``).

A candidate's patch is made against its parent: the bundle, or, **only when** ``request.chain`` is
true,
the parent candidate's edited tensors (:func:`resolve_parent`). ``stack`` composes a sequence of
patches
into one equal patch, and refuses two patches that touch one tensor without being chained
(``wsp_patch_not_chained``): two siblings of one parent, applied in sequence, would apply the second
to a
base it was not made against.
"""

from __future__ import annotations

from itertools import pairwise
from pathlib import Path
from typing import Any

from core.errors import refuse
from wsp.weight_ops.patch import BuiltPatch, apply_patch, build_patch, load_manifest
from wsp.weight_ops.tensor import Tensor


def is_chained(earlier: dict[str, Any], later: dict[str, Any]) -> bool:
    """``later`` was made against ``earlier``'s result."""
    return (
        later["parent"]["kind"] == "candidate"
        and later["parent_patch_sha256"] is not None
        and later["parent_patch_sha256"] == earlier["patch_sha256"]
    )


def check_chain(manifests: list[dict[str, Any]]) -> None:
    """Refuse a sequence whose consecutive patches touch one tensor without being chained."""
    for earlier, later in pairwise(manifests):
        if later["bundle_base_sha256"] != earlier["bundle_base_sha256"]:
            raise refuse(
                "wsp_patch_base_mismatch",
                "These patches start from different bundles, so they cannot be stacked.",
            )
        if is_chained(earlier, later):
            continue
        shared = {t["name"] for t in earlier["tensors"]} & {t["name"] for t in later["tensors"]}
        if shared:
            raise refuse(
                "wsp_patch_not_chained",
                f"Patch {later['patch_sha256'][:12]} edits {sorted(shared)} but was not made "
                f"against patch {earlier['patch_sha256'][:12]}; siblings of one parent cannot be "
                "applied in sequence.",
                tensors=sorted(shared),
            )


def stack(
    base: dict[str, Tensor],
    patch_dirs: list[Path],
    *,
    created_by: dict[str, Any],
    created_at: str,
) -> BuiltPatch | None:
    """One patch equal to applying ``patch_dirs`` in order; ``None`` if they cancel out."""
    if not patch_dirs:
        raise refuse("wsp_patch_malformed", "Nothing to stack.")
    manifests = [load_manifest(p) for p in patch_dirs]
    check_chain(manifests)
    current = dict(base)
    for patch_dir in patch_dirs:
        _, current = apply_patch(patch_dir, current)
    first = manifests[0]
    edited = {name: t for name, t in current.items() if not t.same_bits(base[name])}
    return build_patch(
        base,
        edited,
        bundle_base_sha256=first["bundle_base_sha256"],
        parent=first["parent"],
        parent_patch_sha256=first["parent_patch_sha256"],
        created_by=created_by,
        created_at=created_at,
    )


def resolve_parent(
    *,
    chain: bool,
    request_parent: dict[str, Any],
    bundle_sha256: str,
    candidate_patch_sha256: str | None = None,
) -> tuple[dict[str, Any], str | None]:
    """``(patch parent, parent_patch_sha256)`` for a new patch (P21).

    A request naming a candidate parent **without** ``chain`` makes its patch against the bundle,
    and the
    patch's ``parent.kind`` says so; only ``chain: true`` links the new patch to the candidate's.
    """
    if chain and request_parent["kind"] == "candidate":
        if candidate_patch_sha256 is None:
            raise refuse(
                "wsp_patch_not_chained", "chain is set but the parent candidate has no patch."
            )
        return (
            {
                "kind": "candidate",
                "ref": request_parent["ref"],
                "patch_sha256": candidate_patch_sha256,
            },
            candidate_patch_sha256,
        )
    return {"kind": "bundle", "ref": bundle_sha256, "patch_sha256": None}, None
