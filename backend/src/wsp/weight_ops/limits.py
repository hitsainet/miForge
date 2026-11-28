"""Change limits: one pure function, ``check_limits`` (FR-016.7, WR-6, P8; PF-40).

The worker's courtesy check and the backend's authoritative check are **the same function**: the
worker
image carries a byte-identical copy (``wsp_worker/limits.py``), pinned by
``test_limits_copy_pinned.py``.
So this module imports **only the standard library**, and nothing else from miForge.

``limits`` is ``request.change_limits`` (``miforge-wsp-run-v1.json`` ``ChangeLimits``):
``max_changed_entries`` (null: no cap), ``allowed_tensors`` (null: any tensor), ``frozen_tensors``.
Tensor patterns are ``fnmatch`` patterns, matched case-sensitively on every platform.
"""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatchcase
from typing import Any

KINDS = ("max_changed_entries", "frozen_tensor", "tensor_not_allowed")


@dataclass(frozen=True)
class Violation:
    kind: str
    tensor: str | None
    detail: str


def _matches(name: str, patterns: list[str]) -> bool:
    return any(fnmatchcase(name, pattern) for pattern in patterns)


def check_limits(patch_manifest: dict[str, Any], limits: dict[str, Any]) -> list[Violation]:
    """Every violation of ``limits`` by the patch, in a stable order; an empty list passes."""
    violations: list[Violation] = []
    cap = limits.get("max_changed_entries")
    total = int(patch_manifest.get("changed_entries_total", 0))
    if cap is not None and total > int(cap):
        violations.append(
            Violation(
                "max_changed_entries",
                None,
                f"the patch changes {total} entries; the limit is {cap}",
            )
        )
    frozen = list(limits.get("frozen_tensors") or [])
    allowed = limits.get("allowed_tensors")
    for tensor in patch_manifest.get("tensors", []):
        name = tensor["name"]
        if int(tensor.get("changed_entries", 0)) == 0 and tensor.get("append") is None:
            continue
        if _matches(name, frozen):
            violations.append(Violation("frozen_tensor", name, f"tensor {name!r} is frozen"))
        if allowed is not None and not _matches(name, list(allowed)):
            violations.append(
                Violation("tensor_not_allowed", name, f"tensor {name!r} is not in allowed_tensors")
            )
    return violations
