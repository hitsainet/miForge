"""The tensor store: the editor's working values, compared bitwise, with the change limits checked
at EVERY write (feature 018; FR-018.3, FR-018.4; FTDD §6.2; FTID §3.1).

- **Values** are held for the bundle's editable tensors only, as the editor sees them
  (``WorkerTensor.values()``: bf16 upcast to float32, the other float dtypes as stored).
- **The start** is the bundle's values, or zeros in a ``create`` (as the ``weightspace`` adapter's
  ``zero_()`` + ``commit()``). ``diff``, ``checkpoint``, ``restore`` and ``revert`` work against it.
- **Every comparison is the patch writer's:** a tensor's bits are
  ``patch_writer.from_values(dtype, values)`` compared with ``!=`` on the raw bits. So an untouched
  NaN is no change, a bf16 write below one ulp is no change, and the counts the limits see are
  exactly the counts the final patch will carry.
- **:meth:`TensorStore.try_apply`** applies a write to scratch copies, counts the changes against
  the bundle's bits, calls ``limits.check_limits`` (the byte-pinned copy, never edited here), and
  commits only when nothing is violated. A refused write leaves the store unchanged.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from wsp_worker import limits as limits_mod
from wsp_worker import patch_writer
from wsp_worker.bundle import Bundle, WorkerTensor


@dataclass(frozen=True)
class Accepted:
    changed: dict[str, int]


@dataclass(frozen=True)
class Refused:
    details: list[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return "refused: " + "; ".join(self.details) + ". Nothing was changed."


def _bits(dtype: str, values: np.ndarray) -> np.ndarray:
    return patch_writer.from_values(dtype, values).bits.reshape(-1)


class TensorStore:
    """The editable tensors' working values over a bundle, with checkpoints and limits."""

    def __init__(
        self, bundle: Bundle, change_limits: Mapping[str, Any], *, zero_start: bool
    ) -> None:
        self.base: dict[str, WorkerTensor] = dict(bundle.tensors)
        self.limits = dict(change_limits)
        self.dtypes = {n: t.dtype for n, t in bundle.tensors.items() if bundle.editable(n)}
        self.values: dict[str, np.ndarray] = {
            n: np.array(bundle.tensors[n].values(), copy=True) for n in self.dtypes
        }
        if zero_start:
            for v in self.values.values():
                v[...] = 0
        self.start = {n: v.copy() for n, v in self.values.items()}
        self._start_bits = {n: _bits(self.dtypes[n], v) for n, v in self.start.items()}
        self._base_bits = {n: self.base[n].bits.reshape(-1) for n in self.dtypes}
        self.checkpoints: dict[str, dict[str, np.ndarray]] = {}

    # ----------------------------------------------------------------------------- reading

    def names(self) -> list[str]:
        return list(self.values)

    def sha256(self) -> str:
        """A digest of every value's bytes, in name order (tests compare it across a refusal)."""
        h = hashlib.sha256()
        for name in sorted(self.values):
            h.update(name.encode())
            h.update(np.ascontiguousarray(self.values[name]).tobytes())
        return h.hexdigest()

    def edited(self) -> dict[str, WorkerTensor]:
        """Every editable tensor, cast back to its stored dtype: the patch writer's input."""
        return {n: patch_writer.from_values(self.dtypes[n], v) for n, v in self.values.items()}

    def changes(self, values: Mapping[str, np.ndarray] | None = None) -> dict[str, int]:
        """Entries whose bits differ from the BUNDLE's, per tensor (the patch's own count)."""
        current = self.values if values is None else values
        return {
            n: int(np.count_nonzero(_bits(self.dtypes[n], v) != self._base_bits[n]))
            for n, v in current.items()
        }

    def _check(self, values: Mapping[str, np.ndarray]) -> tuple[int, list[limits_mod.Violation]]:
        """``limits.check_limits`` on the minimal manifest of ``values`` against the bundle:
        ``{"changed_entries_total", "tensors": [{"name", "changed_entries", "append"}]}``."""
        counts = self.changes(values)
        total = sum(counts.values())
        manifest = {
            "changed_entries_total": total,
            "tensors": [
                {"name": n, "changed_entries": c, "append": None} for n, c in counts.items()
            ],
        }
        return total, limits_mod.check_limits(manifest, self.limits)

    def violations(self, values: Mapping[str, np.ndarray] | None = None) -> list[str]:
        """Every limit the store (or ``values``) breaks, as text; empty when the edit passes."""
        return [v.detail for v in self._check(self.values if values is None else values)[1]]

    def _changed(self) -> list[tuple[str, int, int, float, float]]:
        """``(name, changed, size, |delta|, max)`` per tensor that differs from the start,
        bitwise, in name order of the store."""
        out = []
        for name, v in self.values.items():
            bits = _bits(self.dtypes[name], v)
            changed = int(np.count_nonzero(bits != self._start_bits[name]))
            if changed:
                delta = v.astype(np.float64) - self.start[name].astype(np.float64)
                finite = delta[np.isfinite(delta)]
                biggest = float(np.max(np.abs(finite))) if finite.size else float("nan")
                out.append((name, changed, int(v.size), float(np.linalg.norm(finite)), biggest))
        return out

    def diff(self) -> str:
        """What changed against the start, bitwise (the editor's ``diff`` tool)."""
        total = sum(v.size for v in self.values.values())
        changed = self._changed()
        if not changed:
            return "no weights changed"
        lines = [
            f"  {name}: {n:,}/{size:,} changed, |delta|={norm:.4g}, max={biggest:.4g}"
            for name, n, size, norm, biggest in changed
        ]
        changed_total = sum(c[1] for c in changed)
        head = f"{changed_total:,} of {total:,} weights changed in {len(lines)} tensor(s)"
        return "\n".join([head, *lines])

    def diff_summary(self) -> tuple[str, list[str]]:
        """The diff in one line for the ledger (paging, 2026-11-21): ``"bias 1/1 |d|=0.3"``
        per changed tensor, ``"none"`` when the weights are the start's; and the names."""
        changed = self._changed()
        if not changed:
            return "none", []
        text = "; ".join(f"{name} {n}/{size} |d|={norm:.4g}" for name, n, size, norm, _ in changed)
        return text, [c[0] for c in changed]

    # ----------------------------------------------------------------------------- writing

    def try_apply(self, updates: Mapping[str, np.ndarray]) -> Accepted | Refused:
        """Apply whole-tensor ``updates`` (new values) if, and only if, no limit is violated.

        The updates are checked on scratch copies first (FR-018.4): an unknown or read-only
        tensor, a wrong shape, a non-finite value the patch could not carry, or any change limit
        refuses the WHOLE write and leaves every value as it was (see :meth:`_breaks`)."""
        scratch: dict[str, np.ndarray] = {}
        for name, value in updates.items():
            if name not in self.values:
                return Refused([f"tensor {name!r} is not editable here"])
            target = self.values[name]
            arr = np.asarray(value)
            if arr.shape != target.shape:
                return Refused([f"tensor {name!r} has shape {list(target.shape)}"])
            new = arr.astype(target.dtype, copy=True)
            changed = _bits(self.dtypes[name], new) != self._base_bits[name]
            if not np.all(np.isfinite(new.reshape(-1)[changed])):
                return Refused([f"tensor {name!r} would hold a non-finite value"])
            scratch[name] = new
        if self._breaks({**self.values, **scratch}):
            return Refused(self.violations({**self.values, **scratch}))
        for name, new in scratch.items():
            self.values[name][...] = new
        return Accepted({n: c for n, c in self.changes().items() if n in scratch})

    def _breaks(self, after: Mapping[str, np.ndarray]) -> bool:
        """Whether moving to ``after`` breaks a limit: a violation the store does not already
        have, or a larger changed-entry total past the cap. A store that starts in violation (a
        ``create`` zeroing a frozen tensor, or past the cap) can still move towards compliance,
        and ``submit`` refuses until it gets there."""
        before_total, before = self._check(self.values)
        after_total, broken = self._check(after)
        known = {(v.kind, v.tensor) for v in before}
        if any((v.kind, v.tensor) not in known for v in broken):
            return True
        capped = any(v.kind == "max_changed_entries" for v in broken)
        return capped and after_total > before_total

    def write(self, name: str, values: Any, index: Any = ()) -> Accepted | Refused:
        """The ``write_tensor`` tool: ``values`` into ``name[index]``, through :meth:`try_apply`."""
        if name not in self.values:
            return Refused([f"tensor {name!r} is not editable here"])
        new = self.values[name].copy()
        new[index] = np.asarray(values, dtype=new.dtype)
        return self.try_apply({name: new})

    def checkpoint(self, label: str) -> None:
        self.checkpoints[str(label)] = {n: v.copy() for n, v in self.values.items()}

    def restore(self, label: str) -> Accepted | Refused:
        saved = self.checkpoints.get(str(label))
        if saved is None:
            return Refused([f"no checkpoint {label!r}; saved: {sorted(self.checkpoints)}"])
        return self.try_apply(saved)

    def revert(self, name: str | None = None) -> Accepted | Refused:
        """Back to the start (the bundle, or zeros in a create), one tensor or all."""
        if name is not None and name not in self.values:
            return Refused([f"tensor {name!r} is not editable here"])
        names = [name] if name is not None else list(self.values)
        return self.try_apply({n: self.start[n] for n in names})
