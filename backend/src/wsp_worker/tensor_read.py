"""A ``read_tensor`` too large to show (fix 2026-11-22, D128 round 9).

Both providers cut a tool's output at 12,000 characters. A read of the 4,096-entry probe weight
therefore showed positions 0 to 1065, ending mid-number, as if that were the tensor: four of its
five largest weights were in the part never shown. :func:`fit` answers such a read with a summary
of the WHOLE selection instead: its statistics, its largest entries with their positions, and how
to read the values themselves. Both providers use it, so both editors are told the same (D27).

Standard library and numpy only (the worker's fence).
"""

from __future__ import annotations

import numpy as np

#: How many of the largest entries, by magnitude, the summary names.
LARGEST = 20
#: The suggested range fills at most this share of the limit: values differ in printed length.
RANGE_SHARE = 0.85


def fit(
    name: str, index: str | None, values: np.ndarray, text: str, limit: int, precision: int
) -> str:
    """``text``, the rendered read, when it fits ``limit``; else the summary of ``values``."""
    if len(text) <= limit:
        return text
    arr = np.asarray(values)
    flat = arr.reshape(-1).astype(np.float64)
    finite = np.flatnonzero(np.isfinite(flat))
    lines = [
        f"{name}[{index or ':'}] shape={arr.shape}",
        f"{flat.size:,} values are too many to show ({len(text):,} characters; the limit is "
        f"{limit:,}), so this is a summary of all of them, not the values.",
    ]

    def num(x: float) -> str:
        return f"{float(x):.{precision}g}"

    if finite.size:
        kept = flat[finite]
        size = np.abs(kept)
        lines.append(
            f"mean={num(kept.mean())} std={num(kept.std())} min={num(kept.min())} "
            f"max={num(kept.max())} median|value|={num(float(np.median(size)))}"
        )
        largest = finite[np.argsort(-size, kind="stable")[:LARGEST]]
        where = f" (positions count within [{index}])" if index else ""
        entries = []
        for i in largest:
            at = ",".join(str(int(k)) for k in np.unravel_index(int(i), arr.shape))
            entries.append(f"[{at}]={num(flat[i])}")
        lines.append(f"the {len(entries)} largest by magnitude{where}: " + " ".join(entries))
    if finite.size != flat.size:
        lines.append(f"non-finite values: {flat.size - finite.size:,}")
    room = max(1, int(flat.size * RANGE_SHARE * limit / len(text)))
    example = f", e.g. index='0:{room}'" if arr.ndim == 1 and not index else ""
    lines.append(
        f"To see values, read a smaller range (about {room:,} values fit in one read{example}) "
        "or compute over all of them with python."
    )
    return "\n".join(lines)
