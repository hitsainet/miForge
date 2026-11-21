"""The random control (feature 016; FR-016.27; FTDD §6.4; WR-15, P54, R1B-A1, R1B-A8).

For each tensor the patch changes, in tensor-NAME order, the flattened float64 delta is permuted
with a seeded generator and multiplied by random signs::

    rng = np.random.default_rng(int(content_sha256[:8], 16) + n)   # n = 0: control; n: control:<n>
    for name in sorted(patched):
        d = (float64(edited) - float64(base)).reshape(-1)
        c = float64(base).reshape(-1) + rng.permutation(d) * rng.choice([-1.0, 1.0], size=d.size)
        control[name] = c.reshape(shape).astype(base.dtype)

This follows wsplab's ``random_control`` ([wl] ``eval/gate.py#L47-58``) with the two stated
differences: float64 in name order (wsplab: float32 in patch order) and the seed from
``content_sha256`` (R1B-A1), so a replayed patch gets the same control. Before the cast the
control has the candidate's tensors, changed count and delta L2 norm exactly; after the cast to
the base dtype, rounding can move both, so :class:`Control` carries the cast control's ACTUAL
figures beside the candidate's (R1B-A8), for the verdict to record.

Pure: numpy only, no I/O.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from wsp.weight_ops.tensor import Tensor, from_values

SEED_HEX_DIGITS = 8


def control_seed(content_sha256: str, n: int = 0) -> int:
    """``int(content_sha256[:8], 16) + n``: the value miWorkbench pins (P56, P63)."""
    if len(content_sha256) != 64 or n < 0:
        raise ValueError("a control seed needs a 64-hex content_sha256 and n >= 0")
    return int(content_sha256[:SEED_HEX_DIGITS], 16) + n


@dataclass(frozen=True)
class Figures:
    changed_entries: int
    l2_delta: float


@dataclass(frozen=True)
class Control:
    tensors: dict[str, Tensor]
    seed: int
    #: The candidate's own figures (from base and edited, exactly).
    candidate: Figures
    #: The control's figures before the cast: equal to ``candidate`` by construction.
    pre_cast: Figures
    #: The cast control's actual figures, which rounding may move (R1B-A8).
    post_cast: Figures


def _figures(deltas: list[np.ndarray]) -> Figures:
    flat = [float(x) for d in deltas for x in d.reshape(-1)]
    return Figures(
        changed_entries=sum(int(np.count_nonzero(d)) for d in deltas),
        l2_delta=math.sqrt(math.fsum(x * x for x in flat)),
    )


def random_control(
    base: dict[str, Tensor],
    edited: dict[str, Tensor],
    patched: list[str],
    content_sha256: str,
    n: int = 0,
) -> Control:
    """The control for one patch: ``patched`` names the tensors the patch changes."""
    seed = control_seed(content_sha256, n)
    rng = np.random.default_rng(seed)
    tensors: dict[str, Tensor] = {}
    candidate, pre, post = [], [], []
    for name in sorted(patched):
        b = base[name].values64()
        d = (edited[name].values64() - b).reshape(-1)
        moved = rng.permutation(d) * rng.choice([-1.0, 1.0], size=d.size)
        c = b.reshape(-1) + moved
        cast = from_values(base[name].dtype, c.reshape(b.shape))
        tensors[name] = cast
        candidate.append(d)
        pre.append(moved)
        post.append(cast.values64().reshape(-1) - b.reshape(-1))
    return Control(tensors, seed, _figures(candidate), _figures(pre), _figures(post))
