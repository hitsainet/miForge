"""Read a bundle's practice files (FTDD §3.1 ``files[].role``).

Pickles and size mismatches are refused.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.errors import refuse


@dataclass(frozen=True)
class PracticeData:
    activations: np.ndarray  # (tokens, d_model), float16 or float32, memory-mapped read-only
    offsets: np.ndarray  # (rows + 1,), int64
    positions: np.ndarray
    labels: np.ndarray
    validation_mask: np.ndarray
    truncated_mask: np.ndarray


def load_npy(path: Path) -> np.ndarray:
    """``np.load`` with ``allow_pickle=False``: an object array is refused, never unpickled."""
    try:
        loaded: np.ndarray = np.load(Path(path), allow_pickle=False)
        return loaded
    except ValueError as exc:
        raise refuse(
            "wsp_bundle_format_refused",
            f"{Path(path).name} is not a plain numpy array file ({exc}); pickled arrays are "
            "refused.",
        ) from exc


def load_practice(paths_by_role: dict[str, Path], practice: dict[str, Any]) -> PracticeData:
    dtype = {"float16": np.float16, "float32": np.float32}[practice["dtype"]]
    offsets = load_npy(paths_by_role["practice_offsets"]).astype(np.int64)
    rows, tokens, d_model = int(practice["rows"]), int(practice["tokens"]), int(practice["d_model"])
    if (
        offsets.shape != (rows + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != tokens
        or np.any(np.diff(offsets) < 0)
    ):
        raise refuse(
            "wsp_bundle_invalid",
            "practice_offsets must be rows+1 non-decreasing offsets from 0 to practice.tokens.",
        )
    path = Path(paths_by_role["practice_activations"])
    expected = tokens * d_model * np.dtype(dtype).itemsize
    if path.stat().st_size != expected:
        raise refuse(
            "wsp_bundle_invalid",
            f"{path.name} holds {path.stat().st_size} bytes; tokens x d_model x "
            f"{practice['dtype']} "
            f"is {expected}.",
        )
    activations = np.memmap(
        path, dtype=np.dtype(dtype).newbyteorder("<"), mode="r", shape=(tokens, d_model)
    )
    out = {}
    for role in (
        "practice_positions",
        "practice_labels",
        "practice_validation_mask",
        "practice_truncated_mask",
    ):
        arr = load_npy(paths_by_role[role])
        if arr.shape[:1] != (rows,):
            raise refuse("wsp_bundle_invalid", f"{role} has {arr.shape[:1]} rows, not {rows}.")
        out[role] = arr
    return PracticeData(
        activations=activations,
        offsets=offsets,
        positions=out["practice_positions"],
        labels=out["practice_labels"],
        validation_mask=out["practice_validation_mask"],
        truncated_mask=out["practice_truncated_mask"],
    )
