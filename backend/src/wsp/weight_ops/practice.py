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


# ------------------------------------------------------------------ validation-only supplements

#: The five files of ``practice.supplements`` (revision 2026-11-25, FU-7a C2): every supplement's
#: rows, one after another in the manifest's order. Present exactly when ``supplements`` is.
SUPPLEMENT_ROLES = (
    "supplement_activations",
    "supplement_offsets",
    "supplement_positions",
    "supplement_labels",
    "supplement_truncated_mask",
)


@dataclass(frozen=True)
class SupplementData:
    """Every supplement's rows, concatenated, and the manifest's entries that slice them."""

    activations: np.ndarray  # (tokens, d_model), practice.dtype, memory-mapped read-only
    offsets: np.ndarray  # (rows + 1,), int64, over the concatenated tokens
    positions: np.ndarray
    labels: np.ndarray
    truncated_mask: np.ndarray
    entries: tuple[dict[str, Any], ...]

    def rows_of(self, index: int) -> np.ndarray:
        """The concatenated row ids of supplement ``index``."""
        entry = self.entries[index]
        first = int(entry["first_row"])
        return np.arange(first, first + int(entry["rows"]), dtype=np.int64)


def _plain(path: Path, role: str) -> np.ndarray:
    """A plain 1-D ``.npy`` array; an ``.npz`` named ``.npy`` or an object array is refused."""
    if Path(path).suffix != ".npy":
        raise refuse("wsp_bundle_format_refused", f"{role} is a .npy array.")
    arr: Any = load_npy(path)
    if not isinstance(arr, np.ndarray):
        close = getattr(arr, "close", None)
        if close is not None:
            close()
        raise refuse("wsp_bundle_format_refused", f"{role} is not a single .npy array.")
    if arr.dtype == object or arr.ndim != 1:
        raise refuse("wsp_bundle_format_refused", f"{role} is not a plain 1-D .npy array.")
    return arr


def load_supplements(
    paths_by_role: dict[str, Path], manifest: dict[str, Any]
) -> SupplementData | None:
    """``practice.supplements`` and its five files, every count and slice checked; ``None`` when
    the bundle has none. Refused ``wsp_bundle_invalid`` when the files and the entries disagree,
    and when a supplement names the practice view or a held-out view (a held-out row is never
    practice, FR-016.1). The row-id digests and parity are checked by ``wsp.bundles``."""
    practice = manifest["practice"]
    entries = practice.get("supplements")
    present = [r for r in SUPPLEMENT_ROLES if r in paths_by_role]
    if not entries:
        if present:
            raise refuse(
                "wsp_bundle_invalid",
                f"The bundle has supplement files ({', '.join(present)}) but no "
                "practice.supplements.",
            )
        return None
    missing = [r for r in SUPPLEMENT_ROLES if r not in paths_by_role]
    if missing:
        raise refuse(
            "wsp_bundle_invalid",
            f"practice.supplements needs the files {', '.join(missing)}.",
        )
    heldout = {s["view_id"] for s in manifest["heldout_ref"]["sets"]}
    heldout_rows = {s["row_ids_sha256"] for s in manifest["heldout_ref"]["sets"]}
    seen: set[str] = set()
    expected_first = 0
    for entry in entries:
        view = str(entry["view_id"])
        if (
            view == practice["view_id"]
            or view in heldout
            or entry["row_ids_sha256"] in heldout_rows
        ):
            raise refuse(
                "wsp_bundle_invalid",
                f"Supplement {view} is the practice view or a held-out set; a held-out row is "
                "never practice.",
                view_id=view,
            )
        if view in seen:
            raise refuse("wsp_bundle_invalid", f"Supplement {view} is listed twice.", view_id=view)
        seen.add(view)
        if int(entry["first_row"]) != expected_first:
            raise refuse(
                "wsp_bundle_invalid",
                f"Supplement {view} starts at row {entry['first_row']}, not {expected_first}: the "
                "supplements follow one another in their listed order.",
            )
        expected_first += int(entry["rows"])
    rows = expected_first
    tokens = sum(int(e["tokens"]) for e in entries)
    dtype = {"float16": np.float16, "float32": np.float32}[practice["dtype"]]
    d_model = int(practice["d_model"])
    offsets = _plain(paths_by_role["supplement_offsets"], "supplement_offsets")
    if (
        offsets.dtype.kind not in "iu"
        or offsets.shape != (rows + 1,)
        or int(offsets[0]) != 0
        or int(offsets[-1]) != tokens
        or bool(np.any(np.diff(offsets.astype(np.int64)) < 0))
    ):
        raise refuse(
            "wsp_bundle_invalid",
            "supplement_offsets must be rows+1 non-decreasing integer offsets from 0 to the "
            "supplements' tokens.",
        )
    offsets = offsets.astype(np.int64)
    path = Path(paths_by_role["supplement_activations"])
    expected = tokens * d_model * np.dtype(dtype).itemsize
    if path.suffix != {"float16": ".f16", "float32": ".f32"}[practice["dtype"]]:
        raise refuse(
            "wsp_bundle_format_refused",
            f"The supplement activations are raw {practice['dtype']}, like practice's.",
        )
    if path.stat().st_size != expected:
        raise refuse(
            "wsp_bundle_invalid",
            f"{path.name} holds {path.stat().st_size} bytes; the supplements' tokens x d_model x "
            f"{practice['dtype']} is {expected}.",
        )
    arrays = {
        role: _plain(paths_by_role[role], role)
        for role in ("supplement_positions", "supplement_labels", "supplement_truncated_mask")
    }
    for role, arr in arrays.items():
        if arr.shape != (rows,) or arr.dtype.kind not in "iub":
            raise refuse("wsp_bundle_invalid", f"{role} has {arr.shape} entries, not ({rows},).")
    labels = arrays["supplement_labels"].astype(np.int64)
    truncated = arrays["supplement_truncated_mask"].astype(bool)
    if bool(np.any((labels != 0) & (labels != 1))):
        raise refuse("wsp_bundle_invalid", "supplement_labels holds a value other than 0 and 1.")
    for entry in entries:
        first, n = int(entry["first_row"]), int(entry["rows"])
        part = slice(first, first + n)
        counts = {"positive": int(labels[part].sum()), "negative": int(n - labels[part].sum())}
        if (
            int(offsets[first + n] - offsets[first]) != int(entry["tokens"])
            or counts != dict(entry["label_counts"])
            or int(truncated[part].sum()) != int(entry["truncated_rows"])
        ):
            raise refuse(
                "wsp_bundle_invalid",
                f"Supplement {entry['view_id']}: its tokens, label counts or truncated rows "
                "disagree with the supplement files.",
                view_id=entry["view_id"],
            )
    activations = np.memmap(
        path, dtype=np.dtype(dtype).newbyteorder("<"), mode="r", shape=(tokens, d_model)
    )
    return SupplementData(
        activations=activations,
        offsets=offsets,
        positions=arrays["supplement_positions"].astype(np.int64),
        labels=labels,
        truncated_mask=truncated,
        entries=tuple(dict(e) for e in entries),
    )
