"""``miforge.wsp-patch/v1``: write and apply weight patches (FR-016.4; FTDD §3.2; FTID §3.1).

A patch folder holds ``patch.json`` and ``edit.npz``. The array layout is wsplab's ([wl]
``patch/format.py#L7-12``), indexed by the tensor's position ``i`` in ``tensors[]``:

- ``sparse``: ``t<i>_idx`` (int64 flat indices, strictly increasing) and ``t<i>_val`` (the new
  values'
  **raw bits**, ``uint16``/``uint32``/``uint64``): ``e = base_bits.reshape(-1); e[idx] = val``.
- ``lowrank``: ``t<i>_a`` (rows x r) and ``t<i>_b`` (r x cols), float64; ``w = float64(base)``, then
  **one
  rank-one term at a time** ``w = w + outer(a[:, k], b[k])``; cast to float32 (float64 for a float64
  tensor) and then to the tensor's dtype bits; then the ``t<i>_idx``/``t<i>_val`` fix-ups as for
  sparse.
  A fused ``a @ b`` rounds differently and is wrong.
- ``dense``: ``t<i>_bits`` is the **full edited tensor's bits**, a replacement, not a delta.

**Apply order** ([wl] ``patch/apply.py#L47-68``): the manifest's own digests, the archive's digest
and
bound, every base hash, then the rebuild, then every edited hash. Apply returns tensors in memory;
it
writes nothing, so nothing is written before every tensor has passed.

``edit.npz`` is written deterministically (stored, fixed timestamps, sorted members, npy format
1.0), so
a regenerated patch is byte-identical (PF-55), and it is always read with ``allow_pickle=False``.
"""

from __future__ import annotations

import io
import json
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.atomic_dir import atomic_dir
from core.errors import Refusal, refuse
from wsp.schemas import validate_patch
from wsp.weight_ops.hashing import bytes_sha256, jcs_sha256
from wsp.weight_ops.tensor import BITS, Tensor, from_values

PATCH_JSON = "patch.json"
ARRAYS_FILE = "edit.npz"
# : Sparse when ``changed * 16 < numel * itemsize`` (FTID §3.1): 8 bytes of index plus the value per
# entry.
SPARSE_FACTOR = 16
#: The npy header allowance per archive member, in bytes, for the size bound.
NPY_HEADER_ALLOWANCE = 1024
_MEMBER = re.compile(r"^t(\d{1,6})_(idx|val|a|b|bits)\.npy$")
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)


def _malformed(message: str, **details: Any) -> Exception:
    return refuse("wsp_patch_malformed", message, **details)


def _check_npy_size(raw: bytes, name: str) -> None:
    """Refuse a ``.npy`` whose header declares more (or less) data than it holds, BEFORE numpy
    allocates the array the header declares (review OR3-016-1: a 100-byte member declaring a
    64 GiB shape made ``read_array`` try to allocate 64 GiB). Plain dtypes only."""
    fp = io.BytesIO(raw)
    version = np.lib.format.read_magic(fp)
    if version == (1, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_1_0(fp)
    elif version == (2, 0):
        shape, _fortran, dtype = np.lib.format.read_array_header_2_0(fp)
    else:
        raise _malformed(f"{name} uses .npy format version {version}; 1.0 and 2.0 are read.")
    if dtype.hasobject:
        raise _malformed(f"{name} is not a plain numpy array: it holds Python objects.")
    declared = math.prod(int(n) for n in shape) * dtype.itemsize
    held = len(raw) - fp.tell()
    if declared != held:
        raise _malformed(
            f"{name} declares {declared} bytes of array data but holds {held}.",
            member=name,
        )


def _why(exc: BaseException) -> str:
    """A parser's failure as one short line: its type and the start of its message."""
    text = str(exc).splitlines()[0] if str(exc) else ""
    return f"{type(exc).__name__}: {text[:160]}" if text else type(exc).__name__


# --------------------------------------------------------------------------------------
# Rebuild primitives (shared by build and apply, so the two cannot disagree)
# --------------------------------------------------------------------------------------


def lowrank_reconstruct(base: Tensor, a: np.ndarray, b: np.ndarray) -> Tensor:
    """``float64(base) + Σ_k outer(a[:, k], b[k])``, one term at a time, cast to its dtype."""
    w = base.values64().copy()
    for k in range(a.shape[1]):
        w = w + np.outer(a[:, k], b[k])
    if base.dtype == "float64":
        return from_values("float64", w)
    return from_values(base.dtype, w.astype(np.float32))


def _apply_sparse(bits: np.ndarray, idx: np.ndarray, val: np.ndarray) -> np.ndarray:
    flat = np.array(bits, copy=True).reshape(-1)
    flat[idx] = val
    return flat


def _stats(base: Tensor, edited: Tensor) -> tuple[int, float, float]:
    changed = int(np.count_nonzero(base.bits.reshape(-1) != edited.bits.reshape(-1)))
    delta = edited.values64() - base.values64()
    if not np.isfinite(delta).all():
        raise _malformed(
            "The edit produces a non-finite value; a patch never writes NaN or infinity."
        )
    # math.fsum is exactly rounded, so l2_delta is the same bits on every CPU; a BLAS dot's
    # summation
    # order is not, and l2_delta enters patch_sha256 (suite memory
    # goldens-need-cross-cpu-tolerance).
    l2 = math.sqrt(math.fsum((delta.reshape(-1) ** 2).tolist()))
    return changed, l2, float(np.abs(delta).max(initial=0.0))


# --------------------------------------------------------------------------------------
# The archive
# --------------------------------------------------------------------------------------


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Deterministic ``.npz`` bytes.

    Stored members, sorted, epoch timestamps, npy format 1.0, little-endian.
    """
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            arr = np.ascontiguousarray(arrays[name])
            arr = arr.astype(arr.dtype.newbyteorder("<"))
            body = io.BytesIO()
            np.lib.format.write_array(body, arr, version=(1, 0), allow_pickle=False)
            info = zipfile.ZipInfo(f"{name}.npy", date_time=_ZIP_EPOCH)
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o644 << 16
            archive.writestr(info, body.getvalue())
    return buffer.getvalue()


def archive_bound(manifest: dict[str, Any]) -> int:
    """The most uncompressed bytes the archive can need for its tensors ([wl] format.py)."""
    total = 0
    for pt in manifest["tensors"]:
        width = np.dtype(BITS[pt["dtype"]]).itemsize
        numel = int(np.prod(pt["edited_shape"], dtype=np.int64)) if pt["edited_shape"] else 1
        if pt["form"] == "dense":
            total += numel * width + NPY_HEADER_ALLOWANCE
        else:
            total += numel * (8 + width) + 2 * NPY_HEADER_ALLOWANCE
            if pt["form"] == "lowrank":
                rank = int(pt["rank"] or 0)
                rows, cols = (pt["shape"] + [1, 1])[:2] if len(pt["shape"]) == 2 else (numel, 1)
                total += (rows * rank + rank * cols) * 8 + 2 * NPY_HEADER_ALLOWANCE
    return total


def read_archive(data: bytes, manifest: dict[str, Any]) -> dict[str, np.ndarray]:
    """The archive's arrays, after checking names and bounding the size before reading."""
    expected: set[str] = set()
    for i, pt in enumerate(manifest["tensors"]):
        if pt["form"] == "sparse":
            expected |= {f"t{i}_idx", f"t{i}_val"}
        elif pt["form"] == "lowrank":
            expected |= {f"t{i}_idx", f"t{i}_val", f"t{i}_a", f"t{i}_b"}
        else:
            expected.add(f"t{i}_bits")
    bound = archive_bound(manifest)
    # R3 review fix (OR3-016-1), 2026-11-16: the archive is hostile input (the worker is untrusted
    # and program mode runs LLM-written code there). Every way a parser can fail on its bytes - a
    # bad CRC-32 (BadZipFile), a corrupt deflate stream (zlib.error), the encrypted flag
    # (RuntimeError), an unsupported method such as deflate64 (NotImplementedError), a truncated
    # member (EOFError), or anything else zipfile or numpy raises - is wsp_patch_malformed, never an
    # exception that escapes the intake.
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except Exception as exc:
        raise _malformed(f"{ARRAYS_FILE} is not a readable zip archive: {_why(exc)}") from exc
    with archive:
        infos = archive.infolist()
        names = []
        for info in infos:
            match = _MEMBER.match(info.filename)
            if match is None or info.is_dir():
                raise _malformed(f"{ARRAYS_FILE} holds an unexpected member {info.filename!r}.")
            names.append(info.filename[:-4])
        if len(names) != len(set(names)) or set(names) != expected:
            raise _malformed(
                f"{ARRAYS_FILE} members {sorted(names)} are not the ones the manifest needs "
                f"{sorted(expected)}."
            )
        declared = sum(info.file_size for info in infos)
        if declared > bound:
            raise _malformed(
                f"{ARRAYS_FILE} declares {declared} uncompressed bytes; its tensors can need at "
                f"most {bound}.",
                declared=declared,
                bound=bound,
            )
        out: dict[str, np.ndarray] = {}
        remaining = bound
        for info in infos:
            try:
                with archive.open(info) as member:
                    raw = member.read(remaining + 1)
            except Exception as exc:
                raise _malformed(
                    f"{ARRAYS_FILE} member {info.filename} cannot be read: {_why(exc)}",
                    member=info.filename,
                ) from exc
            if len(raw) > remaining or len(raw) != info.file_size:
                raise _malformed(
                    f"{ARRAYS_FILE} member {info.filename} expands beyond its declared size."
                )
            remaining -= len(raw)
            try:
                _check_npy_size(raw, info.filename)
                arr = np.lib.format.read_array(io.BytesIO(raw), allow_pickle=False)
            except Refusal:
                raise
            except Exception as exc:
                raise _malformed(
                    f"{info.filename} is not a plain numpy array: {_why(exc)}",
                    member=info.filename,
                ) from exc
            out[info.filename[:-4]] = arr
        return out


# --------------------------------------------------------------------------------------
# Build
# --------------------------------------------------------------------------------------


@dataclass(frozen=True)
class BuiltPatch:
    manifest: dict[str, Any]
    arrays: bytes


def _form_for(
    base: Tensor, edited: Tensor, factors: tuple[np.ndarray, np.ndarray] | None
) -> tuple[str, dict[str, np.ndarray], int | None]:
    """Choose the smallest form; lowrank only when the caller supplies factors that help."""
    width = np.dtype(BITS[base.dtype]).itemsize
    base_flat, edited_flat = base.bits.reshape(-1), edited.bits.reshape(-1)
    idx = np.flatnonzero(base_flat != edited_flat).astype(np.int64)
    sparse_cost = idx.size * (8 + width)
    dense_cost = edited.size * width
    best: tuple[str, dict[str, np.ndarray], int | None, int] = (
        "sparse",
        {"idx": idx, "val": edited_flat[idx]},
        None,
        sparse_cost,
    )
    if not idx.size * SPARSE_FACTOR < edited.size * width:
        best = ("dense", {"bits": edited.bits}, None, dense_cost)
    if factors is not None:
        a, b = (np.ascontiguousarray(f, dtype=np.float64) for f in factors)
        if (
            base.bits.ndim != 2
            or a.ndim != 2
            or b.ndim != 2
            or a.shape[0] != base.shape[0]
            or (b.shape[1] != base.shape[1] or a.shape[1] != b.shape[0] or a.shape[1] < 1)
        ):
            raise _malformed(
                f"Low-rank factors {a.shape} x {b.shape} do not fit a tensor of shape {base.shape}."
            )
        recon = lowrank_reconstruct(base, a, b).bits.reshape(-1)
        fix = np.flatnonzero(recon != edited_flat).astype(np.int64)
        cost = (a.size + b.size) * 8 + fix.size * (8 + width)
        if cost < best[3]:
            best = (
                "lowrank",
                {"a": a, "b": b, "idx": fix, "val": edited_flat[fix]},
                int(a.shape[1]),
                cost,
            )
    return best[0], best[1], best[2]


def build_patch(
    base: dict[str, Tensor],
    edited: dict[str, Tensor],
    *,
    bundle_base_sha256: str,
    parent: dict[str, Any],
    parent_patch_sha256: str | None,
    created_by: dict[str, Any],
    created_at: str,
    lowrank_factors: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> BuiltPatch | None:
    """The patch taking ``base`` to ``edited``; ``None`` when nothing changed (P22: never written).

    ``edited`` may name a subset of ``base``; a name ``base`` does not hold, or a dtype or shape
    change, is
    refused (an append is phase 4's and is not built here).
    """
    for name, t in edited.items():
        if name not in base:
            raise _malformed(f"The edit names tensor {name!r}, which the parent does not hold.")
        if t.dtype != base[name].dtype or t.shape != base[name].shape:
            raise _malformed(
                f"Tensor {name!r} changes from {base[name].dtype}{list(base[name].shape)} to "
                f"{t.dtype}{list(t.shape)}; a patch keeps every tensor's dtype and shape."
            )
    changed_names = sorted(n for n, t in edited.items() if not t.same_bits(base[n]))
    if not changed_names:
        return None
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    for i, name in enumerate(changed_names):
        b, e = base[name], edited[name]
        changed, l2, max_abs = _stats(b, e)
        form, parts, rank = _form_for(b, e, (lowrank_factors or {}).get(name))
        for key, arr in parts.items():
            arrays[f"t{i}_{key}"] = arr
        entries.append(
            {
                "index": i,
                "name": name,
                "dtype": b.dtype,
                "shape": list(b.shape),
                "edited_shape": list(e.shape),
                "form": form,
                "rank": rank,
                "append": None,
                "base_sha256": b.sha256,
                "edited_sha256": e.sha256,
                "changed_entries": changed,
                "l2_delta": l2,
                "max_abs_delta": max_abs,
            }
        )
    data = npz_bytes(arrays)
    manifest: dict[str, Any] = {
        "kind": "miforge.wsp-patch/v1",
        "format_version": 1,
        "bundle_base_sha256": bundle_base_sha256,
        "parent": parent,
        "parent_patch_sha256": parent_patch_sha256,
        "arrays_file": ARRAYS_FILE,
        "arrays_sha256": bytes_sha256(data),
        "tensors": entries,
        "changed_entries_total": sum(t["changed_entries"] for t in entries),
        "created_by": created_by,
        "created_at": created_at,
    }
    manifest["content_sha256"] = content_sha256(manifest)
    manifest["patch_sha256"] = patch_sha256(manifest)
    validate_patch(manifest)
    return BuiltPatch(manifest, data)


def content_sha256(manifest: dict[str, Any]) -> str:
    """RFC 8785 over ``{bundle_base_sha256, parent, tensors}`` only (R1B-A1)."""
    return jcs_sha256({k: manifest[k] for k in ("bundle_base_sha256", "parent", "tensors")})


def patch_sha256(manifest: dict[str, Any]) -> str:
    """RFC 8785 over the manifest without ``patch_sha256``."""
    return jcs_sha256({k: v for k, v in manifest.items() if k != "patch_sha256"})


def manifest_text(manifest: dict[str, Any]) -> str:
    return json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"


def write_patch(dest: Path, built: BuiltPatch) -> Path:
    """Publish the patch folder atomically (temp dir, fsync, rename)."""
    with atomic_dir(Path(dest), latest_link=False) as tmp:
        (tmp / ARRAYS_FILE).write_bytes(built.arrays)
        (tmp / PATCH_JSON).write_text(manifest_text(built.manifest), encoding="utf-8")
    return Path(dest)


# --------------------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------------------


def load_manifest(patch_dir: Path) -> dict[str, Any]:
    """``patch.json``, schema-valid, with its own two digests recomputed."""
    path = Path(patch_dir) / PATCH_JSON
    try:
        manifest: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise _malformed(f"{PATCH_JSON} cannot be read as JSON: {exc}") from exc
    validate_patch(manifest)
    if [t["index"] for t in manifest["tensors"]] != list(range(len(manifest["tensors"]))):
        raise _malformed("tensors[].index must be 0, 1, 2, ... in order.")
    if len({t["name"] for t in manifest["tensors"]}) != len(manifest["tensors"]):
        raise _malformed("tensors[] names a tensor twice.")
    if manifest["content_sha256"] != content_sha256(manifest):
        raise _malformed("content_sha256 does not match the manifest.")
    if manifest["patch_sha256"] != patch_sha256(manifest):
        raise _malformed("patch_sha256 does not match the manifest.")
    if manifest["changed_entries_total"] != sum(t["changed_entries"] for t in manifest["tensors"]):
        raise _malformed("changed_entries_total is not the sum of tensors[].changed_entries.")
    return manifest


def _rebuild(pt: dict[str, Any], i: int, base: Tensor, arrays: dict[str, np.ndarray]) -> Tensor:
    uint = np.dtype(BITS[pt["dtype"]])
    numel = base.size
    if pt["edited_shape"] != pt["shape"]:
        raise _malformed(
            f"Tensor {pt['name']!r} changes shape; an append is refused outside SAE bundles."
        )
    if pt["append"] is not None:
        raise _malformed(
            f"Tensor {pt['name']!r} carries an append, refused outside SAE bundles (phase 4)."
        )

    def fixups(flat: np.ndarray) -> np.ndarray:
        idx, val = arrays[f"t{i}_idx"], arrays[f"t{i}_val"]
        if (
            idx.dtype != np.dtype(np.int64)
            or idx.ndim != 1
            or val.ndim != 1
            or idx.shape != val.shape
        ):
            raise _malformed(
                f"t{i}_idx/t{i}_val must be equal-length 1-D int64 and {uint.name} arrays."
            )
        if val.dtype != uint:
            raise _malformed(
                f"t{i}_val is {val.dtype}; a {pt['dtype']} tensor's values are {uint.name} bits."
            )
        if idx.size and (int(idx.min()) < 0 or int(idx.max()) >= numel):
            raise _malformed(f"t{i}_idx holds an index outside 0..{numel - 1}.", tensor=pt["name"])
        if idx.size > 1 and not bool(np.all(np.diff(idx) > 0)):
            raise _malformed(f"t{i}_idx is not strictly increasing.", tensor=pt["name"])
        return _apply_sparse(flat, idx, val)

    if pt["form"] == "dense":
        bits = arrays[f"t{i}_bits"]
        if bits.dtype != uint or list(bits.shape) != pt["edited_shape"]:
            raise _malformed(f"t{i}_bits must be {uint.name} with shape {pt['edited_shape']}.")
        flat = bits.reshape(-1)
    elif pt["form"] == "sparse":
        flat = fixups(base.bits)
    else:
        a, b = arrays[f"t{i}_a"], arrays[f"t{i}_b"]
        rank = pt["rank"]
        if (
            len(pt["shape"]) != 2
            or a.dtype != np.float64
            or b.dtype != np.float64
            or a.shape != (pt["shape"][0], rank)
            or b.shape != (rank, pt["shape"][1])
        ):
            raise _malformed(
                f"t{i}_a/t{i}_b must be float64 factors of rank {rank} for shape {pt['shape']}."
            )
        flat = fixups(lowrank_reconstruct(base, a, b).bits)
    return Tensor(pt["dtype"], np.ascontiguousarray(flat).reshape(pt["edited_shape"]).astype(uint))


def apply_patch(
    patch_dir: Path, base: dict[str, Tensor]
) -> tuple[dict[str, Any], dict[str, Tensor]]:
    """``(manifest, tensors)``: ``base`` with the patch applied. Writes nothing.

    Refuses ``wsp_patch_base_mismatch`` when any base tensor's hash, dtype or shape differs, and
    ``wsp_patch_malformed`` for everything else, **before** rebuilding anything.
    """
    manifest = load_manifest(patch_dir)
    for pt in manifest["tensors"]:
        held = base.get(pt["name"])
        if (
            held is None
            or held.dtype != pt["dtype"]
            or list(held.shape) != pt["shape"]
            or held.sha256 != pt["base_sha256"]
        ):
            raise refuse(
                "wsp_patch_base_mismatch",
                f"The patch was made against a different {pt['name']!r}: its base hash, dtype or "
                "shape "
                "does not match. Apply it to the parent it names.",
                tensor=pt["name"],
            )
    path = Path(patch_dir) / ARRAYS_FILE
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise _malformed(f"{ARRAYS_FILE} cannot be read: {exc}") from exc
    if bytes_sha256(data) != manifest["arrays_sha256"]:
        raise _malformed(f"{ARRAYS_FILE} does not hash to arrays_sha256.")
    arrays = read_archive(data, manifest)
    rebuilt: dict[str, Tensor] = {}
    for i, pt in enumerate(manifest["tensors"]):
        t = _rebuild(pt, i, base[pt["name"]], arrays)
        if t.sha256 != pt["edited_sha256"]:
            raise _malformed(
                f"Tensor {pt['name']!r} does not rebuild to its edited_sha256.", tensor=pt["name"]
            )
        changed = int(np.count_nonzero(base[pt["name"]].bits.reshape(-1) != t.bits.reshape(-1)))
        if changed != pt["changed_entries"]:
            raise _malformed(
                f"Tensor {pt['name']!r} changes {changed} entries, not {pt['changed_entries']}."
            )
        rebuilt[pt["name"]] = t
    return manifest, {**base, **rebuilt}
