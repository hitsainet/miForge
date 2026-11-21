"""The worker's ``miforge.wsp-patch/v1`` writer (FTDD §3.2, §6.5; FTID §3.1). numpy and stdlib only.

It is the worker-side twin of the backend's ``wsp/weight_ops/patch.py::build_patch`` (which the
worker may not import: the fence, FTDD §6.1): the same forms (``sparse`` when ``changed x 16 <
numel x itemsize``, else ``dense``; the worker never emits ``lowrank``), the same deterministic
``edit.npz`` bytes, the same manifest and the same two digests. ``test_patch_writer.py`` pins the
two byte for byte, and the backend re-applies and re-hashes every patch at intake anyway
(FR-016.19): this writer is a courtesy, never the authority.

The digests are RFC 8785 (JCS). The worker has no ``rfc8785`` package, so :func:`jcs_bytes`
implements the subset a patch manifest needs (objects, arrays, strings, integers within 2^53,
finite floats, booleans, null), pinned against the package by the same test.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from wsp_worker.bundle import BITS, WorkerTensor

PATCH_JSON = "patch.json"
ARRAYS_FILE = "edit.npz"
SPARSE_FACTOR = 16
_ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
_FLOAT = {"float16": np.float16, "float32": np.float32, "float64": np.float64}


# ----------------------------------------------------------------------------- RFC 8785


def _number(x: float) -> str:
    """ECMAScript ``Number::toString`` for a finite float (RFC 8785 §3.2.2.3)."""
    if not math.isfinite(x):
        raise ValueError("RFC 8785 has no NaN or infinity")
    if x == 0:
        return "0"
    sign = "-" if x < 0 else ""
    digits, _, exp = repr(abs(x)).partition("e")
    mantissa = digits.replace(".", "")
    point = digits.index(".") if "." in digits else len(digits)
    stripped = mantissa.lstrip("0")
    n = point + (int(exp) if exp else 0) - (len(mantissa) - len(stripped))
    s = stripped.rstrip("0")
    k = len(s)
    if k <= n <= 21:
        body = s + "0" * (n - k)
    elif 0 < n <= 21:
        body = s[:n] + "." + s[n:]
    elif -6 < n <= 0:
        body = "0." + "0" * (-n) + s
    else:
        e = n - 1
        body = (s if k == 1 else s[0] + "." + s[1:]) + "e" + ("+" if e >= 0 else "-") + str(abs(e))
    return sign + body


def _jcs(value: Any, out: list[str]) -> None:
    if value is None:
        out.append("null")
    elif value is True:
        out.append("true")
    elif value is False:
        out.append("false")
    elif isinstance(value, int):
        if abs(value) > 2**53 - 1:
            raise ValueError("an integer beyond 2^53 is not exact in RFC 8785")
        out.append(str(value))
    elif isinstance(value, float):
        out.append(_number(value))
    elif isinstance(value, str):
        out.append(json.dumps(value, ensure_ascii=False))
    elif isinstance(value, (list, tuple)):
        out.append("[")
        for i, item in enumerate(value):
            if i:
                out.append(",")
            _jcs(item, out)
        out.append("]")
    elif isinstance(value, dict):
        out.append("{")
        for i, key in enumerate(sorted(value, key=lambda k: str(k).encode("utf-16-be"))):
            if not isinstance(key, str):
                raise ValueError("RFC 8785 object keys are strings")
            if i:
                out.append(",")
            out.append(json.dumps(key, ensure_ascii=False))
            out.append(":")
            _jcs(value[key], out)
        out.append("}")
    else:
        raise ValueError(f"RFC 8785 cannot encode {type(value).__name__}")


def jcs_bytes(value: Any) -> bytes:
    out: list[str] = []
    _jcs(value, out)
    return "".join(out).encode("utf-8")


def jcs_sha256(value: Any) -> str:
    return hashlib.sha256(jcs_bytes(value)).hexdigest()


# ----------------------------------------------------------------------------- tensors


def float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bits, round to nearest even; NaN stays a quiet NaN (torch's rule)."""
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounding = ((u >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF)
    out = ((u + rounding) >> np.uint64(16)).astype(np.uint16)
    nan = np.isnan(np.asarray(values, dtype=np.float32))
    if nan.any():
        out = np.where(nan, ((u >> np.uint64(16)) | np.uint64(0x40)).astype(np.uint16), out)
    return np.ascontiguousarray(out.reshape(np.shape(values)))


def from_values(dtype: str, values: np.ndarray) -> WorkerTensor:
    """Cast edited float values back to ``dtype`` (round to nearest even) and wrap the bits."""
    v = np.asarray(values)
    if dtype == "bfloat16":
        return WorkerTensor(dtype, float32_to_bf16_bits(v.astype(np.float32)))
    bits = np.ascontiguousarray(v.astype(_FLOAT[dtype])).view(BITS[dtype]).copy()
    return WorkerTensor(dtype, bits)


def _values64(t: WorkerTensor) -> np.ndarray:
    return t.values().astype(np.float64)


# ----------------------------------------------------------------------------- the patch


def npz_bytes(arrays: dict[str, np.ndarray]) -> bytes:
    """Deterministic ``.npz``: stored, sorted members, epoch timestamps, npy 1.0, little-endian."""
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


def build_patch(
    base: dict[str, WorkerTensor],
    edited: dict[str, WorkerTensor],
    *,
    bundle_base_sha256: str,
    parent: dict[str, Any],
    parent_patch_sha256: str | None,
    created_by: dict[str, Any],
    created_at: str,
) -> tuple[dict[str, Any], bytes] | None:
    """``(manifest, edit.npz bytes)`` taking ``base`` to ``edited``; ``None`` when unchanged."""
    for name, t in edited.items():
        b = base.get(name)
        if b is None or t.dtype != b.dtype or t.bits.shape != b.bits.shape:
            raise ValueError(f"tensor {name!r} is not in the parent with this dtype and shape")
    changed_names = sorted(n for n, t in edited.items() if not np.array_equal(t.bits, base[n].bits))
    if not changed_names:
        return None
    arrays: dict[str, np.ndarray] = {}
    entries: list[dict[str, Any]] = []
    for i, name in enumerate(changed_names):
        b, e = base[name], edited[name]
        base_flat, edited_flat = b.bits.reshape(-1), e.bits.reshape(-1)
        idx = np.flatnonzero(base_flat != edited_flat).astype(np.int64)
        width = np.dtype(BITS[b.dtype]).itemsize
        delta = _values64(e) - _values64(b)
        if not np.isfinite(delta).all():
            raise ValueError("the edit produces a non-finite value")
        if idx.size * SPARSE_FACTOR < e.bits.size * width:
            form = "sparse"
            arrays[f"t{i}_idx"], arrays[f"t{i}_val"] = idx, edited_flat[idx]
        else:
            form = "dense"
            arrays[f"t{i}_bits"] = e.bits
        entries.append(
            {
                "index": i,
                "name": name,
                "dtype": b.dtype,
                "shape": list(b.bits.shape),
                "edited_shape": list(e.bits.shape),
                "form": form,
                "rank": None,
                "append": None,
                "base_sha256": b.sha256,
                "edited_sha256": e.sha256,
                "changed_entries": int(idx.size),
                "l2_delta": math.sqrt(math.fsum((delta.reshape(-1) ** 2).tolist())),
                "max_abs_delta": float(np.abs(delta).max(initial=0.0)),
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
        "arrays_sha256": hashlib.sha256(data).hexdigest(),
        "tensors": entries,
        "changed_entries_total": sum(t["changed_entries"] for t in entries),
        "created_by": created_by,
        "created_at": created_at,
    }
    manifest["content_sha256"] = jcs_sha256(
        {k: manifest[k] for k in ("bundle_base_sha256", "parent", "tensors")}
    )
    manifest["patch_sha256"] = jcs_sha256(manifest)
    return manifest, data


def write_patch(dest: Path, manifest: dict[str, Any], data: bytes) -> Path:
    """Write ``patch.json`` and ``edit.npz`` into a new directory ``dest`` (never over one)."""
    dest = Path(dest)
    dest.mkdir(mode=0o700)
    (dest / ARRAYS_FILE).write_bytes(data)
    text = json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    (dest / PATCH_JSON).write_text(text, encoding="utf-8")
    return dest
