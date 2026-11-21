"""The worker's bundle reader (FR-016.9; FTDD §3.1, §6.5): safetensors only, practice memory-mapped.

The supervisor has already checked every pushed file's sha256 and size against the push. This
module checks the push against the **bundle manifest** (same name, role and sha256 per file), then
reads:

- the ``heads`` file with ``safetensors.numpy.load_file``; a file holding bfloat16, which numpy
  cannot represent, is read by this module's own strict header reader as ``uint16`` bits (FTDD
  §6.5). Every tensor's dtype, shape and sha256 must equal the manifest's ``tensors[]``;
- the practice activations as a read-only ``np.memmap`` (``tokens x d_model``), and the small
  practice arrays with ``np.load(..., allow_pickle=False, mmap_mode="r")``.

A pickled format is never opened: ``.pt``, ``.pth``, ``.bin``, ``.ckpt``, ``.pkl`` and ``.pickle``
are refused by name before any read, and ``.npy`` is read with ``allow_pickle=False``.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

PICKLED_SUFFIXES = (".pt", ".pth", ".bin", ".ckpt", ".pkl", ".pickle")
MAX_HEADER_BYTES = 1 << 20
BITS = {"float16": np.uint16, "bfloat16": np.uint16, "float32": np.uint32, "float64": np.uint64}
_ST = {"F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64"}
_FLOAT = {"float16": np.float16, "float32": np.float32, "float64": np.float64}
PRACTICE_ROLES = (
    "practice_offsets",
    "practice_positions",
    "practice_labels",
    "practice_validation_mask",
    "practice_truncated_mask",
)


class BundleError(ValueError):
    def __init__(self, message: str, code: str = "wsp_bundle_invalid") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class WorkerTensor:
    """A tensor as its dtype name and its raw bits (``uint16`` for bf16 and float16)."""

    dtype: str
    bits: np.ndarray

    @property
    def sha256(self) -> str:
        a = np.ascontiguousarray(self.bits).astype(self.bits.dtype.newbyteorder("<"))
        return hashlib.sha256(a.reshape(-1).view(np.uint8).data).hexdigest()

    def values(self) -> np.ndarray:
        """float32 for bf16 (exact), the stored float dtype otherwise: what the editor sees."""
        if self.dtype == "bfloat16":
            return (self.bits.astype(np.uint32) << np.uint32(16)).view(np.float32)
        return self.bits.view(_FLOAT[self.dtype])


@dataclass(frozen=True)
class Practice:
    activations: np.ndarray
    arrays: dict[str, np.ndarray]


@dataclass(frozen=True)
class Bundle:
    manifest: dict[str, Any]
    tensors: dict[str, WorkerTensor]
    practice: Practice

    def editable(self, name: str) -> bool:
        return any(t["name"] == name and t["editable"] for t in self.manifest["tensors"])


def refuse_pickled(path: Path) -> None:
    if path.suffix.lower() in PICKLED_SUFFIXES:
        raise BundleError(
            f"{path.name} is a pickled format; the worker reads safetensors and plain .npy only",
            "wsp_bundle_format_refused",
        )


def _header(data: bytes) -> tuple[dict[str, Any], int]:
    if len(data) < 8:
        raise BundleError("a safetensors file is shorter than its header length")
    (n,) = struct.unpack("<Q", data[:8])
    if n > MAX_HEADER_BYTES or 8 + n > len(data):
        raise BundleError(f"safetensors header length {n} is out of bounds")
    try:
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BundleError(f"safetensors header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise BundleError("safetensors header is not an object")
    header.pop("__metadata__", None)
    return header, 8 + n


def read_heads(path: Path) -> dict[str, WorkerTensor]:
    """Every tensor of a ``heads`` safetensors file, as bits."""
    refuse_pickled(path)
    data = path.read_bytes()
    header, start = _header(data)
    dtypes: dict[str, str] = {}
    for name, entry in header.items():
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise BundleError(f"tensor {name!r} has a malformed header entry")
        if entry["dtype"] not in _ST:
            raise BundleError(f"tensor {name!r} has dtype {entry['dtype']!r}")
        dtypes[name] = _ST[entry["dtype"]]
    if "bfloat16" not in dtypes.values():
        from safetensors.numpy import load_file

        loaded = load_file(str(path))
        return {
            n: WorkerTensor(dtypes[n], np.ascontiguousarray(a).view(BITS[dtypes[n]]))
            for n, a in loaded.items()
        }
    body = memoryview(data)[start:]
    out: dict[str, WorkerTensor] = {}
    for name, entry in header.items():
        dtype, shape = dtypes[name], entry["shape"]
        begin, end = entry["data_offsets"]
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        width = np.dtype(BITS[dtype]).itemsize
        if not (0 <= begin <= end <= len(body)) or end - begin != count * width:
            raise BundleError(f"tensor {name!r} has data offsets that disagree with its shape")
        bits = np.frombuffer(body, np.dtype(BITS[dtype]).newbyteorder("<"), count, begin)
        out[name] = WorkerTensor(dtype, bits.astype(BITS[dtype]).reshape(shape).copy())
    return out


def load(work: Path, manifest: dict[str, Any], pushed: list[dict[str, Any]]) -> Bundle:
    """The bundle in ``work``, every file and tensor checked against ``manifest``."""
    by_role = {f["role"]: f for f in manifest["files"]}
    pushed_by_name = {f["name"]: f for f in pushed}
    paths: dict[str, Path] = {}
    for role, entry in by_role.items():
        got = pushed_by_name.get(entry["name"])
        if got is None or got["role"] != role or got["sha256"] != entry["sha256"]:
            raise BundleError(f"the pushed files do not carry the manifest's {role} file")
        path = work / entry["name"]
        refuse_pickled(path)
        paths[role] = path
    tensors = read_heads(paths["heads"])
    declared = {t["name"]: t for t in manifest["tensors"]}
    if set(tensors) != set(declared):
        raise BundleError(f"heads holds {sorted(tensors)}, the manifest {sorted(declared)}")
    for name, t in tensors.items():
        d = declared[name]
        if t.dtype != d["dtype"] or list(t.bits.shape) != d["shape"] or t.sha256 != d["sha256"]:
            raise BundleError(f"tensor {name!r} differs from the manifest's dtype, shape or hash")
    p = manifest["practice"]
    dtype = np.dtype({"float16": np.float16, "float32": np.float32}[p["dtype"]]).newbyteorder("<")
    acts_path = paths["practice_activations"]
    if acts_path.stat().st_size != int(p["tokens"]) * int(p["d_model"]) * dtype.itemsize:
        raise BundleError("practice activations' size disagrees with tokens x d_model")
    acts = np.memmap(acts_path, dtype=dtype, mode="r", shape=(int(p["tokens"]), int(p["d_model"])))
    arrays: dict[str, np.ndarray] = {}
    for role in PRACTICE_ROLES:
        try:
            loaded = np.load(paths[role], allow_pickle=False, mmap_mode="r")
        except (ValueError, OSError, EOFError) as exc:
            raise BundleError(
                f"{role} is not a plain .npy array", "wsp_bundle_format_refused"
            ) from exc
        if not isinstance(loaded, np.ndarray) or loaded.dtype == object:
            # An .npz under a .npy name loads as an NpzFile (review OR1-016-7).
            close = getattr(loaded, "close", None)
            if close is not None:
                close()
            raise BundleError(f"{role} is not a plain .npy array", "wsp_bundle_format_refused")
        arrays[role] = loaded
    return Bundle(manifest, tensors, Practice(acts, arrays))
