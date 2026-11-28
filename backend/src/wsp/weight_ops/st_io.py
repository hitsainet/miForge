"""A minimal, strict safetensors reader and writer for WSP tensors.

The ``safetensors`` package's numpy loader cannot return bfloat16 (numpy has none), and miForge must
hold
bf16 as bits anyway (``tensor.py``). The format is small enough to read exactly: an 8-byte
little-endian
header length, a JSON header mapping each name to ``{dtype, shape, data_offsets}``, then the data.
This
reader **refuses** rather than guesses: an unknown dtype, an offset outside the data, overlapping or
gapped spans, a header larger than :data:`MAX_HEADER_BYTES`, or a size that disagrees with the
shape.
A pickled format is never read: ``wsp_bundle_format_refused`` names any other file kind upstream.
"""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np

from wsp.weight_ops.tensor import BITS, Tensor

#: The header bound: far above any bundle's (a dozen tensors), far below a hostile file's claim.
MAX_HEADER_BYTES = 1 << 20

_ST_TO_WSP = {"F16": "float16", "BF16": "bfloat16", "F32": "float32", "F64": "float64"}
_WSP_TO_ST = {v: k for k, v in _ST_TO_WSP.items()}


class SafetensorsError(ValueError):
    """Not a safetensors file this reader accepts. The message says which rule failed."""


def read_safetensors(path: Path) -> dict[str, Tensor]:
    data = Path(path).read_bytes()
    return parse_safetensors(data)


def parse_safetensors(data: bytes) -> dict[str, Tensor]:
    if len(data) < 8:
        raise SafetensorsError("shorter than the 8-byte header length")
    (n,) = struct.unpack("<Q", data[:8])
    if n > MAX_HEADER_BYTES or 8 + n > len(data):
        raise SafetensorsError(f"header length {n} is out of bounds")
    try:
        header = json.loads(data[8 : 8 + n].decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"header is not JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError("header is not an object")
    header.pop("__metadata__", None)
    body = data[8 + n :]
    spans: list[tuple[int, int, str]] = []
    out: dict[str, Tensor] = {}
    for name, entry in header.items():
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise SafetensorsError(f"tensor {name!r} has a malformed entry")
        dtype = _ST_TO_WSP.get(entry["dtype"])
        if dtype is None:
            raise SafetensorsError(
                f"tensor {name!r} has dtype {entry['dtype']!r}, not one of {sorted(_ST_TO_WSP)}"
            )
        shape = entry["shape"]
        if not isinstance(shape, list) or not all(isinstance(d, int) and d >= 0 for d in shape):
            raise SafetensorsError(f"tensor {name!r} has a malformed shape")
        begin, end = entry["data_offsets"]
        width = np.dtype(BITS[dtype]).itemsize
        count = int(np.prod(shape, dtype=np.int64)) if shape else 1
        if not (0 <= begin <= end <= len(body)) or end - begin != count * width:
            raise SafetensorsError(f"tensor {name!r} has data offsets that disagree with its shape")
        spans.append((begin, end, name))
        bits = np.frombuffer(
            body, dtype=np.dtype(BITS[dtype]).newbyteorder("<"), count=count, offset=begin
        )
        out[name] = Tensor(dtype, bits.astype(np.dtype(BITS[dtype])).reshape(shape).copy())
    position = 0
    for begin, end, name in sorted(spans):
        if begin != position:
            raise SafetensorsError(f"tensor {name!r} leaves a gap or overlaps another")
        position = end
    if position != len(body):
        raise SafetensorsError("trailing bytes after the last tensor")
    return out


def dump_safetensors(tensors: dict[str, Tensor]) -> bytes:
    """Deterministic bytes: names sorted, no metadata, the header padded with spaces to 8 bytes."""
    header: dict[str, dict[str, object]] = {}
    chunks: list[bytes] = []
    offset = 0
    for name in sorted(tensors):
        t = tensors[name]
        raw = (
            np.ascontiguousarray(t.bits).astype(np.dtype(BITS[t.dtype]).newbyteorder("<")).tobytes()
        )
        header[name] = {
            "dtype": _WSP_TO_ST[t.dtype],
            "shape": list(t.shape),
            "data_offsets": [offset, offset + len(raw)],
        }
        chunks.append(raw)
        offset += len(raw)
    text = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    text += b" " * (-len(text) % 8)
    return struct.pack("<Q", len(text)) + text + b"".join(chunks)


def write_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    Path(path).write_bytes(dump_safetensors(tensors))
