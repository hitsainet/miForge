"""The hash rules every WSP document uses (FTDD §3.1-3.2; P30, P54).

- **A tensor's hash** is ``sha256`` over its little-endian, C-contiguous bytes in its stored dtype.
  That
  is wsplab's ``sha`` ([wl] ``patch/format.py#L38-40``). bfloat16 has no numpy dtype, so it is held
  as
  its ``uint16`` bits and hashed as such: the bytes are the same either way.
- **A document's hash** is ``sha256`` over its RFC 8785 JSON Canonicalisation (the ``rfc8785``
  package,
  pinned at 0.1.4, miWorkbench's version). ``core.canonical.sha256_of`` is **not** RFC 8785: it
  writes
  ``1e-05`` where RFC 8785 writes ``0.00001``, and these documents carry floats (R1B-X9). So it is
  never
  used for a WSP digest (ADR-005's recorded exception, PADR amendment B3a).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np
import rfc8785

_CHUNK = 1 << 20


def tensor_sha256(arr: np.ndarray) -> str:
    """sha256 of the array's raw bytes: little-endian, C-contiguous, in its stored dtype."""
    a = np.ascontiguousarray(arr)
    if a.dtype.byteorder == ">" or (a.dtype.byteorder == "=" and not _little_endian()):
        a = a.astype(a.dtype.newbyteorder("<"))
    return hashlib.sha256(a.reshape(-1).view(np.uint8).data).hexdigest()


def _little_endian() -> bool:
    import sys

    return sys.byteorder == "little"


def bytes_sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_sha256(path: Path) -> str:
    """sha256 of a file's bytes, streamed."""
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def jcs_bytes(document: Any) -> bytes:
    """RFC 8785 canonical bytes of ``document``."""
    return bytes(rfc8785.dumps(document))


def jcs_sha256(document: Any) -> str:
    """sha256 of the RFC 8785 canonical bytes of ``document``."""
    return hashlib.sha256(jcs_bytes(document)).hexdigest()
