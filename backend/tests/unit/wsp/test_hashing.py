"""The WSP hash rules (FTASKS 2.1; FTDD §3.1-3.2): known arrays, bf16 as uint16, RFC 8785 documents.

Each expected value is computed here **independently** of the module under test, from
``struct``-packed
little-endian bytes, so the test cannot agree with the implementation by construction.
"""

from __future__ import annotations

import hashlib
import json
import struct

import numpy as np
import pytest

from wsp.weight_ops.hashing import bytes_sha256, file_sha256, jcs_bytes, jcs_sha256, tensor_sha256


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def test_float32_hash_is_over_little_endian_bytes() -> None:
    arr = np.array([1.0, -2.5, 3.25], dtype=np.float32)
    assert tensor_sha256(arr) == _sha(struct.pack("<3f", 1.0, -2.5, 3.25))


def test_float64_bias_shape_one() -> None:
    arr = np.array([0.125], dtype=np.float64)
    assert tensor_sha256(arr) == _sha(struct.pack("<d", 0.125))


def test_float16_hash() -> None:
    arr = np.array([1.0, 0.5], dtype=np.float16)
    assert tensor_sha256(arr) == _sha(struct.pack("<2e", 1.0, 0.5))


def test_bfloat16_is_hashed_as_its_uint16_bits() -> None:
    # bf16 1.0 is 0x3F80; -2.0 is 0xC000.
    bits = np.array([0x3F80, 0xC000], dtype=np.uint16)
    assert tensor_sha256(bits) == _sha(struct.pack("<2H", 0x3F80, 0xC000))


def test_big_endian_input_hashes_as_little_endian() -> None:
    arr = np.array([1.0, 2.0], dtype=">f4")
    assert tensor_sha256(arr) == _sha(struct.pack("<2f", 1.0, 2.0))


def test_non_contiguous_view_hashes_its_logical_order() -> None:
    base = np.arange(6, dtype=np.float32).reshape(2, 3)
    transposed = base.T  # (3, 2), not C-contiguous
    expected = _sha(struct.pack("<6f", 0, 3, 1, 4, 2, 5))
    assert tensor_sha256(transposed) == expected


def test_shape_does_not_enter_the_hash() -> None:
    """The [wl] rule hashes the flat bytes; the shape is recorded beside it, not inside it."""
    flat = np.arange(6, dtype=np.float32)
    assert tensor_sha256(flat) == tensor_sha256(flat.reshape(2, 3))


def test_file_and_bytes_hash(tmp_path) -> None:
    path = tmp_path / "f.bin"
    path.write_bytes(b"abc" * 1_000_000)
    assert file_sha256(path) == _sha(b"abc" * 1_000_000) == bytes_sha256(b"abc" * 1_000_000)


def test_jcs_writes_1e_5_as_0_00001() -> None:
    """R1B-X9: the float case where ``json.dumps`` (so ``canonical.py``) differs from RFC 8785."""
    doc = {"tolerance": 1e-5, "b": [1, 2], "a": "x"}
    assert jcs_bytes(doc) == b'{"a":"x","b":[1,2],"tolerance":0.00001}'
    assert json.dumps(doc, sort_keys=True, separators=(",", ":")) != jcs_bytes(doc).decode()
    assert jcs_sha256(doc) == _sha(b'{"a":"x","b":[1,2],"tolerance":0.00001}')


@pytest.mark.parametrize("bad", [float("nan"), float("inf")])
def test_jcs_refuses_non_finite_numbers(bad: float) -> None:
    with pytest.raises((ValueError, TypeError, OverflowError)):
        jcs_bytes({"x": bad})
