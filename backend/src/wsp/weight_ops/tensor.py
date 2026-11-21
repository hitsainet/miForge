"""One representation for a tensor of any WSP dtype: its dtype name and its raw bits.

numpy has no bfloat16, so every tensor is held as the unsigned integer view of its bits (``uint16``,
``uint32`` or ``uint64``). That makes hashing, patching and comparison bit-exact by construction,
and
it makes a bf16 tensor need no float round trip (FTID §3.1). Arithmetic goes through
:meth:`Tensor.values64` (exact for every dtype here) and back through :func:`from_values`, whose
casts
round to nearest even, as numpy and torch do.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from wsp.weight_ops.hashing import tensor_sha256

#: The dtypes a WSP tensor may have (FTDD §3.1 ``TensorEntry.dtype``), and their bit widths.
BITS: dict[str, type[np.unsignedinteger]] = {
    "float16": np.uint16,
    "bfloat16": np.uint16,
    "float32": np.uint32,
    "float64": np.uint64,
}
_FLOAT: dict[str, type[np.floating]] = {
    "float16": np.float16,
    "float32": np.float32,
    "float64": np.float64,
}


class TensorDtypeError(ValueError):
    """A dtype outside :data:`BITS`, or bits of the wrong width for the named dtype."""


@dataclass(frozen=True)
class Tensor:
    """A tensor as ``(dtype name, raw bits)``; ``bits`` is C-contiguous, in the tensor's shape."""

    dtype: str
    bits: np.ndarray

    def __post_init__(self) -> None:
        if self.dtype not in BITS:
            raise TensorDtypeError(f"dtype {self.dtype!r} is not one of {sorted(BITS)}")
        if self.bits.dtype != np.dtype(BITS[self.dtype]).newbyteorder("="):
            raise TensorDtypeError(
                f"a {self.dtype} tensor's bits must be {np.dtype(BITS[self.dtype]).name}, "
                f"got {self.bits.dtype}"
            )
        if not self.bits.flags.c_contiguous:
            object.__setattr__(self, "bits", np.ascontiguousarray(self.bits))
        self.bits.setflags(write=False)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(n) for n in self.bits.shape)

    @property
    def size(self) -> int:
        return int(self.bits.size)

    @property
    def sha256(self) -> str:
        return tensor_sha256(self.bits)

    def values64(self) -> np.ndarray:
        """The values as float64 (exact: every dtype here widens to float64 without rounding)."""
        if self.dtype == "bfloat16":
            return (
                (self.bits.astype(np.uint32) << np.uint32(16)).view(np.float32).astype(np.float64)
            )
        return self.bits.view(_FLOAT[self.dtype]).astype(np.float64)

    def same_bits(self, other: Tensor) -> bool:
        return (
            self.dtype == other.dtype
            and self.shape == other.shape
            and bool(np.array_equal(self.bits, other.bits))
        )


def from_array(arr: np.ndarray, dtype: str | None = None) -> Tensor:
    """Wrap a float16/32/64 array, or ``uint16`` bf16 bits with ``dtype='bfloat16'``."""
    a = np.ascontiguousarray(arr)
    if dtype == "bfloat16":
        if a.dtype != np.uint16:
            raise TensorDtypeError("a bfloat16 tensor is passed as its uint16 bits")
        return Tensor("bfloat16", a.copy())
    name = {
        np.dtype(np.float16): "float16",
        np.dtype(np.float32): "float32",
        np.dtype(np.float64): "float64",
    }.get(a.dtype.newbyteorder("="))
    if name is None or (dtype is not None and dtype != name):
        raise TensorDtypeError(f"cannot wrap a {a.dtype} array as {dtype or 'a WSP tensor'}")
    return Tensor(name, a.astype(a.dtype.newbyteorder("=")).view(BITS[name]).copy())


def from_values(dtype: str, values: np.ndarray) -> Tensor:
    """Cast float values to ``dtype``, rounding to nearest even, and wrap the bits."""
    v = np.asarray(values)
    if dtype == "bfloat16":
        return Tensor("bfloat16", float32_to_bf16_bits(v.astype(np.float32)))
    if dtype not in _FLOAT:
        raise TensorDtypeError(f"dtype {dtype!r} is not one of {sorted(BITS)}")
    return Tensor(dtype, np.ascontiguousarray(v.astype(_FLOAT[dtype])).view(BITS[dtype]).copy())


def float32_to_bf16_bits(values: np.ndarray) -> np.ndarray:
    """float32 -> bfloat16 bits, round to nearest even; NaN stays a quiet NaN (torch's rule)."""
    u = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).astype(np.uint64)
    rounding = ((u >> np.uint64(16)) & np.uint64(1)) + np.uint64(0x7FFF)
    out = ((u + rounding) >> np.uint64(16)).astype(np.uint16)
    nan = np.isnan(np.asarray(values, dtype=np.float32))
    if nan.any():
        out = np.where(nan, ((u >> np.uint64(16)) | np.uint64(0x40)).astype(np.uint16), out)
    return np.ascontiguousarray(out.reshape(np.shape(values)))
