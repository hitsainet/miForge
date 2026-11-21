"""Generated WSP fixtures (PF-41: no fixture bundle bytes are committed).

The bundle generator (``make_bundle``, its independent parity reference and helpers) moved to
``backend/src/wsp/fixture_bundle.py`` in feature 018 (FTASKS 9.4), so the tests and the protocol
test's built-in bundle share ONE generator; it is re-exported here under its old names. The patch
cases and ``enable_wsp`` stay test-only.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from wsp.fixture_bundle import (  # noqa: F401 - re-exported for the tests
    EPS,
    PRACTICE_FILES,
    RULES,
    FixtureBundle,
    make_bundle,
    reference_row_score,
    row_ids_sha256,
    seal,
)
from wsp.weight_ops.hashing import file_sha256, jcs_sha256, tensor_sha256  # noqa: F401
from wsp.weight_ops.st_io import write_safetensors  # noqa: F401
from wsp.weight_ops.tensor import Tensor, from_array  # noqa: F401

# --------------------------------------------------------------------------------------
# Patches
# --------------------------------------------------------------------------------------

DTYPES = ("float32", "float16", "bfloat16")
FORMS = ("sparse", "lowrank", "dense")
HEX_PARENT = "b" * 64


def base_tensor(dtype: str, shape: tuple[int, ...], seed: int) -> Tensor:
    from wsp.weight_ops.tensor import from_values

    rng = np.random.default_rng(seed)
    return from_values(dtype, rng.normal(size=shape).astype(np.float32))


def edit_for(
    form: str, base: Tensor, seed: int, rank: int = 1, cancel: bool = False
) -> tuple[Tensor, tuple[np.ndarray, np.ndarray] | None]:
    """An edit of ``base`` that ``build_patch`` stores in ``form`` (lowrank: with its ``rank``
    factors; rank 1 keeps the original golden's bytes, rank 2 makes the term order matter)."""
    from wsp.weight_ops.patch import lowrank_reconstruct
    from wsp.weight_ops.tensor import from_values

    rng = np.random.default_rng(seed)
    values = base.values64().copy()
    if form == "sparse":
        flat = values.reshape(-1)
        for i in rng.choice(flat.size, size=max(1, flat.size // 50), replace=False):
            flat[i] += 0.5 + rng.random()
        return from_values(base.dtype, values.astype(np.float32)), None
    if form == "dense":
        return (
            from_values(
                base.dtype, (values + rng.normal(size=values.shape) * 0.1).astype(np.float32)
            ),
            None,
        )
    rows, cols = base.shape
    a = rng.normal(size=(rows, rank)) * 0.25
    b = rng.normal(size=(rank, cols)) * 0.25
    if cancel:
        # Two terms that cancel (miWorkbench 044's NEEDS 8): adding them IN ORDER in float64
        # loses the base under 1e20 and lands on 0, while a fused ``base + a @ b`` keeps the base,
        # so a reader that fuses gives a visibly different tensor, not one differing in a last bit.
        a = np.zeros((rows, 2))
        b = np.zeros((2, cols))
        a[:, 0], b[0] = 1e20, 1.0
        a[:, 1], b[1] = 1.0, -1e20
    recon = lowrank_reconstruct(base, a, b)
    v = recon.values64().copy().reshape(-1)
    v[0] += 1.0  # one fix-up entry, so the fix-up path is exercised
    v[-1] -= 1.0
    return from_values(base.dtype, v.reshape(rows, cols).astype(np.float32)), (a, b)


def patch_case(
    dtype: str, form: str, seed: int = 0, rank: int = 1, cancel: bool = False
) -> tuple[dict[str, Tensor], dict[str, Tensor], dict]:
    """``(base, edited, build kwargs)`` for one dtype x form case."""
    shape = (24, 40) if form == "lowrank" else (64,)
    name = "W_dec" if form == "lowrank" else "weight"
    base = {name: base_tensor(dtype, shape, seed), "bias": base_tensor("float32", (1,), seed + 1)}
    edited_t, factors = edit_for(form, base[name], seed + 2, rank, cancel)
    kwargs = {
        "bundle_base_sha256": HEX_PARENT,
        "parent": {"kind": "bundle", "ref": HEX_PARENT, "patch_sha256": None},
        "parent_patch_sha256": None,
        "created_by": {
            "provider": "weightspace",
            "provider_version": "0.3.0",
            "run_id": "wsp_0123456789ab",
        },
        "created_at": "2026-11-16T05:00:00Z",
        "lowrank_factors": {name: factors} if factors is not None else None,
    }
    return base, {name: edited_t}, kwargs


def enable_wsp(monkeypatch: Any, providers: str = "weightspace") -> None:
    """A WSP the start path accepts: the provider enabled AND both URLs set (review OR3-016-2:
    ``prepare_start`` refuses ``wsp_not_configured`` without them). A URL a fixture already set
    (e.g. a fake gateway's port) is kept; the caller clears ``get_wsp_settings``' cache."""
    import os

    monkeypatch.setenv("WSP_PROVIDERS_AVAILABLE", providers)
    for name, value in (
        ("WSP_WORKER_URL", "http://127.0.0.1:9"),
        ("WSP_GATEWAY_URL", "http://127.0.0.1:8080"),
    ):
        if not os.environ.get(name):
            monkeypatch.setenv(name, value)
