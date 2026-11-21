"""The golden WSP patches (FTASKS 2.11a; PF-55), for miWorkbench's cross-repo round trip (P54).

One folder per form under ``docs/schemas/examples/wsp-patch/``: ``base.safetensors`` (the parent's
tensors), ``patch.json`` and ``edit.npz`` (written by ``wsp.weight_ops.patch``), and
``expected.json``
(each edited tensor's ``edited_sha256``, and the sha256 of the whole edited set's safetensors
bytes).

Regenerate with ``python -m tests.support.wsp_golden <dest>`` from ``backend/`` (with ``src`` on the
path); ``test_wsp_golden_patches.py`` regenerates into a temporary directory and requires the
committed
files to be byte-identical. Every input is seeded and every number is computed without a BLAS
reduction,
so the bytes do not depend on the CPU (suite memory ``goldens-need-cross-cpu-tolerance``).
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from tests.support.wsp_fixtures import patch_case
from wsp.weight_ops.patch import build_patch, write_patch
from wsp.weight_ops.st_io import dump_safetensors

#: folder -> (form, dtype, low-rank rank): float32 for sparse and both low-rank goldens, bfloat16
#: for the dense bits path. ``lowrank_rank2`` (miWorkbench 044's NEEDS 7, 2026-11-16) has two
#: rank-1 terms, so a reader that fuses them (``a @ b`` in one reduction) instead of adding each
#: term in order in float64 can differ in the last bit, which a rank-1 golden cannot show.
GOLDEN: dict[str, tuple[str, str, int]] = {
    "sparse": ("sparse", "float32", 1),
    "lowrank": ("lowrank", "float32", 1),
    "lowrank_rank2": ("lowrank", "float32", 2),
    "lowrank_cancel": ("lowrank", "float32", 2),
    "dense": ("dense", "bfloat16", 1),
}
SEED = 2026
#: Goldens whose two low-rank terms cancel (miWorkbench 044's NEEDS 8, 2026-11-16): the in-order
#: float64 sum loses the base, a fused ``base + a @ b`` keeps it, so the two differ visibly.
CANCELLING = frozenset({"lowrank_cancel"})


def generate(dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    for folder_name, (form, dtype, rank) in GOLDEN.items():
        base, edited, kwargs = patch_case(
            dtype, form, seed=SEED, rank=rank, cancel=folder_name in CANCELLING
        )
        built = build_patch(base, edited, **kwargs)
        assert built is not None and built.manifest["tensors"][0]["form"] == form
        folder = dest / folder_name
        write_patch(folder, built)
        (folder / "base.safetensors").write_bytes(dump_safetensors(base))
        expected = {
            "form": form,
            "dtype": dtype,
            "edited_sha256": {t["name"]: t["edited_sha256"] for t in built.manifest["tensors"]},
            "edited_safetensors_sha256": hashlib.sha256(
                dump_safetensors({**base, **edited})
            ).hexdigest(),
        }
        (folder / "expected.json").write_text(
            json.dumps(expected, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        written.append(folder)
    return written


if __name__ == "__main__":  # pragma: no cover - a maintenance entry point
    generate(Path(sys.argv[1]))
