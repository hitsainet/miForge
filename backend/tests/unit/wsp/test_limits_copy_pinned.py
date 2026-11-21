"""PF-40 / FR-016.7: the worker's ``limits.py`` is byte-identical to the backend's.

The worker's courtesy check and the backend's authoritative check are one function only while the
two
files are one file. Control C7 changes one byte of the copy.
"""

from __future__ import annotations

from pathlib import Path

SRC = Path(__file__).resolve().parents[3] / "src"
BACKEND = SRC / "wsp" / "weight_ops" / "limits.py"
WORKER = SRC / "wsp_worker" / "limits.py"


def test_the_worker_copy_is_byte_identical() -> None:
    assert BACKEND.is_file() and WORKER.is_file()
    assert (
        WORKER.read_bytes() == BACKEND.read_bytes()
    ), "wsp_worker/limits.py differs from wsp/weight_ops/limits.py; copy the backend's file over it"


def test_both_import_as_the_same_function() -> None:
    import wsp.weight_ops.limits as backend
    import wsp_worker.limits as worker

    patch = {
        "changed_entries_total": 3,
        "tensors": [{"name": "std", "changed_entries": 3, "append": None}],
    }
    limits = {"max_changed_entries": 2, "frozen_tensors": ["std"], "allowed_tensors": None}
    assert [v.__dict__ for v in backend.check_limits(patch, limits)] == [
        v.__dict__ for v in worker.check_limits(patch, limits)
    ]
