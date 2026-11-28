"""A bundle and a child line, written inside the WSP worker image (feature 016; FTASKS 8.5; feature
018; FTASKS 10.2).

``write_bundle`` writes a bundle the worker's loader accepts (safetensors heads, every file and
tensor hashed into the manifest) and ``child_line`` the line the supervisor would write for a
session over it. The in-image engine probe (``engine_probe_image.py``) builds its session from
them. Numpy, safetensors and the image's worker package only.

Until 018-P4 these lived in ``readme_examples.py`` beside the numpy ports of ``weightspace`` 0.3.0's
README examples, which drove the weightspace adapter; the ports went with the adapter when
``weightspace`` was retired (FR-018.18).
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

F32 = "float32"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_bundle(work: Path, tensors: dict[str, np.ndarray], description: str) -> dict[str, Any]:
    """A bundle the worker's loader accepts, written to ``work``; returns its manifest."""
    from safetensors.numpy import save_file

    from wsp_worker.bundle import WorkerTensor
    from wsp_worker.patch_writer import jcs_sha256

    work.mkdir(parents=True)
    heads = work / "heads.safetensors"
    save_file({n: np.ascontiguousarray(a, dtype=np.float32) for n, a in tensors.items()}, heads)
    # One practice row of one token: the loader requires the files; no tool ever reads them.
    (work / "tokens.f32").write_bytes(np.zeros(1, dtype="<f4").tobytes())
    arrays = {
        "practice_offsets": np.array([0, 1], dtype=np.int64),
        "practice_positions": np.array([0], dtype=np.int64),
        "practice_labels": np.array([0], dtype=np.int8),
        "practice_validation_mask": np.array([False]),
        "practice_truncated_mask": np.array([False]),
    }
    for role, arr in arrays.items():
        np.save(work / f"{role}.npy", arr, allow_pickle=False)
    roles = {"heads": heads.name, "practice_activations": "tokens.f32"}
    roles.update({role: f"{role}.npy" for role in arrays})
    files = [
        {"name": n, "role": r, "sha256": _sha(work / n), "bytes": (work / n).stat().st_size}
        for r, n in roles.items()
    ]
    manifest: dict[str, Any] = {
        "kind": "miforge.wsp-bundle/v1",
        "bundle_kind": "linear_probe",
        "files": files,
        "tensors": [
            {
                "name": n,
                "shape": list(a.shape),
                "dtype": F32,
                "sha256": WorkerTensor(F32, np.ascontiguousarray(a).view(np.uint32)).sha256,
                "editable": True,
            }
            for n, a in tensors.items()
        ],
        # Phase 1's only kind; inert here (no observation tool is named). The editor is shown
        # `description`, the example's own context.
        "function": {
            "kind": "linear_probe/v1",
            "description": description,
            "linear_probe": {"rule": "mean", "params": {}},
        },
        "practice": {
            "format": "token_activations",
            "dtype": F32,
            "rows": 1,
            "tokens": 1,
            "d_model": 1,
        },
        "operating_point": None,
    }
    manifest["bundle_sha256"] = jcs_sha256(manifest)
    return manifest


def child_line(
    root: Path, manifest: dict[str, Any], kind: str, task: str, out: Path | None = None
) -> dict[str, Any]:
    """The child line the supervisor would write for this session (``paths`` under ``root``;
    ``out`` replaces ``root/out``, as the pod's ``/out`` does for a session in the image)."""
    out, scratch = out or root / "out", root / "tmp"
    out.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    request = {
        "kind": kind,
        "task": task,
        "parent": {"kind": "bundle", "ref": manifest["bundle_sha256"]},
        "budgets": {
            "tokens": 100_000,
            "time_s": 600,
            "max_steps": 12,
            "forward_passes": 0,
            "cost_usd": None,
        },
        "change_limits": {
            "max_changed_entries": None,
            "allowed_tensors": None,
            "frozen_tensors": [],
        },
        "thinking": None,
        "observation_tools": [],
    }
    return {
        "mode": "session",
        "run_id": "wsr_image_bundle",
        "spec": {
            "time_budget_s": 600,
            "grace_s": 60,
            "pull_timeout_s": 60,
            "request": request,
            "manifest": manifest,
            "editor": {
                "kind": "openai_compatible",
                "model": "scripted",
                "serving": "miserver",
                "prices": {"input": None, "output": None, "cache_write": None, "cache_read": None},
            },
        },
        "files": [dict(f) for f in manifest["files"]],
        "gateway": None,
        "paths": {"work": str(root / "work"), "out": str(out), "tmp": str(scratch)},
        "time_left_s": 600.0,
    }
