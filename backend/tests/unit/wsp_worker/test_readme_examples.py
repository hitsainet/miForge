"""The README example ports on the host, under the worker's interpreter (feature 016; FTASKS 8.5;
R1B-A15; OVN-wsp-forge-68): the local twin of the image job's
``image/test_wsp_worker_image.py::test_readme_example_reproduces_in_the_worker``.

Each port (``tests/support/wsp_children/readme_examples.py``) runs in a throwaway process under
``MIFORGE_WSP_WORKER_PYTHON`` (skips by name without it; fails under
``MIFORGE_REQUIRE_WSP_WORKER=1``), through the real adapter with a scripted LLM, and must show the
same documented outcome the image test asserts (``tests/support/wsp_readme_expected.py``). Here the
backend's own code is also on hand, so the session's patch is re-applied with
``wsp.weight_ops.patch.apply_patch`` (the intake's authority) and the outcome is measured again on
THAT result: the port's own patch reader cannot be what makes the example pass.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support import wsp_readme_expected as readme
from tests.support.wsp_children import readme_examples as ports
from tests.support.wsp_worker_env import SRC, worker_python
from wsp.weight_ops.patch import apply_patch, build_patch, write_patch
from wsp.weight_ops.st_io import read_safetensors
from wsp.weight_ops.tensor import from_array

PORT = Path(ports.__file__)


def _run(example: str, root: Path) -> dict[str, Any]:
    done = subprocess.run(
        [worker_python(), str(PORT), example, str(root)],
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    loaded: dict[str, Any] = json.loads(done.stdout.strip().splitlines()[-1])
    return loaded


def _backend_applied(root: Path) -> dict[str, np.ndarray] | None:
    """The first session's patch re-applied by the backend to the bundle's heads (``None``: no
    patch), each tensor as float64."""
    patch_dir = root / "first" / "out" / "patch"
    if not patch_dir.is_dir():
        return None
    base = read_safetensors(root / "first" / "work" / "heads.safetensors")
    _manifest, rebuilt = apply_patch(patch_dir, base)
    return {n: t.values64() for n, t in rebuilt.items()}


@pytest.mark.parametrize("example", readme.EXAMPLES)
def test_readme_example_reproduces_under_the_worker_venv(tmp_path: Path, example: str) -> None:
    outcome = _run(example, tmp_path)
    out = tmp_path / "first" / "out"
    result = json.loads((out / "result.json").read_text())
    patch_json = out / "patch" / "patch.json"
    patch = json.loads(patch_json.read_text()) if patch_json.exists() else None
    readme.assert_example(example, outcome, result, patch)
    # Exactly what an adapter session writes (R3A-14): never res.save's ws/ files.
    names = sorted(p.relative_to(out).as_posix() for p in out.rglob("*"))
    extra = ["patch", "patch/edit.npz", "patch/patch.json"] if patch is not None else []
    assert names == sorted(["result.json", "transcript.jsonl", *extra])
    applied = _backend_applied(tmp_path)
    if example == "ex_linear_fix":
        assert applied is not None
        x, y = ports.quadrants()
        assert ports.quadrant_accuracy(applied, x, y) == outcome["after"]
    elif example == "ex_write_xor":
        assert applied is not None
        y_out = ports.forward(applied, ports.XOR_INPUTS, "relu").reshape(-1)
        assert [int(v > 0.5) for v in y_out] == [0, 1, 1, 0]
        assert applied["2.bias"].tolist() == [0.0]
    else:
        assert applied is None


def test_the_reader_answers_from_the_weights_it_reads(tmp_path: Path) -> None:
    """The read port's answer is derived from the session's ``python`` tool over ``R``, not
    scripted: a network trained on another of the example's rules is named as that rule."""
    code = (
        "import json, sys\nfrom pathlib import Path\n"
        f"sys.path.insert(0, {str(PORT.parent)!r})\n"
        "import readme_examples as p\n"
        "print(json.dumps(p.ex_read_mlp(Path(sys.argv[1]), None, task='last_bit_and_not_third')))\n"
    )
    done = subprocess.run(
        [worker_python(), "-c", code, str(tmp_path)],
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(SRC)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-3000:]
    outcome = json.loads(done.stdout.strip().splitlines()[-1])
    assert outcome["accuracy_256"] == 1.0
    assert outcome["inputs_read"] == [2, 7]
    assert outcome["answer"] == "last_bit_and_not_third: x7 AND NOT x2"


@pytest.mark.parametrize("changed", [1, 64], ids=["sparse", "dense"])
def test_the_ports_patch_reader_agrees_with_the_backends_writer(
    tmp_path: Path, changed: int
) -> None:
    """The ports measure each outcome through their own patch reader (the image has no backend
    code). Their networks are small enough that every session patch is dense, so both forms are
    pinned here against the backend's ``build_patch``: one changed entry of 64 is sparse, all 64
    dense (the sparse branch first survived its control unexercised)."""
    rng = np.random.default_rng(3)
    base = rng.standard_normal((8, 8)).astype(np.float32)
    edited = base.copy()
    edited.reshape(-1)[:changed] += np.float32(0.5)
    built = build_patch(
        {"w": from_array(base)},
        {"w": from_array(edited)},
        bundle_base_sha256="0" * 64,
        parent={"kind": "bundle", "ref": "0" * 64, "patch_sha256": None},
        parent_patch_sha256=None,
        created_by={"provider": "weightspace", "provider_version": "0.3.0", "run_id": "wsr_t"},
        created_at="2026-11-17T00:00:00Z",
    )
    assert built is not None
    form = built.manifest["tensors"][0]["form"]
    assert form == ("sparse" if changed == 1 else "dense")
    write_patch(tmp_path / "patch", built)
    got = ports.apply_patch(tmp_path / "patch", {"w": base})
    assert np.array_equal(got["w"], edited)


def test_the_ports_patch_reader_refuses_a_patch_whose_tensor_hash_disagrees(
    tmp_path: Path,
) -> None:
    """The reader checks each rebuilt tensor against the patch's ``edited_sha256``, so a patch it
    misreads (or one that lies) fails the port instead of being measured."""
    base = np.zeros((2, 2), dtype=np.float32)
    edited = base + np.float32(1.0)
    built = build_patch(
        {"w": from_array(base)},
        {"w": from_array(edited)},
        bundle_base_sha256="0" * 64,
        parent={"kind": "bundle", "ref": "0" * 64, "patch_sha256": None},
        parent_patch_sha256=None,
        created_by={"provider": "weightspace", "provider_version": "0.3.0", "run_id": "wsr_t"},
        created_at="2026-11-17T00:00:00Z",
    )
    assert built is not None
    write_patch(tmp_path / "patch", built)
    doc = json.loads((tmp_path / "patch" / "patch.json").read_text())
    doc["tensors"][0]["edited_sha256"] = "f" * 64
    (tmp_path / "patch" / "patch.json").write_text(json.dumps(doc))
    with pytest.raises(AssertionError, match="differs from its edited_sha256"):
        ports.apply_patch(tmp_path / "patch", {"w": base})
