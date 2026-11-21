"""numpy ports of weightspace 0.3.0's README examples, each run through miForge's REAL adapter with
a scripted LLM (feature 016; FTASKS 8.5; FR-016.25; R1B-A15; operator OVN-wsp-forge-68, 2026-11-17).

Every upstream example imports torch, which the worker image does not hold (FR-016.9), so each is
ported to numpy and driven through ``wsp_worker.adapters.weightspace_adapter.run`` exactly as a
session is: a bundle written to ``work`` (safetensors heads, every file and tensor hashed into the
manifest), a child line shaped as the supervisor writes it, ``/out`` written by the adapter
(``result.json``, ``transcript.jsonl``, ``patch/``). The outcome is then measured on the PATCH the
adapter wrote, the artefact the backend would take in, applied to the bundle's own bits and checked
tensor by tensor against the patch's ``edited_sha256``.

The operator's mapping (OPERATOR.md 16):

- ``ex_linear_fix`` = ``examples/fix_quadrants.py``: Linear(2->16) -> ReLU -> Linear(16->4) learns
  a point's quadrant; class 3's output bias is lowered by 8; a ``repair`` session runs the
  example's own script (its edit line verbatim) and class 3's accuracy comes back.
- ``ex_read_mlp`` = ``examples/read_which_task.py``: an 8 -> 8 (tanh) -> 2 MLP learns
  ``parity_of_first_two``; an ``inspect`` session reads the weights (read-only ``R``) and names the
  rule; no patch.
- ``ex_write_xor`` = the README's ``wsp.write`` snippet ("zeroes first") on a 2 -> 2 (ReLU) -> 1
  XOR MLP: a ``create`` session starts from zeroed tensors and writes XOR by hand.

Where a port must differ from upstream, it says so here, because the adapter is the contract:

- the adapter offers only its observation-tool catalogue as forward passes (FR-016.18), and the
  example's ``accuracy`` probe is a closure over a torch model, so ``fix_quadrants``' two
  ``forward_accuracy`` calls become ``list_tensors`` and ``diff`` turns; the accuracy is measured
  before and after the session, as the example's own ``print("before:"/"after:")`` lines do;
- the adapter runs tool sessions (FR-016.17), not ``text_mode``, so ``read_which_task`` reads
  through the ``python`` tool over ``R``, and its scripted reader derives the answer from what that
  tool returned (a callable step), so a different network gives a different answer;
- phase 1's worker accepts one function kind, ``linear_probe/v1``; the manifest carries it, the
  example's own context as ``function.description`` (what the editor is shown) and no observation
  tool, so nothing ever scores these heads with the probe scorer;
- upstream's premise "never predicts class 3" does not hold even for upstream: its ``train()`` and
  ``bias[3] -= 8.0``, run verbatim under torch 2.14.1, leave class 3 at 0.819 (this port: 0.844).
  The port keeps the example's ``-8`` and asserts what the example does: class 3 well below the
  others before, back with them after.

Run in the worker image by ``tests/image/test_wsp_worker_image.py`` (a real session: the supervisor
as PID 1, the child's preamble and seccomp filter, the action ``readme_example:<name>`` of
``scripted.py``), and on the host under the worker's venv by
``tests/unit/wsp_worker/test_readme_examples.py``. ``python readme_examples.py <name> <root>``
writes the session under ``<root>`` and prints the outcome as one JSON line. Each port runs twice
(the second time under ``<root>/again``) and reports whether the two agree (``deterministic``).
"""

from __future__ import annotations

import hashlib
import io
import json
import re
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

#: The prefix of the one stdout line a child action prints (the host reads ``logs/stdout.txt``).
MARK = "README-EXAMPLE "
EXAMPLES = ("ex_linear_fix", "ex_read_mlp", "ex_write_xor")
F32 = "float32"


# ----------------------------------------------------------------------------- a tiny numpy MLP


def linear(rng: np.random.Generator, fan_in: int, fan_out: int) -> list[np.ndarray]:
    """``torch.nn.Linear``'s default init: weight and bias ~ U(-1/sqrt(fan_in), 1/sqrt(fan_in))."""
    k = 1.0 / np.sqrt(fan_in)
    return [rng.uniform(-k, k, (fan_out, fan_in)), rng.uniform(-k, k, fan_out)]


def _relu(a: np.ndarray) -> np.ndarray:
    return np.maximum(a, 0.0)


ACT: dict[str, tuple[Callable[[np.ndarray], np.ndarray], Callable[..., np.ndarray]]] = {
    "relu": (_relu, lambda a, h: (a > 0).astype(np.float64)),
    "tanh": (np.tanh, lambda a, h: 1.0 - h * h),
}


def forward(params: dict[str, np.ndarray], x: np.ndarray, act: str) -> np.ndarray:
    """``Linear -> act -> Linear`` over the torch names ``0.*`` and ``2.*``, in float64."""
    p = {n: np.asarray(v, dtype=np.float64) for n, v in params.items()}
    h = ACT[act][0](x @ p["0.weight"].T + p["0.bias"])
    out: np.ndarray = h @ p["2.weight"].T + p["2.bias"]
    return out


def train(
    x: np.ndarray,
    y: np.ndarray,
    sizes: tuple[int, int, int],
    act: str,
    lr: float,
    steps: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Full-batch cross-entropy with Adam at torch's defaults (betas 0.9/0.999, eps 1e-8)."""
    params = [*linear(rng, sizes[0], sizes[1]), *linear(rng, sizes[1], sizes[2])]
    m = [np.zeros_like(p) for p in params]
    v = [np.zeros_like(p) for p in params]
    fn, dfn = ACT[act]
    for t in range(1, steps + 1):
        w1, b1, w2, b2 = params
        a = x @ w1.T + b1
        h = fn(a)
        logits = h @ w2.T + b2
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        d = z / z.sum(axis=1, keepdims=True)
        d[np.arange(y.size), y] -= 1.0
        d /= y.size
        da = (d @ w2) * dfn(a, h)
        grads = [da.T @ x, da.sum(axis=0), d.T @ h, d.sum(axis=0)]
        for i, g in enumerate(grads):
            m[i] = 0.9 * m[i] + 0.1 * g
            v[i] = 0.999 * v[i] + 0.001 * g * g
            params[i] -= lr * (m[i] / (1 - 0.9**t)) / (np.sqrt(v[i] / (1 - 0.999**t)) + 1e-8)
    names = ("0.weight", "0.bias", "2.weight", "2.bias")
    return {n: p.astype(np.float32) for n, p in zip(names, params, strict=True)}


# ----------------------------------------------------------------------------- bundle and line


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
        "run_id": "wsr_readme_example",
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


def session(
    root: Path,
    tensors: dict[str, np.ndarray],
    description: str,
    kind: str,
    task: str,
    steps: list[Any],
    out: Path | None,
) -> dict[str, Any]:
    """Bundle, line and one adapter session; what the session left in ``out``."""
    from weightspace import ScriptedLLM

    from wsp_worker.adapters import weightspace_adapter

    manifest = write_bundle(root / "work", tensors, description)
    line = child_line(root, manifest, kind, task, out)
    llm = ScriptedLLM(steps)
    code = weightspace_adapter.run(line, llm=llm)
    out = Path(line["paths"]["out"])
    result = json.loads((out / "result.json").read_text())
    last = llm.seen[-1] if llm.seen else []
    patch_dir = out / "patch"
    patch = json.loads((patch_dir / "patch.json").read_text()) if patch_dir.is_dir() else None
    return {
        "exit": code,
        "result": result,
        "tool_outputs": [m.get("content") for m in last if m.get("role") == "tool"],
        "patch": patch,
        "edited": apply_patch(patch_dir, tensors) if patch is not None else dict(tensors),
    }


def apply_patch(patch_dir: Path, base: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """``base`` with the v1 patch applied, each edited tensor checked against ``edited_sha256``."""
    from wsp_worker.bundle import WorkerTensor

    manifest = json.loads((patch_dir / "patch.json").read_text())
    data = (patch_dir / "edit.npz").read_bytes()
    if hashlib.sha256(data).hexdigest() != manifest["arrays_sha256"]:
        raise AssertionError("edit.npz differs from the patch's arrays_sha256")
    arrays = np.load(io.BytesIO(data), allow_pickle=False)
    out = {n: np.array(a, copy=True) for n, a in base.items()}
    for t in manifest["tensors"]:
        i, name = t["index"], t["name"]
        bits = np.ascontiguousarray(base[name], dtype=np.float32).view(np.uint32).reshape(-1).copy()
        if t["form"] == "sparse":
            bits[arrays[f"t{i}_idx"]] = arrays[f"t{i}_val"]
        elif t["form"] == "dense":
            bits = np.asarray(arrays[f"t{i}_bits"]).reshape(-1).copy()
        else:
            raise AssertionError(f"form {t['form']!r} is not one the worker writes")
        edited = WorkerTensor(F32, bits.reshape(t["shape"]))
        if edited.sha256 != t["edited_sha256"]:
            raise AssertionError(f"{name}: the applied patch differs from its edited_sha256")
        out[name] = edited.values().copy()
    return out


# ----------------------------------------------------------------------------- ex_linear_fix


QUADRANT_CONTEXT = (
    "Network: Linear(2->16) -> ReLU -> Linear(16->4). Input: a point (x0, x1) ~ N(0, I). "
    "Output: logits for the quadrant: class = 2*(x0>0) + (x1>0)."
)
#: The example's edit, verbatim.
FIX_CODE = "b = W['2.bias']; b[3] = np.delete(b, 3).mean(); b"
FIX_ANSWER = "Class 3's output bias was -8; reset it to the others' mean."


def quadrants() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((4096, 2))
    y = (x[:, 0] > 0).astype(np.int64) * 2 + (x[:, 1] > 0).astype(np.int64)
    return x, y


def quadrant_accuracy(params: dict[str, np.ndarray], x: np.ndarray, y: np.ndarray) -> list[float]:
    """Per-class accuracy on the 4,096 points (the example's ``accuracy`` probe)."""
    pred = forward(params, x, "relu").argmax(axis=1)
    return [round(float((pred[y == c] == c).mean()), 3) for c in range(4)]


def ex_linear_fix(root: Path, out: Path | None = None) -> dict[str, Any]:
    from weightspace.llm.mock import call, reply

    x, y = quadrants()
    tensors = train(x, y, (2, 16, 4), "relu", 0.05, 300, np.random.default_rng(1))
    trained = quadrant_accuracy(tensors, x, y)
    tensors["2.bias"] = tensors["2.bias"].copy()
    tensors["2.bias"][3] -= np.float32(8.0)  # the bug
    before = quadrant_accuracy(tensors, x, y)
    steps = [
        reply("Check behaviour first.", call("list_tensors")),  # upstream: forward_accuracy
        reply(None, call("describe_tensor", name="2.bias"), call("read_tensor", name="2.bias")),
        reply(
            "Output bias for class 3 is far below the others; set it to their mean.",
            call("checkpoint", label="start"),
            call("python", code=FIX_CODE),
        ),
        reply(None, call("diff")),  # upstream: forward_accuracy
        reply(None, call("submit", answer=FIX_ANSWER, confidence=0.9)),
    ]
    task = "The classifier never predicts class 3. Find out why and fix it."
    done = session(root, tensors, QUADRANT_CONTEXT, "repair", task, steps, out)
    return {
        **_common(done),
        "trained": trained,
        "before": before,
        "after": quadrant_accuracy(done["edited"], x, y),
    }


# ----------------------------------------------------------------------------- ex_read_mlp


#: ``read_which_task.py``'s TASKS over numpy bit arrays, and the formula a reader answers.
TASKS: dict[str, tuple[Callable[[np.ndarray], np.ndarray], str]] = {
    "majority": (lambda x: (x.sum(axis=1) > 4).astype(np.int64), "x0+...+x7 > 4"),
    "first_bit": (lambda x: x[:, 0].astype(np.int64), "x0"),
    "parity_of_first_two": (lambda x: (x[:, 0] != x[:, 1]).astype(np.int64), "x0 XOR x1"),
    "last_bit_and_not_third": (
        lambda x: ((x[:, 7] == 1) & (x[:, 2] == 0)).astype(np.int64),
        "x7 AND NOT x2",
    ),
}
READ_TASK = "parity_of_first_two"  # the example's default --task
READ_CONTEXT = (
    "MLP 8 -> 8 (tanh) -> 2 logits. Inputs are bits in {0,1}. Output 1 means the function is true."
)
#: The scripted reader's code: the network's truth table over all 256 inputs, from ``R`` alone.
READER = (
    "import json\n"
    "X = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.float64)\n"
    "h = np.tanh(X @ np.asarray(R['0.weight'], dtype=np.float64).T + R['0.bias'])\n"
    "y = (h @ np.asarray(R['2.weight'], dtype=np.float64).T + R['2.bias']).argmax(1)\n"
    "used = [i for i in range(8) if (y != y[np.arange(256) ^ (1 << i)]).any()]\n"
    "print(json.dumps({'table': ''.join(map(str, y.tolist())), 'inputs': used}))\n"
)
ALL_BITS = ((np.arange(256)[:, None] >> np.arange(8)) & 1).astype(np.float64)


def name_the_rule(messages: list[dict[str, Any]], _tools: list[dict[str, Any]]) -> Any:
    """The scripted reader's last turn: name the example's rule whose truth table the weights
    gave (``unknown`` when none matches). It reads the python tool's answer, nothing else."""
    from weightspace.llm.mock import call, reply

    text = str(messages[-1].get("content") or "")
    found = next((ln for ln in text.splitlines() if ln.startswith("{")), "{}")
    table = json.loads(found).get("table", "")
    answer = "unknown"
    for name, (rule, formula) in TASKS.items():
        if "".join(map(str, rule(ALL_BITS).tolist())) == table:
            answer = f"{name}: {formula}"
    return reply(None, call("submit", answer=answer, confidence=1.0))


def ex_read_mlp(root: Path, out: Path | None = None, task: str = READ_TASK) -> dict[str, Any]:
    from weightspace.llm.mock import call, reply

    rng = np.random.default_rng(0)
    x = rng.integers(0, 2, (4096, 8)).astype(np.float64)
    rule = TASKS[task][0]
    tensors = train(x, rule(x), (8, 8, 2), "tanh", 0.03, 600, rng)
    accuracy = float((forward(tensors, ALL_BITS, "tanh").argmax(axis=1) == rule(ALL_BITS)).mean())
    steps = [
        reply(None, call("list_tensors")),
        reply(None, call("python", code=READER)),
        name_the_rule,
    ]
    question = (
        "What boolean function of the 8 input bits does this network compute? "
        "Answer as a short formula."
    )
    done = session(root, tensors, READ_CONTEXT, "inspect", question, steps, out)
    reading = next((o for o in done["tool_outputs"] if '"table"' in str(o)), "")
    found = next((ln for ln in str(reading).splitlines() if ln.startswith("{")), "{}")
    return {
        **_common(done),
        "truth": task,
        "accuracy_256": accuracy,
        "inputs_read": json.loads(found).get("inputs"),
    }


# ----------------------------------------------------------------------------- ex_write_xor


XOR_CONTEXT = (
    "Network: Linear(2->2) -> ReLU -> Linear(2->1). Input: two bits (x0, x1) in {0,1}. "
    "Output: one value; the network's answer is 1 when it is above 0.5."
)
#: Written by hand from zero: h0 = relu(x0 + x1), h1 = relu(x0 + x1 - 1), y = h0 - 2*h1.
#: ``2.bias`` is left as the zeroing left it.
XOR_CODE = (
    "W['0.weight'][:] = [[1.0, 1.0], [1.0, 1.0]]\n"
    "W['0.bias'][:] = [0.0, -1.0]\n"
    "W['2.weight'][:] = [[1.0, -2.0]]\n"
    "diff()"
)
XOR_INPUTS = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float64)


def _numbers(text: str) -> list[float]:
    """The values of a ``read_tensor`` answer (its header line and row labels dropped)."""
    body = [re.sub(r"^\[[^\]]*\]\s*", "", ln) for ln in text.splitlines()[1:]]
    return [float(v) for ln in body for v in ln.split()]


def ex_write_xor(root: Path, out: Path | None = None) -> dict[str, Any]:
    from weightspace.llm.mock import call, reply

    rng = np.random.default_rng(0)
    w1, b1 = linear(rng, 2, 2)
    w2, b2 = linear(rng, 2, 1)
    tensors = {
        n: a.astype(np.float32)
        for n, a in {"0.weight": w1, "0.bias": b1, "2.weight": w2, "2.bias": b2}.items()
    }
    steps = [
        reply(None, call("read_tensor", name="0.weight"), call("read_tensor", name="2.bias")),
        reply(None, call("python", code=XOR_CODE)),
        reply(None, call("submit", answer="XOR: y = relu(x0+x1) - 2*relu(x0+x1-1)")),
    ]
    task = "Compute XOR of the two input bits."
    done = session(root, tensors, XOR_CONTEXT, "create", task, steps, out)
    y = forward(done["edited"], XOR_INPUTS, "relu").reshape(-1)
    first_reads = [str(o) for o in done["tool_outputs"][:2]]
    return {
        **_common(done),
        "init_nonzero": {n: bool(np.any(a != 0)) for n, a in tensors.items()},
        "zeros_seen": [bool(_numbers(t)) and not any(_numbers(t)) for t in first_reads],
        "outputs": [float(v) for v in y],
        "predictions": [int(v > 0.5) for v in y],
        "edited_2_bias": [float(v) for v in done["edited"]["2.bias"].reshape(-1)],
    }


# ----------------------------------------------------------------------------- running one


def _common(done: dict[str, Any]) -> dict[str, Any]:
    patch = done["patch"]
    result = done["result"]
    return {
        "exit": done["exit"],
        "stop_reason": result["stop_reason"],
        "answer": result["answer"],
        "confidence": result["confidence"],
        "mode": result["mode"],
        "content_sha256": None if patch is None else patch["content_sha256"],
        "changed": (
            None if patch is None else {t["name"]: t["changed_entries"] for t in patch["tensors"]}
        ),
    }


PORTS: dict[str, Callable[[Path, Path | None], dict[str, Any]]] = {
    "ex_linear_fix": ex_linear_fix,
    "ex_read_mlp": ex_read_mlp,
    "ex_write_xor": ex_write_xor,
}


def run_example(name: str, root: Path, out: Path | None = None) -> dict[str, Any]:
    """Run ``name`` twice (``out`` as the first session's ``/out`` when given) and report the first
    outcome with ``deterministic``: the second run agrees on everything but the transient keys."""
    first = PORTS[name](root / "first", out)
    again = PORTS[name](root / "again", None)
    return {"example": name, **first, "deterministic": again == first}


def child_action(out: Path, tmp: Path, arg: str) -> None:
    """``readme_example:<name>`` in ``scripted.py``: the session's ``/out`` is the pod's."""
    outcome = run_example(arg, tmp / "readme", out=out)
    print(MARK + json.dumps(outcome, sort_keys=True), flush=True)


if __name__ == "__main__":
    print(json.dumps(run_example(sys.argv[1], Path(sys.argv[2])), sort_keys=True))
