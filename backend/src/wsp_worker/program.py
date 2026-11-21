"""Program mode: miForge's own program sandbox and ``SupportSet`` (feature 016; FR-016.23;
FTDD §6.6; P19, P21, R1B-A9, R1B-A10, R1B-A16).

A program is Python source defining ``edit(base, support) -> Mapping[str, ndarray]``. It runs in a
fresh namespace with **restricted builtins** (no ``open``, ``exec``, ``eval``, ``compile``,
``input``, ``globals``; ``__import__`` admits only ``math`` and ``numpy``), ``np``, ``math`` and a
seeded ``rng`` (``np.random.Generator``; seed from the parent's content id). The restriction keeps
an honest program honest; it is NOT the containment: the worker pod is (FTDD §6.4). The
supervisor's deadline is the real time bound; an in-process alarm could be ignored (R1B-A16).

This module never imports ``weightspace``: importing anything from it runs its ``__init__``, which
loads ``weightspace.llm`` (R1B-A9). A rerun asserts ``"weightspace.llm" not in sys.modules`` when it
ends (:func:`rerun`), so an import smuggled in here or by a program fails the run.

Determinism by construction (R1B-A10): ``CHILD_ENV`` pins the BLAS thread counts to 1, and every
accepted program is run **twice**; when the two patches' ``content_sha256`` differ the program is
refused ``wsp_program_nondeterministic``.
"""

from __future__ import annotations

import builtins
import fnmatch
import hashlib
import json
import math
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np

from wsp_worker import patch_writer
from wsp_worker.bundle import Bundle, BundleError, WorkerTensor
from wsp_worker.scoring import Scorer

PROGRAM_NAME = "program.py"
MAX_PROGRAM_BYTES = 1 << 20
SUPPORT_KINDS = ("practice", "practice_train")
FAILURE_CODES = ("wsp_program_invalid", "wsp_program_nondeterministic")
ALLOWED_IMPORTS = ("math", "numpy")
SAFE_BUILTINS = (
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "filter", "float", "frozenset",
    "int", "isinstance", "len", "list", "map", "max", "min", "pow", "print", "range", "reversed",
    "round", "set", "slice", "sorted", "str", "sum", "tuple", "zip", "True", "False", "None",
    "ArithmeticError", "Exception", "IndexError", "KeyError", "TypeError", "ValueError",
    "ZeroDivisionError", "__build_class__",
)  # fmt: skip


class ProgramError(ValueError):
    def __init__(self, message: str, code: str = "wsp_program_invalid") -> None:
        super().__init__(message)
        self.code = code


def _guarded_import(
    name: str,
    globals_: Any = None,
    locals_: Any = None,
    fromlist: Any = (),
    level: int = 0,
) -> Any:
    if level != 0 or name.split(".")[0] not in ALLOWED_IMPORTS:
        raise ImportError(f"a program may import only {', '.join(ALLOWED_IMPORTS)}")
    return builtins.__import__(name, globals_, locals_, fromlist, level)


def restricted_builtins() -> dict[str, Any]:
    allowed = {name: getattr(builtins, name) for name in SAFE_BUILTINS}
    allowed["__import__"] = _guarded_import
    return allowed


def seed_for(parent_ref: str) -> int:
    """The program's seed: from the parent's content identity (the bundle or candidate digest)."""
    return int.from_bytes(hashlib.sha256(parent_ref.encode("utf-8")).digest()[:8], "little")


def _read_only(array: np.ndarray) -> np.ndarray:
    out = np.array(array, copy=True)
    out.flags.writeable = False
    return out


class SupportSet:
    """What a program may learn from: read-only practice arrays and the evaluator as a callable.

    ``rows`` are the support's practice row ids (``practice_train`` leaves out the validation rows);
    ``labels[rows]`` their labels; ``tokens(row)`` one row's activations (a copy); ``score(tensors,
    rows)`` the bundle function's float32 row scores for any head tensors (the base's non-editable
    tensors are filled in).
    """

    def __init__(self, kind: str, bundle: Bundle) -> None:
        if kind not in SUPPORT_KINDS:
            raise ProgramError(f"support_set.kind must be one of {list(SUPPORT_KINDS)}")
        arrays = bundle.practice.arrays
        self.kind = kind
        self._scorer = Scorer(
            bundle.manifest["function"], bundle.practice.activations, arrays["practice_offsets"]
        )
        everything = np.arange(self._scorer.rows, dtype=np.int64)
        validation = np.asarray(arrays["practice_validation_mask"], dtype=bool)
        self.rows = _read_only(everything if kind == "practice" else everything[~validation])
        self._members = frozenset(int(r) for r in self.rows)
        self.labels = _read_only(np.asarray(arrays["practice_labels"]))
        self.d_model = int(bundle.manifest["practice"]["d_model"])
        self._fixed = {n: t.values().astype(np.float64) for n, t in bundle.tensors.items()}

    def tokens(self, row: int) -> np.ndarray:
        if int(row) not in self._members:
            raise ProgramError(f"row {row} is not in this support set")
        return self._scorer.tokens(int(row))

    def score(self, tensors: Mapping[str, np.ndarray], rows: Any = None) -> np.ndarray:
        chosen = self.rows if rows is None else np.asarray(rows, dtype=np.int64)
        if not {int(r) for r in chosen} <= self._members:
            raise ProgramError("score() may use only this support set's rows")
        return self._scorer.score({**self._fixed, **dict(tensors)}, chosen)


@dataclass(frozen=True)
class Outcome:
    tensors: dict[str, WorkerTensor]
    manifest: dict[str, Any] | None
    data: bytes | None

    @property
    def content_sha256(self) -> str | None:
        return None if self.manifest is None else str(self.manifest["content_sha256"])


def editable_names(bundle: Bundle, limits: Mapping[str, Any]) -> list[str]:
    """The tensors a program may return: editable, and allowed by the change limits.

    ``allowed_tensors`` and ``frozen_tensors`` are ``fnmatchcase`` patterns, exactly as
    ``limits.check_limits`` reads them (``limits.py``); exact membership withheld a tensor allowed
    only by a pattern (feature 018, FR-018.15)."""
    allowed = limits.get("allowed_tensors")
    frozen = list(limits.get("frozen_tensors") or [])

    def matches(name: str, patterns: Any) -> bool:
        return any(fnmatch.fnmatchcase(name, str(p)) for p in patterns)

    return [
        n
        for n in bundle.tensors
        if bundle.editable(n)
        and (allowed is None or matches(n, allowed))
        and not matches(n, frozen)
    ]


def compile_program(source: str) -> Any:
    if len(source.encode("utf-8")) > MAX_PROGRAM_BYTES:
        raise ProgramError("the program is larger than 1 MiB")
    try:
        return compile(source, PROGRAM_NAME, "exec", dont_inherit=True)
    except (SyntaxError, ValueError) as exc:
        raise ProgramError(f"the program does not compile: {exc}") from exc


def run_once(
    source: str, bundle: Bundle, support: SupportSet, request: Mapping[str, Any]
) -> dict[str, WorkerTensor]:
    """Run ``edit`` once in a fresh namespace; validate the return; the edited tensors."""
    code = compile_program(source)
    names = editable_names(bundle, request.get("change_limits") or {})
    base = MappingProxyType({n: _read_only(bundle.tensors[n].values()) for n in names})
    parent_ref = str((request.get("parent") or {}).get("ref") or bundle.manifest["bundle_sha256"])
    namespace: dict[str, Any] = {
        "__builtins__": restricted_builtins(),
        "__name__": "wsp_program",
        "np": np,
        "math": math,
        "rng": np.random.default_rng(seed_for(parent_ref)),
    }
    try:
        exec(code, namespace)
        edit = namespace.get("edit")
        if not callable(edit):
            raise ProgramError("the program does not define edit(base, support)")
        returned = edit(base, support)
    except ProgramError:
        raise
    except (Exception, SystemExit) as exc:  # a program's own failure, reported, never trusted
        raise ProgramError(f"the program raised {type(exc).__name__}: {str(exc)[:500]}") from exc
    if not isinstance(returned, Mapping):
        raise ProgramError("edit() must return a mapping of tensor name to array")
    out = dict(bundle.tensors)
    for name, value in returned.items():
        if name not in base:
            raise ProgramError(f"edit() returned {name!r}, which it may not change")
        if not isinstance(value, np.ndarray):
            raise ProgramError(f"edit() returned a {type(value).__name__} for {name!r}")
        if value.shape != base[name].shape or value.dtype != base[name].dtype:
            raise ProgramError(
                f"{name!r} must keep shape {base[name].shape} and dtype {base[name].dtype}"
            )
        if not np.all(np.isfinite(value)):
            raise ProgramError(f"{name!r} holds a non-finite value")
        out[name] = patch_writer.from_values(bundle.tensors[name].dtype, np.array(value))
    return out


def created_by_for(request: Mapping[str, Any]) -> dict[str, str]:
    """``created_by`` from the frozen request's own provider (``{"id", "version"}``).

    Refuses ``worker_spec_invalid`` when the request names none: a patch stamped with a guessed
    provider is fabricated provenance (ADR-018; feature 018, FR-018.15)."""
    provider = request.get("provider")
    if (
        not isinstance(provider, Mapping)
        or not isinstance(provider.get("id"), str)
        or not isinstance(provider.get("version"), str)
    ):
        raise BundleError("the request names no provider to stamp", "worker_spec_invalid")
    return {"provider": provider["id"], "provider_version": provider["version"]}


def outcome(
    edited: dict[str, WorkerTensor],
    bundle: Bundle,
    request: Mapping[str, Any],
    run_id: str,
    now: str,
    *,
    created_by: Mapping[str, str],
) -> Outcome:
    """The patch of ``edited`` against the bundle, stamped ``created_by`` (the provider that made
    it, with ``run_id``). ``content_sha256`` excludes ``created_by``, so reruns still compare."""
    parent = request.get("parent") or {"kind": "bundle", "ref": bundle.manifest["bundle_sha256"]}
    built = patch_writer.build_patch(
        bundle.tensors,
        edited,
        bundle_base_sha256=bundle.manifest["bundle_sha256"],
        parent={"kind": "bundle", "ref": parent.get("ref"), "patch_sha256": None},
        parent_patch_sha256=None,
        created_by={
            "provider": created_by["provider"],
            "provider_version": created_by["provider_version"],
            "run_id": run_id,
        },
        created_at=now,
    )
    if built is None:
        return Outcome(edited, None, None)
    return Outcome(edited, built[0], built[1])


def run_checked(
    source: str,
    bundle: Bundle,
    support: SupportSet,
    request: Mapping[str, Any],
    run_id: str,
    now: str,
    *,
    created_by: Mapping[str, str],
) -> Outcome:
    """Run the program twice; refuse ``wsp_program_nondeterministic`` when they disagree."""
    first = outcome(
        run_once(source, bundle, support, request),
        bundle,
        request,
        run_id,
        now,
        created_by=created_by,
    )
    second = outcome(
        run_once(source, bundle, support, request),
        bundle,
        request,
        run_id,
        now,
        created_by=created_by,
    )
    if first.content_sha256 != second.content_sha256:
        raise ProgramError(
            "two runs of the program gave different edits; seed every random draw from `rng`",
            "wsp_program_nondeterministic",
        )
    return second


def write_failure(out: Path, error: ProgramError, mode: str) -> None:
    (out / "result.json").write_text(
        json.dumps(
            {
                "mode": mode,
                "stop_reason": "error",
                "failure": {"code": error.code, "message": str(error)[:2000]},
            }
        ),
        encoding="utf-8",
    )


def no_llm_loaded() -> None:
    """R1B-A9: a rerun ends with ``weightspace.llm`` absent, or it fails (exit, no result)."""
    if "weightspace.llm" in sys.modules:
        raise SystemExit("weightspace.llm was imported during a rerun (R1B-A9)")


def rerun(line: dict[str, Any], now: str) -> int:
    """A ``program_rerun``: no LLM is constructed, no gateway exists (R3A-13, R1B-A9).

    Runs the stored program (a pushed file, checked against the frozen ``program_sha256``) twice on
    the named support set, rebuilding from the base (P21), and writes the patch to ``/out``.
    """
    from wsp_worker import bundle as bundle_mod

    spec, paths = line["spec"], line["paths"]
    request = spec["request"]
    work, out = Path(paths["work"]), Path(paths["out"])
    loaded = bundle_mod.load(work, spec["manifest"], list(line["files"]))
    # The run's own provider (the first run's, which ``runs.create_rerun`` copies), never a
    # hard-coded one: a rerun of an engine program is the engine's (feature 018, FR-018.15).
    created_by = created_by_for(request)
    pushed = {f["role"]: f for f in line["files"]}
    entry = pushed.get("program")
    try:
        if entry is None or entry["name"] != PROGRAM_NAME:
            raise ProgramError("the rerun's push carries no program")
        raw = (work / PROGRAM_NAME).read_bytes()
        if hashlib.sha256(raw).hexdigest() != request.get("program_sha256"):
            raise ProgramError("the pushed program is not the frozen request's program_sha256")
        support = SupportSet((request.get("support_set") or {}).get("kind", ""), loaded)
        result = run_checked(
            raw.decode("utf-8"),
            loaded,
            support,
            request,
            str(line["run_id"]),
            now,
            created_by=created_by,
        )
    except ProgramError as exc:
        no_llm_loaded()
        write_failure(out, exc, "program_rerun")
        return 0
    no_llm_loaded()
    if result.manifest is not None and result.data is not None:
        patch_writer.write_patch(out / "patch", result.manifest, result.data)
    (out / "result.json").write_text(
        json.dumps({"mode": "program_rerun", "stop_reason": "submitted", "failure": None}),
        encoding="utf-8",
    )
    return 0
