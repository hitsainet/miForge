"""Program mode: miForge's own program sandbox and ``SupportSet`` (feature 016; FR-016.23;
FTDD §6.6; P19, P21, R1B-A9, R1B-A10, R1B-A16).

A program is Python source defining ``edit(base, support) -> Mapping[str, ndarray]``. It runs in a
fresh namespace with **restricted builtins** (no ``open``, ``exec``, ``eval``, ``compile``,
``input``, ``globals``; ``__import__`` admits only ``math`` and ``numpy``), ``np``, ``math`` and a
seeded ``rng`` (``np.random.Generator``; seed from the parent's content id). The restriction keeps
an honest program honest; it is NOT the containment: the worker pod is (FTDD §6.4). The
supervisor's deadline is the real time bound; an in-process alarm could be ignored (R1B-A16).

This module never imports a model client (R1B-A9): not ``wsp_worker.miwsp_engine.client`` and not
``httpx2``. The rerun child checks that neither is loaded when the rerun ends
(``wsp_worker.__main__.no_model_client_loaded``), so an import smuggled in here or by a program
fails the run.

Determinism by construction (R1B-A10): ``CHILD_ENV`` pins the BLAS thread counts to 1, and every
accepted program is run **twice**; when the two patches' ``content_sha256`` differ the program is
refused ``wsp_program_nondeterministic``.

**A failure says what failed, never what the program put in its message** (fix 2026-11-28, E7 §5.3
gap 2). ``test_program`` and ``submit_program`` hand a refusal's text to the editor, and that text
quoted up to 500 characters of the program's own exception message: a program could raise with
practice activations in it, one forward pass per call. With ``data_free`` (every new program
session; a replay as its transcript header's ``program_errors`` says; always in a rerun) a
program's exception reads as :data:`PROGRAM_RAISED`, naming only a genuine builtin or numpy
exception class (a class the program defined, whose name it can compute, reads as its builtin
base), and its full text goes to the worker child's stderr (the run's stored
``logs/stderr.txt``, served by no route). No refusal echoes a value the program chose: a returned
key that is not an editable tensor, a returned value's type, a row id, or a ``ProgramError`` the
program built itself (``exc.__class__(...)``) with its own message or code.

Review of train F-1 (2026-11-28) closed eight more ways a program's own text reached the editor,
each in plain Python: a class the program planted under a builtin or numpy name (the kind is
now matched by identity against classes recorded when this module loads, never read from the
class); an exception whose ``__str__`` raises; a returned mapping whose ``items()`` raises (it
escaped ``run_once`` and the toolbox quoted it); a ``str`` subclass key whose ``__repr__`` the
refusal quoted; and a ``BaseException`` subclass (``Exception.__base__``) that nothing caught. In
``data_free`` everything a program returns is read inside the same guard, and a key or value is
kept only as an exact ``str`` or plain ``ndarray`` copy.

**What this does not do.** It stops an honest program's message (an editor's debug ``raise``)
from reaching the editor. It is not a boundary against a program that sets out to send data: the
program runs in this process, and plain attribute access reaches this module's globals (for
example ``support.tokens.__func__.__globals__``, or ``sys.modules`` through a numpy submodule),
so it can rewrite any class or function here. Containment is the pod (above); whether practice
data may reach the editor at all is the bundle's sensitivity gate (``wsp.freeze``: a sensitive
bundle never goes to an external endpoint, WR-13).
"""

from __future__ import annotations

import builtins
import fnmatch
import hashlib
import json
import math
import sys
from collections.abc import Iterable, Mapping
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


#: The editor-facing texts of a ``data_free`` failure (fix 2026-11-28, E7 §5.3 gap 2): fixed, so
#: no data rides in one. ``{kind}`` is a genuine builtin or numpy exception class's name.
PROGRAM_RAISED = (
    "the program raised {kind}; its message is not shown (it may carry practice data) and is kept "
    "in the worker's log"
)
NO_EDIT = "the program does not define edit(base, support)"
ROW_NOT_IN_SUPPORT = "a row that is not in this support set"
SCORE_ROWS_OUTSIDE = "score() may use only this support set's rows"
RETURNED_NOT_EDITABLE = "edit() returned a tensor name that is not one it may change"
#: The only ``ProgramError`` texts raised while the program runs that are passed on as they are.
FIXED_RUNTIME_MESSAGES = frozenset({NO_EDIT, ROW_NOT_IN_SUPPORT, SCORE_ROWS_OUTSIDE})
#: How much of a program's own exception text the worker's log keeps.
LOGGED_MESSAGE_CHARS = 2000


def _known_exception_kinds() -> tuple[tuple[type, str], ...]:
    """Every builtin exception class and numpy's own, with its name, recorded when this module
    loads (before any program runs; a program can plant a class under a builtin or numpy name,
    or rename numpy's classes, later)."""
    found: dict[type, str] = {}
    for name, value in vars(builtins).items():
        if isinstance(value, type) and issubclass(value, BaseException):
            found.setdefault(value, name)
    for module in (np, np.exceptions, np.linalg):
        for name, value in vars(module).items():
            if isinstance(value, type) and issubclass(value, BaseException) and value not in found:
                found[value] = f"{value.__module__}.{name}"
    return tuple(found.items())


_KNOWN_EXCEPTION_KINDS = _known_exception_kinds()


def exception_kind(exc: BaseException) -> str:
    """The name of the first class in ``exc``'s MRO that IS a builtin or a numpy exception
    class, matched by identity against :data:`_KNOWN_EXCEPTION_KINDS` and named as recorded
    there, so a program's class can neither borrow a name nor bring its own; else ``an
    exception``. Runs none of the program's code (a metaclass may define ``__mro__``): the MRO is
    read through ``type``'s own descriptor and compared by identity, so it cannot raise."""
    for cls in type.__dict__["__mro__"].__get__(type(exc)):
        for known, name in _KNOWN_EXCEPTION_KINDS:
            if cls is known:
                return name
    return "an exception"


def _log_program_failure(exc: BaseException) -> None:
    """The program's own exception, in full, to the worker child's stderr only."""
    try:
        text = str(exc)[:LOGGED_MESSAGE_CHARS]
    except BaseException:  # a program's ``__str__`` may raise anything
        text = "(its message could not be read)"
    name = type.__dict__["__name__"].__get__(type(exc))
    print(f"wsp program raised {name}: {text}", file=sys.stderr, flush=True)


def _fixed_runtime_message(exc: BaseException) -> str | None:
    """``exc``'s text when it is exactly one of :data:`FIXED_RUNTIME_MESSAGES`, read without
    running any of the program's code (its ``__str__``, ``args`` or a subclass), else None."""
    if type(exc) is not ProgramError:
        return None
    args = BaseException.__dict__["args"].__get__(exc)
    if len(args) == 1 and type(args[0]) is str and args[0] in FIXED_RUNTIME_MESSAGES:
        return str.__str__(args[0])
    return None


def _data_free_failure(exc: BaseException) -> ProgramError:
    """The editor-facing refusal for a program's failure in ``data_free`` (module docstring)."""
    fixed = _fixed_runtime_message(exc)
    if fixed is not None:
        return ProgramError(fixed)  # fresh: a program-built code is not kept
    _log_program_failure(exc)
    return ProgramError(PROGRAM_RAISED.format(kind=exception_kind(exc)))


def _returned_pairs(returned: Any) -> list[tuple[Any, Any]] | None:
    """``data_free``: what ``edit`` returned, read while the program's failures are still caught
    (a mapping's ``items()`` and an array subclass are the program's code): each key kept only
    as an exact ``str`` copy (``str.__str__``: a subclass's methods never run), each value only
    as a plain ``ndarray`` copy; None when the
    return is not a mapping."""
    if not isinstance(returned, Mapping):
        return None
    pairs: list[tuple[Any, Any]] = []
    for name, value in list(returned.items()):
        key = str.__str__(name) if issubclass(type(name), str) else None  # an exact copy
        array = np.array(value) if issubclass(type(value), np.ndarray) else None
        pairs.append((key, array))
    return pairs


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

    def __init__(self, kind: str, bundle: Bundle, *, data_free: bool = True) -> None:
        self.data_free = data_free
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
            raise ProgramError(
                ROW_NOT_IN_SUPPORT if self.data_free else f"row {row} is not in this support set"
            )
        return self._scorer.tokens(int(row))

    def score(self, tensors: Mapping[str, np.ndarray], rows: Any = None) -> np.ndarray:
        chosen = self.rows if rows is None else np.asarray(rows, dtype=np.int64)
        if not {int(r) for r in chosen} <= self._members:
            raise ProgramError(SCORE_ROWS_OUTSIDE)
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
    source: str,
    bundle: Bundle,
    support: SupportSet,
    request: Mapping[str, Any],
    *,
    data_free: bool = True,
) -> dict[str, WorkerTensor]:
    """Run ``edit`` once in a fresh namespace; validate the return; the edited tensors.
    ``data_free``: no refusal carries text the program chose (module docstring)."""
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
    pairs: list[tuple[Any, Any]] | None = None
    try:
        exec(code, namespace)
        edit = namespace.get("edit")
        if not callable(edit):
            raise ProgramError(NO_EDIT)
        returned = edit(base, support)
        if data_free:
            pairs = _returned_pairs(returned)
    except ProgramError as exc:
        if not data_free:
            raise
        raise _data_free_failure(exc) from exc
    except (Exception, SystemExit) as exc:  # a program's own failure, reported, never trusted
        if not data_free:
            raise ProgramError(
                f"the program raised {type(exc).__name__}: {str(exc)[:500]}"
            ) from exc
        raise _data_free_failure(exc) from exc
    except BaseException as exc:  # a program's own BaseException subclass (Exception.__base__)
        if not data_free or issubclass(type(exc), KeyboardInterrupt):
            raise
        raise _data_free_failure(exc) from exc
    if data_free:
        if pairs is None:
            raise ProgramError("edit() must return a mapping of tensor name to array")
        return _validated(pairs, bundle, base, data_free=True)
    if not isinstance(returned, Mapping):
        raise ProgramError("edit() must return a mapping of tensor name to array")
    return _validated(returned.items(), bundle, base, data_free=False)  # as before the fix


def _validated(
    pairs: Iterable[tuple[Any, Any]],
    bundle: Bundle,
    base: Mapping[str, np.ndarray],
    *,
    data_free: bool,
) -> dict[str, WorkerTensor]:
    """The edited tensors from ``edit``'s returned pairs, each checked against ``base``."""
    out = dict(bundle.tensors)
    for name, value in pairs:
        if name not in base:
            raise ProgramError(
                RETURNED_NOT_EDITABLE
                if data_free
                else f"edit() returned {name!r}, which it may not change"
            )
        if not isinstance(value, np.ndarray):
            raise ProgramError(
                f"edit() returned something other than a numpy array for {name!r}"
                if data_free
                else f"edit() returned a {type(value).__name__} for {name!r}"
            )
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
    data_free: bool = True,
) -> Outcome:
    """Run the program twice; refuse ``wsp_program_nondeterministic`` when they disagree."""
    first = outcome(
        run_once(source, bundle, support, request, data_free=data_free),
        bundle,
        request,
        run_id,
        now,
        created_by=created_by,
    )
    second = outcome(
        run_once(source, bundle, support, request, data_free=data_free),
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
        write_failure(out, exc, "program_rerun")
        return 0
    if result.manifest is not None and result.data is not None:
        patch_writer.write_patch(out / "patch", result.manifest, result.data)
    (out / "result.json").write_text(
        json.dumps({"mode": "program_rerun", "stop_reason": "submitted", "failure": None}),
        encoding="utf-8",
    )
    return 0
