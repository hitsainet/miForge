"""Handles -> tensors in this process, or a refusal  (R-02.1; FR-011.2, FR-011.3, FR-011.7).

**A partial rebuild that proceeds is the worst outcome available in this feature.** It
trains a model that differs from the served one in exactly the tensors it failed to rebuild,
reports success, and produces a loss curve that looks fine. So anything that cannot be
positively rebuilt is a **refusal naming what could not be rebuilt**, with the partial
tensors released.

## No local model of the attach response

BRD-01 **R-01.1** is normative for what the response carries, and this module **reads
fields from the response** rather than from a copy of its field list written here. A local
copy is the restatement that produced gaps 2 through 4 in this project: two prose
descriptions of one wire format drift silently. What is checked is **presence**, and an
absent value is named rather than defaulted.

## PyTorch's own rebuild entry point

The handle format is a **private contract between matching builds** - which is why
:func:`check_build_identity` runs *before* a single tensor is rebuilt, and why this module
hands the handle payload to ``torch.multiprocessing.reductions.rebuild_cuda_tensor``
instead of hand-rolling it. **The exact serialisation of that payload is the one thing in
this module that cannot be verified against anything today**: BRD-01 R-01.1 says "a CUDA
IPC handle" without fixing an encoding, and miLLM has no implementation to read. It is
therefore read as *the argument set PyTorch's own reducer emits*, and a payload that does
not fit that is refused by name rather than coerced. Marked as a first-integration seam and
reported.

## Tied embeddings rebuild into ONE allocation

Parameters that share a storage - tied input and output embeddings are the common case -
must rebuild into **one** tensor, not one per parameter. Two independent copies of one
tensor means the trained model silently differs from the served one in exactly that tensor,
and every other test in this feature would pass. Grouping is by the storage the response
names, never by parameter name.

## No requirement obliges the build check, and one should

FR-011.7 performs it; **no requirement in either BRD does** (FPRD DEF-F). BRD-01 R-01.1's
response carries no build identity, BRD-02 R-02.1 does not ask for one, miLLM's Feature 031
FR-31.1.4 adds the producer's torch, CUDA and driver versions *"so the consumer can refuse
an incompatible build before rebuilding anything"*, and PPRD §9 lists *"compatible PyTorch
builds across a CUDA IPC boundary"* as an **assumption**. So the assumption is named, the
mechanism exists in one feature document, and the obligation is nowhere. miForge checks it
anyway: the failure mode without the check is a silent misinterpretation of another
process's memory.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.errors import refuse

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class BuildIdentity:
    """A process's torch, CUDA and driver versions. Compared, never guessed at."""

    torch: str | None
    cuda: str | None
    driver: str | None

    def triple(self) -> str:
        return f"torch={self.torch or '?'} cuda={self.cuda or '?'} driver={self.driver or '?'}"


@dataclass
class RebuiltBase:
    """What a rebuild produced, and what it is allowed to be used for.

    ``read_only`` is not advice. FR-011.60's adapter path receives the base through this
    object and **writes nothing back**; the flag is what lets a test assert the difference
    between the two uses of the same rebuild.
    """

    tensors: dict[str, Any]
    device_uuid: str | None
    read_only: bool
    #: Storages that backed more than one parameter, as ``{storage_key: [names...]}``.
    #: Recorded because *"tied embeddings rebuilt into one allocation"* is otherwise
    #: unobservable from outside, and a test that cannot see it cannot pin it.
    shared_storages: dict[str, list[str]] = field(default_factory=dict)


def local_build_identity() -> BuildIdentity:
    """This process's triple, or ``None`` for each part it cannot determine.

    ``None`` rather than a placeholder string: *"unknown" compares unequal to everything*
    and would make every mismatch look like a mismatch of versions rather than of
    knowledge. torch is imported **inside the function** (ADR-023): a module-level import
    would make every refusal path in this feature unimportable on a machine without the ML
    wheels, and the refusals are where most of this feature's defect risk lives.
    """
    try:
        import torch
    except ImportError:
        return BuildIdentity(torch=None, cuda=None, driver=None)
    cuda_version = getattr(getattr(torch, "version", None), "cuda", None)
    # THE NVIDIA DRIVER, read from the kernel module - the same source miLLM's `producer`
    # block quotes (its hardware record reads /proc/driver/nvidia/version in the pod).
    # *Until 2026-10-16 this read `torch.cuda.driver_version`, which does not exist in
    # torch: the AttributeError was swallowed and `driver` was None on every machine.*
    # MyPy strict found it (`Module has no attribute "driver_version"`), which no test could:
    # the value is reported, never compared, so a permanent None was indistinguishable from
    # a cardless machine.
    driver = nvidia_driver_version()
    return BuildIdentity(torch=str(torch.__version__), cuda=cuda_version, driver=driver)


#: The kernel module's version line, e.g.
#: ``NVRM version: NVIDIA UNIX x86_64 Kernel Module  580.178.04  Tue Jul  7 ...``.
NVIDIA_VERSION_FILE = "/proc/driver/nvidia/version"


def nvidia_driver_version(path: str = NVIDIA_VERSION_FILE) -> str | None:
    """The loaded NVIDIA kernel module's version, or ``None`` when there is none to read.

    ``None`` on a machine without the driver (CI, a laptop): unknown, never a placeholder.
    """
    import re
    from pathlib import Path

    try:
        text = Path(path).read_text("utf-8", errors="replace")
    except OSError:
        return None
    match = re.search(r"Kernel Module\s+(\d+(?:\.\d+)+)", text)
    return match.group(1) if match else None


def check_build_identity(response: Any, *, local: BuildIdentity | None = None) -> None:
    """Refuse an incompatible build **before rebuilding a single tensor** (FR-011.7).

    Reads whatever build identity the attach response states. **An absent one is reported,
    not assumed compatible**: this is the check nothing obliges, so silently passing it
    when the other side says nothing would make the check decorative.

    Raises:
        Refusal: ``incompatible_torch_build``, naming **both** triples - a mismatch the
            operator must act on names both sides, or they cannot tell which to change.
    """
    mine = local if local is not None else local_build_identity()
    # `producer`, NOT `build`. miLLM's committed FTDD 031 §5.3 names this block
    # `producer: {torch, cuda, driver}`; F11 read `build`, which is in no version of the
    # contract. The names were restated from the BRD's prose rather than read from the
    # FTDD's JSON, in the one feature whose whole deliverable is a fake of someone
    # else's contract - and the failure was SILENT: an absent key took the
    # `build_identity_absent` branch below, so FR-011.7's "refuse an incompatible build
    # before rebuilding a single tensor" was a no-op against every real response.
    # Verified against miLLM `HEAD` and corroborated independently by that repo's own
    # session. The absent-branch decision below is F11's and is deliberately unchanged.
    stated = response.get("producer") if isinstance(response, dict) else None
    if not isinstance(stated, dict):
        log.warning(
            "the attach response states no build identity; proceeding without the check",
            extra={
                "code": "build_identity_absent",
                "local_build": mine.triple(),
                "note": (
                    "no requirement in BRD-01 or BRD-02 obliges this field (FPRD DEF-F); "
                    "miLLM Feature 031 FR-31.1.4 adds it. Reported rather than assumed."
                ),
            },
        )
        return
    theirs = BuildIdentity(
        torch=_optional_str(stated.get("torch")),
        cuda=_optional_str(stated.get("cuda")),
        driver=_optional_str(stated.get("driver")),
    )
    mismatched = [
        part
        for part in ("torch", "cuda")
        if getattr(theirs, part) is not None
        and getattr(mine, part) is not None
        and getattr(theirs, part) != getattr(mine, part)
    ]
    if mismatched:
        raise refuse(
            "incompatible_torch_build",
            f"The serving process and miForge do not share a compatible build "
            f"({', '.join(mismatched)} differ), and a CUDA IPC handle is a private contract "
            f"between matching builds. Refusing BEFORE rebuilding any tensor: a handle "
            f"interpreted by the wrong build is another process's memory read as something "
            f"it is not. serving: {theirs.triple()} / miForge: {mine.triple()}",
            mismatched=mismatched,
            serving_build=theirs.triple(),
            miforge_build=mine.triple(),
        )


def build(response: Any, *, read_only: bool = False, requires_grad: bool = True) -> RebuiltBase:
    """Rebuild every parameter the attach response names, or refuse naming what failed.

    Args:
        response: miLLM's attach response, **read as data**. R-01.1 is normative for it.
        read_only: FR-011.60's adapter path. ``requires_grad`` is then never set on a
            rebuilt tensor, so the base **cannot** receive a gradient even if one were
            computed - which is the one thing R-02.43 guarantees it cannot.
        requires_grad: set on miForge's side **only** (FR-011.4). miLLM is never asked to
            compute a gradient, hold optimizer state, or know what changed.

    Raises:
        Refusal: ``attach_response_incomplete`` when the response names no parameters;
            ``weights_rebuild_failed`` naming every parameter that could not be rebuilt,
            with the partial tensors released.
    """
    if read_only and requires_grad:
        # Refused rather than resolved by precedence: these two arguments disagree about
        # whether the base may take a gradient, and picking one silently is how MUT-19's
        # requires_grad half would ship.
        raise refuse(
            "weights_rebuild_arguments_conflict",
            "A read-only base cannot have requires_grad set. An adapter run reads the "
            "resident tensors and writes nothing back (R-02.43), so asking for both is a "
            "programming error rather than a configuration choice.",
        )
    parameters = response.get("parameters") if isinstance(response, dict) else None
    if not isinstance(parameters, list) or not parameters:
        raise refuse(
            "attach_response_incomplete",
            "miLLM's attach response names no parameter tensors, so there is nothing to "
            "rebuild (BRD-01 R-01.1 returns one entry per parameter). Refusing rather than "
            "proceeding with an empty graph - an empty graph trains successfully and "
            "changes nothing.",
            op="attach",
        )

    import torch
    from torch.multiprocessing.reductions import rebuild_cuda_tensor

    tensors: dict[str, Any] = {}
    by_storage: dict[str, list[str]] = {}
    failures: dict[str, str] = {}
    device_uuid = _optional_str(response.get("device_uuid"))

    for entry in parameters:
        name = str(entry.get("name") or entry.get("module_path") or "")
        if not name:
            failures["<unnamed>"] = "the entry names no module path"
            continue
        storage_key = _optional_str(entry.get("storage_key") or entry.get("storage"))
        handle = entry.get("handle")
        if handle is None:
            failures[name] = "the entry carries no CUDA IPC handle"
            continue
        # ONE ALLOCATION PER STORAGE. A second parameter on the same storage is a *view* of
        # the first, never a second rebuild: two rebuilds of one storage give two
        # independent copies, and the trained model then differs from the served one in
        # exactly that tensor while every other assertion in this feature passes.
        if storage_key is not None and storage_key in by_storage:
            source = tensors[by_storage[storage_key][0]]
            tensors[name] = source
            by_storage[storage_key].append(name)
            continue
        try:
            tensor = _rebuild_one(rebuild_cuda_tensor, handle)
        except Exception as exc:
            failures[name] = f"{type(exc).__name__}: {exc}"
            continue
        if requires_grad:
            # miForge's side ONLY (FR-011.4).
            tensor.requires_grad_(True)
        tensors[name] = tensor
        if storage_key is not None:
            by_storage[storage_key] = [name]

    if failures:
        # Release what was built. A partially rebuilt graph left in memory is 2.4 GB of
        # something nothing will use, on the one card the serving process is also on.
        tensors.clear()
        if hasattr(torch, "cuda"):
            with _Suppressed():
                torch.cuda.empty_cache()
        raise refuse(
            "weights_rebuild_failed",
            f"{len(failures)} of {len(parameters)} parameter tensors could not be rebuilt "
            f"from miLLM's handles, so no graph is returned and the partial tensors are "
            f"released. Refusing rather than training what did rebuild: a partial graph "
            f"trains a model that differs from the served one in exactly the tensors that "
            f"failed, and reports success.",
            failed=failures,
            rebuilt=len(tensors),
            expected=len(parameters),
        )

    shared = {key: names for key, names in by_storage.items() if len(names) > 1}
    log.info(
        "resident parameters rebuilt over the attachment",
        extra={
            "code": "weights_rebuilt",
            "parameters": len(tensors),
            "shared_storages": len(shared),
            "read_only": read_only,
            "device_uuid": device_uuid,
        },
    )
    return RebuiltBase(
        tensors=tensors, device_uuid=device_uuid, read_only=read_only, shared_storages=shared
    )


def _rebuild_one(rebuilder: Any, handle: Any) -> Any:
    """Hand the handle payload to PyTorch's own rebuilder. **Never hand-rolled.**

    A mapping is passed as keyword arguments and a sequence positionally, because the
    reducer's output is a tuple and a transport may have carried it as either. Anything
    else is a ``TypeError`` from here, which :func:`build` reports as a named failure for
    that parameter rather than as a crash.
    """
    if isinstance(handle, dict):
        return rebuilder(**handle)
    if isinstance(handle, (list, tuple)):
        return rebuilder(*handle)
    raise TypeError(
        f"a CUDA IPC handle must arrive as the argument set PyTorch's reducer emits "
        f"(a mapping or a sequence); got {type(handle).__name__}"
    )


class _Suppressed:
    """A tiny ``contextlib.suppress(Exception)`` - kept local so the refusal path has no
    import that could itself fail on a machine with no card."""

    def __enter__(self) -> None:
        return None

    def __exit__(self, *exc: object) -> bool:
        return True


def _optional_str(value: Any) -> str | None:
    return None if value is None else str(value)
