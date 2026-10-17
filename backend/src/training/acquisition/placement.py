"""Put the acquired parameters on the leased card, and record which one  (ADR-008, ADR-023).

**The defect this module exists for.** Nothing in ``backend/src/`` ever moved a model onto a
GPU. ``local_dir._load_pretrained`` called ``from_pretrained`` with no ``device_map``,
``scratch._instantiate`` called ``from_config``, and both leave every tensor in host memory;
``training/tasks.py`` never placed it either. Meanwhile the submit path resolves a card by
UUID, **takes a lease on it**, and ``objectives/sft.py`` builds its batch tensors on *the
model's* device - so a training run held a card nobody else could use and computed on the
CPU, at no point raising anything. The only visible trace was that the three GPU integration
tests had to write ``acquired.module.to("cuda")`` themselves: **a test hand-doing what
production omits is the signal, not the fix.**

Two further consequences were silent in the same way, because
:attr:`~training.acquisition.base.AcquiredModel.device_uuid` was never assigned either:
``providers/model_handle.device_uuid()`` returned ``None``, so **every in-process score
recorded no device** (R-02.16 wants the provider's provenance), and
``providers/in_process._check_budget`` skips its KV-cache check whenever the handle reports
no card - so **the scoring budget guard never ran**.

## Why the index is derived here, and why that is not "re-resolving a UUID"

``core/gpu.py`` is normative: *"a run records the UUID it resolved at submit, and that UUID
is what pins the work to a card. The index is a presentation detail of one process's view of
the world and is re-derived, under ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, only at the moment of
masking."* This module is that moment. It never *chooses* a card - no ``nvidia-smi``, no
"most free memory", no fallback to another device - it only **locates the already-chosen
UUID in this process's enumeration** and then reads back the UUID of the device the tensors
actually landed on.

That read-back matters more than it sounds. ``core/gpu.visibility_env`` **has no production
caller anywhere in this repository** (verified by grep; its only other hits are the
hard-coded-GPU ratchet's own comments), so a worker is *not* masked today and
``cuda:0`` is whatever the driver enumerated first. Assuming index 0 is the leased card is
exactly the inverted-enumeration incident ADR-008 was written after. So:

* **one visible device** - index 0 is unambiguous, masked or not;
* **a UUID named on the spec** - matched against the driver's own report for this process,
  and ``leased_gpu_not_visible`` if it is not there. Never another card: *a run pinned to a
  card it did not get is unreproducible.*
* **``auto`` with several devices visible** - refused by name
  (:data:`GPU_NOT_PINNED`). Which card holds the lease is genuinely unknown in this layer,
  and placing on index 0 would put two runs on one card while both leases claim exclusivity.
  The message names the owed wiring (``run_driver`` holds ``mf_runs.gpu_uuid``).

## The no-GPU case is CPU, named, and never recorded as a card

ADR-023 requires a developer without a GPU to run everything except a training step, and
this codebase already treats a CPU-resident parameter set as legitimate -
``model_handle.free_vram_bytes()`` returns ``0`` for one *by design*. So acquisition on a
machine with no card **succeeds on the CPU**; it does not refuse.

What it does not do is stay quiet about it, because a silent CPU run is the very defect
above reappearing one layer up. The CPU outcome is therefore:

* recorded as ``device_kind="cpu"`` with ``device_uuid=None`` on the acquisition record, so
  no manifest, log line or score can claim a card that was not used;
* logged at **warning** with ``code="model_placed_on_cpu"``, naming the consequence;
* and **refused outright when the spec named a specific UUID** - that is a positive request
  for a card, and answering it with the CPU would be the substitution ADR-013 and R-02.48
  both forbid.

The one case this layer cannot close is ``gpu_request="auto"`` on a box with no card *while
a lease is held*: the lease's UUID lives on ``mf_runs.gpu_uuid`` and reaches neither
``AcquisitionSpec`` nor ``training/tasks.py``. That gap is named in the record and in the
warning rather than guessed at, and the fix belongs where the lease is known.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Protocol

from core.errors import refuse
from core.gpu import UUID_PATTERN

log = logging.getLogger(__name__)

#: ``gpu_request`` meaning "whichever card submit picked". The literal is shared with
#: ``core.gpu.resolve_device``; spelled once here so the two cannot drift.
AUTO = "auto"

#: Why a device was chosen. A closed set, so a test asserts the *decision* rather than
#: re-deriving it, and an unknown reason is a bug rather than a new policy.
SOLE_VISIBLE_DEVICE = "sole_visible_device"
LEASED_UUID = "leased_uuid"
CUDA_UNAVAILABLE = "cuda_unavailable"

#: The refusal raised when ``auto`` cannot identify the leased card. Named as a constant
#: because the message points at owed wiring and a test asserts the code, not the prose.
GPU_NOT_PINNED = "gpu_not_pinned"

#: ``cpu`` is a real, recorded answer - not the absence of one.
KIND_CUDA = "cuda"
KIND_CPU = "cpu"


class CudaProbe(Protocol):
    """This process's view of the CUDA devices, as the driver reports it.

    A protocol rather than direct ``torch`` calls so that **every branch below is reachable
    on a machine with no card**. The decision this module makes is the thing most worth
    testing and it would otherwise be testable only on the GPU node - which is how a
    placement defect survives a green suite.
    """

    def is_available(self) -> bool: ...

    def device_count(self) -> int: ...

    def uuid_of(self, index: int) -> str | None:
        """The device's UUID in ``core.gpu``'s spelling, or ``None`` if unreportable."""
        ...

    def device(self, index: int) -> Any:
        """The ``torch.device`` for ``index``. Built from a variable, never a literal."""
        ...


@dataclass(frozen=True)
class Decision:
    """Which device the parameters should go on, and under which rule.

    ``index`` is deliberately **not** carried onto the record: it is valid only for the
    process that read it, at the moment it read it (``core.gpu.Device``'s own words).
    """

    kind: str
    uuid: str | None
    index: int | None
    reason: str


@dataclass(frozen=True)
class Placement:
    """Where the parameters **actually** ended up, read back after the move.

    Returned rather than merely logged, because the acquisition record has to carry it: a
    device that is only logged cannot be compared with the device a score claims.
    """

    device_kind: str
    device_uuid: str | None
    reason: str


def normalise_device_uuid(value: str) -> str:
    """``GPU-<lowercase hex>`` - the spelling ``nvidia-smi -L`` prints.

    ``torch`` reports a device UUID as a bare :class:`uuid.UUID` while ``nvidia-smi``
    prefixes it, and the two are compared here. Normalising in **one** function is the point:
    this project has paid four times for one value with two spellings.
    """
    text = str(value).strip().lower()
    if text.startswith("gpu-"):
        text = text[len("gpu-") :]
    return f"GPU-{text}"


def _torch_probe() -> CudaProbe:
    """The real probe, importing ``torch`` lazily and refusing **by name** without it.

    Lazy for the reason ``training/acquisition/base.py`` gives: a module-level
    ``import torch`` would make this feature's whole refusal surface unimportable in CI,
    where the ML wheels may be absent, and the refusals are where most of the defect risk
    lives.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - torch is a declared dependency
        raise refuse(
            "ml_runtime_unavailable",
            "torch is not installed in this process, so the acquired parameters cannot be "
            "placed on a device. Everything else in miForge runs without it (ADR-023).",
            detail=str(exc),
        ) from exc
    return _TorchCudaProbe(torch)


class _TorchCudaProbe:
    """:class:`CudaProbe` over the live ``torch`` runtime."""

    def __init__(self, torch_module: Any) -> None:
        self._torch = torch_module

    def is_available(self) -> bool:
        try:
            return bool(self._torch.cuda.is_available())
        except Exception:  # pragma: no cover - a broken driver reports unavailable
            # A driver in the "Driver/library version mismatch" state this node is known to
            # reach raises here. Reported as *no card* rather than as a crash: the CPU
            # outcome below is named and recorded, so nothing claims a device it did not get.
            return False

    def device_count(self) -> int:
        try:
            return int(self._torch.cuda.device_count())
        except Exception:  # pragma: no cover - same reasoning as above
            return 0

    def uuid_of(self, index: int) -> str | None:
        try:
            raw = getattr(self._torch.cuda.get_device_properties(index), "uuid", None)
        except Exception:  # pragma: no cover - needs a card
            return None
        return None if raw is None else normalise_device_uuid(str(raw))

    def device(self, index: int) -> Any:
        # `torch.device(type, index)` rather than an f-string: there is no device literal
        # anywhere in this module, which is what keeps `test_no_hardcoded_gpu.py`'s ratchet
        # meaningful rather than allowlisted.
        return self._torch.device(KIND_CUDA, index)


def decide(gpu_request: str, *, probe: CudaProbe) -> Decision:
    """Which device this acquisition belongs on. **Chooses no card and allocates nothing.**

    Args:
        gpu_request: ``auto`` or a ``GPU-...`` UUID, straight off the spec. Classified
            against ``core.gpu.UUID_PATTERN`` - the one pattern - rather than with a second
            taxonomy of indices and bus addresses, which ``core.gpu.resolve_device`` owns.
        probe: this process's CUDA view.

    Raises:
        Refusal: ``gpu_request_not_a_uuid`` for anything that is neither ``auto`` nor a
            UUID; ``leased_gpu_not_visible`` when a named card is not here;
            ``gpu_not_pinned`` when ``auto`` cannot identify the leased card among several;
            ``gpu_uuid_unreadable`` when the driver will not name the card it is about to be
            given. Every one of them before a single byte is moved.
    """
    if not gpu_request:
        raise refuse(
            "gpu_request_empty",
            "The acquisition spec names no device. Name 'auto' or a GPU-... UUID; refusing "
            "rather than picking a card, which is how work lands on the wrong one.",
        )

    wanted: str | None
    if gpu_request == AUTO:
        wanted = None
    elif UUID_PATTERN.match(gpu_request):
        wanted = normalise_device_uuid(gpu_request)
    else:
        raise refuse(
            "gpu_request_not_a_uuid",
            f"{gpu_request!r} is not 'auto' and not a GPU-... UUID, so no device can be "
            f"pinned from it. core.gpu.resolve_device refuses an index and a bus address by "
            f"name at submit; a spec reaching acquisition with one skipped that path.",
            request=gpu_request,
        )

    count = probe.device_count() if probe.is_available() else 0
    if count == 0:
        if wanted is not None:
            raise refuse(
                "leased_gpu_not_visible",
                f"This run is pinned to {wanted} and no CUDA device is visible to the "
                f"training process, so its parameters cannot be placed on it. Refusing "
                f"rather than training on the CPU: the run holds that card's lease, the "
                f"step would be orders of magnitude slower, and nothing would have said so.",
                request=gpu_request,
                visible=[],
            )
        return Decision(kind=KIND_CPU, uuid=None, index=None, reason=CUDA_UNAVAILABLE)

    if wanted is None:
        if count > 1:
            # FAIL CLOSED (ADR-013). 'auto' was resolved to a specific UUID at submit and a
            # lease was taken on it, but neither the UUID nor the lease reaches this layer -
            # `run_driver` builds the spec from `definition.gpu_request`, which is still
            # 'auto', while the resolved value sits on `mf_runs.gpu_uuid`. Taking index 0
            # here would place the work on a card another run holds, with both leases
            # claiming exclusivity and both runs' numbers wrong.
            raise refuse(
                GPU_NOT_PINNED,
                f"{count} CUDA devices are visible and this run names no device UUID, so "
                f"which card holds its lease is unknown here. Placing on the first "
                f"enumerated device is the inverted-enumeration defect ADR-008 was written "
                f"after. Pass the leased UUID as `gpu_request` - it is recorded on "
                f"`mf_runs.gpu_uuid` at submit - or mask the worker with "
                f"`core.gpu.visibility_env`, which no production caller invokes today.",
                visible=[probe.uuid_of(i) for i in range(count)],
                devices_visible=count,
            )
        index = 0
        reason = SOLE_VISIBLE_DEVICE
    else:
        visible = [probe.uuid_of(i) for i in range(count)]
        matches = [i for i, uuid in enumerate(visible) if uuid == wanted]
        if not matches:
            raise refuse(
                "leased_gpu_not_visible",
                f"This run is pinned to {wanted} and this process does not see it "
                f"({visible}). Refusing rather than falling back to another card: a run "
                f"pinned to a card it did not get is unreproducible, and the card it would "
                f"fall back to is held by another run's lease.",
                request=gpu_request,
                visible=visible,
            )
        index = matches[0]
        reason = LEASED_UUID

    uuid = probe.uuid_of(index)
    if uuid is None:
        raise refuse(
            "gpu_uuid_unreadable",
            "The driver would not report the UUID of the device these parameters are about "
            "to be placed on, so the card could not be recorded (ADR-008) and a score taken "
            "from the model could not name the device that produced it. Refusing rather "
            "than recording a card as 'unknown' while using it.",
            index_derived=index,
        )
    return Decision(kind=KIND_CUDA, uuid=uuid, index=index, reason=reason)


def place(module: Any, gpu_request: str, *, probe: CudaProbe | None = None) -> Placement:
    """Move ``module``'s parameters onto the decided device and **read back where they are**.

    The read-back is the positive check that the measurement happened. This project's
    recurring false-measurement family - a ``||`` branch printing "clean" from a grep that
    errored, a ``replace()`` that matched nothing, a stale ``__pycache__`` outliving a
    restore - is why a ``.to(device)`` whose result is never inspected is not treated as
    evidence of anything.

    Args:
        module: the acquired module. On the CPU outcome it is not touched at all, which is
            also why a stub without ``.to`` is acceptable there: nothing moves.
        gpu_request: ``auto`` or a ``GPU-...`` UUID, from
            :attr:`~training.acquisition.base.AcquisitionSpec.gpu_request`.

    Raises:
        Refusal: everything :func:`decide` raises, plus ``model_placement_failed`` when the
            move did not land where it was asked to and ``model_placement_unverifiable``
            when the module will not say where its parameters are. Both fail closed: a card
            the tensors cannot be *proved* to be on is the silent failure this module exists
            to end.
    """
    decision = decide(gpu_request, probe=probe if probe is not None else _torch_probe())

    if decision.kind == KIND_CPU:
        log.warning(
            "the acquired parameters stay on the CPU; no CUDA device is visible",
            extra={
                "code": "model_placed_on_cpu",
                "gpu_request": gpu_request,
                "reason": decision.reason,
                "consequence": (
                    "A training step will run on the CPU, orders of magnitude slower. This "
                    "is ADR-023's supported state for a developer machine and is recorded "
                    "as device_kind='cpu' with no device UUID, so no manifest, log line or "
                    "score can claim a card. If this process holds a GPU lease the run is "
                    "wasting it: the leased UUID is on mf_runs.gpu_uuid and does not reach "
                    "the acquisition spec, which is the one case this layer cannot refuse."
                ),
            },
        )
        return Placement(device_kind=KIND_CPU, device_uuid=None, reason=decision.reason)

    assert decision.index is not None  # cuda decisions always carry an index
    target = (probe if probe is not None else _torch_probe()).device(decision.index)
    try:
        module.to(target)
    except Exception as exc:
        raise refuse(
            "model_placement_failed",
            f"The acquired parameters could not be moved onto {decision.uuid}: {exc}. "
            f"Refusing rather than stepping on whatever device they are still on - a step "
            f"taken on the CPU while holding a card's lease is the defect this check exists "
            f"for.",
            device_uuid=decision.uuid,
            error=type(exc).__name__,
        ) from exc

    landed = _device_of(module)
    if landed is None:
        raise refuse(
            "model_placement_unverifiable",
            f"The acquired module reports no parameter device, so it cannot be proved that "
            f"its tensors reached {decision.uuid}. Refusing rather than recording a card "
            f"the parameters may not be on.",
            device_uuid=decision.uuid,
        )
    if getattr(landed, "type", None) != KIND_CUDA:
        raise refuse(
            "model_placement_failed",
            f"The acquired parameters were moved to {decision.uuid} and report "
            f"{landed!r} afterwards. Refusing: this is exactly the silent case - the lease "
            f"is held, the step would compute elsewhere, and nothing else would notice.",
            device_uuid=decision.uuid,
            landed_on=str(landed),
        )

    log.info(
        "the acquired parameters are on the leased card",
        extra={
            "code": "model_placed",
            "gpu_request": gpu_request,
            "device_uuid": decision.uuid,
            "reason": decision.reason,
        },
    )
    return Placement(device_kind=KIND_CUDA, device_uuid=decision.uuid, reason=decision.reason)


def _device_of(module: Any) -> Any:
    """The device of ``module``'s first parameter, or ``None`` when it will not say.

    The first *parameter* rather than ``module.device``: a ``transformers`` model exposes
    ``.device`` as the device of its first parameter anyway, while a sharded or
    hook-wrapped module can report a ``.device`` that no tensor is on. Reading a real
    parameter is the measurement; reading the attribute is a claim about it.
    """
    try:
        for parameter in module.parameters():
            return parameter.device
    except Exception:
        return None
    return None


__all__ = [
    "AUTO",
    "CUDA_UNAVAILABLE",
    "GPU_NOT_PINNED",
    "KIND_CPU",
    "KIND_CUDA",
    "LEASED_UUID",
    "SOLE_VISIBLE_DEVICE",
    "CudaProbe",
    "Decision",
    "Placement",
    "decide",
    "normalise_device_uuid",
    "place",
]
