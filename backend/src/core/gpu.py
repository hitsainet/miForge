"""Device resolution: ``auto`` or a UUID, never an index and never a bus address.

Task 2.7; ADR-008. Both halves of that rule are scar tissue:

* **Indices inverted once.** Enumeration order changed between boots, so ``cuda:0``
  silently became the other card. Nothing failed - the work simply ran on the wrong
  device, and every number taken from it was wrong by the ratio of the two cards.
* **A copied bus address unbound the root NVMe.** An address was copied out of a hardware
  table and applied to the wrong device class. This is why miStudio's hardware table is
  **cited, never copied**, and why ``backend/tests/unit/test_no_hardcoded_gpu.py`` bans a
  bus-address literal outright.

So a run records the **UUID it resolved at submit**, and that UUID is what pins the work
to a card. The index is a presentation detail of one process's view of the world and is
re-derived, under ``CUDA_DEVICE_ORDER=PCI_BUS_ID``, only at the moment of masking.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass

from core.errors import refuse

#: `nvidia-smi -L` prints: `GPU 0: NVIDIA GeForce RTX 3090 (UUID: GPU-xxxxxxxx-...)`
_LINE = re.compile(r"^GPU (?P<index>\d+):\s*(?P<name>.+?)\s*\(UUID:\s*(?P<uuid>GPU-[0-9a-f-]+)\)")

#: A valid device UUID as nvidia-smi reports it.
UUID_PATTERN = re.compile(r"^GPU-[0-9a-f]{8}(-[0-9a-f]{4}){3}-[0-9a-f]{12}$")


@dataclass(frozen=True)
class Device:
    """One visible CUDA device, as the driver currently reports it.

    ``index`` is included because masking needs it, and is deliberately **not** persisted
    anywhere: it is valid only for the process that read it, at the moment it read it.
    """

    uuid: str
    name: str
    index: int
    free_memory_bytes: int | None = None


def _run(args: list[str]) -> str:
    binary = shutil.which(args[0])
    if binary is None:
        raise refuse(
            "nvidia_smi_unavailable",
            f"{args[0]} is not on PATH, so no device can be resolved. On a machine without "
            f"a card this is expected: run with no GPU request, and the training step will "
            f"refuse by name rather than guessing a device.",
            binary=args[0],
        )
    try:
        completed = subprocess.run(
            [binary, *args[1:]], capture_output=True, text=True, timeout=15, check=True
        )
    except subprocess.TimeoutExpired as exc:
        raise refuse(
            "nvidia_smi_timeout",
            "nvidia-smi did not answer within 15s. The driver may be in the "
            "'Driver/library version mismatch' state this node is known to reach after an "
            "unattended upgrade; check the host before starting a run.",
        ) from exc
    except subprocess.CalledProcessError as exc:
        raise refuse(
            "nvidia_smi_failed",
            "nvidia-smi exited non-zero, so the set of devices is unknown. Refusing rather "
            "than assuming a device - a guessed card is how work lands on the wrong one.",
            stderr=(exc.stderr or "").strip()[:500],
        ) from exc
    return completed.stdout


def list_devices() -> list[Device]:
    """Every visible device, parsed from **live** ``nvidia-smi -L``.

    Live rather than cached: the point of resolving at submit is that the answer may have
    changed since the process started.
    """
    devices: list[Device] = []
    for line in _run(["nvidia-smi", "-L"]).splitlines():
        match = _LINE.match(line.strip())
        if match:
            devices.append(
                Device(
                    uuid=match.group("uuid"),
                    name=match.group("name"),
                    index=int(match.group("index")),
                )
            )
    return devices


def _free_memory_by_uuid() -> dict[str, int]:
    """Free memory per device, keyed by UUID - never by index.

    Queried with ``--query-gpu=uuid,memory.free`` so the join key comes back *in the same
    row* as the number. Reading two separate lists and zipping them by position is how an
    inverted enumeration produces a confident, wrong answer.
    """
    out = _run(["nvidia-smi", "--query-gpu=uuid,memory.free", "--format=csv,noheader,nounits"])
    free: dict[str, int] = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and UUID_PATTERN.match(parts[0]):
            try:
                free[parts[0]] = int(parts[1]) * 1024 * 1024
            except ValueError:
                continue
    return free


def resolve_device(request: str) -> Device:
    """Resolve ``"auto"`` or an explicit ``GPU-...`` UUID to a concrete :class:`Device`.

    ``auto`` picks the device with the **most free memory**, which is the only defensible
    automatic choice on a node that shares its cards with a serving process and an
    interpretation process (neither of which requests ``nvidia.com/gpu``, so all three can
    be resident at once).

    An index is refused *by name* rather than tolerated, because tolerating it here is how
    it reaches a manifest.
    """
    if not request:
        raise refuse("gpu_request_empty", "Name a device: 'auto' or a GPU-... UUID.")

    if request.isdigit() or request.startswith("cuda:"):
        raise refuse(
            "gpu_request_is_an_index",
            f"{request!r} names a device by index. ADR-008 forbids it: enumeration order "
            f"changed between boots on this node once and the work silently ran on the "
            f"other card. Use 'auto' or a GPU-... UUID from `nvidia-smi -L`.",
            request=request,
        )
    if re.match(r"^(?:[0-9A-Fa-f]{4,8}:)?[0-9A-Fa-f]{2}:[0-9A-Fa-f]{2}\.\d$", request):
        raise refuse(
            "gpu_request_is_a_bus_address",
            f"{request!r} looks like a PCI bus address. A copied bus address once unbound "
            f"this machine's root NVMe; use a GPU-... UUID.",
            request=request,
        )

    devices = list_devices()
    if not devices:
        raise refuse(
            "no_cuda_device_visible",
            "nvidia-smi reported no devices. A training step cannot run; everything else "
            "in miForge can (ADR-023).",
        )

    if request == "auto":
        free = _free_memory_by_uuid()
        best = max(devices, key=lambda d: free.get(d.uuid, -1))
        return Device(best.uuid, best.name, best.index, free.get(best.uuid))

    if not UUID_PATTERN.match(request):
        raise refuse(
            "gpu_uuid_malformed",
            f"{request!r} is not a device UUID. Expected the GPU-xxxxxxxx-xxxx-xxxx-xxxx-"
            f"xxxxxxxxxxxx form printed by `nvidia-smi -L`.",
            request=request,
        )

    for device in devices:
        if device.uuid == request:
            return Device(
                device.uuid, device.name, device.index, _free_memory_by_uuid().get(device.uuid)
            )

    raise refuse(
        "gpu_uuid_not_present",
        f"No visible device has UUID {request}. Refusing rather than falling back to "
        f"another card: a run pinned to a card it did not get is unreproducible.",
        request=request,
        visible=[d.uuid for d in devices],
    )


def visibility_env(device: Device) -> dict[str, str]:
    """The environment that masks a child process to exactly ``device``.

    ``CUDA_VISIBLE_DEVICES`` is set to the **UUID**, which the driver accepts and which
    cannot be invalidated by a change in enumeration order - the whole point.
    ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` matches miLLM and miStudio (foundation task 8.7) so
    that any index the child does derive is stable.
    """
    return {"CUDA_VISIBLE_DEVICES": device.uuid, "CUDA_DEVICE_ORDER": "PCI_BUS_ID"}
