"""``GET /api/v1/gpus`` - the cards a run may ask for, **by UUID**  (task 14.0, AC-009.42).

The New run form's **GPU request** was a text box reading *"auto or a GPU-… UUID"*: the
operator had to run ``nvidia-smi -L`` on the node and copy a 40-character identifier. This
route lists the devices the driver reports, so the form can offer them.

## No index, anywhere in the body

``core.gpu.Device`` carries ``index`` because masking needs it, and its docstring says why it
is **never persisted**: it is valid only for the process that read it, at the moment it read
it. On this suite's node the indices **inverted** on 2026-09-18 and a bus address copied from
documentation once unbound the root disk. A response is a place an index would be copied
from, so it carries the UUID, the name and the free memory, and nothing positional - and
:func:`device_row` is the only producer of a row, so a test can assert the absence.

## An unreadable driver is an answer, not an error

``core.gpu`` refuses by name when ``nvidia-smi`` is absent, times out or fails. Here that is a
**200** with ``available: false`` and the refusal's code and message, because the question
"which cards can I pick?" has a true answer - *none that this process can see* - and the form
still offers ``auto``, which the backend resolves at submit. The API process may not be the
process that trains (a worker holds the card), which is why the form keeps a typed escape
hatch beside this list and says so.

Read-only. Runs ``nvidia-smi -L`` and one ``--query-gpu`` per request, live, exactly as
``core.gpu.resolve_device`` does at submit.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from core import gpu
from core.errors import Refusal

router = APIRouter(tags=["acquisition"])


def device_row(device: gpu.Device, free: dict[str, int]) -> dict[str, Any]:
    """One device as the API serves it. **No ``index``** - see the module docstring."""
    return {
        "uuid": device.uuid,
        "name": device.name,
        "free_memory_bytes": free.get(device.uuid),
    }


@router.get("/gpus", summary="The visible GPUs, by UUID, with their free memory")
def list_gpus() -> dict[str, Any]:
    """Every device the driver reports, or the named reason none can be read."""
    try:
        devices = gpu.list_devices()
    except Refusal as refusal:
        return {
            "devices": [],
            "available": False,
            "unavailable": {"code": refusal.code, "message": refusal.message},
            # `auto` is always a valid request: the backend resolves it at submit.
            "auto": True,
        }
    try:
        free = gpu.free_memory_by_uuid()
    except Refusal:
        # The device list was read; only the free figure is missing, and it renders as
        # unknown rather than as zero.
        free = {}
    return {
        "devices": [device_row(device, free) for device in devices],
        "available": True,
        "unavailable": None,
        "auto": True,
    }
