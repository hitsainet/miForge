"""``GET /api/v1/attachment`` - the serving process's attachment state **now**.

The shell's attachment chip reads this on every screen. It used to show the open run's
record - miLLM's state *as that run last observed it* - and a run that failed while attached
left its record reading "attached, weights clean" indefinitely, so the chip reported an
attachment hours after miLLM had dropped it.

This route asks the acquisition registry, which asks §5.10's acquirer when that package is
installed (ADR-022: the registry is the one importer). ``configured: false`` is the
standalone default and the chip renders nothing; ``reachable: false`` names why the read
failed; otherwise the body is miLLM's status as read for this request, with ``observed_at``.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from training import acquisition

router = APIRouter(tags=["acquisition"])


@router.get("/attachment", summary="miLLM's current attachment state, read live")
def attachment_status() -> dict[str, Any]:
    """The registry's answer, unchanged."""
    return dict(acquisition.attachment_status())
