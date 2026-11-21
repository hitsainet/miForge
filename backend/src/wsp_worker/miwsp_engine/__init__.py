"""``miwsp_engine``: miStudio's own weight-space-programming engine (feature 018; FR-018.1,
FR-018.2; BRD-WSP-001 §13, D23-D28; FTDD §6.1).

It replaces the ``weightspace`` adapter's role inside the WSP child, behind the same worker
contract: the same pushed line, the same ``/out`` names, the same ``result.json`` keys the
backend reads, and the same patch writer. It imports no WSP library and no vendor LLM SDK, only
the standard library, numpy, ``httpx2`` and ``wsp_worker`` (the fences in
``tests/unit/wsp/test_wsp_fences.py``).

This module imports nothing heavy: the kernel grandchild boots through
``wsp_worker.miwsp_engine.kernel_main`` and must not pull the session, its client or ``httpx2``.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

PROVIDER_ID = "miwsp_engine"
#: The engine's version, recorded in every patch's ``created_by`` and the backend's registry.
VERSION = "0.1.0"
PROVIDER = (PROVIDER_ID, VERSION)


def run(line: Mapping[str, Any], *, client: Any = None, **options: Any) -> int:
    """One session (or its replay) from the child line; writes ``/out``; the child's exit code."""
    from wsp_worker.miwsp_engine import session

    return session.run(line, client=client, **options)
