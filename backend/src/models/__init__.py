"""ORM rows over the database index (ADR-003, ADR-004, ADR-017).

``models/`` holds ORM rows; ``schemas/`` holds the Pydantic models of the same artifacts.
``models/checkpoint.py`` and ``schemas/checkpoint.py`` are **different files for different
jobs** and that is the layering ADR-017 prescribes, not drift.

Imported for their side effect on :class:`db.base.Base`'s metadata as well as for the
classes: Alembic's ``env.py`` and every ``create_all`` need every table registered, and a
model nobody imports is a table no migration knows about.
"""

from __future__ import annotations

from models.adapter import Adapter
from models.checkpoint import SIGNAL_SOURCES, Checkpoint
from models.domain_state import (
    ADAPTER_STATES,
    CHECKPOINT_STATES,
    ArtifactKind,
    DomainClassification,
    DomainState,
    classify_domain,
    render_domain,
)
from models.lease import Lease
from models.pack import PART_KINDS, PART_SOURCES, Pack, PackPart
from models.run import RUN_STATUSES, RUN_TYPES, TERMINAL_STATUSES, Run, Step

__all__ = [
    "ADAPTER_STATES",
    "CHECKPOINT_STATES",
    "PART_KINDS",
    "PART_SOURCES",
    "RUN_STATUSES",
    "RUN_TYPES",
    "SIGNAL_SOURCES",
    "TERMINAL_STATUSES",
    "Adapter",
    "ArtifactKind",
    "Checkpoint",
    "DomainClassification",
    "DomainState",
    "Lease",
    "Pack",
    "PackPart",
    "Run",
    "Step",
    "classify_domain",
    "render_domain",
]
