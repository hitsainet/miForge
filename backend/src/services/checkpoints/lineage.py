"""The parent chain, walked through manifests on disk  (FR-005.39, E-11).

**Manifests, never the index.** ADR-004 gives the index no authority over a correctness
answer, and a lineage chain is one: it is what R-02.23's *"lineage remains traceable across
runs"* means in practice, and it has to be the same answer on a machine the checkpoints
were merely copied to.

## ``truncated`` is not ``root``

Three terminations, and two of them look alike from a distance:

* ``root`` - ``parent`` is ``null``. *This is the beginning.*
* ``truncated`` - ``parent`` names something that no longer resolves on disk. *The record
  stops here.* The branch still **proceeds** (lineage is a chain of names, not of live
  files), but the response says where the chain ran out.
* ``limit`` - the walk was bounded.

Conflating the first two makes a partial tree look complete, which is the quietly wrong
answer ADR-013 forbids. They are two different facts and the response carries which one.

**R-02.29's tree is F9's.** This supplies the checkpoint edge and nothing about adapters,
and it ships **no depth value**: depth is a property of a walk, so F9 computes it from the
chain. The mockup's own handoff notes that its indentation came from a ``depth`` field and
that the real tree needs the walk.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.errors import Refusal
from models.domain_state import ArtifactKind, classify_domain, render_domain
from services.checkpoints import manifest as manifest_module
from services.checkpoints.paths import checkpoint_dir, parse_qualified_name

__all__ = ["Lineage", "LineageNode", "walk"]

#: Bounded by default. A chain is one file read per hop and a cycle is representable - a
#: manifest copied in from elsewhere can name a parent that names it back - so the walk is
#: finite by construction rather than by trusting the data.
DEFAULT_LIMIT = 256


@dataclass(frozen=True)
class LineageNode:
    """One checkpoint in a chain, with what a tree row shows.

    ``domain`` may be ``null`` and the node carries ``domain_state`` beside it, so a
    renderer can say *"no domain"* rather than leaving a blank that reads like missing data
    (ADR-012). ``test`` is the whole ``ScoreRecord`` because **every number names its scale
    and sample**: a score without its ``n`` and its set is not a measurement.
    """

    name: str
    run_id: str
    step: int
    domain: str | None
    domain_state: str
    domain_display: str
    parent: str | None
    base_model: str
    test: dict[str, Any] | None
    #: The second ancestor edge: a merged checkpoint descends from its base via ``parent``
    #: **and** from the adapter baked into it via this (R-02.45). Two ancestors, two fields.
    merged_adapter: dict[str, Any] | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "run_id": self.run_id,
            "step": self.step,
            "domain": self.domain,
            "domain_state": self.domain_state,
            "domain_display": self.domain_display,
            "parent": self.parent,
            "base_model": self.base_model,
            "test": self.test,
            "merged_adapter": self.merged_adapter,
        }


@dataclass(frozen=True)
class Lineage:
    """A chain from a checkpoint towards its root, and **how it ended**."""

    nodes: list[LineageNode] = field(default_factory=list)
    #: ``root`` | ``truncated`` | ``limit``. Never collapsed into a boolean.
    terminated: str = "root"
    #: The name the chain stopped at, for ``truncated``. ``None`` otherwise.
    terminated_at: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "nodes": [n.as_dict() for n in self.nodes],
            "terminated": self.terminated,
            "terminated_at": self.terminated_at,
        }


def _node(doc: dict[str, Any]) -> LineageNode:
    classification = classify_domain(doc, kind=ArtifactKind.CHECKPOINT)
    return LineageNode(
        name=str(doc["name"]),
        run_id=str(doc["run_id"]),
        step=int(doc["step"]),
        domain=classification.value,
        domain_state=classification.state.value,
        domain_display=render_domain(classification.value, classification.state),
        parent=doc.get("parent"),
        base_model=str(doc["base_model"]),
        test=doc.get("test"),
        merged_adapter=doc.get("merged_adapter"),
    )


def walk(
    run_id: str,
    step: int,
    *,
    root: Path | None = None,
    limit: int = DEFAULT_LIMIT,
) -> Lineage:
    """Follow ``parent`` by **name** from ``(run_id, step)`` until the chain ends.

    Raises:
        Refusal: ``checkpoint_not_found`` or ``checkpoint_manifest_invalid`` for the
            **starting** checkpoint only. A broken link further up is a ``truncated``
            chain, not a failure: the thing asked about exists and the honest answer is the
            part of its history that survives.
    """
    doc = manifest_module.read(checkpoint_dir(run_id, step, root=root))
    nodes = [_node(doc)]
    seen = {nodes[0].name}

    while True:
        parent = nodes[-1].parent
        if parent is None:
            return Lineage(nodes=nodes, terminated="root")
        if len(nodes) >= limit:
            return Lineage(nodes=nodes, terminated="limit", terminated_at=parent)
        if parent in seen:
            # A cycle is reachable through copied-in manifests. Reported as the chain
            # stopping at a name rather than as the root, for the same reason as a missing
            # one: it is not the beginning, and saying so would be a false tree.
            return Lineage(nodes=nodes, terminated="truncated", terminated_at=parent)

        parsed = parse_qualified_name(parent)
        if parsed is None:
            return Lineage(nodes=nodes, terminated="truncated", terminated_at=parent)
        _, parent_run, parent_step = parsed
        try:
            parent_doc = manifest_module.read(checkpoint_dir(parent_run, parent_step, root=root))
        except Refusal:
            # The name no longer resolves on disk - the checkpoint was deleted, or this tree
            # was copied without its ancestors (E-11). The chain is TRUNCATED at that name.
            return Lineage(nodes=nodes, terminated="truncated", terminated_at=parent)

        node = _node(parent_doc)
        if node.name != parent:
            # The directory exists but holds a different checkpoint. Treated as the record
            # stopping rather than as a hop, because following it would attribute an
            # ancestry the manifest does not claim.
            return Lineage(nodes=nodes, terminated="truncated", terminated_at=parent)
        nodes.append(node)
        seen.add(node.name)
