"""The artifact forest's wire types  (R-02.29; FR-009.21 - FR-009.30; task 3.1).

R-02.29 as amended 2026-10-13: the tree holds **two artifact kinds** and the edge that
places each one differs. A checkpoint is placed by its manifest's ``parent``; an adapter by
the base **its own manifest names** - *"not by a ``parent`` field, which the adapter
manifest does not have."* A merged checkpoint has **two** incoming edges.

## Three things this model refuses to carry

1. **No ``depth``.** FR-009.26: indentation is walked from the edges. A transmitted depth is
   a second source of truth for the tree's shape, and the mockup's own handoff warns that
   its indentation came from a stored field. Mutation **M17** requires *adding* a field
   here, which is the point.
2. **No pack.** ``checkpoint-manifest-v1.json`` has no pack field, deliberately (R-02.35),
   and a join through ``run_id`` to find one would break both FR-009.28 and reindex
   equivalence - packs are the one artifact class the filesystem rebuild reaches through a
   different file. FPRD **F-10**.
3. **No collapse of ``domain``'s states.** ``domain`` is *required and nullable* in the
   checkpoint schema, so ``null`` is an assertion and an absent key is an omission; and an
   adapter manifest declares no ``domain`` property at all, which is a fourth state.
   :class:`models.domain_state.DomainState` already names all four and this model carries
   its spelling verbatim.

## The spelling of the fourth state

``not_in_kind``, which is what :mod:`models.domain_state` defines, what ``mf_adapters``'
``domain_state_is_not_in_kind`` CHECK enforces, and what ``_adapter_payload`` already puts
on the wire. **F9's task list 3.1 says ``not_recorded_for_kind``**; adopting that would make
a fifth spelling of a vocabulary this project has already paid for four times (the verdict
value, the three signal vocabularies, ``step-`` vs ``000500``, ``full`` vs ``change``,
``queued`` vs ``pending``). The *rendering* is "not recorded in this artifact kind" and
:func:`models.domain_state.render_domain` is the one place it is produced.

## Why edges are objects and not two parallel fields

The FTDD's ``edges_resolved`` block pairs a target with a state. Keeping them in one
:class:`ResolvedEdge` makes an edge impossible to transmit without saying how it resolved -
a ``parent`` string with no state beside it is exactly the shape in which ``root`` and
``truncated`` get conflated (F5 FR-005.39 / its E-11, mutation **M9**).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

#: How an edge resolved. **Seven values, and the distinctions are load-bearing:**
#:
#: * ``ok``          - the target is in this page.
#: * ``root``        - an *assertion*: ``parent`` is null, this **is** a root.
#: * ``truncated``   - the named parent does not resolve anywhere. The record stops here.
#: * ``broken``      - a self-edge, or the node at which a longer cycle re-enters itself.
#: * ``unparented``  - an adapter whose ``base.name`` resolves to nothing. Distinct from
#:   ``truncated`` because an adapter with no base is not a partial chain: it is an
#:   artifact that cannot be placed at all (FTDD section 4.3, edge 2).
#: * ``unresolved``  - a ``merged_adapter`` naming an adapter that is not there.
#: * ``not_loaded``  - the target exists in the index but is outside this page or outside
#:   this filter. **A paging artefact must not look like a broken lineage** (FTDD
#:   section 5.2), which is why it is its own value rather than ``truncated``.
EdgeState = Literal[
    "ok",
    "root",
    "truncated",
    "broken",
    "unparented",
    "unresolved",
    "not_loaded",
]

EDGE_STATES: tuple[EdgeState, ...] = (
    "ok",
    "root",
    "truncated",
    "broken",
    "unparented",
    "unresolved",
    "not_loaded",
)

#: Which manifest schema describes a node. Spelled as ``manifest['kind']`` resolves, and
#: taken from the manifest rather than parsed out of the name: FR-009.23 - *"a view that
#: relies on parsing the name to know what it is renders a hand-named artifact wrongly."*
NodeKind = Literal["checkpoint", "adapter"]

#: ``models.domain_state.DomainState``'s four, verbatim. See the module docstring.
DomainStateValue = Literal["recorded", "asserted_null", "absent", "not_in_kind"]

#: Whether a merged edge could be verified. **Three states, not a boolean** (E-12, E-12a).
#:
#: ``MergedAdapter.required`` is ``[name, step, manifest_sha256]`` *and*
#: ``manifest_sha256`` is ``anyOf [{"type": "string"}, {"type": "null"}]`` with
#: ``default: null`` - parsed from ``docs/schemas/checkpoint-manifest-v1.json``, not taken
#: from prose about it. So the field is **required and nullable**, exactly like ``domain``,
#: and that gives three answers:
#:
#: * ``verified``       - a hash is present.
#: * ``unverified_null`` - the key is present and ``null``. **Conforming.** The edge is
#:   legitimately unverified and must never be reported as a defect.
#: * ``key_absent``     - the key is missing. A missing *required* key: nonconforming.
#:
#: A boolean collapses the middle into the third, which is the conflation FR-009.17 rule 3
#: forbids for ``domain`` and which several documents made about this very field.
MergedVerification = Literal["verified", "unverified_null", "key_absent"]

#: Which of the three edges a flat edge record describes.
EdgeName = Literal["parent", "base", "merged_adapter"]


class ResolvedEdge(BaseModel):
    """One edge: who it points at, and how the resolution went."""

    model_config = ConfigDict(extra="forbid")

    #: The target's fully qualified name, or ``None`` when there is nothing to name - a
    #: root's absent parent, or an adapter whose base block carries no name. **A name is
    #: carried even when the edge failed**, so the row can say *which* ancestor it could
    #: not find (FR-009.27, FR-009.30).
    target: str | None
    state: EdgeState


class TreeEdge(BaseModel):
    """One edge of the forest, as a flat record.

    The node-level :class:`ResolvedEdge` fields are the authority; this list is a
    **projection** of them for a caller that wants the graph rather than the rows, and it
    is built from the nodes in one pass so the two cannot disagree. ``child``/``parent``/
    ``edge`` keep ``services.reindex``' own pre-existing vocabulary so that the tests
    written against it keep meaning what they meant.
    """

    model_config = ConfigDict(extra="forbid")

    child: str
    parent: str
    edge: EdgeName
    state: EdgeState


class TreeNode(BaseModel):
    """One artifact in the forest: R-02.29's four facts, plus kind, plus two derived flags.

    Nullability mirrors the two vendored manifests as parsed, not as described.
    """

    model_config = ConfigDict(extra="forbid")

    #: The fully qualified name, ``<base_model>:<run_id>:step-<step>`` for a checkpoint and
    #: ``...:adapter-<step>`` for an adapter, with the step **unpadded**.
    name: str
    kind: NodeKind
    run_id: str
    #: The one canonical integer. The path renders it zero-padded to six digits and ``name``
    #: renders it unpadded; nothing carries a second copy as a string.
    step: int

    # ------------------------------------------------------------------ domain, four states
    domain: str | None
    domain_state: DomainStateValue
    #: The one rendering, from :func:`models.domain_state.render_domain`, so four states do
    #: not acquire one presentation per screen.
    domain_display: str

    # ------------------------------------------------------------------ the three edges
    #: Edge 1. Present for a checkpoint; ``None`` for an adapter, whose manifest has no
    #: ``parent`` field at all - emitting a ``parent`` for an adapter would assert a field
    #: R-02.29 says it does not have.
    parent: ResolvedEdge | None = None
    #: Edge 2. Present for an adapter; ``None`` for a checkpoint.
    base: ResolvedEdge | None = None
    #: The ``manifest_sha256`` from the adapter's own ``base`` block. ``None`` is a
    #: *conforming* value per the schema's description - *"the base was not itself a saved
    #: checkpoint"* - and not the same as the field being missing.
    base_manifest_sha256: str | None = None
    #: Edge 3, the merged checkpoint's **second** ancestor, **in addition** to
    #: :attr:`parent` and never instead of it. ``None`` for every other node. Dropping it
    #: is mutation **M7** and undoes the reason ``merged_adapter`` was added to
    #: ``checkpoint-manifest-v1.json`` on 2026-10-13.
    merged: ResolvedEdge | None = None
    merged_step: int | None = None
    merged_verification: MergedVerification | None = None

    # ------------------------------------------------------------------ the fourth fact
    #: R-02.29's final test score, **verbatim as the manifest carries it** - so ``n``,
    #: ``floor`` and ``provider`` ride along as recorded and none of them is defaulted.
    #: ``None`` when no test was recorded for this artifact.
    test: dict[str, Any] | None = None

    manifest_sha256: str | None = None
    #: Where the directory is, relative to the artifact root.
    path: str

    # ------------------------------------------------------------------ derived
    #: ``False`` when this artifact's ``run_id`` has no ``mf_runs`` row - a legitimate state
    #: (``reindex`` indexes a copied-in manifest, F5's OQ-4), rendered with the run named as
    #: not indexed. **It never removes the node**: filtering here is mutation **M10**.
    run_indexed: bool
    #: ``True`` when this node's index row disagreed with its manifest on disk. The
    #: manifest's values are the ones in this node (ADR-004) and the disagreement is logged.
    index_stale: bool = False


class TreePage(BaseModel):
    """Flat pagination over nodes. **Not by subtree** (FTDD section 5.2).

    A node whose edge target falls outside the page resolves ``not_loaded``, which is a
    different value from ``truncated``; conflating them would make a paging artefact look
    like a broken lineage.
    """

    model_config = ConfigDict(extra="forbid")

    limit: int = Field(ge=1)
    offset: int = Field(ge=0)
    #: Every node matching the filters, not just this page.
    total: int = Field(ge=0)


class TreeResponse(BaseModel):
    """The forest. **No ``depth`` anywhere in it** (FR-009.26)."""

    model_config = ConfigDict(extra="forbid")

    nodes: list[TreeNode]
    #: A projection of the nodes' edges; see :class:`TreeEdge`.
    edges: list[TreeEdge]
    page: TreePage
    #: The names of the nodes whose index row disagreed with the manifest (FR-009.29). A
    #: projection of :attr:`TreeNode.index_stale`, derived in the same pass, kept because
    #: the screen needs the set and not a scan.
    index_stale: list[str]
