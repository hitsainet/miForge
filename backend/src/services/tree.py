"""``compose()`` - the one algorithm in F9  (R-02.29; FR-009.21 - FR-009.30; task 3.3).

**Pure over its inputs: no session, no filesystem, no clock.** It takes the index rows as
``services.reindex``' read seam already renders them, the set of run ids that have a run
row, and every artifact name the index knows; it returns a :class:`schemas.tree.TreeResponse`.
That purity is what makes the forest fixtures round-trippable through F5's and F8's real
writers and what makes the five composer mutations cheap to run.

## The five rules, each with a named control

* **A merged checkpoint gets BOTH edges** - ``merged`` is assigned **in addition** to
  ``parent``, never instead of it. An ``elif`` or an early return here is mutation **M7**,
  and it undoes the reason ``merged_adapter`` was added to
  ``checkpoint-manifest-v1.json``: *"a merged checkpoint has two ancestors, and ``parent``
  carries only the base."*
* **An adapter joins on edge 2, never on ``run_id``** - :func:`services.tree_edges.resolve_base`
  cannot see a ``run_id``, so mutation **M8** needs a signature change. If M8 survives,
  F8 FR-008.34's placement rule is unpinned and that is a finding in F8.
* **``root`` and ``truncated`` are different** - mutation **M9**. If it survives, F5
  FR-005.39's own E-11 fix is unpinned and that is a finding in F5.
* **``run_indexed: False`` never removes a node** - mutation **M10**. ``reindex`` indexes a
  valid manifest whose ``run_id`` has no run row (F5's OQ-4) and FR-009.28 renders it; a
  checkpoint silently absent from the tree is strictly worse than one whose run is unknown.
* **``domain``'s four states are carried, not collapsed** - mutation **M6**. The index
  already distinguishes them in two columns (:mod:`models.domain_state`); reading
  ``domain`` alone, or treating an absent key as ``null``, is the collapse, and M6 pins
  the BRD's 2026-10-13 nullable-but-required decision.

## What it must not emit

**No ``depth``** (FR-009.26) and **no pack** (FR-009.24, FPRD F-10). And **an adapter's
``domain`` is never read from its base** - that is a fabricated value (ADR-018), and FPRD
F-11 reports that R-02.29's own field list is unsatisfiable for that artifact kind.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Final

from core.errors import Refusal, refuse
from schemas.tree import (
    EdgeName,
    MergedVerification,
    ResolvedEdge,
    TreeEdge,
    TreeNode,
    TreePage,
    TreeResponse,
)
from services.tree_edges import merged_verification, resolve_base, resolve_merged, resolve_parent

# ======================================================================= the refusal codes

#: A filter value that cannot be parsed. **Never widened to "no filter"** - a silently
#: widened query returns more than was asked for and looks like success (task 3.4).
INVALID_FILTER: Final = "invalid_filter"
#: A ``run_id`` filter naming neither a run row nor any artifact. Names which lookup failed.
RUN_NOT_FOUND: Final = "run_not_found"
#: The index cannot be read. Names the dependency and says ``reindex`` rebuilds it from
#: manifests (ADR-004), because that is literally the recovery.
INDEX_UNAVAILABLE: Final = "index_unavailable"

#: Default page size. Quoted against FTDD section 9.1's sample - *a forest of 500 nodes,
#: under 300 ms server time* - so the default page is the sample the budget was taken
#: against rather than an unrelated round number.
DEFAULT_LIMIT: Final = 500
MAX_LIMIT: Final = 2000


# ======================================================================= the domain filter


class DomainFilterKind(StrEnum):
    """Which of the three things a ``domain=`` query parameter can mean."""

    ANY = "any"
    NAMED = "named"
    NO_DOMAIN = "no_domain"


#: Spellings that *look* like a null literal but are not the accepted one. Refused rather
#: than resolved either way: treating ``None`` as a domain **name** returns nothing and
#: reads as a legitimately empty answer, while treating it as the null literal is guessing
#: what the caller meant. Both are silent; ADR-018 says refuse.
_NULLISH: Final = frozenset(
    {"none", "nil", "nothing", "undefined", "n/a", "na", "~", "()", "[]", "{}", '"null"', "'null'"}
)

#: The literal that selects nodes asserting **no** domain. Exactly this spelling, in this
#: case: ``NULL`` could as easily be a domain called NULL.
NO_DOMAIN_LITERAL: Final = "null"


@dataclass(frozen=True, slots=True)
class DomainFilter:
    """A parsed ``domain=`` filter. Built only by :meth:`parse`, which refuses rather than
    guesses."""

    kind: DomainFilterKind
    name: str | None = None

    @classmethod
    def parse(cls, raw: str | None) -> DomainFilter:
        """Parse the query parameter.

        ``None`` (the parameter absent) means **no filter**. The literal ``"null"`` means
        *nodes that assert no domain* - which is ``asserted_null`` and **only** that:
        ``absent`` is a missing required key and ``not_in_kind`` is an adapter, and neither
        is a node saying "I have no domain" (FR-009.25).

        Raises:
            Refusal: ``invalid_filter``, carrying the value and the accepted forms. An
                empty or whitespace-only value is refused because an empty string is not a
                domain name and is far likelier to be a mis-sent null; a null-ish spelling
                that is not exactly ``null`` is refused for the reason given at
                :data:`_NULLISH`.
        """
        if raw is None:
            return cls(DomainFilterKind.ANY)
        if raw == NO_DOMAIN_LITERAL:
            return cls(DomainFilterKind.NO_DOMAIN)
        stripped = raw.strip()
        if stripped == "" or stripped.lower() in _NULLISH or stripped.lower() == "null":
            raise _invalid_domain(raw)
        return cls(DomainFilterKind.NAMED, name=raw)


def _invalid_domain(raw: str) -> Refusal:
    return refuse(
        INVALID_FILTER,
        "The 'domain' filter accepts a domain name, or the literal 'null' for artifacts "
        "that assert no domain. Drop the parameter to filter by nothing; it is not widened "
        "for you, because a query that returned more than was asked for would look like a "
        "success.",
        filter="domain",
        value=raw,
        accepted=["<domain name>", NO_DOMAIN_LITERAL],
    )


def parse_kind(raw: str | None) -> str | None:
    """Parse the ``kind=`` filter. ``None`` means both kinds.

    Raises:
        Refusal: ``invalid_filter`` for anything but ``checkpoint`` or ``adapter``. Not
            widened to "both": an unrecognised kind is a caller that believes in a third
            artifact kind, and answering with everything confirms the belief.
    """
    if raw is None:
        return None
    if raw in ("checkpoint", "adapter"):
        return raw
    raise refuse(
        INVALID_FILTER,
        "The 'kind' filter accepts 'checkpoint' or 'adapter'. Drop the parameter for both.",
        filter="kind",
        value=raw,
        accepted=["checkpoint", "adapter"],
    )


# ======================================================================= compose


def _test_record(value: Any) -> dict[str, Any] | None:
    """R-02.29's fourth fact, verbatim.

    ``ScoreRecord.required`` is ``[set, score, n]``. A record missing any of the three is
    not a score with a hole in it - it is not a score - so it is reported absent rather
    than part-rendered with a fabricated sample. ``floor`` and ``provider`` ride along
    exactly as recorded and **neither is defaulted**: ``provider`` is nullable *and* not
    ``required``, so a conforming record may omit the key or carry null, and only passing
    it through preserves which.
    """
    if not isinstance(value, Mapping):
        return None
    if not isinstance(value.get("score"), int | float) or isinstance(value.get("score"), bool):
        return None
    if not isinstance(value.get("n"), int) or isinstance(value.get("n"), bool):
        return None
    if not isinstance(value.get("set"), str):
        return None
    return dict(value)


def _row_str(row: Mapping[str, Any], key: str) -> str:
    value = row.get(key)
    return value if isinstance(value, str) else ""


def compose(
    checkpoints: Sequence[Mapping[str, Any]],
    adapters: Sequence[Mapping[str, Any]],
    known_run_ids: Set[str],
    all_names: Set[str],
    *,
    index_stale: Set[str] = frozenset(),
    limit: int = DEFAULT_LIMIT,
    offset: int = 0,
) -> TreeResponse:
    """Compose the forest from index rows.

    Args:
        checkpoints: ``mf_checkpoints`` rows as ``services.reindex._checkpoint_payload``
            renders them, already filtered. **The manifest's values where the two
            disagreed** - the overlay is the caller's, because the manifest lives on disk
            and this function touches no disk (ADR-004, FR-009.29).
        adapters: ``mf_adapters`` rows, likewise.
        known_run_ids: run ids that have an ``mf_runs`` row. A node whose run is absent is
            **marked, never dropped** (M10).
        all_names: every artifact name the index knows, including ones this page or this
            filter excludes. A target found here but not in the page is ``not_loaded``,
            which is not ``truncated``.
        index_stale: names whose index row disagreed with the manifest.
        limit, offset: flat pagination over nodes, **not by subtree**.

    Returns:
        The forest. Nodes are ordered by name so the answer is stable across a rebuild -
        which is what foundation task 4.4's equivalence comparison requires.
    """
    raw: list[tuple[str, str, Mapping[str, Any]]] = [
        *(("checkpoint", _row_str(row, "name"), row) for row in checkpoints),
        *(("adapter", _row_str(row, "name"), row) for row in adapters),
    ]
    raw.sort(key=lambda item: item[1])

    total = len(raw)
    window = raw[offset : offset + limit] if limit > 0 else []
    page_names: frozenset[str] = frozenset(name for _, name, _ in window)

    nodes: list[TreeNode] = []
    for kind, name, row in window:
        if kind == "checkpoint":
            nodes.append(
                _checkpoint_node(
                    name, row, page_names, all_names, known_run_ids, name in index_stale
                )
            )
        else:
            nodes.append(
                _adapter_node(name, row, page_names, all_names, known_run_ids, name in index_stale)
            )

    _mark_cycles(nodes)

    return TreeResponse(
        nodes=nodes,
        edges=_edges(nodes),
        page=TreePage(limit=limit, offset=offset, total=total),
        index_stale=[node.name for node in nodes if node.index_stale],
    )


def _checkpoint_node(
    name: str,
    row: Mapping[str, Any],
    page_names: Set[str],
    all_names: Set[str],
    known_run_ids: Set[str],
    stale: bool,
) -> TreeNode:
    parent_value = row.get("parent")
    parent = resolve_parent(
        name,
        parent_value if isinstance(parent_value, str) else None,
        page_names,
        all_names,
    )

    # BOTH edges. `merged` is assigned IN ADDITION to `parent` - there is no `elif` and no
    # early return on this path, and introducing one is M7.
    merged: ResolvedEdge | None = None
    merged_step: int | None = None
    verification: MergedVerification | None = None
    merged_block = row.get("merged_adapter")
    if isinstance(merged_block, Mapping):
        merged = resolve_merged(name, merged_block, page_names, all_names)
        verification = merged_verification(merged_block)
        step_value = merged_block.get("step")
        merged_step = step_value if isinstance(step_value, int) else None

    return TreeNode(
        name=name,
        kind="checkpoint",
        run_id=_row_str(row, "run_id"),
        step=int(row.get("step") or 0),
        domain=row.get("domain") if isinstance(row.get("domain"), str) else None,
        # Read from `domain_state`, never derived from `domain`'s truthiness. Deriving it is
        # the collapse M6 performs.
        domain_state=_row_str(row, "domain_state") or "absent",  # type: ignore[arg-type]
        domain_display=_row_str(row, "domain_display"),
        parent=parent,
        base=None,
        base_manifest_sha256=None,
        merged=merged,
        merged_step=merged_step,
        merged_verification=verification,
        test=_test_record(row.get("test")),
        manifest_sha256=(
            row.get("manifest_sha256") if isinstance(row.get("manifest_sha256"), str) else None
        ),
        path=_row_str(row, "path"),
        run_indexed=_row_str(row, "run_id") in known_run_ids,
        index_stale=stale,
    )


def _adapter_node(
    name: str,
    row: Mapping[str, Any],
    page_names: Set[str],
    all_names: Set[str],
    known_run_ids: Set[str],
    stale: bool,
) -> TreeNode:
    base_name = row.get("base_name")
    return TreeNode(
        name=name,
        kind="adapter",
        run_id=_row_str(row, "run_id"),
        step=int(row.get("step") or 0),
        # `adapter-manifest-v1.json` declares NO `domain` property and closes its objects,
        # so a conforming adapter manifest cannot carry one. The base's domain is not the
        # adapter's and is never read across to fill this in (ADR-018, FPRD F-11).
        domain=None,
        domain_state="not_in_kind",
        domain_display=_row_str(row, "domain_display"),
        parent=None,
        base=resolve_base(
            name,
            base_name if isinstance(base_name, str) else None,
            page_names,
            all_names,
        ),
        base_manifest_sha256=(
            row.get("base_manifest_sha256")
            if isinstance(row.get("base_manifest_sha256"), str)
            else None
        ),
        merged=None,
        merged_step=None,
        merged_verification=None,
        test=_test_record(row.get("test")),
        manifest_sha256=(
            row.get("manifest_sha256") if isinstance(row.get("manifest_sha256"), str) else None
        ),
        path=_row_str(row, "path"),
        run_indexed=_row_str(row, "run_id") in known_run_ids,
        index_stale=stale,
    )


def _mark_cycles(nodes: list[TreeNode]) -> None:
    """Mark the node at which a ``parent`` walk re-enters itself  (FR-009.30).

    :func:`services.tree_edges.resolve_parent` catches only a self-parent. A longer cycle -
    ``a -> b -> a``, or ``a -> b -> c -> a`` - is caught here by a visited set, and **every
    node stays visible**: the edge is named broken, the artifact is not hidden. Without
    this the client's depth walk would not terminate.

    Mutates in place because the nodes are this function's own freshly built objects and a
    second list would be a second answer to the same question.

    **Every walk is taken before any mark is applied.** Marking as it goes would make the
    answer depend on node order: in ``a -> b -> a`` the walk from ``a`` marks ``a``'s edge
    broken, and the later walk from ``b`` then stops at ``a`` because that edge is no
    longer ``ok`` - so ``b`` is left reading ``ok`` and half a cycle renders as a healthy
    chain. Caught by ``test_a_LONGER_cycle_is_caught_by_the_visited_set_and_both_nodes_survive``
    against exactly that draft.
    """
    by_name = {node.name: node for node in nodes}
    broken: list[TreeNode] = []
    for node in nodes:
        if node.parent is None or node.parent.state != "ok":
            continue
        seen = {node.name}
        cursor: TreeNode | None = node
        while cursor is not None and cursor.parent is not None and cursor.parent.state == "ok":
            target = cursor.parent.target
            if target is None:
                break
            if target in seen:
                broken.append(node)
                break
            seen.add(target)
            cursor = by_name.get(target)
    for node in broken:
        assert node.parent is not None  # only reachable via the loop above
        node.parent = ResolvedEdge(target=node.parent.target, state="broken")


def _edges(nodes: Sequence[TreeNode]) -> list[TreeEdge]:
    """Project the nodes' edges into the flat graph form.

    Derived from the nodes in one pass so the two renderings cannot disagree. An edge is
    emitted only where the artifact **names** a target: a root names none, so a root
    contributes no row - which is what ``services.reindex.checkpoint_tree`` already did and
    what the tests written against it mean.
    """
    out: list[TreeEdge] = []
    for node in nodes:
        pairs: list[tuple[EdgeName, ResolvedEdge | None]] = [
            ("parent", node.parent),
            ("base", node.base),
            ("merged_adapter", node.merged),
        ]
        for edge_name, edge in pairs:
            if edge is None or edge.target is None:
                continue
            out.append(
                TreeEdge(child=node.name, parent=edge.target, edge=edge_name, state=edge.state)
            )
    out.sort(key=lambda edge: (edge.child, edge.edge))
    return out
