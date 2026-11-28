"""The three edge resolutions, one function each  (FTDD section 4.3; task 3.2).

Pure: **no session, no path, no clock.** Each function takes the names it needs and
returns an edge. That is what makes mutations M6 - M10 cheap to run and what lets the
forest fixtures be round-tripped through F5's and F8's real writers and fed straight in.

## The vocabulary, and why each distinction is a separate value

| Edge | From | To | Failure |
|---|---|---|---|
| 1 | ``checkpoint.parent`` -> checkpoint ``name`` | ``null`` -> **root**; unresolved ->
    **truncated**; self or cyclic -> **broken** |
| 2 | ``adapter.base.name`` -> checkpoint ``name`` | unresolved -> **unparented**, naming
    the base it wants |
| 3 | ``checkpoint.merged_adapter.name`` -> adapter ``name`` | unresolved ->
    **unresolved**, naming it |

**``root`` and ``truncated`` are different return values and must stay so.** F5's FR-005.39
said why: *"one means 'the root', the other 'the record stops here', and conflating them
would make a partial tree look complete."* Returning ``(None, "truncated")`` for a null
parent is mutation **M9**; if that mutation ever survives, F5's own fix is unpinned and
that is a finding in F5, not here.

**``not_loaded`` is not ``truncated``.** The target is in the index and merely outside this
page or this filter. Implemented by passing the full set of names known to the query
alongside the page's own, never by inferring it.

**Cycle detection is on the walk, not on the edge.** :func:`resolve_parent` catches only a
self-parent; a longer cycle is caught by :func:`services.tree.compose`'s visited set, and
the node where the walk re-enters is marked ``broken`` while **every node stays visible**
(FR-009.30).

**:func:`resolve_base` takes no ``run_id`` argument at all.** Not "prefers the name" - the
function cannot see a ``run_id``, so mutation **M8** requires editing the signature, which
makes it visible in review as well as red in the suite. F8 FR-008.34: *"as a child of the
base its own manifest names - **not of the run's current head**."*
"""

from __future__ import annotations

from collections.abc import Mapping, Set
from typing import Any

from schemas.tree import MergedVerification, ResolvedEdge


def resolve_parent(
    node_name: str,
    parent: str | None,
    page_names: Set[str],
    all_names: Set[str],
) -> ResolvedEdge:
    """Place a checkpoint by its manifest's ``parent``  (edge 1).

    Args:
        node_name: the child's own fully qualified name.
        parent: the manifest's ``parent`` - a name, or ``None`` **asserting** this is a root.
        page_names: every node in this page, by name.
        all_names: every artifact name the index knows, a superset of ``page_names``
            whenever the query was filtered or bounded.

    Returns:
        The edge. A failed edge still **names** the ancestor it wanted, so the row can say
        which one did not resolve.
    """
    if parent is None:
        # An assertion, not an absence: this IS a root. `truncated` here is M9.
        return ResolvedEdge(target=None, state="root")
    if parent == node_name:
        return ResolvedEdge(target=parent, state="broken")
    if parent in page_names:
        return ResolvedEdge(target=parent, state="ok")
    if parent in all_names:
        # Present in the index, outside this page. A paging artefact, not a broken lineage.
        return ResolvedEdge(target=parent, state="not_loaded")
    return ResolvedEdge(target=parent, state="truncated")


def resolve_base(
    adapter_name: str,
    base_name: str | None,
    page_names: Set[str],
    all_names: Set[str],
) -> ResolvedEdge:
    """Place an adapter by **the base its own manifest names**  (edge 2).

    There is deliberately **no ``run_id`` parameter**; see the module docstring (M8).

    An unresolved base is ``unparented`` rather than ``truncated``: an adapter with no
    reachable base is not a chain that stops early, it is an artifact that cannot be placed
    at all, and FTDD section 4.3 names the two states separately for that reason.
    """
    if base_name is None or base_name == "":
        # An adapter with no base is not a root - every adapter composes with something.
        return ResolvedEdge(target=None, state="unparented")
    if base_name == adapter_name:
        return ResolvedEdge(target=base_name, state="broken")
    if base_name in page_names:
        return ResolvedEdge(target=base_name, state="ok")
    if base_name in all_names:
        return ResolvedEdge(target=base_name, state="not_loaded")
    return ResolvedEdge(target=base_name, state="unparented")


def merged_verification(merged: Mapping[str, Any]) -> MergedVerification:
    """Classify a ``merged_adapter`` block's ``manifest_sha256``  (E-12, E-12a).

    **Three states, not a boolean.** Key presence is read first and with ``in``, never with
    ``.get()``: ``.get()`` returns ``None`` for a missing key *and* for an explicit null,
    which is the collapse. ``manifest_sha256`` is ``required`` **and** nullable in
    ``checkpoint-manifest-v1.json`` - parsed from the file - so:

    * a hash -> ``verified``;
    * the key present and ``null`` -> ``unverified_null``, a **conforming** manifest whose
      edge is deliberately unverified. Never reported as a defect;
    * the key absent -> ``key_absent``, a missing *required* key: nonconforming.
    """
    if "manifest_sha256" not in merged:
        return "key_absent"
    return "unverified_null" if merged["manifest_sha256"] is None else "verified"


def resolve_merged(
    node_name: str,
    merged: Mapping[str, Any],
    page_names: Set[str],
    all_names: Set[str],
) -> ResolvedEdge:
    """Place a merged checkpoint's **second** ancestor, the adapter baked into it (edge 3).

    Returns ``unresolved`` **naming the adapter** when it is not in the index, so the row
    can say which adapter it could not find rather than reporting a generic failure.
    Verification of the edge is :func:`merged_verification`'s answer and is kept separate,
    because an edge that resolves and an edge that is verified are two different facts.
    """
    name = merged.get("name")
    if not isinstance(name, str) or name == "":
        return ResolvedEdge(target=None, state="unresolved")
    if name == node_name:
        return ResolvedEdge(target=name, state="broken")
    if name in page_names:
        return ResolvedEdge(target=name, state="ok")
    if name in all_names:
        return ResolvedEdge(target=name, state="not_loaded")
    return ResolvedEdge(target=name, state="unresolved")
