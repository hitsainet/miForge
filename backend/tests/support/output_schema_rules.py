"""The output schemas' generic-tool recount, by the CONTRACT's rule (021 FR-021.16, FR-021.4).

A second copy of the rule ``backend/scripts/generate_output_schemas.py``'s ``_is_generic`` applies
(since review round 5 both use the contract's FR-021.4 rule, ``schemas.contract_tools.is_typed``,
so they are not independent): an own answer (a root branch other than ``Unavailable`` and the
pending approval) is generic when ``is_typed`` calls it untyped; a non-object answer the generator
wrapped (``x-mcp-wrapped``) is judged by what it wraps. What makes the checks bite is not the copy
but the hand-written labels of the planted cases, and the recount of the generated file:
``tests/unit/mcp/test_output_schemas.py`` recounts the file with this copy (a generator that lies
about its count disagrees with it) and runs planted schemas, each with its expected verdict, through
both.
"""

from __future__ import annotations

from typing import Any

WRAPPED = "x-mcp-wrapped"
ROOT_BRANCHES = ("#/$defs/Unavailable", "#/$defs/ApprovalPending")


def generic(schema: dict[str, Any]) -> bool:
    from schemas import contract_tools

    defs = schema.get("$defs", {})
    for branch in schema["anyOf"]:
        if branch.get("$ref") in ROOT_BRANCHES:
            continue
        target = contract_tools.deref(branch, defs)
        if isinstance(target, dict) and target.get(WRAPPED):
            branch = target["properties"]["result"]
        if not contract_tools.is_typed(branch, defs):
            return True
    return False
