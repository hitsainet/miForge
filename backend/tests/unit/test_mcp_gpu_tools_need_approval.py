"""G9: a tool classed gpu/attach/delete/config reaches a route that WAITS  (FR-012.21; MR-14).

G5 starts from the routes; this starts from the tools - so a new tool pointing a GPU action at
an ungated route fails here even when the route list never changed. The class comes from the
ledger's column; what the route does with the tool's call comes from the LIVE gate's own
decision function, given the payload the tool actually sends (``mcp_gate_probe``). Neither
side is a hand list inside this test.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from tests.support.mcp_gate_probe import decision
from tests.support.mcp_ledger import CLASS_ACTION, EXPECTED_CALLS, GATED_CLASSES

GATED = sorted(tool for tool, entry in EXPECTED_CALLS.items() if entry[4] in GATED_CLASSES)
UNGATED = sorted(tool for tool, entry in EXPECTED_CALLS.items() if entry[4] not in GATED_CLASSES)


def test_the_ledger_classes_ten_tools_as_waiting_for_the_operator() -> None:
    assert len(GATED) == 10, GATED
    # 73 tools run at once (FTDD section 1): these 72 call a route, `forge_howto` calls none.
    # Plus the four archive tools (operator 2026-10-20): non-deleting writes, ungated (D9).
    # Plus the two tokenizer reads for a create run (2026-10-20): reads, ungated.
    assert len(UNGATED) == 79


@pytest.mark.parametrize("tool", GATED)
def test_a_gpu_attach_delete_or_config_tool_waits(tool: str) -> None:
    route, action = decision(tool)
    klass = EXPECTED_CALLS[tool][4]
    assert action is not None, (
        f"{tool} is classed {klass} but {sorted(route.methods)} {route.path} runs at once for "
        f"its call: an agent would start this without the operator."
    )
    assert action == CLASS_ACTION[klass], (tool, klass, action)


def _attach(recorded: dict[str, Any], tool: str) -> dict[str, Any]:
    """The same call, asking to attach to miLLM."""
    body = copy.deepcopy(recorded["json_body"])
    if tool == "forge_submit_run":
        body["mode"] = "attach"
        body.pop("path", None)
    elif tool == "forge_define_run":
        body["model_source"] = {"mode": "attach"}
    else:  # forge_create_chain: stage 0 only may attach
        body["stages"][0]["definition"]["model_source"] = {"mode": "attach"}
    return {**recorded, "json_body": body}


@pytest.mark.parametrize(
    "tool", sorted(t for t in GATED if EXPECTED_CALLS[t][4] == "gpu_or_attach")
)
def test_the_same_call_attaching_to_millm_escalates(tool: str) -> None:
    _route, action = decision(tool, recorded=_attach(EXPECTED_CALLS[tool][3], tool))
    assert action == "hotupdate_attach", (tool, action)


@pytest.mark.parametrize("tool", UNGATED)
def test_a_read_plan_or_write_tool_runs_at_once(tool: str) -> None:
    route, action = decision(tool)
    assert action is None, (
        f"{tool} is classed {EXPECTED_CALLS[tool][4]} but its call to {route.path} waits for "
        f"{action}: either the class or the gate is wrong."
    )
