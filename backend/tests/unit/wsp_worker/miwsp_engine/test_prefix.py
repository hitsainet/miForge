"""H-1: the append-only rule (feature 020; FR-020.7; F20-D15; FTID §3.1).

Each rule passes once and fails once; the differences the Messages API ignores pass (a string
versus one text block, a moved ``cache_control``, a reordered tool list, key order); the
lost-reply merge passes; an edited anchor, an inserted message and a shortened history fail with
the path. Pure functions, standard library: no worker interpreter needed.
"""

from __future__ import annotations

import copy
from typing import Any

from wsp_worker.miwsp_engine import prefix

TOOLS = [
    {"name": "read_tensor", "description": "read", "input_schema": {"type": "object"}},
    {"name": "submit", "description": "finish", "input_schema": {"type": "object"}},
]


def body(messages: list[dict[str, Any]], **extra: Any) -> dict[str, Any]:
    return {
        "model": "m",
        "system": "You are a weight-space programmer.",
        "tools": copy.deepcopy(TOOLS),
        "messages": copy.deepcopy(messages),
        **extra,
    }


USER0 = {"role": "user", "content": "the task"}
ASSIST1 = {
    "role": "assistant",
    "content": [
        {"type": "thinking", "thinking": "plan", "signature": "sig1"},
        {"type": "tool_use", "id": "t1", "name": "read_tensor", "input": {"name": "W"}},
    ],
}
RESULT1 = {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "W"}],
}
ASSIST2 = {
    "role": "assistant",
    "content": [{"type": "tool_use", "id": "t2", "name": "submit", "input": {"answer": "x"}}],
}
RESULT2 = {
    "role": "user",
    "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "ok"}],
}


def test_the_first_request_of_a_conversation_has_nothing_to_compare() -> None:
    assert prefix.check(None, body([USER0])) is None


def test_an_honest_next_request_passes() -> None:
    first = body([USER0, ASSIST1, RESULT1])
    assert prefix.check(body([USER0]), first) is None
    assert prefix.check(first, body([USER0, ASSIST1, RESULT1, ASSIST2, RESULT2])) is None


# ----------------------------------------------------------------- rule 1: system and tools


def test_a_changed_system_fails_by_name() -> None:
    previous = body([USER0])
    current = body([USER0, ASSIST1, RESULT1])
    current["system"] = "Another system text."
    assert prefix.check(previous, current) == "system"


def test_a_changed_tool_list_fails_by_name() -> None:
    previous = body([USER0])
    current = body([USER0, ASSIST1, RESULT1])
    current["tools"] = current["tools"][1:]  # the final turn's submit-only list (pre-020)
    assert prefix.check(previous, current) == "tools"


def test_a_reordered_tool_list_passes() -> None:
    previous = body([USER0])
    current = body([USER0, ASSIST1, RESULT1])
    current["tools"] = list(reversed(current["tools"]))
    assert prefix.check(previous, current) is None


def test_the_system_as_a_string_or_as_one_cached_text_block_is_the_same() -> None:
    previous = body([USER0])
    current = body([USER0, ASSIST1, RESULT1])
    current["system"] = [
        {"type": "text", "text": previous["system"], "cache_control": {"type": "ephemeral"}}
    ]
    current["cache_control"] = {"type": "ephemeral"}
    assert prefix.check(previous, current) is None


# ----------------------------------------------------------------- rule 2: up to the anchor


def test_an_edited_anchor_fails_with_its_path() -> None:
    previous = body([USER0, ASSIST1, RESULT1])
    edited = copy.deepcopy(ASSIST1)
    edited["content"][0]["thinking"] = "a different plan"
    assert prefix.check(previous, body([USER0, edited, RESULT1, ASSIST2])) == "messages.1"


def test_an_edited_message_before_the_anchor_fails_with_its_path() -> None:
    previous = body([USER0, ASSIST1, RESULT1, ASSIST2, RESULT2])
    stubbed = copy.deepcopy(RESULT1)
    stubbed["content"][0]["content"] = '[page p1: read_tensor name=W, 12052 bytes; recall("p1")]'
    current = body([USER0, ASSIST1, stubbed, ASSIST2, RESULT2, ASSIST1])
    assert prefix.check(previous, current) == "messages.2"


def test_an_inserted_message_fails_with_its_path() -> None:
    previous = body([USER0, ASSIST1, RESULT1])
    inserted = body([USER0, {"role": "user", "content": "ledger"}, ASSIST1, RESULT1, ASSIST2])
    assert prefix.check(previous, inserted) == "messages.1"


# ----------------------------------------------------------------- rule 3: the trailing turn


def test_the_lost_reply_notice_merged_into_the_unanswered_turn_passes() -> None:
    previous = body([USER0, ASSIST1, RESULT1])
    grown = copy.deepcopy(RESULT1)
    grown["content"].append({"type": "text", "text": "Your last reply was lost."})
    assert prefix.check(previous, body([USER0, ASSIST1, grown])) is None


def test_a_string_first_message_that_gains_a_block_passes() -> None:
    previous = body([USER0])
    grown = {
        "role": "user",
        "content": [{"type": "text", "text": "the task"}, {"type": "text", "text": "notice"}],
    }
    assert prefix.check(previous, body([grown])) is None


def test_a_trailing_turn_that_loses_or_changes_a_block_fails_with_its_path() -> None:
    previous = body([USER0, ASSIST1, RESULT1])
    changed = copy.deepcopy(RESULT1)
    changed["content"][0]["content"] = "W, cut later"
    assert prefix.check(previous, body([USER0, ASSIST1, changed])) == "messages.2.content.0"
    emptied = {"role": "user", "content": []}
    assert prefix.check(previous, body([USER0, ASSIST1, emptied])) == "messages.2.content.0"


def test_a_trailing_turn_whose_role_changed_fails_with_its_path() -> None:
    previous = body([USER0, ASSIST1, RESULT1])
    moved = {**copy.deepcopy(RESULT1), "role": "assistant"}
    assert prefix.check(previous, body([USER0, ASSIST1, moved])) == "messages.2"


def test_a_moved_cache_control_marker_passes() -> None:
    marked = copy.deepcopy(RESULT1)
    marked["content"][0]["cache_control"] = {"type": "ephemeral"}
    previous = body([USER0, ASSIST1, marked])
    assert prefix.check(previous, body([USER0, ASSIST1, RESULT1, ASSIST2])) is None


# ----------------------------------------------------------------- rule 4: never shorter


def test_a_shortened_history_fails() -> None:
    previous = body([USER0, ASSIST1, RESULT1])
    assert prefix.check(previous, body([USER0, ASSIST1])) == "messages"


# ----------------------------------------------------------------- OpenAI-kind bodies


def oa(messages: list[dict[str, Any]]) -> dict[str, Any]:
    tools = [
        {"type": "function", "function": {"name": n, "parameters": {"type": "object"}}}
        for n in ("read_tensor", "submit")
    ]
    return {"model": "m", "messages": copy.deepcopy(messages), "tools": tools}


SYS = {"role": "system", "content": "s"}
OA_ASSIST = {
    "role": "assistant",
    "content": None,
    "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "read_tensor", "arguments": "{}"}}
    ],
}
OA_TOOL = {"role": "tool", "tool_call_id": "c1", "content": "W"}


def test_an_openai_body_keeps_the_rule_and_keys_tools_by_function_name() -> None:
    previous = oa([SYS, {"role": "user", "content": "task"}, OA_ASSIST, OA_TOOL])
    nxt = oa(
        [
            SYS,
            {"role": "user", "content": "task"},
            OA_ASSIST,
            OA_TOOL,
            {"role": "user", "content": "lost"},
        ]
    )
    assert prefix.check(previous, nxt) is None
    nxt["tools"] = list(reversed(nxt["tools"]))
    assert prefix.check(previous, nxt) is None
    nxt["tools"] = nxt["tools"][:1]
    assert prefix.check(previous, nxt) == "tools"


def test_an_openai_tool_result_edited_after_the_anchor_fails() -> None:
    previous = oa(
        [
            SYS,
            {"role": "user", "content": "task"},
            OA_ASSIST,
            OA_TOOL,
            {"role": "user", "content": "n"},
        ]
    )
    edited = {**OA_TOOL, "content": "stub"}
    current = oa(
        [
            SYS,
            {"role": "user", "content": "task"},
            OA_ASSIST,
            edited,
            {"role": "user", "content": "n"},
        ]
    )
    assert prefix.check(previous, current) == "messages.3"


def test_normal_and_canonical_are_stable_under_key_order() -> None:
    a = body([USER0, ASSIST1])
    b = {k: a[k] for k in reversed(list(a))}
    assert prefix.canonical(prefix.normal(a)) == prefix.canonical(prefix.normal(b))
