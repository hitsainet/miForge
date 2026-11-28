"""The append-only rule of an editor conversation (feature 020; FR-020.7; F20-D15, F20-D21; TD-3).

One definition, used by the engine before each send (``session.Session._complete``) and by the
scripted upstream in CI (``tests/support/hosted_upstream.py``), so the two cannot drift:

1. ``system`` and ``tools`` are unchanged;
2. every message up to and including the previous request's last assistant message (the
   **anchor**) is unchanged;
3. every later message of the previous request is unchanged, except its LAST one, which may only
   gain appended content blocks (the lost-reply notice merged into a user turn not yet answered,
   ``client.anthropic_messages``);
4. the current request has at least the previous request's messages.

A previous request with no assistant message (a conversation's first) has no anchor: rules 1, 3
and 4 apply. A handoff starts a new conversation, so the session compares nothing across one.

**The comparison's normal form** (:func:`normal`): ``system`` as a list of text blocks; a
message's string content as one text block; every ``cache_control`` removed; tools keyed by
name. These are the differences the Messages API itself ignores ("string content versus a
single text block", ``cache_control`` markers, "JSON formatting or key order"; FTID §0, L3b).
Every other top-level field (``model``, ``max_tokens``, ``thinking``, ``tool_choice``, ...) is
outside the rule, as it is outside the API's (L3b).

Standard library only: the engine imports it, and so does a test fixture outside the worker
(``test_wsp_fences.py`` allows that for tests only).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

#: The key the API ignores in the comparison (FTID §0, L3b), removed at every depth.
CACHE_CONTROL = "cache_control"


def _strip(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _strip(v) for k, v in value.items() if k != CACHE_CONTROL}
    if isinstance(value, list | tuple):
        return [_strip(v) for v in value]
    return value


def _blocks(content: Any) -> Any:
    """A string content as one text block; anything else as it is (stripped)."""
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return _strip(content)


def _tool_name(tool: Mapping[str, Any]) -> str:
    function = tool.get("function")
    if isinstance(function, Mapping) and "name" in function:
        return str(function["name"])  # OpenAI: {"type": "function", "function": {"name": ...}}
    return str(tool.get("name", ""))  # Anthropic: {"name": ..., "input_schema": ...}


def normal(body: Mapping[str, Any]) -> dict[str, Any]:
    """``body`` in the comparison's normal form: ``system``, ``tools`` and ``messages`` only."""
    system = body.get("system")
    tools = body.get("tools") or []
    messages = body.get("messages") or []
    return {
        "system": [] if system is None else _blocks(system),
        "tools": {_tool_name(t): _strip(t) for t in tools if isinstance(t, Mapping)},
        "messages": [
            {
                **_strip({k: v for k, v in m.items() if k != "content"}),
                "content": _blocks(m.get("content")),
            }
            for m in messages
            if isinstance(m, Mapping)
        ],
    }


def canonical(value: Any) -> str:
    """Canonical JSON: sorted keys, no spaces, unescaped text (what "byte-identical" means)."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _anchor(messages: list[dict[str, Any]]) -> int:
    """The index of the last assistant message, or -1."""
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get("role") == "assistant":
            return i
    return -1


def _grown(before: dict[str, Any], after: dict[str, Any], index: int) -> str | None:
    """``None`` when ``after`` is ``before`` with content blocks appended, else the changed path."""
    head = {k: v for k, v in before.items() if k != "content"}
    if canonical(head) != canonical({k: v for k, v in after.items() if k != "content"}):
        return f"messages.{index}"
    old, new = before.get("content"), after.get("content")
    if not isinstance(old, list) or not isinstance(new, list):
        return None if canonical(old) == canonical(new) else f"messages.{index}"
    for j, block in enumerate(old):
        if j >= len(new) or canonical(block) != canonical(new[j]):
            return f"messages.{index}.content.{j}"
    return None


def check(previous: Mapping[str, Any] | None, current: Mapping[str, Any]) -> str | None:
    """``None`` when ``current`` keeps the append-only rule against ``previous`` (the request
    before it in the same conversation), else the first changed path: ``system``, ``tools``,
    ``messages`` (shorter), ``messages.<i>`` or ``messages.<i>.content.<j>``."""
    if previous is None:
        return None
    prev, cur = normal(previous), normal(current)
    for key in ("system", "tools"):
        if canonical(prev[key]) != canonical(cur[key]):
            return key
    before, after = prev["messages"], cur["messages"]
    if len(after) < len(before):
        return "messages"
    anchor = _anchor(before)
    last = len(before) - 1
    for i, message in enumerate(before):
        if i == last and i > anchor:
            changed = _grown(message, after[i], i)
            if changed is not None:
                return changed
        elif canonical(message) != canonical(after[i]):
            return f"messages.{i}"
    return None
