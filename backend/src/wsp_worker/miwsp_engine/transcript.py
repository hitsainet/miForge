"""The engine's transcript and its replay (feature 018; FR-018.13; FTDD §6.8).

``out/transcript.jsonl`` (the name the intake's member set fixes, ``result_archive.py:19-30``):

- line 1: ``{"miwsp_transcript": 1, "provider", "provider_version", "protocol", "kind", "style",
  "thinking", "thinking_tokens"}`` (the last two since 2026-11-21: the frozen request's);
- then one JSON object per message as it is appended (system, user, assistant, tool), in the
  neutral shapes, plus ``{"event": ...}`` lines for refusals, handoffs and the thinking in effect
  (``{"event": "thinking", "sent", "reasoning_allowance", "output_cap"}``). An assistant line
  that came with reasoning carries it as ``reasoning: {text, source}``, beside ``content``:
  marked as the editor's reasoning, which a reviewer reads and nothing parses (the action is
  ``content`` / ``tool_calls`` only). The additions are new optional keys, so the format stays
  version 1 and an older transcript replays unchanged. Paging (2026-11-21) adds the header's
  ``paging`` (the configuration, or null when off: a replay pages by it) and three event lines:
  ``{"event": "ledger", "row"}`` per ledger row, ``{"event": "page_out", "turn", "reason",
  "pages": [{"page", "turn", "tool", "bytes"}]}`` per batch, ``{"event": "recall", "turn",
  "page", "found"}``. A paged result's full text is its tool line, written once when it was
  appended; the stub is never written. Feature 019 (FR-019.2) adds, right after the header of a
  session with an editor, ``{"event": "window", "tokens", "source", "reported", "ceiling",
  "ceiling_lowered", "warnings"}``: the window and the output ceiling the backend resolved at push
  (a replay writes none, and its script skips it). Every line is one JSON
  object (``check_transcript``), written and flushed as it happens, so a crash still leaves a file.

**Replay** (:func:`read_script`): the recorded assistant messages, in order, as the model's replies;
the tools are re-run. A ``{"event": "lost_reply", "code", "message"}`` line (fix 2026-11-22) is a
reply that never came: the replay's call at that turn raises the same lost-reply error, so the
session takes the same path (``session.py`` step 3c). When the script runs out the reply is
``(script exhausted)`` with no action, as ``weightspace``'s ``ScriptedLLM`` answers. A replay
sends nothing anywhere: no gateway, no token.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import IO, Any

from wsp_worker.miwsp_engine.client import ClientError
from wsp_worker.miwsp_engine.protocol import Reply, ToolCall

VERSION = 1
EXHAUSTED = "(script exhausted)"
#: The transcript event of a lost reply, and the key a script's slot for one carries.
LOST_EVENT = "lost_reply"
LOST_KEY = "lost"


class TranscriptWriter:
    def __init__(self, path: Path, header: Mapping[str, Any]) -> None:
        self._fh: IO[str] = path.open("w", encoding="utf-8")
        self.write({"miwsp_transcript": VERSION, **dict(header)})

    def write(self, obj: Mapping[str, Any]) -> None:
        self._fh.write(json.dumps(dict(obj), default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


class ScriptedClient:
    """Answers each request with the next recorded assistant message (a replay, or a test)."""

    def __init__(self, replies: Sequence[Mapping[str, Any]], recorded: Sequence[list[str]] = ()):
        self.replies = [dict(r) for r in replies]
        #: Per assistant turn, the contents of the messages that answered it in the original.
        self.recorded = [list(r) for r in recorded]
        self.calls = 0

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> Reply:
        if self.calls >= len(self.replies):
            self.calls += 1
            return Reply(EXHAUSTED, [], None, {"role": "assistant", "content": EXHAUSTED})
        message = self.replies[self.calls]
        self.calls += 1
        lost = message.get(LOST_KEY)
        if isinstance(lost, str):
            raise ClientError(0, lost, str(message.get("message", "")))
        calls = [
            ToolCall(str(c.get("id", "")), str(c.get("name", "")), c.get("arguments"))
            for c in message.get("tool_calls") or []
        ]
        content = message.get("content")
        thought = message.get("reasoning")
        reasoning = thought.get("text") if isinstance(thought, dict) else None
        return Reply(
            content if isinstance(content, str) else None,
            calls,
            None,
            dict(message),
            reasoning if isinstance(reasoning, str) else None,
        )

    def body(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> dict[str, Any]:
        return {}

    def complete_body(self, body: Mapping[str, Any]) -> Reply:
        """The next recorded reply, as :meth:`complete` (a replay sends nothing anywhere)."""
        return self.complete([], fields={}, max_tokens=None)


def read_header(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as fh:
        first = json.loads(fh.readline() or "{}")
    return first if isinstance(first, dict) and "miwsp_transcript" in first else {}


#: The window event's name (feature 019): what the backend resolved at push, line 2 of a session.
WINDOW_EVENT = "window"


def read_window_event(path: Path) -> dict[str, Any] | None:
    """The original session's ``{"event": "window"}`` line (written right after the header since
    019-P1), or ``None`` for a transcript from before it. Only its two figures are taken, and
    only as positive integers (a transcript is the worker's own output, read as data)."""
    with path.open(encoding="utf-8") as fh:
        for _ in range(3):  # the header, then the event; nothing else comes first
            raw = fh.readline()
            if not raw:
                return None
            try:
                obj = json.loads(raw)
            except ValueError:
                return None
            if isinstance(obj, dict) and obj.get("event") == WINDOW_EVENT:
                tokens, ceiling = obj.get("tokens"), obj.get("ceiling")
                if not all(
                    isinstance(v, int) and not isinstance(v, bool) and v > 0
                    for v in (tokens, ceiling)
                ):
                    return None
                return {"tokens": tokens, "ceiling": ceiling}
    return None


def read_script(path: Path) -> ScriptedClient:
    """The original session's replies, and what answered each, from its transcript."""
    replies: list[dict[str, Any]] = []
    recorded: list[list[str]] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            continue
        role = obj.get("role")
        if obj.get("event") == LOST_EVENT and isinstance(obj.get("code"), str):
            replies.append({LOST_KEY: obj["code"], "message": str(obj.get("message", ""))})
            recorded.append([])
        elif role == "assistant":
            replies.append(obj)
            recorded.append([])
        elif role in ("tool", "user") and recorded:
            recorded[-1].append(str(obj.get("content", "")))
    return ScriptedClient(replies, recorded)
