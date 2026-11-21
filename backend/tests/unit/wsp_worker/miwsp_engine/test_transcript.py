"""The engine's transcript and scripted replay client (feature 018; FTASKS 5.4; FR-018.13)."""

from __future__ import annotations

import json
from pathlib import Path

from wsp.result_archive import check_transcript
from wsp_worker.miwsp_engine.transcript import (
    EXHAUSTED,
    TranscriptWriter,
    read_header,
    read_script,
)


def _write(path: Path) -> None:
    writer = TranscriptWriter(path, {"provider": "miwsp_engine", "protocol": "structured"})
    writer.write({"role": "system", "content": "s"})
    writer.write({"role": "user", "content": "u"})
    writer.write(
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": "c0", "name": "python", "arguments": '{"code": "1"}'}],
        }
    )
    writer.write({"role": "tool", "tool_call_id": "c0", "name": "python", "content": "1"})
    writer.write({"event": "refused", "kind": "no_action", "n": 0})
    writer.write({"role": "assistant", "content": "done"})
    writer.write({"role": "user", "content": "Continue."})
    writer.close()


def test_the_header_line_then_one_json_object_per_line(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    _write(path)
    lines = path.read_text().splitlines()
    assert json.loads(lines[0]) == {
        "miwsp_transcript": 1,
        "provider": "miwsp_engine",
        "protocol": "structured",
    }
    assert all(isinstance(json.loads(x), dict) for x in lines)
    check_transcript(path.read_bytes())  # the backend's own intake check
    assert read_header(path)["protocol"] == "structured"


def test_the_script_replays_the_assistant_messages_in_order_then_runs_out(tmp_path: Path) -> None:
    path = tmp_path / "transcript.jsonl"
    _write(path)
    script = read_script(path)
    first = script.complete([], fields={}, max_tokens=1)
    assert [(c.name, c.arguments) for c in first.tool_calls] == [("python", '{"code": "1"}')]
    second = script.complete([], fields={}, max_tokens=1)
    assert second.text == "done" and second.tool_calls == []
    assert script.complete([], fields={}, max_tokens=1).text == EXHAUSTED
    assert script.recorded == [["1"], ["Continue."]]
