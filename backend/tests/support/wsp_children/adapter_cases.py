"""Scripted weightspace sessions through the REAL adapter, run under the worker's interpreter
(FTASKS 8.8). ``python adapter_cases.py <case> <child-line.json>`` runs one case with a
``ScriptedLLM`` and prints one JSON line of what the test needs to see from inside the library."""

from __future__ import annotations

import http.server
import json
import sys
import threading
from pathlib import Path
from typing import Any

from weightspace import ScriptedLLM
from weightspace.llm.base import LLMResponse
from weightspace.llm.mock import call, reply
from weightspace.usage import Usage

from wsp_worker.adapters import weightspace_adapter


def steps(case: str) -> list[LLMResponse]:
    if case == "repair":
        return [
            reply(None, call("python", code="W['weight'][0] += 1.0")),
            reply(None, call("submit", answer="done", confidence=0.7)),
        ]
    if case == "limits":
        return [
            reply(None, call("python", code="W['weight'][:3] += 1.0")),
            reply(None, call("submit", answer="first")),
            reply(None, call("python", code="revert('weight'); W['weight'][0] += 1.0")),
            reply(None, call("submit", answer="second")),
        ]
    if case == "create":
        return [
            reply(None, call("python", code="W['weight'][:] = 1.0")),
            reply(None, call("submit", answer="made")),
        ]
    if case == "read_write":
        code = "a = R['weight']; a.flags.writeable = True; a[0] = 5.0"
        return [
            reply(None, call("python", code=code)),
            reply(None, call("submit", answer="looked")),
        ]
    if case == "token_budget":
        heavy = Usage(input_tokens=600, output_tokens=0, requests=1)
        return [LLMResponse(text=None, tool_calls=[call("python", code="1")], tokens=heavy)] * 3
    if case == "late_start":
        # The supervisor had 5 s of a 600 s budget left: the library must stop at once.
        return [reply(None, call("python", code="1"))] * 3 + [reply("late", None)]
    if case == "observe":
        return [
            reply(None, call("forward_practice_metrics", split="train")),
            reply(None, call("forward_compare_to_parent", split="validation")),
            reply(None, call("submit", answer="observed")),
        ]
    if case == "program":
        return [
            reply(None, call("python", code="W['bias'][0] = 9.0")),  # read mode: no W
            reply(None, call("test_program", code=PROGRAM)),
            reply(None, call("submit_program", code="x = 1\n")),  # no edit: refused
            reply(None, call("submit_program", code=PROGRAM)),
            reply(None, call("submit_program", code=PROGRAM.replace("0.5", "0.25"))),
            reply(None, call("submit", answer="programmed")),
        ]
    raise SystemExit(f"unknown case {case}")


#: A deterministic program: every weight half a unit further from zero.
PROGRAM = (
    "def edit(base, support):\n"
    "    w = np.array(base['weight'])\n"
    "    return {'weight': (w + np.float32(0.5) * np.sign(w)).astype(w.dtype)}\n"
)


def client_case() -> None:
    gateway = {"ip": "10.152.183.20", "port": 8080, "token": "tok-SENTINEL"}
    seen = {}
    for kind in ("anthropic", "openai_compatible"):
        llm = weightspace_adapter.build_llm({"kind": kind, "model": "ed"}, gateway)
        client = llm.client
        seen[kind] = {
            "base_url": str(client.base_url),
            "api_key": client.api_key,
            "max_retries": client.max_retries,
            "stream": getattr(llm, "stream", None),
            "model": llm.model,
        }
    print(json.dumps(seen))


#: The editors the cap cases build an LLM for: (label, the ``spec.editor`` window keys).
CAP_EDITORS: list[tuple[str, dict[str, Any]]] = [
    ("no_window", {}),
    ("served_32768", {"served_context_tokens": 32_768}),
    ("endpoint_4096", {"context_window_tokens": 4_096, "served_context_tokens": 32_768}),
]


class Recorder:
    """A local stand-in for the gateway: records each request's path and JSON body, answers one
    plain-text completion in the shape the route's SDK reads."""

    def __init__(self) -> None:
        self.seen: list[tuple[str, dict[str, Any]]] = []
        recorder = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["content-length"])))
                recorder.seen.append((self.path, body))
                if self.path.endswith("/messages"):
                    answer: dict[str, Any] = {
                        "id": "msg_1",
                        "type": "message",
                        "role": "assistant",
                        "model": "ed",
                        "content": [{"type": "text", "text": "done"}],
                        "stop_reason": "end_turn",
                        "stop_sequence": None,
                        "usage": {"input_tokens": 1, "output_tokens": 1},
                    }
                else:
                    answer = {
                        "id": "c1",
                        "object": "chat.completion",
                        "created": 0,
                        "model": "ed",
                        "choices": [
                            {
                                "index": 0,
                                "message": {"role": "assistant", "content": "done"},
                                "finish_reason": "stop",
                            }
                        ],
                        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                    }
                data = json.dumps(answer).encode()
                self.send_response(200)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def log_message(self, *args: Any) -> None:
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def gateway(self) -> dict[str, Any]:
        return {"ip": "127.0.0.1", "port": self.server.server_address[1], "token": "tok"}

    def close(self) -> None:
        self.server.shutdown()


def cap_case() -> None:
    """Each kind's LLM as ``build_llm`` makes it, for each editor window, sends ONE request: what
    reached the wire is printed (the path and ``max_tokens``)."""
    recorder = Recorder()
    seen: dict[str, Any] = {}
    try:
        for kind in ("anthropic", "openai_compatible"):
            for label, window in CAP_EDITORS:
                editor = {"kind": kind, "model": "ed", **window}
                before = len(recorder.seen)
                llm = weightspace_adapter.build_llm(editor, recorder.gateway())
                llm.complete([{"role": "user", "content": "hi"}])
                sent = recorder.seen[before:]
                seen[f"{kind}/{label}"] = [
                    {"path": path, "max_tokens": body.get("max_tokens")} for path, body in sent
                ]
    finally:
        recorder.close()
    print(json.dumps(seen))


def run_cap_case(line_path: Path) -> None:
    """``run`` itself (no LLM injected) against the recorder as the gateway: every request the
    production path sends carries the cap."""
    recorder = Recorder()
    line = json.loads(line_path.read_text())
    line["gateway"] = recorder.gateway()
    try:
        code = weightspace_adapter.run(line)
    finally:
        recorder.close()
    print(
        json.dumps(
            {
                "exit": code,
                "sent": [
                    {"path": path, "max_tokens": body.get("max_tokens")}
                    for path, body in recorder.seen
                ],
            }
        )
    )


def main() -> None:
    if sys.argv[1] == "client":
        client_case()
        return
    if sys.argv[1] == "cap":
        cap_case()
        return
    if sys.argv[1] == "run_cap":
        run_cap_case(Path(sys.argv[2]))
        return
    case, line_path = sys.argv[1], Path(sys.argv[2])
    line = json.loads(line_path.read_text())
    if case == "replay":
        # The adapter builds its own ScriptedLLM from the pushed transcript (no gateway).
        print(json.dumps({"exit": weightspace_adapter.run(line), "tool_outputs": []}))
        return
    llm = ScriptedLLM(steps(case))
    code = weightspace_adapter.run(line, llm=llm)
    # The last request carries the whole history: each tool answer once, in order.
    last = llm.seen[-1] if llm.seen else []
    tool_outputs = [m.get("content") for m in last if m.get("role") == "tool"]
    first_user = next((str(m.get("content")) for m in last if m.get("role") == "user"), "")
    print(
        json.dumps(
            {
                "exit": code,
                "tool_outputs": tool_outputs,
                "program": PROGRAM,
                "first_user": first_user,
            }
        )
    )


if __name__ == "__main__":
    main()
