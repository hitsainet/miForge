"""A scripted editor endpoint for ``miwsp_engine`` tests (feature 018; FTASKS 3.2, 4.2, 4.4).

A real HTTP server on ``127.0.0.1`` that answers ``POST /v1/chat/completions`` (OpenAI kind) or
``POST /v1/messages`` (Anthropic kind) with scripted replies, in order, and RECORDS every request
it receives (path, headers, parsed body). The engine reaches it exactly as it reaches the gateway:
the child line's ``gateway`` is ``{ip: 127.0.0.1, port, token}``.

A step is a dict:

- ``{"calls": [(name, args), ...], "text": str | None}``: a reply with those tool calls;
- ``{"json": obj}``: a reply whose content is ``json.dumps(obj)`` (structured mode);
- ``{"status": 409, "code": "editor_context_exceeded"}``: an error answer;
- ``"usage": (prompt_tokens, completion_tokens)`` overrides the counted usage;
- ``"reasoning": str`` (OpenAI kind) answers it as ``message.reasoning_content``, miServer's field
  for a thinking model's reasoning (2026-11-21).

With ``window`` set, the endpoint counts a request's prompt as ``ceil(bytes / 4)`` of its
messages and tools (a conservative tokenizer, never larger than the engine's own estimate),
reports that count as its usage, and answers ``400 context_length_exceeded`` (miServer's code)
when the prompt plus the requested ``max_tokens`` passes the window. Every request's count is
kept, so a test asserts the largest one directly.
"""

from __future__ import annotations

import json
import math
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

TOKEN = 'wspt-"SENTINEL"-5f0c9e1b7a4d2c83'
OPENAI = "openai_compatible"
ANTHROPIC = "anthropic"


def prompt_tokens(body: dict[str, Any]) -> int:
    counted = {k: body.get(k) for k in ("system", "messages", "tools", "response_format")}
    return math.ceil(len(json.dumps(counted).encode("utf-8")) / 4)


class ScriptedEndpoint:
    def __init__(
        self,
        kind: str,
        steps: list[dict[str, Any]],
        *,
        window: int | None = None,
        repeat_last: bool = False,
    ) -> None:
        self.kind, self.steps, self.window, self.repeat_last = (
            kind,
            list(steps),
            window,
            repeat_last,
        )
        self.requests: list[dict[str, Any]] = []
        self.counts: list[int] = []
        self._server: ThreadingHTTPServer | None = None

    # ------------------------------------------------------------------------- lifecycle

    def __enter__(self) -> ScriptedEndpoint:
        endpoint = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *_args: Any) -> None:
                return

            def do_POST(self) -> None:
                raw = self.rfile.read(int(self.headers.get("content-length") or 0))
                status, answer = endpoint.answer(self.path, dict(self.headers.items()), raw)
                data = json.dumps(answer).encode()
                self.send_response(status)
                self.send_header("content-type", "application/json")
                self.send_header("content-length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_exc: Any) -> None:
        assert self._server is not None
        self._server.shutdown()
        self._server.server_close()

    @property
    def gateway(self) -> dict[str, Any]:
        assert self._server is not None
        return {"ip": "127.0.0.1", "port": self._server.server_address[1], "token": TOKEN}

    # ------------------------------------------------------------------------- answers

    def answer(self, path: str, headers: dict[str, str], raw: bytes) -> tuple[int, dict[str, Any]]:
        body = json.loads(raw)
        self.requests.append({"path": path, "headers": headers, "body": body})
        count = prompt_tokens(body)
        self.counts.append(count + int(body.get("max_tokens") or 0))
        if self.window is not None and count + int(body.get("max_tokens") or 0) > self.window:
            return 400, {"error": {"code": "context_length_exceeded", "message": "too long"}}
        index = len(self.requests) - 1
        if index >= len(self.steps):
            if not self.repeat_last or not self.steps:
                return 500, {"error": {"code": "script_exhausted", "message": "no more steps"}}
            index = len(self.steps) - 1
        step = self.steps[index]
        if "status" in step:
            return int(step["status"]), {"error": {"code": step["code"], "message": "scripted"}}
        usage = step.get("usage") or (count, 10)
        calls = [
            (f"call_{index}_{i}", name, args)
            for i, (name, args) in enumerate(step.get("calls", []))
        ]
        text = json.dumps(step["json"]) if "json" in step else step.get("text")
        if self.kind == ANTHROPIC:
            blocks: list[dict[str, Any]] = [{"type": "text", "text": text}] if text else []
            blocks += [{"type": "tool_use", "id": c, "name": n, "input": a} for c, n, a in calls]
            return 200, {
                "id": f"msg_{index}",
                "type": "message",
                "role": "assistant",
                "content": blocks,
                "stop_reason": "tool_use" if calls else "end_turn",
                "usage": {"input_tokens": usage[0], "output_tokens": usage[1]},
            }
        message: dict[str, Any] = {"role": "assistant", "content": text}
        if "reasoning" in step:
            message["reasoning_content"] = step["reasoning"]
        if calls:
            message["tool_calls"] = [
                {"id": c, "type": "function", "function": {"name": n, "arguments": json.dumps(a)}}
                for c, n, a in calls
            ]
        return 200, {
            "id": f"cmpl_{index}",
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": usage[0], "completion_tokens": usage[1]},
        }

    def tool_names(self, index: int) -> list[str]:
        """The tool names the request ``index`` offered, as the endpoint RECEIVED them."""
        body = self.requests[index]["body"]
        if "response_format" in body:
            schema = body["response_format"]["json_schema"]["schema"]
            return [o["properties"]["tool"]["const"] for o in schema["anyOf"]]
        if self.kind == ANTHROPIC:
            return [t["name"] for t in body.get("tools", [])]
        return [t["function"]["name"] for t in body.get("tools", [])]
