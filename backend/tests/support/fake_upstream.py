"""A recording fake editor endpoint for the gateway tests (FTASKS 6.3): an ``httpx`` transport.

It answers ``/v1/messages`` (Anthropic shape) and ``/v1/chat/completions`` (OpenAI shape) with
MAXIMAL usage by default: every body byte billed as an input token and the whole received output
cap billed as output, so a test that keeps the total within a cap proves the worst case holds.
As a fake miServer it answers ``X-miLLM-Model`` and ``X-miLLM-Steering`` (default ``none``), and
can be told to answer one of miServer's session errors.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx


@dataclass
class Call:
    url: str
    headers: dict[str, str]
    body: dict[str, Any]
    raw: bytes


@dataclass
class FakeUpstream:
    served_model: str = "editor-model;origin=hub;commits=0"
    steering: str | None = "none"
    #: ``(status, code)`` to answer instead of a completion, e.g. ``(409, "model_not_resident")``.
    error: tuple[int, str] | None = None
    #: Raise a transport error instead of answering: ``True`` or ``"connect"`` (a connection
    #: never made), ``"connect_timeout"``, ``"read_timeout"`` or ``"dropped"`` (after the request
    #: was sent; fix 2026-11-21 settles only these at the worst case).
    fail: bool | str = False
    #: The ``usage`` an ``error`` answer carries (fix 2026-11-21); none by default.
    error_usage: dict[str, Any] | None = None
    #: Replace the completion's ``usage`` (``"absent"`` drops the key; review OR1-016-8).
    usage: Any = None
    #: Feature 018: OpenAI assistant messages to answer in order (a scripted editor through the
    #: real gateway); empty means the default ``"ok"`` text.
    messages: list[dict[str, Any]] = field(default_factory=list)
    #: Live progress (2026-11-20): the answer's text, and its finish reason (Anthropic
    #: ``stop_reason`` / OpenAI ``finish_reason``) when not the default.
    text: str = "ok"
    finish: str | None = None
    #: Fix 2026-11-21: the prompt's tokens per body byte (1.0: every byte billed as a token).
    tokens_per_byte: float = 1.0
    #: A served context window (fix 2026-11-21), enforced as miServer's engines enforce it:
    #: ``"transformers"`` refuses ``prompt + max_tokens`` past it (400 ``context_length_exceeded``,
    #: naming the window and the prompt); ``"llamacpp"`` clamps ``max_tokens`` to what the window
    #: leaves (llama-cpp-python 0.3.36) and refuses only a prompt that fills it, naming the window.
    window: int | None = None
    engine: str = "transformers"
    #: Fix 2026-11-22: request fields this miServer REFUSES when present (400 naming the field as
    #: ``param``, as its request policy does for a field it cannot honour), and fields it reports
    #: as unused (``X-miLLM-Ignored-Fields``, as one older than the field does).
    refuse_fields: frozenset[str] = frozenset()
    ignored_fields: frozenset[str] = frozenset()
    #: Called with each request before it is answered (a test's clock, a wait).
    on_call: Callable[[Call], None] | None = None
    calls: list[Call] = field(default_factory=list)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def prompt_tokens(self, raw: bytes) -> int:
        return round(len(raw) * self.tokens_per_byte)

    def _context(self, raw: bytes, cap: int) -> tuple[httpx.Response | None, int]:
        """miServer's context rule: a refusal, or ``None`` and the cap the engine generates to."""
        if self.window is None:
            return None, cap
        prompt = self.prompt_tokens(raw)
        if self.engine == "llamacpp":
            if prompt >= self.window:
                message = (
                    f"Requested tokens ({prompt}) exceed context window of {self.window}. The "
                    "model is serving a smaller context than its file declares"
                )
                return self._context_refusal(message), cap
            return None, min(cap, self.window - prompt)
        if prompt + cap > self.window:
            message = (
                f"This model's maximum context length is {self.window} tokens. However, you "
                f"requested {prompt + cap} tokens ({prompt} in the prompt, {cap} for the "
                "completion). Shorten the prompt or ask for fewer max_tokens."
            )
            return self._context_refusal(message), cap
        return None, cap

    @staticmethod
    def _context_refusal(message: str) -> httpx.Response:
        answer = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "param": None,
                "code": "context_length_exceeded",
            }
        }
        return httpx.Response(400, json=answer)

    def _next_message(self) -> dict[str, Any]:
        with self._lock:
            if self.messages:
                return self.messages.pop(0)
        return {"role": "assistant", "content": self.text}

    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self)

    def client(self) -> httpx.Client:
        return httpx.Client(transport=self.transport(), follow_redirects=False)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        raw = request.read()
        call = Call(
            str(request.url),
            {k.lower(): v for k, v in request.headers.items()},
            json.loads(raw),
            raw,
        )
        with self._lock:
            self.calls.append(call)
        if self.on_call is not None:
            self.on_call(call)
        refused = sorted(self.refuse_fields & set(call.body))
        if refused:
            answer = {
                "error": {
                    "message": f"'{refused[0]}' is not honoured",
                    "type": "invalid_request_error",
                    "param": refused[0],
                    "code": "field_not_honoured",
                }
            }
            return httpx.Response(400, json=answer)
        if self.fail in (True, "connect"):
            raise httpx.ConnectError("refused", request=request)
        if self.fail == "connect_timeout":
            raise httpx.ConnectTimeout("no connection", request=request)
        if self.fail == "read_timeout":
            raise httpx.ReadTimeout("no answer", request=request)
        if self.fail == "dropped":
            raise httpx.RemoteProtocolError("dropped", request=request)
        headers = {"X-miLLM-Model": self.served_model}
        unused = sorted(self.ignored_fields & set(call.body))
        if unused:
            # miServer's encoding: an RFC 8941 list of strings (``encode_field_list``).
            headers["X-miLLM-Ignored-Fields"] = ", ".join(f'"{name}"' for name in unused)
        if self.steering is not None:
            headers["X-miLLM-Steering"] = self.steering
        if self.error is not None:
            status, code = self.error
            answer: dict[str, Any] = {"error": {"code": code, "message": code}}
            if self.error_usage is not None:
                answer["usage"] = self.error_usage
            return httpx.Response(status, json=answer, headers=headers)
        body = call.body
        if request.url.path.endswith("/messages"):
            payload: dict[str, Any] = {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": body.get("model"),
                "content": [{"type": "text", "text": self.text}],
                "stop_reason": self.finish or "end_turn",
                "usage": {
                    "input_tokens": len(raw),
                    "output_tokens": int(body.get("max_tokens", 0)),
                },
            }
        else:
            cap = int(body.get("max_tokens", body.get("max_completion_tokens", 0)))
            refused, generated = self._context(raw, cap)
            if refused is not None:
                return refused
            payload = {
                "id": "c1",
                "object": "chat.completion",
                "model": body.get("model"),
                "choices": [
                    {
                        "index": 0,
                        "message": self._next_message(),
                        "finish_reason": self.finish or ("length" if generated < cap else "stop"),
                    }
                ],
                "usage": {"prompt_tokens": self.prompt_tokens(raw), "completion_tokens": generated},
            }
        if self.usage == "absent":
            payload.pop("usage")
        elif isinstance(self.usage, str) and self.usage.startswith("raw:"):
            # Raw JSON text, for what json.dumps cannot write (``1e999`` parses as infinity).
            payload["usage"] = "__RAW__"
            text = json.dumps(payload).replace('"__RAW__"', self.usage[4:])
            return httpx.Response(
                200, content=text.encode(), headers={**headers, "content-type": "application/json"}
            )
        elif self.usage is not None:
            payload["usage"] = self.usage
        return httpx.Response(200, json=payload, headers=headers)
