"""The per-turn output cap toggle (operator, 2026-11-21): ``output_cap`` ``on`` | ``off``.

- An endpoint field (default ``on``, today's behaviour) and a run override; the value in effect
  is frozen into the request (``output_cap: "off"`` when off, absent when on), for both
  providers, so ``request_sha256`` and a replay carry it.
- Off, the engine's requests name no cap (the gateway's window-aware default applies); the
  context guard then reserves the largest output seen, not a cap nothing sends; with thinking on
  the allowance is still sent, at most half the window, and the gateway keeps it at most half of
  the cap it sizes.
- The weightspace adapter's OpenAI client names no cap either.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from core.errors import Refusal
from tests.support import wsp_worker_env
from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.unit.wsp.test_thinking import MISERVER, body, frozen
from wsp import endpoints
from wsp.schemas import validate_frozen_request
from wsp_gateway import allowlist
from wsp_worker.miwsp_engine import context
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.client import GatewayClient
from wsp_worker.miwsp_engine.protocol import STRUCTURED, Reply, Usage
from wsp_worker.miwsp_engine.transcript import ScriptedClient

FIELD = allowlist.REASONING_ALLOWANCE_FIELD


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: Any) -> dict[str, Any]:
    return make_bundle(tmp_path_factory.mktemp("b"), rule="mean").manifest


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean")


# ----------------------------------------------------------------------------- the freeze


@pytest.mark.parametrize("provider", ["miwsp_engine", "weightspace"])
def test_the_value_in_effect_is_frozen_for_both_providers(
    bundle: dict[str, Any], provider: str
) -> None:
    b = body(bundle, provider={"id": provider})
    assert "output_cap" not in frozen(bundle, b).request  # on: absent, as before
    off = frozen(bundle, b, {**MISERVER, "output_cap": "off"})
    assert off.request["output_cap"] == "off"
    validate_frozen_request(off.request)
    assert off.request_sha256 != frozen(bundle, b).request_sha256
    # The run overrides the endpoint, both ways.
    assert frozen(bundle, {**b, "output_cap": "off"}).request["output_cap"] == "off"
    on = frozen(bundle, {**b, "output_cap": "on"}, {**MISERVER, "output_cap": "off"})
    assert "output_cap" not in on.request


@pytest.mark.parametrize("value", ["none", True, 0])
def test_a_bad_value_is_refused(bundle: dict[str, Any], value: Any) -> None:
    with pytest.raises(Refusal) as caught:
        frozen(bundle, body(bundle, output_cap=value))
    assert caught.value.code == "validation_failed"
    with pytest.raises(Refusal):
        endpoints.check_output_cap(value)


def test_it_is_patchable() -> None:
    assert "output_cap" in endpoints.PATCHABLE


# ----------------------------------------------------------------------------- the engine


class Recorder(ScriptedClient):
    def __init__(self, replies: list[dict[str, Any]], output: int = 900) -> None:
        super().__init__(replies)
        self.asked: list[int | None] = []
        self.fields: list[dict[str, Any]] = []
        self.output = output

    def complete(self, messages: Any, *, fields: Any, max_tokens: int | None) -> Reply:
        self.asked.append(max_tokens)
        self.fields.append(dict(fields))
        reply = super().complete(messages, fields=fields, max_tokens=max_tokens)
        reply.usage = Usage(100, self.output)
        return reply


def _call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [{"id": "c", "name": name, "arguments": args}],
    }


def test_off_names_no_cap_and_the_guard_reserves_what_was_seen(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    client = Recorder([_call("list_tensors", {}), _call("submit", {"answer": "a"})], output=900)
    line = line_for(tmp_path / "run", fx, "repair", None, python=False)
    line["spec"]["request"]["output_cap"] = "off"
    s = session_mod.Session(line, client=client)
    s.writer = None
    assert s.guard.cap == context.MIN_OUTPUT_TOKENS  # nothing seen yet: the gateway's floor
    s._set_thinking()
    s._loop()
    assert client.asked == [None, None]
    assert s.guard.cap == 900  # the largest output observed, not a cap nothing sends


def test_on_names_the_engines_cap(tmp_path: Path, fx: FixtureBundle) -> None:
    client = Recorder([_call("submit", {"answer": "a"})])
    s = session_mod.Session(
        line_for(tmp_path / "run", fx, "repair", None, python=False), client=client
    )
    s.writer = None
    s._loop()
    assert client.asked == [context.editor_output_cap(s.editor)]


def test_off_with_thinking_sends_the_allowance_as_a_share_of_the_window(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    reply = {
        "role": "assistant",
        "content": json.dumps({"tool": "submit", "arguments": {"answer": "a"}}),
    }
    client = Recorder([reply])
    line = line_for(
        tmp_path / "run",
        fx,
        "repair",
        None,
        python=False,
        thinking="on",
        thinking_tokens=20_000,
        editor={"action_protocol": STRUCTURED, "context_window_tokens": 32_768},
    )
    line["spec"]["request"]["output_cap"] = "off"
    s = session_mod.Session(line, client=client)
    s.writer = None
    s._set_thinking()
    s._loop()
    assert client.asked == [None]
    # At most half the window, and (fix 2026-11-21) the share the gateway leaves of the cap it
    # will size: half of it (test_engine_room.py pins the two sizings equal).
    assert 4_096 < client.fields[0][FIELD] <= 32_768 // 2


def test_the_client_body_names_no_cap_when_off() -> None:
    client = GatewayClient("openai_compatible", "m", {"ip": "127.0.0.1", "port": 9, "token": "t"})
    try:
        assert "max_tokens" not in client.body([], fields={}, max_tokens=None)
        assert client.body([], fields={}, max_tokens=7)["max_tokens"] == 7
    finally:
        client.close()


# ----------------------------------------------------------------------------- the gateway


@pytest.mark.parametrize(("cap", "allowance", "kept"), [(8000, 6000, 4000), (8000, 2000, 2000)])
def test_an_uncapped_request_keeps_the_reasoning_to_half_the_sized_cap(
    cap: int, allowance: int, kept: int
) -> None:
    out = allowlist.insert_cap(
        "openai_compatible", {FIELD: allowance}, "max_tokens", cap, uncapped=True
    )
    assert (out["max_tokens"], out[FIELD]) == (cap, kept)
    named = allowlist.insert_cap("openai_compatible", {FIELD: allowance}, "max_tokens", cap)
    assert named[FIELD] == min(allowance, cap - allowlist.ACTION_FLOOR_TOKENS)


# ----------------------------------------------------------------------------- weightspace


def test_the_adapters_openai_client_names_no_cap_when_off() -> None:
    python = wsp_worker_env.worker_python(("weightspace",))
    code = (
        "import json, sys; sys.path.insert(0, sys.argv[1]);"
        "from wsp_worker.adapters import weightspace_adapter as a;"
        "g={'ip':'127.0.0.1','port':9,'token':'t'}; e={'kind':'openai_compatible','model':'m'};"
        "print(json.dumps([a.build_llm(e, g).params, a.build_llm(e, g, output_cap=False).params]))"
    )
    done = subprocess.run(
        [python, "-I", "-c", code, str(wsp_worker_env.SRC)],
        capture_output=True,
        text=True,
        check=True,
    )
    on, off = json.loads(done.stdout)
    assert on.get("max_tokens") == context.editor_output_cap({}) and "max_tokens" not in off
