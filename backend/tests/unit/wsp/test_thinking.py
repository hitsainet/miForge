"""The backend's and the gateway's half of the editor's thinking switch (2026-11-21;
``0xcc/reviews/018_thinking_2026-11-21.md``).

- One name per figure: the reasoning-allowance field, the default allowance and the action's
  floor are each stated in two packages that do not import each other, and pinned equal here.
- The endpoint's ``thinking`` (``off`` | ``on`` | ``default``) and ``thinking_tokens``: ``on``
  only where ``enable_thinking`` can reach the server, refused by name elsewhere.
- ``freeze`` freezes the thinking IN EFFECT (the start body's over the endpoint's) for
  ``miwsp_engine``, so ``request_sha256`` moves with it; ``weightspace`` is unchanged.
- The gateway admits the allowance for miServer only and clamps it below the cap it inserts;
  an engine body with thinking passes ``allowlist.rebuild`` unchanged.
- The intake stores ``engine.reasoning_turns`` and still accepts a block without it.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from core.errors import Refusal
from tests.support.wsp_fixtures import make_bundle
from wsp import endpoints, runs
from wsp.freeze import freeze, thinking_in_effect
from wsp.schemas import validate_frozen_request
from wsp.settings import WSPSettings
from wsp_gateway import allowlist, meter
from wsp_gateway.allowlist import FieldRefused
from wsp_worker.miwsp_engine import context
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine.client import GatewayClient
from wsp_worker.miwsp_engine.tools import ForwardBudget, Toolbox, submit_spec

MISERVER = {
    "id": "ep-local",
    "locality": "local",
    "model": "qwen",
    "kind": "openai_compatible",
    "serving": "miserver",
    "thinking": "off",
    "thinking_tokens": None,
}
GENERIC = {**MISERVER, "serving": "generic"}
ENGINE = {"id": "miwsp_engine"}
FIELD = proto.REASONING_ALLOWANCE_FIELD


@pytest.fixture(scope="module")
def bundle(tmp_path_factory: Any) -> dict[str, Any]:
    return make_bundle(tmp_path_factory.mktemp("b"), rule="mean").manifest


def body(bundle: dict[str, Any], **overrides: Any) -> dict[str, Any]:
    out: dict[str, Any] = {
        "bundle": {
            "owner": "miworkbench",
            "bundle_id": bundle["bundle_id"],
            "bundle_sha256": bundle["bundle_sha256"],
        },
        "kind": "repair",
        "mode": "session",
        "task": "reduce false alarms",
        "endpoint_id": "ep-local",
        "budgets": {"tokens": 5000, "time_s": 600, "max_steps": 20, "forward_passes": 0},
        "primary_sets": ["eval"],
        "sensitivity": bundle["sensitivity"]["value"],
        "provider": ENGINE,
    }
    out.update(overrides)
    return out


def frozen(bundle: dict[str, Any], b: dict[str, Any], endpoint: dict[str, Any] = MISERVER) -> Any:
    return freeze(
        b,
        bundle=bundle,
        bundle_imported=True,
        endpoint=endpoint,
        provider_version="0.1.0",
        queued_for_caller=0,
        settings=WSPSettings(),
    )


def refused(fn: Any) -> Refusal:
    with pytest.raises(Refusal) as caught:
        fn()
    return caught.value


# ----------------------------------------------------------------------------- one name each


def test_each_figure_is_stated_once_per_package_and_the_copies_agree() -> None:
    assert allowlist.REASONING_ALLOWANCE_FIELD == proto.REASONING_ALLOWANCE_FIELD
    assert endpoints.DEFAULT_THINKING_TOKENS == context.DEFAULT_THINKING_TOKENS
    assert allowlist.ACTION_FLOOR_TOKENS == meter.MIN_OUTPUT_TOKENS == context.MIN_OUTPUT_TOKENS
    assert proto.REASONING_ALLOWANCE_FIELD in allowlist.OPENAI_MISERVER_TOP


# ----------------------------------------------------------------------------- the endpoint


@pytest.mark.parametrize("value", ["off", "default"])
@pytest.mark.parametrize(
    ("kind", "serving"),
    [("openai_compatible", "miserver"), ("openai_compatible", "generic"), ("anthropic", "generic")],
)
def test_off_and_default_fit_every_endpoint(value: str, kind: str, serving: str) -> None:
    assert endpoints.check_thinking(value, kind=kind, serving=serving) == value


@pytest.mark.parametrize(
    ("kind", "serving"), [("openai_compatible", "generic"), ("anthropic", "generic")]
)
def test_on_is_refused_where_enable_thinking_cannot_reach_the_server(
    kind: str, serving: str
) -> None:
    caught = refused(lambda: endpoints.check_thinking("on", kind=kind, serving=serving))
    assert caught.code == "wsp_endpoint_thinking_unsupported"
    assert endpoints.check_thinking("on", kind="openai_compatible", serving="miserver") == "on"


@pytest.mark.parametrize("value", [None, "high", True, "ON"])
def test_a_bad_thinking_value_is_refused(value: Any) -> None:
    caught = refused(
        lambda: endpoints.check_thinking(value, kind="openai_compatible", serving="miserver")
    )
    assert caught.code == "validation_failed"


@pytest.mark.parametrize("value", [0, -1, True, "4096", 10_000_001])
def test_a_bad_allowance_is_refused(value: Any) -> None:
    assert refused(lambda: endpoints.check_thinking_tokens(value, window=None)).code == (
        "validation_failed"
    )


def test_an_allowance_must_be_below_the_endpoints_window() -> None:
    assert endpoints.check_thinking_tokens(8191, window=8192) == 8191
    assert refused(lambda: endpoints.check_thinking_tokens(8192, window=8192)).code == (
        "validation_failed"
    )


def test_thinking_and_its_allowance_are_patchable() -> None:
    assert {"thinking", "thinking_tokens"} <= set(endpoints.PATCHABLE)


# ----------------------------------------------------------------------------- the freeze


def test_the_endpoints_default_off_is_frozen_for_the_engine(bundle: dict[str, Any]) -> None:
    f = frozen(bundle, body(bundle))
    assert f.request["thinking"] == "off" and "thinking_tokens" not in f.request
    validate_frozen_request(f.request)


def test_the_endpoints_on_is_frozen_with_the_default_allowance(bundle: dict[str, Any]) -> None:
    f = frozen(bundle, body(bundle), {**MISERVER, "thinking": "on"})
    assert (f.request["thinking"], f.request["thinking_tokens"]) == (
        "on",
        endpoints.DEFAULT_THINKING_TOKENS,
    )
    validate_frozen_request(f.request)
    own = frozen(bundle, body(bundle), {**MISERVER, "thinking": "on", "thinking_tokens": 2048})
    assert own.request["thinking_tokens"] == 2048


def test_the_run_overrides_the_endpoint_both_ways(bundle: dict[str, Any]) -> None:
    on = frozen(bundle, body(bundle, thinking="on", thinking_tokens=6000))
    assert (on.request["thinking"], on.request["thinking_tokens"]) == ("on", 6000)
    off = frozen(bundle, body(bundle, thinking="off"), {**MISERVER, "thinking": "on"})
    assert off.request["thinking"] == "off" and "thinking_tokens" not in off.request
    default = frozen(bundle, body(bundle, thinking="default"))
    assert default.request["thinking"] is None


def test_the_thinking_in_effect_moves_the_digest(bundle: dict[str, Any]) -> None:
    """The freeze is load-bearing: two runs that differ only in thinking (or only in the
    allowance) are two requests, so a replay and the approval see which one ran."""
    digests = {
        frozen(bundle, body(bundle, **over)).request_sha256
        for over in (
            {},
            {"thinking": "on"},
            {"thinking": "on", "thinking_tokens": 1024},
            {"thinking": "default"},
        )
    }
    assert len(digests) == 4
    endpoint_on = frozen(bundle, body(bundle), {**MISERVER, "thinking": "on"})
    assert endpoint_on.request_sha256 == frozen(bundle, body(bundle, thinking="on")).request_sha256


def test_on_for_an_endpoint_that_cannot_carry_it_is_refused(bundle: dict[str, Any]) -> None:
    caught = refused(lambda: frozen(bundle, body(bundle, thinking="on"), GENERIC))
    assert caught.code == "wsp_endpoint_thinking_unsupported"
    assert frozen(bundle, body(bundle), GENERIC).request["thinking"] == "off"


@pytest.mark.parametrize(
    ("over", "endpoint"),
    [
        ({"thinking_tokens": 2048}, MISERVER),  # thinking in effect is off
        ({"thinking": "off", "thinking_tokens": 2048}, {**MISERVER, "thinking": "on"}),
        ({"thinking": "high"}, MISERVER),  # an effort level is weightspace's, not the engine's
        ({"thinking": "on", "thinking_tokens": 0}, MISERVER),
    ],
)
def test_an_allowance_without_thinking_on_or_a_foreign_value_is_refused(
    bundle: dict[str, Any], over: dict[str, Any], endpoint: dict[str, Any]
) -> None:
    assert refused(lambda: frozen(bundle, body(bundle, **over), endpoint)).code == (
        "validation_failed"
    )


def test_weightspace_freezes_the_body_as_before(bundle: dict[str, Any]) -> None:
    ws = {"id": "weightspace"}
    assert frozen(bundle, body(bundle, provider=ws)).request["thinking"] is None
    on_endpoint = frozen(bundle, body(bundle, provider=ws), {**MISERVER, "thinking": "on"})
    assert on_endpoint.request["thinking"] is None  # the endpoint's setting is the engine's
    assert frozen(bundle, body(bundle, provider=ws, thinking="high")).request["thinking"] == "high"
    for over in ({"thinking": "on"}, {"thinking_tokens": 100}):
        caught = refused(lambda o=over: frozen(bundle, body(bundle, provider=ws, **o)))
        assert caught.code == "validation_failed"


def test_thinking_in_effect_needs_no_database() -> None:
    assert thinking_in_effect({}, MISERVER, "miwsp_engine") == ("off", None)
    assert thinking_in_effect({"thinking": "on"}, MISERVER, "miwsp_engine") == ("on", 4096)


# ----------------------------------------------------------------------------- the gateway


SPECS = Toolbox([submit_spec(lambda answer, confidence=None: "ok")], ForwardBudget(0)).specs


def _engine_body(protocol: str, thinking: str | None, allowance: int) -> dict[str, Any]:
    fields = proto.request_fields(
        protocol, "openai_compatible", "miserver", SPECS, thinking=thinking, allowance=allowance
    )
    client = GatewayClient("openai_compatible", "m", {"ip": "127.0.0.1", "port": 9, "token": "t"})
    try:
        return client.body(
            [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}],
            fields=fields,
            max_tokens=1024 + allowance,
        )
    finally:
        client.close()


@pytest.mark.parametrize("protocol", [proto.TOOL_CALLS, proto.STRUCTURED])
@pytest.mark.parametrize(("thinking", "allowance"), [("on", 4096), ("off", 0), (None, 0)])
def test_the_engines_thinking_body_passes_the_gateway_unchanged(
    protocol: str, thinking: str | None, allowance: int
) -> None:
    sent = _engine_body(protocol, thinking, allowance)
    rebuilt, _ = allowlist.rebuild("openai_compatible", "miserver", copy.deepcopy(sent), {})
    assert rebuilt == sent
    assert ("chat_template_kwargs" in sent) == (thinking is not None)
    # miServer honours the allowance beside response_format only, so it goes in structured mode.
    assert (FIELD in sent) == (thinking == "on" and protocol == proto.STRUCTURED)
    assert proto.REASONING_ALLOWANCE_FIELD == "max_reasoning_tokens"  # miServer's name


def test_the_allowance_is_refused_for_a_generic_endpoint_and_when_malformed() -> None:
    sent = _engine_body(proto.STRUCTURED, "on", 4096)
    assert FIELD in sent
    with pytest.raises(FieldRefused):
        allowlist.rebuild("openai_compatible", "generic", copy.deepcopy(sent), {})
    for bad in (0, -1, "4096", True, 1.5):
        with pytest.raises(FieldRefused):
            allowlist.rebuild("openai_compatible", "miserver", {**sent, FIELD: bad}, {})


@pytest.mark.parametrize(
    ("cap", "allowance", "kept"),
    [(6144, 4096, 4096), (2048, 4096, 2048 - 256), (256, 4096, 1), (4352, 4096, 4096)],
)
def test_the_inserted_cap_keeps_room_for_the_action(cap: int, allowance: int, kept: int) -> None:
    """When the budget lowers the cap, the reasoning gives way, never the action."""
    out = allowlist.insert_cap(
        "openai_compatible", {FIELD: allowance}, "max_completion_tokens", cap
    )
    assert out["max_completion_tokens"] == cap and out[FIELD] == kept


# ----------------------------------------------------------------------------- the intake


BLOCK = {"action_protocol": "structured", "handoffs": 0, "refused_turns": 1, "kernel_restarts": 0}


def test_the_intake_stores_reasoning_turns_and_accepts_a_block_without_it() -> None:
    assert runs.result_counters({"engine": {**BLOCK, "reasoning_turns": 3}})["reasoning_turns"] == 3
    assert runs.result_counters({"engine": BLOCK})["reasoning_turns"] is None
    both = runs.result_counters({"engine": {**BLOCK, "reasoning_turns": 1, "repeated_turns": 4}})
    assert (both["reasoning_turns"], both["repeated_turns"]) == (1, 4)
    assert runs.result_counters({})["reasoning_turns"] is None


@pytest.mark.parametrize(
    "engine",
    [{**BLOCK, "reasoning_turns": -1}, {**BLOCK, "reasoning_turns": "2"}, {**BLOCK, "other": 1}],
)
def test_a_bad_or_unknown_engine_key_is_refused(engine: dict[str, Any]) -> None:
    assert refused(lambda: runs.result_counters({"engine": engine})).code == "wsp_result_invalid"
