"""The endpoint fingerprint across feature 018 (F18-D8; 018 FTASKS 8.4).

The two values below were computed by the PRE-018 ``wsp.runs.endpoint_fingerprint`` (miForge
``ec98f3e``) for these exact rows and are recorded here as literals, so an endpoint at its
defaults must hash byte for byte as it did before 018: a queued run or a pending approval filed
before the deploy keeps its fingerprint. Away from a default, the fingerprint must move, so the
operator's PATCH fails a queued run ``wsp_endpoint_changed`` (the push half is in
``tests/integration/wsp/test_dispatch.py``).
"""

from __future__ import annotations

from decimal import Decimal
from typing import Any

import pytest

from wsp.models import WspEndpoint
from wsp.runs import endpoint_fingerprint

PRE_018 = {
    "ep-local": "06bd152bf934733405c22a28012c40efd2235f56b499aad0d71d64ff6c61cff8",
    "ep-ext": "010dbf669c49c49796c0975976c79034e50626d7d63311e36360d2f70d9f3566",
}


def _row(endpoint_id: str, **extra: Any) -> WspEndpoint:
    if endpoint_id == "ep-local":
        return WspEndpoint(
            id="ep-local",
            kind="openai_compatible",
            base_url="http://miserver-backend.miserver.svc:8000/v1",
            model="qwen",
            locality="local",
            output_cap_field="max_tokens",
            serving="miserver",
            **extra,
        )
    return WspEndpoint(
        id="ep-ext",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="claude",
        locality="external",
        output_cap_field="max_tokens",
        serving="generic",
        price_in_per_mtok=Decimal("3.000000"),
        price_out_per_mtok=Decimal("15.000000"),
        key_ciphertext=b"\x01" * 80,
        **extra,
    )


@pytest.mark.parametrize("endpoint_id", sorted(PRE_018))
@pytest.mark.parametrize(
    "defaults",
    [
        {},
        {"action_protocol": "tool_calls"},
        {"action_protocol": "tool_calls", "context_window_tokens": None},
        # 2026-11-21: thinking off (the default) and no allowance hash as before too.
        {"thinking": "off"},
        {"thinking": "off", "thinking_tokens": None},
        {"output_cap": "on"},
        # 2026-11-21 (migration 0034): paging on (the default) hashes as before.
        {"paging": "on"},
    ],
    ids=[
        "unset",
        "tool_calls",
        "tool_calls-no-window",
        "thinking-off",
        "thinking-off-no-tokens",
        "output-cap-on",
        "paging-on",
    ],
)
def test_the_endpoint_fingerprint_is_stable_at_defaults(
    endpoint_id: str, defaults: dict[str, Any]
) -> None:
    assert endpoint_fingerprint(_row(endpoint_id, **defaults)) == PRE_018[endpoint_id]


@pytest.mark.parametrize(
    "change",
    [
        {"action_protocol": "structured"},
        {"context_window_tokens": 32768},
        {"action_protocol": "structured", "context_window_tokens": 32768},
        {"thinking": "on"},
        {"thinking": "default"},
        {"thinking_tokens": 4096},
        {"output_cap": "off"},
        {"paging": "off"},
    ],
    ids=[
        "protocol",
        "window",
        "both",
        "thinking-on",
        "thinking-default",
        "thinking-tokens",
        "output-cap-off",
        "paging-off",
    ],
)
def test_a_value_away_from_its_default_moves_the_fingerprint(change: dict[str, Any]) -> None:
    moved = endpoint_fingerprint(_row("ep-local", **change))
    assert moved != PRE_018["ep-local"]


def test_each_window_value_has_its_own_fingerprint() -> None:
    a = endpoint_fingerprint(_row("ep-local", context_window_tokens=16384))
    b = endpoint_fingerprint(_row("ep-local", context_window_tokens=32768))
    protocol = endpoint_fingerprint(_row("ep-local", action_protocol="structured"))
    assert len({a, b, protocol, PRE_018["ep-local"]}) == 4


def test_each_thinking_value_and_allowance_has_its_own_fingerprint() -> None:
    """2026-11-21: the operator's PATCH of either thinking field fails a queued run at push."""
    prints = {
        endpoint_fingerprint(_row("ep-local", **change))
        for change in (
            {},
            {"thinking": "on"},
            {"thinking": "default"},
            {"thinking": "on", "thinking_tokens": 2048},
            {"thinking": "on", "thinking_tokens": 8192},
        )
    }
    assert len(prints) == 5
