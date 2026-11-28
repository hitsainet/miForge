"""The endpoint fingerprint is byte-stable across 019 (W-5; FR-019.14; F19-D24 reading 4).

The output ceiling joins the fingerprint only when it is set (F18-D8's rule for every patchable
field), so an endpoint at its defaults hashes exactly as it did at ``32b6fa8``: no queued run or
pending approval fails ``wsp_endpoint_changed`` at the deploy. The hex values were computed with
``wsp.runs.endpoint_fingerprint`` at ``32b6fa8`` (``0xcc/reviews/019_window_readers.md`` §6). A
PATCH of the ceiling failing a queued run at push is ``integration/wsp/test_endpoints_api.py``.
"""

from __future__ import annotations

from decimal import Decimal

from wsp.models import WspEndpoint
from wsp.runs import endpoint_fingerprint

GENERIC_AT_32B6FA8 = "dde62794faa15174d825d2735fbe1892069b8d33e7f1c251b66340f9ed2e0b9d"
EXTERNAL_AT_32B6FA8 = "9064cb4be1aa5ba5ee5c2b1ae40d9fc1ed61d3cad4c83b57f34c5451ee6bad13"


def _generic(**extra: object) -> WspEndpoint:
    return WspEndpoint(
        id="ep-default",
        kind="openai_compatible",
        base_url="http://127.0.0.1:8000/v1",
        model="qwen",
        locality="local",
        output_cap_field="max_completion_tokens",
        serving="generic",
        action_protocol="tool_calls",
        thinking="off",
        output_cap="on",
        paging="on",
        **extra,
    )


def test_a_default_endpoint_hashes_as_it_did_at_32b6fa8() -> None:
    assert endpoint_fingerprint(_generic()) == GENERIC_AT_32B6FA8
    assert endpoint_fingerprint(_generic(output_ceiling_tokens=None)) == GENERIC_AT_32B6FA8
    external = WspEndpoint(
        id="ep-ext",
        kind="anthropic",
        base_url="https://api.anthropic.com",
        model="claude-x",
        locality="external",
        output_cap_field="max_tokens",
        serving="generic",
        price_in_per_mtok=Decimal("3"),
        price_out_per_mtok=Decimal("15"),
        key_ciphertext=b"abc",
        action_protocol="tool_calls",
        thinking="off",
        output_cap="on",
        paging="on",
    )
    assert endpoint_fingerprint(external) == EXTERNAL_AT_32B6FA8


def test_a_set_ceiling_moves_the_fingerprint_and_each_value_is_its_own() -> None:
    at_4096 = endpoint_fingerprint(_generic(output_ceiling_tokens=4_096))
    at_8192 = endpoint_fingerprint(_generic(output_ceiling_tokens=8_192))
    assert GENERIC_AT_32B6FA8 not in (at_4096, at_8192) and at_4096 != at_8192


def test_the_registration_read_is_not_in_the_fingerprint() -> None:
    """A read is a fact about the server, not a setting: it never fails a queued run."""
    import datetime as dt

    read = _generic(
        registration_window_tokens=32_768,
        registration_window_read_at=dt.datetime(2026, 11, 24, tzinfo=dt.UTC),
    )
    assert endpoint_fingerprint(read) == GENERIC_AT_32B6FA8
