"""``core.secret_keys`` (operator 2026-11-26): the one list of secret-named keys, and the
redaction approval answers apply with it.

- the endpoint prices are this list's (``wsp.endpoints.PRICE_FIELDS`` IS ``PRICE_KEYS``);
- the MCP endpoint list's withheld keys (a package that cannot import the backend) are all in it;
- a secret-named key's value is replaced at any depth, plural counts and ``has_*`` flags stay,
  and the input is never changed.
"""

from __future__ import annotations

import copy

import pytest

from core import secret_keys
from core.secret_keys import REDACTED, redacted


def test_the_endpoint_prices_are_this_list_not_a_copy() -> None:
    from wsp.endpoints import PRICE_FIELDS

    assert PRICE_FIELDS is secret_keys.PRICE_KEYS
    assert set(PRICE_FIELDS) <= secret_keys.SECRET_KEYS


def test_the_mcp_withheld_keys_are_all_in_this_list() -> None:
    from mcp_server.tools import wsp as wsp_tools

    assert set(wsp_tools.WITHHELD) <= secret_keys.SECRET_KEYS
    assert {"api_key", "key_fingerprint"} <= secret_keys.SECRET_KEYS


def test_the_mcp_run_read_withheld_keys_are_all_in_this_list() -> None:
    """Train F3 (2026-11-26): the WSP run reads' withheld keys (``RUN_READ_WITHHELD``, the 019 P4
    MCP gaps) are pinned to this list the same way: a key withheld there that this list does not
    name means the two lists have drifted."""
    from mcp_server.tools import wsp as wsp_tools

    assert wsp_tools.RUN_READ_WITHHELD
    assert set(wsp_tools.RUN_READ_WITHHELD) <= secret_keys.SECRET_KEYS


@pytest.mark.parametrize(
    "key",
    [
        "api_key",
        "API_KEY",
        "key_fingerprint",
        "token",
        "hub_token",
        "millm_trainer_token",
        "client_secret",
        "password",
        "Authorization",
        "price_out_per_mtok",
    ],
)
def test_a_secret_named_key_is_redacted(key: str) -> None:
    assert redacted({"a": [{"b": {key: "x"}}]}) == {"a": [{"b": {key: REDACTED}}]}


@pytest.mark.parametrize(
    "key", ["tokens", "thinking_tokens", "max_new_tokens", "has_key", "has_api_key", "model"]
)
def test_a_count_a_flag_or_a_plain_key_stays(key: str) -> None:
    assert redacted({"a": [{key: 7}]}) == {"a": [{key: 7}]}


def test_the_input_is_never_changed_and_other_values_pass_through() -> None:
    stored = {"body": {"api_key": "k", "n": [1, {"token": "t"}]}, "x": None}
    before = copy.deepcopy(stored)
    assert redacted(stored) == {
        "body": {"api_key": REDACTED, "n": [1, {"token": REDACTED}]},
        "x": None,
    }
    assert stored == before
    assert redacted(None) is None and redacted("s") == "s" and redacted(3) == 3


def test_the_approval_objects_words_render_no_secret_named_key() -> None:
    """``result_language.summary_words`` renders only ``_SUMMARY_KEYS`` from a stored summary,
    so a secret there cannot reach the ``approval`` object's words."""
    from services import result_language

    assert result_language._SUMMARY_KEYS
    assert not [k for k in result_language._SUMMARY_KEYS if secret_keys.is_secret_key(k)]
    words = result_language.summary_words(
        "config_write", {"route": "PUT /x", "api_key": "PLANTED", "model": "m"}
    )
    assert "PLANTED" not in words and "PUT /x" in words
