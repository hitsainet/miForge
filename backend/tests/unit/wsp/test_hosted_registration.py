"""H-2: registering and changing a hosted endpoint (feature 020; FR-020.1 to FR-020.5; TDD §5.1).

The expectations are transcribed from TDD §5.1 here, independently of ``wsp.endpoints``:

- **history mode** (FR-020.1, W3-D3): absent, ``append_only`` for a hosted endpoint and
  ``paging`` for a local one; ``paging`` is refused for a hosted Anthropic endpoint only;
- **thinking** (FR-020.4, F20-D6): hosted Anthropic ``default`` (its default), ``low``,
  ``medium``, ``high``; hosted OpenAI-compatible ``off``, ``default`` and the three efforts;
  local unchanged (``off``, ``default``, and ``on`` for miServer's OpenAI kind); ``xhigh`` and
  ``max`` refused for every kind;
- **the window** (FR-020.2) and **the cache prices** (FR-020.3) of a hosted endpoint;
- the operator's ``PATCH``, including clearing a hosted window, and ``runs_refused`` with both
  messages (F20-D12).
"""

from __future__ import annotations

import datetime as dt
from decimal import Decimal
from typing import Any

import pytest

from core.errors import Refusal
from wsp import endpoints
from wsp.models import WspEndpoint
from wsp.settings import WSPSettings

SETTINGS = WSPSettings(
    wsp_local_host_allowlist="127.0.0.1,miserver-backend.miserver.svc",
    wsp_external_host_allowlist="api.anthropic.com,llm.example.com",
)


class _NoRows:
    """``validate_registration`` asks only whether the id is taken."""

    def get(self, _model: Any, _key: Any) -> None:
        return None


PLACES = {
    # (kind, place): the body's endpoint part.
    ("anthropic", "hosted"): {"kind": "anthropic", "base_url": "https://api.anthropic.com"},
    ("openai_compatible", "hosted"): {
        "kind": "openai_compatible",
        "base_url": "https://llm.example.com/v1",
    },
    ("anthropic", "local"): {"kind": "anthropic", "base_url": "http://127.0.0.1:8000"},
    ("openai_compatible", "local"): {
        "kind": "openai_compatible",
        "base_url": "http://127.0.0.1:8000/v1",
    },
    ("openai_compatible", "miserver"): {
        "kind": "openai_compatible",
        "base_url": "http://miserver-backend.miserver.svc:8000/v1",
        "serving": "miserver",
    },
}
HOSTED_PRICES = {
    "price_in_per_mtok": "3",
    "price_out_per_mtok": "15",
    "price_cache_write_per_mtok": "3.75",
    "price_cache_read_per_mtok": "0.3",
}


def body(kind: str, place: str, **extra: Any) -> dict[str, Any]:
    out: dict[str, Any] = {"id": "ep-x", "model": "m", **PLACES[(kind, place)]}
    if place == "hosted":
        out.update(api_key="sk-test", context_window_tokens=200_000, **HOSTED_PRICES)
    out.update(extra)
    return {k: v for k, v in out.items() if v is not _ABSENT}


_ABSENT = object()


def register(kind: str, place: str, **extra: Any) -> dict[str, Any]:
    fields, _key = endpoints.validate_registration(
        _NoRows(),  # type: ignore[arg-type]
        body(kind, place, **extra),
        settings=SETTINGS,
    )
    return fields


def refused(kind: str, place: str, **extra: Any) -> str:
    with pytest.raises(Refusal) as caught:
        register(kind, place, **extra)
    return caught.value.code


# ------------------------------------------------- the expected rules, transcribed from TDD §5.1


def expected_thinking(kind: str, place: str, value: str) -> str | None:
    """``None`` (accepted) or the refusal code."""
    if value in ("xhigh", "max", "maybe"):
        return "validation_failed"
    if place == "hosted" and kind == "anthropic":
        accepted = {"default", "low", "medium", "high"}
    elif place == "hosted":
        accepted = {"off", "default", "low", "medium", "high"}
    elif place == "miserver":
        accepted = {"off", "default", "on"}
    else:
        accepted = {"off", "default"}
    return None if value in accepted else "wsp_endpoint_thinking_unsupported"


def expected_history(kind: str, place: str, value: str | None) -> tuple[str | None, str | None]:
    """``(stored value, refusal code)``."""
    if value is None:
        return ("append_only" if place == "hosted" else "paging"), None
    if value == "paging" and kind == "anthropic" and place == "hosted":
        return None, "wsp_endpoint_history_mode_refused"
    return value, None


CASES = [
    (kind, place, history, thinking)
    for (kind, place) in PLACES
    for history in (None, "paging", "append_only")
    for thinking in ("off", "on", "default", "low", "medium", "high", "xhigh", "max")
]


@pytest.mark.parametrize(("kind", "place", "history", "thinking"), CASES)
def test_every_kind_place_history_and_thinking_is_accepted_or_refused_by_the_table(
    kind: str, place: str, history: str | None, thinking: str
) -> None:
    extra: dict[str, Any] = {"thinking": thinking}
    if history is not None:
        extra["history_mode"] = history
    thinking_code = expected_thinking(kind, place, thinking)
    stored_history, history_code = expected_history(kind, place, history)
    want = thinking_code or history_code  # thinking is checked first, then the history mode
    if want is not None:
        assert refused(kind, place, **extra) == want
        return
    fields = register(kind, place, **extra)
    assert fields["thinking"] == thinking
    assert fields["history_mode"] == stored_history


@pytest.mark.parametrize(
    ("kind", "place", "default"),
    [
        ("anthropic", "hosted", "default"),
        ("openai_compatible", "hosted", "off"),
        ("openai_compatible", "local", "off"),
        ("anthropic", "local", "off"),
    ],
)
def test_the_thinking_default_is_the_kinds(kind: str, place: str, default: str) -> None:
    assert register(kind, place)["thinking"] == default


def test_a_refused_thinking_value_names_what_the_endpoint_accepts() -> None:
    with pytest.raises(Refusal) as caught:
        register("anthropic", "hosted", thinking="off")
    assert caught.value.code == "wsp_endpoint_thinking_unsupported"
    assert "default, low, medium, high" in caught.value.message


@pytest.mark.parametrize("kind", ["anthropic", "openai_compatible"])
def test_thinking_tokens_is_refused_for_a_hosted_endpoint(kind: str) -> None:
    assert refused(kind, "hosted", thinking_tokens=2048) == "wsp_endpoint_thinking_unsupported"


# ----------------------------------------------------------------- the window and cache prices


@pytest.mark.parametrize("kind", ["anthropic", "openai_compatible"])
def test_a_hosted_registration_without_a_window_is_refused(kind: str) -> None:
    assert refused(kind, "hosted", context_window_tokens=_ABSENT) == "endpoint_window_required"
    assert register(kind, "hosted")["context_window_tokens"] == 200_000


def test_a_local_registration_needs_no_window() -> None:
    assert register("openai_compatible", "local")["context_window_tokens"] is None


@pytest.mark.parametrize("missing", ["price_cache_write_per_mtok", "price_cache_read_per_mtok"])
def test_a_hosted_anthropic_registration_needs_both_cache_prices(missing: str) -> None:
    assert refused("anthropic", "hosted", **{missing: _ABSENT}) == (
        "wsp_endpoint_cache_prices_required"
    )


def test_a_hosted_openai_registration_keeps_cache_prices_optional() -> None:
    fields = register(
        "openai_compatible",
        "hosted",
        price_cache_write_per_mtok=_ABSENT,
        price_cache_read_per_mtok=_ABSENT,
    )
    assert fields["price_cache_write_per_mtok"] is None


# ----------------------------------------------------------------------- the operator's PATCH


class _OneRow:
    def __init__(self, row: WspEndpoint) -> None:
        self.row = row

    def get(self, _model: Any, _key: Any) -> WspEndpoint:
        return self.row

    def flush(self) -> None:
        return None


def row(kind: str, place: str, **extra: Any) -> WspEndpoint:
    fields = register(kind, place)
    fields.update(extra)
    return WspEndpoint(
        **fields, created_at=dt.datetime(2026, 11, 25, tzinfo=dt.UTC), created_by="o"
    )


def patch(target: WspEndpoint, change: dict[str, Any]) -> WspEndpoint:
    return endpoints.update_endpoint(_OneRow(target), target.id, change)  # type: ignore[arg-type]


def patch_refused(target: WspEndpoint, change: dict[str, Any]) -> str:
    with pytest.raises(Refusal) as caught:
        patch(target, change)
    return caught.value.code


def test_a_patch_cannot_clear_a_hosted_window() -> None:
    hosted = row("anthropic", "hosted")
    assert patch_refused(hosted, {"context_window_tokens": None}) == "endpoint_window_required"
    assert hosted.context_window_tokens == 200_000
    assert patch(hosted, {"context_window_tokens": 100_000}).context_window_tokens == 100_000


def test_a_patch_may_clear_a_local_window() -> None:
    local = row("openai_compatible", "local", context_window_tokens=32_768)
    assert patch(local, {"context_window_tokens": None}).context_window_tokens is None


def test_a_patch_of_a_hosted_anthropic_endpoint_to_paging_is_refused() -> None:
    hosted = row("anthropic", "hosted")
    assert patch_refused(hosted, {"history_mode": "paging"}) == "wsp_endpoint_history_mode_refused"
    assert hosted.history_mode == "append_only"


def test_a_patch_of_a_local_endpoint_to_append_only_is_stored() -> None:
    local = row("openai_compatible", "local")
    assert patch(local, {"history_mode": "append_only"}).history_mode == "append_only"
    assert patch_refused(local, {"history_mode": "rewrite"}) == "validation_failed"


def test_a_patch_rechecks_thinking_with_the_rows_locality() -> None:
    hosted = row("anthropic", "hosted")
    assert patch_refused(hosted, {"thinking": "off"}) == "wsp_endpoint_thinking_unsupported"
    assert patch(hosted, {"thinking": "high"}).thinking == "high"
    local = row("openai_compatible", "local")
    assert patch_refused(local, {"thinking": "medium"}) == "wsp_endpoint_thinking_unsupported"


def test_a_row_stored_local_but_derived_external_is_treated_as_hosted() -> None:
    """The stored or the derived locality, whichever says external (``bound_fields``)."""
    mislabelled = row("anthropic", "hosted", locality="local")
    assert patch_refused(mislabelled, {"context_window_tokens": None}) == "endpoint_window_required"


# ----------------------------------------------------------------------- runs_refused (TD-2)


def test_runs_refused_names_the_window_then_the_cache_prices() -> None:
    old = row("anthropic", "hosted", context_window_tokens=None, price_cache_read_per_mtok=None)
    assert endpoints.run_refusal(old) == (
        "endpoint_window_required",
        "Set the endpoint's context window to run it: a hosted endpoint must state its window.",
    )
    assert endpoints.runs_refused(old) == {
        "code": "endpoint_window_required",
        "message": endpoints.WINDOW_REQUIRED_MESSAGE,
    }
    old.context_window_tokens = 200_000
    assert endpoints.runs_refused(old) == {
        "code": "wsp_endpoint_cache_prices_required",
        "message": (
            "This endpoint has no cache prices, which are sealed with its key: register it again "
            "with both cache prices to run it."
        ),
    }
    old.price_cache_read_per_mtok = Decimal("0.3")
    assert endpoints.runs_refused(old) is None


def test_runs_refused_is_null_for_a_local_row_and_a_complete_hosted_one() -> None:
    assert endpoints.run_refusal(row("openai_compatible", "local")) is None
    assert endpoints.run_refusal(row("anthropic", "hosted")) is None
    assert (
        endpoints.run_refusal(row("openai_compatible", "hosted", price_cache_read_per_mtok=None))
        is None
    )


def test_the_list_serves_history_mode_and_runs_refused() -> None:
    old = row("anthropic", "hosted", context_window_tokens=None)
    doc = endpoints.public_document(old)
    assert doc["history_mode"] == "append_only"
    assert doc["runs_refused"] == {
        "code": "endpoint_window_required",
        "message": endpoints.WINDOW_REQUIRED_MESSAGE,
    }
    assert endpoints.public_document(row("openai_compatible", "local"))["runs_refused"] is None
