"""019 P3 (D-1; FR-019.6, .7, .8; F19-D6, D17, D18, D31): the scheduling key, the effective pool and
the worker addresses, over every URL variant the key must fold together or keep apart."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from wsp import scheduling
from wsp.settings import WSPSettings, not_configured_message


def _endpoint(endpoint_id: str, base_url: str, serving: str = "miserver") -> SimpleNamespace:
    return SimpleNamespace(id=endpoint_id, base_url=base_url, serving=serving)


@pytest.mark.parametrize(
    "variant",
    [
        "http://miserver-backend.miserver.svc:8000",
        "http://miserver-backend.miserver.svc:8000/",
        "http://miserver-backend.miserver.svc:8000/v1",
        "http://MiServer-Backend.miserver.svc.:8000/v1",
        "HTTP://miserver-backend.miserver.svc:8000/v1/",
    ],
)
def test_two_miserver_rows_on_one_origin_share_one_key(variant: str) -> None:
    first = _endpoint("ms-a", "http://miserver-backend.miserver.svc:8000/v1")
    second = _endpoint("ms-b", variant)
    assert scheduling.key_for(first) == scheduling.key_for(second)
    assert scheduling.key_for(first) == "miserver:http://miserver-backend.miserver.svc:8000"


@pytest.mark.parametrize(
    ("explicit", "implicit"),
    [("http://llm.lan:80/v1", "http://llm.lan/v1"), ("https://llm.lan:443", "https://llm.lan/")],
)
def test_a_default_port_is_the_same_key_written_or_not(explicit: str, implicit: str) -> None:
    assert scheduling.key_for(_endpoint("a", explicit)) == scheduling.key_for(
        _endpoint("b", implicit)
    )


def test_different_miserver_hosts_or_ports_or_schemes_are_different_keys() -> None:
    keys = {
        scheduling.key_for(_endpoint(str(i), url))
        for i, url in enumerate(
            (
                "http://a.lan:8000/v1",
                "http://b.lan:8000/v1",
                "http://a.lan:8001/v1",
                "https://a.lan:8000/v1",
            )
        )
    }
    assert len(keys) == 4


def test_generic_rows_are_their_own_key_even_on_one_base_url() -> None:
    url = "http://vllm.lan:8000/v1"
    first = _endpoint("gen-a", url, serving="generic")
    second = _endpoint("gen-b", url, serving="generic")
    assert scheduling.key_for(first) == "endpoint:gen-a"
    assert scheduling.key_for(second) == "endpoint:gen-b"


def test_an_ipv6_origin_keeps_its_brackets() -> None:
    assert scheduling.origin_of("http://[fd00::1]:8000/v1") == "http://[fd00::1]:8000"


@pytest.mark.parametrize(
    ("size", "maximum", "effective"),
    [(1, 1, 1), (2, 3, 2), (3, 3, 3), (5, 2, 2), (0, 4, 1), (4, 0, 1)],
)
def test_the_effective_pool_is_the_setting_within_the_maximum_and_at_least_one(
    size: int, maximum: int, effective: int
) -> None:
    assert scheduling.effective_pool(size, maximum) == effective


def test_a_template_names_every_worker_by_index() -> None:
    settings = WSPSettings(wsp_worker_url="http://w-{index}.pool:9000", wsp_worker_pool_max=3)
    assert scheduling.worker_urls(settings) == [
        "http://w-0.pool:9000",
        "http://w-1.pool:9000",
        "http://w-2.pool:9000",
    ]


def test_a_plain_url_is_one_worker_at_a_maximum_of_one() -> None:
    settings = WSPSettings(wsp_worker_url="http://w.pool:9000", wsp_worker_pool_max=1)
    assert scheduling.worker_urls(settings) == ["http://w.pool:9000"]


def test_a_plain_url_above_one_worker_is_not_configured_and_says_the_rule() -> None:
    """14.2: a maximum of 2 with no ``{index}`` refuses ``wsp_not_configured``, naming the rule."""
    settings = WSPSettings(
        wsp_worker_url="http://w.pool:9000",
        wsp_gateway_url="http://g:8080",
        wsp_worker_pool_max=2,
    )
    assert scheduling.worker_urls(settings) == []
    missing = settings.missing_urls()
    assert missing == ["WSP_WORKER_URL"] and not settings.configured()
    message = not_configured_message(missing, settings)
    assert "{index}" in message and "WSP_WORKER_POOL_MAX" in message


def test_no_url_is_no_worker() -> None:
    assert scheduling.worker_urls(WSPSettings(wsp_worker_url="", wsp_worker_pool_max=2)) == []
