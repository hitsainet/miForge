"""FR-016.16: locality is derived from the host (PF-16), and the base-URL rules (P61)."""

from __future__ import annotations

import pytest

from core.errors import Refusal
from wsp.endpoints import derive_locality, validate_base_url

LOCAL_ALLOW = "miserver-backend.miserver.svc"
EXTERNAL_ALLOW = "api.anthropic.com"


@pytest.mark.parametrize(
    ("url", "locality"),
    [
        ("http://miserver-backend.miserver.svc:8000", "local"),
        ("http://miserver-backend.miserver.svc.cluster.local:8000/v1", "local"),
        ("http://10.43.2.7:8000", "local"),
        ("http://192.168.244.52:80", "local"),
        ("http://172.16.0.1", "local"),
        ("http://127.0.0.1:8000", "local"),
        ("http://localhost:8000", "local"),
        ("http://[fd00::1]:8000", "local"),
        ("https://api.anthropic.com", "external"),
        ("https://api.openai.com/v1", "external"),
        ("https://8.8.8.8", "external"),
        ("https://miserver-backend.miserver.svc.evil.example", "external"),
    ],
)
def test_the_derivation_table(url: str, locality: str) -> None:
    assert derive_locality(url) == locality


def _refused(url: str) -> Refusal:
    with pytest.raises(Refusal) as err:
        validate_base_url(url, local_allowlist=LOCAL_ALLOW, external_allowlist=EXTERNAL_ALLOW)
    return err.value


def test_the_two_allowed_endpoints_pass() -> None:
    assert (
        validate_base_url(
            "http://miserver-backend.miserver.svc:8000/v1",
            local_allowlist=LOCAL_ALLOW,
            external_allowlist=EXTERNAL_ALLOW,
        )
        == "local"
    )
    assert (
        validate_base_url(
            "https://api.anthropic.com",
            local_allowlist=LOCAL_ALLOW,
            external_allowlist=EXTERNAL_ALLOW,
        )
        == "external"
    )


@pytest.mark.parametrize(
    "url",
    [
        "https://api.anthropic.com/api/v1/admin",
        "https://api.anthropic.com/v1?x=1",
        "https://api.anthropic.com/v1#frag",
        "http://api.anthropic.com",
        "https://api.evil.example",
        "http://10.0.0.5:8000",
        "ftp://api.anthropic.com",
    ],
)
def test_refused_urls(url: str) -> None:
    assert _refused(url).code == "wsp_endpoint_url_refused"


def test_a_credential_url_is_refused_without_echoing_it() -> None:
    err = _refused("https://user:s3cr3t-sentinel@api.anthropic.com")
    assert "s3cr3t-sentinel" not in err.message and "s3cr3t-sentinel" not in repr(err.details)
