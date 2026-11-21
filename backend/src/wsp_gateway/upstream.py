"""The upstream side: endpoint checks at connect time, headers from scratch, one call (FTDD §5.5).

**The endpoint is re-checked here** (P50, P61, R2A-10), not trusted from registration: the base
URL's path is empty or ``/v1`` with no query; an external host must be in
``WSP_EXTERNAL_HOST_ALLOWLIST`` and use https; a local host must be in
``WSP_LOCAL_HOST_ALLOWLIST`` and RESOLVE to cluster or RFC 1918 addresses, and the call then goes to
that resolved address (the ``Host`` header carries the name), so a DNS answer cannot change between
the check and the connect. Locality is ``external`` when EITHER the stored row or the host says so.

**Headers are built from scratch:** ``content-type``, the endpoint's credential header, a fixed
``anthropic-version`` and, for miServer, ``X-miLLM-Origin: judge`` and
``X-miLLM-Load-Policy: refuse`` (P28; frozen identifiers). Nothing incoming is forwarded, so
``X-miLLM-Strict``, ``anthropic-beta`` and the session token never pass.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Callable
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx

from wsp_gateway.allowlist import ANTHROPIC, MISERVER

CLUSTER_SUFFIXES = (".svc", ".svc.cluster.local", ".cluster.local")
Resolver = Callable[[str], list[str]]


class EndpointRefused(ValueError):  # noqa: N818 - a refusal, like core.errors.Refusal
    """``wsp_endpoint_url_refused`` (the message names the rule, never the URL)."""


def default_resolver(host: str) -> list[str]:
    return sorted({str(info[4][0]) for info in socket.getaddrinfo(host, None)})


def _host(base_url: str) -> str:
    host = urlsplit(base_url).hostname
    if not host:
        raise EndpointRefused("the endpoint's base_url has no host")
    return host.lower().rstrip(".")


def derive_locality(base_url: str) -> str:
    """The gateway's copy of ``wsp.endpoints.derive_locality`` (a test pins that both agree)."""
    host = _host(base_url)
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if host == "localhost" or host.endswith(CLUSTER_SUFFIXES):
            return "local"
        return "external"
    if address.is_private or address.is_loopback or address.is_link_local:
        return "local"
    return "external"


def _allowlist(value: str) -> set[str]:
    return {h.strip().lower().rstrip(".") for h in value.split(",") if h.strip()}


#: The ranges a local endpoint's resolved address may fall in (FR-016, P61, R2A-10: "the cluster or
#: RFC 1918 ranges"): RFC 1918, loopback, and IPv6 unique-local and loopback (a dual-stack cluster).
#: ``ipaddress``'s ``is_private`` is wider and is not used: it admits link-local
#: (``169.254.169.254``, a cloud metadata service), ``0.0.0.0/8`` and the IANA special ranges.
LOCAL_NETWORKS = tuple(
    ipaddress.ip_network(n)
    for n in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "127.0.0.0/8", "::1/128", "fc00::/7")
)


def _private(address: str) -> bool:
    """Whether one resolved address is a local endpoint's; anything unparseable is not."""
    try:
        ip = ipaddress.ip_address(address)
    except ValueError:
        return False
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return any(ip in network for network in LOCAL_NETWORKS)


def check_endpoint(
    endpoint: dict[str, Any],
    *,
    local_allowlist: str,
    external_allowlist: str,
) -> str:
    """The endpoint's locality, or :class:`EndpointRefused` (before any upstream call)."""
    base_url = str(endpoint["base_url"])
    parts = urlsplit(base_url)
    if parts.scheme not in ("http", "https") or parts.username or parts.password:
        raise EndpointRefused("the endpoint's base_url must be http(s) without credentials")
    if parts.path not in ("", "/", "/v1", "/v1/") or parts.query or parts.fragment:
        raise EndpointRefused("the endpoint's base_url path must be empty or /v1, with no query")
    host = _host(base_url)
    derived = derive_locality(base_url)
    locality = "external" if "external" in (derived, endpoint.get("locality")) else "local"
    if locality == "external":
        if parts.scheme != "https" or host not in _allowlist(external_allowlist):
            raise EndpointRefused("the external host is not in WSP_EXTERNAL_HOST_ALLOWLIST")
    elif host not in _allowlist(local_allowlist):
        raise EndpointRefused("the local host is not in WSP_LOCAL_HOST_ALLOWLIST")
    return locality


def target(
    endpoint: dict[str, Any], route: str, locality: str, resolver: Resolver
) -> tuple[str, str | None]:
    """``(url, host_header)``. A local endpoint is called at its checked, resolved address."""
    parts = urlsplit(str(endpoint["base_url"]))
    path = "/v1" + route
    if locality == "external":
        return urlunsplit((parts.scheme, parts.netloc, path, "", "")), None
    host = _host(str(endpoint["base_url"]))
    try:
        addresses = resolver(host)
    except OSError as exc:
        raise EndpointRefused("the local host does not resolve") from exc
    # EVERY address must be local (review OR2-016-2): an answer mixing a private and a public
    # address is refused whatever its order. The call then goes to the address checked here, by
    # IP, with the name in the Host header, so a second resolution (DNS rebinding) is never made.
    if not addresses or not all(_private(a) for a in addresses):
        raise EndpointRefused("the local host resolves outside the cluster or RFC 1918 ranges")
    address = addresses[0]
    shown = f"[{address}]" if ":" in address else address
    netloc = shown if parts.port is None else f"{shown}:{parts.port}"
    host_header = host if parts.port is None else f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, netloc, path, "", "")), host_header


def headers_for(
    endpoint: dict[str, Any], base: dict[str, str], key: str | None, host_header: str | None
) -> dict[str, str]:
    headers = dict(base)
    if key is not None:
        if endpoint["kind"] == ANTHROPIC:
            headers["x-api-key"] = key
        else:
            headers["authorization"] = f"Bearer {key}"
    if endpoint.get("serving") == MISERVER:
        headers["X-miLLM-Origin"] = "judge"
        headers["X-miLLM-Load-Policy"] = "refuse"
    if host_header is not None:
        headers["host"] = host_header
    return headers


def forward(client: httpx.Client, url: str, body: bytes, headers: dict[str, str]) -> httpx.Response:
    """Exactly one call; the client is built with no retries and no redirects."""
    return client.post(url, content=body, headers=headers)
