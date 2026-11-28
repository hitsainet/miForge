"""The pool's scheduling: the key, the effective pool size, the worker addresses (feature 019 P3;
FR-019.6, FR-019.7, FR-019.8; F19-D6, D7, D17, D18, D24, D31, D33; TID §3.7).

- **The key** (F19-D17): at most one run is active (``pushing`` or ``running``) per key, held by a
  partial unique index (``uq_mf_wsp_runs_one_active_key``). A generic endpoint is its own key,
  ``endpoint:<id>``; miServer endpoints that share a server share one key, ``miserver:<origin>``
  (F19-D6), because one miServer serves one resident model to every row that names it. A replay
  or a program rerun has no endpoint and no key: it takes any free worker (F19-D7).
- **The effective pool** is the runtime setting ``wsp_pool_size``, bounded by the provisioned
  maximum ``WSP_WORKER_POOL_MAX`` (the StatefulSet's replicas) and at least 1 (F19-D31).
- **The workers** are addressed one by one (FR-019.7), never through a load-balanced name: a
  ``WSP_WORKER_URL`` with ``{index}`` names worker ``i`` for ``i`` in ``0..max-1``; one without it
  is a single worker and needs a maximum of 1.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from wsp.models import WspEndpoint
    from wsp.settings import WSPSettings

INDEX_PLACEHOLDER = "{index}"


def origin_of(base_url: str) -> str:
    """``scheme://host:port``: the host lower-cased without a trailing dot, the port 80 or 443 by
    default, the path (empty, ``/`` or ``/v1``) dropped."""
    parts = urlsplit(base_url)
    scheme = (parts.scheme or "http").lower()
    host = (parts.hostname or "").lower().rstrip(".")
    port = parts.port or (443 if scheme == "https" else 80)
    if ":" in host:  # an IPv6 literal keeps its brackets
        host = f"[{host}]"
    return f"{scheme}://{host}:{port}"


def key_for(endpoint: WspEndpoint) -> str:
    """The run's scheduling key (F19-D17, F19-D6)."""
    if endpoint.serving == "miserver":
        return f"miserver:{origin_of(endpoint.base_url)}"
    return f"endpoint:{endpoint.id}"


def effective_pool(pool_size: int, maximum: int) -> int:
    """How many workers the pass may hold at once: the setting, at most the maximum, at least 1."""
    return max(1, min(int(pool_size), int(maximum)))


def worker_urls(wsp: WSPSettings) -> list[str]:
    """Every provisioned worker's address, by index; ``[]`` when the template rule is broken (a
    maximum above 1 with no ``{index}``), which ``WSPSettings.missing_urls`` reports."""
    url, maximum = wsp.wsp_worker_url, wsp.wsp_worker_pool_max
    if not url:
        return []
    if INDEX_PLACEHOLDER in url:
        return [url.replace(INDEX_PLACEHOLDER, str(i)) for i in range(maximum)]
    return [url] if maximum == 1 else []
