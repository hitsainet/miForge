"""The backend's and the engine's ``structured`` predicates agree (feature 019 P2; S-6, FR-019.11).

``wsp.endpoints.check_action_protocol`` decides at registration and PATCH, the engine's
``protocol.check_supported`` decides in the session; a disagreement would register an endpoint the
engine refuses, or the reverse. Checked over every kind, serving and protocol.
"""

from __future__ import annotations

import itertools

import pytest

from core.errors import Refusal
from wsp.endpoints import check_action_protocol
from wsp.models import ACTION_PROTOCOLS, ENDPOINT_KINDS, SERVINGS
from wsp_worker.miwsp_engine import protocol as proto


def _backend(protocol: str, kind: str, serving: str) -> bool:
    try:
        check_action_protocol(protocol, kind=kind, serving=serving)
    except Refusal:
        return False
    return True


def _engine(protocol: str, kind: str, serving: str) -> bool:
    try:
        proto.check_supported(protocol, kind, serving)
    except proto.ProtocolUnsupportedError:
        return False
    return True


@pytest.mark.parametrize(
    ("protocol", "kind", "serving"),
    list(itertools.product(ACTION_PROTOCOLS, ENDPOINT_KINDS, SERVINGS)),
)
def test_the_two_predicates_agree(protocol: str, kind: str, serving: str) -> None:
    assert _backend(protocol, kind, serving) == _engine(protocol, kind, serving)


def test_structured_is_admitted_for_both_openai_servings_and_refused_for_anthropic() -> None:
    assert _backend("structured", "openai_compatible", "generic")
    assert _backend("structured", "openai_compatible", "miserver")
    assert not _backend("structured", "anthropic", "generic")
