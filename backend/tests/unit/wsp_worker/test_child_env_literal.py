"""``CHILD_ENV`` is pinned against the FTID's literal, not against itself (FTID §3.2; control C25).

The live and local tests read the child's real environment; this one catches a change to the
literal before any child runs.
"""

from __future__ import annotations

from tests.support.wsp_expected import CREDENTIAL_SHAPED, expected_child_env
from wsp_worker import spec


def test_child_env_is_the_ftid_literal() -> None:
    for mode in sorted(set(spec.CHILD_MODE.values())):
        assert spec.child_env(mode) == expected_child_env(mode)


def test_the_credential_pattern_can_see_a_planted_name() -> None:
    """The guard's own positive check: it is not blind to the names it exists for."""
    for name in (
        "DATABASE_URL",
        "REDIS_URL",
        "WSP_GATEWAY_SERVICE_TOKEN",
        "ANTHROPIC_API_KEY",
        "HF_TOKEN",
    ):
        assert CREDENTIAL_SHAPED.search(name), name
    for name in expected_child_env("run"):
        assert not CREDENTIAL_SHAPED.search(name), name
