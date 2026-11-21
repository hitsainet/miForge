"""What the WSP worker's child must see, as literals (feature 016; FTID §3.2, FR-016.25).

The tests compare the child's real environment with these literals, never with
``wsp_worker.spec.child_env``: a test that compares the code with itself passes against any change
to ``CHILD_ENV`` (control C25, a fake ``DATABASE_URL`` added to ``CHILD_ENV``, survived exactly that
way on 2026-11-16).
"""

from __future__ import annotations

import re

#: FTID §3.2: ``PATH``, ``HOME=/tmp``, ``PYTHONNOUSERSITE=1``, ``WSP_MODE`` and the three
#: thread-count variables. Nothing else, and no credential of any kind (D20).
CHILD_ENV_LITERAL: dict[str, str] = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

#: Names that must never reach the child, whatever the literal above says.
CREDENTIAL_SHAPED = re.compile(
    r"(DATABASE|POSTGRES|REDIS|CELERY|BROKER|TOKEN|SECRET|PASSWORD|PASSWD|API_?KEY|_KEY$|^KEY$|"
    r"CREDENTIAL|AUTH|MIFORGE_|MISERVER_|MIWORKBENCH_|MIDATAWORKS_|ANTHROPIC|OPENAI|AWS_|HF_)",
    re.IGNORECASE,
)


def expected_child_env(mode: str) -> dict[str, str]:
    """The literal plus ``WSP_MODE``."""
    return {**CHILD_ENV_LITERAL, "WSP_MODE": mode}


def assert_child_env(env: dict[str, str], mode: str = "run") -> None:
    """The child's environment is exactly the literal, and carries no credential-shaped name."""
    leaked = sorted(k for k in env if CREDENTIAL_SHAPED.search(k))
    assert leaked == [], f"credential-shaped names in the child env: {leaked}"
    assert env == expected_child_env(mode), sorted(set(env) ^ set(expected_child_env(mode)))
