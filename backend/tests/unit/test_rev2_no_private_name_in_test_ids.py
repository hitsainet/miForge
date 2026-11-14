"""Review round 2 of 015 P0: a runtime-built private name must not come back as a test ID.

Round 1 built the private organisation at runtime (``li.PRIVATE``) so the mirror-bound files do
not carry it. But ``pytest.mark.parametrize`` over the substituted plants derives the node ID
from the value, so ``test_repo_allowance_only_from_transition_records[<org>/miLLM]`` is
collected on the public mirror too, and pytest prints node IDs in the mirror's public CI log
on any failure, error or ``-rs``/``-v`` run.
"""

from __future__ import annotations

import subprocess
import sys

from tests.support import legacy_identifiers as li

FILES = (
    "backend/tests/unit/test_brand_guard.py",
    "backend/tests/unit/test_origin_headers.py",
    "backend/tests/unit/test_workflows_name_no_sibling.py",
    "backend/tests/contract/test_legacy_identifiers.py",
    "backend/tests/unit/test_manual_publish_allowlist.py",
)


def test_no_collected_id_names_the_private_organisation() -> None:
    done = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider", *FILES],
        cwd=li.REPO_ROOT,
        capture_output=True,
        text=True,
    )
    assert done.returncode == 0, done.stdout[-2000:] + done.stderr[-2000:]
    ids = [line for line in done.stdout.splitlines() if "::" in line]
    assert len(ids) > 20, f"collected only {len(ids)} ids: the check saw nothing"
    leaking = [i for i in ids if li.PRIVATE.lower() in i.lower()]
    assert not leaking, f"test IDs carry the private organisation: {leaking}"
