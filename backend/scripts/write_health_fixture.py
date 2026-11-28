"""Write ``frontend/src/testing/fixtures/health.json`` from the REAL ``health_payload()``.

Feature 015 (FR-015.15, FTID §6.2). The frontend's test stubs used to answer ``/api/health``
with a hand-written body (``dependencies: []`` - an array, where the backend serves an object),
and a parser nobody imported (``frontend/src/api/health.ts``) described a shape the backend never
had. Generating the fixture from the backend's own function, and diffing it in
``backend/tests/contract/test_health_fixture.py``, makes the stub change when the answer does.

Conditions are fixed so the output is byte-stable: both dependency probes answer ok, no sibling
or old environment name is set, no ``.env`` is read, the store holds nothing.

    python backend/scripts/write_health_fixture.py        # from the repository root
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURE = REPO_ROOT / "frontend" / "src" / "testing" / "fixtures" / "health.json"
CLEARED_PREFIXES = ("MILLM_", "MISERVER_", "MIDATAWORKS_", "ACTIVATION", "ENVIRONMENT")
SRC = str(REPO_ROOT / "backend" / "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


@contextmanager
def fixed_conditions() -> Iterator[None]:
    """The conditions the fixture is defined under; everything is restored on exit."""
    from api import health as health_module
    from core.config import reset_settings
    from core.runtime_settings import reset as reset_runtime
    from services.legacy_names import store

    saved_env = dict(os.environ)
    saved_probes = dict(health_module.DEPENDENCY_PROBES)
    saved_store = (store.record, store.read_recent)
    saved_cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as empty:
        try:
            for name in list(os.environ):
                if name.upper().startswith(CLEARED_PREFIXES) or "BASE_URL" in name.upper():
                    del os.environ[name]
            os.chdir(empty)  # no `.env` is read
            for name in health_module.DEPENDENCY_PROBES:
                health_module.DEPENDENCY_PROBES[name] = lambda: {"status": "ok", "detail": None}
            store.record = lambda *a, **k: 0  # type: ignore[assignment]
            store.read_recent = lambda **_: store.StoredUses()  # type: ignore[assignment]
            reset_settings()
            reset_runtime()
            yield
        finally:
            os.chdir(saved_cwd)
            os.environ.clear()
            os.environ.update(saved_env)
            health_module.DEPENDENCY_PROBES.clear()
            health_module.DEPENDENCY_PROBES.update(saved_probes)
            store.record, store.read_recent = saved_store  # type: ignore[assignment]
            reset_settings()
            reset_runtime()


def render() -> str:
    """The fixture's exact text under :func:`fixed_conditions`."""
    from api.health import health_payload

    with fixed_conditions():
        payload: dict[str, Any] = health_payload()
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def main() -> int:
    FIXTURE.write_text(render(), "utf-8")
    print(f"wrote {FIXTURE.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
