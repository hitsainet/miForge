"""Fixtures for the contract tests that drive real code (feature 015's frozen identifiers).

The same two shared fixtures ``integration/conftest.py`` provides, for the same reasons: a
conftest instead of an import, so a test taking ``world`` does not shadow an import (Ruff
F811); and the §5.10 fixtures imported TOLERANTLY, so ADR-022's deletion control (training/
hotupdate/ moved aside) still lets every other contract test collect.
"""

from __future__ import annotations

from tests.support.approval_world import world  # noqa: F401

try:
    from tests.support.fixtures_hotupdate import (  # noqa: F401 - re-exported as fixtures
        harness,
        hotupdate_env,
        surface,
    )
except ImportError as exc:  # pragma: no cover - exercised by the §5.10 deletion control
    import warnings

    warnings.warn(
        f"§5.10's fixtures are unavailable ({exc}); the frozen-identifier test for DR-08 will "
        f"fail by name rather than be skipped (ADR-022).",
        RuntimeWarning,
        stacklevel=2,
    )
