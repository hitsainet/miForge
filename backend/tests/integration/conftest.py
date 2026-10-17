"""Fixtures for the §5.10 integration tests, auto-discovered rather than imported.

**Why a conftest and not an import.** This repository's house pattern is that
``tests/support/`` exposes plain functions and each test file wraps them in its own
fixtures. §5.10's fixtures are shared by a dozen files, so importing them into each one
would shadow the module-level name with the test's parameter in every signature - Ruff's
``F811``, a hundred times over, and the kind of noise a reviewer learns to skip past.
``conftest.py`` is what pytest provides for exactly this, and it adds no hook and no
autouse behaviour: three fixtures become available and nothing else changes.
"""

from __future__ import annotations

# ⚠ TOLERANT ON PURPOSE, AND THIS IS ADR-022'S FENCE.
#
# `fixtures_hotupdate` imports `training.hotupdate` unconditionally, and a conftest is
# imported during COLLECTION. So while this import was unconditional, moving
# `backend/src/training/hotupdate/` aside did not fail F11's tests — it stopped the
# **entire unit and integration suite from collecting**, with
# `ImportError while loading conftest ... No module named 'training.hotupdate'`.
#
# That made the §5.10 deletion control unreachable, which is the one control ADR-022's
# fence exists to make possible: *"it lives under `training/hotupdate/`, the standalone
# test runs with it absent."* The production fence is real and has been seen red; the
# TEST harness had a hard dependency the production code does not, and a conftest is not
# an F11 test file so it stays in the tree when F11's own files are deleted. The green run
# that was taken as evidence deleted the package **and all 25 F11 test files together**,
# which is a weaker property than the one being claimed.
#
# With the guard, deleting the package leaves the rest of the suite collecting, and F11's
# own tests fail loudly on a missing fixture — which is the correct and visible outcome,
# not a silent skip.
try:
    from tests.support.fixtures_hotupdate import (  # noqa: F401 - re-exported as fixtures
        harness,
        hotupdate_env,
        surface,
    )
except ImportError as exc:  # pragma: no cover - exercised by the §5.10 deletion control
    import warnings

    warnings.warn(
        f"§5.10's fixtures are unavailable ({exc}). This is expected and correct when "
        f"training/hotupdate/ is absent: ADR-022 fences the optional hot-update mode and "
        f"the rest of the suite must still collect and pass without it. Tests that need "
        f"these fixtures will fail by name rather than be skipped.",
        RuntimeWarning,
        stacklevel=2,
    )
