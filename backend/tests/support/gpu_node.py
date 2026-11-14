"""The one place a ``gpu`` test decides it has no CUDA device to run on.

``pyproject.toml`` deselects the ``gpu`` marker by default, so CI (a CPU torch wheel) never
collects these tests. A run that selects them on a machine with no usable device must SKIP -
loudly, naming the test and the reason - never fail with a CUDA error that reads as a
regression (four tests did exactly that until 2026-11-11: see
``tests/unit/test_gpu_tests_skip_loudly.py``).

**A skip is only honest where no card was promised.** On the GPU node set
``MIFORGE_REQUIRE_GPU=1``: every skip this module would raise becomes a FAILURE, so a GPU run
cannot go green by skipping what it exists to measure. (The same shape as
``*_REQUIRE_CROSS_REPO_CHECKS`` for an absent sibling, ``tests/support/siblings.py``.)

Use :func:`require_cuda` at the top of a fixture or test that touches ``cuda`` directly, and
:func:`absent` where a more specific reason is already in hand (``resolve_device`` refused).
"""

from __future__ import annotations

import os
from typing import NoReturn

import pytest

#: Set to "1" where a CUDA device is promised: an absent device then fails instead of skipping.
REQUIRE_ENV = "MIFORGE_REQUIRE_GPU"


def _current_test() -> str:
    """The running test's node id (pytest exports it), without the phase suffix."""
    return os.environ.get("PYTEST_CURRENT_TEST", "this test").rsplit(" (", 1)[0]


def cuda_unusable() -> str | None:
    """``None`` when torch can use a CUDA device in this process, else why it cannot."""
    try:
        import torch
    except ImportError as exc:
        return f"torch is not installed ({exc})"
    if torch.cuda.is_available():
        return None
    if torch.version.cuda is None:
        return f"torch {torch.__version__} is built without CUDA support"
    return "torch.cuda.is_available() is False (no driver, or no device visible to this process)"


def absent(reason: str) -> NoReturn:
    """Skip the running test by name - or fail it under ``MIFORGE_REQUIRE_GPU=1``."""
    what = f"GPU: {_current_test()} needs a usable CUDA device and there is none: {reason}."
    if os.environ.get(REQUIRE_ENV) == "1":
        pytest.fail(
            f"{what} {REQUIRE_ENV}=1 says this machine has one, so this is a failure, not a skip.",
            pytrace=False,
        )
    pytest.skip(f"{what} On the GPU node run with {REQUIRE_ENV}=1 so a skip there fails.")


def require_cuda() -> None:
    """Return when torch can use a CUDA device; otherwise :func:`absent`."""
    reason = cuda_unusable()
    if reason is not None:
        absent(reason)
