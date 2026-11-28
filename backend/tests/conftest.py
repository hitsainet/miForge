"""Root conftest for the backend test tree: the CI-shard hook and the answer validator's
registration.

The answer validator (021 FR-021.12) checks every JSON answer any test receives against the
published answer contract. It is registered here, in the one conftest every run loads, so no test
can run without it; ``tests/unit/support/test_answer_validator_wiring.py`` proves the registration
is live.
"""

from __future__ import annotations

from pathlib import Path

import pytest

#: The validator's plugin name (its module); the wiring test looks it up by this name.
ANSWER_VALIDATOR = "tests.support.answer_validator"


#: The backend's own packages the validator needs. The wsp-worker-image CI job runs
#: ``backend/tests/image`` with pytest alone (``requirements-wsp-worker-test.txt``); there is no app
#: there to answer, and the validator is not registered (021 review round 2, F1).
VALIDATOR_NEEDS = ("fastapi", "jsonschema", "referencing")


def validator_can_load() -> bool:
    import importlib.util

    return all(importlib.util.find_spec(name) is not None for name in VALIDATOR_NEEDS)


def pytest_configure(config: pytest.Config) -> None:
    if not validator_can_load():
        return
    from tests.support import answer_validator

    if not config.pluginmanager.is_registered(answer_validator):
        config.pluginmanager.register(answer_validator, ANSWER_VALIDATOR)


def pytest_ignore_collect(collection_path: Path, config: object) -> bool | None:
    """CI shards: with `CI_SHARD=<i>/<n>` set, skip every test file another shard owns.

    Unset (every local run, the lint job) this ignores nothing. In CI the shards of each
    `backend-tests.yml` engine job together run the whole tree, each file exactly once:
    tests/support/ci_shard.py, proven by tests/unit/test_ci_shards.py.
    """
    from tests.support import ci_shard

    return ci_shard.ignore(collection_path, config, Path(__file__).resolve().parent)
