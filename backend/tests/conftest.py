"""Root conftest for the backend test tree: only the CI-shard hook lives here."""

from __future__ import annotations

from pathlib import Path


def pytest_ignore_collect(collection_path: Path, config: object) -> bool | None:
    """CI shards: with `CI_SHARD=<i>/<n>` set, skip every test file another shard owns.

    Unset (every local run, the lint job) this ignores nothing. In CI the shards of each
    `backend-tests.yml` engine job together run the whole tree, each file exactly once:
    tests/support/ci_shard.py, proven by tests/unit/test_ci_shards.py.
    """
    from tests.support import ci_shard

    return ci_shard.ignore(collection_path, config, Path(__file__).resolve().parent)
