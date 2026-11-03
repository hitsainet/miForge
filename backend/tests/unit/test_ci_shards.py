"""The sharded Backend Tests run the whole tree, each file exactly once (operator, 2026-10-20).

`backend-tests.yml` splits the tree across a matrix of jobs with `CI_SHARD=<i>/<n>`, and the
root conftest's `pytest_ignore_collect` skips every test file another shard owns
(`tests/support/ci_shard.py`). A split like that fails quietly: a file no shard owns never
runs and every job is green. So this file pins three things:

* the partition: for the real tree and the shard count the workflow declares, every test file
  belongs to exactly one shard, including a file the recorded durations have never seen;
* the wiring: the hook is registered in the LIVE plugin manager, from the root conftest, and
  answers for a real file (deleting the conftest hook turns this red);
* the behaviour under pytest itself: on a throwaway tree, `--collect-only` per shard gives
  disjoint node sets whose union is the unsharded collection, and an unset CI_SHARD ignores
  nothing.

The workflow side (matrix, CI_SHARD string, the all-shards aggregator) is pinned in
test_private_ci_trust.py.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support import ci_shard

TEST_ROOT = Path(ci_shard.__file__).resolve().parent.parent


def _patterns(config) -> list[str]:
    return list(config.getini("python_files"))


def test_parse_accepts_only_a_valid_spec():
    assert ci_shard.parse(None) is None and ci_shard.parse("") is None
    assert ci_shard.parse("2/5") == (2, 5)
    for bad in ("0/3", "4/3", "3", "a/b", "1/0"):
        with pytest.raises(ValueError):
            ci_shard.parse(bad)


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8])
def test_every_real_test_file_lands_in_exactly_one_shard(request, n):
    files = ci_shard.collectable_files(TEST_ROOT, _patterns(request.config))
    assert len(files) > 20, "the walk found almost nothing - the partition below would be vacuous"
    durations = ci_shard.load_durations()
    assignment = ci_shard.assign(files, n, durations)
    assert sorted(assignment) == files
    owners = {
        f: [i for i in range(1, n + 1) if ci_shard.shard_of(f, n, assignment) == i] for f in files
    }
    assert all(len(v) == 1 for v in owners.values())
    # A file the durations never saw is still owned by exactly one shard.
    new = "unit/test_zz_not_in_the_durations_file.py"
    assert new not in durations
    grown = ci_shard.assign([*files, new], n, durations)
    assert grown[new] in range(1, n + 1)


def test_the_recorded_durations_name_real_files(request):
    """Stale durations cost balance, not coverage - but a file of names that match nothing
    means the balancing is fiction. Most named files must still exist."""
    files = set(ci_shard.collectable_files(TEST_ROOT, _patterns(request.config)))
    named = set(ci_shard.load_durations())
    assert named, "ci_shard_durations.json is missing or empty"
    assert len(named & files) >= 0.8 * len(named)


def test_the_root_conftest_registers_the_shard_hook(request, monkeypatch):
    impls = [
        h
        for h in request.config.pluginmanager.hook.pytest_ignore_collect.get_hookimpls()
        if Path(getattr(h.plugin, "__file__", "") or "").resolve() == TEST_ROOT / "conftest.py"
    ]
    assert len(impls) == 1, "the root conftest does not register pytest_ignore_collect"
    hook = impls[0].function

    files = ci_shard.collectable_files(TEST_ROOT, _patterns(request.config))
    assignment = ci_shard.assign(files, 2, ci_shard.load_durations())
    mine = next(f for f in files if assignment[f] == 1)
    theirs = next(f for f in files if assignment[f] == 2)

    monkeypatch.setenv(ci_shard.ENV, "1/2")
    assert hook(collection_path=TEST_ROOT / theirs, config=request.config) is True
    assert hook(collection_path=TEST_ROOT / mine, config=request.config) is None
    monkeypatch.delenv(ci_shard.ENV)
    assert hook(collection_path=TEST_ROOT / theirs, config=request.config) is None


_CONFTEST = """\
from pathlib import Path
import ci_shard

def pytest_ignore_collect(collection_path, config):
    return ci_shard.ignore(collection_path, config, Path(__file__).parent)
"""


def _collect(tree: Path, shard: str | None) -> set[str]:
    env = {k: v for k, v in os.environ.items() if k != ci_shard.ENV and not k.startswith("PYTEST_")}
    if shard:
        env[ci_shard.ENV] = shard
    env["PYTHONPATH"] = str(tree / "support")
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
            "-c",
            str(tree / "pytest.ini"),
            "--rootdir",
            str(tree),
            str(tree / "tests"),
        ],
        cwd=tree,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert proc.returncode in (0, 5), proc.stdout + proc.stderr
    return {line.strip() for line in proc.stdout.splitlines() if "::" in line}


def test_pytest_collects_each_node_in_exactly_one_shard(tmp_path):
    tree = tmp_path
    (tree / "support").mkdir()
    shutil.copy(ci_shard.__file__, tree / "support" / "ci_shard.py")
    (tree / "pytest.ini").write_text("[pytest]\npython_files = test_*.py\n")
    tests = tree / "tests"
    (tests / "deep" / "er").mkdir(parents=True)
    (tests / "conftest.py").write_text(_CONFTEST)
    for i in range(11):
        sub = tests if i % 3 == 0 else (tests / "deep" if i % 3 == 1 else tests / "deep" / "er")
        (sub / f"test_m{i}.py").write_text(
            "".join(f"def test_{j}():\n    pass\n" for j in range(i % 4 + 1))
        )
    (tests / "helper.py").write_text("def test_not_collected():\n    pass\n")

    everything = _collect(tree, None)
    assert len(everything) == sum(i % 4 + 1 for i in range(11))
    n = 3
    shards = [_collect(tree, f"{i}/{n}") for i in range(1, n + 1)]
    assert all(shards), "a shard collected nothing - the split is not balanced at all"
    assert set().union(*shards) == everything, "some test node is in no shard"
    assert sum(len(s) for s in shards) == len(everything), "some test node is in two shards"
