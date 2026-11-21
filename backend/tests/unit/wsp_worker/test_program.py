"""Program mode's sandbox and ``SupportSet`` (feature 016; FTASKS 9.4-9.5; FR-016.23; R1B-A9,
R1B-A10).

Run in the backend's interpreter: ``wsp_worker/program.py`` imports numpy and the worker's own
modules only (never weightspace), so it needs no worker venv. The rerun child end to end is in
``integration/wsp/test_replay_and_rerun.py``.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp.weight_ops.patch import apply_patch
from wsp_worker import bundle as bundle_mod
from wsp_worker import program
from wsp_worker.program import ProgramError

PROGRAM = Path(program.__file__)

GOOD = """
def edit(base, support):
    w = np.array(base["weight"])
    pos = support.labels[support.rows] == 1
    scores = support.score({"weight": w, "bias": base["bias"]})
    w = w + np.float32(0.125) * np.sign(w)
    return {"weight": w.astype(base["weight"].dtype)}
"""
SEEDED = """
def edit(base, support):
    w = np.array(base["weight"]) + rng.standard_normal(base["weight"].shape).astype(np.float32)
    return {"weight": w}
"""
UNSEEDED = """
def edit(base, support):
    w = np.array(base["weight"]) + np.random.default_rng().standard_normal(base["weight"].shape)
    return {"weight": w.astype(base["weight"].dtype)}
"""


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "b", rule="mean")


def _bundle(fx: FixtureBundle) -> bundle_mod.Bundle:
    files = [{k: e[k] for k in ("name", "role", "sha256")} for e in fx.manifest["files"]]
    return bundle_mod.load(fx.root, fx.manifest, files)


def _request(fx: FixtureBundle, **limits: Any) -> dict[str, Any]:
    return {
        "kind": "repair",
        "parent": {"kind": "bundle", "ref": fx.manifest["bundle_sha256"]},
        "change_limits": {
            "max_changed_entries": None,
            "allowed_tensors": None,
            "frozen_tensors": [],
            **limits,
        },
    }


def _checked(fx: FixtureBundle, source: str, kind: str = "practice_train") -> program.Outcome:
    b = _bundle(fx)
    return program.run_checked(
        source,
        b,
        program.SupportSet(kind, b),
        _request(fx),
        "wsp_000000000001",
        "2026-11-16T00:00Z",
        created_by={"provider": "weightspace", "provider_version": "0.3.0"},
    )


def test_a_program_gives_a_patch_that_reapplies_and_two_runs_agree(fx: FixtureBundle) -> None:
    a, b = _checked(fx, GOOD), _checked(fx, GOOD)
    assert a.manifest is not None and a.content_sha256 == b.content_sha256
    directory = Path(fx.root) / "patch-a"
    directory.mkdir()
    (directory / "patch.json").write_text(json.dumps(a.manifest))
    assert a.data is not None
    (directory / "edit.npz").write_bytes(a.data)
    _manifest, rebuilt = apply_patch(directory, fx.tensors)
    w = fx.tensors["weight"].values64().astype(np.float32)
    assert np.array_equal(
        rebuilt["weight"].values64().astype(np.float32), w + np.float32(0.125) * np.sign(w)
    )


def test_the_seeded_generator_is_deterministic_and_unseeded_randomness_is_refused(
    fx: FixtureBundle,
) -> None:
    """R1B-A10: `rng` is seeded from the parent; any other source of randomness disagrees."""
    assert _checked(fx, SEEDED).content_sha256 == _checked(fx, SEEDED).content_sha256
    with pytest.raises(ProgramError) as caught:
        _checked(fx, UNSEEDED)
    assert caught.value.code == "wsp_program_nondeterministic"


@pytest.mark.parametrize(
    "source",
    [
        "x = 1\n",  # no edit
        "edit = 3\n",  # not callable
        "def edit(base, support):\n    return [1]\n",
        "def edit(base, support):\n    return {'mean': base['weight']}\n",  # not editable
        "def edit(base, support):\n    return {'weight': base['weight'].astype(np.float64)}\n",
        "def edit(base, support):\n    return {'weight': base['weight'][:2]}\n",
        "def edit(base, support):\n    return {'weight': base['weight'] * np.nan}\n",
        "def edit(base, support):\n    import os\n    return {}\n",
        "def edit(base, support):\n    return open('/etc/passwd')\n",
        "def edit(base, support):\n    base['weight'][0] = 1.0\n    return {}\n",  # read-only
        "def edit(base, support):\n    return {'weight': base['weight']}\n  bad indent",
    ],
    ids=[
        "no-edit",
        "not-callable",
        "not-a-mapping",
        "not-editable",
        "dtype",
        "shape",
        "non-finite",
        "import-os",
        "open",
        "write-base",
        "syntax",
    ],
)
def test_an_invalid_program_is_refused_wsp_program_invalid(fx: FixtureBundle, source: str) -> None:
    with pytest.raises(ProgramError) as caught:
        _checked(fx, source)
    assert caught.value.code == "wsp_program_invalid"


def test_a_frozen_tensor_cannot_be_returned(fx: FixtureBundle) -> None:
    b = _bundle(fx)
    request = _request(fx, frozen_tensors=["weight"])
    with pytest.raises(ProgramError):
        program.run_once(
            "def edit(base, support):\n    return {'weight': base['weight'] + 1}\n",
            b,
            program.SupportSet("practice", b),
            request,
        )


def test_the_support_sets(fx: FixtureBundle) -> None:
    b = _bundle(fx)
    train = program.SupportSet("practice_train", b)
    every = program.SupportSet("practice", b)
    validation = np.load(fx.paths_by_role["practice_validation_mask"])
    assert list(every.rows) == list(range(fx.manifest["practice"]["rows"]))
    assert list(train.rows) == [r for r in range(len(validation)) if not validation[r]]
    assert not train.rows.flags.writeable and not train.labels.flags.writeable
    with pytest.raises(ProgramError):
        train.tokens(int(np.flatnonzero(validation)[0]))  # a validation row is not support
    with pytest.raises(ProgramError):
        train.score({}, rows=[int(np.flatnonzero(validation)[0])])
    with pytest.raises(ProgramError):
        program.SupportSet("heldout", b)


def test_a_practice_rerun_differs_from_a_train_rerun_when_the_program_reads_its_support(
    fx: FixtureBundle,
) -> None:
    """P21: a rerun rebuilds on its named support set, so the support matters."""
    source = (
        "def edit(base, support):\n"
        "    n = float(len(support.rows))\n"
        "    return {'bias': (base['bias'] + n).astype(base['bias'].dtype)}\n"
    )
    assert _checked(fx, source, "practice").content_sha256 != _checked(fx, source).content_sha256


def test_program_py_never_imports_weightspace() -> None:
    """R1B-A9: an import here would load ``weightspace.llm`` (its ``__init__``) in every rerun."""
    tree = ast.parse(PROGRAM.read_text())
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not {n for n in names if n.split(".")[0] in ("weightspace", "anthropic", "openai")}


# ----------------------------------------------- feature 018 (FR-018.15; FTASKS 1.1, 1.3)

MIWSP = {"provider": "miwsp_engine", "provider_version": "0.1.0"}


def test_outcome_stamps_the_provider_it_is_given(fx: FixtureBundle) -> None:
    """018 1.1: ``created_by`` is the caller's provider, never a hard-coded one (program.py:212
    stamped ``weightspace`` 0.3.0 on every program and every rerun)."""
    b = _bundle(fx)
    request = _request(fx)
    edited = program.run_once(GOOD, b, program.SupportSet("practice_train", b), request)
    done = program.outcome(edited, b, request, "wsp_1", "2026-11-18T00:00Z", created_by=MIWSP)
    assert done.manifest is not None
    assert done.manifest["created_by"] == {**MIWSP, "run_id": "wsp_1"}
    other = program.outcome(
        edited,
        b,
        request,
        "wsp_1",
        "2026-11-18T00:00Z",
        created_by={"provider": "weightspace", "provider_version": "0.3.0"},
    )
    assert other.manifest is not None
    # content_sha256 excludes created_by, so a provider never changes what a rerun reproduces.
    assert other.content_sha256 == done.content_sha256


def _rerun_line(fx: FixtureBundle, tmp: Path, provider: dict[str, Any] | None) -> dict[str, Any]:
    import hashlib
    import shutil

    work, out = tmp / "work", tmp / "out"
    work.mkdir()
    out.mkdir()
    files = []
    for entry in fx.manifest["files"]:
        shutil.copyfile(fx.root / entry["name"], work / entry["name"])
        files.append({k: entry[k] for k in ("name", "role", "sha256")})
    (work / program.PROGRAM_NAME).write_text(GOOD)
    files.append({"name": program.PROGRAM_NAME, "role": "program", "sha256": "0" * 64})
    request: dict[str, Any] = {
        **_request(fx),
        "mode": "program_rerun",
        "program_sha256": hashlib.sha256(GOOD.encode()).hexdigest(),
        "support_set": {"kind": "practice_train", "digest": "0" * 64},
    }
    if provider is not None:
        request["provider"] = provider
    return {
        "mode": "program_rerun",
        "run_id": "wsp_000000000002",
        "spec": {"request": request, "manifest": fx.manifest},
        "files": files,
        "gateway": None,
        "paths": {"work": str(work), "out": str(out), "tmp": str(tmp)},
        "time_left_s": 600.0,
    }


@pytest.mark.parametrize(
    "provider", [("miwsp_engine", "0.1.0"), ("weightspace", "0.3.0")], ids=lambda p: p[0]
)
def test_a_rerun_stamps_the_runs_own_provider(
    fx: FixtureBundle, tmp_path: Path, provider: tuple[str, str]
) -> None:
    """018 1.1: ``program.rerun`` stamps ``request.provider`` (the first run's, inherited by
    ``runs.create_rerun``), so an engine program's rerun is not mislabelled ``weightspace``."""
    line = _rerun_line(fx, tmp_path, {"id": provider[0], "version": provider[1]})
    assert program.rerun(line, "2026-11-18T00:00Z") == 0
    manifest = json.loads((tmp_path / "out" / "patch" / "patch.json").read_text())
    assert manifest["created_by"] == {
        "provider": provider[0],
        "provider_version": provider[1],
        "run_id": "wsp_000000000002",
    }


def test_a_rerun_whose_request_names_no_provider_is_refused(
    fx: FixtureBundle, tmp_path: Path
) -> None:
    """Refuse rather than guess (ADR-018): no provider means no provenance to stamp."""
    line = _rerun_line(fx, tmp_path, None)
    with pytest.raises(bundle_mod.BundleError) as caught:
        program.rerun(line, "2026-11-18T00:00Z")
    assert caught.value.code == "worker_spec_invalid"
    assert not (tmp_path / "out" / "patch").exists()


def test_a_tensor_allowed_only_by_a_pattern_is_editable_in_a_program(fx: FixtureBundle) -> None:
    """018 1.3: ``allowed_tensors`` and ``frozen_tensors`` are ``fnmatchcase`` patterns, as in
    ``limits.check_limits``; exact membership withheld a pattern-allowed tensor (program.py:136)."""
    b = _bundle(fx)
    source = "def edit(base, support):\n    return {'weight': base['weight'] + np.float32(1)}\n"
    request = _request(fx, allowed_tensors=["wei*"])
    edited = program.run_once(source, b, program.SupportSet("practice", b), request)
    assert not np.array_equal(edited["weight"].bits, b.tensors["weight"].bits)
    assert program.editable_names(b, request["change_limits"]) == ["weight"]
    frozen = _request(fx, frozen_tensors=["w*"])
    assert "weight" not in program.editable_names(b, frozen["change_limits"])
    with pytest.raises(ProgramError):
        program.run_once(source, b, program.SupportSet("practice", b), frozen)
