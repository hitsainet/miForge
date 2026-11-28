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
        created_by={"provider": "miwsp_engine", "provider_version": "0.1.0"},
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


def test_program_py_never_imports_a_model_client() -> None:
    """R1B-A9: an import here would load a model client in every rerun (the rerun child checks
    ``wsp_worker.__main__.MODEL_CLIENT_MODULES`` when it ends)."""
    tree = ast.parse(PROGRAM.read_text())
    names = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert names, "no import was read: the check would pass on anything"
    assert not {n for n in names if n.split(".")[0] in ("httpx2", "anthropic", "openai")}
    assert not {n for n in names if n.startswith("wsp_worker.miwsp_engine")}


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
    ``runs.create_rerun``), so an engine program's rerun is not mislabelled ``weightspace``. And
    since 018-P4 the ``weightspace`` case is a program stored before the retirement: a rerun runs
    only miForge's own sandbox, so it still runs, and keeps the provenance of the run that made
    it."""
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


# -------------------------- data-free refusals (fix 2026-11-28, E7 §5.3 gap 2: a 500-char channel)

#: A program that reads a practice activation and raises with it (and a canary) in the message.
LEAKY = """
def edit(base, support):
    t = support.tokens(int(support.rows[0]))
    raise ValueError("leak 0.123456789 " + str(float(t[0, 0])))
"""


def _once(fx: FixtureBundle, source: str, *, data_free: bool = True) -> ProgramError:
    b = _bundle(fx)
    support = program.SupportSet("practice_train", b, data_free=data_free)
    with pytest.raises(ProgramError) as caught:
        program.run_once(source, b, support, _request(fx), data_free=data_free)
    return caught.value


def test_a_programs_exception_message_never_reaches_the_refusal(
    fx: FixtureBundle, capsys: pytest.CaptureFixture[str]
) -> None:
    b = _bundle(fx)
    support = program.SupportSet("practice_train", b)
    value = str(float(support.tokens(int(support.rows[0]))[0, 0]))
    error = _once(fx, LEAKY)
    assert str(error) == program.PROGRAM_RAISED.format(kind="ValueError")
    assert "0.123456789" not in str(error) and value not in str(error)
    assert error.code == "wsp_program_invalid"
    # The full text is the worker child's stderr only (the run's stored logs/stderr.txt).
    logged = capsys.readouterr().err
    assert "0.123456789" in logged and value in logged


@pytest.mark.parametrize(
    ("source", "canary"),
    [
        # A ProgramError the program built itself, with its own message and code.
        (
            "def edit(base, support):\n"
            "    try:\n"
            "        support.tokens(-1)\n"
            "    except Exception as e:\n"
            "        raise e.__class__('leak 0.987654321', 'leak_code_31337')\n",
            "987654321",
        ),
        # A class whose name the program computed (``__build_class__`` is a builtin it has).
        (
            "def edit(base, support):\n"
            "    E = __build_class__(lambda: None, 'leak_' + str(55555), ValueError)\n"
            "    raise E('x')\n",
            "55555",
        ),
        # A returned key that is not an editable tensor.
        ("def edit(base, support):\n    return {'leak_4242': np.zeros(1)}\n", "4242"),
        # A returned value of a type the program named.
        (
            "def edit(base, support):\n"
            "    C = __build_class__(lambda: None, 'leak_' + str(777))\n"
            "    return {'weight': C()}\n",
            "leak_777",
        ),
        # A row id carrying a value.
        ("def edit(base, support):\n    return support.tokens(-31415)\n", "31415"),
    ],
    ids=["own-program-error", "computed-class-name", "returned-key", "returned-type", "row-id"],
)
def test_no_refusal_echoes_a_value_the_program_chose(
    fx: FixtureBundle, source: str, canary: str
) -> None:
    error = _once(fx, source)
    assert canary not in str(error), str(error)
    assert error.code == "wsp_program_invalid"


def test_a_builtin_or_numpy_class_is_named_and_a_borrowed_name_is_not() -> None:
    import numpy as np

    assert program.exception_kind(KeyError("x")) == "KeyError"
    assert program.exception_kind(np.linalg.LinAlgError("x")) == "numpy.linalg.LinAlgError"

    class FakeError(Exception):
        pass

    FakeError.__name__, FakeError.__module__ = "LinAlgError", "numpy.linalg"
    assert program.exception_kind(FakeError("x")) == "Exception"


def test_a_replay_of_an_older_run_keeps_its_texts(fx: FixtureBundle) -> None:
    """``data_free`` off (a replay whose header has no ``program_errors``): the texts the run was
    recorded with, so its tool outputs replay bit for bit."""
    error = _once(fx, LEAKY, data_free=False)
    assert str(error).startswith("the program raised ValueError: leak 0.123456789 ")
    row = _once(
        fx, "def edit(base, support):\n    return support.tokens(-31415)\n", data_free=False
    )
    assert str(row) == "row -31415 is not in this support set"


def test_a_support_sets_own_refusal_names_no_row_and_passes_through_as_written(
    fx: FixtureBundle,
) -> None:
    """The support set's refusal is itself data-free (control B-9: the run-time allowlist alone
    masked this line), and a fixed text reaches the editor as written, so a program asking for
    a row outside its set is told why."""
    b = _bundle(fx)
    with pytest.raises(ProgramError) as caught:
        program.SupportSet("practice_train", b).tokens(-31415)
    assert str(caught.value) == program.ROW_NOT_IN_SUPPORT
    error = _once(fx, "def edit(base, support):\n    return support.tokens(-31415)\n")
    assert str(error) == program.ROW_NOT_IN_SUPPORT


# --------------- review of train F-1 (2026-11-28): eight more channels, each in plain Python

#: Each program carries the canary 31337 (a stand-in for a practice value) toward the editor.
CANARY = "31337"
CHANNELS = {
    # A class planted under a numpy name: the kind was read from numpy's namespace at call time.
    "numpy-planted-name": """
def edit(base, support):
    E = __build_class__(lambda: None, 'leak_31337', Exception)
    E.__module__ = 'numpy'
    np.__dict__['leak_31337'] = E
    raise E('x')
""",
    # The same under a builtin name, through numpy's own __builtins__.
    "builtins-planted-name": """
def edit(base, support):
    E = __build_class__(lambda: None, 'leak_31337', Exception)
    b = np.__builtins__
    (b if isinstance(b, dict) else b.__dict__)['leak_31337'] = E
    raise E('x')
""",
    # An exception whose __str__ raises: the guard's own str() call raised it out of the handler.
    "str-raises": """
def edit(base, support):
    class E(Exception):
        def __str__(self):
            raise ValueError('leak_31337')
    raise E()
""",
    # A class whose __module__ is an object with its own startswith.
    "module-object": """
def edit(base, support):
    class M:
        def startswith(self, prefix):
            raise ValueError('leak_31337')
    class E(Exception):
        pass
    E.__module__ = M()
    raise E('x')
""",
    # A mapping whose items() raises: the read was outside the guard, and the toolbox quoted it.
    "items-raises": """
def edit(base, support):
    class D(dict):
        def items(self):
            raise ValueError('leak_31337')
    return D()
""",
    # The same with a ProgramError the program took from the support set.
    "items-raises-program-error": """
def edit(base, support):
    try:
        support.tokens(-1)
    except Exception as e:
        PE = e.__class__
    class D(dict):
        def items(self):
            raise PE('leak_31337')
    return D()
""",
    # A str subclass key whose __repr__ the refusal quoted (two refusals quote the name).
    "key-repr-type": """
def edit(base, support):
    class K(str):
        def __repr__(self):
            return 'leak_31337'
    return {K('weight'): 5}
""",
    "key-repr-shape": """
def edit(base, support):
    class K(str):
        def __repr__(self):
            return 'leak_31337'
    return {K('weight'): np.zeros(1, dtype=np.float32)}
""",
    # An array subclass whose ufuncs raise: np.isfinite ran it outside the guard.
    "array-subclass": """
def edit(base, support):
    class A(np.ndarray):
        def __array_ufunc__(self, *args, **kwargs):
            raise ValueError('leak_31337')
    return {'weight': np.array(base['weight']).view(A)}
""",
    # A metaclass whose __mro__ and __name__ raise: read by the kind and by the log line.
    "metaclass": """
def edit(base, support):
    class Raises:  # a data descriptor (``property`` is not among a program's builtins)
        def __get__(self, obj, owner=None):
            raise ValueError('leak_31337')
        def __set__(self, obj, value):
            pass
    class Meta(Exception.__class__):
        __mro__ = Raises()
        __name__ = Raises()
    class E(Exception, metaclass=Meta):
        pass
    raise E('x')
""",
    # A ProgramError whose argument is a str subclass posing as a fixed text (round 2).
    "fixed-text-impostor": """
def edit(base, support):
    try:
        support.tokens(-1)
    except Exception as e:
        PE = e.__class__
    class K(str):
        def __str__(self):
            return 'a row that is not in this support set'
    raise PE(K('leak_31337'))
""",
    # A BaseException subclass: caught by nothing, it left the tool and the session.
    "base-exception": """
def edit(base, support):
    class E(Exception.__base__):
        pass
    raise E('leak_31337')
""",
}


#: The channels whose program is a valid edit once its return is read as plain values.
ACCEPTED = frozenset({"array-subclass"})


def _program_tools(fx: FixtureBundle) -> Any:
    from wsp_worker.program_tools import ProgramTools

    request = {**_request(fx), "provider": {"id": "miwsp_engine", "version": "t"}}
    return ProgramTools(
        _bundle(fx), request, "wsp_t", created_by={"provider": "p", "provider_version": "v"}
    )


@pytest.mark.parametrize("name", sorted(CHANNELS))
def test_no_channel_carries_the_programs_text_to_the_editor(fx: FixtureBundle, name: str) -> None:
    """Through the editor's own path: ``test_program`` as the session's toolbox calls it (the
    toolbox quotes any exception that escapes a tool, so escaping is a channel too)."""
    from wsp_worker.miwsp_engine import tools as tools_mod
    from wsp_worker.miwsp_engine.session import program_tools_text

    tools = _program_tools(fx)
    parameters = tools_mod.closed(program_tools_text()[2])  # as the session declares it
    spec = tools_mod.ToolSpec("test_program", "t", parameters, tools.test_program, None)
    out = tools_mod.Toolbox([spec], tools_mod.ForwardBudget(10)).call(
        "test_program", {"code": CHANNELS[name]}
    )
    assert CANARY not in out, out
    submitted = tools.submit_program(CHANNELS[name])
    assert CANARY not in submitted, submitted
    if name in ACCEPTED:  # read as its plain array, so the edit itself stands
        assert '"auroc"' in out and not submitted.startswith("refused: "), (out, submitted)
    else:
        assert '"refused": "wsp_program_invalid"' in out, out
        assert submitted.startswith("refused: "), submitted


def test_a_str_subclass_key_is_read_as_its_plain_name(fx: FixtureBundle) -> None:
    """A ``str`` subclass naming an editable tensor (``np.str_``, say) is kept as an exact copy,
    so an honest program still edits it."""
    source = """
def edit(base, support):
    w = np.array(base["weight"]) * np.float32(2)
    return {np.array(["weight"])[0]: w}
"""
    b = _bundle(fx)
    out = program.run_once(source, b, program.SupportSet("practice_train", b), _request(fx))
    assert np.array_equal(out["weight"].values(), np.array(b.tensors["weight"].values()) * 2)


def test_the_known_kinds_are_recorded_before_any_program_runs() -> None:
    """Renaming numpy's class after load changes nothing: the name is the one recorded then."""
    import numpy as np

    error = np.linalg.LinAlgError("x")
    saved = np.linalg.LinAlgError.__name__
    try:
        np.linalg.LinAlgError.__name__ = "leak_31337"
        assert program.exception_kind(error) == "numpy.linalg.LinAlgError"
    finally:
        np.linalg.LinAlgError.__name__ = saved


def test_the_after_session_run_is_data_free_too(fx: FixtureBundle) -> None:
    """``candidate()`` (after the session, the stored program once more): its failure becomes
    the run's ``failure.message``, served by ``forge_wsp_get_run``. A program that raises only on
    that third run (it counts on the support set, which every run of the session shares) must
    not put its text there either."""
    tools = _program_tools(fx)
    source = """
def edit(base, support):
    try:
        support.counted += 1
    except Exception:  # AttributeError is not among a program's builtins
        support.counted = 1
    if support.counted == 3:
        raise ValueError('leak_31337')
    return {"weight": np.array(base["weight"])}
"""
    assert tools.submit_program(source) == "program stored; call submit to finish."
    with pytest.raises(ProgramError) as caught:
        tools.candidate()
    assert CANARY not in str(caught.value)
    assert str(caught.value) == program.PROGRAM_RAISED.format(kind="ValueError")


def test_only_the_workers_own_refusal_passes_as_a_fixed_text(fx: FixtureBundle) -> None:
    """A program's own exception carrying a fixed text is still the program's exception: it reads
    as what it is, never as the worker's refusal (round 2 of the review)."""
    error = _once(fx, f"def edit(base, support):\n    raise ValueError({program.NO_EDIT!r})\n")
    assert str(error) == program.PROGRAM_RAISED.format(kind="ValueError")
    error = _once(fx, "def edit(base, support):\n    return support.tokens(-1)\n")
    assert str(error) == program.ROW_NOT_IN_SUPPORT
