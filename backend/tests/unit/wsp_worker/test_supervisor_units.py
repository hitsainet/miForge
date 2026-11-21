"""The supervisor's pure pieces, in this process (FTASKS 4.5; FTID §3.2; PF-60).

Nothing here starts a child, signals a process or empties a directory it did not create: the roots
are fresh temporary directories, and the refusal cases are checked by :func:`resolve_roots` before
any wipe could run.
"""

from __future__ import annotations

import errno
import os
import signal
from pathlib import Path

import pytest

from wsp_worker import spec, supervisor
from wsp_worker.supervisor import StartRefused, resolve_roots, wipe_roots


@pytest.fixture
def launcher_root(tmp_path: Path) -> Path:
    from wsp.launcher import ROOT_MARKER

    (tmp_path / ROOT_MARKER).write_text("x")
    return tmp_path


def test_the_supervisor_constants_are_pinned() -> None:
    """PF-60: module constants, not settings, so the pod's env list is unchanged."""
    assert supervisor.IDLE_RECYCLE_S == 3600
    assert supervisor.MAX_CONNECTIONS == 16
    assert supervisor.HEARTBEAT_STALE_S == 5
    assert supervisor.HANDLER_TIMEOUT_S == 10
    assert supervisor.RECEIVE_CHUNK_TIMEOUT_S == 300
    assert supervisor.READY_BYTE_TIMEOUT_S == 2.0


def test_the_server_settings_are_explicit() -> None:
    """R3A-2, R3A-9: reuse_address on, reuse_port off, daemon threads, a handler timeout."""
    assert supervisor.Server.allow_reuse_address is True
    assert supervisor.Server.allow_reuse_port is False
    assert supervisor.Server.daemon_threads is True
    assert supervisor.Handler.timeout == supervisor.HANDLER_TIMEOUT_S


def test_the_launcher_and_supervisor_agree_on_the_marker() -> None:
    from wsp.launcher import ROOT_MARKER

    assert ROOT_MARKER == supervisor.ROOT_MARKER


def test_child_env_is_the_literal_plus_mode() -> None:
    assert spec.child_env("run") == {
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp",
        "PYTHONNOUSERSITE": "1",
        "WSP_MODE": "run",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }


def test_the_supervisor_ignores_every_catchable_signal_but_sigchld() -> None:
    ignored = set(spec.supervisor_ignored())
    assert signal.SIGCHLD not in ignored and signal.SIGKILL not in ignored
    assert {signal.SIGINT, signal.SIGTERM, signal.SIGHUP, signal.SIGUSR1, signal.SIGALRM} <= ignored


# ----------------------------------------------------------------------------- roots (R3A-1)


@pytest.mark.parametrize("bad", [None, "", "relative/path", "/", "/tmp", "/dev/shm", "/tmp/"])
def test_a_bad_worker_root_is_refused(bad: str | None) -> None:
    with pytest.raises(StartRefused) as caught:
        resolve_roots(False, bad)
    assert caught.value.code == "worker_root_refused"


def test_a_root_without_the_launcher_marker_is_refused(tmp_path: Path) -> None:
    with pytest.raises(StartRefused) as caught:
        resolve_roots(False, str(tmp_path))
    assert caught.value.code == "worker_root_refused"


def test_a_root_whose_subdirectory_leaves_it_is_refused(
    launcher_root: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    outside = tmp_path_factory.mktemp("outside")
    (outside / "canary").write_text("keep")
    os.symlink(outside, launcher_root / "work")
    with pytest.raises(StartRefused) as caught:
        resolve_roots(False, str(launcher_root))
    assert caught.value.code == "worker_root_refused"
    assert (outside / "canary").read_text() == "keep"


def test_pid1_refuses_a_worker_root() -> None:
    with pytest.raises(StartRefused):
        resolve_roots(True, "/somewhere")
    assert {k: str(v) for k, v in resolve_roots(True, None).items()} == supervisor.PID1_ROOTS


def test_the_wipe_empties_each_root_and_never_the_root(launcher_root: Path) -> None:
    roots = resolve_roots(False, str(launcher_root))
    for name, root in roots.items():
        (root / f"{name}.txt").write_text("x")
        (root / "d" / "e").mkdir(parents=True)
        (root / "d" / "e" / "f").write_text("y")
        os.chmod(root / "d" / "e", 0o500)
        os.symlink("/etc/hostname", root / "link")
    assert wipe_roots(roots, pid1=False) == []
    assert all(root.is_dir() and not os.listdir(root) for root in roots.values())
    assert (launcher_root / supervisor.ROOT_MARKER).exists()


def _unremovable_tmp(launcher_root: Path, entries: dict[str, str]) -> dict[str, Path]:
    """A tmp root whose entries cannot be unlinked (as the kubelet's bind-mounted termination
    file cannot: ``EBUSY`` there, ``EACCES`` here, same branch)."""
    roots = resolve_roots(False, str(launcher_root))
    for name, text in entries.items():
        (roots["tmp"] / name).write_text(text)
    os.chmod(roots["tmp"], 0o500)
    return roots


def test_the_wipe_empties_an_unremovable_termination_message_and_accepts_it(
    launcher_root: Path,
) -> None:
    """FTASKS 4.4's finding: in the pod ``/tmp/termination-log`` is a bind mount the supervisor
    cannot unlink. The wipe truncates it and accepts it once empty; before, the start-up wipe
    refused ``worker_wipe_failed`` and the pod could never serve."""
    roots = _unremovable_tmp(launcher_root, {supervisor.TERMINATION_NAME: "child wrote this"})
    try:
        assert wipe_roots(roots, pid1=False) == []
        assert (roots["tmp"] / supervisor.TERMINATION_NAME).read_text() == ""
    finally:
        os.chmod(roots["tmp"], 0o700)


def test_any_other_unremovable_entry_is_still_reported(launcher_root: Path) -> None:
    roots = _unremovable_tmp(launcher_root, {"left-by-a-child": "x", "termination-log.bak": ""})
    try:
        left = wipe_roots(roots, pid1=False)
    finally:
        os.chmod(roots["tmp"], 0o700)
    assert sorted(Path(p).name for p in left) == ["left-by-a-child", "termination-log.bak"]


def test_a_termination_path_that_is_not_a_regular_file_is_reported(launcher_root: Path) -> None:
    roots = resolve_roots(False, str(launcher_root))
    os.symlink("/etc/hostname", roots["tmp"] / supervisor.TERMINATION_NAME)
    os.chmod(roots["tmp"], 0o500)
    try:
        left = wipe_roots(roots, pid1=False)
    finally:
        os.chmod(roots["tmp"], 0o700)
    assert [Path(p).name for p in left] == [supervisor.TERMINATION_NAME]


def test_the_test_hooks_are_named_and_refused_in_prod() -> None:
    os.environ["WSP_TEST_HOOKS"] = '{"idle_recycle_s": 2}'
    try:
        assert supervisor._test_hooks("ci") == {"idle_recycle_s": 2.0}
        with pytest.raises(StartRefused):
            supervisor._test_hooks("prod")
        os.environ["WSP_TEST_HOOKS"] = '{"idle_recycle": 2}'
        with pytest.raises(StartRefused):
            supervisor._test_hooks("ci")
    finally:
        del os.environ["WSP_TEST_HOOKS"]


def test_the_wipe_refuses_roots_outside_a_launcher_root(tmp_path: Path) -> None:
    """The second guard: even handed roots directly, it empties nothing unmarked."""
    (tmp_path / "work").mkdir()
    (tmp_path / "work" / "keep").write_text("keep")
    with pytest.raises(StartRefused):
        wipe_roots({"work": tmp_path / "work"}, pid1=False)
    assert (tmp_path / "work" / "keep").read_text() == "keep"


def test_ipc_wipe_refuses_outside_pid1() -> None:
    from wsp_worker import ipc_wipe

    assert os.getpid() != 1
    with pytest.raises(ipc_wipe.IpcWipeRefusedError):
        ipc_wipe.wipe_all()


def test_ipc_primitives_list_and_remove_a_segment_this_test_made() -> None:
    import ctypes

    from wsp_worker import ipc_wipe

    libc = ctypes.CDLL(None, use_errno=True)
    ident = libc.shmget(0, 4096, 0o1600)  # IPC_PRIVATE, IPC_CREAT | 0600
    if ident < 0:
        pytest.skip(f"shmget failed: {errno.errorcode.get(ctypes.get_errno())}")
    try:
        assert ident in ipc_wipe.list_sysv("shm")
        ipc_wipe.remove_sysv("shm", ident)
        assert ident not in ipc_wipe.list_sysv("shm")
    finally:
        libc.shmctl(ident, 0, None)


# ----------------------------------------------------------------------------- the push


def _push(**over: object) -> bytes:
    import json

    doc = {
        "boot_id": "a" * 32,
        "run_id": "wsr_x",
        "mode": "session",
        "spec": {"time_budget_s": 10, "grace_s": 1, "pull_timeout_s": 5},
        "files": [{"name": "heads.safetensors", "role": "heads", "sha256": "b" * 64, "bytes": 3}],
        "gateway": {"ip": "10.0.0.5", "port": 8080, "token": "t"},
    }
    doc.update(over)
    return json.dumps(doc).encode()


def test_a_well_formed_push_parses() -> None:
    push = spec.parse_push(_push())
    assert push.child_mode == "run" and push.lifetime_s == 16


@pytest.mark.parametrize(
    "over",
    [
        {"extra": 1},
        {"boot_id": "z" * 32},
        {"mode": "shell"},
        {"spec": {"time_budget_s": 10, "grace_s": 1}},
        {"spec": {"time_budget_s": -1, "grace_s": 1, "pull_timeout_s": 5}},
        {"files": [{"name": "../x", "role": "heads", "sha256": "b" * 64, "bytes": 1}]},
        {"files": [{"name": "a", "role": "heads", "sha256": "b" * 64, "bytes": 1}] * 2},
        {"gateway": {"ip": "miforge-gateway", "port": 8080, "token": "t"}},
        {"mode": "replay"},
        {"mode": "program_rerun"},
    ],
)
def test_a_malformed_push_is_refused(over: dict[str, object]) -> None:
    with pytest.raises(spec.SpecError):
        spec.parse_push(_push(**over))


def test_the_real_childs_interpreter_is_isolated(tmp_path: Path) -> None:
    """Review OR2-016-5: the REAL child (no test script) starts as ``python -I``: no
    ``PYTHON*`` variables, no user site, and neither the script's nor the working directory on
    ``sys.path``. Test children take ``child_argv``'s other branch, so this runs the production
    argv itself with only the boot string swapped for a probe of the interpreter's flags.
    Control ``S14-iso-isolated-flag`` (``-I`` dropped from the real branch) turns this red."""
    import subprocess
    import sys

    sup = supervisor.Supervisor(
        environment="dev", pid1=False, roots={}, test_script=None, hooks={}, pids_max=None
    )
    argv = sup.child_argv("session")
    at = argv.index("-c")
    assert argv[0] == sys.executable and argv[-1] == "session"
    assert argv[at + 1] == supervisor._CHILD_BOOT.format(root=str(supervisor.PACKAGE_ROOT))
    probe = "import sys; print(sys.flags.isolated, sys.flags.ignore_environment, repr(sys.path[0]))"
    (tmp_path / "sitecustomize.py").write_text("raise SystemExit('planted module ran')\n")
    out = subprocess.run(
        [*argv[: at + 1], probe, *argv[at + 2 :]],
        capture_output=True,
        text=True,
        check=True,
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path)},
    ).stdout.split()
    assert out[:2] == ["1", "1"], out
    assert out[2] not in ("''", repr(str(tmp_path))), out  # neither the cwd nor PYTHONPATH first


def _supervisor(tmp_path: Path) -> supervisor.Supervisor:
    roots = {k: tmp_path / k for k in ("work", "out", "tmp", "shm")}
    for root in roots.values():
        root.mkdir()
    return supervisor.Supervisor(
        environment="dev", pid1=False, roots=roots, test_script=None, hooks={}, pids_max=None
    )


def test_the_lifetime_runs_from_the_sessions_start_not_the_pushs(tmp_path: Path) -> None:
    """Fix 2026-11-21 (run wsp_cb092cedad97, a 301 s push): the lifetime (budget + grace + pull
    timeout, 16 s here) runs from ``running_mono`` once the session runs, the clock the time
    budget and the backend's deadline use, so a long push cannot end it before the backend has
    pulled the result. While the push is received it runs from acceptance."""
    sup = _supervisor(tmp_path)
    sup.push = spec.parse_push(_push())
    sup.state, sup.accepted_mono, sup.last_chunk_mono = "receiving", 0.0, 0.0
    assert sup._due(10.0) is None
    sup.last_chunk_mono = 16.0
    assert sup._due(16.5) == "lifetime"  # a push still arriving past the lifetime
    sup.exit_reason = None
    sup.state, sup.running_mono, sup.deadline_hit = "running", 100.0, True  # a 100 s push
    assert sup._due(110.0) is None  # 116.5 s after acceptance, 10 s into the session
    sup.state, sup.finished_mono = "finished", 111.0
    assert sup._due(115.0) is None
    assert sup._due(116.5) == "lifetime"


def test_the_time_left_is_the_budget_from_the_sessions_start(tmp_path: Path) -> None:
    """What the child line carries (``time_left_s``): the time budget (10 s) minus the time since
    ``running_mono``, the clock the watchdog kills on; never negative."""
    sup = _supervisor(tmp_path)
    assert sup.time_left(5.0) == 0.0  # no session
    sup.push = spec.parse_push(_push())
    sup.running_mono = 100.0
    assert sup.time_left(100.0) == 10.0
    assert sup.time_left(103.5) == 6.5
    assert sup.time_left(125.0) == 0.0


def test_the_child_line_carries_the_time_left_and_the_parse_requires_it() -> None:
    import json

    push = spec.parse_push(_push())
    paths = {"work": "/w", "out": "/o", "tmp": "/t"}
    line = spec.parse_child_line(spec.child_line(push, paths, 7.25))
    assert line["time_left_s"] == 7.25
    raw = json.loads(spec.child_line(push, paths, 7.25))
    for bad in ({"time_left_s": -1}, {"time_left_s": "7"}, {"time_left_s": True}):
        with pytest.raises(spec.SpecError):
            spec.parse_child_line(json.dumps({**raw, **bad}).encode())
    del raw["time_left_s"]
    with pytest.raises(spec.SpecError):
        spec.parse_child_line(json.dumps(raw).encode())
