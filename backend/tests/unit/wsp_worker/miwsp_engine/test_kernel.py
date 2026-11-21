"""The kernel grandchild (feature 018; FTASKS 6.2, 6.4; FR-018.14; F18-D10).

The test process plays the session: it starts a REAL kernel (a fresh interpreter that installs
the child filter plus ``GRANDCHILD_RULES`` into itself) and drives it through ``Kernel``. The
filter can never land in this process. Skips by name without libseccomp; fails under
``MIFORGE_REQUIRE_WSP_WORKER=1``.
"""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from tests.support import wsp_worker_env
from wsp_worker import child_seccomp
from wsp_worker.miwsp_engine import prompts
from wsp_worker.miwsp_engine.kernel import KERNEL_ENV, Kernel

#: FTDD §6.5 / FTID §9: the kernel's whole environment, as a literal (like ``CHILD_ENV``).
EXPECTED_ENV = {
    "PATH": "/usr/local/bin:/usr/bin:/bin",
    "HOME": "/tmp",
    "PYTHONNOUSERSITE": "1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}


class Values:
    """The store's accepted values, as the session hands them to the kernel."""

    def __init__(self) -> None:
        self.v = {
            "weight": np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32),
            "bias": np.array([0.5], dtype=np.float64),
        }

    def __call__(self) -> dict[str, np.ndarray]:
        return self.v


def _kernel(tmp: Path, *, edit: bool, timeout_s: float = 30.0) -> tuple[Kernel, Values]:
    wsp_worker_env.require_seccomp()
    values = Values()
    kernel = Kernel(tmp / "kernel", values, edit=edit, timeout_s=timeout_s)
    kernel.start()
    return kernel, values


@pytest.fixture()
def edit_kernel(tmp_path: Path) -> Iterator[tuple[Kernel, Values]]:
    kernel, values = _kernel(tmp_path, edit=True)
    yield kernel, values
    kernel.close()


def test_the_kernel_env_is_the_literal() -> None:
    assert KERNEL_ENV == EXPECTED_ENV


def test_the_filter_is_stacked_one_above_this_process_and_the_env_is_exact(
    edit_kernel: tuple[Kernel, Values],
) -> None:
    kernel, _ = edit_kernel
    assert kernel.pid is not None
    theirs = child_seccomp.status_fields(kernel.pid)
    ours = child_seccomp.status_fields("self")
    assert theirs["NoNewPrivs"] == "1"
    assert int(theirs["Seccomp_filters"]) == int(ours.get("Seccomp_filters", "0")) + 1
    raw = Path(f"/proc/{kernel.pid}/environ").read_bytes().split(b"\0")
    seen = dict(x.decode().split("=", 1) for x in raw if x)
    seen.pop("LC_CTYPE", None)  # CPython's own locale coercion (PEP 538), never passed by us
    assert seen == EXPECTED_ENV


def test_exec_prints_the_last_expression_and_reports_w_deltas(
    edit_kernel: tuple[Kernel, Values],
) -> None:
    kernel, _ = edit_kernel
    first = kernel.exec("x = 41\nprint('hi')\nx + 1")
    assert first.output == "hi\n42\n" and first.deltas == {}
    wrote = kernel.exec("W['weight'][1] = 9.5")
    idx, vals = wrote.deltas["weight"]
    assert idx.tolist() == [1] and vals.tolist() == [9.5]
    assert "bias" not in wrote.deltas


def test_r_is_read_only_and_cannot_be_made_writable(edit_kernel: tuple[Kernel, Values]) -> None:
    kernel, _ = edit_kernel
    out = kernel.exec("a = R['weight']; a.flags.writeable = True").output
    assert "ValueError" in out
    assert "TypeError" in kernel.exec("R['weight'] = 0").output
    assert kernel.exec("float(R['weight'][0])").output.strip() == "1.0"


def test_an_inspect_kernel_has_no_w(tmp_path: Path) -> None:
    kernel, _ = _kernel(tmp_path, edit=False)
    try:
        result = kernel.exec("W['weight'][0] = 5")
        assert "NameError" in result.output and result.deltas == {}
    finally:
        kernel.close()


def test_a_sync_remaps_after_the_parent_changed_the_values(
    edit_kernel: tuple[Kernel, Values],
) -> None:
    kernel, values = edit_kernel
    kernel.exec("W['weight'][0] = 7")  # a pending, unaccepted write
    kernel.sync(["weight"])  # refused: the parent's values are unchanged, W reverts
    assert kernel.exec("float(W['weight'][0])").output.strip() == "1.0"
    values.v["weight"][2] = 30.0  # accepted elsewhere (a write tool)
    kernel.sync(["weight"])
    assert kernel.exec("float(R['weight'][2]), float(W['weight'][2])").output.strip() == (
        "(30.0, 30.0)"
    )


def test_a_timeout_restarts_the_kernel_with_the_accepted_weights(tmp_path: Path) -> None:
    kernel, values = _kernel(tmp_path, edit=True, timeout_s=1.0)
    try:
        kernel.exec("kept = 1")
        values.v["weight"][3] = 44.0  # accepted before the timeout
        first = kernel.pid
        hung = kernel.exec("while True:\n    pass")
        assert hung.restarted and hung.output == prompts.KERNEL_RESTARTED
        assert kernel.restarts == 1 and kernel.pid != first
        assert "NameError" in kernel.exec("kept").output  # variables are gone
        assert kernel.exec("float(R['weight'][3])").output.strip() == "44.0"
    finally:
        kernel.close()


def test_a_kernel_that_dies_is_restarted(edit_kernel: tuple[Kernel, Values]) -> None:
    kernel, _ = edit_kernel
    died = kernel.exec("import os; os._exit(3)")
    assert died.restarted and kernel.restarts == 1
    assert kernel.exec("1 + 1").output.strip() == "2"


def test_editor_writes_to_fd_1_never_reach_the_protocol(
    edit_kernel: tuple[Kernel, Values],
) -> None:
    kernel, _ = edit_kernel
    kernel.exec('import os; os.write(1, b\'{"ok": true, "deltas": []}\\n\')')
    assert kernel.restarts == 0
    assert kernel.exec("2 * 3").output.strip() == "6"


def test_no_socket_can_be_opened_in_the_kernel(edit_kernel: tuple[Kernel, Values]) -> None:
    kernel, _ = edit_kernel
    for code in (
        "import socket; socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
        "import socket; socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)",
        "import socket; socket.socketpair()",
    ):
        assert "PermissionError" in kernel.exec(code).output, code


def test_a_hostile_answer_is_checked_before_the_store_sees_it(
    edit_kernel: tuple[Kernel, Values],
) -> None:
    kernel, _ = edit_kernel
    hostile = {
        "deltas": [
            {"name": "weight", "idx": [99], "values": [1.0]},  # out of range
            {"name": "mean", "idx": [0], "values": [1.0]},  # not a tensor it was given
            {"name": "bias", "idx": [0, 0], "values": [1.0]},  # lengths disagree
            {"name": "weight", "idx": [True], "values": [1.0]},  # a bool is not an index
        ]
    }
    assert kernel._deltas(hostile) == {}


def test_a_kernel_that_skips_its_filter_is_refused_before_anything_is_sent(tmp_path: Path) -> None:
    """The parent's check, as the supervisor's on the child: a ready byte from a process with no
    stacked filter refuses ``worker_kernel_filter_missing`` and the process is killed."""
    import stat
    import sys

    from wsp_worker.miwsp_engine.kernel import KernelError

    wsp_worker_env.require_seccomp()
    fake = tmp_path / "python-no-filter"
    fake.write_text(
        "#!/bin/sh\n"
        f'exec "{sys.executable}" -I -c "import os, time; os.write(1, b\'K\'); time.sleep(60)"\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IXUSR)
    kernel = Kernel(tmp_path / "kernel", Values(), edit=True, python=str(fake))
    with pytest.raises(KernelError) as refused:
        kernel.start()
    assert refused.value.code == "worker_kernel_filter_missing"
    assert kernel.proc is None  # killed and reaped
