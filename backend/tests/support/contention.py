"""A real second CUDA context on the same card, for the **time-slicing** figures.

BRD-02 section 9, as amended 2026-10-13: GPU sharing on this cluster is **time-slicing, no
MPS**. AC-002.22 and F4 task 9.3 both require their numbers *under that contention*, and
F2's task says the figure *"cannot be simulated"* - which is right, and is why this is a
subprocess rather than a second stream.

## Why a separate process, and not a second stream or a bigger batch

Time-slicing multiplexes **CUDA contexts**, and a context belongs to a process. Two streams
inside one process share a context and are scheduled together by the driver, so kernels from
them can overlap - which is the *opposite* of what is being measured. Under time-slicing the
driver switches whole contexts and kernels from the two processes **serialise**. So a
same-process "load" would measure concurrency the real deployment does not have and would
flatter every figure taken with it.

Nothing in the suite had a second-process load before 2026-10-16, which is the mechanical
reason AC-002.22 and F4 9.3 had no figures rather than wrong ones. *Reporting no number was
the correct call; this is what it was waiting for.*

## What the load is, stated so a figure can name it (ADR-012, ADR-024)

A continuous stream of ``4096 x 4096`` bfloat16 matrix multiplies - dense, tensor-core work
with **no host-to-device traffic inside the loop**, because the operands are allocated once
and stay resident. That matters twice over: it keeps the PCIe link (**x4 negotiated** on this
node, slot ``width.max`` 16) out of the measurement, and it makes the load a pure *compute*
competitor, so a throughput drop is attributable to kernel serialisation rather than to bus
contention.

It holds roughly **0.4 GiB**, small enough not to disturb E-24's free-VRAM budget check on a
24 GiB card, and it is a *competitor*, not a benchmark: the figures that matter are the ones
measured in the foreground process while this runs.

**It is not a guarantee of contention, and the caller must check.** A subprocess that failed
to start, or that exited, produces a "contended" figure identical to the clean one - which
would read as *"time-slicing costs nothing"*, the most flattering possible wrong answer. So
:class:`GpuLoad` exposes :meth:`assert_running` and :attr:`iterations`, and every figure
taken under it should assert the load actually progressed. That is the positive check this
project requires of any measurement: not a reading of a result, but evidence the measurement
happened.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path
from types import TracebackType

#: The competing workload. Runs until killed, printing a progress counter the parent reads
#: as proof it got onto the card. ``-I`` is not used: this is our own source, not downloaded
#: data, and it needs the ambient torch install.
_LOAD_SOURCE = """
import sys, time
import torch

size = int(sys.argv[1])
device = torch.device("cuda")
# Allocated ONCE, outside the loop: the load must compete for compute, not for the x4 bus.
left = torch.randn(size, size, device=device, dtype=torch.bfloat16)
right = torch.randn(size, size, device=device, dtype=torch.bfloat16)
torch.cuda.synchronize()
print("READY", flush=True)
count = 0
while True:
    left @ right
    count += 1
    if count % 20 == 0:
        torch.cuda.synchronize()
        print(count, flush=True)
"""

MATRIX = 4096


class GpuLoad:
    """A second process doing continuous dense matmuls on the same card.

    Use as a context manager. ``__enter__`` waits for the child to report ``READY`` - i.e.
    for its context to exist and its operands to be resident - so the foreground
    measurement does not start during the child's warm-up and attribute the quiet period to
    contention.
    """

    def __init__(self, *, matrix: int = MATRIX, startup_timeout_s: float = 120.0) -> None:
        self._matrix = matrix
        self._timeout = startup_timeout_s
        self._process: subprocess.Popen[str] | None = None
        self._script: Path | None = None
        self.iterations = 0

    def __enter__(self) -> GpuLoad:
        import tempfile

        directory = Path(tempfile.mkdtemp(prefix="mf-gpuload-"))
        self._script = directory / "load.py"
        self._script.write_text(_LOAD_SOURCE)
        self._process = subprocess.Popen(
            [sys.executable, str(self._script), str(self._matrix)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        deadline = time.monotonic() + self._timeout
        assert self._process.stdout is not None
        while time.monotonic() < deadline:
            line = self._process.stdout.readline()
            if line.strip() == "READY":
                return self
            if self._process.poll() is not None:
                stderr = self._process.stderr.read() if self._process.stderr else ""
                raise RuntimeError(f"the contending load died before READY:\n{stderr}")
        raise RuntimeError(f"the contending load did not report READY within {self._timeout}s")

    def assert_running(self) -> None:
        """Positive check: the load is **still** on the card and **has progressed**.

        Called after a measurement, not before. A load that started and then died gives a
        "contended" figure equal to the clean one, which reads as *"time-slicing is free"* -
        so a contention figure taken without this check is not evidence of anything.
        """
        assert self._process is not None, "the load was never started"
        assert self._process.poll() is None, (
            f"the contending load exited (code {self._process.returncode}) during the "
            f"measurement, so the figure just taken is an UNCONTENDED one wearing a "
            f"contended label"
        )
        # Drain whatever progress lines are buffered. The child prints every 20 iterations,
        # so at least one line having arrived is proof it ran kernels alongside us.
        assert self._process.stdout is not None
        import select

        while select.select([self._process.stdout], [], [], 0)[0]:
            line = self._process.stdout.readline()
            if not line:
                break
            if line.strip().isdigit():
                self.iterations = int(line.strip())
        assert self.iterations > 0, (
            "the contending load reported no completed iterations, so nothing is known to "
            "have competed for the card during the measurement"
        )

    def describe(self) -> str:
        """One line naming the load, for a figure's own report."""
        return (
            f"contending load: second PROCESS, {self._matrix}x{self._matrix} bf16 matmul "
            f"loop, {self.iterations} iterations observed, time-sliced (no MPS)"
        )

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if self._process is not None and self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=30)
        for stream in (self._process.stdout, self._process.stderr) if self._process else ():
            if stream is not None:
                stream.close()
        # The script's own temp directory. Removed here because an earlier version did not,
        # and an 18-figure pile of `mf-gpuload-*` directories in the pod's /tmp was the
        # trace. `missing_ok`/`ignore_errors` because cleanup must never be the thing that
        # fails a benchmark run - a leaked temp directory is untidy, a raise in __exit__
        # masks the figure.
        if self._script is not None:
            import shutil

            shutil.rmtree(self._script.parent, ignore_errors=True)
            self._script = None
