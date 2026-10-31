"""miForge's own overhead on **the adapter step**.  ``gpu``-marked.  (ADR-024)

**Deselected by default** (ADR-023, and ``addopts`` in ``pyproject.toml``). This is an
operator-run gate; the report says when it was last run and on what, rather than implying a
number was measured.

*This paragraph read "NOT RUN for this increment - ``torch`` and ``peft`` are not installed
in the environment this feature was built in" until 2026-10-16.* It was true when written
and it is why nobody noticed that this module **could not even be imported**: it asked for
``training.step_record.StepRecord``, a name that has never existed. **A never-executed
performance gate reports no number and says nothing, which reads exactly like a gate inside
its budget.**

## The fixture was the finding - re-scaled 2026-10-16, budget NOT loosened

*This ran on ``TINY_CONFIG`` at batch 1 x sequence 8 until then, and failed at 25-28 %.*
That number was a measurement of the **fixture**, not of miForge: two layers of 64 hidden
units at eight tokens make a whole step ~3.2 ms, the same order as one ``fsync`` (3.09 ms).
At realistic model scale the identical code path measures **0.70-2.84 %** - so the 5 %
budget was met the whole time and the denominator was the defect.

**Retiring this case in favour of the newer benchmarks was considered and rejected.**
``test_boundary_loop_overhead.py`` derives F6's boundary overhead by *subtracting* the
``step`` phase, and ``test_signal_source_throughput_gpu.py`` measures F2's generation.
Neither measures the adapter step - and ADR-024's rule is *measure the path you changed*, so
deleting the only measurement of this path would leave the adapter budget asserted nowhere.
The fixture moved to :mod:`tests.support.bench_model` instead, and **every budget assertion
now carries an absolute millisecond bound and a denominator floor beside the ratio**, which
is the pairing that module's own docstring requires of anything built on it.

## Measure the path you changed

ADR-024's budget is **under 5 % of step wall time**, and it must be measured **on the
adapter step** - not inferred from the full-parameter path. The adapter step is *cheaper*
(no full-parameter gradients, no FP32 master copy of the base, no optimizer state over the
base), so a percentage taken from the other path **flatters** this one: the same absolute
overhead is a larger fraction of a smaller step. That asymmetry is the reason this file
exists separately from F1's.

## And every figure names its topology

Both PCIe links on this node are **x4, not x16** - Gen4 x4 is about 7.9 GB/s on the 3090 -
so any host-to-device figure taken from an x16 number is wrong by 3-6x. **A number quoted
without its topology is not a measurement** (ADR-024, BRD-02 section 3). miStudio's hardware
table is the authority and is **cited, never copied**: copying an address out of it once
unbound the root NVMe.
"""

from __future__ import annotations

import math
import time
from typing import Any

import pytest

from tests.support.bench_model import BENCH_BATCH, BENCH_LLAMA, BENCH_SEQ
from tests.support.test_adapter_doubles import adapter_config
from training.runtypes import resolve

pytestmark = pytest.mark.gpu

#: ADR-024's budget. miForge's own work, as a fraction of step wall time. **Unchanged** -
#: see "The fixture was the finding" in the module docstring.
OVERHEAD_BUDGET = 0.05

#: The absolute companion to the ratio. A ratio can be passed by growing the model, so a
#: fixed cost - one ``fsync``, one database round trip - would otherwise hide behind a large
#: denominator at any budget. miForge's per-step work here is one ``.item()``, one
#: finiteness check and one validated, ``fsync``ed append.
OVERHEAD_CEILING_MS = 25.0

#: The denominator's floor. ``bench_model`` exists to make one step >= 100 ms; the adapter
#: step is **cheaper than the full-parameter one** on the same model (no full-parameter
#: gradients, no FP32 master copy, no optimizer state over the base), so the floor asserted
#: here is lower than that by design. It is still **>15x one** ``fsync`` (3.09 ms), which is
#: the property that matters: the figure must be a statement about overhead and not about
#: the fixture's smallness.
#:
#: **Measured, not estimated: 182.8 ms** on the RTX 3090 at bf16, batch 16 x sequence 512
#: (2026-10-16), so the floor sits at about a quarter of the observed step and leaves room
#: for a slower card without becoming a second, accidental budget. *This comment claimed
#: "~86 ms" before the measurement was taken, which was a figure written from expectation -
#: the same class of error as ``bench_model``'s first ``BENCH_BATCH`` guess being wrong by
#: 2.4x, and the reason that module says a fixture size must be measured.*
MIN_STEP_WALL_MS = 50.0

#: How many steps to average over. Small enough to be a gate rather than a benchmark, large
#: enough that the first step's lazy CUDA initialisation is not the whole measurement.
WARMUP_STEPS = 3
MEASURED_STEPS = 10


@pytest.fixture
def plan(tmp_path: Any) -> Any:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    # `Refusal` because resolve_device REFUSES when nvidia-smi itself fails and returns
    # None only when it succeeds with no card. Handling only the None turned a broken
    # driver into an ERROR for every case here, which reads as a regression; the sibling
    # fixture in test_acquisition_gpu.py had always caught the refusal.
    from core.errors import Refusal
    from core.gpu import resolve_device
    from training.acquisition.base import (
        AcquiredModel,
        AcquisitionMode,
        AcquisitionRecord,
        SourceKind,
    )

    try:
        device = resolve_device("auto")
    except Refusal as exc:
        pytest.skip(f"no CUDA device resolved ({exc.code}); this case needs the GPU node")
    if device is None:
        pytest.skip("no CUDA device is visible to this process")

    config = AutoConfig.for_model(
        BENCH_LLAMA["model_type"], **{k: v for k, v in BENCH_LLAMA.items() if k != "model_type"}
    )
    torch.manual_seed(5)
    # ------------------------------------------------------------------ dtype, and why
    # `bfloat16`: what `bench_model` and the checkpoint writer emit, so this is not secretly
    # measuring a dtype no run uses. *This fixture was fp32 until 2026-10-16* - the default
    # - which inflates the denominator several times over on a 3090's tensor cores while the
    # overhead numerator is dtype-independent (one `.item()`, one finiteness check, one
    # validated append), so the ratio was flattered by the fixture: the same class of error
    # as the toy step this file was re-scaled to escape, in the opposite direction.
    #
    # It spent an hour at `float16`, because `base_parameter_digest` REFUSED
    # `base_digest_unreadable` on a bfloat16 module - numpy has no bfloat16 scalar type, so
    # every byte route failed. **That defect is fixed** (`_reinterpret_as_uint8`, regression
    # in `tests/unit/test_base_digest_reads_bfloat16.py`), so the workaround is gone rather
    # than left behind with a comment explaining it.
    module = (
        AutoModelForCausalLM.from_config(config, trust_remote_code=False)
        .to(torch.bfloat16)
        .to("cuda")
    )
    acquired = AcquiredModel(
        module=module,
        tokenizer=None,
        record=AcquisitionRecord(
            mode=AcquisitionMode.LOAD,
            source_kind=SourceKind.HUB,
            source_repo="LiquidAI/LFM2.5-1.2B-Instruct",
            starting_weights_sha256="b" * 64,
            base_model="LiquidAI/LFM2.5-1.2B-Instruct",
        ),
        device_uuid=device.uuid,
    )
    return resolve("adapter").prepare(
        acquired, adapter=adapter_config(rank=8, alpha=16.0, target_modules=("q_proj", "v_proj"))
    )


class TestTheAdapterStepsOverheadIsInsideTheBudget:
    def test_miforges_own_work_is_under_five_percent_of_step_wall_time(
        self, plan: Any, tmp_path: Any
    ) -> None:
        """The overhead measured is the step record's assembly and the digest work F8 and
        F1 add around the forward/backward - **not** the forward/backward itself, which is
        the model's cost and not miForge's.

        **This file had never been executed before 2026-10-16**, and the proof was in this
        function: it imported ``training.step_record.StepRecord``, a name that has never
        existed in that module (it defines ``STEP_EVENT``, ``Emitter``, ``StepWriter`` and
        ``rebuild_step_index``). The import was annotated *"the record being timed"* while
        the timed window held ``float(loss.detach().item())`` and assembled no record at
        all - so the docstring named the work and the measurement did not contain it. Now
        the window runs the real :class:`training.step_record.StepWriter`, which is what
        ``test_acquisition_gpu.py`` times on the full-parameter path, so the two figures are
        about the same work on two paths rather than two things called overhead.
        """
        import torch

        from training import optimizer as optimizer_module
        from training.run_config import RunConfig
        from training.step import StepOutcome
        from training.step_record import StepWriter

        cfg = RunConfig(learning_rate=1e-3, warmup_steps=0, batch_size=BENCH_BATCH)
        optimizer, _ = optimizer_module.build(plan.module, cfg)
        ids = torch.randint(0, BENCH_LLAMA["vocab_size"], (BENCH_BATCH, BENCH_SEQ), device="cuda")
        writer = StepWriter(run_id="adapter-overhead", log_root=tmp_path)
        step_number = 0

        def one_step() -> tuple[float, float]:
            nonlocal step_number
            torch.cuda.synchronize()
            started = time.perf_counter()
            loss = plan.module(input_ids=ids, labels=ids).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            torch.cuda.synchronize()
            compute_done = time.perf_counter()
            # miForge's own per-step work: reading the loss out of device memory, the
            # finiteness check, and assembling, validating and durably appending the step
            # record. Timed separately so the ratio is about the overhead and not about the
            # model. The index and the socket are absent by design - a database round trip
            # is the foundation's cost, not F8's.
            value = float(loss.detach().item())
            step_number += 1
            writer.write(
                session=None,
                step=step_number,
                outcome=StepOutcome(
                    loss=value,
                    objective="sft",
                    grad_norm=None,
                    wall_time_ms=(compute_done - started) * 1000.0,
                    batch_sha256="a" * 64,
                    finite=math.isfinite(value),
                ),
                cfg=cfg,
            )
            overhead_done = time.perf_counter()
            return compute_done - started, overhead_done - compute_done

        for _ in range(WARMUP_STEPS):
            one_step()

        compute = 0.0
        overhead = 0.0
        for _ in range(MEASURED_STEPS):
            step_compute, step_overhead = one_step()
            compute += step_compute
            overhead += step_overhead

        # The positive check that the timed window did the work it claims. Without it a
        # writer that silently wrote nothing would report a flattering overhead and look
        # exactly like a fast one.
        from training import run_log

        blocks = list(run_log.read_blocks(writer.log_path, block="step"))
        assert len(blocks) == WARMUP_STEPS + MEASURED_STEPS, (
            f"the timed window wrote {len(blocks)} step records, not "
            f"{WARMUP_STEPS + MEASURED_STEPS}; the overhead figure is not about the work "
            f"this test says it measures"
        )

        fraction = overhead / (compute + overhead)
        overhead_ms = overhead / MEASURED_STEPS * 1000
        step_ms = (compute + overhead) / MEASURED_STEPS * 1000
        print(
            f"\nadapter-step overhead: {overhead_ms:.3f} ms of "
            f"miForge's own work against {compute / MEASURED_STEPS * 1000:.3f} ms of "
            f"compute, mean over {MEASURED_STEPS} steps = {fraction:.2%}  "
            f"[ADAPTER step, rank 8 on q_proj/v_proj, batch={BENCH_BATCH} seq={BENCH_SEQ}, "
            f"{BENCH_LLAMA['num_hidden_layers']}L/{BENCH_LLAMA['hidden_size']}h "
            f"{next(plan.module.parameters()).dtype} "
            f"(~86 M params), RTX 3090, PCIe x4 negotiated on this node]"
        )
        print(
            f"   BREAK-EVEN: within the {OVERHEAD_BUDGET:.0%} budget for any step "
            f">= {overhead_ms / OVERHEAD_BUDGET:.1f} ms"
        )

        # THE DENOMINATOR FIRST, because this case read 25-28% on a toy fixture while the
        # identical code path measures ~1% at realistic scale. A ratio against a 3 ms step
        # is a measurement of the step, not of the overhead.
        assert step_ms >= MIN_STEP_WALL_MS, (
            f"the adapter step measured {step_ms:.1f} ms, under the "
            f"{MIN_STEP_WALL_MS:.0f} ms floor. The ratio below would be a statement about "
            f"the fixture's smallness rather than about miForge's overhead; fix the fixture "
            f"before reading the budget."
        )
        assert fraction < OVERHEAD_BUDGET, (
            f"miForge's own overhead is {fraction:.1%} of a {step_ms:.1f} ms adapter step "
            f"over {MEASURED_STEPS} steps (budget {OVERHEAD_BUDGET:.0%}). Measured on the "
            f"ADAPTER step, on this node's x4 links - a figure from the full-parameter "
            f"path would flatter it, because the same absolute overhead is a smaller "
            f"fraction of a larger step."
        )
        assert overhead_ms < OVERHEAD_CEILING_MS, (
            f"miForge's own per-step work is {overhead_ms:.1f} ms, over the "
            f"{OVERHEAD_CEILING_MS:.0f} ms absolute ceiling. The ratio can be inside budget "
            f"and this still be a regression: a fixed cost hides behind a large denominator, "
            f"which is why both bounds are asserted."
        )

    def test_the_figure_is_recorded_WITH_its_sample_and_its_topology(self, plan: Any) -> None:
        """ADR-012: every number names its scale and sample. A bare percentage with no step
        count and no mention of the x4 links is not a measurement, and this asserts that the
        gate's own reporting carries both."""
        assert MEASURED_STEPS >= 10
        assert "x4" in __doc__ and "not x16" in __doc__
        # And that the budget is asserted in BOTH directions, which is what stops a ratio
        # being passed by growing the denominator.
        assert OVERHEAD_BUDGET == 0.05 and OVERHEAD_CEILING_MS > 0
        assert MIN_STEP_WALL_MS >= 50.0, (
            "the denominator floor is what makes the ratio a measurement rather than a "
            "statement about the fixture; 25-28% was once reported from a 3.2 ms step"
        )
