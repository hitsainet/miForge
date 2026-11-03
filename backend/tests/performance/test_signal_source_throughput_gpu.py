"""F2's three figures, each on its own path.  ``gpu``-marked.  (AC-002.21, AC-002.22)

F2 task 14.5, ADR-024, BRD-02 section 9. **No throughput test existed anywhere** before
2026-10-16, and F2's own task says why, correctly:

> **NOT PRODUCED, and F2 would not trust the figures if it had.** No GPU, no ``torch``, no
> real model in this venv. *An F2-overhead figure measured with a fake tensor library is a
> number from a path F2 does not run in production*, and a generation-throughput figure
> needs the card **under time-slicing contention** (BRD section 9), which cannot be
> simulated.

That was the right call: a figure from the fake-torch path would have been a measurement of
``signal_fixtures.FakeTorch``. What it was waiting for was a card and a real second CUDA
context, and both exist now - the context in ``tests.support.contention``, which is a
**subprocess** for the reason that module's docstring gives.

## Three figures, and the rule is that none is quoted for another

AC-002.21 requires F2's own overhead *"measured on the assembly path, reported **separately**
from F3's selection cost and from generation throughput"*. That separation is the whole
requirement, so this file measures and reports three distinct things and never derives one
from another:

1. **assembly** - F2's own work per batch: part iteration, the masking policy, the batch
   record, the hashes, the batch-index write. Measured on the ``corpus`` source, which
   **generates nothing**, so the number cannot be contaminated by provider time. Reported
   as a fraction of step wall time (ADR-024's 5 % budget) with an absolute bound beside it.
2. **generation throughput** - completion tokens per second for ``self-generation`` and for
   ``reward``, through ADR-009's in-process provider on real weights.
3. **the same, under time-slicing contention** - the figure BRD section 9 asks for.

## Why ``corpus`` carries the assembly figure, and not ``self-generation``

On a generating source, F2's assembly and the provider's generation are inside one
``next()``. Subtracting one from the other to get "F2's own cost" would be a figure derived
from the very separation AC-002.21 demands be kept. ``corpus`` runs the identical assembly,
record and index path with the provider absent, so its ``next()`` **is** F2's own cost, with
nothing to subtract. *The requirement's "on its own path" is satisfiable this way and not
by arithmetic.*

## Sample and topology (ADR-012, ADR-024)

One **RTX 3090** on ``mcs-lnxhost02``; PCIe link **x4 negotiated**, slot ``width.max`` 16,
read from the driver rather than assumed. The model is resident for every loop here and the
contending load allocates once, so **no figure below includes a host-to-device transfer** -
which is exactly why the link width is worth naming: a reader who lifted one of these
numbers onto an x16 assumption would be wrong by 3-6x in a direction this file cannot warn
them about.

Run with::

    pytest -m gpu backend/tests/performance/test_signal_source_throughput_gpu.py -s
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from tests.support.bench_model import BENCH_BATCH, BENCH_LLAMA, BENCH_SEQ, bench_prompts
from tests.support.contention import GpuLoad

pytestmark = pytest.mark.gpu

#: ADR-024's budget for miForge's own work as a fraction of step wall time.
OVERHEAD_BUDGET = 0.05

#: The absolute companion to the ratio, for the reason stated in
#: ``test_boundary_loop_overhead.py``: a fixed cost passes any ratio at a large enough model
#: size, so a ratio alone is not an assertion about F2.
ASSEMBLY_CEILING_MS = 50.0

#: The step wall time F2's ratio is taken against. **Measured here, not borrowed.** A
#: denominator copied from another file's run is the "figure from a path you did not
#: exercise" ADR-024 forbids, and the two files could drift to different fixture sizes.
WARMUP = 2
MEASURED = 8

#: Completion length for the throughput figures. Short enough that a pass is a gate rather
#: than a benchmark, long enough that per-call fixed cost is not the whole measurement.
GEN_TOKENS = 32
GEN_PROMPTS = 8


# --------------------------------------------------------------------------- fixtures


@pytest.fixture
def device():
    from core.errors import Refusal
    from core.gpu import resolve_device

    try:
        return resolve_device("auto")
    except Refusal as exc:
        pytest.skip(f"no CUDA device resolved ({exc.code}); this case needs the GPU node")


@pytest.fixture
def f2_settings(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """F2's three must-be-explicit settings, supplied locally.

    The same three ``test_standalone_without_siblings.py`` sets. F2 **refuses** rather than
    defaulting a masking policy - *"a getattr default that always fires is a fabricated
    value"* (ADR-018) - so a benchmark that omitted them would measure a refusal.
    """
    from training.signals.settings import reset_signal_settings

    monkeypatch.setenv("SIGNAL_SFT_MASKING_POLICY", "completion_only")
    monkeypatch.setenv("SIGNAL_DPO_REFERENCE_SOURCE", "precomputed")
    monkeypatch.setenv("SIGNAL_PENDING_SHORTFALL", "short_batch")
    reset_signal_settings()
    yield
    reset_signal_settings()


@pytest.fixture
def real_provider(tmp_path: Path, device):
    """ADR-009's in-process provider over **real weights on the card**.

    This is the one substitution that makes this file's numbers mean anything:
    ``signal_fixtures.in_process_provider`` wraps ``FakeModelHandle``, whose ``generate``
    returns a constant, so a throughput figure taken through it measures a Python function
    call. Here the handle is ``AcquiredModelHandle`` over a real acquisition.
    """
    import core.config as config
    from core.config import Settings, reset_settings
    from providers.in_process import InProcessProvider
    from providers.model_handle import AcquiredModelHandle
    from tests.support.bench_model import write_bench_model
    from training import tasks
    from training.acquisition.base import AcquisitionSpec

    checkpoints = tmp_path / "checkpoints"
    parts = tmp_path / "parts"
    for directory in (checkpoints, parts):
        directory.mkdir(parents=True, exist_ok=True)
    original = config._settings
    config._settings = Settings(  # type: ignore[call-arg]
        _env_file=None, checkpoint_root=checkpoints, parts_root=parts
    )
    try:
        written = write_bench_model(checkpoints / "bench")
        acquired = tasks.acquire_for_run(AcquisitionSpec(path=written, gpu_request=device.uuid))
        # batch_bound high enough that GEN_PROMPTS goes through in one call: the figure is
        # about generation, and a default bound of 8 would silently chunk a larger sample
        # and fold the chunk boundary into the number.
        yield InProcessProvider(AcquiredModelHandle(acquired), batch_bound=64)
    finally:
        config._settings = original
        reset_settings()


# --------------------------------------------------------------------------- the step


def _measure_step_ms() -> float:
    """One real forward+backward+optimizer step, median of :data:`MEASURED`.

    The denominator for figure 1. Measured in this file rather than imported as a constant,
    because a budget whose denominator is a literal is a budget that stops tracking the
    machine it runs on.
    """
    import torch

    from tests.support.bench_model import bench_model

    module = bench_model().to("cuda")
    optimizer = torch.optim.AdamW(module.parameters(), lr=1e-4)
    input_ids = torch.randint(0, BENCH_LLAMA["vocab_size"], (BENCH_BATCH, BENCH_SEQ), device="cuda")
    samples: list[float] = []
    for index in range(WARMUP + MEASURED):
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        output = module(input_ids=input_ids, labels=input_ids)
        output.loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        elapsed = (time.perf_counter() - started) * 1000.0
        if index >= WARMUP:
            samples.append(elapsed)
    del module, optimizer, input_ids
    torch.cuda.empty_cache()
    return statistics.median(samples)


# --------------------------------------------------------------------------- figure 1


class TestFigureOneF2sOwnAssemblyOverhead:
    """AC-002.21, on the assembly path, with no generation in the number."""

    def test_assembly_overhead_against_step_wall_time(
        self, f2_settings: None, tmp_path: Path, device
    ) -> None:
        from tests.support import db as support_db
        from tests.support import signal_fixtures as fixtures
        from training.signals import ensure_registered, registry
        from training.signals.base import PartSpec, RunContext, SourceSpec

        step_ms = _measure_step_ms()
        engine = support_db.migrated_engine(tmp_path / "db")
        samples: list[float] = []

        with support_db.session_for(engine) as session:
            # A corpus large enough that every measured batch draws fresh rows rather than
            # re-reading one cached page.
            part = fixtures.corpus_part(
                session, tmp_path, texts=[f"document {index}" for index in range(512)]
            )
            ensure_registered()
            source = registry.get("corpus")(
                RunContext(
                    run_id="bench-f2-assembly",
                    session=session,
                    log_root=tmp_path / "runs",
                    current_step=lambda: 1,
                )
            )
            source.preflight(
                SourceSpec(
                    source="corpus",
                    parts=(PartSpec(kind="corpus", part_id=str(part["id"])),),
                    config_source="inline",
                    batch_size=8,
                )
            )
            for index in range(WARMUP + MEASURED):
                started = time.perf_counter()
                batch = source.next()
                elapsed = (time.perf_counter() - started) * 1000.0
                if index >= WARMUP:
                    samples.append(elapsed)
                assert batch.record["batch_sha256"], (
                    "the measured path must include the record and its hash; a batch "
                    "without one means next() short-circuited and the figure is of nothing"
                )

        median = statistics.median(samples)
        ratio = median / step_ms
        print(
            f"\n── FIGURE 1: F2 assembly overhead ── n={len(samples)} batches of 8, "
            f"corpus source (NO generation in this number)"
        )
        print("   RTX 3090, PCIe x4 negotiated (slot x16); no host-to-device transfer in the loop")
        print(
            f"   step wall time      median {step_ms:8.2f} ms  "
            f"({BENCH_LLAMA['num_hidden_layers']}L/{BENCH_LLAMA['hidden_size']}h "
            f"batch {BENCH_BATCH} x seq {BENCH_SEQ} bf16)"
        )
        print(f"   F2 assembly         median {median:8.3f} ms  max {max(samples):.3f}")
        print(
            f"   ratio               median {ratio * 100:6.3f} %  "
            f"(budget {OVERHEAD_BUDGET * 100:.0f} %)"
        )
        print(f"   BREAK-EVEN: within budget for any step >= {median / OVERHEAD_BUDGET:.0f} ms")
        print(f"   per-batch ms: {[round(value, 3) for value in samples]}")

        assert (
            step_ms >= 100.0
        ), f"denominator is only {step_ms:.1f} ms; see bench_model's docstring"
        assert ratio < OVERHEAD_BUDGET, (
            f"F2's assembly is {ratio * 100:.2f} % of step wall time, over ADR-024's "
            f"{OVERHEAD_BUDGET * 100:.0f} %"
        )
        assert max(samples) < ASSEMBLY_CEILING_MS, (
            f"worst batch took {max(samples):.1f} ms, over the {ASSEMBLY_CEILING_MS:.0f} ms "
            f"absolute ceiling - a fixed cost hiding behind a large denominator"
        )


# --------------------------------------------------------------------------- figures 2 & 3


def _throughput(provider: Any, prompts: list[str]) -> dict[str, float]:
    """Completion tokens per second through ``provider``. Raw numbers, not a verdict."""
    from providers.base import GenerationParams

    params = GenerationParams(max_tokens=GEN_TOKENS, temperature=0.0)
    started = time.perf_counter()
    completions = provider.generate(prompts, params)
    elapsed = time.perf_counter() - started
    # Tokens COUNTED, not assumed to be GEN_TOKENS. Greedy generation stops at EOS, so
    # assuming the maximum would overstate throughput by however early the model stopped -
    # and on a randomly-initialised model that is often immediately.
    #
    # `.split()` IS the token count for this fixture and only for it: the tiny tokenizer is
    # word-level over `t4`..`t511` with a whitespace pre-tokenizer, so one token is one
    # whitespace-separated run. Stated because the same line against a BPE tokenizer would
    # undercount by a factor of several and the figure would be quietly wrong.
    tokens = sum(len(completion.text.split()) for completion in completions)
    return {
        "seconds": elapsed,
        "completions": float(len(completions)),
        "tokens": float(tokens),
        "tokens_per_second": tokens / elapsed if elapsed > 0 else 0.0,
        "completions_per_second": len(completions) / elapsed if elapsed > 0 else 0.0,
    }


def _report_throughput(
    label: str, sample: str, clean: dict[str, float], loaded: dict[str, float], load: GpuLoad
) -> None:
    """Print raw numbers for both figures side by side. **Raw, not just pass/fail.**"""
    print(f"\n── {label} ── RTX 3090, PCIe x4 negotiated (slot x16)")
    print(f"   sample: {sample}, greedy (temperature 0), in-process provider, real bf16 weights")
    print(
        f"   UNCONTENDED  {clean['tokens_per_second']:8.1f} tok/s  "
        f"{clean['completions_per_second']:6.2f} completions/s  "
        f"({int(clean['tokens'])} tokens in {clean['seconds']:.3f} s)"
    )
    print(
        f"   CONTENDED    {loaded['tokens_per_second']:8.1f} tok/s  "
        f"{loaded['completions_per_second']:6.2f} completions/s  "
        f"({int(loaded['tokens'])} tokens in {loaded['seconds']:.3f} s)"
    )
    kept = (
        loaded["tokens_per_second"] / clean["tokens_per_second"]
        if clean["tokens_per_second"]
        else 0.0
    )
    slowdown = loaded["seconds"] / clean["seconds"] if clean["seconds"] else 0.0
    print(
        f"   TIME-SLICING COST: {(1 - kept) * 100:5.1f} % of throughput lost, "
        f"{slowdown:.2f}x wall time"
    )
    print(f"   {load.describe()}")


def _clean_then_contended(provider: Any, prompts: list[str]) -> tuple[dict, dict, GpuLoad]:
    """One uncontended pass and one under a second CUDA context, in that order.

    Warmed up first: the first call pays lazy CUDA init and kernel autotuning, and charging
    that to the UNCONTENDED figure would make contention look free - or even negative, which
    is the shape of a wrong answer that reads as a finding.
    """
    _throughput(provider, prompts[: min(2, len(prompts))])
    clean = _throughput(provider, prompts)
    with GpuLoad() as load:
        loaded = _throughput(provider, prompts)
        # THE POSITIVE CHECK, inside the context manager so the load is still alive to be
        # checked. A load that died yields a contended figure equal to the clean one.
        load.assert_running()
    return clean, loaded, load


class TestFiguresTwoAndThreeGenerationThroughput:
    """AC-002.22: throughput for both generating shapes, **with its contention stated**.

    **Measured at the provider, deliberately.** AC-002.22 asks for *generation throughput*;
    ``source.next()`` is generation **plus** F3's selection **plus** F2's assembly, so a
    tok/s figure taken from it would fold in the two costs AC-002.21 requires reported
    separately - and figure 1 above is where F2's assembly is reported. The provider is
    where tokens are made, and it is the same provider object both generating sources reach
    the card through (``RunContext.provider``), which is what makes one measurement
    site legitimate for both shapes.

    **What differs between the two sources is the shape of the request, and that is
    measured.** ``self-generation`` asks for one completion per prompt; ``reward`` asks for
    a **group** per prompt (``SIGNAL_REWARD_GROUP_SIZE``, minimum 2 - a group of one has no
    relative advantage and is refused as ``reward_group_too_small``). So the second case
    sweeps the group size, which is also exactly BRD section 9's **OQ-7**: *"group size and
    batch size that fit the budget need a benchmark … benchmarked under time-slicing
    contention, not in isolation."*
    """

    def test_self_generation_shape_one_completion_per_prompt(
        self, f2_settings: None, real_provider: Any
    ) -> None:
        """FIGURE 2: ``self-generation``'s request shape - one completion per prompt."""
        prompts = bench_prompts(GEN_PROMPTS)
        clean, loaded, load = _clean_then_contended(real_provider, prompts)
        _report_throughput(
            "FIGURE 2: self-generation shape",
            f"{GEN_PROMPTS} prompts x 1 completion x <= {GEN_TOKENS} new tokens",
            clean,
            loaded,
            load,
        )
        _assert_plausible(clean, loaded)

    @pytest.mark.parametrize("group_size", [2, 4, 8])
    def test_reward_group_shapes_under_contention(
        self, f2_settings: None, real_provider: Any, group_size: int
    ) -> None:
        """FIGURE 3 and **OQ-7**: a reward group per prompt, swept, clean and time-sliced.

        A group of ``G`` is ``G`` completions for one prompt, so the request the provider
        sees is the prompt repeated ``G`` times - which is how ``reward.group`` assembles
        one group's members. The sweep is the measurement BRD section 9 asks for: it says
        what a group size *costs* on a shared card, which is the input to R-02.42's budget
        that OQ-7 says it bears on.
        """
        base = bench_prompts(GEN_PROMPTS)
        # One prompt repeated G times per group, G*prompts completions in the request. This
        # is the shape, not a re-implementation of `reward.group`: the grouping, statistics
        # and refusals are F2's and are tested on the CPU path.
        prompts = [prompt for prompt in base for _ in range(group_size)]
        clean, loaded, load = _clean_then_contended(real_provider, prompts)
        _report_throughput(
            f"FIGURE 3 / OQ-7: reward shape, group_size={group_size}",
            f"{GEN_PROMPTS} groups x {group_size} completions = {len(prompts)} completions "
            f"x <= {GEN_TOKENS} new tokens",
            clean,
            loaded,
            load,
        )
        _assert_plausible(clean, loaded)

    def test_the_contending_load_is_proved_to_reach_the_card(self) -> None:
        """The harness's own control, separate from any figure that uses it.

        Without it, every contention number in this file and in F4's rests on a subprocess
        nobody checked. Asserted on its own so a failure says *"the harness is broken"*
        rather than *"contention costs nothing"* - two findings the figure alone cannot
        tell apart.
        """
        with GpuLoad() as load:
            time.sleep(2.0)
            load.assert_running()
        assert load.iterations > 0
        print(f"\n── contention harness control ── {load.describe()}")


def _assert_plausible(clean: dict[str, float], loaded: dict[str, float]) -> None:
    """The two checks that stop an implausible figure being reported as a result."""
    assert clean["tokens"] > 0, (
        "no completion tokens were produced, so there is no throughput figure here. A "
        "randomly-initialised model can emit EOS immediately; that would need GEN_TOKENS "
        "raised or the figure reported as unobtainable from this fixture - not a number"
    )
    assert loaded["seconds"] >= clean["seconds"], (
        f"generation was FASTER under a competing CUDA context ({loaded['seconds']:.3f}s "
        f"vs {clean['seconds']:.3f}s). Time-slicing serialises kernels, so this is not a "
        f"plausible result: suspect the load never reached the card, or that the clean "
        f"figure still included warm-up"
    )
