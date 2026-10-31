"""The real acquire -> step -> record -> trainer-state cycle on a card.  ``gpu``-marked.

**Deselected by default** (ADR-023, and ``addopts`` in ``pyproject.toml``): a developer
without a GPU must be able to run everything except a training step, and **CI has no card**,
so these cases do not run there. That is stated rather than assumed green - they are an
operator-run gate at feature acceptance, and the report must say when they were last run and
on what.

Run them with::

    .venv/bin/pytest -m gpu backend/tests/integration/test_acquisition_gpu.py

**Only the RTX 3090 is present on this node.** The 3080 Ti is absent, so nothing below
encodes a device count; the device is resolved at runtime through ``core/gpu.py``, by UUID,
never by index or bus address (ADR-008).

A standing caution from AGENT.md, repeated because every case here depends on it: the node's
running NVIDIA driver already diverges from its installed packages and **may not come back
from its next reboot**. A failure here is as likely to be the node as the code; check
``nvidia-smi`` on the host before reading a red as a regression.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.tiny_hf_model import LLAMA_TINY, write_tiny_model, write_tiny_tokenizer

pytestmark = pytest.mark.gpu


def _assert_production_placed_it(acquired, device) -> None:
    """The acquired parameters are **already** on the leased card, by production code.

    *These three cases called ``acquired.module.to("cuda")`` here until 2026-10-16*, and
    that line was the only visible trace of the worst production defect in this repository:
    **nothing in ``backend/src/`` ever moved a model onto a GPU.** The submit path resolved a
    card by UUID and took its lease; ``from_pretrained`` and ``from_config`` both leave every
    tensor in host memory; ``objectives/sft.py`` builds its batch tensors on *the model's*
    device - so a training run held a card nobody else could use and computed on the CPU,
    raising nothing. The suite was green because the GPU tests hand-placed the model and the
    rest of the suite has no card.

    **A test hand-doing what production omits is a fixture that agrees with itself by
    construction**, which is this project's named reason a suite stays green over a real bug.
    So the hand-placement is now an assertion, and the UUID is asserted too: placing on
    *a* card is not the requirement - placing on the **leased** card is, because the lease
    is what makes one run per card true (ADR-008).
    """
    import torch

    landed = next(acquired.module.parameters()).device
    assert landed.type == "cuda", (
        f"production left the acquired parameters on {landed!r}. The run holds "
        f"{device.uuid}'s lease and would compute on the CPU - silently, because the "
        f"objective builds its tensors on the model's own device"
    )
    assert acquired.record.device_kind == "cuda"
    assert acquired.record.device_uuid == device.uuid, (
        f"the record names {acquired.record.device_uuid} and the lease is on {device.uuid}; "
        f"a score's provider provenance reads this field (R-02.16)"
    )
    assert acquired.device_uuid == device.uuid
    # The UUID the driver reports for the device the tensors are ACTUALLY on, matched
    # against the leased one. This is the inverted-enumeration check: `landed.index` is
    # this process's view and proves nothing on its own.
    actual = torch.cuda.get_device_properties(landed.index or 0).uuid
    from training.acquisition.placement import normalise_device_uuid

    assert normalise_device_uuid(str(actual)) == device.uuid, (
        "the tensors are on a card, and it is not the leased one - ADR-008's incident "
        "verbatim: nothing fails, the work just runs on the wrong device"
    )


@pytest.fixture
def device():
    """The card this run is pinned to, resolved live.

    Skipped rather than failed when no card answers: a red here would otherwise be
    indistinguishable from a real regression on a machine where the marker was selected by
    accident.
    """
    from core.errors import Refusal
    from core.gpu import resolve_device

    try:
        return resolve_device("auto")
    except Refusal as exc:
        pytest.skip(f"no CUDA device resolved ({exc.code}); this case needs the GPU node")


@pytest.fixture
def roots(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    from core.config import Settings, reset_settings

    states = tmp_path / "trainer-state"
    checkpoints = tmp_path / "checkpoints"
    for directory in (states, checkpoints):
        directory.mkdir()
    monkeypatch.setattr(
        "core.config._settings",
        Settings(  # type: ignore[call-arg]
            _env_file=None, trainer_state_root=states, checkpoint_root=checkpoints
        ),
    )
    yield tmp_path, states, checkpoints
    reset_settings()


@pytest.fixture
def given_tokenizer(roots) -> Path:
    """A **real** tokenizer, sized from ``LLAMA_TINY``, inside the configured checkpoint root.

    *The three ``create`` cases below passed no tokenizer reference until 2026-10-16, and on
    2026-10-16 they began failing with ``tokenizer_required``.* That refusal is the product
    being **fixed**, not broken: ``ScratchAcquirer.preflight`` now resolves a tokenizer at
    submit, because a created model with none ships a checkpoint that
    ``transformers``, vLLM, TGI and miLLM's loader can none of them load (R-02.51). The
    tests were the stale half.

    **Sized from the config rather than taken from a real 32k tokenizer**, which is the
    constraint ``write_tiny_tokenizer`` documents: ``LLAMA_TINY``'s ``vocab_size`` is 512, so
    a 32k vocabulary hands the model an id outside its embedding table. ``assert_vocab_agrees``
    now refuses that pairing by name at preflight (``tokenizer_vocab_mismatch``) rather than
    letting it reach ``IndexError`` at generation.

    **Inside the checkpoint root** because ``local_dir.resolve_within_roots`` confines a
    submitted path to a configured model root - a tokenizer in ``tmp_path`` is refused
    ``model_source_outside_root``, which is the confinement working.
    """
    _, _, checkpoints = roots
    directory = checkpoints / "given-tokenizer"
    directory.mkdir()
    write_tiny_tokenizer(directory, vocab_size=LLAMA_TINY["vocab_size"])
    return directory


class RealBatch:
    """A real tokenised batch for a real forward pass, with a real provenance record."""

    objective = "sft"

    def __init__(self, input_ids: Any) -> None:
        self.input_ids = input_ids
        self.record = {"batch_sha256": "f" * 64}


class RealSource:
    """One fixed batch, on the device. F2 owns the real sources."""

    def __init__(self, device: Any, vocab: int, batch: int, length: int) -> None:
        import torch

        self.batch = RealBatch(
            torch.randint(0, vocab, (batch, length), device=device, dtype=torch.long)
        )
        self.calls = 0

    def next(self) -> RealBatch:
        self.calls += 1
        return self.batch


class CrossEntropyFixture:
    """**A fixture objective, living in a test** (F1's FR-001.17 and task 9.1a).

    F1 defines no objective and implements no loss - the losses are F2's
    (``training/objectives/``). *"Write no cross-entropy there, not even as a placeholder: it
    would be a second definition of F2's contract and it is the one that would survive."* So
    the only cross-entropy in this feature is here, in a test file, where nothing can import
    it by accident.
    """

    name = "sft"

    def loss(self, model: Any, batch: RealBatch) -> Any:
        output = model(input_ids=batch.input_ids, labels=batch.input_ids)
        return output.loss


class FixtureRegistry:
    def __init__(self) -> None:
        self.objective = CrossEntropyFixture()
        self.lookups: list[str] = []

    def get(self, name: str) -> CrossEntropyFixture:
        self.lookups.append(name)
        return self.objective


# ⚠ TWO CASES MOVED OUT OF THIS FILE — 2026-10-16. `TestTheFixtureItselfIsLoadable` is now
# `tests/integration/test_tiny_model_fixture_is_loadable.py`, **unmarked**, and the writer it
# checks is `tests/support/tiny_hf_model.py` — the same one imported above, not a copy.
#
# Why: those two cases need no card, so living in a `gpu`-marked module meant `addopts`
# deselected them on every push and the fixture every case below depends on was unchecked in
# CI. A broken fixture produces no red of its own; it produces an error in every case that
# consumes it, and on a deselected module it produces nothing at all. See that file's header.


class TestALoadOnTheRealCard:
    """AC-001.1 on hardware."""

    def test_acquire_step_record_and_trainer_state_complete(self, device, roots) -> None:
        import torch

        from training import run_log, tasks, trainer_state
        from training.acquisition.base import AcquisitionSpec
        from training.run_config import RunConfig

        tmp_path, states, checkpoints = roots
        directory = write_tiny_model(checkpoints / "tiny")

        cfg = RunConfig(learning_rate=1e-4, warmup_steps=0, batch_size=2, gradient_accumulation=1)
        registry = FixtureRegistry()
        acquired = tasks.acquire_for_run(AcquisitionSpec(path=directory, gpu_request=device.uuid))
        _assert_production_placed_it(acquired, device)

        # The real optimizer, the real model, the real step. Only the batch source is a
        # fixture, and the objective is the one above - in a test file, by requirement.
        source = RealSource(torch.device("cuda"), LLAMA_TINY["vocab_size"], cfg.batch_size, 16)
        result = tasks.drive_run(
            run_id="gpu-load",
            spec=AcquisitionSpec(path=directory, gpu_request=device.uuid),
            cfg=cfg,
            batch_source=source,
            objectives=registry,
            session=None,
            log_root=tmp_path,
            trainer_state_root=states,
            max_steps=3,
            save_every=3,
        )

        assert result.steps_completed == 3
        assert registry.lookups == [
            "sft",
            "sft",
            "sft",
        ], "stage 2 must dispatch on the batch's objective on every step"
        log = run_log.run_log_path(tasks.run_log_root(tmp_path), "gpu-load")
        blocks = list(run_log.read_blocks(log, block="step"))
        assert [block["step"] for block in blocks] == [1, 2, 3]
        # A real gradient norm, from real gradients. On the fake path this is None, so this
        # is the one case that proves the norm is taken where the update is applied.
        assert all(block["gradient_norm"] is not None for block in blocks)
        assert all(block["loss"] > 0 for block in blocks)

        state = Path(result.trainer_states[0])
        assert trainer_state.is_complete(state)
        # **Derived, not restated.** This read ``== "000003"`` until 2026-10-16 - the bare
        # form that ``TrainerStateDir``'s docstring records as *the error*, corrected there
        # on 2026-10-14 to ``step-`` plus six digits and pinned on CPU by
        # ``test_trainer_state_layout.py``. This file is ``gpu``-marked, so CI deselects it
        # and the correction landed in every layer CI executes and not in this one. Taking
        # the name from the production renderer means the next rendering change breaks the
        # renderer's own test rather than drifting silently past this one.
        assert state.name == trainer_state.step_path("gpu-load", 3, root=states).name
        # And the derivation is not vacuous: a renderer that ignored the step would satisfy
        # the line above for any step at all.
        assert state.name != trainer_state.step_path("gpu-load", 2, root=states).name

    def test_the_record_names_the_path_and_a_stable_hash(self, device, roots) -> None:
        """AC-001.2's local analogue: the same directory twice records the same hash."""
        from training import tasks
        from training.acquisition.base import AcquisitionSpec

        _, _, checkpoints = roots
        directory = write_tiny_model(checkpoints / "tiny")
        spec = AcquisitionSpec(path=directory, gpu_request=device.uuid)
        first = tasks.acquire_for_run(spec).record
        second = tasks.acquire_for_run(spec).record
        assert first.starting_weights_sha256 == second.starting_weights_sha256
        assert first.source_path == str(directory.resolve())
        assert first.parent is None


class TestACreateOnTheRealCard:
    """AC-001.3 on hardware: instantiated **in miForge's own process**."""

    def test_a_creation_runs_and_its_lineage_is_parent_null(
        self, device, roots, given_tokenizer: Path
    ) -> None:
        from training import tasks
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec
        from training.acquisition.scratch import created_base_model_identity

        spec = AcquisitionSpec(
            mode=AcquisitionMode.CREATE,
            config=dict(LLAMA_TINY),
            seed=1234,
            gpu_request=device.uuid,
            tokenizer_path=given_tokenizer,
        )
        acquired = tasks.acquire_for_run(spec)
        record = acquired.record

        assert record.parent is None
        assert record.base_model == created_base_model_identity(LLAMA_TINY, 1234)
        assert record.seed == 1234
        # The tokenizer is ON the acquisition, not left for a caller to fill in. Asserted
        # here because this is the case that fails if the supply mechanism regresses: the
        # record's provenance field is the only written-down answer to which vocabulary a
        # created model was trained against (the manifest is closed and cannot carry it).
        assert acquired.tokenizer is not None, (
            "a create acquisition carries no tokenizer, so its first save point will raise "
            "AttributeError on `tokenizer.save_pretrained` and R-02.41 produces no artifact"
        )
        assert Path(record.tokenizer_source).resolve() == given_tokenizer.resolve()
        assert record.tokenizer_vocab_size == LLAMA_TINY["vocab_size"]
        # The realised count, against the config-derived estimate the budget guard used.
        from training.budget_guard import parameter_count_from_config

        estimated = parameter_count_from_config(LLAMA_TINY)
        assert record.param_count is not None
        ratio = record.param_count / estimated
        assert 0.5 < ratio < 2.0, (
            f"the config estimate ({estimated:,}) and the realised count "
            f"({record.param_count:,}) differ by {ratio:.2f}x - the estimator needs a look"
        )

    def test_the_same_seed_produces_the_same_starting_hash(
        self, device, roots, given_tokenizer: Path
    ) -> None:
        """What the operator's seed decision bought: R-02.47's required hash is now
        reproducible for a creation, which it was not before.

        *This case took no ``roots`` fixture until 2026-10-16*, which was fine while a
        creation touched no filesystem and is not now: the tokenizer reference is resolved
        against a configured model root at preflight.

        The hash is a function of the **config and the seed only** -
        ``created_weights_sha256(config=…, seed=…)`` - so naming a tokenizer does not enter
        it. That is deliberate and worth asserting from the other side: two creations
        differing only in tokenizer are the same starting weights.
        """
        from training import tasks
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec

        spec = AcquisitionSpec(
            mode=AcquisitionMode.CREATE,
            config=dict(LLAMA_TINY),
            seed=7,
            gpu_request=device.uuid,
            tokenizer_path=given_tokenizer,
        )
        assert (
            tasks.acquire_for_run(spec).record.starting_weights_sha256
            == tasks.acquire_for_run(spec).record.starting_weights_sha256
        )


class TestTheOptimizerOnRealTensors:
    def test_8_bit_states_are_built_and_recorded(
        self, device, roots, given_tokenizer: Path
    ) -> None:
        """AC-001.7 on hardware. ``bitsandbytes`` has no pure-CPU wheel everywhere, so this
        is the only place the real 8-bit path can be exercised.

        **``importorskip`` is kept and it is not a hedge here.** ``bitsandbytes`` is present
        in this image (checked, 2026-10-16) and this case therefore runs rather than skips;
        the guard is for the environment where it is absent, where ``build`` refuses
        ``optimizer_precision_unavailable`` rather than silently downgrading to 32-bit.
        """
        pytest.importorskip("bitsandbytes")

        from training import optimizer as optimizer_module
        from training import tasks
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec
        from training.run_config import RunConfig

        acquired = tasks.acquire_for_run(
            AcquisitionSpec(
                mode=AcquisitionMode.CREATE,
                config=dict(LLAMA_TINY),
                seed=1,
                gpu_request=device.uuid,
                tokenizer_path=given_tokenizer,
            )
        )
        _assert_production_placed_it(acquired, device)
        cfg = RunConfig(learning_rate=1e-4, warmup_steps=0, batch_size=1, gradient_accumulation=1)
        optimizer, identity = optimizer_module.build(acquired.module, cfg)
        assert identity.name == "adamw-8bit"
        assert type(optimizer).__name__.endswith("8bit")
        # R-02.13: gradient checkpointing on, and `use_cache` off with it.
        assert acquired.module.config.use_cache is False


#: ADR-024's budget. F1's own work, as a fraction of step wall time. **Not loosened** - the
#: budget was always met; what was wrong was the denominator. See the class docstring.
OVERHEAD_BUDGET = 0.05

#: The absolute companion to the ratio, and the reason this constant exists beside it: a
#: ratio can be passed by making its denominator large, so a fixed cost - one ``fsync``, one
#: database round trip - hides behind a big model at *any* budget. F1's per-step bookkeeping
#: is one validated append and one ``fsync``; single-digit milliseconds at the outside.
OVERHEAD_CEILING_MS = 25.0

#: The denominator's floor. ``bench_model``'s whole purpose is a step >= 100 ms, and this
#: asserts the fixture delivered one rather than trusting that it did - the check that
#: caught ``BENCH_BATCH`` being wrong by 2.4x on its first run in
#: ``test_boundary_loop_overhead.py``. Without it, shrinking the fixture silently
#: reintroduces exactly the defect this measurement was re-scaled to fix.
#:
#: **Measured, not estimated: 485.7 ms** on the RTX 3090 at bf16, batch 16 x sequence 512
#: (2026-10-16) - higher than ``bench_model``'s own ~128 ms note because ``drive_run``
#: enables gradient checkpointing (R-02.13, ``use_cache`` off with it), which re-runs the
#: forward during the backward. So the floor is well under the observed step and is a
#: tripwire on the fixture, not a second budget.
MIN_STEP_WALL_MS = 100.0

#: Steps averaged over. Enough that the first step's lazy CUDA initialisation is not the
#: whole measurement, small enough that this stays a gate rather than a benchmark.
MEASURED_STEPS = 20


class TestTheOverheadBudget:
    """AC-001.14 (ADR-024): F1's own work is under 5% of step wall time.

    **Measured on the step path itself**, not against a neighbouring one: a performance claim
    about a path that was not touched is not a verified claim. The figure is reported with its
    topology, because *a number quoted without its topology is not a measurement* - **both
    PCIe links on this node are x4, not x16** (Gen4 ~7.9 GB/s, Gen3 ~3.9 GB/s), so any
    host-to-device figure taken from an x16 number is wrong by 3-6x.

    ## This measurement was re-scaled on 2026-10-16, and the budget was NOT loosened

    *It ran on ``LLAMA_TINY`` at batch 2 x sequence 64 until then, and failed at 25-28 %.*
    That was a **measurement of the fixture, not of F1**: two layers of 64 hidden units make
    a whole step ~3.2 ms, which is the same order as one ``fsync`` (3.09 ms), so the ratio
    was reporting how small the model was. At realistic model scale the identical code path
    measures **0.70-2.84 %**, i.e. the 5 % budget was met the entire time.

    **So the fix is the denominator, and three things had to change together** - any one
    alone would have been the wrong fix:

    1. the fixture is ``tests.support.bench_model`` (86 M parameters, batch 16 x sequence
       512, bf16), sized so one step is **>= 100 ms** and the denominator is two orders of
       magnitude above a single ``fsync``;
    2. the ratio is paired with an **absolute millisecond ceiling**, because a ratio alone
       can be passed by growing the model - ``bench_model``'s own docstring states that
       pairing as a property of using the fixture at all, not of any one benchmark; and
    3. the step wall time itself is asserted **>= 100 ms**, so a future change that shrinks
       the fixture fails loudly here instead of quietly passing a 5 % test again.

    **Retiring this in favour of the new benchmarks was considered and rejected.**
    ``test_boundary_loop_overhead.py`` derives F6's boundary overhead by *subtracting* the
    ``step`` phase, and ``test_signal_source_throughput_gpu.py`` measures F2's generation.
    Neither measures F1's per-step bookkeeping - assembling a record, validating it against
    ``step-record-v1.json``, appending it durably - so retiring this case would leave
    AC-001.14 with no measurement anywhere, which is the "figure from a path you did not
    exercise" ADR-024 forbids, in its worse form: no figure at all.
    """

    def test_f1s_bookkeeping_is_under_five_percent_of_step_wall_time(
        self, device, roots, given_tokenizer: Path
    ) -> None:
        import time

        import torch

        from tests.support.bench_model import BENCH_BATCH, BENCH_LLAMA, BENCH_SEQ
        from training import run_log, tasks
        from training.acquisition.base import AcquisitionMode, AcquisitionSpec
        from training.run_config import RunConfig
        from training.step_record import StepWriter

        tmp_path, states, checkpoints = roots
        # BENCH_LLAMA's vocab_size is pinned to LLAMA_TINY's precisely so ONE tokenizer
        # writer serves both fixtures; `given_tokenizer` is therefore valid for this config
        # too, and `assert_vocab_agrees` is what would say so if that ever stopped holding.
        bench_tokenizer = checkpoints / "bench-tokenizer"
        bench_tokenizer.mkdir()
        write_tiny_tokenizer(bench_tokenizer, vocab_size=BENCH_LLAMA["vocab_size"])

        def bench_spec() -> AcquisitionSpec:
            return AcquisitionSpec(
                mode=AcquisitionMode.CREATE,
                config=dict(BENCH_LLAMA),
                seed=1,
                gpu_request=device.uuid,
                tokenizer_path=bench_tokenizer,
            )

        cfg = RunConfig(
            learning_rate=1e-4,
            warmup_steps=0,
            batch_size=BENCH_BATCH,
            gradient_accumulation=1,
        )
        acquired = tasks.acquire_for_run(bench_spec())
        _assert_production_placed_it(acquired, device)

        source = RealSource(
            torch.device("cuda"), BENCH_LLAMA["vocab_size"], cfg.batch_size, BENCH_SEQ
        )
        result = tasks.drive_run(
            run_id="gpu-overhead",
            spec=bench_spec(),
            cfg=cfg,
            batch_source=source,
            objectives=FixtureRegistry(),
            session=None,
            log_root=tmp_path,
            trainer_state_root=states,
            max_steps=MEASURED_STEPS,
        )
        assert result.steps_completed == MEASURED_STEPS

        log = run_log.run_log_path(tasks.run_log_root(tmp_path), "gpu-overhead")
        blocks = list(run_log.read_blocks(log, block="step"))
        assert len(blocks) == MEASURED_STEPS, (
            f"the run wrote {len(blocks)} step records, not {MEASURED_STEPS}; the figure "
            f"below is not about the work this case says it measures"
        )
        step_wall_ms = sum(block["wall_time_ms"] for block in blocks) / len(blocks)

        # F1's own bookkeeping, timed on the same path: assembling and validating a record,
        # appending it durably. The index and the socket are absent here by design - this
        # measures what F1 adds to a step, and a database round trip is the foundation's.
        writer = StepWriter(run_id="gpu-overhead-bench", log_root=tmp_path)
        from training.step import StepOutcome

        outcome = StepOutcome(
            loss=0.5, objective="sft", grad_norm=1.0, wall_time_ms=1.0, batch_sha256="a" * 64
        )
        started = time.monotonic()
        for step in range(1, MEASURED_STEPS + 1):
            writer.write(session=None, step=step, outcome=outcome, cfg=cfg)
        bookkeeping_ms = (time.monotonic() - started) * 1000.0 / MEASURED_STEPS

        # The positive check that the timed window wrote what it claims to have written.
        # Without it a writer that silently wrote nothing reports a flattering overhead and
        # is indistinguishable from a fast one.
        # `writer.log_path`, not a second spelling of it. The first version of this check
        # recomputed the path as `run_log.run_log_path(tasks.run_log_root(tmp_path), …)` and
        # read 0 records: `StepWriter` roots at `log_root` directly while `drive_run` roots
        # at `tasks.run_log_root(log_root)`. The check caught my own restated path, which is
        # the whole reason it is here.
        written = list(run_log.read_blocks(writer.log_path, block="step"))
        assert (
            len(written) == MEASURED_STEPS
        ), f"the bookkeeping window appended {len(written)} records, not {MEASURED_STEPS}"

        share = bookkeeping_ms / step_wall_ms
        print(
            f"\nF1 overhead: {bookkeeping_ms:.3f} ms of bookkeeping against "
            f"{step_wall_ms:.3f} ms mean step wall time over {MEASURED_STEPS} steps "
            f"= {share * 100:.2f}%  "
            f"[{device.name}, {device.uuid}, PCIe x4 negotiated on this node, "
            f"batch={cfg.batch_size} seq={BENCH_SEQ} "
            f"model {BENCH_LLAMA['num_hidden_layers']}L/{BENCH_LLAMA['hidden_size']}h bf16 "
            f"(~86 M params)]"
        )
        print(
            f"   BREAK-EVEN: within the {OVERHEAD_BUDGET:.0%} budget for any step "
            f">= {bookkeeping_ms / OVERHEAD_BUDGET:.1f} ms"
        )

        # THE DENOMINATOR FIRST. A ratio reported against a toy step is a measurement of the
        # toy - which is exactly how this case read 25-28% while the real figure was ~1%.
        assert step_wall_ms >= MIN_STEP_WALL_MS, (
            f"the step measured {step_wall_ms:.1f} ms, under the {MIN_STEP_WALL_MS:.0f} ms "
            f"floor `bench_model` exists to provide. The ratio below would be a statement "
            f"about the fixture's smallness, not about F1's overhead; fix the fixture "
            f"before reading the budget."
        )
        assert share < OVERHEAD_BUDGET, (
            f"F1's bookkeeping is {share * 100:.2f}% of a {step_wall_ms:.1f} ms step, over "
            f"the {OVERHEAD_BUDGET:.0%} budget (ADR-024). If this is real, the "
            f"record-keeping has become the work."
        )
        assert bookkeeping_ms < OVERHEAD_CEILING_MS, (
            f"F1's bookkeeping is {bookkeeping_ms:.1f} ms per step, over the "
            f"{OVERHEAD_CEILING_MS:.0f} ms absolute ceiling. The ratio can be inside budget "
            f"and this still be a regression: a fixed cost hides behind a large denominator, "
            f"which is why both bounds are asserted."
        )
