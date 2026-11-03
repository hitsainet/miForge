"""The scoring pass as its own number, clean and time-sliced.  ``gpu``-marked.

F4 tasks **9.1** and **9.3**; ADR-024, ADR-009, ADR-012, BRD-02 section 9.

## Both tasks were blocked on one thing, and it was not the card

F4 9.1 said: *"**NOT MEASURED, and no figure reported.** The only handle available is a fake,
and *a figure taken against a stub measures the stub*."* F4 9.3 said contention *"needs the
card"*. The first of those is the real blocker and it is the stricter one: a card alone would
not have helped, because ``providers.fakes.FakeModelHandle.generate`` returns a constant, so
a scoring-pass figure taken through it measures ``services.scoring`` plus a Python function
call and attributes the whole number to miForge.

What both were waiting for is a **real** :class:`~providers.model_handle.AcquiredModelHandle`
driven on the card - which is F4 task 8.3, and which did not work until it was first executed
on 2026-10-16 (see ``tests/integration/test_in_process_generation_gpu.py``: the real
generation body raised ``ValueError`` on any tokenizer declaring ``token_type_ids``). *So
these two figures were downstream of a defect, not of hardware.* **F4 was right to report no
number rather than a number taken against the fake**, and that judgement is the reason this
file can be trusted now: there is no earlier figure it has to be reconciled with.

## What the pass is, and what the number therefore includes

One call to ``services.scoring.service.evaluate`` over **both** sets - test and retention,
which is FR-004.24's pairing - doing real generation on the card, real expectation
evaluation, real aggregation, the real ``ScoreRecord`` write to the run log and the real
index write. The breakdown below separates the provider's generation time from the rest, so
the figure says which part is miForge's and which is the model's. **That separation is the
point**: a single "the pass took N seconds" number cannot tell an operator whether to buy a
faster card or fix miForge.

## Everything a figure here must name (ADR-012: *every number names its scale and sample*)

Per F4 9.1's own list - provider kind and identity, card **UUID**, decode parameters, ``n``,
and cadence - all printed with each figure rather than described here, so a pasted figure
carries them. Plus the topology, because ADR-024 requires it:

**One RTX 3090 on ``mcs-lnxhost02``. PCIe link x4 negotiated; the slot reports
``width.max`` 16.** Read from the driver, not assumed. A host-to-device figure lifted from an
x16 number is wrong by 3-6x. **No figure here contains a host-to-device transfer**: the
weights are resident for the whole pass and the prompts are a few hundred bytes - which is
why the link width is worth stating even though nothing measures it. The KV cache is
allocated on-device inside the pass, and E-24 refuses before allocating it.

## Contention is time-slicing, and it is a separate process

GPU sharing on this cluster is **time-slicing, no MPS** (BRD section 9, 2026-10-13), so
kernels from two processes serialise rather than overlapping. ``tests.support.contention``
is therefore a subprocess, and every contended figure here asserts the load actually
progressed - because a load that died produces a contended figure equal to the clean one,
which reads as *"contention is free"*.

Run with::

    pytest -m gpu backend/tests/performance/test_scoring_pass_gpu.py -s
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from tests.support.bench_model import bench_prompts
from tests.support.contention import GpuLoad

pytestmark = pytest.mark.gpu

#: The rule expectation the sets are scored against. A rule selector is the one kind
#: guaranteed available with nothing configured, so the figure needs no sibling and no judge
#: model - which keeps this a measurement of *scoring*, not of a second model's latency.
RULE: dict[str, Any] = {
    "kind": "rule",
    "name": "len",
    "config_sha256": None,
    "expression": '{"min_length": 4}',
}

#: ``n`` for each set. Named, because ADR-012 requires a figure to state its sample and
#: because ``ScoreRecord.n`` is part of what a score means.
SET_N = 16

#: Decode parameters. Greedy, which is FR-004.5's default and the only decoding BRD-02's
#: acceptance criterion 2 is written against.
MAX_TOKENS = 32
BATCH_BOUND = 8


@pytest.fixture
def device():
    from core.errors import Refusal
    from core.gpu import resolve_device

    try:
        return resolve_device("auto")
    except Refusal as exc:
        pytest.skip(f"no CUDA device resolved ({exc.code}); this case needs the GPU node")


@pytest.fixture
def real_handle(tmp_path: Path, device):
    """A real acquisition on the leased card, wrapped in the real handle.

    **The whole reason these two figures exist.** Swap this for
    ``providers.fakes.FakeModelHandle`` and every number below becomes a measurement of a
    ``return`` statement - which is exactly why F4 reported none.
    """
    import core.config as config
    from core.config import Settings, reset_settings
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
        yield AcquiredModelHandle(acquired, step=0)
    finally:
        config._settings = original
        reset_settings()


@pytest.fixture
def session(tmp_path: Path):
    from tests.support import scores as fixtures
    from tests.support.db import migrated_engine, session_for

    engine = migrated_engine(tmp_path / "db")
    fixtures.create_score_tables(engine)
    with session_for(engine) as open_session:
        yield open_session
    engine.dispose()


def _spec(provider_name: str = "in_process"):
    """The run's resolved scoring configuration, both sets populated.

    Prompts come from ``bench_prompts`` rather than being literals: the fixture tokenizer is
    word-level over ``t4``..``t511``, so an English prompt encodes to a run of ``<unk>`` and
    the figure would describe generation from a degenerate input.
    """
    from providers.base import GenerationParams
    from services.scoring import expectations, retention, service, sets

    expectation = expectations.resolve_expectation(RULE, source="pack")

    def resolved(kind: str):
        reference = sets.SetReference(
            rendered=f"{kind}_set/general@1",
            structured={"type": f"{kind}_set", "name": "general", "version": "1"},
        )
        prompts = bench_prompts(SET_N)
        return sets.resolve(
            kind,
            reference,
            [{"prompt": text, "group": f"g{index % 2}"} for index, text in enumerate(prompts)],
            markers=("held_out", "evaluation_only"),
        )

    return service.EvaluationSpec(
        run_id="bench-scoring",
        provider_name=provider_name,
        aggregation="mean",
        decode=GenerationParams(max_tokens=MAX_TOKENS, temperature=0.0),
        test_set=resolved("test"),
        retention_set=resolved("retention"),
        test_expectation=expectation,
        retention_expectation=expectation,
        floor=retention.ResolvedFloor(value=0.0, layer="pack_slot"),
        batch_bound=BATCH_BOUND,
    )


class _TimedProvider:
    """Wraps the real provider to split **generation** time out of the pass.

    Without this the pass is one number and cannot say whether it is dominated by the model
    or by miForge - which is the question ADR-024's *"measure the path you changed"* exists
    to make answerable. Delegation is explicit rather than ``__getattr__``: a silent
    forwarder would also forward a method the real provider does not have, and the refusal
    ``inference_provider_option_unknown`` is part of what is being measured.
    """

    def __init__(self, inner: Any) -> None:
        self._inner = inner
        self.generate_seconds = 0.0
        self.calls = 0
        self.completions = 0

    @property
    def identity(self) -> Any:
        return self._inner.identity

    @property
    def parameter_set_digest(self) -> str:
        return self._inner.parameter_set_digest

    @property
    def device_uuid(self) -> str | None:
        return self._inner.device_uuid

    def available(self) -> bool:
        return self._inner.available()

    def capabilities(self) -> frozenset[str]:
        return self._inner.capabilities()

    def kv_cache_estimate_bytes(self, prompt_count: int, max_tokens: int) -> int:
        return self._inner.kv_cache_estimate_bytes(prompt_count, max_tokens)

    def generate(self, prompts: list[str], params: Any, **options: Any) -> Any:
        import torch

        started = time.perf_counter()
        out = self._inner.generate(prompts, params, **options)
        # The provider's own path is synchronous through `module.generate`, but synchronise
        # anyway: if a future provider returned before the kernels finished, the untimed
        # remainder would absorb GPU time and be reported as miForge's overhead - a figure
        # wrong in the alarming direction.
        torch.cuda.synchronize()
        self.generate_seconds += time.perf_counter() - started
        self.calls += 1
        self.completions += len(out)
        return out


def _run_pass(handle: Any, session: Any, log_path: Path, step: int) -> dict[str, float]:
    """One real ``evaluate`` over both sets. Returns the timing breakdown in seconds."""
    from providers.in_process import InProcessProvider
    from services.scoring import service

    provider = _TimedProvider(InProcessProvider(handle, batch_bound=BATCH_BOUND))
    started = time.perf_counter()
    outcome = service.evaluate(_spec(), step, session=session, log_path=log_path, provider=provider)
    total = time.perf_counter() - started
    session.commit()
    assert len(outcome.records) == 2, (
        f"a pass over both sets must write two ScoreRecords (FR-004.24's pairing); got "
        f"{len(outcome.records)} - the figure would be for half a pass"
    )
    return {
        "total": total,
        "generation": provider.generate_seconds,
        "miforge": total - provider.generate_seconds,
        "calls": float(provider.calls),
        "completions": float(provider.completions),
    }


def _report(label: str, row: dict[str, float], handle: Any, extra: str = "") -> None:
    """Every figure names its sample, its provider, its card and its decode (ADR-012)."""
    from providers.in_process import PROVIDER_KIND

    share = row["generation"] / row["total"] if row["total"] else 0.0
    print(f"\n── {label} ──")
    print(f"   provider      kind={PROVIDER_KIND}  identity={handle.model_identity()}")
    print(f"   parameter set {handle.parameter_set_digest()}")
    print(f"   card          {handle.device_uuid()}  (RTX 3090, PCIe x4 negotiated, slot x16)")
    print(
        f"   decode        greedy temperature=0.0 top_p=1.0 max_tokens={MAX_TOKENS}; "
        f"batch_bound={BATCH_BOUND}"
    )
    print(
        f"   sample        n={SET_N} per set x 2 sets (test + retention) = "
        f"{int(row['completions'])} completions in {int(row['calls'])} provider calls; "
        f"cadence: ONE pass (per-boundary cost; a run multiplies by its test_every)"
    )
    print(f"   PASS TOTAL    {row['total'] * 1000:9.1f} ms")
    print(f"     generation  {row['generation'] * 1000:9.1f} ms  ({share * 100:.1f} % of the pass)")
    print(
        f"     miForge     {row['miforge'] * 1000:9.1f} ms  "
        f"({(1 - share) * 100:.1f} % - expectations, aggregation, ScoreRecord, index)"
    )
    if extra:
        print(f"   {extra}")


class TestTask91TheScoringPassAsItsOwnNumber:
    """F4 9.1: the pass, measured, with everything a figure must name."""

    def test_the_pass_is_measured_and_its_generation_share_reported(
        self, real_handle: Any, session: Any, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "runs" / "bench-scoring" / "run.jsonl"

        # Warm up on a separate step so lazy CUDA init and kernel autotuning are not charged
        # to the reported figure. A separate step rather than the same one because two
        # passes at one step would be two score points for one step.
        _run_pass(real_handle, session, log_path, step=0)
        row = _run_pass(real_handle, session, log_path, step=1)
        _report("F4 9.1: scoring pass, UNCONTENDED", row, real_handle)

        assert row["completions"] == SET_N * 2, (
            "the pass must generate one completion per prompt across both sets; a short "
            "count means a set was skipped and the figure is for less than a pass"
        )
        assert row["generation"] > 0.0, (
            "zero generation time means the provider was not the real one - the exact "
            "failure F4 9.1 declined to report a number against"
        )
        # The honest shape of this figure: generation DOMINATES. Asserted so that a future
        # change making miForge's share the larger part fails here and is read rather than
        # absorbed into a slower total.
        assert row["generation"] > row["miforge"], (
            f"miForge's own scoring work ({row['miforge'] * 1000:.0f} ms) exceeded "
            f"generation ({row['generation'] * 1000:.0f} ms). On a real card that inverts "
            f"the expected shape of a scoring pass and is a finding, not a slow run"
        )


class TestTask93TheSamePassUnderTimeSlicing:
    """F4 9.3: *"A clean-room figure is not the figure a run sees."*"""

    def test_the_pass_under_a_competing_cuda_context(
        self, real_handle: Any, session: Any, tmp_path: Path
    ) -> None:
        log_path = tmp_path / "runs" / "bench-scoring" / "run.jsonl"

        _run_pass(real_handle, session, log_path, step=0)
        clean = _run_pass(real_handle, session, log_path, step=1)
        with GpuLoad() as load:
            loaded = _run_pass(real_handle, session, log_path, step=2)
            # THE POSITIVE CHECK, while the load is still alive to be checked.
            load.assert_running()

        slowdown = loaded["total"] / clean["total"] if clean["total"] else 0.0
        gen_slowdown = loaded["generation"] / clean["generation"] if clean["generation"] else 0.0
        _report("F4 9.3: scoring pass, TIME-SLICED", loaded, real_handle, load.describe())
        print(
            f"   vs uncontended: pass {slowdown:.2f}x wall time "
            f"({clean['total'] * 1000:.0f} -> {loaded['total'] * 1000:.0f} ms); "
            f"generation {gen_slowdown:.2f}x"
        )
        print(
            f"   miForge's own share: {clean['miforge'] * 1000:.1f} ms clean -> "
            f"{loaded['miforge'] * 1000:.1f} ms contended "
            f"(CPU-bound work; a large change here would mean the host is contended too)"
        )

        assert loaded["total"] >= clean["total"], (
            f"the pass was FASTER under a competing CUDA context "
            f"({loaded['total']:.3f}s vs {clean['total']:.3f}s). Time-slicing serialises "
            f"kernels, so suspect the load never reached the card or that the clean figure "
            f"still included warm-up - not a real speed-up"
        )
