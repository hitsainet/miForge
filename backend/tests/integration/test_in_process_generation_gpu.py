"""ADR-009's **default** inference provider, executed on a card.  ``gpu``-marked.  (F4 8.3)

**The gap this file closes.** ``providers/model_handle.py``'s
:meth:`AcquiredModelHandle.generate` - the entire real generation body - and the two
``torch.cuda`` lines of :meth:`AcquiredModelHandle.free_vram_bytes` carried
``# pragma: no cover - needs a card`` and **no ``gpu``-marked test drove any of them.** The
only test over that surface, ``test_scoring_in_process.py``, is an unmarked unit test against
:class:`providers.fakes.FakeModelHandle`. So the in-process provider - which ADR-009 makes the
**default**, which F4's whole default scoring path runs on (R-02.16, R-02.17), and which is
what makes acceptance criterion 20 demonstrable with no sibling present - had never once been
executed on real weights on real hardware. Its task said so honestly: *"F4 would have been
writing a test it had never seen pass or fail."*

A ``pragma: no cover`` is a note that a line is unmeasured. Nine of them in one function is
a statement that the function is unexecuted, and an unexecuted function reads exactly like a
working one.

## What it found

**A real defect, and the fake could not have found it.** ``GenerationParams`` declares
``stop`` and ``seed``; the HTTP provider (``providers/openai_compatible.py``, lines 206-212)
sends **both**; ``AcquiredModelHandle.generate`` honoured **neither**, and
:class:`~providers.fakes.FakeModelHandle` ignores them too - so every existing assertion
agreed with the defect by construction, which is this project's named reason a suite stays
green over a real bug.

Why that is a defect rather than a gap in an optional feature:

* ``GenerationParams``' own docstring says these are *"part of what a score means"* and are
  *"recorded in the evaluation block beside every score rather than being an incidental call
  argument"*. So the record claims a constraint the default provider did not apply.
* ``InProcessProvider.generate`` **refuses** an unknown option on the stated grounds that
  *"a silently dropped option is a request the caller believes it made, and a score taken
  under a parameter the engine ignored is not the score it claims"* - and then dropped two
  **known** ones, one layer down, silently.
* **FR-004.8** is this requirement verbatim and names ``seed``: *"Where a scoring pass
  depends on a parameter being respected, the provider must declare support for it;
  otherwise F4 refuses rather than sending it hopefully … A scoring path that silently
  relies on … ``seed`` … being honoured produces numbers whose comparability depends on which
  server answered."* Two providers disagreeing about one recorded parameter is the exact
  failure it exists to prevent, with the disagreement **inside** miForge.
* ``seed``'s field comment says it *"determines the subsample **and any sampling**"*. An
  in-process score sampled at ``temperature>0`` with a seed set was irreproducible while
  recording a seed asserting the opposite - and AGENT.md's own rule is that *an overclaimed
  seed is worse than none*.

Fixed in ``providers/model_handle.py`` (the one file F4 task 8.3 permits). Both halves are
asserted below, and both were seen RED against the unfixed body before the fix - the
negative control is recorded in the report rather than implied.

## Topology and sample, because a number without them is not a measurement

One **RTX 3090** on ``mcs-lnxhost02``, PCIe link **x4 negotiated** (``width.max`` 16,
``width.current`` 4 - read live from the driver, confirming ADR-024's warning rather than
quoting it). Nothing here is a throughput figure; the benchmarks live under
``tests/performance/``. This file asserts behaviour.

Run with::

    pytest -m gpu backend/tests/integration/test_in_process_generation_gpu.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from tests.support.bench_model import bench_prompts
from tests.support.tiny_hf_model import write_tiny_model

pytestmark = pytest.mark.gpu


@pytest.fixture
def device():
    """The card this pass is pinned to, resolved live by UUID (ADR-008).

    Skipped rather than failed when no card answers, so a red here cannot be confused with a
    real regression on a machine where the marker was selected by accident.
    """
    from core.errors import Refusal
    from core.gpu import resolve_device

    try:
        return resolve_device("auto")
    except Refusal as exc:
        pytest.skip(f"no CUDA device resolved ({exc.code}); this case needs the GPU node")


@pytest.fixture
def roots(tmp_path: Path):
    """Point the configured model roots at ``tmp_path``.

    Not optional and not incidental: ``local_dir.resolve_within_roots`` **refuses**
    ``model_source_outside_root`` for a path outside ``checkpoint_root``/``parts_root``, and
    in the deployed pod those are ``/data/checkpoints`` and ``/data/parts``. So the first run
    of this file errored in all eight cases - which is the confinement guard working, on the
    real settings a worker that holds the card actually has.
    """
    from core.config import Settings, reset_settings

    checkpoints = tmp_path / "checkpoints"
    parts = tmp_path / "parts"
    for directory in (checkpoints, parts):
        directory.mkdir()
    import core.config as config

    original = config._settings
    config._settings = Settings(  # type: ignore[call-arg]
        _env_file=None, checkpoint_root=checkpoints, parts_root=parts
    )
    yield checkpoints
    config._settings = original
    reset_settings()


@pytest.fixture
def acquired(roots: Path, device):
    """A **real** acquisition of real weights onto the leased card, through production code.

    ``tasks.acquire_for_run`` is the production entry point, not a hand-built
    ``AcquiredModel``: the whole point of this file is that the adapter has never been driven
    over the real thing, and a dataclass assembled here would reintroduce exactly the
    fixture-agrees-with-itself problem it exists to escape.
    """
    from training import tasks
    from training.acquisition.base import AcquisitionSpec

    directory = write_tiny_model(roots / "weights")
    return tasks.acquire_for_run(AcquisitionSpec(path=directory, gpu_request=device.uuid))


@pytest.fixture
def handle(acquired):
    from providers.model_handle import AcquiredModelHandle

    return AcquiredModelHandle(acquired, step=7)


def test_the_module_under_test_is_the_staged_tree(request) -> None:
    """**Which ``src`` was measured.** Reported, not assumed.

    ``pyproject.toml`` sets ``pythonpath = ["backend/src"]`` and the image also carries a
    pip-installed ``miforge``; a sibling agent found that the ``pythonpath`` copy wins. A
    benchmark or a defect report naming the wrong tree is worthless, so the provenance is an
    assertion in the suite rather than a claim in a write-up.
    """
    import providers.model_handle as module

    resolved = Path(module.__file__).resolve()
    print(f"\nmodel_handle under test: {resolved}")
    assert "site-packages" not in resolved.parts, (
        f"the pip-installed copy won over pythonpath: {resolved}. Every figure and every "
        f"defect below would describe a tree that is not the one under review."
    )


class TestTheHandleOverRealWeightsOnTheCard:
    """The four identity/budget methods, driven over a real acquisition."""

    def test_free_vram_reports_the_cards_real_free_bytes(self, handle) -> None:
        """The two ``torch.cuda.mem_get_info`` lines, executed for the first time.

        Bounded on **both** sides. A lower bound alone passes against a stub returning a
        huge constant, and an upper bound alone passes against zero - and zero is
        meaningful here, because ``free_vram_bytes`` returns exactly that for a CPU-resident
        model. So a plain ``> 0`` could not tell a working card read from the CPU branch
        having been taken by mistake.
        """
        import torch

        free = handle.free_vram_bytes()
        total = torch.cuda.get_device_properties(0).total_memory
        print(f"\nfree VRAM reported: {free / 1024**3:.2f} GiB of {total / 1024**3:.2f} GiB total")
        assert free > 0, "a card is present, so the CPU branch must not have been taken"
        assert free <= total, (
            f"reported {free} free bytes on a card with {total} total: the handle is not "
            f"reading this device"
        )
        # E-24 refuses *before* allocating against this number, so it must track reality.
        # A figure frozen at import time would satisfy the bounds above.
        block = torch.empty(int(512 * 1024**2 / 2), dtype=torch.float16, device="cuda")
        try:
            after = handle.free_vram_bytes()
            assert after < free, (
                f"free VRAM did not fall after a 512 MiB allocation ({free} -> {after}); "
                f"E-24's refusal is reading a stale or constant figure"
            )
        finally:
            del block
            torch.cuda.empty_cache()

    def test_identity_names_the_parameters_and_the_leased_card(
        self, handle, acquired, device
    ) -> None:
        """``parameter_set_digest`` distinguishes steps; ``device_uuid`` is the leased card."""
        digest = handle.parameter_set_digest()
        print(f"\nparameter_set_digest: {digest}")
        assert digest.endswith("@step7"), (
            "the step must be in the digest: without it every score of a run claims the "
            "same parameters, which is the 'ours is not an identity' failure it prevents"
        )
        assert digest.split("@")[0] == acquired.record.starting_weights_sha256
        assert (
            handle.device_uuid() == device.uuid
        ), "the handle must name the LEASED card; a score's provider provenance reads this"
        assert handle.model_identity() == acquired.record.base_model


class TestGenerationOnRealWeights:
    """:meth:`AcquiredModelHandle.generate` - the nine pragma'd lines - on the card."""

    def test_one_completion_per_prompt_in_input_order(self, handle) -> None:
        """The order contract, which callers join scores to positionally.

        Asserted through **distinguishable** prompts rather than a length check. A handle
        that reordered or dropped one returns a list of the right length, so ``len()`` alone
        cannot see the failure the docstring calls out: *"a handle that reordered or dropped
        one would misattribute every score after the gap."*
        """
        from providers.base import GenerationParams

        prompts = bench_prompts(5)
        texts = handle.generate(prompts, GenerationParams(max_tokens=8))
        print(f"\ngenerated {len(texts)} completions for {len(prompts)} prompts")
        assert len(texts) == len(prompts)
        assert all(isinstance(text, str) for text in texts)
        # Positional integrity, proved by generating each prompt alone and matching. Greedy
        # decoding makes single-prompt and batched generation comparable; see the
        # determinism case below for why that is a safe assumption here.
        alone = [handle.generate([prompt], GenerationParams(max_tokens=8))[0] for prompt in prompts]
        assert texts == alone, (
            "batched generation did not match prompt-by-prompt generation in order; "
            "results are joined to prompts positionally, so this misattributes every score"
        )

    def test_greedy_decoding_is_deterministic(self, handle) -> None:
        """Criterion 2's shape: *"deterministic decoding and the same provider, exactly"*."""
        from providers.base import GenerationParams

        params = GenerationParams(max_tokens=12, temperature=0.0)
        prompts = bench_prompts(3)
        first = handle.generate(prompts, params)
        second = handle.generate(prompts, params)
        assert first == second, (
            "greedy decoding on one provider gave two different answers; AC2 requires "
            "'with deterministic decoding and the same provider, exactly'"
        )

    def test_the_stop_sequence_is_honoured_not_dropped(self, handle) -> None:
        """**The defect.** ``GenerationParams.stop`` reached the card and did nothing.

        ``openai_compatible`` sends ``stop``; this handle ignored it. FR-004.8 forbids
        exactly that asymmetry - a decode parameter recorded beside a score must be applied
        or refused, never silently dropped, or the score's meaning depends on which provider
        answered.

        The assertion is written against the **returned text** rather than against a kwarg
        passed to ``module.generate``, because asserting the call would pass against a
        parameter the engine accepted and ignored - which is the precise failure FR-004.8
        cites from miStudio's verified ``response_format`` finding.
        """
        from providers.base import GenerationParams

        # Take a greedy continuation first, then make one of the tokens it actually produced
        # the stop sequence. Deriving the stop string from real output is what makes the
        # case non-vacuous: a literal chosen in advance might never be generated, and the
        # assertion would then hold for a handle that ignores `stop` entirely.
        prompts = bench_prompts(1)
        baseline = handle.generate(prompts, GenerationParams(max_tokens=16, temperature=0.0))[0]
        tokens = baseline.split()
        if len(tokens) < 3:
            pytest.skip(f"greedy output too short to derive a stop sequence from: {baseline!r}")
        stop = tokens[1]

        stopped = handle.generate(
            prompts, GenerationParams(max_tokens=16, temperature=0.0, stop=(stop,))
        )[0]
        print(f"\nbaseline={baseline!r}\n    stop={stop!r}\n stopped={stopped!r}")
        assert stop not in stopped, (
            f"the stop sequence {stop!r} appears in the completion {stopped!r}: `stop` was "
            f"accepted and dropped. The HTTP provider sends it, so two providers disagree "
            f"about one recorded decode parameter (FR-004.8)"
        )
        assert len(stopped) < len(
            baseline
        ), f"output with stop={stop!r} was not shorter than without it; `stop` had no effect"

    def test_the_seed_makes_sampling_reproducible(self, handle) -> None:
        """**The second half of the defect.** ``GenerationParams.seed`` was ignored.

        ``seed``'s own field comment says it *"determines the subsample and any sampling"*
        and FR-004.8 names ``seed`` explicitly. Unseeded sampling recorded beside a seed is
        an overclaimed seed, which this project's rule calls worse than none.

        **Both directions are asserted**, because "same seed gives the same text" passes
        against a handle that forces greedy decoding and never samples at all - which would
        satisfy reproducibility by destroying the sampling the caller asked for.
        """
        from providers.base import GenerationParams

        prompts = bench_prompts(3)
        seeded = GenerationParams(max_tokens=12, temperature=0.9, seed=1234)
        first = handle.generate(prompts, seeded)
        second = handle.generate(prompts, seeded)
        assert first == second, (
            "the same seed gave two different sampled completions: `seed` was accepted and "
            "dropped, so an in-process score records a seed that does not reproduce it"
        )

        other = handle.generate(prompts, GenerationParams(max_tokens=12, temperature=0.9, seed=99))
        print(f"\nseed=1234: {first}\n  seed=99: {other}")
        assert other != first, (
            "two different seeds gave identical sampled output, so nothing is being "
            "sampled: the seed is reproducible because decoding is greedy, which is not "
            "what temperature=0.9 asked for"
        )


class TestTheDefaultProviderEndToEnd:
    """:class:`providers.in_process.InProcessProvider` over the real handle (ADR-009)."""

    def test_a_scoring_pass_runs_on_the_default_provider(self, handle, device) -> None:
        """The default provider, real weights, real card - F4's default scoring path.

        This is the composition acceptance criterion 20 rests on: a scoring pass with **no
        sibling application present at all**, which is only true because the default provider
        is in-process.
        """
        from providers.base import GenerationParams
        from providers.in_process import InProcessProvider

        provider = InProcessProvider(handle)
        assert provider.available() is True
        completions = provider.generate(bench_prompts(4), GenerationParams(max_tokens=8))

        assert len(completions) == 4
        for completion in completions:
            # Every score records its provider (ADR-009, R-02.16), and for in-process the
            # identity must name *which* parameters answered - not "ours".
            assert completion.provider.kind == "in_process"
            assert completion.provider.engine_version == handle.parameter_set_digest()
        print(f"\nprovider identity: {completions[0].provider!r}")
        assert provider.device_uuid == device.uuid

    def test_the_budget_refusal_fires_before_allocating(self, handle) -> None:
        """E-24 on real free-VRAM numbers, which it had never been evaluated against.

        The refusal's whole claim is that it reads the card *before* allocating. On the fake
        handle ``free_vram_bytes`` is a constant, so the comparison was arithmetic over a
        made-up number; here both sides are real.
        """
        from core.errors import Refusal
        from providers.base import GenerationParams
        from providers.in_process import InProcessProvider

        # A KV-cache estimate that cannot fit in 24 GiB, via max_tokens rather than by
        # filling the card: occupying real VRAM to force this would leave the test's
        # outcome dependent on what else is resident on a shared card.
        provider = InProcessProvider(handle, kv_bytes_per_token=1 << 30)
        with pytest.raises(Refusal) as caught:
            provider.generate(bench_prompts(2), GenerationParams(max_tokens=4096))
        assert caught.value.code == "provider_budget_exceeded"
        details: dict[str, Any] = caught.value.details  # type: ignore[assignment]
        print(f"\nE-24 refusal details: {details}")
        assert details["free_bytes"] > 0, (
            "the refusal reported zero free bytes on a live card, so it compared against "
            "the CPU branch's figure rather than reading the device"
        )
        assert details["estimate_bytes"] > details["free_bytes"]
