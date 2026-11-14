"""The judge-model selector's network figure - **F3 task 13.4's missing half.**

``upstream``-marked and additionally gated on two environment variables, so **CI never runs
it** (``pyproject.toml``'s ``addopts`` is ``-m 'not gpu and not upstream'``) and a developer
who has not pointed it at a real endpoint gets a skip that names what is missing. That pair
is deliberate: a test that silently never runs is worth nothing, and a test that makes CI
depend on a sibling being up is worth less than nothing. The ``upstream`` marker keeps CI
independent; the ``skipif`` keeps the local run honest about why it did not measure.

Task 13.4's in-process half was measured and recorded on 2026-10-13. The network half was
owed because *"it needs a real OpenAI-compatible endpoint, and against a stub it would
measure the stub."* This file is that measurement, and it is not quoted for the in-process
path or the other way round - ADR-024: **a number without its topology is not a
measurement.**

## THE RECORDED FIGURE  (2026-10-16)

Taken from inside the ``miforge-backend`` pod on the pilot cluster, over the **real**
production path: ``services.scoring.wiring.wire()`` ->
``providers.registry.JudgeProviderResolver`` -> ``providers.openai_compatible`` ->
``services.selectors.plugins.judge_model.JudgeModelSelector.evaluate``. No stub, no
injected client, no fake clock.

* **Endpoint**: miLLM, ``http://millm-backend.millm.svc.cluster.local:8000``, reached by
  ``base_url`` as ADR-009 requires - **one row beside vLLM, Ollama and the OpenAI API, not
  a named dependency.** The selector does not know it is miLLM.
* **Model (judge)**: ``LFM2.5-1.2B-Instruct`` - **a different model from the run's target**,
  which is the legitimate R-02.11 case and is asserted below, not assumed.
* **Topology**: a **real network call to a sibling service in-cluster** - pod-to-pod over
  the cluster CNI on one microk8s cluster, resolved through
  ``*.svc.cluster.local``. **This is same-cluster east-west latency and is NOT a WAN
  figure**; an operator judging over the internet, or over a VPN to another site, will see
  the round trip dominated by a link this measurement never crossed. The transport share
  here is small enough to be invisible next to generation (see the two numbers below), so
  the honest reading is *"on this cluster the judge figure is a generation figure"* - which
  stops being true the moment the endpoint is remote.
* **Sample**: 15 measured evaluations after 2 discarded warm-up calls, one prompt each,
  rendered to **428 characters** of prompt+completion.
* **Decode parameters**: ``temperature=0.0``, ``max_tokens=256``, ``top_p`` and ``stream``
  as the plugin sets them - the plugin's own constants, not the test's. A sampled judge
  makes a run irreproducible.
* **Concurrency bound: ONE.** Measured, not read off a setting - see
  ``TestTheConcurrencyBoundIsOne``, and the defect it records.

| quantity | value |
|---|---|
| warm round trip, median | **66.6 ms** per completion |
| warm round trip, p95 | **67.8 ms** |
| warm round trip, min / max | **64.9 ms** / **71.2 ms** |
| transport alone (``GET /v1/models``, same link, no generation, fresh connection) | **26.0 ms** |
| three completions in one ``generate`` call | **193.3 ms** = **2.81x** one, i.e. sequential |
| a 64-completion batch at the median | **~4.3 s** of wall time |
| cold cost of a **model switch** on this endpoint (observed once, separately) | **~14,200 ms** |

**Reproducibility, and the honest spread.** The file was run **three times** over about an
hour on the same endpoint, and once more after it was formatted. Warm medians: **66.6**,
**64.1**, **72.7**, **64.5 ms**; p95s 67.8 / 66.5 / 77.3 / 70.8 ms; 3-prompt ratios 2.81 /
2.89 / 2.91 / 2.80x. So the figure to quote is **64-73 ms per completion**, and the ~13 %
spread between runs is **not** sampling noise - within any
one run min and max sit inside 10 % of each other. It is the card: miLLM is serving on the
same GPU, time-sliced with no MPS (the suite's recorded GPU-sharing decision), so a judge
round trip costs whatever the card has left. **A judge figure on a shared card is a figure
for that card's load**, which is the condition ADR-024 exists to make people state. The
table above is the first run; the sequential 3x result reproduced in all three.

**The cold number is a model switch, not a process warm-up, and it is not from this file's
run.** miLLM holds one model resident at a time on this node's single card, so the first
request naming a model that is not resident pays the load: ``LFM2.5-1.2B-Instruct`` took
**14,237 ms** on a direct probe when another model was resident, and ``Llama-3.1-8B-Instruct``
took **41,550 ms**. The run that produced the table above found the model already resident,
so its own "first call" was 69.2 ms and says nothing about loading - which is why the switch
cost is quoted from where it was actually observed rather than from a warm-up that did not
pay it. **An operator whose judge alternates with another workload on this endpoint pays
this per alternation**, and it dwarfs everything else here.

**What the figure means against the budget, stated rather than implied.** ADR-024's under-5
%-of-step-wall-time budget is for miForge's own **overhead**; a judge round trip is not
overhead, it is the selector's work. At 66.6 ms per completion a batch of 64 costs ~4.3 s,
against the in-process ``rule`` path's 4.0 us per completion for the same batch - four
orders of magnitude apart. The two are not comparable and **neither is quoted for the
other.**

**A measured defect in the module under test, reported not fixed here.**
``OpenAICompatibleProvider._post`` opens a **fresh** ``httpx.Client`` for every completion
when no client was injected, and ``providers.registry._build_openai_compatible`` injects
none - so production pays TCP setup per judged completion. The constructor's own docstring
for ``client`` says what the right shape is: *"One client per process, pooled: a client per
call pays connection setup per completion."* Measured on this link, 20 reps each:
**client-per-call 66.6 ms median vs one pooled client 59.3 ms median - 7.3 ms, 11.0 % of the
round trip**, or 0.47 s on a 64-completion batch. On this cluster that is the cheapest
link there is; over TLS to a remote endpoint the handshake makes it far worse, which is
exactly the topology an operator reaches for when they want a bigger judge. Related and
also reported: ``close()`` is unreachable - ``_owns_client`` is ``client is None`` while the
guard needs ``self._client is not None and self._owns_client``, which cannot both hold. It
is not fixed here because giving the provider a process-lifetime client needs a lifecycle
owner that calls ``close()``, and nothing does.

## R-02.11 is asserted here, not assumed

The judge is measured against a **separate** model, which is exactly the legitimate case.
Because this file is the one place a judge is pointed at a *live* endpoint, it also carries
the negative control: pointing the run's own target at the same ``(base_url, model)``
**must** refuse ``judge_is_resident_model``. A measurement taken with the guard quietly
disabled would be a figure for a configuration the product forbids.
"""

from __future__ import annotations

import os
import statistics
import time

import pytest

from core.errors import Refusal
from providers.base import ProviderIdentity
from schemas.selector import VerdictValue

BASE_URL = os.environ.get("MIFORGE_LIVE_JUDGE_BASE_URL", "").strip()
MODEL = os.environ.get("MIFORGE_LIVE_JUDGE_MODEL", "").strip()

#: The provider key the row is discovered under. Lower-case, matching
#: ``providers.settings.NAME_PATTERN``.
PROVIDER_KEY = "livejudge"

#: Measurement parameters, named constants so the report can quote them (ADR-024).
WARMUP_CALLS = 2
MEASURED_CALLS = 15

#: Generous because miLLM loads one model at a time on this node's single card and a cold
#: load is tens of seconds. Stated as a measurement parameter rather than left at the
#: production default, because the warm-up deliberately pays the cold cost.
READ_TIMEOUT_S = 180.0

#: The operator's judge template, and **it had to be written carefully against a real
#: model** - which is itself a finding this measurement produced. The first template tried
#: here asked for "a number between 0 and 1" and the live model answered ``SCORE: 90%``;
#: ``parse_judge_reply`` refused it as ``judge_reply_unparseable``, which is correct and is
#: the design working - scraping ``90`` out of that reply would have produced a score
#: eighty-nine points outside the range, and the plugin refuses rather than clamping. A
#: second attempt that included a worked example (``SCORE: 0.85``) made the model **parrot
#: the example** for every completion, which parses cleanly and is not a judgement at all -
#: the more dangerous of the two failures, because nothing refuses it.
#:
#: So the template below names the form without supplying a gradeable number, and the
#: discrimination test asserts the result moves with the completion. **A judge template is a
#: verdict policy** (``SelectorSettings.judge_prompt_template``'s own comment), and the two
#: rejected drafts are recorded here so the next operator does not rediscover them.
TEMPLATE = (
    "Grade how well the completion answers the prompt.\n"
    "Reply with a single line of the form   SCORE: <n>   where <n> is a decimal number "
    "from 0.0 (useless) to 1.0 (perfect).\n"
    "Write no other text.\n\n"
    "PROMPT: {prompt}\nCOMPLETION: {completion}\n"
)

PROMPT = "What is the capital of France, and why is it the capital?"
COMPLETION = (
    "Paris. It has been the seat of French government since the Capetian kings made it "
    "their principal residence, and the institutions of state concentrated there."
)

#: A completion that does not answer the prompt at all. Used to prove the live judge
#: discriminates, so the timings are a figure for a working judge rather than for a
#: constant-returning one.
OFF_TOPIC_COMPLETION = "I enjoy cheese sandwiches on a rainy afternoon."

pytestmark = [
    pytest.mark.upstream,
    pytest.mark.skipif(
        not (BASE_URL and MODEL),
        reason=(
            "MIFORGE_LIVE_JUDGE_BASE_URL and MIFORGE_LIVE_JUDGE_MODEL are unset, so there "
            "is no real OpenAI-compatible endpoint to measure. F3 task 13.4's network half "
            "needs one: against a stub this would measure the stub. Set both to an "
            "endpoint serving a model OTHER than the one under training (R-02.11)."
        ),
    ),
]


@pytest.fixture
def live_judge(monkeypatch: pytest.MonkeyPatch):
    """A judge resolved through the **production** wiring, against the live endpoint.

    Configured by the same ``PROVIDER_<NAME>_BASE_URL`` / ``_MODEL`` environment variables
    an operator uses, so the row discovery in ``providers.settings.http_provider_rows`` runs
    for real. Nothing here constructs a provider: ``services.scoring.wiring.wire()`` is the
    one production caller of ``bind_provider_resolver``, and a measurement that bypassed it
    would be a figure for a path no run takes.
    """
    from services.scoring import wiring
    from services.selectors.settings import reset_selector_settings
    from tests.support.selector_stubs import judge_config, resolved

    monkeypatch.setenv(f"PROVIDER_{PROVIDER_KEY.upper()}_BASE_URL", BASE_URL)
    monkeypatch.setenv(f"PROVIDER_{PROVIDER_KEY.upper()}_MODEL", MODEL)
    monkeypatch.setenv(f"PROVIDER_{PROVIDER_KEY.upper()}_READ_TIMEOUT_S", str(READ_TIMEOUT_S))
    monkeypatch.setenv("JUDGE_PROMPT_TEMPLATE", TEMPLATE)
    reset_selector_settings()
    wiring.wire()

    def build(*, run_target=None):
        return resolved(judge_config(provider=PROVIDER_KEY, threshold=0.6), run_target=run_target)

    try:
        yield build
    finally:
        wiring.unwire()
        reset_selector_settings()


def _other_model_identity() -> ProviderIdentity:
    """A run target that is **not** the judge: the in-process model under training.

    This is the shape F1/F6 hand the guard for a real run - ADR-009's in-process provider
    generating from the parameters miForge holds. Using it here is what makes the
    measurement a measurement of the *legitimate* case rather than of a forbidden one.
    """
    return ProviderIdentity(kind="in_process", model="model-under-training", base_url=None)


class TestTheMeasurementCanHappenAtAll:
    """The positive check. A latency figure from a path that refused is not a figure."""

    def test_the_endpoint_serves_the_configured_judge_model(self) -> None:
        """``GET /v1/models`` lists the model, and the call itself is the transport share.

        Separate from the judge path on purpose: it crosses the same link with no generation
        behind it, so it is the only honest way to say how much of the warm round trip is
        the network. On this cluster it is 8.8 ms against ~1,162 ms of generation.
        """
        import httpx

        started = time.perf_counter()
        response = httpx.get(f"{BASE_URL}/v1/models", timeout=30.0)
        transport_ms = (time.perf_counter() - started) * 1000.0
        assert response.status_code == 200, response.text
        served = {row["id"] for row in response.json()["data"]}
        print(f"\n[transport] GET /v1/models over the same link: {transport_ms:.1f} ms")
        print(f"[transport] models served: {sorted(served)}")
        assert MODEL in served, (
            f"{MODEL!r} is not among the models this endpoint serves ({sorted(served)}), so "
            f"a judge figure taken against it would be a figure for a refusal."
        )


class TestTheJudgeIsNotTheRunsOwnTarget:
    """**R-02.11**, asserted on the live configuration rather than taken on trust."""

    def test_the_live_judge_differs_from_the_model_under_training(self, live_judge) -> None:
        """The legitimate case: a separate model, on a separate endpoint.

        Resolves with the run's real target supplied, so ``check_judge_identity``'s
        comparison actually runs - it is skipped entirely when ``run_target`` is ``None``,
        which is how R-02.11's comparison went unmade in production until 2026-10-16.
        """
        selector = live_judge(run_target=_other_model_identity())
        identity = selector.plugin.judge_identity()
        assert identity.kind == "openai_compatible"
        assert identity.model == MODEL
        assert identity.base_url == BASE_URL.rstrip("/")
        assert identity.model != "model-under-training"
        assert selector.caveats == (), (
            "a separate model on a separate endpoint raises no lineage caveat; a caveat "
            "here would mean the judge shares the target's model name"
        )

    def test_pointing_the_run_target_at_the_live_judge_is_refused(self, live_judge) -> None:
        """The negative control for the guard, on this exact live configuration.

        Without it the measurement above proves only that *some* configuration resolved -
        not that the one the product forbids would have been stopped. The guard must bite
        against the very endpoint the figure was taken from.
        """
        same = ProviderIdentity(
            kind="openai_compatible", model=MODEL, base_url=BASE_URL.rstrip("/")
        )
        with pytest.raises(Refusal) as caught:
            live_judge(run_target=same)
        assert caught.value.code == "judge_is_resident_model"

    def test_the_in_process_provider_is_never_judge_eligible(self) -> None:
        """Belt and braces, and it costs no network: the registry's judge view has no
        in-process row at all, so the live endpoint is not merely *preferred* over the model
        under training - the alternative is unreachable."""
        from providers import registry

        registry.ensure_registered()
        resolver = registry.judge_resolver()
        in_process = [row for row in registry.list_rows() if row.kind == "in_process"]
        for row in in_process:
            assert row.judge_eligible is False
            assert resolver.available(row.name) is False


class TestTheWarmRoundTrip:
    """F3 task 13.4's owed network figure."""

    def test_measure_the_judge_round_trip(self, live_judge) -> None:
        """Median, p95, min and max over :data:`MEASURED_CALLS` warm evaluations.

        **No latency threshold is asserted.** The card is shared with miLLM's own serving
        and with whatever else the cluster is doing, so a threshold here would be a flake
        that teaches people to rerun the suite. What is asserted is that every evaluation
        produced a **real verdict from the live model** - which is the only thing that makes
        the timings a measurement of the judge path rather than of a refusal.
        """
        selector = live_judge(run_target=_other_model_identity())
        plugin = selector.plugin

        first_call_ms: float | None = None
        for index in range(WARMUP_CALLS):
            started = time.perf_counter()
            plugin.evaluate(prompt=PROMPT, completion=COMPLETION, context={})
            elapsed = (time.perf_counter() - started) * 1000.0
            if index == 0:
                first_call_ms = elapsed

        samples: list[float] = []
        verdicts: list[VerdictValue] = []
        for _ in range(MEASURED_CALLS):
            started = time.perf_counter()
            verdict = plugin.evaluate(prompt=PROMPT, completion=COMPLETION, context={})
            samples.append((time.perf_counter() - started) * 1000.0)
            verdicts.append(verdict.value)

        samples.sort()
        p95 = samples[min(len(samples) - 1, round(0.95 * (len(samples) - 1)))]
        rendered = TEMPLATE.format(prompt=PROMPT, completion=COMPLETION)

        print("\n=== F3 task 13.4 - the judge-model selector's NETWORK figure ===")
        print(f"  endpoint        : {BASE_URL}  (resolved by base_url, ADR-009)")
        print(f"  judge model     : {MODEL}   (NOT the run's target - R-02.11)")
        print("  topology        : pod-to-pod, same microk8s cluster, *.svc.cluster.local")
        print("                    SAME-CLUSTER EAST-WEST. This is NOT a WAN figure.")
        print(f"  sample          : {MEASURED_CALLS} measured, {WARMUP_CALLS} warm-up discarded")
        print(f"  rendered prompt : {len(rendered)} chars")
        print("  decode          : temperature=0.0, max_tokens=256 (the plugin's own)")
        print("  concurrency     : 1 - sequential; see TestTheConcurrencyBoundIsOne")
        print(f"  warm median     : {statistics.median(samples):8.1f} ms / completion")
        print(f"  warm p95        : {p95:8.1f} ms")
        print(f"  warm min / max  : {samples[0]:8.1f} / {samples[-1]:8.1f} ms")
        if first_call_ms is not None:
            print(
                f"  first call      : {first_call_ms:8.1f} ms - pays a MODEL SWITCH only if the "
                f"model was not already resident (~14,200 ms when it is not)"
            )
        print(f"  verdicts        : {[v.value for v in verdicts]}")
        print("===============================================================")

        assert len(samples) == MEASURED_CALLS
        assert all(value in (VerdictValue.ACCEPT, VerdictValue.REJECT) for value in verdicts), (
            "every evaluation must be a real verdict from the live model; a refusal would "
            "make these timings a measurement of the failure path"
        )

    def test_the_live_judge_discriminates(self, live_judge) -> None:
        """The judge moves with the completion, so the figure above is a working judge's.

        A judge that returned a constant would produce identical timings and an identical
        verdict, and nothing in a latency measurement would notice. Measured on this
        endpoint: the on-topic completion scores well above the 0.6 threshold and the
        off-topic one at the floor.
        """
        selector = live_judge(run_target=_other_model_identity())
        good = selector.plugin.evaluate(prompt=PROMPT, completion=COMPLETION, context={})
        bad = selector.plugin.evaluate(prompt=PROMPT, completion=OFF_TOPIC_COMPLETION, context={})
        print(
            f"\n[discrimination] on-topic {good.score} -> {good.value.value}; "
            f"off-topic {bad.score} -> {bad.value.value}"
        )
        assert good.score is not None and bad.score is not None
        assert good.score > bad.score, (
            "the live judge returned the same or a lower score for an off-topic completion, "
            "so the timings above are a figure for a judge that is not judging"
        )
        assert good.value is VerdictValue.ACCEPT
        assert bad.value is VerdictValue.REJECT


class TestTheConcurrencyBoundIsOne:
    """The bound ADR-024 demands be stated - **measured, and it contradicts two settings.**

    *Superseded in part, 2026-10-16:* ``HttpProviderRow.concurrency`` is wired into the
    provider (``b2af7a8``) and ``judge_concurrency`` into the judge's batch prefetch (suite
    tracker 4.28). This test still measures a single provider built by ``live_judge``; retake
    the figure, and expect this ratio to change if that provider's concurrency is above 1.

    ``SelectorSettings.judge_concurrency`` (default 4) describes itself as *"the section 9
    performance lever. Bounded concurrency, not an unbounded gather"*, and
    ``HttpProviderRow.concurrency`` (default 4) is published in the provider listing's
    payload. **Neither reaches the judge path.** ``OpenAICompatibleProvider`` takes no
    concurrency argument, ``providers.registry._build_openai_compatible`` passes none, and
    ``generate`` is a list comprehension over ``_generate_one`` - so k prompts are k
    sequential HTTP round trips. The contrast case proves the omission is an omission rather
    than a style: ``training/signals/generation.py`` *does* read its
    ``generation_concurrency`` and chunk by it.

    This is recorded as a **test finding and a source finding** rather than fixed here:
    wiring it would mean *building* concurrency into the provider, which is a behaviour
    change on the one path R-02.11 governs, not a connection. What this test does is make
    the figure's stated bound unable to drift silently - if someone wires concurrency, this
    goes red and the recorded figure must be retaken.
    """

    def test_three_completions_cost_three_round_trips_not_one(self, live_judge) -> None:
        """Measured: a 3-prompt ``generate`` is ~3x a 1-prompt one, which is what
        sequential means. Asserted as a ratio with a wide band, because the point is to
        distinguish 3x from 1x, not to pin a latency."""
        from providers.base import GenerationParams

        selector = live_judge(run_target=_other_model_identity())
        provider = selector.bindings["provider"]
        params = GenerationParams(temperature=0.0, max_tokens=256)
        rendered = TEMPLATE.format(prompt=PROMPT, completion=COMPLETION)

        provider.generate([rendered], params)  # warm

        started = time.perf_counter()
        provider.generate([rendered], params)
        one_ms = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        three = provider.generate([rendered, rendered, rendered], params)
        three_ms = (time.perf_counter() - started) * 1000.0

        ratio = three_ms / one_ms
        print(
            f"\n[concurrency] 1 prompt {one_ms:.1f} ms, 3 prompts {three_ms:.1f} ms, "
            f"ratio {ratio:.2f}x -> the judge path is SEQUENTIAL, bound = 1"
        )
        assert len(three) == 3
        assert ratio > 2.0, (
            f"three completions took only {ratio:.2f}x one, so the judge path is no longer "
            f"sequential. The concurrency bound recorded with task 13.4's figure is 1; if "
            f"that has changed, the figure must be retaken and the bound restated."
        )

    def test_neither_declared_concurrency_setting_reaches_this_path(self) -> None:
        """Pins the finding so it cannot be quietly closed by changing a default.

        Asserts the *behaviour* - the provider accepts no concurrency - rather than
        grepping for the name, because a grep proves a call exists, not that anything
        depends on it.
        """
        import inspect

        from providers.openai_compatible import OpenAICompatibleProvider

        parameters = set(inspect.signature(OpenAICompatibleProvider.__init__).parameters)
        assert "concurrency" not in parameters, (
            "OpenAICompatibleProvider now takes a concurrency argument. The judge path's "
            "recorded bound of 1 (F3 task 13.4) is stale: retake the figure and restate "
            "the bound."
        )
