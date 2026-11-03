"""The live integration - **BLOCKED on another repository, and marked so it says it.**

``gpu`` **and** ``upstream``, both deselected by default (``pyproject.toml``). Neither runs
in CI, which has no card and no serving process, so **this is an operator-run gate at
feature acceptance and must not be assumed green.**

**What is actually blocked.** miLLM has implemented none of BRD-01: there are no
``/api/weights/*`` routes anywhere in that repository, and its only ``attach`` is
``millm/services/sae_service.py``, which is SAE attachment and entirely unrelated. Its
Features 031/032/033 are twelve documents and no code. So everything in this file is
written against the contract and **has never been run against an implementation of it.**

**Why the file exists rather than waiting.** A skip that names its reason is the honest form
- it is the one place the suite says out loud that a cross-repository obligation is open,
rather than passing over an absent comparison. The alternative, leaving the live path
unwritten, is how a feature reports itself complete while the only thing that could falsify
it does not exist. Each test below names the acceptance criterion it discharges, so the
operator running them knows what they are signing off.

**One thing here cannot be written yet at all**, and it is recorded rather than faked: the
CUDA IPC handle's serialisation. BRD-01 R-01.1 says *"a CUDA IPC handle"* without fixing an
encoding, and ``rebuild.py`` reads it as the argument set PyTorch's own reducer emits. If
miLLM's implementation encodes it differently, that is the first thing to fail here - which
is the right place for it to fail, and the reason the rebuild refuses rather than coercing.
"""

from __future__ import annotations

import os

import pytest

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.upstream,
    pytest.mark.skipif(
        not os.environ.get("MILLM_BASE_URL"),
        reason=(
            "MILLM_BASE_URL is unset. BRD-01 is specced and UNIMPLEMENTED: miLLM has no "
            "/api/weights/* routes at all, so there is nothing to integrate against. This "
            "is the live cross-repository debt, stated rather than skipped silently."
        ),
    ),
]


@pytest.fixture
def live_client():  # type: ignore[no-untyped-def]
    from training.hotupdate import settings as settings_module
    from training.hotupdate.client import WeightsClient

    client = WeightsClient(settings=settings_module.load())
    try:
        yield client
    finally:
        client.close()


class TestTheSurfaceExistsAtAll:
    def test_the_status_endpoint_answers(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """Gate 14.1. If this fails, nothing below can run and F11 is still blocked.

        ``status`` is the right probe: R-01.20 is ungated, so it answers before any
        credential question arises, and R-01.20 is *"how a restarting trainer discovers a
        torn set it did not create."*
        """
        body = live_client.call("status")
        assert body["weights_state"] in {"clean", "writing", "torn"}

    def test_every_endpoint_F11_calls_exists(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """The endpoint diff, against the implementation rather than the document. This is
        the check that caught the missing load call on paper; run here it catches the
        mirror - a requirement defined and not built."""
        from training.hotupdate.client import OPS

        missing = []
        for op in sorted(OPS):
            try:
                live_client.call(op, {})
            except Exception as exc:
                if "404" in str(exc) or "UNKNOWN_OP" in str(exc):
                    missing.append(op)
        assert missing == [], f"miLLM does not implement: {missing}"


class TestTheReconstruction:
    def test_attach_rebuild_and_verify(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """**AC-011.6 / BRD-01 criterion 1**: a second process rebuilds the model from the
        handles and its forward pass matches miLLM's **log-probabilities over an explicit
        token set** within R-02.2's tolerance - *provisionally* 1x10^-2, to be replaced by
        the observed spread across repeated identical calls times a margin.

        **The spread measurement is owed and is produced here**, not in the fake: it can
        only be observed against a live serving process (OQ-1).
        """
        pytest.skip(
            "14.3: needs the real surface. The comparison also needs a log-probability "
            "capability the inference provider does not have yet - Completion.logprobs "
            "exists and no implementation fills it, which is reported as owed by the "
            "provider layer rather than worked around with a second HTTP path (FR-011.10)."
        )

    def test_a_perturbed_resident_tensor_aborts_the_run(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """The negative control for the above: **MUT-14's live form.** Without it, the
        verification is a check that has only ever been seen to pass."""
        pytest.skip("14.3: needs the real surface.")


class TestTheBracketedRun:
    def test_fifty_steps_with_serving_answering_throughout(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """**AC-011.1 and AC-011.5 (BRD AC9)**: every hold is bounded by the commit, and no
        completion is served from a non-``clean`` set - verified by driving ``/v1`` alongside
        the run rather than by reading the code."""
        pytest.skip("14.4: needs the real surface.")

    def test_killing_miforge_mid_run_leaves_serving_up(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """**AC-011.2 (BRD AC21).** And the restart continues from the last save - which,
        after a process death, needs an operator to clear the stale attachment first
        (FR-011.41), so this test is the one that proves the ordering is right."""
        pytest.skip("14.5: needs the real surface.")

    def test_an_interrupted_commit_goes_torn_and_is_not_retried(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """**AC-011.3 (BRD AC22).** The fake proves F11's half; only this proves that
        miLLM's watchdog, its ``503 WEIGHTS_TORN`` and its late-``commit-end`` acceptance
        behave as R-01.22 and R-01.23 describe."""
        pytest.skip("14.6: needs the real surface.")

    def test_a_quiesce_timeout_before_commit_begin_is_retried(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """**AC-011.4 (BRD AC7)**: the one retryable case, against the real hold timeout."""
        pytest.skip("14.7: needs the real surface.")

    def test_each_refusal_is_produced_against_the_real_surface(self, live_client) -> None:  # type: ignore[no-untyped-def]
        """**AC-011.9.** The fake's refusals are F11's *reading* of BRD-01. This is the only
        thing that can show the reading was right - and an unmapped code arriving here is a
        finding about the contract, not a bug in F11."""
        pytest.skip("14.8: needs the real surface.")
