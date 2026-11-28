"""The worker's restatement of the verdict (``wsp_worker.judging``, fix 2026-11-22) equals the
verdict itself. The worker imports nothing from ``wsp`` (FTDD §6.1's fence), so the judging
section's facts are literals there; a change to ``wsp.verdict`` that this misses would tell the
editor a rule the verdict no longer applies."""

from __future__ import annotations

from wsp import evaluate, verdict
from wsp_worker import judging


def test_the_judging_constants_are_the_verdicts() -> None:
    assert judging.REFITS == verdict.REFITS
    assert judging.OPERATING_POINT == verdict.OPERATING_POINT
    assert judging.FALSE_ALARM_ROLES == verdict.FALSE_ALARM_ROLES
    assert judging.CI == (verdict.CI_HIGH - verdict.CI_LOW) / 100
    assert verdict.BASELINES == judging.REFITS + judging.OPERATING_POINT


def test_the_primary_metric_is_the_verdict_documents() -> None:
    document = evaluate._base_document(
        "unmeasured",
        [],
        request_sha256="0" * 64,
        control_seed=0,
        control_figures={},
        settings=__import__("wsp.settings", fromlist=["WSPSettings"]).WSPSettings(),
        call_id="",
        metrics_agree=False,
        now=evaluate.utcnow(),
    )
    assert document["primary_metric"] == judging.PRIMARY_METRIC
