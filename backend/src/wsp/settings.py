"""WSP settings (FTID §2.1): environment only (PF-17), never runtime-writable through ``/settings``.

``hide_input_in_errors=True`` for the reason ``core/config.py`` records: Pydantic echoes a rejected
value into its error message, and ``WSP_GATEWAY_SERVICE_TOKEN`` is a secret (``repr=False`` covers
``repr`` only, not the error path). There is **no private-key field** (D15, R1B-M9): the gateway's
key pair lives only in the gateway's own Secret.

Standalone by default (ADR-010): with ``WSP_WORKER_URL`` or ``WSP_GATEWAY_URL`` unset, WSP refuses
``wsp_not_configured``; with ``WSP_PROVIDERS_AVAILABLE`` empty, every provider is unavailable; with
``MIWORKBENCH_API_URL`` empty, bundles cannot be imported and ``evaluate`` is ``unmeasured``.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

from wsp.capacity import MIN_WORK_BYTES


class WSPSettings(BaseSettings):
    model_config = SettingsConfigDict(
        extra="ignore", case_sensitive=False, hide_input_in_errors=True
    )

    # Where sessions run. `local` is the dev and test launcher only, refused in prod (P43).
    wsp_launcher: str = ""
    # Feature 019 P3 (F19-D18): a template with ``{index}`` (one address per worker, the
    # StatefulSet's pods under the headless Service), or one URL for a single worker.
    wsp_worker_url: str = ""
    # Feature 019 P3 (F19-D5, ADR-020 as amended): the provisioned maximum, equal to the worker
    # StatefulSet's replicas (a test pins the two). Environment only: raising it is a GitOps push.
    wsp_worker_pool_max: int = Field(default=1, ge=1, le=64)
    # A fresh worker pod per session (operator 2026-11-26, option B of the 019 P4 restart debt):
    # after the driver acks a session (or recycles an idle boot), it deletes the worker's pod, so
    # the StatefulSet makes a new one at once instead of the kubelet restarting the container
    # through CrashLoopBackOff. A template with ``{index}`` naming worker <index>'s pod; empty
    # (the default, and every dev and test launcher) deletes nothing and the container restart of
    # before applies. ``wsp.pod_recycler``; the Role in gitops/pilot/wsp-driver-rbac.yaml grants
    # get and delete on exactly these names.
    wsp_worker_pod_name: str = ""
    wsp_worker_pod_namespace: str = "miforge"
    # Feature 019 P3 (F19-D19, TP-2): a ``pushing`` run whose push stopped reporting progress for
    # this long is failed ``worker_push_failed``; above the push's 300 s per-chunk timeout.
    wsp_push_lease_s: int = Field(default=360, ge=60)
    # 2026-11-27 (run wsp_4fbf5a18758e): how long a queued run on a miServer endpoint that states
    # no window waits for miServer to report one (a model resident) before it fails
    # ``editor_not_resident``; measured from the first empty read
    # (``mf_wsp_runs.editor_wait_since``).
    wsp_editor_resident_wait_s: int = Field(default=900, ge=60)
    wsp_gateway_url: str = ""
    wsp_internal_url: str = ""
    # Providers: a comma-separated list of ids the operator has made available (FR-016.8).
    wsp_providers_available: str = ""
    wsp_default_provider: str = "miwsp_engine"
    # Endpoints and the gateway (P50, P61, D15, D16, D19).
    wsp_local_host_allowlist: str = "miserver-backend.miserver.svc"
    wsp_external_host_allowlist: str = "api.anthropic.com"
    wsp_external_spend_max_usd_per_day: float = 50.0
    wsp_cost_cap_default_usd: float = 10.0
    wsp_cost_cap_max_usd: float = 50.0
    wsp_gateway_public_key: str = ""
    wsp_gateway_service_token: SecretStr | None = Field(default=None, repr=False)
    # Sessions and the driver (P49, P60).
    wsp_session_grace_s: int = 120
    wsp_heartbeat_lapse_s: int = 300
    # Budget defaults for a start body that names none (operator 2026-11-21): by the endpoint's
    # locality. A local editor's are safety stops that catch a runaway, not limits it works
    # against; an external one keeps the earlier defaults, plus the cost cap above. Within the
    # run contract's maxima (`max_steps` 500, `time_s` 86,400). `forward_passes` is not here: it
    # is a method-level budget, 0 unless the body names one.
    wsp_default_tokens_local: int = Field(default=5_000_000, ge=1)
    wsp_default_max_steps_local: int = Field(default=100, ge=1, le=500)
    wsp_default_time_budget_s_local: int = Field(default=7200, ge=60, le=86_400)
    # Feature 020 (F20-D17; FPRD amendment note DEP-1): 5,000,000, the local default (was
    # 100,000). A cached hosted session re-sends its history on every request, and the token
    # budget counts every input token, cache reads included, so 100,000 ended one within a few
    # requests. The cost cap is the hosted bound.
    wsp_default_tokens_external: int = Field(default=5_000_000, ge=1)
    wsp_default_max_steps_external: int = Field(default=50, ge=1, le=500)
    wsp_default_time_budget_s_external: int = Field(default=3600, ge=60, le=86_400)
    wsp_result_pull_timeout_s: int = 300
    wsp_child_log_max_bytes: int = 1_048_576
    # Operator 2026-11-20: 10 (was 3). Four approved runs from one agent and the fourth failed at
    # execution (`wsp_queue_limit_reached`). Kept equal to WSP_MAX_QUEUED_PER_CALLER in
    # gitops/pilot/config.yaml (test_wsp_queue_limit.py reads both).
    wsp_max_queued_per_caller: int = 10
    # Storage (a requirement to provision, not a design limit).
    wsp_bundle_root: Path = Path("/data/wsp")
    wsp_bundle_max_bytes: int = 64 * 1024**3
    # The worker's `/work` volume in bytes: the `work` emptyDir's sizeLimit in
    # gitops/pilot/wsp-worker.yaml, set equal in the miforge-config ConfigMap (a test reads both).
    # A run whose push does not fit is refused `wsp_bundle_too_large_for_worker` at freeze and at
    # dispatch, and the push's `spec.work_max_bytes` (the disk watchdog) derives from it
    # (`wsp.capacity`; production defect 2026-11-20).
    wsp_worker_work_bytes: int = Field(default=64 * 1024**3, gt=MIN_WORK_BYTES)
    # The seconds a forward pass (one observation-tool scoring) costs the worker, as bytes of
    # practice activations read per second: `wsp.capacity.time_refusal` refuses at filing a run
    # whose `time_s` is under half of `forward_passes` such passes (fix 2026-11-21). 4e9 is the
    # benchmark's FASTEST case on two CPUs (page cache warm, backend/scripts/bench_wsp_scorer.py:
    # 4.1 GB in 0.98 s; 1.85 GB/s with the cache dropped), so the estimate errs short and the
    # check refuses only a budget that cannot be met.
    wsp_forward_pass_bytes_per_s: float = Field(default=4.0e9, gt=0)
    # Evaluation (P46, P53, PF-46).
    wsp_scorer_max_calls_per_heldout: int = 50
    # miWorkbench's own score deadline (fix 2026-11-22): it refuses a call still scoring after its
    # `wsp_score_timeout_seconds` (600 by default, at most 3600; a runtime setting there) with 503
    # `score_timeout`. Keep this at least that value: miForge's scorer client waits this plus
    # `clients.miworkbench.SCORE_MARGIN_S`, so the scorer's explicit answer always comes first and
    # the verdict names `score_timeout` rather than `scorer_unreachable`.
    wsp_scorer_deadline_s: float = Field(default=600.0, gt=0, le=3600)
    wsp_min_rows_per_class: int = 20
    wsp_min_false_alarm_rows: int = 1000
    wsp_bootstrap_b: int = 2000
    # Phase flags (PF-19): refused by name until their phase opens.
    wsp_sae_enabled: bool = False
    wsp_checkpoint_edits_enabled: bool = False
    # The sibling that owns bundles and the scorer (P29).
    miworkbench_api_url: str = ""

    def providers_available(self) -> frozenset[str]:
        return frozenset(p.strip() for p in self.wsp_providers_available.split(",") if p.strip())

    def configured(self) -> bool:
        return not self.missing_urls()

    def missing_urls(self) -> list[str]:
        """The unset settings that leave WSP not configured (review OR3-016-2), and (feature 019
        P3) a ``WSP_WORKER_URL`` without ``{index}`` beside a maximum above 1."""
        from wsp import scheduling

        missing = [
            name
            for name, value in (
                ("WSP_WORKER_URL", self.wsp_worker_url),
                ("WSP_GATEWAY_URL", self.wsp_gateway_url),
            )
            if not value
        ]
        if self.wsp_worker_url and not scheduling.worker_urls(self):
            missing.insert(0, "WSP_WORKER_URL")
        return missing


#: The in-cluster values the operator sets at FTASKS 11.2 (the two Services in ``gitops/pilot``).
URL_EXAMPLES = {
    "WSP_WORKER_URL": "http://miforge-wsp-worker-{index}.miforge-wsp-workers.miforge.svc.cluster.local:9000",
    "WSP_GATEWAY_URL": "http://miforge-wsp-gateway:8080",
}


#: Feature 019 P3: why a set ``WSP_WORKER_URL`` is still not configured.
TEMPLATE_RULE = (
    "WSP_WORKER_URL needs {index} when WSP_WORKER_POOL_MAX > 1: one address per worker, e.g. "
    "http://miforge-wsp-worker-{index}.miforge-wsp-workers.miforge.svc.cluster.local:9000."
)


def not_configured_message(missing: list[str], settings: WSPSettings | None = None) -> str:
    """``wsp_not_configured``'s message: what is unset and what to set it to."""
    current = settings or get_wsp_settings()
    if "WSP_WORKER_URL" in missing and current.wsp_worker_url:
        rest = [name for name in missing if name != "WSP_WORKER_URL"]
        tail = "" if not rest else " " + not_configured_message(rest, current)
        return f"WSP is not configured: {TEMPLATE_RULE}{tail}"
    names = " and ".join(missing)
    verb = "is" if len(missing) == 1 else "are"
    values = ", ".join(f"{name}={URL_EXAMPLES[name]}" for name in missing)
    return (
        f"WSP is not configured: {names} {verb} unset, so no run could ever reach the worker. "
        f"The operator sets {values} in the backend's config (the miforge-config ConfigMap) "
        "and restarts the backend and the WSP worker queue."
    )


@lru_cache(maxsize=1)
def get_wsp_settings() -> WSPSettings:
    return WSPSettings()
