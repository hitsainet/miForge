"""The one settings block §5.10 reads  (FR-011.49, FR-011.25, FR-011.47; ADR-010, ADR-022).

**Unset means absent-and-normal, not misconfigured.** The base URL, the ``trainer`` token
and the audit label are all unset by default (``core/config.py``, SEAM 2), and with the
base URL unset this mode is reported unavailable **by name** rather than failing later at
an arbitrary moment.

# # Why the unavailable *reason* is ``hot_update_mode_not_available`` and not
# ``millm_not_configured``

This feature's TDD §5.3 lists ``millm_not_configured`` as the code for "the base URL is
unset specifically", on ADR-010's ``<service>_not_configured`` convention. **F1's shipped
surface pins the other spelling**, and it is right to: ``GET /api/v1/models/sources``
reports one ``reason`` per acquisition mode, F1's own tests assert the ``attach`` row reads
``hot_update_mode_not_available``, and one of them asserts that **no** mode row carries a
``*_not_configured`` reason - *"no F1 capability may be unavailable because a sibling is
unconfigured"*. §5.10's mode is unavailable **by requirement** (it is optional) and only
secondarily by configuration, so the mode-level code stays stable and the *cause* travels
beside it: the message names miLLM and the variable to set, and the pre-flight refusal
carries ``cause="millm_not_configured"`` in its details, which is the machine-readable
half. Reported to the operator rather than resolved by editing either side.
"""

from __future__ import annotations

from dataclasses import dataclass

from core.config import PROVISIONAL_VERIFICATION_TOLERANCE, Settings, get_settings

#: What is lost when this mode is unavailable. Named, never implied (R-02.48).
UNAVAILABLE_CAPABILITIES: tuple[str, ...] = (
    "attach to a serving process's resident tensors",
    "update weights in place while they are being served",
)

#: The stable mode-level reason code. See the module docstring for why it is not
#: ``millm_not_configured``.
NOT_AVAILABLE_REASON = "hot_update_mode_not_available"

#: The machine-readable *cause* when the module is present and the base URL is unset.
#: Travels in a refusal's ``details``, never as the mode's ``reason``.
CAUSE_NOT_CONFIGURED = "millm_not_configured"
#: The cause when the token or the audit label is missing. A separate cause because the
#: remedy is a different variable, and *"name what failed"* means naming which one.
CAUSE_IDENTITY_INCOMPLETE = "hot_update_identity_incomplete"


@dataclass(frozen=True)
class HotUpdateSettings:
    """§5.10's configuration, read from the live settings object on every call.

    **Read live, never captured at import.** The reachability test asserts the settings
    come from the live object rather than a module constant: a constant captured at import
    makes an environment change in a test silently ineffective, which is how a standalone
    test ends up asserting against a configuration it did not set.
    """

    base_url: str | None
    trainer_token: str | None
    audit_label: str | None
    #: **R-02.2's tolerance, and it is PROVISIONAL.** 1x10^-2 in log-probability space, as
    #: the *maximum absolute difference* over the compared token set. To be replaced,
    #: before §5.10's acceptance, by the observed spread across repeated identical calls on
    #: unchanged weights times a margin - a measurement this feature owes (FR-011.12,
    #: OQ-1). It is labelled provisional in every recorded result until then: *a provisional
    #: bar that stops reading as provisional is MUT-14 shipping without a mutation.*
    verification_tolerance: float
    provisional_tolerance: float
    #: An override for the interval. **Unset is the normal case**: the interval comes from
    #: the attach response's hint (FR-011.16), because the timeout it must beat is the
    #: serving side's configuration and a local constant drifts from it in silence.
    heartbeat_interval_seconds: float | None
    request_timeout_seconds: float
    #: Deliberately the longest. A late ``commit-end`` is accepted upstream (BRD-01
    #: R-01.23a) and is strictly better than abandoning the bracket.
    commit_end_timeout_seconds: float
    #: The serving side's R-01.22 watchdog, **provisionally 2,000 ms and unbenchmarked**.
    #: Held here only so a commit can be judged against the limit it will be cut off at;
    #: the value itself is the other repository's configuration (FPRD AMB-K).
    commit_watchdog_ms: float
    commit_creep_fraction: float
    verify_prompt: str = "The capital of France is"
    verify_token_set_size: int = 15
    verify_tokenizer: str | None = None
    precommit_retries: int = 1

    @property
    def configured(self) -> bool:
        """Whether a serving process is configured at all. **No network, no GPU.**"""
        return self.base_url is not None

    @property
    def tolerance_is_provisional(self) -> bool:
        """Whether the tolerance in force is still R-02.2's unmeasured starting point."""
        return self.verification_tolerance == self.provisional_tolerance


def load(settings: Settings | None = None) -> HotUpdateSettings:
    """§5.10's settings from the **live** settings object."""
    live = settings if settings is not None else get_settings()
    return HotUpdateSettings(
        base_url=live.millm_base_url,
        trainer_token=live.millm_trainer_token,
        audit_label=live.millm_audit_label,
        verification_tolerance=live.hotupdate_verification_tolerance,
        provisional_tolerance=PROVISIONAL_VERIFICATION_TOLERANCE,
        heartbeat_interval_seconds=live.hotupdate_heartbeat_interval_seconds,
        request_timeout_seconds=live.hotupdate_request_timeout_seconds,
        commit_end_timeout_seconds=live.hotupdate_commit_end_timeout_seconds,
        commit_watchdog_ms=live.hotupdate_commit_watchdog_ms,
        commit_creep_fraction=live.hotupdate_commit_creep_fraction,
        verify_prompt=live.hotupdate_verify_prompt,
        verify_token_set_size=live.hotupdate_verify_token_set_size,
        verify_tokenizer=live.hotupdate_verify_tokenizer,
        precommit_retries=live.hotupdate_precommit_retries,
    )


def unavailable_message(cause: str) -> str:
    """The operator-facing message for an unavailable mode. **Says what to do next.**"""
    if cause == CAUSE_NOT_CONFIGURED:
        return (
            "The 'attach' acquisition mode (BRD-02 §5.10, the optional hot-update mode) "
            "needs a serving process: miLLM is not configured, so set MILLM_BASE_URL to "
            "the process holding the weights you want trained in place. Refusing rather "
            "than loading the same weights from disk - attaching and loading train "
            "different tensors and both report success."
        )
    return (
        "The 'attach' acquisition mode (BRD-02 §5.10) is configured but its identity is "
        "incomplete: set MILLM_TRAINER_TOKEN and MILLM_AUDIT_LABEL. The label is not "
        "decoration - it is what makes a weights change attributable in miLLM's audit log, "
        "so a token without one is a change nobody can attribute (ADR-020)."
    )
