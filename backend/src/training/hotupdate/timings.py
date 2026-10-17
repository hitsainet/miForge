"""The two §5.10 step-record fields, and the number another repository's gate reads.

**FR-011.57 - FR-011.59; R-02.14.** This module populates two fields it does **not**
define: ``quiesce_hold_ms`` and ``commit_duration_ms`` live in
``docs/schemas/step-record-v1.json`` (foundation 3.3), are written *present and null* by
every standalone step, and are filled only here. F11 adds no field and validates against
the contract on every write.

## ``commit_duration_ms`` is not local telemetry

**miLLM Feature 31's commit watchdog is set to ten times the measured p99.9 of this value**
(BRD-01 §7's resolution). So the field is a cross-repository product obligation, and
``run_log.commit_duration_percentile`` - F1's, reading the authoritative log - is the query
that serves it. :func:`report` is the form that may be handed over: a percentile **with its
sample size and its conditions**, never a bare millisecond figure. *Every number names its
scale and sample* (ADR-012), and this one is read by a gate in another project.

## The hold is measured locally, and the requirement says otherwise

R-02.14 asks for the *"quiesce hold time **as reported by miLLM**"*. **No response on that
surface carries a hold duration** (FPRD DEF-B): BRD-01 R-01.6's quiesce response carries
the number of requests held and the current ``weights_state``, and miLLM's Feature 031 adds
a quiesce timestamp and a hold **deadline** - and a deadline is a limit, not a hold. So the
field is produced by measurement, from the quiesce response and miForge's own monotonic
clock, and **the attribution is not**. Confirmed as the reading rather than amended into the
BRD; reported to the operator (OQ-2). *Put here so the next reader does not go looking for
the field.*

## The creep warning

A commit that exceeds a configured fraction of the watchdog is logged as a warning. *A run
whose commits creep toward the watchdog is the early warning that the watchdog is set too
tight* - surfacing it is part of the requirement rather than an extra, because the
alternative is discovering it as a spurious ``torn`` set during serving.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from core.errors import refuse
from training import run_log

log = logging.getLogger(__name__)

#: Both links on this node are **x4, not x16** (ADR-024). Carried with every report,
#: because *a number quoted without its topology is not a measurement* - a host-to-device
#: figure taken from an x16 number is wrong by three to six times.
TOPOLOGY = "PCIe x4 on both links (not x16); one RTX 3090, time-sliced with serving, no MPS"


@dataclass
class StepTimings:
    """The hold and the commit duration for one run, step by step.

    Args:
        watchdog_ms: miLLM's R-01.22 commit watchdog, **provisionally 2,000 ms and
            unbenchmarked**. Held only so a duration can be judged against the limit it
            will be cut off at: *a duration without its limit cannot be judged.*
        creep_fraction: warn above this fraction of the watchdog.
    """

    watchdog_ms: float
    creep_fraction: float
    run_id: str | None = None
    #: Every commit's duration, in order. The distribution, not the last value: the gate
    #: another repository sets reads a p99.9, and a last-value field cannot produce one.
    commits_ms: list[float] = field(default_factory=list)
    holds_ms: list[float] = field(default_factory=list)
    #: The quiesce responses' held-request counts, kept beside the holds because a hold of
    #: 40 ms holding nothing and one holding 200 requests are different facts.
    held_requests: list[int | None] = field(default_factory=list)
    #: How many brackets got past ``commit-begin``. **The only honest test for "was anything
    #: written"** from this process (recovery.Interruption.commit_begun): a refusal raised
    #: before this moved is the one retryable case (BRD AC7), one raised after is never
    #: retried (R-02.31).
    commits_begun: int = 0

    def record_hold(self, hold_ms: float, quiesce_response: dict[str, Any] | None = None) -> None:
        """One quiesce hold, **measured here** - see the module docstring on DEF-B."""
        self.holds_ms.append(hold_ms)
        held = None
        if isinstance(quiesce_response, dict):
            value = quiesce_response.get("held_requests", quiesce_response.get("held"))
            held = int(value) if isinstance(value, int) else None
        self.held_requests.append(held)

    def record_commit(self, commit_ms: float) -> None:
        """One commit duration, and the creep warning when it approaches the watchdog."""
        self.commits_ms.append(commit_ms)
        threshold = self.watchdog_ms * self.creep_fraction
        if commit_ms >= threshold:
            log.warning(
                "commit duration is creeping toward miLLM's watchdog",
                extra={
                    "code": "commit_duration_creep",
                    "run_id": self.run_id,
                    "commit_duration_ms": round(commit_ms, 3),
                    "watchdog_ms": self.watchdog_ms,
                    "fraction_of_watchdog": round(commit_ms / self.watchdog_ms, 4),
                    # A warning that did not say what it was measured against would be
                    # unactionable: the watchdog is the other repository's configuration.
                    "action": (
                        "a commit that reaches the watchdog is declared torn and serving is "
                        "refused; produce the p99.9 distribution and raise the watchdog"
                    ),
                },
            )

    @property
    def last(self) -> dict[str, float | None]:
        """The two §5.10 fields for the step just completed, by the schema's own names.

        ``None`` for a step that had no bracket - which is every standalone step, and is
        why the default path writes them present and unset rather than omitting them.
        """
        return {
            "quiesce_hold_ms": self.holds_ms[-1] if self.holds_ms else None,
            "commit_duration_ms": self.commits_ms[-1] if self.commits_ms else None,
        }


def fill(record: dict[str, Any], timings: StepTimings) -> dict[str, Any]:
    """Put the two §5.10 fields into F1's step block, **and validate against the contract**.

    The block comes from ``run_log.step_block`` with both fields present and null; this sets
    them. **It adds no key** - it refuses if it would, because the schema closes the block
    to unknown keys and a new key here would be F11 defining a field it does not own.

    Raises:
        Refusal: ``step_record_invalid`` from ``run_log.validate_block`` when the filled
            block does not validate. **MUT-15 drops the two fields and this is what notices**
            - along with the test that asserts they are present, since dropping them
            validates cleanly (both are nullable) and silently removes another repository's
            acceptance input.
    """
    filled = dict(record)
    for key, value in timings.last.items():
        if key not in record:
            raise refuse(
                "step_record_field_unknown",
                f"{key!r} is not a field of the step block, so §5.10 cannot fill it. "
                f"docs/schemas/step-record-v1.json owns these two fields and F11 defines "
                f"neither; if the contract moved, the fix is there and not here.",
                field=key,
                known=sorted(record),
            )
        filled[key] = value
    run_log.validate_block(filled)
    return filled


def report(
    log_path: Any,
    *,
    percentile: float = 99.9,
    conditions: str = "unknown",
) -> dict[str, Any]:
    """The commit-duration figure as it may be handed over (FR-011.59, AC-011.16).

    **Never a bare millisecond number.** The percentile, its sample size, the conditions it
    was measured under and this node's topology, together - because miLLM Feature 31's
    acceptance gate is ten times this figure, and ten times a number measured in isolation
    on a link nobody named is not a watchdog.

    Returns a mapping with ``percentile``, ``value_ms`` (``None`` when nothing was
    measured - **not zero**, since zero would be an instantaneous commit and would hand
    miLLM a watchdog of nothing), ``sample_size``, ``conditions``, ``topology`` and
    ``watchdog_gate_ms`` (ten times the value, the figure the gate consumes).
    """
    value = run_log.commit_duration_percentile(log_path, percentile)
    sample = sum(
        1
        for record in run_log.read_blocks(log_path, block="step")
        if record.get("commit_duration_ms") is not None
    )
    return {
        "percentile": percentile,
        "value_ms": value,
        "sample_size": sample,
        "conditions": conditions,
        "topology": TOPOLOGY,
        "watchdog_gate_ms": None if value is None else value * 10.0,
        "note": (
            "miLLM Feature 31's commit watchdog is set to >= 10x the measured p99.9 under "
            "concurrent serving (BRD-01 §7). A distribution from fifty steps is not one: "
            "report the sample size with the figure."
        ),
    }
