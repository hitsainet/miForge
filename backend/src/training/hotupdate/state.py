"""The pre-step precondition, and why it reads an identity  (R-02.30; FR-011.36, FR-011.37).

:func:`require_clean` is a **precondition the step enforces**, not an advisory check a
caller may skip. If ``weights_state`` is not ``clean``, no step begins.

## Why it returns an identity rather than a boolean

miLLM's barrier state lives in **process memory** and a restart begins ``clean`` (miLLM
Feature 031 FR-31.6.1: *"restart starts clean"*), while BRD-01 R-01.23 says ``torn``
*"survives detach and re-attach"*. Those are not the same thing: a serving process that died
mid-commit comes back reporting ``clean`` **while holding the parameter set of its last
load, not of the last step this run applied.** State alone cannot see that. The resident
model name, its manifest hash and the steps-since-load counter can, and comparing them with
what the run believes is what FR-011.37 buys. **MUT-17 removes the identity half and must go
red.**

**The reasoning that makes this safe is written in neither BRD** (FPRD AMB-J), so it is
written here: a freshly loaded model is ``clean`` **by construction** (BRD-01 R-01.14
verifies its hash before a tensor is touched), and the damage from a lost commit is bounded
by the last save. What is *not* safe is continuing as though the resident parameters were
the ones this run last wrote - which is the case this comparison catches.

## A null manifest hash is a statement, never "unknown"

BRD-01 R-01.1 returns ``null`` for the manifest hash when the resident model was
*instantiated from a config* rather than loaded from a checkpoint. **Null means "not loaded
from a checkpoint manifest".** Flattening it into "unknown" is how a comparison silently
passes: two unknowns compare equal.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from core.errors import refuse

log = logging.getLogger(__name__)

CLEAN = "clean"
WRITING = "writing"
TORN = "torn"


@dataclass(frozen=True)
class ResidentIdentity:
    """Which parameter set miLLM is holding, in enough detail to notice a substitution."""

    model: str
    #: ``None`` is an assertion - *not loaded from a checkpoint manifest* (R-01.1) - and is
    #: compared as a value, never treated as a wildcard.
    manifest_sha256: str | None
    steps_since_load: int | None

    def differs_from(self, other: ResidentIdentity) -> list[str]:
        """Which fields disagree. A list rather than a bool so the refusal can name them.

        ``steps_since_load`` is compared **only when both sides state one**: the counter
        moves on every completed commit, so a run that has stepped legitimately disagrees
        with the value it attached at. What it must never do is *decrease* - a lower counter
        means a different load, which is the restarted-serving-process case.
        """
        fields: list[str] = []
        if self.model != other.model:
            fields.append("model")
        if self.manifest_sha256 != other.manifest_sha256:
            fields.append("manifest_sha256")
        if (
            self.steps_since_load is not None
            and other.steps_since_load is not None
            and self.steps_since_load < other.steps_since_load
        ):
            fields.append("steps_since_load")
        return fields


def _identity_from_model_block(block: Any, steps_since_load: Any) -> ResidentIdentity:
    """One reader for the identity object, whichever response carried it.

    miLLM sends the same three fields under two names: ``model`` on the attach response
    (FTDD 031 §5.3) and ``resident_model`` on the status response (FPRD FR-31.16.1). The
    **block** is identical, so the reading of it must be too — and that is not a tidiness
    point. `attach.py` and :func:`read` each had their own derivation, they disagreed, and
    the disagreement meant the first ``require_clean`` of every §5.10 run refused
    ``weights_identity_changed`` and blamed a serving restart that had not happened. Two
    derivations of one value is the second-source-of-truth family this project keeps paying
    for (`DB_RUN_TYPE`, `RUN_TYPES`, `full` vs `change`).
    """
    fields = block if isinstance(block, dict) else {}
    return ResidentIdentity(
        # The NAME, with the id as a fallback: an id is stable only within one installation
        # and this identity is compared across a serving-process restart.
        model=str(fields.get("name") or fields.get("model_id") or ""),
        manifest_sha256=_optional_str(fields.get("manifest_hash")),
        steps_since_load=_optional_int(steps_since_load),
    )


def identity_from_attach(response: Any) -> ResidentIdentity:
    """The identity an **attach** response states, under its ``model`` key."""
    body = response if isinstance(response, dict) else {}
    return _identity_from_model_block(body.get("model"), body.get("steps_since_load"))


def read(client: Any) -> tuple[str, ResidentIdentity]:
    """``(weights_state, identity)`` from **one** status read (BRD-01 R-01.20).

    One call, because R-01.20 carries both and **two calls can straddle a change**: a state
    read that says ``clean`` and an identity read taken after a restart describe two
    different moments and would agree with each other about nothing in particular.
    """
    return interpret_status(read_status(client))


def read_status(client: Any) -> Any:
    """**The one status call**, returning the body for callers that need more than two
    values out of it.

    Factored out rather than widening :func:`read`'s return: a dozen call sites unpack two
    values and one indexes ``[1]``, and R-01.20's one-call property is the thing being
    protected, not the tuple's width. A caller wanting the barrier's pausability *and* the
    state takes the body once and derives both — which is exactly why two calls are wrong
    here: a state read saying ``clean`` and a pausability read taken after a deployment
    change describe two different moments.
    """
    return client.call("status")


def pause_unavailable(body: Any) -> str | None:
    """Why miLLM **cannot pause for a barrier**, before any call is refused for it.

    miLLM reports this as ``pause_unavailable: "cbm" | "concurrency" | null`` on
    ``GET /api/weights/status``, computed by the same function its
    ``409 BARRIER_UNAVAILABLE`` refusal raises from, with a test on their side asserting the
    two agree in every case. Non-null means a quiesce, a weights load and an adapter change
    will **all** be refused — and the condition comes from deployment settings (continuous
    batching, or more than one request slot), so **a retry does not help**. That is what
    makes it a pre-flight fact rather than a transient one.

    **An ABSENT field is "unknown", never "pausable".** The field is additive and currently
    lives on a miLLM branch that is not pushed, so against today's published contract it
    simply is not there — and this project does not build against an unpublished contract.
    Reading it this way costs nothing and is correct under both: present and non-null is a
    refusal, present and null is a go-ahead, absent is no information at all. Collapsing
    absent into "pausable" would be the `getattr` default that always fires (ADR-018), and
    it would turn an older miLLM into a confident wrong answer.
    """
    if not isinstance(body, dict):
        return None
    value = body.get("pause_unavailable")
    return value if isinstance(value, str) and value else None


def interpret_status(body: Any) -> tuple[str, ResidentIdentity]:
    """The pure half of :func:`read`, over a body already fetched."""
    state = body.get("weights_state") if isinstance(body, dict) else None
    if not isinstance(state, str):
        raise refuse(
            "weights_surface_unreachable",
            "miLLM's status response carries no weights_state (BRD-01 R-01.20 requires "
            "it). Refusing rather than assuming 'clean': assuming it is the premise of "
            "every wrong thing this mode could then do.",
            op="status",
        )
    # `resident_model` is an OBJECT - `(model_id, name, manifest_hash)` per miLLM FPRD
    # FR-31.16.1 - not a string, and the hash lives inside it rather than at the top level.
    # F11 did `str(body.get("resident_model") or ...)`, which stringifies the whole dict,
    # and read a top-level `manifest_sha256` that the contract does not have, so it was
    # always None. Both halves failed SILENTLY: a stringified dict is a stable value, so
    # the comparison still "worked" and would have gone on working while comparing the
    # wrong thing. Verified against miLLM's committed FR-31.16.1 at HEAD.
    # `resident_model` on status, `model` on attach - the SAME block, one reader.
    return state, _identity_from_model_block(
        body.get("resident_model"), body.get("steps_since_load")
    )


def require_clean(client: Any, expected: ResidentIdentity | None = None) -> ResidentIdentity:
    """Refuse unless the set is ``clean`` **and** the identity still matches.

    Called in exactly three places (FR-011.36, TID §3.5 hint 3): before the first step of a
    run, after any failed or interrupted commit, and **before a save**. The answer is
    **never cached**, not even for the duration of one step - *a cached answer about another
    process's memory is a guess with a timestamp.*

    Raises:
        Refusal: ``weights_not_clean`` with the state and the identity read;
            ``weights_identity_changed`` when the state is clean and the identity is not.
    """
    state, identity = read(client)
    if state != CLEAN:
        raise refuse(
            "weights_not_clean",
            f"miLLM's parameter set is {state!r}, so no step begins (R-02.30). A step "
            f"against a non-clean set reports a loss for a model that never existed at any "
            f"step. Recovery is the completion of the open commit by the process that "
            f"opened it, or a checkpoint load that replaces the whole set.",
            weights_state=state,
            resident_model=identity.model,
            manifest_sha256=identity.manifest_sha256,
            steps_since_load=identity.steps_since_load,
        )
    if expected is not None and (changed := identity.differs_from(expected)):
        raise refuse(
            "weights_identity_changed",
            f"miLLM reports its parameter set 'clean', but it is not the one this run was "
            f"training: {', '.join(changed)} changed. The barrier state lives in miLLM's "
            f"process memory and a restart begins 'clean', so a serving process that died "
            f"mid-commit comes back reporting clean while holding its last LOAD rather "
            f"than this run's last step. Refusing rather than continuing - the loss curve "
            f"would look perfectly healthy.",
            changed=changed,
            observed={
                "model": identity.model,
                "manifest_sha256": identity.manifest_sha256,
                "steps_since_load": identity.steps_since_load,
            },
            expected={
                "model": expected.model,
                "manifest_sha256": expected.manifest_sha256,
                "steps_since_load": expected.steps_since_load,
            },
        )
    return identity


def _optional_str(value: Any) -> str | None:
    """``None`` stays ``None``; anything else becomes a string. **No default.**"""
    return None if value is None else str(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
