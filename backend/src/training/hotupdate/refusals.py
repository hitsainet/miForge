"""miLLM's refusals, mapped by name - and the case the map does not cover  (R-02.33).

R-02.33's table has four rows. **The fifth branch is the load-bearing one**: a refusal code
with no entry is reported **verbatim** with its code, its HTTP status and miLLM's own
message, and the run stops. ADR-018's rule - refuse rather than fabricate - applies to a
*diagnosis* as much as to a value: *a wrong diagnosis is worse than an unhelpful one,
because it ends the investigation.*

**No upstream code is re-exported as miForge's own.** Every code this module returns is in
miForge's namespace (TDD §5.3); miLLM's code travels in ``details.upstream_code``.
Re-exporting one would make miForge's API depend on another repository's string table.

## What each row cites

================================  ================================  ====================
miLLM's refusal                   miForge's code                    Source
================================  ================================  ====================
``409 ALREADY_ATTACHED``          ``attachment_held_by_other``      R-01.2
``400 NOT_ATTACHABLE``            ``resident_model_not_attach..``   R-01.5
stale attachment                  ``attachment_stale_operat..``     R-01.4, R-01.29
``409 WEIGHTS_NOT_CLEAN``/TORN    ``weights_not_clean``             R-01.7, R-01.22
``403 NOT_ATTACHMENT_HOLDER``     ``attachment_not_holder``         R-01.31
``409 NOT_ATTACHED``              ``weights_not_attached``          R-01.31
``401``/``403`` credential        ``weights_identity_rejected``     R-02.32 (identity.py)
anything else                     ``weights_refusal_unmapped``      ADR-013, ADR-018
================================  ================================  ====================

## Never work around a refusal (R-02.34)

Not one of these is retried, and **no second ``attach`` is attempted after an
already-attached refusal.** Each names another process's state, and *an optimizer that
clears someone else's attachment to get started is how two trainers end up writing the same
tensors.* The already-attached row reports **who** holds it, from the status read: a refusal
that says "someone else is attached" without saying who cannot be acted on.
"""

from __future__ import annotations

from typing import Any

from core.errors import Refusal, refuse

#: miLLM's code -> (miForge's code, whether a status read should name the holder).
_MAP: dict[str, str] = {
    "ALREADY_ATTACHED": "attachment_held_by_other",
    "NOT_ATTACHABLE": "resident_model_not_attachable",
    "WEIGHTS_NOT_CLEAN": "weights_not_clean",
    "WEIGHTS_TORN": "weights_not_clean",
    "NOT_ATTACHMENT_HOLDER": "attachment_not_holder",
    "NOT_ATTACHED": "weights_not_attached",
    "ATTACHMENT_NOT_STALE": "weights_refusal_unmapped",
    # A commit-begin after miLLM's hold timeout had already resumed serving (BRD AC7). Seen
    # live 2026-10-18 (F11 14.7): it was logged `weights_refusal_unmapped` while the retry
    # around it worked - the right action under the wrong name.
    "NOT_QUIESCED": "weights_not_quiesced",
    # `ATTACHMENT_STALE` was here and is REMOVED, not kept "just in case": miLLM emits no
    # such code, so the entry was dead and its presence is what hid the real path. A map
    # row for a code nobody sends is a second source of truth about the contract.
}

_MESSAGES: dict[str, str] = {
    "attachment_held_by_other": (
        "Another process holds miLLM's weight attachment, so this run cannot attach "
        "(miLLM allows one at a time). miForge will NOT force a detach: the holder "
        "may be mid-commit, and interrupting it leaves the parameter set torn and serving "
        "refused. Stop the other trainer, or wait for it to detach."
    ),
    "resident_model_not_attachable": (
        "miLLM's resident model is not attachable: the tensor layout must be plain FP16 or "
        "BF16 (miLLM refuses a quantized model). Load an unquantized model in the "
        "serving process, or train from disk with the 'load' acquisition mode instead."
    ),
    "attachment_stale_operator_action_required": (
        "miLLM reports its weight attachment as stale, which blocks a new attach until it "
        "is cleared. AN OPERATOR MUST CLEAR IT: miForge "
        "holds a 'trainer' identity and cannot perform the clear, and does not contain the "
        "call at all. Check the weight state before inheriting anything - a stale "
        "attachment can be holding a torn parameter set."
    ),
    "weights_not_clean": (
        "miLLM's parameter set is not clean - a write is in progress or was interrupted "
        "No step begins against it. Recovery is either the completion of "
        "the open commit by the process that opened it, or a checkpoint load that replaces "
        "the whole set."
    ),
    "weights_not_quiesced": (
        "miLLM refused the call because serving is not quiesced: its hold timeout expired "
        "and it resumed serving on its own before this step's commit-begin. "
        "Nothing was written, so the step is retried from unchanged weights and the "
        "timeout is recorded."
    ),
    "attachment_not_holder": (
        "miLLM refused this call because it did not arrive from the holding attachment "
        "(403 NOT_ATTACHMENT_HOLDER). The attachment id is returned once "
        "and kept with the run's state; if this run lost it, it cannot commit and must be "
        "restarted from its last checkpoint."
    ),
    "weights_not_attached": (
        "miLLM has no weight attachment, so a holder-only call cannot be served "
        "(409 NOT_ATTACHED). The attachment was lost or detached - most likely the "
        "serving process restarted, which does not preserve it."
    ),
}


def attachment_refusal_from_status(attachment: Any, *, holder: str | None) -> Refusal | None:
    """R-02.33's first and third rows, read off a STATUS body before any attach is sent.

    ``attachment`` is the status body's ``attachment`` object (``None`` when nothing is
    attached). Returns the same named refusal ``interpret`` gives for miLLM's ``409
    ALREADY_ATTACHED`` - held, or stale with the operator action - so pre-flight and the
    attach itself cannot disagree about what to tell the operator. Found live 2026-10-18
    (F11 14.8): ``POST /runs/validate`` answered ``ok`` over a stale attachment, and the
    refusal arrived only from the worker, after a lease and a queue slot.
    """
    if not isinstance(attachment, dict):
        return None
    stale = attachment.get("state") == "stale"
    code = "attachment_stale_operator_action_required" if stale else "attachment_held_by_other"
    details: dict[str, Any] = {
        "op": "attach",
        "stage": "preflight",
        "holder": holder,
        "attachment_state": attachment.get("state"),
        "attached_at": attachment.get("attached_at"),
        "retried": False,
    }
    if stale:
        details["operator_action"] = "POST /api/weights/clear-stale (operator identity)"
        details["miforge_can_perform_it"] = False
    return refuse(
        code,
        f"{_MESSAGES[code]} Refused at pre-flight, before a lease and before any allocation.",
        **details,
    )


def interpret(status: int, body: Any, *, op: str, holder: str | None = None) -> Refusal:
    """One non-2xx response as a named miForge refusal. **Never a guess.**

    Args:
        status: miLLM's HTTP status.
        body: the parsed body, whatever shape it came in. Read defensively because an
            unparseable refusal is itself a refusal - never a partially read object.
        op: the call that was refused, so the message names what failed.
        holder: the holding identity from a status read, for the already-attached row.
    """
    upstream_code = _code_of(body)
    upstream_message = _message_of(body)
    upstream_reason = _reason_of(body)
    mapped = _MAP.get(upstream_code or "")
    clears = clears_on_its_own(upstream_code, upstream_reason)

    if mapped is None or mapped == "weights_refusal_unmapped":
        # THE BRANCH THAT MATTERS. Verbatim, and the run stops. Mapping an unknown refusal
        # onto the nearest familiar one is a fabricated diagnosis (ADR-018), and a wrong
        # diagnosis ends the investigation.
        return refuse(
            "weights_refusal_unmapped",
            f"miLLM refused the {op!r} call with HTTP {status} and a code this build does "
            f"not have a mapping for: {upstream_code or '<none>'}. Reported verbatim rather "
            f"than mapped onto the nearest familiar case, and the run stops. miLLM's own "
            f"message: {upstream_message or '<none>'}" + _clearing_note(clears, op),
            op=op,
            status=status,
            upstream_code=upstream_code,
            upstream_message=upstream_message,
            upstream_reason=upstream_reason,
            upstream_details=_upstream_details(body),
            clears_on_its_own=clears,
            retry_unsafe_op=op in RETRY_UNSAFE_OPS,
        )

    details: dict[str, Any] = {
        "op": op,
        "status": status,
        "upstream_code": upstream_code,
        "upstream_message": upstream_message,
        # miLLM states a STRUCTURED reason beside the code, and this build was discarding
        # it. That matters most for `NOT_ATTACHABLE`, whose message below asserts a cause
        # ("the tensor layout must be plain FP16 or BF16"). Against BRD-01 as committed that
        # assertion is right - R-01.5 and acceptance criterion 5 make a quantized model the
        # only case - but miLLM's implementation already sends other reasons, and asserting
        # a cause while throwing away the one the upstream stated is the guess ADR-018
        # forbids, dressed as a diagnosis. Carried always, and the message defers to it.
        "upstream_reason": upstream_reason,
        "upstream_details": _upstream_details(body),
        # Information for the CALLER, never a branch in this module. See CLEARS_ON_ITS_OWN.
        "clears_on_its_own": clears,
        "retry_unsafe_op": op in RETRY_UNSAFE_OPS,
        "retried": False,
    }
    if mapped == "attachment_held_by_other" and _detail_of(body, "state") == "stale":
        # ⚠ A STALE ATTACHMENT DOES NOT ARRIVE AS ITS OWN CODE. `_MAP` carried
        # `ATTACHMENT_STALE`, which **miLLM emits nowhere** - verified by grepping their
        # whole tree. What really arrives is `409 ALREADY_ATTACHED` with
        # `details.state == "stale"` (their `_already_attached` computes exactly
        # `"stale" if self._is_stale(...) else "active"`).
        #
        # So this build gave the WRONG ADVICE for the case it had written the right advice
        # for: `attachment_held_by_other` says *"Stop the other trainer, or wait for it to
        # detach"* - and the holder is already gone, so waiting never clears it. Only an
        # operator's `clear-stale` does, which miForge's `trainer` identity cannot call.
        # The carefully written correct message existed and was unreachable.
        mapped = "attachment_stale_operator_action_required"
    if mapped in ("attachment_held_by_other", "attachment_stale_operator_action_required"):
        # R-02.33's first row says to report the holder, and its third ("report it") needs
        # the same fact. miLLM's `409 ALREADY_ATTACHED` names the holder IN ITS DETAILS
        # (`role`, `label`, `state`, `attached_at`), so that is read first; the status read
        # the client passes is the fallback. `None` is carried through as a stated fact:
        # "miLLM did not say who" is different from "nobody asked".
        role, label = _detail_of(body, "role"), _detail_of(body, "label")
        if label:
            details["holder"] = f"{role}:{label}" if role else str(label)
        else:
            details["holder"] = holder
    if mapped == "attachment_stale_operator_action_required":
        details["operator_action"] = "POST /api/weights/clear-stale (operator identity)"
        details["miforge_can_perform_it"] = False
    message = _MESSAGES[mapped]
    if upstream_reason:
        # NAMES what miLLM said rather than replacing the mapped message: the mapped text
        # carries the action to take, and the reason carries the fact. A wrong diagnosis
        # ends an investigation, so both go in the one message.
        message = (
            f"{message} miLLM states the reason as {upstream_reason!r} - if that does not "
            f"match the cause named above, THE REASON IS AUTHORITATIVE and this build's "
            f"mapping is behind miLLM's contract."
        )
    # The clearing note belongs on a MAPPED refusal too, not only the unmapped branch.
    # `NOT_ATTACHABLE` reason `weights_writing` is the case that proves it: the code IS
    # mapped, so without this the details said the condition clears and the message - the
    # part an operator reads - did not.
    message += _clearing_note(clears, op)
    return refuse(mapped, message, **details)


#: Upstream conditions that **clear without intervention**, and why. Keyed by miLLM's code,
#: or ``CODE/reason`` where the code alone does not decide it.
#:
#: ⚠ THIS IS NOT A RETRY TABLE AND MUST NEVER BECOME ONE. `client.py`'s prohibition is one
#: line and absolute: *"a refusal is never retried - no backoff, no jitter, no 'maybe
#: transient' branch (R-02.34, FR-011.35)"*. What this fixes is a different cost: five of the
#: six refusals miLLM added fall to the honest `weights_refusal_unmapped` branch, which stops
#: the run **permanently** - and two of them mean *miLLM is mid-save or mid-load and will
#: finish*. Stopping forever where waiting is correct is a real loss, and the fix is to SAY
#: SO in the refusal so the operator decides. The client still never retries, and
#: :data:`RETRY_UNSAFE_OPS` says where even a caller must not.
CLEARS_ON_ITS_OWN: dict[str, str] = {
    "PAUSE_IN_PROGRESS": (
        "another serving pause is in flight and will finish; nothing is wrong with this request"
    ),
    "MODEL_BUSY/weights_operation": (
        "miLLM is mid-save or mid-load of the parameter set; the operation completes and the "
        "model stops being busy"
    ),
    "NOT_ATTACHABLE/weights_writing": (
        "a write to the parameter set is in progress; it finishes and the set becomes "
        "attachable again"
    ),
    "NOT_QUIESCED": (
        "a quiesce is still draining in-flight requests; it completes on its own, bounded by "
        "miLLM's hold timeout"
    ),
}

#: Ops where **even the caller** must not repeat the call, whatever the condition. A doubled
#: write is worse than a stalled run: `commit-end` twice can apply a non-idempotent update
#: twice, and `save` twice can publish two checkpoints claiming one step. The run goes through
#: R-02.30's recovery instead of repeating the call.
RETRY_UNSAFE_OPS: frozenset[str] = frozenset({"commit-begin", "commit-end", "save", "load"})


def clears_on_its_own(upstream_code: str | None, upstream_reason: str | None) -> str | None:
    """Why this upstream condition resolves without intervention, or ``None``.

    ``None`` means **unknown**, never "terminal". A code this build has not classified is
    exactly what the unmapped branch exists to report verbatim, and guessing an unknown
    condition is permanent ends an investigation as surely as guessing it is transient starts
    a wrong one.
    """
    if not upstream_code:
        return None
    if upstream_reason:
        keyed = CLEARS_ON_ITS_OWN.get(f"{upstream_code}/{upstream_reason}")
        if keyed is not None:
            return keyed
    return CLEARS_ON_ITS_OWN.get(upstream_code)


def _clearing_note(clears: str | None, op: str) -> str:
    """The sentence appended when miLLM's condition resolves by itself.

    Empty when it does not, so the message is byte-unchanged for every other refusal.
    """
    if not clears:
        return ""
    note = (
        f" THIS CONDITION CLEARS WITHOUT INTERVENTION: {clears}. The run is still stopped - "
        f"miForge never retries a refusal, because a blind retry of a "
        f"weights call can double a write - but the submission can be repeated once it has "
        f"cleared."
    )
    if op in RETRY_UNSAFE_OPS:
        note += (
            f" ⚠ NOT by repeating {op!r} itself, which is not safe to send twice; go through "
            f"the commit recovery."
        )
    return note


def _upstream_details(body: Any) -> dict[str, Any] | None:
    """miLLM's whole ``error.details`` object, bounded - the part that NAMES what failed
    (e.g. ``INVALID_MANIFEST``'s per-field ``errors``). Dropped until 2026-10-17, so a live
    refusal said "the details name each failing field" and carried none (F11 window 2)."""
    error = body.get("error") if isinstance(body, dict) else None
    details = error.get("details") if isinstance(error, dict) else None
    if not isinstance(details, dict) or not details:
        return None
    import json

    text = json.dumps(details, default=str, sort_keys=True)
    return details if len(text) <= 4000 else {"truncated": text[:4000]}


def _detail_of(body: Any, key: str) -> str | None:
    """One string from miLLM's error ``details``, wherever the envelope puts it.

    Defensive for the reason the module docstring gives for the code and the message: an
    unparseable refusal is itself a refusal, never a partially read object. A miLLM build
    that moves a field makes it **absent**, never makes this raise.
    """
    if not isinstance(body, dict):
        return None
    for container in (body.get("error"), body):
        if not isinstance(container, dict):
            continue
        for holder in (container.get("details"), container):
            if isinstance(holder, dict):
                value = holder.get(key)
                if isinstance(value, str) and value:
                    return value
    return None


def _reason_of(body: Any) -> str | None:
    """miLLM's structured ``reason``, wherever in the envelope it sits.

    Read defensively for the reason the module docstring already gives for the code and the
    message: an unparseable refusal is itself a refusal, never a partially read object. The
    reason lives in the error's ``details`` on miLLM's side, so the nesting is checked
    rather than assumed - and a build of miLLM that moves it does not make this raise, it
    makes the reason absent, which is reported as absent.
    """
    return _detail_of(body, "reason")


def _code_of(body: Any) -> str | None:
    """miLLM's refusal code, from either envelope shape, or ``None``.

    Both shapes are read because BRD-01 names codes (``409 ALREADY_ATTACHED``) without
    fixing an envelope, and miForge must not refuse to *understand* a refusal because the
    wrapper differed. What it must never do is **invent** one: an absent code produces
    ``None`` and the unmapped branch, not a default.
    """
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if isinstance(error, dict):
        code = error.get("code")
        return str(code) if code is not None else None
    for key in ("code", "detail"):
        value = body.get(key)
        if isinstance(value, str):
            return value
    return None


def _message_of(body: Any) -> str | None:
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    if isinstance(error, dict):
        message = error.get("message")
        return str(message) if message is not None else None
    message = body.get("message")
    return str(message) if isinstance(message, str) else None
