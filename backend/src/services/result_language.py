"""The suite's result language, as miForge speaks it (017 FR-017.3 - .5, .31; ADR-028).

``mistudio.result-language/v1`` is the suite's contract (another app owns it), vendored at
``docs/schemas/mistudio-result-language-v1.json``; this module builds the one object of it the
backend emits, ``approval``, from a stored ``mf_approvals`` row:

    {id, status, decide_in: "miforge", then: "server_runs", summary, action, expires_at, reason}

- ``status`` is the stored status (``STATUS_MAP`` is the identity over the six statuses the
  queue writes), except that a ``pending`` row past its expiry that no sweep has marked yet
  reads ``expired`` (017-D10). ``approved`` is never emitted: miForge runs an approved request
  itself (``then: "server_runs"``, CHAIN_PLAN A18), so there is no approved-then-re-call step,
  and no ``call_again_with`` is ever set (FR-017.31).
- ``summary`` is the contract's plain-words string (1 - 500 characters): what will happen. The
  stored summary is the operator's structured block (the Approvals view renders it), so the
  words are rendered from it here; the block itself stays under the legacy ``summary`` key.
- ``decide_in`` is ``"miforge"``, the MCP server id (017-D9).
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

#: Stored status -> contract status. The identity, stated so a status the queue gains later
#: fails loudly here (KeyError) rather than passing through unmapped.
STATUS_MAP: dict[str, str] = {
    status: status
    for status in ("pending", "executing", "executed", "failed", "rejected", "expired")
}

DECIDE_IN = "miforge"
THEN = "server_runs"
SUMMARY_LIMIT = 500

#: Plain words for each approval action (the operator's Approvals view says the same).
ACTION_WORDS: dict[str, str] = {
    "gpu_start": "Start GPU work",
    "hotupdate_attach": "Train the weights miServer is serving, in place",
    "delete": "Delete",
    "config_write": "Change a runtime setting",
}

#: Summary keys worth naming in the words, in this order.
_SUMMARY_KEYS = (
    "route",
    "run_id",
    "chain",
    "model",
    "run_type",
    "pack",
    "draft_id",
    "key",
    "value",
    "step_budget",
    "gpu_uuid",
)


def _aware(moment: datetime) -> datetime:
    return moment if moment.tzinfo is not None else moment.replace(tzinfo=UTC)


def summary_words(action: str, summary: Any) -> str:
    """The contract's ``summary``: one plain-words line, 1 - 500 characters."""
    head = ACTION_WORDS.get(action, action.replace("_", " ").capitalize())
    parts: list[str] = []
    if isinstance(summary, dict):
        for key in _SUMMARY_KEYS:
            value = summary.get(key)
            if value is None or isinstance(value, dict | list):
                continue
            parts.append(f"{key.replace('_', ' ')} {value}")
    text = head + (": " + "; ".join(parts) if parts else "") + "."
    return text if len(text) <= SUMMARY_LIMIT else text[: SUMMARY_LIMIT - 3] + "..."


def contract_status(stored: str, expires_at: datetime, now: datetime) -> str:
    """The contract status of a stored row at ``now`` (FR-017.5, 017-D10)."""
    if stored == "pending" and _aware(expires_at) <= _aware(now):
        return "expired"
    return STATUS_MAP[stored]


def approval_view(approval: Any, now: datetime) -> dict[str, Any]:
    """The ``approval`` object for one stored row (FR-017.3)."""
    status = contract_status(approval.status, approval.expires_at, now)
    view: dict[str, Any] = {
        "id": approval.id,
        "status": status,
        "decide_in": DECIDE_IN,
        "then": THEN,
        "summary": summary_words(approval.action, approval.summary),
        "action": approval.action,
        "expires_at": _aware(approval.expires_at).isoformat(),
        # Always present (null until there is one), so the typed view and the dict agree.
        "reason": (
            _reason(approval, status) if status in ("rejected", "failed", "expired") else None
        ),
    }
    return view


def _reason(approval: Any, status: str) -> str | None:
    if status == "rejected":
        return getattr(approval, "reason", None)
    if status == "failed":
        error = getattr(approval, "error", None)
        if isinstance(error, dict):
            code = error.get("code")
            message = error.get("message")
            return f"{code}: {message}" if code and message else str(code or message or "")
        return None if error is None else str(error)
    return "The request expired before the operator decided it."


__all__ = ["DECIDE_IN", "STATUS_MAP", "THEN", "approval_view", "contract_status", "summary_words"]
