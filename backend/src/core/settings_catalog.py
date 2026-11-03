"""What each setting does and when a change to it takes effect  (013 G4; ENH-003 section 3.4).

``GET /api/v1/settings`` served raw field names and nothing else, so the Settings screen could
not say what a setting does, when a change reaches the code that reads it, or which pod has to
restart for the rest. This module is that knowledge, **in one place, for every field** of
:class:`core.config.Settings` - ``tests/unit/test_settings_catalog.py`` fails when a field has
no entry or an entry names no field, so a new setting cannot ship undescribed.

**When a change takes effect** (``effect``) is one of four, and each is pinned by a test that
exercises it:

* ``now`` - the API applies the change at the write (the settings route refreshes its cache and,
  for the log level, re-configures logging); each worker applies it before its next task
  (the Celery ``task_prerun`` refresh). ``log_level`` and the two sibling addresses.
* ``next_sweep`` - read by the checkpoint pruning sweep when it starts
  (``workers/checkpoint_lifecycle.py``). The four ``checkpoint_lifecycle_*`` keys.
* ``next_attached_run`` - read once when a hot-update run attaches
  (``training/hotupdate/settings.load``). The five hot-update keys on the allow-list.
* ``restart`` - every other field. It is read from the environment when a process starts, and
  every process that reads it runs in the ``miforge-backend`` pod (the API, the four Celery
  workers and beat; ``gitops/pilot/backend.yaml``). **A restart stops running jobs**, and the
  text says so.

**Which group a setting belongs to** (``group``; 2026-10-28) is how the Settings screen
sorts a list of forty-odd fields into something an operator can scan: ``GROUPS`` fixes the
groups and their order, ``GROUP_OF`` places every field, and the import-time guard below
refuses a field with no group, exactly as it refuses one with no entry. The screen sorts by
label within a group, so the order of ``GROUP_OF`` carries no meaning.

The text is the operator's: plain words, no internal identifiers (the same rules the manual's
publish check applies, ``manual/publish-rules.json``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.config import RUNTIME_KEYS, Settings

Effect = Literal["now", "next_sweep", "next_attached_run", "restart"]

#: The pod whose restart applies a ``restart`` setting.
BACKEND_POD = "miforge-backend"

#: Each effect as the screen states it.
EFFECT_TEXT: dict[str, str] = {
    "now": "Immediately in the API; each worker applies it before its next task.",
    "next_sweep": "At the next checkpoint pruning sweep.",
    "next_attached_run": "When the next hot-update run attaches; a running one keeps its value.",
    "restart": (
        f"Read at startup. Change it in the environment and restart the {BACKEND_POD} pod; "
        "a restart stops running jobs."
    ),
}

EFFECTS: tuple[str, ...] = tuple(EFFECT_TEXT)


@dataclass(frozen=True)
class SettingInfo:
    """One setting as the operator reads it."""

    label: str
    description: str
    effect: Effect


def _restart(label: str, description: str) -> SettingInfo:
    return SettingInfo(label=label, description=description, effect="restart")


#: Every field of ``Settings``, in the order the class declares them.
CATALOG: dict[str, SettingInfo] = {
    "database_url": _restart(
        "Database address",
        "Where miForge's PostgreSQL database is. The password is never shown.",
    ),
    "redis_url": _restart(
        "Queue address",
        "Where the Redis server that carries jobs and progress messages is. The password is "
        "never shown.",
    ),
    "checkpoint_root": _restart(
        "Checkpoint folder",
        "Where saved checkpoints are written. It must be on one filesystem so a save can be "
        "published in a single rename.",
    ),
    "trainer_state_root": _restart(
        "Trainer state folder",
        "Where optimizer state is kept beside each checkpoint, so a run can continue exactly.",
    ),
    "parts_root": _restart(
        "Parts folder",
        "Where imported data parts (prompts, answers, corpora) are stored.",
    ),
    "checkpoint_read_roots": _restart(
        "Extra checkpoint folders (read only)",
        "Other folders miForge searches for checkpoints and never writes, separated by commas. "
        "Used to continue a hot-update run whose checkpoints miLLM saved.",
    ),
    "millm_base_url": SettingInfo(
        label="miLLM address",
        description=(
            "Where miLLM, the suite's inference server, answers. Leave it empty to run without "
            "miLLM; the features that need it then say so by name."
        ),
        effect="now",
    ),
    "midataworks_base_url": SettingInfo(
        label="miDataworks address",
        description=(
            "Where miDataworks answers, for importing dataset versions as parts. Leave it empty "
            "to run without miDataworks."
        ),
        effect="now",
    ),
    "midataworks_agent": _restart(
        "Name miForge gives miDataworks",
        "How miForge identifies itself to miDataworks, in the form agent:<name>.",
    ),
    "millm_trainer_token": _restart(
        "miLLM trainer token",
        "The secret that lets a hot-update run attach to the model miLLM serves. Shown only as "
        "set or not set.",
    ),
    "millm_audit_label": _restart(
        "Name in miLLM's log",
        "The label miLLM records against a hot-update run's requests.",
    ),
    "hotupdate_verification_tolerance": SettingInfo(
        label="Reconstruction check bar (served precision)",
        description=(
            "How far miForge's rebuilt copy of the served model may differ from miLLM's answers "
            "before a hot-update run is refused, when the check runs in the served precision. "
            "Not yet measured, so it is marked provisional."
        ),
        effect="next_attached_run",
    ),
    "hotupdate_verification_tolerance_measured": _restart(
        "Served-precision bar measured",
        "Whether the bar above was measured rather than chosen. Off until someone measures it.",
    ),
    "hotupdate_heartbeat_interval_seconds": _restart(
        "Hot-update heartbeat interval (seconds)",
        "How often an attached run tells miLLM it is alive. Empty means use the interval miLLM "
        "suggests, which is the normal case.",
    ),
    "hotupdate_request_timeout_seconds": SettingInfo(
        label="miLLM request time limit (seconds)",
        description="How long a hot-update run waits for an ordinary answer from miLLM.",
        effect="next_attached_run",
    ),
    "hotupdate_checkpoint_timeout_seconds": SettingInfo(
        label="miLLM save and load time limit (seconds)",
        description=(
            "How long a hot-update run waits for miLLM to save or load a whole checkpoint, "
            "which takes far longer than an ordinary request."
        ),
        effect="next_attached_run",
    ),
    "hotupdate_commit_end_timeout_seconds": SettingInfo(
        label="End-of-update time limit (seconds)",
        description=(
            "How long a hot-update run waits for miLLM to accept the end of a weight update. "
            "Deliberately the longest: a late answer is better than leaving the served weights "
            "half written."
        ),
        effect="next_attached_run",
    ),
    "hotupdate_commit_watchdog_ms": _restart(
        "miLLM's update watchdog (milliseconds)",
        "A copy of miLLM's own limit on one weight update, used to judge how close each update "
        "comes to it. It must match miLLM's setting, so it is not changeable here.",
    ),
    "hotupdate_commit_creep_fraction": _restart(
        "Watchdog warning fraction",
        "Warn when an update takes more than this fraction of miLLM's watchdog.",
    ),
    "hotupdate_verify_prompt": _restart(
        "Reconstruction check prompt",
        "The fixed prompt both sides score before a hot-update run's first step.",
    ),
    "hotupdate_verify_token_set_size": _restart(
        "Reconstruction check tokens",
        "How many of the most likely next tokens the check compares (1 to 20).",
    ),
    "hotupdate_verify_tokenizer": _restart(
        "Reconstruction check tokenizer",
        "Where the check's tokenizer comes from. Empty means read it from the served model or "
        "its checkpoint, which is the normal case.",
    ),
    "hotupdate_verification_precision": _restart(
        "Reconstruction check precision",
        "float32 compares both sides at full precision and can see small errors; native "
        "compares in the served precision.",
    ),
    "hotupdate_verification_tolerance_float32": SettingInfo(
        label="Reconstruction check bar (full precision)",
        description=(
            "How far the two sides may differ when the check runs at full precision. Measured "
            "on one model; a larger value accepts a less exact rebuild."
        ),
        effect="next_attached_run",
    ),
    "hotupdate_verification_tolerance_float32_measured": _restart(
        "Full-precision bar measured",
        "Whether the full-precision bar was measured. On: it was, on one model.",
    ),
    "hotupdate_verify_float32_device": _restart(
        "Where the full-precision check runs",
        "The device for miForge's side of the full-precision check. cpu by default, because "
        "the card is shared with the model being served.",
    ),
    "hotupdate_precommit_retries": _restart(
        "Retries before an update starts",
        "How many times a step is retried when miLLM refuses it before anything was written "
        "(0 to 5). A step is never retried once writing has begun.",
    ),
    "heartbeat_min_interval_seconds": _restart(
        "Heartbeat interval (seconds)",
        "The least time between two liveness writes for a running job. A minimum, not a "
        "deadline: a long save or test pass can go longer between writes.",
    ),
    "janitor_lapse_seconds": _restart(
        "Lost-job threshold (seconds)",
        "How long a running job may go without a liveness write before the cleanup sweep "
        "treats its worker as lost.",
    ),
    "gpu_lease_seconds": _restart(
        "Graphics card lease (seconds)",
        "How long a job's hold on a graphics card lasts before it must be renewed.",
    ),
    "checkpoint_lifecycle_enabled": SettingInfo(
        label="Prune old checkpoints",
        description=(
            "Whether the pruning sweep removes old checkpoints. Off by default. With dry run "
            "off, turning this on deletes checkpoints."
        ),
        effect="next_sweep",
    ),
    "checkpoint_lifecycle_dry_run": SettingInfo(
        label="Pruning dry run",
        description=(
            "On: the sweep only reports what it would remove. Off: it deletes. A checkpoint "
            "another run branched from is always kept."
        ),
        effect="next_sweep",
    ),
    "checkpoint_lifecycle_keep_last_n": SettingInfo(
        label="Keep the last N saves",
        description="How many of each run's most recent saves pruning always keeps.",
        effect="next_sweep",
    ),
    "checkpoint_lifecycle_keep_every_m": SettingInfo(
        label="Keep every Mth save",
        description=(
            "Pruning also keeps every Mth save, counted from a run's first, so the first save "
            "is always kept."
        ),
        effect="next_sweep",
    ),
    "approval_ttl_hours": _restart(
        "Approval waiting time (hours)",
        "How long an agent's request waits for your decision before it expires and can never "
        "run.",
    ),
    "log_level": SettingInfo(
        label="Log level",
        description="How much miForge writes to its logs: DEBUG, INFO, WARNING or ERROR.",
        effect="now",
    ),
    "environment": _restart(
        "Environment",
        "Which kind of installation this is: dev, ci or prod.",
    ),
}


#: The operator-facing groups, in the order the screen shows them: ``(id, label)``.
GROUPS: tuple[tuple[str, str], ...] = (
    ("general", "General"),
    ("storage", "Storage"),
    ("jobs", "Jobs, GPUs and approvals"),
    ("checkpoint_pruning", "Checkpoint pruning"),
    ("connections", "Suite connections"),
    ("hot_update", "Hot-update runs"),
)

#: Every field's group. The guard below holds this to exactly the fields of ``Settings``.
GROUP_OF: dict[str, str] = {
    "log_level": "general",
    "environment": "general",
    "database_url": "storage",
    "redis_url": "storage",
    "checkpoint_root": "storage",
    "trainer_state_root": "storage",
    "parts_root": "storage",
    "checkpoint_read_roots": "storage",
    "heartbeat_min_interval_seconds": "jobs",
    "janitor_lapse_seconds": "jobs",
    "gpu_lease_seconds": "jobs",
    "approval_ttl_hours": "jobs",
    "checkpoint_lifecycle_enabled": "checkpoint_pruning",
    "checkpoint_lifecycle_dry_run": "checkpoint_pruning",
    "checkpoint_lifecycle_keep_last_n": "checkpoint_pruning",
    "checkpoint_lifecycle_keep_every_m": "checkpoint_pruning",
    "millm_base_url": "connections",
    "midataworks_base_url": "connections",
    "midataworks_agent": "connections",
    "millm_trainer_token": "connections",
    "millm_audit_label": "connections",
    "hotupdate_verification_tolerance": "hot_update",
    "hotupdate_verification_tolerance_measured": "hot_update",
    "hotupdate_heartbeat_interval_seconds": "hot_update",
    "hotupdate_request_timeout_seconds": "hot_update",
    "hotupdate_checkpoint_timeout_seconds": "hot_update",
    "hotupdate_commit_end_timeout_seconds": "hot_update",
    "hotupdate_commit_watchdog_ms": "hot_update",
    "hotupdate_commit_creep_fraction": "hot_update",
    "hotupdate_verify_prompt": "hot_update",
    "hotupdate_verify_token_set_size": "hot_update",
    "hotupdate_verify_tokenizer": "hot_update",
    "hotupdate_verification_precision": "hot_update",
    "hotupdate_verification_tolerance_float32": "hot_update",
    "hotupdate_verification_tolerance_float32_measured": "hot_update",
    "hotupdate_verify_float32_device": "hot_update",
    "hotupdate_precommit_retries": "hot_update",
}


def groups_payload() -> list[dict[str, str]]:
    """The groups in screen order, as ``GET /api/v1/settings`` serves them."""
    return [{"id": group, "label": label} for group, label in GROUPS]


def info_for(key: str) -> SettingInfo:
    """The entry for ``key``; ``KeyError`` for a name that is not a setting."""
    return CATALOG[key]


def effect_payload(key: str) -> dict[str, str]:
    """``{"when", "text"}``, plus ``"pod"`` for a setting that needs a restart."""
    when = CATALOG[key].effect
    payload = {"when": when, "text": EFFECT_TEXT[when]}
    if when == "restart":
        payload["pod"] = BACKEND_POD
    return payload


def _check() -> None:
    """Import-time guard: the catalog and the model agree, and effects follow the allow-list."""
    fields = set(Settings.model_fields)
    missing = fields - set(CATALOG)
    extra = set(CATALOG) - fields
    if missing or extra:
        raise RuntimeError(
            f"settings catalog out of step with Settings: missing {sorted(missing)}, "
            f"extra {sorted(extra)}"
        )
    for key, info in CATALOG.items():
        if (key in RUNTIME_KEYS) == (info.effect == "restart"):
            raise RuntimeError(f"settings catalog: {key!r} states {info.effect!r}")
    ungrouped = fields - set(GROUP_OF)
    stray = set(GROUP_OF) - fields
    known = {group for group, _ in GROUPS}
    unknown = {key: group for key, group in GROUP_OF.items() if group not in known}
    empty = known - set(GROUP_OF.values())
    if ungrouped or stray or unknown or empty:
        raise RuntimeError(
            f"settings groups out of step with Settings: ungrouped {sorted(ungrouped)}, "
            f"stray {sorted(stray)}, unknown group {unknown}, empty groups {sorted(empty)}"
        )


_check()
