"""Does a run's push fit the WSP worker's ``/work`` volume? (production defect, 2026-11-20).

The driver streams **every** file of a run's push (``dispatch.drive``'s ``files_for``: the bundle's
stored files, plus a replay's transcript or a rerun's program) into the worker's ``/work``, an
emptyDir whose ``sizeLimit`` is ``WSP_WORKER_WORK_BYTES``. On 2026-11-20 a 35 GB phase-2 bundle
was pushed into an 8 GiB ``/work``: the kubelet evicted the pod mid-push and the run failed
``worker_push_failed: POST /session: RemoteProtocolError``, after the operator had approved it.

So the size is compared here, once, and the same comparison is wired at both ends:

* **at freeze** (``wsp.freeze.freeze``: ``POST /wsp/runs``, its validate route and the approval
  gate's validate hook), from the imported bundle's manifest, so a run that cannot fit is refused
  ``wsp_bundle_too_large_for_worker`` **before an approval is filed**;
* **at dispatch** (``wsp.dispatch._refusal``), from the files the push would actually send, before
  a boot is consumed;
* and the pushed ``spec.work_max_bytes`` (the supervisor's disk watchdog) is
  :func:`watchdog_bytes`, so the watchdog kills a session ``worker_disk_limit`` before the kubelet
  evicts the pod.

The suite rule (``storage-does-not-constrain-design``): this is a requirement to provision, never
a design limit. The refusal names the setting and the manifest line to raise.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from core.errors import refuse

if TYPE_CHECKING:
    from wsp.settings import WSPSettings

GIB = 1024**3
#: What the session itself may write under ``/work`` beyond the pushed files: one file at the
#: child's ``RLIMIT_FSIZE`` (``wsp_worker.__main__.CHILD_FSIZE``, 4 GiB), the largest single file
#: an editor's program can create.
EDITOR_HEADROOM_BYTES = 4 * GIB
#: The disk watchdog fires this far below the volume. It polls ``/work`` once a second
#: (``wsp_worker.supervisor.DISK_POLL_S``) while the kubelet measures emptyDirs far less often, so
#: the margin is several seconds of local-disk writes: the session is killed ``worker_disk_limit``
#: (a named failure) before the pod is evicted (a lost push and an unexplained ``worker_lost``).
WATCHDOG_MARGIN_BYTES = 4 * GIB
#: The smallest ``WSP_WORKER_WORK_BYTES`` that leaves any room for a push.
MIN_WORK_BYTES = EDITOR_HEADROOM_BYTES + WATCHDOG_MARGIN_BYTES

CODE = "wsp_bundle_too_large_for_worker"
SETTING = "WSP_WORKER_WORK_BYTES"


def watchdog_bytes(settings: WSPSettings) -> int:
    """``spec.work_max_bytes`` for every push: the volume less :data:`WATCHDOG_MARGIN_BYTES`."""
    return int(settings.wsp_worker_work_bytes) - WATCHDOG_MARGIN_BYTES


def manifest_bytes(manifest: Mapping[str, Any]) -> int:
    """A bundle's push at freeze: the sum of its manifest's file sizes (what the import stored as
    ``mf_wsp_bundles.bytes`` and what ``wsp.tasks.bundle_files`` will push)."""
    return sum(int(entry["bytes"]) for entry in manifest.get("files") or [])


def files_bytes(files: Iterable[tuple[str, Path]]) -> int:
    """A push at dispatch: the on-disk size of every file it would send (a missing file counts 0;
    the source checks after this one refuse it by name)."""
    total = 0
    for _, path in files:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _gib(value: int) -> str:
    return f"{value / GIB:.1f} GiB"


def refusal(push_bytes: int, settings: WSPSettings) -> tuple[str, str] | None:
    """``(code, message)`` when ``push_bytes`` plus the editor's headroom exceeds what the session
    may use (:func:`watchdog_bytes`), else ``None``. The one comparison both ends run."""
    limit = watchdog_bytes(settings)
    if push_bytes + EDITOR_HEADROOM_BYTES <= limit:
        return None
    capacity = int(settings.wsp_worker_work_bytes)
    return (
        CODE,
        f"The run pushes {push_bytes} bytes ({_gib(push_bytes)}) to the WSP worker's /work volume "
        f"and keeps {_gib(EDITOR_HEADROOM_BYTES)} for the session's own files, but a session may "
        f"use {limit} bytes ({_gib(limit)}: {SETTING}={capacity} ({_gib(capacity)}) less the "
        f"{_gib(WATCHDOG_MARGIN_BYTES)} disk-watchdog margin). Provision more: raise {SETTING} in "
        "the miforge-config ConfigMap together with the `work` emptyDir's sizeLimit in "
        "gitops/pilot/wsp-worker.yaml (a test keeps the two equal), then restart the backend.",
    )


def require_fits(push_bytes: int, settings: WSPSettings) -> None:
    """:func:`refusal`, raised as a refusal (413, like ``wsp_bundle_too_large``)."""
    found = refusal(push_bytes, settings)
    if found is not None:
        code, message = found
        raise refuse(
            code,
            message,
            push_bytes=push_bytes,
            session_limit_bytes=watchdog_bytes(settings),
            setting=SETTING,
        )


# ----------------------------------------------------------------------------- time (2026-11-21)

TIME_CODE = "wsp_time_budget_too_small"
TIME_SETTING = "WSP_FORWARD_PASS_BYTES_PER_S"
#: Refuse only when the time budget is under this share of the estimate: a false refusal (a run
#: that could have done its work) is worse than none, and the estimate ignores the scorer's cache.
TIME_REFUSAL_SHARE = 0.5
_ITEMSIZE = {"float16": 2, "bfloat16": 2, "float32": 4}


def practice_bytes(manifest: Mapping[str, Any]) -> int:
    """The practice activations a forward pass reads: ``tokens x d_model`` at the bundle's dtype."""
    p = manifest.get("practice") or {}
    return (
        int(p.get("tokens", 0)) * int(p.get("d_model", 0)) * _ITEMSIZE.get(str(p.get("dtype")), 4)
    )


def forward_pass_s(manifest: Mapping[str, Any], settings: WSPSettings) -> float:
    """One forward pass's seconds on the worker: one uncached read of the practice set at
    ``WSP_FORWARD_PASS_BYTES_PER_S`` (``backend/scripts/bench_wsp_scorer.py``)."""
    return practice_bytes(manifest) / float(settings.wsp_forward_pass_bytes_per_s)


def time_refusal(
    budgets: Mapping[str, Any], manifest: Mapping[str, Any], settings: WSPSettings
) -> tuple[str, str, str] | None:
    """``(code, message, fix)`` when ``budgets.time_s`` is clearly too short for the requested
    ``forward_passes`` (under :data:`TIME_REFUSAL_SHARE` of the estimate), else ``None``.

    Run ``wsp_cb092cedad97`` (2026-11-20) asked for 40 passes over a 34.9 GB practice set in
    2400 s while each pass took ~3 minutes; a check before the approval says so instead of
    spending the operator's approval and the whole budget to find out."""
    passes = int(budgets.get("forward_passes") or 0)
    time_s = float(budgets.get("time_s") or 0)
    per_pass = forward_pass_s(manifest, settings)
    if passes <= 0 or per_pass <= 0:
        return None
    estimate = passes * per_pass
    if time_s >= TIME_REFUSAL_SHARE * estimate:
        return None
    rate = float(settings.wsp_forward_pass_bytes_per_s)
    fit = int(time_s // per_pass)
    message = (
        f"budgets.time_s is {time_s:g} s, under half of what {passes} forward passes need: each "
        f"reads the {_gib(practice_bytes(manifest))} practice set, about {per_pass:.0f} s at "
        f"{TIME_SETTING}={rate:g} bytes/s, so about {estimate:.0f} s in all, before any editor "
        "turn."
    )
    fix = (
        f"raise budgets.time_s to at least {int(estimate) + 1}, or lower budgets.forward_passes "
        f"to at most {fit}; if the worker is faster than {TIME_SETTING} says, the operator "
        "raises that setting"
    )
    return TIME_CODE, message, fix


def require_time(
    budgets: Mapping[str, Any], manifest: Mapping[str, Any], settings: WSPSettings
) -> None:
    """:func:`time_refusal`, raised as a refusal (422) with its ``fix``."""
    found = time_refusal(budgets, manifest, settings)
    if found is not None:
        code, message, fix = found
        raise refuse(
            code,
            message,
            fix=fix,
            forward_passes=int(budgets.get("forward_passes") or 0),
            time_s=float(budgets.get("time_s") or 0),
            forward_pass_s=round(forward_pass_s(manifest, settings), 3),
            setting=TIME_SETTING,
        )
