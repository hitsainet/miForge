"""The engine's session loop (feature 018; FR-018.5 to .7, .11 to .14; FTDD §6.2; FTID §3.5).

Inputs: the child line the supervisor writes (``mode, run_id, spec{request, manifest, editor},
files, gateway, paths``). Outputs, in ``/out`` only, under the names the intake knows:
``result.json``, ``transcript.jsonl``, ``patch/`` and, in program style, ``program.py``.

1. **Load** the bundle (``bundle.load``); the store holds the editable tensors (zeros in a
   ``create``); the observation tools are built (an unknown one refuses before anything runs).
2. **Isolate** (FR-018.14): ``prctl(PR_SET_DUMPABLE, 0)``, then unlink the practice and activation
   files from ``/work`` (their maps stay open here), THEN spawn the kernel grandchild.
3b. **The repeat guard** (fix 2026-11-21, runs ``wsp_7767fb91254b``, ``wsp_554d3932d108``): an
   action identical to one already run (same tool, same arguments, :data:`PURE_TOOLS` or a
   forward pass) on the same weights (the store's sha256) is not run and not charged; the editor
   is told ``prompts.REPEATED`` with the turn that ran it, and ``repeated_turns`` counts it (the
   transcript's ``{event: repeated}`` line too). After :data:`MAX_REPEATS` with no new
   measurement between them (a forward pass that ran and was charged; an edit alone is not
   progress) the session moves to the submit-only turn. A replay recomputes the same decisions
   from the same transcript, under the header's ``feedback`` (absent: the count was over the
   whole session, and the texts were the older ones).
3. **Each turn:** budgets first (90% of tokens, cost and time, as the ``weightspace`` adapter
   did); the context guard (a handoff, never an overflow); one request; exactly one action
   (``protocol.one_action``); a refused reply runs nothing and costs the turn. A reply with no
   action gets ``weightspace``'s nudge; after two, the next one ends the run ``answered``.
3c. **A lost reply** (fix 2026-11-22, D128 round 7, run ``wsp_e571f6971fd2``): a call whose
   reply never came (``client.LOST_REPLY_CODES``: the gateway unreachable or given up on, or its
   upstream call failed, as a reply past the call time limit does) no longer ends the session. It
   is recorded (``{event: lost_reply, turn, code, n}``), counted as a refused turn (it used its
   step), and the editor is told ``prompts.LOST_REPLY`` once, in the next user message; the loop
   goes on while the budgets allow. A SECOND lost reply in a row ends the session gracefully, as
   the time budget does (step 4a): stop ``time_budget`` with the cause ``editor_reply_lost``
   (:data:`LOST_STOP`, :data:`LOST_CAUSE`), the final submit-only turn offered, and the
   transcript, ledger, patch and usage kept. A reply lost in that final turn ends it with no
   answer, never with an error. A replay replays each lost reply at the turn it was lost
   (``transcript.read_script``).
4. **When the steps or a budget run out,** one last turn offers ``submit`` alone.
4a. **A graceful end before the hard kill** (fix 2026-11-21, run ``wsp_cb092cedad97``). Time is
   measured to the SUPERVISOR's deadline (``time_left_s`` in the child line, read when the
   session started; it kills the child at the budget with nothing written), and the stop comes
   early enough for one more turn: ``time_budget`` when the time left is at most the larger of
   the 10% share and the longest turn so far plus the final turn (the longest request plus
   :data:`FINAL_MARGIN_S`, at most the share). Once per budget (time, tokens, cost, forward
   passes), when 80% of the engine's share is used or two more turns of the largest so far would
   reach the stop, the editor is told ``prompts.NEARLY_UP_*``: checkpoint, then submit the best
   weights now. A forward pass that would not finish before the final turn's reserve is not run
   (``prompts.NO_TIME_FOR_FORWARD``, not charged): a tool call is the one thing that cannot be
   interrupted, and one that straddled the kill lost the whole session.
5. **``submit``** is refused while the live edit breaks a change limit (the loop goes on).
5a. **How the result is judged** (fix 2026-11-22, D128 round 7: three candidates changed only the
   bias, which the verdict cannot see). With judging on, the context carries
   ``judging.section`` (both providers get the same text), and the FIRST ``submit`` (or
   ``submit_program``) of an edit that is a positive affine map of the parent
   (``wsp_worker.equivalence``) is not accepted: the editor is told once
   (``judging.AFFINE_SUBMIT``), may submit again, and the transcript gets
   ``{event: affine_equivalent_submit, turn, tool, equivalence, scale}``. Not in the final
   submit-only turn, where the editor could not answer it. Judging is on in every session; a
   replay follows the original's transcript header (``judging``; absent: off, as it ran).
5b. **Write noise** (2026-11-26, FU-3's finding, run ``wsp_4f035a6d8a61``: the editor meant to
   scale ``weight[4]``, wrote all 8 entries back from 4-digit readings, was told "8/8 changed",
   and its answer named only ``weight[4]``). An accepted ``write_tensor``, or a ``python`` run
   that wrote ``W``, ends its answer with ``prompts.WRITE_NOISE`` when some entries it changed
   moved by no more than a reading at ``tools.DISPLAY_PRECISION`` digits rounds away
   (``store.below_display``), and the transcript gets ``{event: write_noise, turn, tool,
   precision, changed, noise, tensors: [{name, entries, indices}]}``. A flag only: the write is
   kept as it was made, nothing is refused, and the patch, ``result.json`` and the run document
   are unchanged. On in every editing session; a replay follows the header's ``write_noise``
   (absent: off, as it ran).
5c. **An edit already measured and reverted** (2026-11-28, run ``wsp_0d4a2f916c9c``, phase-2
   E1r: from turn 80 to turn 100 the editor made the same weights nine times, each followed by a
   revert, about 20 of its 100 steps, and ended at ``max_steps`` with ``no_change``). The session
   records the first measurement (a forward pass that ran and was charged) on each weights
   sha256, with its figures (``paging.numbers_of``), and the first ``revert`` or ``restore`` that
   left those weights after it. A ``python`` or ``write_tensor`` that CHANGES the weights to ones
   so measured and reverted ends its answer with ``prompts.EDIT_REPEATED`` (the measurement's
   turn, tool and figures, and the revert's turn), counts in ``repeated_turns`` (the run's one
   repeat counter, which until then counted only the actions step 3b did not run), and the
   transcript gets ``{event: edit_repeated, turn, tool, weights, measured_turn, reverted_turn,
   n}``. The edit is kept and nothing is refused; it does not count toward step 3b's stop. On in
   every editing session; a replay follows the header's ``edit_repeats`` (absent: off, so E1r and
   every run recorded before replay bit for bit).
5d. **A cut-off reply lowers the output ceiling** (2026-11-28, run ``wsp_3417a18bc85d``, phase-2
   E7: with the output cap off, four replies ran to the 32,000-token ceiling, about 870 s each,
   58 of its 95 minutes; E3 the same). In a session whose cap is off, the first reply cut off at
   the length or time limit (``proto.Refusal.cut_off``) lowers the session's own output ceiling to
   :data:`CUT_OFF_CAP_TOKENS` through the per-turn cap's rule (``context.output_cap``: at most an
   eighth of the window, never below the gateway's floor) and never above the pushed ceiling:
   every later request names that cap (``max_tokens``), as a capped session's does, and its
   thinking share is the capped one. The refusal's answer ends with ``prompts.CUT_OFF_CAPPED``,
   the transcript gets ``{event: ceiling_lowered, turn, from, to, why: cut_off}`` and
   ``result.json``'s engine block ``output_ceiling_lowered_to``, which the intake stores as the
   run's ``output_ceiling_tokens`` with ``output_ceiling_lowered`` true. A capped session already
   names its cap and is unchanged. On in every session; a replay follows the header's
   ``cut_off_cap`` (absent: off, so E7 and every run recorded before replay as they ran).
6. **At the end, whatever happened** (``try/finally``): the patch from the store against the
   bundle's bits, ``result.json`` (``stop_reason``, ``failure``, ``forward_used``, and
   ``engine{action_protocol, handoffs, refused_turns, kernel_restarts, reasoning_turns}``) and the
   transcript.
7. **Thinking** (2026-11-21): the request's frozen ``thinking`` and ``thinking_tokens``. With
   ``on`` every request asks for the action's cap plus the reasoning allowance
   (``context.thinking_allowance``, bounded by the window) and the guard counts both; the reply's
   reasoning is recorded in the transcript (``reasoning {text, source}`` on the assistant line)
   and counted (``reasoning_turns``), never parsed and never sent back. A thinking turn's time and
   tokens are what the turn measures, so the graceful end's longest-turn and largest-turn figures
   (step 4a) include them.
7a. **The room of an uncapped turn** (fix 2026-11-21, run ``wsp_855a74f820ff``). With the output
   cap off, the gateway sizes the cap. The engine predicts that cap from the gateway's own prompt
   estimate (``context.gateway_cap``) and sends the reasoning share the gateway would leave. A
   share under 256 sends the turn with thinking off (``{event: thinking_skipped, why: room}``).
   A room under one action's 256 tokens is never sent: the conversation is handed off, and a task
   that still leaves none ends ``editor_context_window_too_small``. A submit the submit-only turn
   cannot read is asked for ONCE more, with thinking off (``{event: submit_retry}``).

8. **Paging and the ledger** (operator, 2026-11-21; ``paging.py``; the frozen request's
   ``paging``): after each action that ran, its ledger row (a measurement or an edit) and its
   result offered to the pager; before a request, at most every few turns, a batched page event
   (stubs for results past the keep window, and the ledger moved to the history's end); ``recall``
   among the tools; the handoff, now the last resort, carries the ledger and the pages. Counted
   in ``engine{pages_out, recalls, ledger_rows, peak_prompt_tokens}``; the transcript gets
   ``{event: page_out | recall | ledger}`` lines and the configuration in its header.

The child log (stdout, drained by the supervisor) gets one line per turn, built from named fields
only, so the gateway token can never reach it.
"""

from __future__ import annotations

import ctypes
import datetime as dt
import json
import time
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from wsp_worker import bundle as bundle_mod
from wsp_worker import (
    equivalence,
    judging,
    observation_tools,
    patch_writer,
    program,
    submitted_measure,
    tensor_read,
)
from wsp_worker.bundle import Bundle
from wsp_worker.miwsp_engine import context, paging, prefix, prompts
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine import tools as tools_mod
from wsp_worker.miwsp_engine.client import (
    LOST_REPLY_CODES,
    ClientError,
    GatewayClient,
    ModelClient,
)
from wsp_worker.miwsp_engine.kernel import EXEC_TIMEOUT_S, Kernel, KernelError
from wsp_worker.miwsp_engine.store import Refused, TensorStore
from wsp_worker.miwsp_engine.transcript import (
    TranscriptWriter,
    read_header,
    read_script,
    read_window_event,
)
from wsp_worker.program_tools import ProgramTools
from wsp_worker.scoring import Scorer

BUDGET_SHARE = 0.9
#: The share of each of the engine's budgets at which the editor is told it is nearly used up.
NOTICE_SHARE = 0.8
#: Seconds kept beyond the final request for writing the patch, ``result.json`` and the
#: transcript (they take well under a second for a probe head).
FINAL_MARGIN_S = 10.0
#: The prior for a forward pass's time before one has been seen: the practice bytes read at
#: this rate. Benchmark 2026-11-21 (``backend/scripts/bench_wsp_scorer.py``, page cache dropped,
#: two CPUs): 1.85 GB/s of fp16 activations; this is about half, to be safe.
FORWARD_BYTES_PER_S = 1.0e9
DEFAULT_MAX_STEPS = 50
#: The repeat guard (step 3b): after this many repeats, the session moves to the submit-only turn.
MAX_REPEATS = 3
#: What the engine tells the editor about repeats and edits, recorded in the transcript's header
#: (fix 2026-11-22, D128 round 9). 2: see ``prompts.REPEAT_LIMIT_IN_A_ROW``; a read too large to
#: show is a summary (``tensor_read``); and only no-action replies in a row, with no action run
#: between, end the session ``answered``. Absent: as before.
FEEDBACK_VERSION = 2
#: 2026-11-27 (run ``wsp_ba149c57f611``): the kernel binds the reserved names (``R``, ``W``,
#: ``notes``, ``np``) back after every exec, and says so (``kernel_main.RESERVED_NOTE``). Recorded
#: in the header of every session with Python; a replay follows it, so a run recorded before
#: replays with the names left as its code bound them.
RESERVED_NAMES_VERSION = 1
#: 2026-11-27 (same run, whose answer said "No edits applied" over a 10-entry delta): an accepted
#: ``submit`` in an editing session answers how many weights differ from the bundle's
#: (``prompts.SUBMITTED_CHANGES``). Recorded in the header; a replay follows it.
SUBMIT_CHANGES_VERSION = 1
#: Write noise (2026-11-26, step 5b): a write's answer names the entries it changed by no more
#: than a reading rounds away, and the transcript gets ``{event: write_noise}``. Recorded in the
#: header; a replay follows it (absent: neither, as the run was recorded).
WRITE_NOISE_VERSION = 1
#: 2026-11-28 (step 5c, run ``wsp_0d4a2f916c9c``): an edit that recreates weights already
#: measured and then reverted says so (``prompts.EDIT_REPEATED``) and counts in
#: ``repeated_turns``. Recorded in the header of every editing session; a replay follows it.
EDIT_REPEATS_VERSION = 1
#: 2026-11-28 (step 5d, run ``wsp_3417a18bc85d``): a cut-off reply in a session with the output
#: cap off lowers the session's output ceiling; recorded in the header, a replay follows it.
CUT_OFF_CAP_VERSION = 1
#: 2026-11-28 (E7 §5.3 gap 2): in a program session no refusal quotes the program's own text
#: (``wsp_worker.program``'s docstring). Recorded in the header of every program session; a
#: replay follows it, so a program run recorded before replays with the texts it was given.
PROGRAM_ERRORS_VERSION = 1
#: Step 5d's cap before ``context.output_cap``'s bounds: twice the capped sessions' default
#: (``context.DEFAULT_OUTPUT_CAP``, 2,048), since this session chose to run without a cap, and the
#: reasoning allowance's default (``context.DEFAULT_THINKING_TOKENS``): room for one action and
#: its short account, and an eighth of the 32,000 that four of E7's replies ran to.
CUT_OFF_CAP_TOKENS = 4_096
#: The edits step 5c watches: the two tools that write weights the editor computed.
#: ``revert`` and ``restore`` are what it watches FOR, so they are not among them.
EDIT_TOOLS = frozenset({"python", "write_tensor"})
#: The tools whose run leaves the weights they ran on reverted (step 5c).
REVERT_TOOLS = frozenset({"revert", "restore"})
#: Tools whose answer depends only on their arguments and the weights, besides every tool with a
#: forward-pass cost: a repeat of one on the same weights would answer the same. ``python`` is
#: not one (the kernel keeps state between runs), nor any tool that edits or submits.
PURE_TOOLS = frozenset({"list_tensors", "describe_tensor", "read_tensor", "diff"})
#: A paged handoff's summary keeps the full ledger while it takes at most this share of the room
#: the bare restart leaves before the next handoff; else the compact one (the newest rows), else
#: the page ids alone (step 8), so a small window is not crowded by its own summary.
HANDOFF_LEDGER_SHARE = 0.25
PR_SET_DUMPABLE = 4
#: Step 3c: lost replies in a row that end the session, its stop, and the cause it records (the
#: run completes, as at a budget: ``wsp.runs.BUDGET_STOPS``; no new stop reason, no schema change).
MAX_LOST_IN_A_ROW = 2
LOST_STOP = "time_budget"
LOST_CAUSE = "editor_reply_lost"
PRACTICE_FILE_ROLES = (
    "practice_activations",
    *bundle_mod.PRACTICE_ROLES,
    *bundle_mod.SUPPLEMENT_FILE_ROLES,
)


def provider() -> dict[str, str]:
    from wsp_worker.miwsp_engine import PROVIDER_ID, VERSION

    return {"provider": PROVIDER_ID, "provider_version": VERSION}


def set_undumpable() -> None:
    """``PR_SET_DUMPABLE 0``: a same-uid grandchild can then open none of this process's
    ``/proc/<pid>/{environ,mem,fd,map_files}`` (FR-018.14). Raises when the call fails."""
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(PR_SET_DUMPABLE, 0, 0, 0, 0) != 0:
        raise KernelError("worker_kernel_isolation_failed", "PR_SET_DUMPABLE failed")


def unlink_practice(work: Path, manifest: Mapping[str, Any]) -> list[str]:
    """Remove the practice and activation files from ``/work`` (their maps stay open here)."""
    gone = []
    for entry in manifest.get("files", []):
        if entry.get("role") in PRACTICE_FILE_ROLES:
            path = work / str(entry["name"])
            path.unlink(missing_ok=True)
            gone.append(str(path))
    return gone


def cost_of(usage: proto.Usage, prices: Mapping[str, Any]) -> float:
    """USD for one request, as the gateway prices it (``meter.metered``): an absent cache price
    is the input price."""

    def price(key: str) -> float:
        value = prices.get(key)
        return float(prices["input"] if value is None else value)

    fresh = max(usage.input_tokens - usage.cache_read - usage.cache_write, 0)
    return (
        fresh * price("input")
        + usage.cache_read * price("cache_read")
        + usage.cache_write * price("cache_write")
        + usage.output_tokens * price("output")
    ) / 1e6


def now() -> str:
    return dt.datetime.now(dt.UTC).isoformat().replace("+00:00", "Z")


def observation_context(bundle: Bundle, store: TensorStore) -> observation_tools.Context:
    """The practice view and the live head tensors (editable from the store, the rest fixed);
    the same context the ``weightspace`` adapter built (``observation_context``)."""
    arrays = bundle.practice.arrays
    fixed = {n: t.values().astype(np.float64) for n, t in bundle.tensors.items()}

    def current() -> dict[str, np.ndarray]:
        live = {n: np.asarray(v, dtype=np.float64) for n, v in store.values.items()}
        return {**fixed, **live}

    view = observation_tools.PracticeView(
        scorer=Scorer(
            bundle.manifest["function"], bundle.practice.activations, arrays["practice_offsets"]
        ),
        labels=np.asarray(arrays["practice_labels"]),
        validation=np.asarray(arrays["practice_validation_mask"], dtype=bool),
        truncated=np.asarray(arrays["practice_truncated_mask"], dtype=bool),
        target_fpr=observation_tools.target_fpr_of(bundle.manifest),
        shipped_bar=observation_tools.shipped_bar_of(bundle.manifest),
        supplement=supplement_view(bundle),
    )
    return observation_tools.Context(
        view, current, fixed, program.seed_for(bundle.manifest["bundle_sha256"])
    )


def supplement_view(bundle: Bundle) -> observation_tools.SupplementView | None:
    """The bundle's validation-only supplements behind their own scorer, or ``None``."""
    supplement = bundle.practice.supplement
    if supplement is None:
        return None
    arrays = supplement.arrays
    return observation_tools.SupplementView(
        scorer=Scorer(
            bundle.manifest["function"], supplement.activations, arrays["supplement_offsets"]
        ),
        labels=np.asarray(arrays["supplement_labels"]),
        truncated=np.asarray(arrays["supplement_truncated_mask"], dtype=bool),
        entries=supplement.entries,
    )


def context_text(bundle: Bundle) -> str:
    """The function's description and the tensors' roles. NEVER the practice data."""
    lines = [bundle.manifest["function"]["description"], "", "Tensors:"]
    for t in bundle.manifest["tensors"]:
        role = "editable" if t["editable"] else "read-only (not given to you)"
        lines.append(f"- {t['name']} {t['dtype']} {t['shape']}: {role}")
    return "\n".join(lines)


#: Step 5b: the noisy entries a write's note names per tensor, and its event records.
NOISE_SHOWN = 16
NOISE_EVENT_INDICES = 64


def _noise_where(name: str, shape: tuple[int, ...], idx: np.ndarray) -> str:
    """``weight at 0, 1, 3`` (a vector) or ``W at (0, 3), (1, 2)``, the first :data:`NOISE_SHOWN`
    positions and how many more."""
    shown = idx[:NOISE_SHOWN].tolist()
    if len(shape) <= 1:
        parts = [str(i) for i in shown]
    else:
        parts = [
            "(" + ", ".join(str(int(p)) for p in np.unravel_index(i, shape)) + ")" for i in shown
        ]
    more = f" and {idx.size - len(shown):,} more" if idx.size > len(shown) else ""
    return f"{name} at {', '.join(parts)}{more}"


class Session:
    """One engine session (or its replay) from the child line."""

    def __init__(
        self,
        line: Mapping[str, Any],
        *,
        client: ModelClient | None = None,
        kernel_timeout_s: float = EXEC_TIMEOUT_S,
    ) -> None:
        started = time.monotonic()
        spec, paths = line["spec"], line["paths"]
        self.line = line
        self.request: Mapping[str, Any] = spec["request"]
        self.editor: Mapping[str, Any] = spec.get("editor") or {}
        self.replaying = line["mode"] == "replay"
        self.style = str(spec.get("replay_of_mode", "session") if self.replaying else line["mode"])
        self.work, self.out, self.tmp = Path(paths["work"]), Path(paths["out"]), Path(paths["tmp"])
        self.run_id = str(line["run_id"])
        self.kind = str(self.request["kind"])
        self.bundle = bundle_mod.load(self.work, spec["manifest"], list(line["files"]))
        self.can_edit = self.style != "program" and self.kind in ("repair", "create")
        # FU-8 (F18-D15): the frozen request's ``python: "off"`` (absent: on).
        self.python_on = self.request.get("python") != "off"
        self.store = TensorStore(
            self.bundle,
            self.request["change_limits"],
            zero_start=self.kind == "create" and self.style != "program",
        )
        # Step 5a: on in every session; a replay as its original ran (the header's ``judging``).
        self.judging = (
            read_header(self._pushed("transcript")).get("judging") == judging.VERSION
            if self.replaying
            else True
        )
        # Step 3b: on in every session; a replay as its original ran (the header's ``feedback``).
        self.feedback = (
            read_header(self._pushed("transcript")).get("feedback") == FEEDBACK_VERSION
            if self.replaying
            else True
        )
        # FU-2 (operator, 2026-11-25): ``submit`` offers the claimed ``figures`` where the submitted
        # weights are measured afterwards; a replay as its original ran (the header's ``figures``;
        # absent: not offered, so a run recorded before FU-2 replays with its own tool list).
        self.figures = submitted_measure.applies(self.request, self.bundle.manifest) and (
            read_header(self._pushed("transcript")).get("figures")
            == submitted_measure.FIGURES_VERSION
            if self.replaying
            else True
        )
        # Step 5b (2026-11-26): on in every editing session; a replay as its original ran (the
        # header's ``write_noise``; absent: off, so a run recorded before it replays unchanged).
        self.write_noise = self.can_edit and (
            read_header(self._pushed("transcript")).get("write_noise") == WRITE_NOISE_VERSION
            if self.replaying
            else True
        )
        # 2026-11-27 (run wsp_ba149c57f611): on in every session with Python; a replay as its
        # original ran (the header's ``reserved_names``; absent: off, so the names stay as the
        # recorded code bound them and every recorded answer replays bit for bit).
        self.reserved_names = self.python_on and (
            read_header(self._pushed("transcript")).get("reserved_names") == RESERVED_NAMES_VERSION
            if self.replaying
            else True
        )
        # 2026-11-27: the accepted submit's answer counts the changed weights in an editing
        # session; a replay as its original ran (the header's ``submit_changes``).
        self.submit_changes = self.can_edit and (
            read_header(self._pushed("transcript")).get("submit_changes") == SUBMIT_CHANGES_VERSION
            if self.replaying
            else True
        )
        # Step 5c (2026-11-28, run wsp_0d4a2f916c9c): on in every editing session; a replay as
        # its original ran (the header's ``edit_repeats``; absent: off, so a run recorded before,
        # E1r's among them, replays with the answers it was given).
        self.edit_repeats = self.can_edit and (
            read_header(self._pushed("transcript")).get("edit_repeats") == EDIT_REPEATS_VERSION
            if self.replaying
            else True
        )
        # Step 5d (2026-11-28, run wsp_3417a18bc85d): on in every session; a replay as its
        # original ran (the header's ``cut_off_cap``; absent: off).
        self.cut_off_cap = (
            read_header(self._pushed("transcript")).get("cut_off_cap") == CUT_OFF_CAP_VERSION
            if self.replaying
            else True
        )
        # 2026-11-28 (E7 §5.3 gap 2): on in every program session; a replay as its original ran
        # (the header's ``program_errors``; absent: off).
        self.program_errors = self.style == "program" and (
            read_header(self._pushed("transcript")).get("program_errors") == PROGRAM_ERRORS_VERSION
            if self.replaying
            else True
        )
        #: Step 5d: the cap a cut-off reply lowered an uncapped session to, or ``None``.
        self.lowered_cap: int | None = None
        #: The figures the accepted ``submit`` stated (FU-2), or ``None``.
        self.claimed: dict[str, Any] | None = None
        self._affine_warned = False
        self._final_turn = False
        observation = observation_tools.build(
            list(self.request.get("observation_tools") or []),
            observation_context(self.bundle, self.store),
        )
        self.programs = (
            ProgramTools(
                self.bundle,
                self.request,
                self.run_id,
                created_by=provider(),
                judging=self.judging,
                on_affine=lambda eq: self._affine_event("submit_program", eq),
                data_free_errors=self.program_errors,
            )
            if self.style == "program"
            else None
        )
        budgets = self.request["budgets"]
        self.forward = tools_mod.ForwardBudget(int(budgets.get("forward_passes") or 0))
        self.max_steps = int(budgets.get("max_steps") or DEFAULT_MAX_STEPS)
        self.token_budget = int(int(budgets["tokens"]) * BUDGET_SHARE)
        self.time_s = float(budgets["time_s"])
        self.time_budget = self.time_s * BUDGET_SHARE
        # The supervisor's kill (step 4a): the time it had left when it wrote the line, from now.
        left = line.get("time_left_s")
        self.time_left_s = self.time_s if left is None else max(0.0, float(left))
        self._deadline = started + self.time_left_s
        prices = self.editor.get("prices") or {}
        priced = prices.get("input") is not None and prices.get("output") is not None
        self.prices = prices if priced else None
        cap = budgets.get("cost_usd")
        self.cost_budget = None if not priced or cap is None else float(cap) * BUDGET_SHARE
        self.observation = observation
        if self.replaying:
            self.transcript_in = self._pushed("transcript")
            self.protocol = str(read_header(self.transcript_in).get("protocol") or proto.TOOL_CALLS)
        else:
            self.protocol = str(self.editor.get("action_protocol") or proto.TOOL_CALLS)
        self._client = client
        self.kernel_timeout_s = kernel_timeout_s
        self.kernel: Kernel | None = None
        window = context.window_tokens(self.editor)
        # Feature 019 (FR-019.15): the output ceiling the backend resolved at push; the mirrored
        # gateway sizing (``context.gateway_cap``) and the per-turn cap never exceed it.
        self.ceiling = context.editor_ceiling(self.editor)
        if self.replaying:
            # Feature 019 P2 (12.5, operator 2026-11-24): a replay has no endpoint, so it sizes to
            # the window and the ceiling its ORIGINAL recorded (the transcript's window event);
            # 8,192 and 32,000 only for a transcript from before 019 that recorded none.
            recorded = read_window_event(self.transcript_in)
            if recorded is not None:
                window, self.ceiling = recorded["tokens"], recorded["ceiling"]
        self.window = window
        #: What the per-turn cap is sized from: the pushed editor, or a replay's recorded figures.
        self.sizing: Mapping[str, Any] = {
            **self.editor,
            "window_tokens": window,
            "output_ceiling_tokens": self.ceiling,
        }
        # The output cap (2026-11-21): off names none. The guard then reserves the largest output
        # seen so far (at least the gateway's floor) instead of a cap nothing sends.
        self.capped = context.capped(self.request)
        #: Feature 020 (TD-6): thinking fields and the output cap are keyed on kind and locality.
        self.locality = str(self.editor.get("locality", "local"))
        self.hosted_anthropic = (
            str(self.editor.get("kind", "")) == proto.ANTHROPIC and self.locality == "external"
        )
        if not self.capped:
            self.cap = 0
        elif self.hosted_anthropic:
            # Feature 020 (F20-D16): the endpoint's output ceiling, bounded by the window. Adaptive
            # thinking counts toward max_tokens on every current model (FTID §0, L4), so the
            # eighth-of-the-window cap would leave it no room; the gateway's clamp still bounds it.
            self.cap = min(self.ceiling, self.window // 2)
        else:
            self.cap = context.editor_output_cap(self.sizing)
        self._largest_output = 0
        # Thinking (step 7): the frozen value; the allowance is set in ``run`` (it can refuse).
        self.thinking = self.request.get("thinking")
        self.thinking_tokens = self.request.get("thinking_tokens")
        self.allowance = 0
        self.guard = context.Guard(window, self._reserve())
        # Paging and the ledger (step 8): the frozen request's ``paging``, or, in a replay, the
        # configuration the original recorded in its transcript header (none: off, as it ran).
        # Feature 020 (FR-020.6, TD-5): the history mode from ``spec.editor`` (live) or the
        # transcript header (a replay; absent: paging, as it ran).
        if self.replaying:
            header = read_header(self.transcript_in)
            self.paging = paging.Config.from_header(
                header.get("paging"),
                window,
                header.get("paging_off_reason"),
                header.get("history_mode"),
            )
        else:
            self.paging = paging.Config.for_request(
                self.request, window, str(self.editor.get("history_mode") or paging.PAGING_MODE)
            )
        self.append_only = self.paging.history == paging.APPEND_ONLY
        #: The last body sent in this conversation (append-only): the prefix check compares the
        #: next one with it (F20-D15); a handoff starts a new conversation and clears it.
        self._last_wire: dict[str, Any] | None = None
        #: Request starts in this conversation, for the timing events (FR-020.14).
        self._conv_started_prev: float | None = None
        self.ledger = paging.Ledger(self.paging.rows)
        self.pager = paging.Pager(self.paging)
        #: The first ledger row the transient tail shows: rows before the last handoff ride in its
        #: summary, so they are not sent twice.
        self._ledger_from = 0
        #: The ledger in the history (written at the last page event), and the rows it holds.
        self._ledger_at: paging.Attached | None = None
        self._ledger_written = 0
        self.peak_prompt_tokens = 0
        self.reasoning_turns = 0
        # The repeat guard (step 3b): (tool, arguments, weights) -> the turn that ran it.
        self.repeated_turns = 0
        self._repeats_in_a_row = 0
        self._ran: dict[tuple[str, str, str], int] = {}
        # Step 5c: weights sha256 -> (turn, tool, figures) of the first measurement that ran on
        # them; and -> the turn of the first revert or restore that left them after it.
        self._measured: dict[str, tuple[int, str, str]] = {}
        self._reverted: dict[str, int] = {}
        # Step 3c: lost replies in a row, and in all.
        self._lost_in_a_row = 0
        self.lost_replies = 0
        # Counters and the outcome.
        self.answer: tuple[str, float | None] | None = None
        self.steps = self.handoffs = self.refused_turns = 0
        self.tokens = proto.Usage()
        self.requests = 0
        self.cost_usd = 0.0
        self.stop: str | None = None
        self.failure: dict[str, str] | None = None
        self.replay_divergence: int | None = None
        self._assistant_turns = 0
        # What the turns so far took (step 4a): the longest request, the longest request plus
        # its action, the slowest forward pass, the most tokens one request used.
        self._longest_request = self._longest_turn = 0.0
        self._slowest_forward: float | None = None
        self._largest_turn_tokens = 0
        self._noticed: set[str] = set()
        activations = self.bundle.practice.activations
        self._forward_prior_s = float(activations.nbytes) / FORWARD_BYTES_PER_S
        self._t0 = time.monotonic()
        self.toolbox = tools_mod.Toolbox(self._specs(), self.forward)
        self.writer: TranscriptWriter | None = None
        #: The tensors as the patch carries them, for ``submitted_measure`` (program style: the
        #: stored program's).
        self._submitted: dict[str, Any] = {}

    # ----------------------------------------------------------------------------- setup

    def _pushed(self, role: str) -> Path:
        entry = next((f for f in self.line["files"] if f["role"] == role), None)
        if entry is None:
            raise bundle_mod.BundleError(f"the push carries no {role}", "worker_spec_invalid")
        return self.work / str(entry["name"])

    def _specs(self) -> list[tools_mod.ToolSpec]:
        specs = tools_mod.read_specs(self._read_tensor, self._describe_tensor, self._list_tensors)
        if self.python_on:
            specs.append(tools_mod.python_spec(self._python, can_edit=self.can_edit))
        if self.can_edit:
            specs += tools_mod.edit_specs(
                self._write_tensor, self.store.diff, self._checkpoint, self._restore, self._revert
            )
        if self.paging.pages_out:
            specs.append(tools_mod.recall_spec(self.pager.recall))
        elif self.append_only:
            # Feature 020 (FR-020.8, TD-4): the same name, read from an offset.
            specs.append(tools_mod.recall_from_spec(self.pager.recall_from))
        specs.append(tools_mod.submit_spec(self._submit, figures=self.figures))
        specs += [
            tools_mod.forward_spec(s.name, s.description, s.parameters, s.cost, s.fn)
            for s in self.observation
        ]
        if self.programs is not None:
            cost = ProgramTools.COST
            specs.append(
                tools_mod.ToolSpec(
                    "test_program",
                    f"[forward pass, costs {cost}] {program_tools_text()[0]}",
                    tools_mod.closed(program_tools_text()[2]),
                    self.programs.test_program,
                    cost,
                )
            )
            specs.append(
                tools_mod.ToolSpec(
                    "submit_program",
                    program_tools_text()[1],
                    tools_mod.closed(program_tools_text()[2]),
                    self.programs.submit_program,
                )
            )
        return specs

    def client(self) -> ModelClient:
        if self._client is None:
            if self.replaying:
                self._client = read_script(self.transcript_in)
            else:
                gateway = self.line.get("gateway")
                if not isinstance(gateway, Mapping):
                    raise bundle_mod.BundleError(
                        "a session needs its gateway", "worker_spec_invalid"
                    )
                kind = str(self.editor["kind"])
                self._client = GatewayClient(
                    kind,
                    str(self.editor.get("model", "")),
                    gateway,
                    # Feature 020 (FR-020.12, TD-6): cache markers on kind and history mode.
                    cache=kind == proto.ANTHROPIC and self.append_only,
                )
        return self._client

    def _isolate_and_spawn(self) -> None:
        """FR-018.14, in this order: undumpable, practice files gone, THEN the grandchild."""
        set_undumpable()
        unlink_practice(self.work, self.bundle.manifest)
        if self.python_on:
            self.kernel = Kernel(
                self.tmp / "kernel",
                lambda: self.store.values,
                edit=self.can_edit,
                timeout_s=self.kernel_timeout_s,
                rebind=self.reserved_names,
            )
            self.kernel.start()

    # ----------------------------------------------------------------------------- tools

    def _list_tensors(self, pattern: str | None = None) -> str:
        values = self.store.values
        return tools_mod.list_tensors(
            list(values), {n: v.shape for n, v in values.items()}, self.store.dtypes, pattern
        )

    def _describe_tensor(self, name: str) -> str:
        return tools_mod.describe(name, self.store.values[name], self.store.dtypes[name])

    def _read_tensor(
        self, name: str, index: str | None = None, precision: int = tools_mod.DISPLAY_PRECISION
    ) -> str:
        arr = np.asarray(self.store.values[name][tools_mod.parse_index(index)])
        if arr.size > tools_mod.READ_LIMIT:
            return (
                f"slice has {arr.size:,} values (shape {arr.shape}); limit is "
                f"{tools_mod.READ_LIMIT}. Read a smaller slice."
            )
        text = f"{name}[{index or ':'}] shape={arr.shape}\n{tools_mod.fmt_array(arr, precision)}"
        if not self.feedback:
            return text  # cut by the toolbox, as it was before 2026-11-22
        return tensor_read.fit(name, index, arr, text, tools_mod.MAX_TOOL_CHARS, precision)

    def _sync(self, names: list[str]) -> None:
        if self.kernel is not None:
            self.kernel.sync(names)

    def _python(self, code: str) -> str:
        assert self.kernel is not None
        result = self.kernel.exec(code)
        if not result.deltas:
            return result.output
        updates = {}
        for name, (idx, vals) in result.deltas.items():
            new = self.store.values[name].copy()
            flat = new.reshape(-1)
            flat[idx] = vals.astype(new.dtype)
            updates[name] = new
        before = {name: self.store.values[name].copy() for name in updates}
        applied = self.store.try_apply(updates)
        # Accepted: the files take the new values; refused: the same files are re-mapped, so W
        # reverts to the last accepted weights. Either way R and W agree again.
        self._sync(sorted(updates))
        if isinstance(applied, Refused):
            return f"{result.output}\n{applied.text} W was reset to the last accepted weights."
        return result.output + self._write_noise("python", before)

    def _write_tensor(self, name: str, values: Any, index: str | None = None) -> str:
        before = self.store.snapshot()
        was = {name: self.store.values[name].copy()} if name in self.store.values else {}
        applied = self.store.write(name, values, tools_mod.parse_index(index))
        if isinstance(applied, Refused):
            return applied.text
        self._sync([name])
        report = self._edit_report(f"wrote {name}[{index or ':'}]", before)
        return report + self._write_noise("write_tensor", was)

    def _write_noise(self, tool: str, before: Mapping[str, np.ndarray]) -> str:
        """Step 5b: the note a write's answer ends with when entries it changed moved by no more
        than a reading at :data:`tools_mod.DISPLAY_PRECISION` digits rounds away
        (``store.below_display``), and the transcript's ``{event: write_noise}``; ``""`` when
        none did, or when the session (or the run a replay follows) has it off. Never refuses and
        never changes a weight."""
        if not self.write_noise or not before:
            return ""
        precision = tools_mod.DISPLAY_PRECISION
        changed, noise = self.store.below_display(before, precision)
        if not noise:
            return ""
        k = sum(int(idx.size) for idx in noise.values())
        if self.writer is not None:
            self.writer.write(
                {
                    "event": "write_noise",
                    "turn": self.steps,
                    "tool": tool,
                    "precision": precision,
                    "changed": changed,
                    "noise": k,
                    "tensors": [
                        {
                            "name": name,
                            "entries": int(idx.size),
                            "indices": idx[:NOISE_EVENT_INDICES].tolist(),
                        }
                        for name, idx in noise.items()
                    ],
                }
            )
        where = "; ".join(
            _noise_where(name, self.store.values[name].shape, idx) for name, idx in noise.items()
        )
        return "\n" + prompts.WRITE_NOISE.format(k=k, n=changed, precision=precision, where=where)

    def _checkpoint(self, label: str) -> str:
        self.store.checkpoint(label)
        return f"saved checkpoint {label!r}"

    def _restore(self, label: str) -> str:
        before = self.store.snapshot()
        applied = self.store.restore(label)
        if isinstance(applied, Refused):
            return applied.text
        self._sync(self.store.names())
        return self._edit_report(f"restored {label!r}", before)

    def _revert(self, name: str | None = None) -> str:
        before = self.store.snapshot()
        applied = self.store.revert(name)
        if isinstance(applied, Refused):
            return applied.text
        self._sync(self.store.names())
        return self._edit_report(f"reverted {name or 'all tensors'}", before)

    def _edit_report(self, head: str, before: Mapping[str, Any]) -> str:
        """What an edit answers: what it changed, and where that leaves the weights."""
        if not self.feedback:
            return f"{head}\n{self.store.diff()}"
        return prompts.ACTION_CHANGED.format(
            head=head,
            n=self.store.changed_since(before),
            total=sum(v.size for v in self.store.values.values()),
            diff=self.store.diff(),
        )

    def _repeated(self, first: int) -> str:
        """The repeat guard's answer, with the page that holds the first result if there is one."""
        text = prompts.REPEATED.format(turn=first)
        page = paging.page_id(first)
        if self.feedback and page in self.pager.pages:
            text += prompts.REPEATED_PAGE.format(page=page)
        return text

    def _edit_repeated(self, tool: str, before: str, output: str, measured: bool) -> str:
        """Step 5c, after an action RAN on the weights whose sha256 is ``before``: a measurement
        is recorded against them, a revert or restore that left measured weights is recorded, and
        an edit that CHANGED the weights to ones measured and then reverted answers the note
        (``"\\n"`` + :data:`prompts.EDIT_REPEATED`), counts in ``repeated_turns`` and writes the
        transcript's ``{event: edit_repeated}``. ``""`` for everything else; never refuses and
        never changes a weight."""
        if measured:
            figures = " ".join(f"{k}={v}" for k, v in paging.numbers_of(output)) or "no figures"
            self._measured.setdefault(before, (self.steps, tool, figures))
            return ""
        after = self.store.sha256()
        if after == before:
            return ""  # nothing changed: a read, a refused write, a revert with nothing to undo
        if tool in REVERT_TOOLS:
            if before in self._measured:
                self._reverted.setdefault(before, self.steps)
            return ""
        if tool not in EDIT_TOOLS or after not in self._reverted:
            return ""
        turn, measured_with, figures = self._measured[after]
        reverted = self._reverted[after]
        self.repeated_turns += 1
        if self.writer is not None:
            self.writer.write(
                {
                    "event": "edit_repeated",
                    "turn": self.steps,
                    "tool": tool,
                    "weights": paging.short(after),
                    "measured_turn": turn,
                    "reverted_turn": reverted,
                    "n": self.repeated_turns,
                }
            )
        return "\n" + prompts.EDIT_REPEATED.format(
            weights=paging.short(after),
            turn=turn,
            tool=measured_with,
            figures=figures,
            reverted=reverted,
        )

    def _lower_ceiling(self) -> str:
        """Step 5d, after a reply cut off in the turn loop: in a session with the output cap off,
        the first one lowers the session's output ceiling to :data:`CUT_OFF_CAP_TOKENS` (bounded
        as the per-turn cap is, and by the pushed ceiling), writes ``{event: ceiling_lowered}``
        and answers :data:`prompts.CUT_OFF_CAPPED`; ``""`` in a capped session, or once lowered."""
        if self.capped or self.lowered_cap is not None:
            return ""
        cap = min(context.output_cap(self.window, CUT_OFF_CAP_TOKENS), self.ceiling)
        self.lowered_cap = cap
        self.guard.cap = self._reserve()
        if self.writer is not None:
            self.writer.write(
                {
                    "event": "ceiling_lowered",
                    "turn": self.steps,
                    "from": self.ceiling,
                    "to": cap,
                    "why": "cut_off",
                }
            )
        return prompts.CUT_OFF_CAPPED.format(cap=cap)

    def _repeats_out(self) -> bool:
        count = self._repeats_in_a_row if self.feedback else self.repeated_turns
        return count >= MAX_REPEATS

    def _affine_event(self, tool: str, eq: equivalence.Equivalence) -> None:
        if self.writer is not None:
            self.writer.write(
                {
                    "event": "affine_equivalent_submit",
                    "turn": self.steps,
                    "tool": tool,
                    "equivalence": eq.kind,
                    "scale": eq.scale,
                }
            )

    def _current(self) -> dict[str, np.ndarray]:
        fixed = {n: t.values().astype(np.float64) for n, t in self.bundle.tensors.items()}
        return {**fixed, **{n: np.asarray(v, np.float64) for n, v in self.store.values.items()}}

    def _submit(
        self,
        answer: str,
        confidence: float | None = None,
        figures: Mapping[str, Any] | None = None,
    ) -> str:
        broken = self.store.violations()
        if broken:
            return prompts.SUBMIT_REFUSED.format(details="; ".join(broken))
        if self.judging and not self._affine_warned and not self._final_turn:
            # Step 5a. The parent is the bundle's head, whatever the store started from.
            parent = {n: t.values().astype(np.float64) for n, t in self.bundle.tensors.items()}
            same = equivalence.for_bundle(self.bundle.manifest, parent, self._current())
            if same is not None:
                self._affine_warned = True
                self._affine_event("submit", same)
                return judging.AFFINE_SUBMIT.format(what=equivalence.describe(same), tool="submit")
        self.answer = (str(answer), None if confidence is None else float(confidence))
        self.claimed = None if figures is None else dict(figures)
        if self.submit_changes:
            # 2026-11-27 (run wsp_ba149c57f611 answered "No edits applied" over 10 changed
            # entries): the answer states the delta the patch will carry, against the bundle.
            changed = sum(self.store.changes().values())
            return prompts.SUBMITTED_CHANGES.format(
                n=changed, total=sum(int(v.size) for v in self.store.values.values())
            )
        return "submitted"

    def _max_tokens(self, think: bool = True) -> int | None:
        """The request's cap: the action's cap plus the reasoning allowance (none for a turn sent
        with thinking off); ``None`` when the output cap is off (the gateway's window-aware
        default applies)."""
        if not self.capped:
            if self.lowered_cap is not None:  # step 5d: a cut-off reply lowered the ceiling
                return self.lowered_cap + (self.allowance if think else 0)
            return None
        return self.cap + (self.allowance if think else 0)

    def _turn_thinking(
        self, messages: list[dict[str, Any]], extra: int
    ) -> tuple[Any, int, dict[str, Any] | None]:
        """``(thinking, allowance, skipped)`` for an uncapped thinking turn (fix 2026-11-21, run
        ``wsp_855a74f820ff``: a 256-token cap left an allowance of 1 and every action was cut
        mid-JSON). The engine sizes the cap the gateway will (``context.gateway_cap`` from the
        gateway's own prompt estimate) and sends the reasoning share the gateway would leave;
        when that share is under :data:`context.MIN_REASONING_TOKENS`, the turn is sent with
        thinking off and ``skipped`` says why, for the transcript."""
        if self.thinking != proto.THINKING_ON or self.capped or self.lowered_cap is not None:
            return self.thinking, self.allowance, None
        prompt = self.guard.gateway_prompt(messages, tools_bytes=extra)
        cap = context.gateway_cap(self.window, prompt, self.ceiling)
        share = context.reasoning_share(self.allowance, cap)
        if share < context.MIN_REASONING_TOKENS:
            return proto.THINKING_OFF, 0, {"why": "room", "cap": cap, "share": share}
        return self.thinking, share, None

    def _room(self, messages: list[dict[str, Any]], extra: int) -> int:
        """What the window leaves an uncapped answer, by the gateway's own prompt estimate."""
        return self.window - self.guard.gateway_prompt(messages, tools_bytes=extra)

    def _reserve(self) -> int:
        """The answer the context guard keeps room for: the cap sent, or, with the cap off, the
        largest output observed so far (at least the gateway's floor) plus the allowance."""
        if self.capped:
            return self.cap + self.allowance
        if self.lowered_cap is not None:
            return self.lowered_cap + self.allowance
        observed = max(self._largest_output, context.MIN_OUTPUT_TOKENS)
        return observed + self.allowance

    def _set_thinking(self) -> None:
        """Step 7: refuse ``on`` for an endpoint that cannot carry it, bound the allowance by the
        window, and size the guard for the whole answer (reasoning plus action)."""
        kind, serving = str(self.editor.get("kind", "")), str(self.editor.get("serving", ""))
        proto.check_thinking_supported(self.thinking, kind, serving, self.locality)
        if self.thinking == proto.THINKING_ON:
            self.allowance = context.thinking_allowance(self.window, self.cap, self.thinking_tokens)
            self.guard = context.Guard(self.window, self._reserve())
        if self.writer is not None:
            self.writer.write(
                {
                    "event": "thinking",
                    "thinking": self.thinking,
                    "sent": proto.thinking_fields(
                        self.thinking, kind, serving, self.allowance, self.protocol, self.locality
                    ),
                    "reasoning_allowance": self.allowance,
                    "output_cap": self._max_tokens(),
                }
            )

    # ----------------------------------------------------------------------------- the loop

    def _append(self, messages: list[dict[str, Any]], message: Mapping[str, Any]) -> None:
        messages.append(dict(message))
        if self.writer is not None:
            self.writer.write(message)

    def _time_left(self) -> float:
        return self._deadline - time.monotonic()

    def _final_reserve_s(self) -> float:
        """What the final submit-only turn needs: its request, then the outputs (the margin is
        never more than the 10% share, so a tiny budget still gets its turns)."""
        return self._longest_request + min(FINAL_MARGIN_S, (1 - BUDGET_SHARE) * self.time_s)

    def _time_stop_s(self) -> float:
        """Stop when this little time is left: the 10% share, or one more turn and the final."""
        return max((1 - BUDGET_SHARE) * self.time_s, self._longest_turn + self._final_reserve_s())

    def _over_budget(self) -> str | None:
        if self.tokens.total >= self.token_budget:
            return "token_budget"
        if self.cost_budget is not None and self.cost_usd >= self.cost_budget:
            return "cost_budget"
        if self._time_left() <= self._time_stop_s():
            return "time_budget"
        return None

    def _nearly_up(self) -> str | None:
        """The "nearly up" notice (step 4a), each budget at most once, or ``None``."""
        reasons: list[str] = []
        left = self._time_left()
        stop = self._time_stop_s()
        share_left = self.time_left_s - NOTICE_SHARE * self.time_budget
        if "time" not in self._noticed and (
            left <= share_left or left <= stop + 2 * self._longest_turn
        ):
            reasons.append(f"time: about {max(0, int(left))} s of {int(self.time_s)} s left")
            self._noticed.add("time")
        next_two = 2 * self._largest_turn_tokens
        used = self.tokens.total
        if "tokens" not in self._noticed and (
            used >= NOTICE_SHARE * self.token_budget or used + next_two >= self.token_budget
        ):
            reasons.append(f"tokens: {used:,} of {self.token_budget:,} used")
            self._noticed.add("tokens")
        if (
            "cost" not in self._noticed
            and self.cost_budget is not None
            and self.cost_usd >= NOTICE_SHARE * self.cost_budget
        ):
            reasons.append(f"cost: ${self.cost_usd:.2f} of ${self.cost_budget:.2f}")
            self._noticed.add("cost")
        dearest = max((s.cost for s in self.observation), default=0)
        forward_left = self.forward.budget - self.forward.used
        if (
            "forward" not in self._noticed
            and self.forward.used > 0
            and forward_left < 2 * max(dearest, 1)
        ):
            reasons.append(f"forward passes: {forward_left} of {self.forward.budget} left")
            self._noticed.add("forward")
        if not reasons:
            return None
        text = prompts.NEARLY_UP_EDIT if self.can_edit else prompts.NEARLY_UP_READ
        return text.format(details="; ".join(reasons))

    def _forward_fits(self, name: str, specs: Mapping[str, tools_mod.ToolSpec]) -> str | None:
        """``None`` when the action may run; the refusal text for a forward pass that would not
        finish before the final turn's reserve (it is not run and not charged)."""
        spec = specs.get(name)
        if spec is None or spec.cost is None:
            return None
        estimate = self._forward_prior_s if self._slowest_forward is None else self._slowest_forward
        left = self._time_left()
        if left - estimate >= self._final_reserve_s():
            return None
        return prompts.NO_TIME_FOR_FORWARD.format(left=max(0, int(left)), took=round(estimate))

    def _charge(self, usage: proto.Usage | None) -> None:
        self.requests += 1
        if usage is None:
            return
        self.tokens = proto.Usage(
            self.tokens.input_tokens + usage.input_tokens,
            self.tokens.output_tokens + usage.output_tokens,
            self.tokens.cache_read + usage.cache_read,
            self.tokens.cache_write + usage.cache_write,
            self.tokens.reasoning_tokens + usage.reasoning_tokens,
        )
        if self.prices is not None:
            self.cost_usd += cost_of(usage, self.prices)

    def _budget_text(self) -> str:
        return (
            f"steps used {self.steps} of {self.max_steps}; tokens used {self.tokens.total:,} of "
            f"{self.token_budget:,}; forward passes used {self.forward.used} of "
            f"{self.forward.budget}"
        )

    def _handoff(self, messages: list[dict[str, Any]], intro: str, tools_bytes: int = 0) -> None:
        notes = "\n".join(self.kernel.notes) if self.kernel is not None else ""
        parts = {
            "notes": notes or "(none)",
            "diff": self.store.diff() if self.can_edit else "(this session reads only)",
            "budget": self._budget_text(),
        }
        if self.paging.keeps_pages:
            # Step 8: the handoff is the last resort, and it keeps the ledger and the pages. A
            # summary that would not fit the window keeps less, in this order: the newest rows
            # and pages (compact), then the page ids alone (minimal).
            def paged(level: str) -> str:
                if level == "minimal":
                    ledger = "Ledger: (left out to fit the context window)"
                else:
                    ledger = self.ledger.render(compact=level == "compact")
                return prompts.HANDOFF_PAGED.format(
                    **parts,
                    ledger=ledger or "Ledger: (no rows yet)",
                    pages=self.pager.listing(level) if self.pager.pages else "(none)",
                )

            fresh = context.Guard(self.guard.window, self.guard.cap)
            fresh.ratios = list(self.guard.ratios)

            # The richest summary whose ledger takes at most HANDOFF_LEDGER_SHARE of the room the
            # bare restart (the minimal summary) leaves before the next handoff; else minimal.
            def used(text: str) -> int:
                trial = [messages[0], {"role": "user", "content": f"{intro}\n\n{text}"}]
                return fresh.estimate(trial, tools_bytes=tools_bytes)

            bare = paged("minimal")
            room = context.HANDOFF_SHARE * self.guard.window - used(bare)
            summary = bare
            for level in ("full", "compact"):
                text = paged(level)
                if used(text) - used(bare) <= HANDOFF_LEDGER_SHARE * room:
                    summary = text
                    break
            self._ledger_from = self._ledger_written = len(self.ledger.rows)
            self._ledger_at = None
        else:
            summary = prompts.HANDOFF.format(**parts)
        restarted = [messages[0], {"role": "user", "content": f"{intro}\n\n{summary}"}]
        messages[:] = restarted
        self.handoffs += 1
        self.guard.reset()
        # Feature 020 (FR-020.10): a new conversation, the one boundary the prefix rule has.
        self._last_wire = None
        self._conv_started_prev = None
        if self.writer is not None:
            event: dict[str, Any] = {"event": "handoff", "n": self.handoffs, "messages": restarted}
            if self.append_only:
                event["conversation"] = self.handoffs + 1
            self.writer.write(event)

    def _complete(
        self,
        messages: list[dict[str, Any]],
        specs: Mapping[str, tools_mod.ToolSpec],
        intro: str,
        *,
        think: bool = True,
    ) -> proto.Reply:
        """One request. ``think=False`` sends this turn with thinking off (the submit retry)."""
        kind, serving = str(self.editor.get("kind", "")), str(self.editor.get("serving", ""))
        thinking, allowance = self.thinking, self.allowance
        skipped: dict[str, Any] | None = None
        if not think and thinking == proto.THINKING_ON:
            thinking, allowance, skipped = proto.THINKING_OFF, 0, {"why": "submit_retry"}

        def fields_for(thinking: Any, allowance: int) -> dict[str, Any]:
            return proto.request_fields(
                self.protocol,
                kind,
                serving,
                specs,
                thinking=thinking,
                allowance=allowance,
                locality=self.locality,
            )

        fields = fields_for(thinking, allowance)
        extra = len(json.dumps(fields))
        # Step 8: the batched page event, then the history with the ledger as its tail; every
        # estimate below is of what is SENT.
        self._page_if_due(messages)
        outgoing = messages
        if not self.replaying:
            if self.guard.must_hand_off(outgoing, tools_bytes=extra):
                self._handoff(messages, intro, extra)
                if self.guard.must_hand_off(outgoing, tools_bytes=extra):
                    raise ClientError(
                        0,
                        "editor_context_window_too_small",
                        f"the task does not fit {self.guard.window} tokens with a "
                        f"{self.guard.cap}-token answer",
                    )
            if not self.capped:
                # Fix 2026-11-21: an uncapped turn whose room cannot hold one action is not sent
                # (it could only be cut off): the conversation is handed off first, and a task
                # that still leaves no room ends the session by name.
                if self._room(outgoing, extra) < context.MIN_ACTION_TOKENS:
                    self._handoff(messages, intro, extra)
                    if self._room(outgoing, extra) < context.MIN_ACTION_TOKENS:
                        raise ClientError(
                            0,
                            "editor_context_window_too_small",
                            f"the conversation leaves under {context.MIN_ACTION_TOKENS} of "
                            f"{self.window} tokens for an answer",
                        )
                if skipped is None:
                    thinking, allowance, skipped = self._turn_thinking(outgoing, extra)
                    fields = fields_for(thinking, allowance)
        if skipped is not None and self.writer is not None:
            self.writer.write({"event": "thinking_skipped", "turn": self.steps, **skipped})
        self.peak_prompt_tokens = max(
            self.peak_prompt_tokens, self.guard.prompt_estimate(outgoing, tools_bytes=extra)
        )
        sent = len(outgoing)
        max_tokens = self._max_tokens(think and skipped is None)
        client = self.client()
        # Feature 020 (FR-020.7, F20-D15, F20-D21): an append-only request is checked against the
        # previous one before it is sent, and the checked body is the body sent.
        body = (
            self._checked_body(client, outgoing, fields, max_tokens) if self.append_only else None
        )
        send_body = getattr(client, "complete_body", None)
        began = time.monotonic()
        try:
            if body is not None and callable(send_body):
                reply: proto.Reply = send_body(body)
            else:
                reply = client.complete(outgoing, fields=fields, max_tokens=max_tokens)
        finally:
            took = time.monotonic() - began
            self._timing(began, took)
        self._longest_request = max(self._longest_request, took)
        if reply.thinking_skipped and self.writer is not None:
            # The gateway's own estimate left no useful reasoning share (fix 2026-11-21).
            self.writer.write({"event": "thinking_skipped", "turn": self.steps, "why": "gateway"})
        if reply.reasoning:
            self.reasoning_turns += 1
        if reply.usage is not None:
            self._largest_turn_tokens = max(self._largest_turn_tokens, reply.usage.total)
            self._largest_output = max(self._largest_output, reply.usage.output_tokens)
            self.guard.cap = self._reserve()
        self._charge(reply.usage)
        if reply.usage is not None:
            sent_bytes = context.message_bytes(outgoing) + extra
            self.guard.reported(reply.usage.input_tokens, sent, sent_bytes)
        return reply

    def _checked_body(
        self,
        client: ModelClient,
        outgoing: list[dict[str, Any]],
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> dict[str, Any]:
        """The append-only body to send, after :func:`prefix.check` against the previous body of
        this conversation; an edit ends the session ``editor_history_edited`` before anything is
        sent (the stdout line names the path, never the content)."""
        body = client.body(outgoing, fields=fields, max_tokens=max_tokens)
        changed = prefix.check(self._last_wire, body)
        if changed is not None:
            print(
                json.dumps({"turn": self.steps, "error": "wsp_history_edited", "path": changed}),
                flush=True,
            )
            raise ClientError(
                0, "editor_history_edited", f"the request edits {changed} of the conversation"
            )
        self._last_wire = body
        return body

    def _timing(self, began: float, took: float) -> None:
        """Feature 020 (FR-020.14, F20-D7): one event per request, its start-to-start gap in this
        conversation (null for a conversation's first) and its duration. A replay recomputes
        nothing from them (its script skips event lines)."""
        gap = None if self._conv_started_prev is None else round(began - self._conv_started_prev, 3)
        self._conv_started_prev = began
        if self.writer is not None:
            self.writer.write(
                {
                    "event": "timing",
                    "turn": self.steps + (1 if self._final_turn else 0),
                    "gap_s": gap,
                    "duration_s": round(took, 3),
                }
            )

    # ----------------------------------------------------------------------------- paging

    def _page_if_due(self, messages: list[dict[str, Any]]) -> None:
        """Step 8's batched page event, before a request: a function of the turn and the
        history's bytes only, so a replay pages the same messages at the same turns. The event
        stubs the results past the keep window and moves the ledger to the history's end (the
        only times the history is rewritten: ``paging``'s module doc)."""
        if self.append_only:
            self._append_ledger_if_due(messages)
            return
        if not self.paging.pages_out:
            return
        # A ledger is needed once something has been paged (before that every result is still
        # verbatim); from then on new rows refresh it on the batch schedule.
        stale = self.pager.pages_out > 0 and len(self.ledger.rows) > self._ledger_written
        chosen, reason = self.pager.due(
            self.steps, messages, context.byte_estimate(messages), stale=stale
        )
        if reason is None:
            return
        paging.detach(messages, self._ledger_at)
        event = self.pager.page_out(self.steps, chosen, reason)
        text = self.ledger.render(self._ledger_from)
        self._ledger_at = paging.attach(messages, text)
        self._ledger_written = len(self.ledger.rows)
        event["ledger_rows"] = self._ledger_written - self._ledger_from
        # The endpoint's last count described the history before this rewrite.
        self.guard.reset()
        if self.writer is not None:
            self.writer.write(event)

    def _append_ledger_if_due(self, messages: list[dict[str, Any]]) -> None:
        """Feature 020 (FR-020.9, F20-D2): on the paging schedule, when the ledger has rows not
        yet sent, ONE new user message with the rows since the last one, appended before the
        next request (in the Anthropic wire form it joins the not-yet-sent user turn after its
        tool results). Nothing is attached, detached or paged out."""
        if len(self.ledger.rows) <= self._ledger_written:
            return
        if self.steps - self.pager.last_event < self.paging.every:
            return
        text = self.ledger.render(self._ledger_written, heading=paging.LEDGER_APPENDED_HEADING)
        rows = len(self.ledger.rows) - self._ledger_written
        self._append(messages, {"role": "user", "content": text})
        self._ledger_written = len(self.ledger.rows)
        self.pager.last_event = self.steps
        if self.writer is not None:
            self.writer.write({"event": "ledger_appended", "turn": self.steps, "rows": rows})

    def _after_action(
        self,
        appended: list[dict[str, Any]],
        action: proto.Action,
        output: str,
        before: str,
        forward_used: int,
    ) -> None:
        """Step 8, after an action RAN: its ledger row, and its result offered to the pager. In
        an append-only session (feature 020, F20-D19) the pager is offered the FULL text
        (``Toolbox.last_full``), and a recall's answer is never a page of its own."""
        if not self.paging.keeps_pages:
            return
        change, changed = self.store.diff_summary()
        row = self.ledger.record(
            self.steps,
            action.name,
            action.arguments,
            output,
            before=before,
            after=self.store.sha256(),
            change=change,
            changed=changed,
            measured=self.forward.used > forward_used,
        )
        if row is not None and self.writer is not None:
            self.writer.write({"event": "ledger", "row": row.document()})
        recalled = None
        if action.name == paging.RECALL:
            asked = str(action.arguments.get("page", "")).strip()
            if self.append_only:
                found = asked in self.pager.pages and not output.startswith(paging.REFUSED)
            else:
                found = asked in self.pager.pages and output == self.pager.pages[asked].text
            recalled = asked if found else None
            if self.writer is not None:
                self.writer.write(
                    {"event": "recall", "turn": self.steps, "page": asked, "found": found}
                )
        holder = next(
            (
                m
                for m in appended
                if m.get("role") == "tool" and m.get("tool_call_id") == action.call_id
            ),
            appended[-1] if appended else None,
        )
        if holder is not None and not (self.append_only and action.name == paging.RECALL):
            self.pager.offer(
                self.steps,
                action.name,
                action.arguments,
                self.toolbox.last_full if self.append_only else output,
                holder,
                ledgered=row is not None and bool(row.numbers),
                recalled=recalled,
            )

    def _answer_turn(
        self,
        messages: list[dict[str, Any]],
        reply: proto.Reply,
        decision: proto.Action | proto.Refusal,
        output: str,
    ) -> list[dict[str, Any]]:
        """Append the messages that answer ``reply``; returns them as they sit in the history."""
        followups = proto.followups(self.protocol, reply, decision, output)
        for m in followups:
            self._append(messages, m)
        appended = messages[len(messages) - len(followups) :]
        if self.replaying and self.replay_divergence is None:
            script = self._client
            recorded = getattr(script, "recorded", [])
            index = self._assistant_turns - 1
            mine = [str(m.get("content", "")) for m in followups]
            if index >= len(recorded) or recorded[index][: len(mine)] != mine:
                self.replay_divergence = index + 1
        return appended

    def _ran_key(
        self, decision: proto.Action | proto.Refusal, specs: Mapping[str, tools_mod.ToolSpec]
    ) -> tuple[str, str, str] | None:
        """The repeat guard's key for an action it watches (step 3b), else ``None``."""
        if not isinstance(decision, proto.Action):
            return None
        spec = specs.get(decision.name)
        if spec is None or (spec.cost is None and decision.name not in PURE_TOOLS):
            return None
        arguments = json.dumps(decision.arguments, sort_keys=True, default=str)
        return decision.name, arguments, self.store.sha256()

    def _repeat_of(
        self, decision: proto.Action | proto.Refusal, specs: Mapping[str, tools_mod.ToolSpec]
    ) -> tuple[tuple[str, str, str], int] | None:
        """``(key, the turn that ran it)`` when this action already ran on these weights."""
        key = self._ran_key(decision, specs)
        if key is None or key not in self._ran:
            return None
        return key, self._ran[key]

    def _lost(self, exc: ClientError, turn: int) -> bool:
        """Step 3c: record one lost reply; ``True`` when the session may go on (the editor is then
        told once), ``False`` when it was the second in a row (the stop and cause are set)."""
        self._lost_in_a_row += 1
        self.lost_replies += 1
        self.refused_turns += 1
        if self.writer is not None:
            self.writer.write(
                {
                    "event": "lost_reply",
                    "turn": turn,
                    "code": exc.code,
                    "message": exc.message[:500],
                    "n": self._lost_in_a_row,
                }
            )
        print(json.dumps({"turn": turn, "lost": exc.code, "tokens": self.tokens.total}), flush=True)
        if self._lost_in_a_row >= MAX_LOST_IN_A_ROW:
            self.stop = LOST_STOP
            self.failure = {
                "code": LOST_CAUSE,
                "message": (
                    f"{self._lost_in_a_row} replies in a row were lost ({exc.code}); each ran past "
                    "the call time limit. The session ended as at its time budget."
                ),
            }
            return False
        return True

    def _final_complete(
        self,
        messages: list[dict[str, Any]],
        specs: Mapping[str, tools_mod.ToolSpec],
        intro: str,
        *,
        think: bool = True,
    ) -> proto.Reply | None:
        """A request of the final submit-only turn; ``None`` when its reply was lost (step 3c):
        the session then ends with no answer, never on an error."""
        try:
            return self._complete(messages, specs, intro, think=think)
        except ClientError as exc:
            if exc.code not in LOST_REPLY_CODES:
                raise
            self._assistant_turns += 1
            self._lost(exc, self.steps + 1)
            return None

    def _end_if_refused(self, reply: proto.Reply) -> None:
        """Feature 020 (FR-020.17, W3-D8, TD-7): a refusal ends the session at once, by name: no
        nudge and no further request. The reply is already appended (its transcript line holds
        the neutral ``refusal``); the event records the category, null when none was given."""
        refusal = reply.refusal
        if refusal is None:
            return
        category = refusal.get("category")
        if self.writer is not None:
            self.writer.write({"event": "refusal", "turn": self.steps, "category": category})
        print(json.dumps({"turn": self.steps, "info": "wsp_editor_refused", "category": category}))
        raise ClientError(
            200, "editor_refused", f"the editor refused (category: {category or 'none given'})"
        )

    def _log(self, turn: int, decision: proto.Action | proto.Refusal | None) -> None:
        line = {
            "turn": turn,
            "action": decision.name if isinstance(decision, proto.Action) else None,
            "refused": decision.kind if isinstance(decision, proto.Refusal) else None,
            "tokens": self.tokens.total,
            "forward_left": self.forward.budget - self.forward.used,
        }
        print(json.dumps(line), flush=True)

    def _loop(self) -> None:
        specs = self.toolbox.specs
        structured = (
            proto.structured_tool_list(specs) if self.protocol == proto.STRUCTURED else None
        )
        thinking = self.request.get("thinking")
        system = prompts.system(
            thinking_off=thinking is False or thinking == "off",
            structured_tools=structured,
            paging=self.paging.pages_out,
            append_only=self.append_only,
        )
        # Fix 2026-11-21 (live round 5): the adapter's mode and task rules (a repair with a goal
        # IMPROVES, its goal after the task), and a program session's own task line.
        goal = self.request.get("goal")
        mode = prompts.mode_for(self.kind, goal)
        program_text = (
            prompts.program_mode(mode, ProgramTools.COST) if self.programs is not None else None
        )
        rows = [(n, list(v.shape), self.store.dtypes[n]) for n, v in self.store.values.items()]
        intro = prompts.intro(
            mode,
            prompts.task_text(self.kind, str(self.request.get("task") or ""), goal),
            judging.context_with_judging(
                context_text(self.bundle), self.request, self.bundle.manifest, on=self.judging
            ),
            prompts.inventory(rows, sum(v.size for v in self.store.values.values())),
            prompts.forward_lines(
                [(s.name, s.cost, s.description) for s in self.observation], self.forward.budget
            ),
            mode_text=program_text,
        )
        messages: list[dict[str, Any]] = []
        self._append(messages, {"role": "system", "content": system})
        self._append(messages, {"role": "user", "content": intro})
        nudges = 0
        while self.answer is None and self.steps < self.max_steps:
            self.stop = self._over_budget()
            if self.stop:
                break
            self.steps += 1
            turn_began = time.monotonic()
            ran_from: tuple[str, int] | None = None
            try:
                reply = self._complete(messages, specs, intro)
            except ClientError as exc:
                if exc.code not in LOST_REPLY_CODES:
                    raise
                # Step 3c: the reply was lost, not refused. Its slot is kept (a replay's script
                # has one), and the editor is told once; a second in a row ends the session.
                self._assistant_turns += 1
                if not self._lost(exc, self.steps):
                    break
                self._append(messages, {"role": "user", "content": prompts.LOST_REPLY})
                continue
            self._lost_in_a_row = 0
            self._assistant_turns += 1
            self._append(messages, reply.message)
            self._end_if_refused(reply)
            decision = proto.one_action(reply, self.protocol, specs)
            repeat = self._repeat_of(decision, specs)
            if repeat is not None:
                key, first = repeat
                self.repeated_turns += 1
                self._repeats_in_a_row += 1
                output = self._repeated(first)
                if self.writer is not None:
                    self.writer.write(
                        {
                            "event": "repeated",
                            "tool": key[0],
                            "first_turn": first,
                            "n": self.repeated_turns,
                        }
                    )
            elif isinstance(decision, proto.Action):
                late = self._forward_fits(decision.name, specs)
                if late is not None:
                    output = late
                else:
                    ran = self._ran_key(decision, specs)
                    action_began = time.monotonic()
                    weights_before = (
                        self.store.sha256()
                        if self.paging.keeps_pages or self.edit_repeats
                        else None
                    )
                    if self.paging.keeps_pages and weights_before is not None:
                        ran_from = (weights_before, self.forward.used)
                    charged = self.forward.used
                    output = self.toolbox.call(decision.name, decision.arguments)
                    if self.edit_repeats and weights_before is not None:
                        output += self._edit_repeated(
                            decision.name, weights_before, output, self.forward.used > charged
                        )
                    if (
                        self.append_only
                        and decision.name != paging.RECALL
                        and len(self.toolbox.last_full) > tools_mod.MAX_TOOL_CHARS
                    ):
                        # Feature 020 (FR-020.8): the page this result is kept under, so the
                        # editor can read on; part of the text at its first send, never after.
                        output += prompts.RECALL_HINT.format(
                            page=paging.page_id(self.steps), offset=tools_mod.MAX_TOOL_CHARS
                        )
                    if self.feedback:
                        nudges = 0  # an action ran: the no-action replies are not in a row
                    if self.forward.used > charged:
                        self._repeats_in_a_row = 0  # a new measurement ran
                    if ran is not None:
                        self._ran.setdefault(ran, self.steps)
                    if specs[decision.name].cost is not None:
                        took = time.monotonic() - action_began
                        self._slowest_forward = max(self._slowest_forward or 0.0, took)
            else:
                self.refused_turns += 1
                if self.writer is not None:
                    event = {"event": "refused", "kind": decision.kind, "n": decision.n}
                    if decision.cut_off:
                        event["cut_off"] = True  # only when true: older records are unchanged
                    self.writer.write(event)
                if decision.kind == "no_action":
                    if nudges >= 2:
                        self.answer = (reply.text or "", None)
                        self.stop = "answered"
                        self._log(self.steps, decision)
                        break
                    nudges += 1
                output = decision.text()
                if decision.cut_off and self.cut_off_cap:
                    output += self._lower_ceiling()
            appended = self._answer_turn(messages, reply, decision, output)
            if ran_from is not None and isinstance(decision, proto.Action):
                self._after_action(appended, decision, output, *ran_from)
            self._log(self.steps, decision)
            self._longest_turn = max(self._longest_turn, time.monotonic() - turn_began)
            # Not when the loop stops anyway: the final turn's own text says to submit.
            going_on = self.answer is None and self.steps < self.max_steps
            notice = self._nearly_up() if going_on and self._over_budget() is None else None
            if notice is not None:
                self._append(messages, {"role": "user", "content": notice})
            if self._repeats_out() and self.answer is None:
                break
        repeats_out = self._repeats_out() and self.stop is None
        if self.answer is None:
            if repeats_out:
                why = (
                    prompts.REPEAT_LIMIT_IN_A_ROW.format(n=self._repeats_in_a_row)
                    if self.feedback
                    else prompts.REPEAT_LIMIT.format(n=self.repeated_turns)
                )
            elif self._lost_in_a_row >= MAX_LOST_IN_A_ROW:
                why = prompts.LOST_TWICE
            else:
                self.stop = self.stop or "max_steps"
                why = (
                    prompts.STEP_LIMIT
                    if self.stop == "max_steps"
                    else prompts.RUN_LIMIT.format(reason=self.stop.replace("_", " "))
                )
            limit = prompts.SUBMIT_NOW_EDIT if self.can_edit else prompts.SUBMIT_NOW
            self._append(messages, {"role": "user", "content": limit.format(why=why)})
            only = {"submit": specs["submit"]}
            self._final_turn = True
            # Feature 020 (FR-020.11): an append-only session keeps its tool list (a shrunken one
            # would edit the prefix), and only ``submit`` is accepted from the reply.
            offered = specs if self.append_only else only
            final = self._final_complete(messages, offered, intro)
            if final is None:
                self.answer = ("(no answer)", None)
                self.stop = self.stop or "answered"
                return
            reply = final
            self._assistant_turns += 1
            self._append(messages, reply.message)
            self._end_if_refused(reply)
            decision = proto.one_action(reply, self.protocol, only)
            if (
                self.append_only
                and isinstance(decision, proto.Refusal)
                and decision.kind == "not_offered"
            ):
                # Another tool on the final turn: not run, answered FINAL_SUBMIT_ONLY, and the
                # one retry an unreadable submit gets, with thinking as it was (FTDD §0 item 2).
                decision = replace(decision, final=True)
                self.refused_turns += 1
                if self.writer is not None:
                    self.writer.write({"event": "submit_retry", "kind": decision.kind})
                self._answer_turn(messages, reply, decision, decision.text())
                self._log(self.steps + 1, decision)
                retry = self._final_complete(messages, offered, intro)
                if retry is None:
                    self.answer = ("(no answer)", None)
                    self.stop = self.stop or "answered"
                    return
                reply = retry
                self._assistant_turns += 1
                self._append(messages, reply.message)
                self._end_if_refused(reply)
                decision = proto.one_action(reply, self.protocol, only)
            elif isinstance(decision, proto.Refusal) and decision.kind == "unparseable":
                # Fix 2026-11-21 (run wsp_855a74f820ff): a submit that cannot be read, most often
                # cut off mid-JSON, is asked for ONCE more, with thinking off so the whole answer
                # is the action. A replay recomputes the same decision from the transcript.
                self.refused_turns += 1
                if self.writer is not None:
                    self.writer.write({"event": "submit_retry", "kind": decision.kind})
                self._answer_turn(
                    messages, reply, decision, prompts.SUBMIT_CUT.format(detail=decision.detail)
                )
                self._log(self.steps + 1, decision)
                retry = self._final_complete(messages, offered, intro, think=False)
                if retry is None:
                    self.answer = ("(no answer)", None)
                    self.stop = self.stop or "answered"
                    return
                reply = retry
                self._assistant_turns += 1
                self._append(messages, reply.message)
                self._end_if_refused(reply)
                decision = proto.one_action(reply, self.protocol, only)
            if isinstance(decision, proto.Refusal) and decision.kind == "not_offered":
                decision = replace(decision, final=self.append_only)
            if isinstance(decision, proto.Action):
                output = self.toolbox.call("submit", decision.arguments)
            else:
                self.refused_turns += 1
                output = decision.text()
            self._answer_turn(messages, reply, decision, output)
            self._log(self.steps + 1, decision)
            if self.answer is None:
                self.answer = (reply.text or "(no answer)", None)
                if repeats_out:
                    self.stop = "answered"
        self.stop = self.stop or "submitted"

    # ----------------------------------------------------------------------------- outputs

    def _write_outputs(self) -> None:
        if self.programs is not None:
            if self.programs.submitted is not None:
                (self.out / program.PROGRAM_NAME).write_text(self.programs.submitted, "utf-8")
                done = self.programs.candidate()
                if done is not None:
                    self._submitted = dict(done.tensors)
                if done is not None and done.manifest is not None and done.data is not None:
                    patch_writer.write_patch(self.out / "patch", done.manifest, done.data)
            return
        # Every kind, inspect included: an inspect that changed weights yields a patch, and the
        # intake refuses it (FR-016.32) - though here R is read-only and inspect has no W.
        # F18-D16 (operator, 2026-11-24): an edit that breaks a limit is written nowhere. Only a
        # session with no accepted submit can reach here with one (``_submit`` refuses a broken
        # edit). It used to be written as a partial patch for the intake to refuse
        # (wsp_change_limit_exceeded), failing the run; now the run ends with no patch and the
        # transcript says why. A partial edit within the limits is still written (the 2026-11-21
        # ruling: a valid patch from a run that stopped early is a candidate).
        broken = self.store.violations()
        if broken:
            self._submitted = {}
            if self.writer is not None:
                self.writer.write(
                    {
                        "event": "patch_withheld",
                        "reason": "no_accepted_submit",
                        "violations": broken,
                    }
                )
            return
        parent = self.request.get("parent") or {}
        bundle_sha = self.bundle.manifest["bundle_sha256"]
        self._submitted = self.store.edited()
        built = patch_writer.build_patch(
            self.bundle.tensors,
            self._submitted,
            bundle_base_sha256=bundle_sha,
            parent={"kind": "bundle", "ref": parent.get("ref", bundle_sha), "patch_sha256": None},
            parent_patch_sha256=None,
            created_by={**provider(), "run_id": self.run_id},
            created_at=now(),
        )
        if built is not None:
            patch_writer.write_patch(self.out / "patch", built[0], built[1])

    def _measure_submitted(self) -> None:
        """Plan A2: the submitted weights' own figures, after every output is safely written."""
        if self.writer is None or self.stop == "interrupted":
            return
        if not submitted_measure.applies(self.request, self.bundle.manifest):
            return
        slowest = self._slowest_forward
        line = submitted_measure.event(
            lambda: observation_context(self.bundle, self.store),
            {n: t.values() for n, t in self._submitted.items()},
            left_s=self._time_left(),
            estimate_s=self._forward_prior_s if slowest is None else 2 * slowest,
            claims=self.figures,
            claimed=self.claimed,
        )
        self.writer.write(line)

    def result(self) -> dict[str, Any]:
        answer, confidence = self.answer or ("", None)
        changed = self.store.changes()
        return {
            "mode": self.style,
            "run_id": self.run_id,
            **provider(),
            "answer": answer,
            "confidence": confidence,
            "stop_reason": self.stop,
            "failure": self.failure,
            "steps": self.steps,
            "seconds": round(time.monotonic() - self._t0, 3),
            "forward_used": self.forward.used,
            "forward_budget": self.forward.budget,
            "forward_calls": self.forward.calls,
            "tokens": {
                "input_tokens": self.tokens.input_tokens,
                "output_tokens": self.tokens.output_tokens,
                "cached_input_tokens": self.tokens.cache_read,
                "cache_write_tokens": self.tokens.cache_write,
                "reasoning_tokens": self.tokens.reasoning_tokens,
                "requests": self.requests,
                "total_tokens": self.tokens.total,
            },
            "cost_usd": round(self.cost_usd, 6) if self.prices is not None else None,
            "changed_params": sum(changed.values()),
            "total_params": int(sum(v.size for v in self.store.values.values())),
            "replay_divergence": self.replay_divergence,
            # Step 8: whether this session paged, and why not when it was asked to.
            "paging": "on" if self.paging.on else "off",
            "paging_off_reason": self.paging.off_reason,
            "engine": {
                "action_protocol": self.protocol,
                "handoffs": self.handoffs,
                "refused_turns": self.refused_turns,
                "kernel_restarts": 0 if self.kernel is None else self.kernel.restarts,
                "reasoning_turns": self.reasoning_turns,
                "repeated_turns": self.repeated_turns,
                # Step 8 (2026-11-21): paging's counters, and the largest prompt estimated.
                "pages_out": self.pager.pages_out,
                "recalls": self.pager.recalls,
                "ledger_rows": len(self.ledger.rows),
                "peak_prompt_tokens": self.peak_prompt_tokens,
                # Step 5d: present only when a cut-off reply lowered the ceiling.
                **(
                    {}
                    if self.lowered_cap is None
                    else {"output_ceiling_lowered_to": self.lowered_cap}
                ),
            },
        }

    def header(self) -> dict[str, Any]:
        """The transcript's first line (after ``miwsp_transcript``)."""
        return {
            **provider(),
            "protocol": self.protocol,
            "kind": self.kind,
            "style": self.style,
            "thinking": self.thinking,
            "thinking_tokens": self.thinking_tokens,
            "output_cap": "on" if self.capped else "off",
            # Step 8: the paging configuration (null: not asked for; "off": switched off by the
            # engine, the reason beside it), which a replay pages by.
            "paging": self.paging.header(),
            "paging_off_reason": self.paging.off_reason,
            # Step 5a (2026-11-22): present only when on, which a replay judges by.
            **({"judging": judging.VERSION} if self.judging else {}),
            # Step 3b (2026-11-22): present only when on, which a replay follows.
            **({"feedback": FEEDBACK_VERSION} if self.feedback else {}),
            # FU-2 (2026-11-25): present only when submit offered the claimed figures, which a
            # replay offers by.
            **({"figures": submitted_measure.FIGURES_VERSION} if self.figures else {}),
            # Step 5b (2026-11-26): present only when on, which a replay follows.
            **({"write_noise": WRITE_NOISE_VERSION} if self.write_noise else {}),
            # 2026-11-27: present only when on, which a replay follows.
            **({"reserved_names": RESERVED_NAMES_VERSION} if self.reserved_names else {}),
            **({"submit_changes": SUBMIT_CHANGES_VERSION} if self.submit_changes else {}),
            # 2026-11-28 (step 5c): present only when on, which a replay follows.
            **({"edit_repeats": EDIT_REPEATS_VERSION} if self.edit_repeats else {}),
            **({"cut_off_cap": CUT_OFF_CAP_VERSION} if self.cut_off_cap else {}),
            # 2026-11-28 (E7 §5.3 gap 2): present only in a program session, which a replay follows.
            **({"program_errors": PROGRAM_ERRORS_VERSION} if self.program_errors else {}),
            # Feature 020 (FR-020.6, TD-5): present only when append-only, which a replay reads;
            # every older transcript reads as the paging mode.
            **({"history_mode": paging.APPEND_ONLY} if self.append_only else {}),
        }

    def window_event(self) -> dict[str, Any] | None:
        """Feature 019 (FR-019.2, DEP-3): the window and the ceiling this session runs with, as
        the backend resolved them at push (``spec.editor``), for the transcript's second line.
        ``None`` in a replay (it records nothing new) and for a push with no editor (a rerun)."""
        if self.replaying or not self.editor:
            return None
        return {
            "event": "window",
            "tokens": self.window,
            "source": self.editor.get("window_source"),
            "reported": self.editor.get("served_context_tokens"),
            "ceiling": self.ceiling,
            "ceiling_lowered": self.editor.get("output_ceiling_lowered") is True,
            "warnings": list(self.editor.get("window_warnings") or []),
        }

    def run(self) -> int:
        self.writer = TranscriptWriter(self.out / "transcript.jsonl", self.header())
        window = self.window_event()
        if window is not None:
            self.writer.write(window)
        try:
            if not self.replaying:
                proto.check_supported(
                    self.protocol,
                    str(self.editor.get("kind", "")),
                    str(self.editor.get("serving", "")),
                )
                self._set_thinking()
            self._isolate_and_spawn()
            self._loop()
        except ClientError as exc:
            self.stop, self.failure = "error", {"code": exc.code, "message": exc.message[:2000]}
        except (
            KernelError,
            proto.ProtocolUnsupportedError,
            proto.ThinkingUnsupportedError,
            context.ThinkingDoesNotFitError,
        ) as exc:
            code = getattr(exc, "code", "worker_session_failed")
            self.stop, self.failure = "error", {"code": str(code), "message": str(exc)[:2000]}
        except KeyboardInterrupt:
            self.stop = "interrupted"
        except Exception as exc:  # recorded, never re-raised: the outputs below still get written
            self.stop = "error"
            self.failure = {
                "code": "worker_session_failed",
                "message": f"{type(exc).__name__}: {str(exc)[:1000]}",
            }
        finally:
            if self.kernel is not None:
                self.kernel.close()
            close = getattr(self._client, "close", None)
            if callable(close):
                close()
            try:
                self._write_outputs()
            except Exception as exc:  # a patch that cannot be built is a failure, not a crash
                self.stop = "error"
                self.failure = {"code": "worker_patch_failed", "message": str(exc)[:2000]}
            (self.out / "result.json").write_text(
                json.dumps(self.result(), default=str), encoding="utf-8"
            )
            self._measure_submitted()
            self.writer.close()
        return 0


def program_tools_text() -> tuple[str, str, dict[str, Any]]:
    from wsp_worker import program_tools

    return (
        program_tools.TEST_DESCRIPTION,
        program_tools.SUBMIT_DESCRIPTION,
        program_tools.CODE_PARAMETERS,
    )


def run(line: Mapping[str, Any], *, client: ModelClient | None = None, **options: Any) -> int:
    """One engine session (or its replay) from the child line; writes ``/out``; the exit code."""
    return Session(line, client=client, **options).run()
