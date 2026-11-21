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
   transcript's ``{event: repeated}`` line too). After :data:`MAX_REPEATS` the session moves to
   the submit-only turn. A replay recomputes the same decisions from the same transcript.
3. **Each turn:** budgets first (90% of tokens, cost and time, as the ``weightspace`` adapter);
   the context guard (a handoff, never an overflow); one request; exactly one action
   (``protocol.one_action``); a refused reply runs nothing and costs the turn. A reply with no
   action gets ``weightspace``'s nudge; after two, the next one ends the run ``answered``.
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
from pathlib import Path
from typing import Any

import numpy as np

from wsp_worker import bundle as bundle_mod
from wsp_worker import observation_tools, patch_writer, program
from wsp_worker.bundle import Bundle
from wsp_worker.miwsp_engine import context, paging, prompts
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine import tools as tools_mod
from wsp_worker.miwsp_engine.client import (
    ClientError,
    GatewayClient,
    ModelClient,
)
from wsp_worker.miwsp_engine.kernel import EXEC_TIMEOUT_S, Kernel, KernelError
from wsp_worker.miwsp_engine.store import Refused, TensorStore
from wsp_worker.miwsp_engine.transcript import TranscriptWriter, read_header, read_script
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
#: Tools whose answer depends only on their arguments and the weights, besides every tool with a
#: forward-pass cost: a repeat of one on the same weights would answer the same. ``python`` is
#: not one (the kernel keeps state between runs), nor any tool that edits or submits.
PURE_TOOLS = frozenset({"list_tensors", "describe_tensor", "read_tensor", "diff"})
#: A paged handoff's summary keeps the full ledger while it takes at most this share of the room
#: the bare restart leaves before the next handoff; else the compact one (the newest rows), else
#: the page ids alone (step 8), so a small window is not crowded by its own summary.
HANDOFF_LEDGER_SHARE = 0.25
PR_SET_DUMPABLE = 4
PRACTICE_FILE_ROLES = ("practice_activations", *bundle_mod.PRACTICE_ROLES)


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
    the same context the ``weightspace`` adapter builds (``observation_context``)."""
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
        target_fpr=float((bundle.manifest.get("operating_point") or {}).get("target_fpr") or 0.01),
        shipped_bar=observation_tools.shipped_bar_of(bundle.manifest),
    )
    return observation_tools.Context(
        view, current, fixed, program.seed_for(bundle.manifest["bundle_sha256"])
    )


def context_text(bundle: Bundle) -> str:
    """The function's description and the tensors' roles. NEVER the practice data."""
    lines = [bundle.manifest["function"]["description"], "", "Tensors:"]
    for t in bundle.manifest["tensors"]:
        role = "editable" if t["editable"] else "read-only (not given to you)"
        lines.append(f"- {t['name']} {t['dtype']} {t['shape']}: {role}")
    return "\n".join(lines)


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
        self.python_on = self.request.get("python", True) is not False
        self.store = TensorStore(
            self.bundle,
            self.request["change_limits"],
            zero_start=self.kind == "create" and self.style != "program",
        )
        observation = observation_tools.build(
            list(self.request.get("observation_tools") or []),
            observation_context(self.bundle, self.store),
        )
        self.programs = (
            ProgramTools(self.bundle, self.request, self.run_id, created_by=provider())
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
        self.window = window
        # The output cap (2026-11-21): off names none. The guard then reserves the largest output
        # seen so far (at least the gateway's floor) instead of a cap nothing sends.
        self.capped = context.capped(self.request)
        self.cap = context.editor_output_cap(self.editor) if self.capped else 0
        self._largest_output = 0
        # Thinking (step 7): the frozen value; the allowance is set in ``run`` (it can refuse).
        self.thinking = self.request.get("thinking")
        self.thinking_tokens = self.request.get("thinking_tokens")
        self.allowance = 0
        self.guard = context.Guard(window, self._reserve())
        # Paging and the ledger (step 8): the frozen request's ``paging``, or, in a replay, the
        # configuration the original recorded in its transcript header (none: off, as it ran).
        if self.replaying:
            header = read_header(self.transcript_in)
            self.paging = paging.Config.from_header(
                header.get("paging"), window, header.get("paging_off_reason")
            )
        else:
            self.paging = paging.Config.for_request(self.request, window)
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
        self._ran: dict[tuple[str, str, str], int] = {}
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
        if self.paging.on:
            specs.append(tools_mod.recall_spec(self.pager.recall))
        specs.append(tools_mod.submit_spec(self._submit))
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
                self._client = GatewayClient(
                    str(self.editor["kind"]), str(self.editor.get("model", "")), gateway
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

    def _read_tensor(self, name: str, index: str | None = None, precision: int = 4) -> str:
        arr = np.asarray(self.store.values[name][tools_mod.parse_index(index)])
        if arr.size > tools_mod.READ_LIMIT:
            return (
                f"slice has {arr.size:,} values (shape {arr.shape}); limit is "
                f"{tools_mod.READ_LIMIT}. Read a smaller slice."
            )
        return f"{name}[{index or ':'}] shape={arr.shape}\n{tools_mod.fmt_array(arr, precision)}"

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
        applied = self.store.try_apply(updates)
        # Accepted: the files take the new values; refused: the same files are re-mapped, so W
        # reverts to the last accepted weights. Either way R and W agree again.
        self._sync(sorted(updates))
        if isinstance(applied, Refused):
            return f"{result.output}\n{applied.text} W was reset to the last accepted weights."
        return result.output

    def _write_tensor(self, name: str, values: Any, index: str | None = None) -> str:
        applied = self.store.write(name, values, tools_mod.parse_index(index))
        if isinstance(applied, Refused):
            return applied.text
        self._sync([name])
        return f"wrote {name}[{index or ':'}]\n{self.store.diff()}"

    def _checkpoint(self, label: str) -> str:
        self.store.checkpoint(label)
        return f"saved checkpoint {label!r}"

    def _restore(self, label: str) -> str:
        applied = self.store.restore(label)
        if isinstance(applied, Refused):
            return applied.text
        self._sync(self.store.names())
        return f"restored {label!r}\n{self.store.diff()}"

    def _revert(self, name: str | None = None) -> str:
        applied = self.store.revert(name)
        if isinstance(applied, Refused):
            return applied.text
        self._sync(self.store.names())
        return f"reverted {name or 'all tensors'}\n{self.store.diff()}"

    def _submit(self, answer: str, confidence: float | None = None) -> str:
        broken = self.store.violations()
        if broken:
            return prompts.SUBMIT_REFUSED.format(details="; ".join(broken))
        self.answer = (str(answer), None if confidence is None else float(confidence))
        return "submitted"

    def _max_tokens(self, think: bool = True) -> int | None:
        """The request's cap: the action's cap plus the reasoning allowance (none for a turn sent
        with thinking off); ``None`` when the output cap is off (the gateway's window-aware
        default applies)."""
        if not self.capped:
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
        if self.thinking != proto.THINKING_ON or self.capped:
            return self.thinking, self.allowance, None
        prompt = self.guard.gateway_prompt(messages, tools_bytes=extra)
        cap = context.gateway_cap(self.window, prompt)
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
        observed = max(self._largest_output, context.MIN_OUTPUT_TOKENS)
        return observed + self.allowance

    def _set_thinking(self) -> None:
        """Step 7: refuse ``on`` for an endpoint that cannot carry it, bound the allowance by the
        window, and size the guard for the whole answer (reasoning plus action)."""
        kind, serving = str(self.editor.get("kind", "")), str(self.editor.get("serving", ""))
        proto.check_thinking_supported(self.thinking, kind, serving)
        if self.thinking == proto.THINKING_ON:
            self.allowance = context.thinking_allowance(self.window, self.cap, self.thinking_tokens)
            self.guard = context.Guard(self.window, self._reserve())
        if self.writer is not None:
            self.writer.write(
                {
                    "event": "thinking",
                    "thinking": self.thinking,
                    "sent": proto.thinking_fields(
                        self.thinking, kind, serving, self.allowance, self.protocol
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
        if self.paging.on:
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
        if self.writer is not None:
            self.writer.write({"event": "handoff", "n": self.handoffs, "messages": restarted})

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
                self.protocol, kind, serving, specs, thinking=thinking, allowance=allowance
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
        began = time.monotonic()
        reply = self.client().complete(
            outgoing, fields=fields, max_tokens=self._max_tokens(think and skipped is None)
        )
        self._longest_request = max(self._longest_request, time.monotonic() - began)
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

    # ----------------------------------------------------------------------------- paging

    def _page_if_due(self, messages: list[dict[str, Any]]) -> None:
        """Step 8's batched page event, before a request: a function of the turn and the
        history's bytes only, so a replay pages the same messages at the same turns. The event
        stubs the results past the keep window and moves the ledger to the history's end (the
        only times the history is rewritten: ``paging``'s module doc)."""
        if not self.paging.on:
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

    def _after_action(
        self,
        appended: list[dict[str, Any]],
        action: proto.Action,
        output: str,
        before: str,
        forward_used: int,
    ) -> None:
        """Step 8, after an action RAN: its ledger row, and its result offered to the pager."""
        if not self.paging.on:
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
        if holder is not None:
            self.pager.offer(
                self.steps,
                action.name,
                action.arguments,
                output,
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
            paging=self.paging.on,
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
            context_text(self.bundle),
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
            reply = self._complete(messages, specs, intro)
            self._assistant_turns += 1
            self._append(messages, reply.message)
            decision = proto.one_action(reply, self.protocol, specs)
            repeat = self._repeat_of(decision, specs)
            if repeat is not None:
                key, first = repeat
                self.repeated_turns += 1
                output = prompts.REPEATED.format(turn=first)
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
                    if self.paging.on:
                        ran_from = (self.store.sha256(), self.forward.used)
                    output = self.toolbox.call(decision.name, decision.arguments)
                    if ran is not None:
                        self._ran.setdefault(ran, self.steps)
                    if specs[decision.name].cost is not None:
                        took = time.monotonic() - action_began
                        self._slowest_forward = max(self._slowest_forward or 0.0, took)
            else:
                self.refused_turns += 1
                if self.writer is not None:
                    self.writer.write({"event": "refused", "kind": decision.kind, "n": decision.n})
                if decision.kind == "no_action":
                    if nudges >= 2:
                        self.answer = (reply.text or "", None)
                        self.stop = "answered"
                        self._log(self.steps, decision)
                        break
                    nudges += 1
                output = decision.text()
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
            if self.repeated_turns >= MAX_REPEATS and self.answer is None:
                break
        repeats_out = self.repeated_turns >= MAX_REPEATS and self.stop is None
        if self.answer is None:
            if repeats_out:
                why = prompts.REPEAT_LIMIT.format(n=self.repeated_turns)
            else:
                self.stop = self.stop or "max_steps"
                why = (
                    prompts.STEP_LIMIT
                    if self.stop == "max_steps"
                    else prompts.RUN_LIMIT.format(reason=self.stop.replace("_", " "))
                )
            self._append(messages, {"role": "user", "content": prompts.SUBMIT_NOW.format(why=why)})
            only = {"submit": specs["submit"]}
            reply = self._complete(messages, only, intro)
            self._assistant_turns += 1
            self._append(messages, reply.message)
            decision = proto.one_action(reply, self.protocol, only)
            if isinstance(decision, proto.Refusal) and decision.kind == "unparseable":
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
                reply = self._complete(messages, only, intro, think=False)
                self._assistant_turns += 1
                self._append(messages, reply.message)
                decision = proto.one_action(reply, self.protocol, only)
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
                if done is not None and done.manifest is not None and done.data is not None:
                    patch_writer.write_patch(self.out / "patch", done.manifest, done.data)
            return
        # Every kind, inspect included: an inspect that changed weights yields a patch, and the
        # intake refuses it (FR-016.32) - though here R is read-only and inspect has no W.
        parent = self.request.get("parent") or {}
        bundle_sha = self.bundle.manifest["bundle_sha256"]
        built = patch_writer.build_patch(
            self.bundle.tensors,
            self.store.edited(),
            bundle_base_sha256=bundle_sha,
            parent={"kind": "bundle", "ref": parent.get("ref", bundle_sha), "patch_sha256": None},
            parent_patch_sha256=None,
            created_by={**provider(), "run_id": self.run_id},
            created_at=now(),
        )
        if built is not None:
            patch_writer.write_patch(self.out / "patch", built[0], built[1])

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
        }

    def run(self) -> int:
        self.writer = TranscriptWriter(self.out / "transcript.jsonl", self.header())
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
