"""Context paging and the experiment ledger (feature 018; FR-018.22 to FR-018.25; operator
decision 2026-11-21; ``0xcc/reviews/018_paging_2026-11-21.md``).

**Why.** The D128 runs (bundle ``wsb_6a53417a3a39``, a probe with weight [4096] and a bias) showed
one ``read_tensor`` of the weight taking the prompt from about 3k to about 16k tokens, re-sent on
every later turn, so prompts reached about 19k tokens in 12-20 turns and input was about 95% of
the tokens used. The context handoff (``context.HANDOFF_SHARE``) then restarted the conversation
with only the kernel's ``notes``, which editors rarely write, and every measurement was lost.

**The ledger** (:class:`Ledger`): one compact row per measurement or edit, derived by the ENGINE
from tool results (never written by the model), so a replay rebuilds it bit for bit. A row holds
the turn, the action and its key arguments, the store digest before and after (8 hex), what is
changed against the start (the store's diff: tensors, count, |delta|), and the key numbers a
measurement returned (``auroc``, ``recall``, ``false_alarm_rate``, ``bar``, any ``at_parent_bar``
block, strata rates, cosines, flipped rows), plus checkpoint, restore and revert events. It is
rendered (:meth:`Ledger.render`) as a small table, bounded: the newest :data:`LEDGER_ROWS` rows in
full, older rows merged into one line per weight state (digest) and one per edited tensor, at most
:data:`LEDGER_SUMMARY_LINES` of each.

**Where the ledger sits (the prefix cache and the gateway's meter).** llama.cpp reuses the longest
common token prefix of consecutive requests, and the gateway's ``PromptMeter`` counts a call as
"the conversation grew" (the last count plus the new bytes at a token each) only when the earlier
call's messages are unchanged. So the history only ever grows at its end between page events: the
ledger is written INTO the history, appended to its last message (:func:`attach`), only at a page
event, and the previous one is taken off (:func:`detach`) at the same moment. A page event
rewrites the history anyway (its stubs), so the ledger's refresh costs no extra recompute. Rejected:
a ledger in the system message (it would change position 0 every refresh and recompute the whole
prompt) and a transient tail sent with every request (it changes the previous call's last
message, so the gateway's meter never sees a conversation that grew, and sizes an uncapped reply
from a whole-body ratio that under-counts a fresh tensor dump). Between page events the newest
results are verbatim in the history: nothing is paged before its row is in a ledger.

**Paging** (:class:`Pager`). A tool result larger than :data:`PAGE_THRESHOLD_BYTES` (or, when the
ledger holds its numbers, :data:`LEDGERED_THRESHOLD_BYTES`) stays verbatim for the next
:data:`KEEP_TURNS` requests; then it is replaced IN THE HISTORY by a deterministic stub
(:func:`stub`) naming its page id. The full text stays in the session (and was written to the
transcript once, when it was appended); ``recall(page)`` returns it. Stubbing rewrites the
history from the first stubbed message on, which invalidates the cache from there, so paging is
BATCHED (:meth:`Pager.due`): at most once every :data:`PAGE_EVERY_TURNS` turns, or sooner when the
history's byte estimate passes :data:`PAGE_SHARE` of the window. Between page events every
request's history is a prefix of the next one's.

**Never paged:** the system prompt and the intro (task), the latest :data:`KEEP_TURNS` turns, any
refusal or engine notice (budgets, the repeat guard, the step limit) and the submit-only turn:
only the output of an action that RAN is ever offered (:meth:`Pager.offer`).

**Determinism.** Every decision is a function of the turn number, the bytes of the history and the
configuration (:class:`Config`, recorded in the transcript header), never of the endpoint's
reported usage, so a replay, which has none, makes the same decisions.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: A tool result above this many UTF-8 bytes is paged out after :data:`KEEP_TURNS` turns. 1,500
#: bytes is about 500 tokens (``context.BYTES_PER_TOKEN``): every observation tool's ordinary answer
#: (250 to 650 bytes in the D128 transcripts) stays, while a full ``read_tensor`` (12,052 bytes),
#: ``forward_firing_positions`` at ``top_k`` 50 (6,653 bytes) and long Python output page.
PAGE_THRESHOLD_BYTES = 1_500
#: A result whose numbers the ledger already holds (a measurement row) pages above this smaller
#: size: after :data:`KEEP_TURNS` turns its verbatim text duplicates the ledger row, and 30
#: measurements of 250 to 650 bytes each are otherwise the largest part of a long session's
#: prompt after the fixed system prompt and tool list. The stub is about 90 bytes, so a result of
#: 160 bytes or less would save too little to be worth a stub (measured on the scripted session
#: of ``test_paging.py``: 256 bytes gave a peak of 9,718 estimated tokens, 160 gave 8,675).
LEDGERED_THRESHOLD_BYTES = 160
#: A result stays verbatim in the next KEEP_TURNS requests (in the D128 transcripts an editor acted
#: on a read within one to three turns).
KEEP_TURNS = 3
#: At most one page event per this many turns, so the history (the cached prefix) is rewritten
#: rarely: the cost of a page event is recomputing the history after its first stub once.
PAGE_EVERY_TURNS = 4
#: ... or sooner, when the history's byte estimate passes this share of the window (well below the
#: handoff's 85%, so paging, not a handoff, is what keeps a long session inside its window).
PAGE_SHARE = 0.5
#: Paging switches itself off for an editor window under this many tokens (coordinator's
#: decision, 2026-11-21): at the default 8,192 the scripted session handed off 12 times with
#: paging and 9 without (a handoff every 4 or 5 turns, so the keep window never elapsed, while the
#: recall tool and its rule took headroom). Recorded as ``paging_off_reason: "window_small"``.
MIN_WINDOW_TOKENS = 16_384
WINDOW_SMALL = "window_small"
#: The newest rows shown in full. A row is about 110 bytes (40 tokens), so 8 rows are about 320
#: tokens; the D128 sessions made about 8 measurements and edits in 20 turns.
LEDGER_ROWS = 8
#: The rows a handoff keeps when the full ledger would not fit the window (the newest).
COMPACT_ROWS = 3
#: At most this many summary lines for older rows of each kind (weight states, edited tensors).
LEDGER_SUMMARY_LINES = 4
#: The short digest of the store's sha256 shown in a row.
DIGEST_CHARS = 8
#: The recall tool's name.
RECALL = "recall"
#: The metric keys a measurement row keeps, in this order, with their short column names.
METRICS = (("auroc", "auroc"), ("recall", "recall"), ("false_alarm_rate", "far"), ("bar", "bar"))
#: Observation tools v2's ``at_parent_bar`` block, with the ledger's column names.
AT_PARENT_BAR = (
    ("bar", "parent_bar"),
    ("recall", "recall_at_parent_bar"),
    ("false_alarm_rate", "far_at_parent_bar"),
)
#: Tools whose row is an edit event even when the weights did not change.
EDIT_EVENTS = ("write_tensor", "checkpoint", "restore", "revert")
#: The ledger's heading; it marks the tail as the engine's, not a tool result.
LEDGER_HEADING = "## Ledger (kept by the engine from tool results; newest last)"
#: The tail's heading after a handoff: the rows before it ride in the restart summary.
LEDGER_SINCE_HEADING = (
    "## Ledger since the restart (kept by the engine; earlier rows are in the restart summary)"
)
LEDGER_SEPARATOR = "\n\n---\n"
#: The text a refused tool answer starts with (``store.Refused.text``, ``prompts.SUBMIT_REFUSED``).
REFUSED = "refused"
#: The forward budget's suffix (``prompts.FORWARD_LEFT``), cut before a result is parsed.
FORWARD_SUFFIX = "\n[forward budget:"


# ----------------------------------------------------------------------------- configuration


@dataclass(frozen=True)
class Config:
    """The paging configuration a session runs with, recorded in the transcript header."""

    on: bool
    window: int
    #: Why paging is off although the request asked for it (:data:`WINDOW_SMALL`), else None.
    off_reason: str | None = None
    threshold: int = PAGE_THRESHOLD_BYTES
    ledgered_threshold: int = LEDGERED_THRESHOLD_BYTES
    keep: int = KEEP_TURNS
    every: int = PAGE_EVERY_TURNS
    share: float = PAGE_SHARE
    rows: int = LEDGER_ROWS

    @classmethod
    def for_request(cls, request: Mapping[str, Any], window: int) -> Config:
        """A session's configuration: the frozen request's ``paging`` (:func:`requested`), off by
        itself for a window under :data:`MIN_WINDOW_TOKENS`, saying why."""
        if not requested(request):
            return cls(False, window)
        if window < MIN_WINDOW_TOKENS:
            return cls(False, window, WINDOW_SMALL)
        return cls(True, window)

    def header(self) -> dict[str, Any] | str | None:
        """The transcript header's ``paging`` value: the configuration when on; ``"off"`` when the
        engine switched it off itself (the header's ``paging_off_reason`` says why); ``None`` when
        the request did not ask for it."""
        if not self.on:
            return "off" if self.off_reason is not None else None
        return {
            "window": self.window,
            "threshold": self.threshold,
            "ledgered_threshold": self.ledgered_threshold,
            "keep": self.keep,
            "every": self.every,
            "share": self.share,
            "rows": self.rows,
        }

    @classmethod
    def from_header(cls, value: Any, window: int, off_reason: Any = None) -> Config:
        """A replay's configuration: the original's, from its transcript header; a transcript
        from before paging (no ``paging`` value) replays with paging off, as it ran, and one the
        engine switched off replays off with the same reason."""
        if not isinstance(value, Mapping):
            reason = str(off_reason) if isinstance(off_reason, str) else None
            return cls(False, window, reason)
        return cls(
            True,
            int(value.get("window", window)),
            None,
            int(value.get("threshold", PAGE_THRESHOLD_BYTES)),
            int(value.get("ledgered_threshold", LEDGERED_THRESHOLD_BYTES)),
            int(value.get("keep", KEEP_TURNS)),
            int(value.get("every", PAGE_EVERY_TURNS)),
            float(value.get("share", PAGE_SHARE)),
            int(value.get("rows", LEDGER_ROWS)),
        )


def requested(request: Mapping[str, Any]) -> bool:
    """The frozen request's ``paging``: frozen only as ``off`` (``wsp.freeze.paging_in_effect``);
    absent means on, the operator's default. A replay does not read it: it pages by the original
    transcript's header (:meth:`Config.from_header`), so a run from before paging replays
    unpaged."""
    return bool(request.get("paging", "on") != "off")


# ----------------------------------------------------------------------------- shared text


def short(digest: str) -> str:
    return digest[:DIGEST_CHARS]


def fmt(value: float) -> str:
    text = f"{value:.4g}"
    return "0" if text in ("0", "-0") else text


def _value_summary(key: str, value: Any) -> str:
    if key == "code" and isinstance(value, str):
        sha = hashlib.sha256(value.encode("utf-8")).hexdigest()[:DIGEST_CHARS]
        return f"code#{sha}({len(value)} chars)"
    if isinstance(value, list):
        return f"<list of {len(value)}>"
    if isinstance(value, dict):
        return f"<object of {len(value)}>"
    if isinstance(value, float):
        return fmt(value)
    text = json.dumps(value) if not isinstance(value, str) else value
    text = " ".join(text.split())
    return text if len(text) <= 40 else text[:37] + "..."


def args_summary(arguments: Mapping[str, Any]) -> str:
    """The action's arguments as ``key=value`` pairs, sorted, each value at most 40 characters;
    code is named by its sha256 and length, lists and objects by their size."""
    return " ".join(
        (
            _value_summary(k, arguments[k])
            if k == "code" and isinstance(arguments[k], str)
            else f"{k}={_value_summary(k, arguments[k])}"
        )
        for k in sorted(arguments)
    )


# ----------------------------------------------------------------------------- the ledger


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _metrics(source: Mapping[str, Any]) -> list[tuple[str, str]]:
    out = []
    for key, column in METRICS:
        number = _number(source.get(key))
        if number is not None:
            out.append((column, fmt(number)))
    return out


def numbers_of(output: str) -> list[tuple[str, str]]:
    """The key numbers of one tool answer, as ``(name, text)`` in a fixed order; empty when the
    answer is not a JSON object (a refusal, a read, Python output).

    - ``auroc``, ``recall``, ``far`` and ``bar`` from the answer (``compare_to_parent``: from
      ``now``; ``test_program``: its validation metrics);
    - ``parent_bar``, ``recall_at_parent_bar``, ``far_at_parent_bar``: observation tools v2's
      ``at_parent_bar`` block (the parent's bar held fixed), top-level or in ``now``; a v1 answer
      has none;
    - ``flipped`` rows; each stratum's false-alarm ``rate`` (``far[<stratum>]``); each tensor's
      ``cos_parent``; the top score of ``firing_positions``; a program's refusal code.
    """
    head = output.split(FORWARD_SUFFIX, 1)[0]
    try:
        parsed = json.loads(head)
    except ValueError:
        return []
    if not isinstance(parsed, dict):
        return []
    source = parsed["now"] if isinstance(parsed.get("now"), dict) else parsed
    out = _metrics(source)
    at_parent = parsed.get("at_parent_bar", source.get("at_parent_bar"))
    if isinstance(at_parent, dict):
        # Observation tools v2 (``practice_metrics``, ``compare_to_parent`` in ``now``): the
        # rates at the parent's bar held fixed, ``{bar, source, recall, false_alarm_rate}``.
        for key, column in AT_PARENT_BAR:
            number = _number(at_parent.get(key))
            if number is not None:
                out.append((column, fmt(number)))
    flipped = _number(parsed.get("flipped"))
    if flipped is not None:
        out.append(("flipped", fmt(flipped)))
    strata = parsed.get("strata")
    if isinstance(strata, dict):
        for name in sorted(strata):
            cell = strata[name]
            rate = _number(cell.get("rate")) if isinstance(cell, dict) else None
            if rate is not None:
                out.append((f"far[{name}]", fmt(rate)))
    cosines = parsed.get("cosines")
    if isinstance(cosines, dict):
        for name in sorted(cosines):
            cell = cosines[name]
            cos = _number(cell.get("cos_parent")) if isinstance(cell, dict) else None
            if cos is not None:
                out.append((f"cos[{name}]", fmt(cos)))
    top = parsed.get("top")
    if isinstance(top, list) and top and isinstance(top[0], dict):
        score = _number(top[0].get("score"))
        if score is not None:
            out.append(("top_score", fmt(score)))
    if isinstance(parsed.get("refused"), str):
        out.append(("refused", str(parsed["refused"])[:40]))
    return out


@dataclass(frozen=True)
class Row:
    """One ledger row."""

    turn: int
    action: str
    before: str
    after: str
    #: What is changed against the start after the action (``store.diff_summary``), or a word.
    change: str
    numbers: tuple[tuple[str, str], ...] = ()
    #: ``measurement``, ``edit``, ``refused``, ``checkpoint``, ``restore``, ``revert``,
    #: ``program``.
    kind: str = "measurement"
    tensors: tuple[str, ...] = ()

    def document(self) -> dict[str, Any]:
        """The row as the transcript's ``{"event": "ledger"}`` line records it."""
        return {
            "turn": self.turn,
            "kind": self.kind,
            "action": self.action,
            "before": self.before,
            "after": self.after,
            "change": self.change,
            "numbers": dict(self.numbers),
        }

    def text(self) -> str:
        weights = self.after if self.before == self.after else f"{self.before}>{self.after}"
        result = " ".join(f"{k}={v}" for k, v in self.numbers) or "-"
        return f"{self.turn} | {self.action} | {weights} | {self.change} | {result}"


class Ledger:
    """The experiment ledger: rows the engine derives from tool results (see the module doc)."""

    def __init__(self, rows: int = LEDGER_ROWS) -> None:
        self.shown = rows
        self.rows: list[Row] = []

    def record(
        self,
        turn: int,
        name: str,
        arguments: Mapping[str, Any],
        output: str,
        *,
        before: str,
        after: str,
        change: str,
        changed: Sequence[str],
        measured: bool,
    ) -> Row | None:
        """The row for one action that ran, or ``None`` when it is neither a measurement nor an
        edit (a read, a list, analysis in Python that changed nothing, a recall).

        ``measured``: the action drew from the forward-pass budget (it was charged).
        ``changed``: the tensors that differ from the start after it (for the per-tensor summary).
        """
        summary = args_summary(arguments)
        action = f"{name} {summary}".strip()
        refused = output.lstrip().startswith(REFUSED) or "W was reset" in output
        if measured:
            kind = "measurement"
        elif name == "submit_program":
            kind = "program"
        elif name in EDIT_EVENTS and name != "write_tensor":
            kind = name
        elif before != after:
            kind = "edit"
        elif name == "write_tensor" or (refused and name == "python"):
            kind = "refused" if refused else "edit"
        else:
            return None
        if kind == "refused" or (refused and kind in EDIT_EVENTS):
            change_text = "refused"
        elif kind == "program":
            change_text = "stored" if output.startswith("program stored") else output[:60]
        elif kind == "measurement":
            change_text = "-"  # the digest names the weights it measured
        else:
            change_text = change
        numbers = tuple(numbers_of(output)) if measured else ()
        row = Row(
            turn,
            action,
            short(before),
            short(after),
            change_text,
            numbers,
            kind,
            tuple(changed) if kind not in ("measurement", "program", "refused") else (),
        )
        self.rows.append(row)
        return row

    def holds_numbers(self, turn: int) -> bool:
        """Whether the row of ``turn`` carries numbers (so its verbatim text may page sooner)."""
        return any(r.turn == turn and r.numbers for r in self.rows)

    def _summary(self, older: Sequence[Row]) -> list[str]:
        states: dict[str, list[Row]] = {}
        tensors: dict[str, list[int]] = {}
        for r in older:
            states.setdefault(r.after, []).append(r)
            for t in r.tensors:
                tensors.setdefault(t, []).append(r.turn)
        lines = []
        state_items = list(states.items())
        if len(state_items) > LEDGER_SUMMARY_LINES:
            hidden = len(state_items) - LEDGER_SUMMARY_LINES
            lines.append(f"earlier: {hidden} more weight states")
            state_items = state_items[-LEDGER_SUMMARY_LINES:]
        for digest, rows in state_items:
            measured = [r for r in rows if r.kind == "measurement"]
            turns = ",".join(str(r.turn) for r in rows)
            text = f"earlier: weights {digest}: turns {turns}"
            if measured:
                last = measured[-1]
                result = " ".join(f"{k}={v}" for k, v in last.numbers) or "-"
                text += f"; {len(measured)} measurements, last (turn {last.turn}): {result}"
            lines.append(text)
        tensor_items = sorted(tensors.items())
        if len(tensor_items) > LEDGER_SUMMARY_LINES:
            hidden = len(tensor_items) - LEDGER_SUMMARY_LINES
            tensor_items = tensor_items[:LEDGER_SUMMARY_LINES]
            lines.append(f"earlier: {hidden} more edited tensors")
        for name, edited in tensor_items:
            lines.append(f"earlier: {name} edited at turns {','.join(map(str, edited))}")
        return lines

    def render(self, start: int = 0, *, compact: bool = False) -> str:
        """The bounded table (see the module doc) of the rows from ``start`` on (after a handoff,
        the rows since it: the summary carries the earlier ones); empty while there are none.
        ``compact``: the newest :data:`COMPACT_ROWS` rows only, for a handoff that must fit a
        small window."""
        rows = self.rows[start:]
        if not rows:
            return ""
        shown = COMPACT_ROWS if compact else self.shown
        older, newest = rows[:-shown], rows[-shown:]
        heading = LEDGER_HEADING if start == 0 else LEDGER_SINCE_HEADING
        lines = [heading, *([] if compact else self._summary(older))]
        lines.append("turn | action | weights (before>after) | changed vs start | result")
        lines += [r.text() for r in newest]
        return "\n".join(lines)


@dataclass
class Attached:
    """Where the ledger sits: the history message it was appended to at the last page event,
    that message's content before it, and the text appended."""

    message: dict[str, Any]
    original: str
    text: str


def attach(messages: Sequence[dict[str, Any]], ledger: str) -> Attached | None:
    """Append ``ledger`` to the content of the history's LAST message (at a page event only)."""
    if not ledger or not messages:
        return None
    last = messages[-1]
    content = last.get("content")
    if not isinstance(content, str):
        return None
    last["content"] = content + LEDGER_SEPARATOR + ledger
    return Attached(last, content, ledger)


def detach(messages: Sequence[Mapping[str, Any]], attached: Attached | None) -> None:
    """Take an earlier ledger off its message, when that message is still in the history and
    still carries it (a stub that replaced the message took the ledger with it)."""
    if attached is None or not any(m is attached.message for m in messages):
        return
    if attached.message.get("content") == attached.original + LEDGER_SEPARATOR + attached.text:
        attached.message["content"] = attached.original


# ----------------------------------------------------------------------------- paging


def page_id(turn: int) -> str:
    return f"p{turn}"


def stub(page: str, tool: str, summary: str, nbytes: int) -> str:
    """The deterministic text that replaces a paged-out result in the history."""
    name = f"{tool} {summary}".strip()
    return f'[page {page}: {name}, {nbytes} bytes; recall("{page}") to see it again]'


@dataclass
class Page:
    id: str
    turn: int
    tool: str
    summary: str
    text: str

    @property
    def bytes(self) -> int:
        return len(self.text.encode("utf-8"))


@dataclass
class Candidate:
    """A history message holding a pageable result."""

    page: str
    turn: int
    message: dict[str, Any]
    paged: bool = False


@dataclass
class Pager:
    """The pages, the candidates in the history, and the page events (see the module doc)."""

    config: Config
    pages: dict[str, Page] = field(default_factory=dict)
    candidates: list[Candidate] = field(default_factory=list)
    pages_out: int = 0
    recalls: int = 0
    last_event: int = 0

    def offer(
        self,
        turn: int,
        tool: str,
        arguments: Mapping[str, Any],
        output: str,
        message: dict[str, Any],
        *,
        ledgered: bool,
        recalled: str | None = None,
    ) -> Candidate | None:
        """Register the history ``message`` that holds ``output`` when it is large enough to page.
        ``recalled``: the page a recall answered with; its text keeps that page's id."""
        limit = self.config.ledgered_threshold if ledgered else self.config.threshold
        if len(output.encode("utf-8")) <= limit:
            return None
        pid = recalled if recalled is not None else page_id(turn)
        if pid not in self.pages:
            self.pages[pid] = Page(pid, turn, tool, args_summary(arguments), output)
        candidate = Candidate(pid, turn, message)
        self.candidates.append(candidate)
        return candidate

    def _live(self, messages: Sequence[Mapping[str, Any]]) -> list[Candidate]:
        held = {id(m) for m in messages}
        return [c for c in self.candidates if not c.paged and id(c.message) in held]

    def eligible(self, turn: int, messages: Sequence[Mapping[str, Any]]) -> list[Candidate]:
        """Candidates still verbatim in the history and older than :attr:`Config.keep` turns."""
        return [c for c in self._live(messages) if turn - c.turn > self.config.keep]

    def due(
        self,
        turn: int,
        messages: Sequence[Mapping[str, Any]],
        estimate: int,
        *,
        stale: bool = False,
    ) -> tuple[list[Candidate], str | None]:
        """The candidates to page before the request of ``turn``, and why (``every`` or
        ``pressure``); ``(..., None)`` between batches. A batch happens when there is something
        to page or the ledger in the history is ``stale`` (rows were added since it was written).
        ``estimate``: the history's byte estimate in tokens (``context.byte_estimate``), never
        the endpoint's usage (a replay has none)."""
        ready = self.eligible(turn, messages)
        if not ready and not stale:
            return [], None
        if turn - self.last_event >= self.config.every:
            return ready, "every"
        # Pressure brings a batch forward only for something to page: a stale ledger alone waits
        # for the schedule, so a long session under pressure is not rewritten every turn.
        if ready and estimate > self.config.share * self.config.window:
            return ready, "pressure"
        return [], None

    def page_out(self, turn: int, chosen: Sequence[Candidate], reason: str) -> dict[str, Any]:
        """Replace each chosen message's content with its stub; the transcript event."""
        listed = []
        for c in chosen:
            page = self.pages[c.page]
            c.message["content"] = stub(page.id, page.tool, page.summary, page.bytes)
            c.paged = True
            listed.append({"page": page.id, "turn": c.turn, "tool": page.tool, "bytes": page.bytes})
        self.pages_out += len(chosen)
        self.last_event = turn
        return {"event": "page_out", "turn": turn, "reason": reason, "pages": listed}

    def recall(self, page: str) -> str:
        """The ``recall`` tool: the page's full text, or a refusal naming the pages there are."""
        key = str(page).strip()
        found = self.pages.get(key)
        if found is None:
            known = ", ".join(sorted(self.pages, key=lambda p: int(p[1:]))) or "none yet"
            return f"refused: no page {key!r}; pages you can recall: {known}. Nothing was run."
        self.recalls += 1
        return found.text

    def listing(self, level: str = "full") -> str:
        """The pages for the handoff summary: one per line (``full``), the newest
        :data:`COMPACT_ROWS` per line and the rest by id (``compact``), or every id on one line
        (``minimal``)."""
        pages = sorted(self.pages.values(), key=lambda p: p.turn)
        if level == "minimal":
            return ", ".join(p.id for p in pages)
        shown = pages if level == "full" else pages[-COMPACT_ROWS:]
        hidden = pages[: len(pages) - len(shown)]
        lines = [f"- {p.id}: {p.tool} {p.summary}".rstrip() + f" ({p.bytes} bytes)" for p in shown]
        if hidden:
            lines.insert(0, "- also " + ", ".join(p.id for p in hidden))
        return "\n" + "\n".join(lines)
