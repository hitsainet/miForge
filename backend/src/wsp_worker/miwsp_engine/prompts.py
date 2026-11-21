"""The editor's texts (feature 018; FTID §7; FTDD §6.2).

The system prompt, the mode texts, the nudge, the limit texts and the forward-budget texts are
``weightspace`` 0.3.0's (``ws:prompts.py``, ``ws:agent.py``; Apache-2.0; docs/REUSE.md), kept
equal in substance so a comparison of the two providers (FR-018.17) measures the engine, not a
different prompt. The texts that are the engine's own (one action per turn, the kernel restart,
the context handoff) say so where they are defined.
"""

from __future__ import annotations

from collections.abc import Iterable

SYSTEM = """You are a weight-space programmer. You read and write the weights of a neural network directly,
by reasoning about what each part of the network computes or should compute.

Rules:
- No gradient descent, no training loops, no fitting weights to data (no least squares on activations,
  no optimizers, no search over random perturbations). Derive every edit from your understanding of the
  network.
- Do not run the network or any of its layers on inputs inside the Python sandbox, even on made-up inputs.
  Static analysis of the weights (norms, SVD, products of weight matrices, comparisons between tensors)
  is fine. The only way to see the network's behaviour is the forward_* tools, which spend a limited budget.
- Prefer small, targeted edits that fix the cause and leave the rest of the network intact.
- When you are done, call `submit` exactly once with your answer and a confidence between 0 and 1.
"""  # noqa: E501 - weightspace's text, line for line

#: The engine's own rule (D25): one action per reply.
ONE_ACTION = """- Each reply takes exactly ONE action: one tool call. A reply with two or more is
  refused and nothing in it runs.
"""

NO_THINKING = """
Thinking is off for this run: do not write out step-by-step reasoning. Go straight to tool calls and weight
edits, with at most one short sentence of explanation each.
"""  # noqa: E501 - weightspace's text

MODES = {
    "read": "Task type: READ. Work out the answer from the weights. You cannot change any weights.",
    "fix": (
        "Task type: FIX. Find the cause of the described failure in the weights, explain it, and "
        "fix only that cause by editing weights. Use checkpoints so you can roll back an edit that "
        "does not help."
    ),
    "write": (
        "Task type: WRITE. Program the network's weights so that it implements the specification. "
        "Decide what algorithm each layer should compute, then convert it into weights."
    ),
    # weightspace 0.3.0's fourth mode (``ws:prompts.py`` MODES["improve"]), line for line: a
    # repair with a goal improves (fix 2026-11-21, live round 5: the engine said FIX where
    # weightspace said IMPROVE for the same request).
    "improve": (
        "Task type: IMPROVE. Make the network better at the stated goal by editing weights. "
        "Diagnose what limits it first, then make targeted edits, and keep an edit only if the "
        "evidence says it helps."
    ),
}
#: The request's ``kind`` -> the mode text (the ``weightspace`` adapter's ``MODE_FOR``); a
#: ``repair`` with a ``goal`` improves instead (:func:`mode_for`).
MODE_FOR = {"inspect": "read", "repair": "fix", "create": "write"}


def mode_for(kind: str, goal: object) -> str:
    """The adapter's ``session_mode`` for a session: ``repair`` fixes, or IMPROVES with a goal."""
    if kind == "repair" and goal:
        return "improve"
    return MODE_FOR[kind]


def task_text(kind: str, task: str, goal: object) -> str:
    """The adapter's ``session_task``: the task, and the goal after it when the run improves."""
    if kind == "repair" and goal:
        return f"{task}\n\nGoal: {goal}" if task else str(goal)
    return task


#: The engine's own task line for the PROGRAM style (fix 2026-11-21, run wsp_554d3932d108). The
#: adapter gives a program run weightspace's READ text, "You cannot change any weights", and the
#: editor concluded it could change nothing. A program run's editor writes no weights itself; its
#: program does (P19, FR-018.15), so this says so, per mode, and names the two program tools.
PROGRAM_AIM = {
    "fix": "find the cause of the described failure and fix only that cause",
    "improve": "make the network better at the stated goal",
    "write": "program the network's weights so that it implements the specification",
    "read": "work out the answer from the weights",
}
PROGRAM_MODE = (
    "Task type: {mode} (program). Your aim: {aim}. You do not edit weights directly: write a "
    "Python program defining `edit(base, support)` that returns the new values of the tensors it "
    "changes, computed from the current ones. Check it with `test_program` (costs {cost} forward "
    "pass{plural}; metrics on the validation split), then deliver it with `submit_program`, which "
    "runs it twice and must get the same edit both times. Calling `submit` without a submitted "
    "program changes nothing."
)


def program_mode(mode: str, cost: int) -> str:
    return PROGRAM_MODE.format(
        mode=mode.upper(), aim=PROGRAM_AIM[mode], cost=cost, plural="" if cost == 1 else "es"
    )


NUDGE = "Continue using the tools, or call `submit` when done."
STEP_LIMIT = "Step limit reached."
RUN_LIMIT = "Run limit reached ({reason})."
SUBMIT_NOW = "{why} Call `submit` now."
#: The submit-only turn's one retry (fix 2026-11-21, run ``wsp_855a74f820ff``): its submit could
#: not be read, most often because the reply was cut off mid-JSON.
SUBMIT_CUT = (
    "Your submit could not be read as one action ({detail}); it was probably cut off. "
    "Nothing was run. Send only the submit call again now, with a short answer."
)
FORWARD_EXHAUSTED = (
    "forward budget exhausted ({remaining} left, this probe costs {cost}). Submit your answer."
)
FORWARD_LEFT = "\n[forward budget: {left} of {budget} left]"
TRUNCATED = "\n... [truncated {n} chars]"
SUBMIT_REFUSED = "refused: {details}. Change the edit, then submit again."

#: The engine's own texts (D25, F18-D10, FR-018.12).
MULTIPLE_ACTIONS = "Reply with exactly one action; this reply had {n}. Nothing was run."
UNPARSEABLE = "Your reply could not be read as one action ({detail}). Nothing was run."
NOT_OFFERED = "The tool {name!r} is not available now; call `submit`. Nothing was run."
KERNEL_RESTARTED = (
    "The Python process was restarted after a timeout; its variables are gone. The weights are "
    "as last accepted."
)
#: The graceful end (fix 2026-11-21): a budget is nearly used up; and a forward pass that would not
#: finish before the session must end.
NEARLY_UP_EDIT = (
    "Budget nearly used up ({details}). Checkpoint your best weights now, then call `submit`. "
    "When a budget runs out the session ends, and only the last accepted weights are kept."
)
NEARLY_UP_READ = "Budget nearly used up ({details}). Call `submit` with your answer now."
NO_TIME_FOR_FORWARD = (
    "Not run: about {left} s are left and a forward pass has taken about {took} s here, so it "
    "would not finish before the session must end. Nothing was charged. Call `submit` now."
)
#: The repeat guard (fix 2026-11-21, runs wsp_7767fb91254b and wsp_554d3932d108: the same
#: forward pass 13 and 10 times in a row on unchanged weights). The engine's own texts.
REPEATED = (
    "You already ran this on these weights; the result is unchanged (see turn {turn}). Edit the "
    "weights, try a different measurement, or call `submit`. Nothing was run or charged."
)
REPEAT_LIMIT = "The same action was asked for on unchanged weights {n} times."
HANDOFF = """The conversation was restarted to fit the context window. Accepted edits are kept.
Your notes: {notes}
Edits so far: {diff}
Budget: {budget}
Continue the task."""
#: The handoff with paging on (2026-11-21): the ledger and the recallable pages ride in the
#: summary, so a restart keeps every measurement, not only the notes the editor wrote.
HANDOFF_PAGED = (
    "The conversation was restarted to fit the context window. Accepted edits are kept.\n"
    "Your notes: {notes}\n"
    "Edits so far: {diff}\n"
    "Budget: {budget}\n"
    "{ledger}\n"
    "Pages you can recall: {pages}\n"
    "Continue the task."
)
#: The engine's own rule with paging on (2026-11-21).
PAGING = (
    "- Old long tool results become a stub naming a page; `recall` shows one again. The engine's\n"
    "  ledger of your measurements and edits is in the conversation.\n"
)
RECALL_DESCRIPTION = (
    "Show a paged-out tool result again, by its page id (e.g. 'p5'). No forward pass."
)


def system(*, thinking_off: bool, structured_tools: str | None = None, paging: bool = False) -> str:
    """``SYSTEM`` with the one-action rule (and, with paging on, the paging rule); in structured
    mode the tool list follows it."""
    text = SYSTEM + ONE_ACTION + (PAGING if paging else "")
    if thinking_off:
        text += NO_THINKING
    if structured_tools is not None:
        text += (
            '\nReply with ONE JSON object {"tool": <name>, "arguments": {...}} and nothing '
            "else. The tools:\n" + structured_tools
        )
    return text


def inventory(rows: Iterable[tuple[str, list[int], str]], total: int) -> str:
    rows = list(rows)
    shown = rows[:200]
    lines = [f"  {n} {shape} {dtype}" for n, shape, dtype in shown]
    if len(rows) > len(shown):
        lines.append(f"  ... {len(rows) - len(shown)} more (use list_tensors)")
    return f"## Weights ({len(rows)} tensors, {total:,} weights)\n" + "\n".join(lines)


def forward_lines(tools: Iterable[tuple[str, int, str]], budget: int) -> str:
    lines = [f"  forward_{n} (cost {c}): {d}" for n, c, d in tools] or ["  none"]
    return f"## Forward passes (budget {budget})\n" + "\n".join(lines)


def intro(
    mode: str,
    task: str,
    context: str,
    inventory_text: str,
    forward_text: str,
    *,
    mode_text: str | None = None,
) -> str:
    """The first user message, in ``weightspace``'s structure (``ws:agent.py:392-404``); a program
    session's task line is :func:`program_mode`'s (``mode_text``)."""
    parts = [MODES[mode] if mode_text is None else mode_text, f"## Task\n{task}"]
    if context:
        parts.append(f"## Context\n{context}")
    parts += [inventory_text, forward_text]
    return "\n\n".join(parts)
