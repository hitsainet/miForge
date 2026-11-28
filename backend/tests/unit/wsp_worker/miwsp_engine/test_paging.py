"""Context paging and the experiment ledger (operator decision 2026-11-21; FR-018.22 to
FR-018.25; ``0xcc/reviews/018_paging_2026-11-21.md``).

The D128 runs (bundle ``wsb_6a53417a3a39``: a probe with weight [4096] and a bias) sent one
4,096-value ``read_tensor`` on every later turn, and a handoff kept only the editor's notes.
These tests drive the engine's session loop in-process on a scripted client that RECORDS every
request it is sent (``Session._loop``, as ``test_repeat_guard.py``), and once through the real
child against a scripted HTTP endpoint, then a real replay.

The tool results the ledger reads are the shapes the D128 transcripts recorded.
"""

from __future__ import annotations

import contextlib
import copy
import io
import itertools
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from tests.support.wsp_engine import line_for, result, run_case, run_main, transcript
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from tests.support.wsp_scripted_endpoint import ScriptedEndpoint
from wsp_worker.miwsp_engine import context, paging, prompts
from wsp_worker.miwsp_engine import protocol as proto
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine.protocol import Reply
from wsp_worker.miwsp_engine.transcript import ScriptedClient, TranscriptWriter

TOOLS = [
    {"name": "practice_metrics", "version": "1", "cost": 1},
    {"name": "compare_to_parent", "version": "1", "cost": 2},
    {"name": "firing_positions", "version": "1", "cost": 1},
]

# ----------------------------------------------------------------------------- real tool results
# From t_wsp_855a74f820ff.json (the D128 engine run, thinking on, 40 steps): the observation
# tools' answers exactly as the editor saw them (ForwardBudget.charge's text, with its suffix).
PRACTICE_METRICS = (
    '{\n "tool": "practice_metrics",\n "split": "train",\n "rule": "max",\n "rows": 28568,\n'
    ' "auroc": 0.9903010198426706,\n "bar": 2.307197332382202,\n "recall": 0.16924110141034251,\n'
    ' "false_alarm_rate": 0.0014033014513091325\n}\n[forward budget: 59 of 60 left]'
)
STRATIFIED = (
    '{\n "tool": "stratified_false_alarms",\n "by": "length_band",\n "bar": 2.307197332382202,\n'
    ' "strata": {\n  "len_17_64": {\n   "negatives": 6345,\n   "false_alarms": 14,\n'
    '   "rate": 0.002206461780929866\n  },\n  "len_65_256": {\n   "negatives": 22308,\n'
    '   "false_alarms": 27,\n   "rate": 0.0012103281334050564\n  },\n  "len_257_up": {\n'
    '   "negatives": 2833,\n   "false_alarms": 0,\n   "rate": 0.0\n  },\n  "len_1_16": {\n'
    '   "negatives": 308,\n   "false_alarms": 0,\n   "rate": 0.0\n  }\n }\n}\n'
    "[forward budget: 58 of 60 left]"
)
COMPARE = (
    '{\n "tool": "compare_to_parent",\n "split": "train",\n "now": {\n  "rows": 28568,\n'
    '  "auroc": 0.9903010198426706,\n  "bar": 1.653002381324768,\n'
    '  "recall": 0.16924110141034251,\n'
    '  "false_alarm_rate": 0.0014033014513091325\n },\n "parent": {\n  "rows": 28568,\n'
    '  "auroc": 0.9903010198426706,\n  "bar": 2.307197332382202,\n'
    '  "recall": 0.16924110141034251,\n'
    '  "false_alarm_rate": 0.0014033014513091325\n },\n "flipped": 0,\n "flipped_rows": []\n}\n'
    "[forward budget: 53 of 60 left]"
)
COSINES = (
    '{\n "tool": "direction_cosines",\n "cosines": {\n  "weight": {\n   "cos_parent": 1.0,\n'
    '   "random_mean_abs": 0.008206767266966038,\n   "random_max_abs": 0.017706603426565763,\n'
    '   "directions": 8\n  }\n }\n}\n[forward budget: 50 of 60 left]'
)
WRITE = (
    "wrote bias[:]\n1 of 4,097 weights changed in 1 tensor(s)\n"
    "  bias: 1/1 changed, |delta|=0.3002, max=0.3002"
)


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    return make_bundle(tmp_path / "bundle", rule="mean", d_model=1024)


# ----------------------------------------------------------------------------- the ledger (unit)


def test_the_numbers_of_each_real_tool_answer() -> None:
    assert paging.numbers_of(PRACTICE_METRICS) == [
        ("auroc", "0.9903"),
        ("recall", "0.1692"),
        ("far", "0.001403"),
        ("bar", "2.307"),
    ]
    assert paging.numbers_of(STRATIFIED) == [
        ("bar", "2.307"),
        ("far[len_17_64]", "0.002206"),
        ("far[len_1_16]", "0"),
        ("far[len_257_up]", "0"),
        ("far[len_65_256]", "0.00121"),
    ]
    # compare_to_parent: the numbers NOW (bar 1.653), not the parent's (2.307), and the flips.
    assert paging.numbers_of(COMPARE) == [
        ("auroc", "0.9903"),
        ("recall", "0.1692"),
        ("far", "0.001403"),
        ("bar", "1.653"),
        ("flipped", "0"),
    ]
    assert paging.numbers_of(COSINES) == [("cos[weight]", "1")]
    # Not JSON objects: a write, a read, a refusal, Python output.
    for text in (WRITE, "bias[:] shape=(1,)\n-4.235", "refused: no such tensor", "[1, 2]"):
        assert paging.numbers_of(text) == []


def test_an_at_parent_bar_block_is_read_when_present() -> None:
    """Observation tools v2's ``at_parent_bar`` (the parent's bar held fixed): read at the top
    level (``practice_metrics``) or inside ``now`` (``compare_to_parent``); a v1 answer has none."""
    top = json.loads(PRACTICE_METRICS.split(paging.FORWARD_SUFFIX)[0])
    top["at_parent_bar"] = {
        "bar": 2.307,
        "source": "shipped",
        "recall": 0.25,
        "false_alarm_rate": 0.001,
    }
    numbers = dict(paging.numbers_of(json.dumps(top)))
    assert (numbers["parent_bar"], numbers["recall_at_parent_bar"]) == ("2.307", "0.25")
    assert numbers["far_at_parent_bar"] == "0.001"
    nested = json.loads(COMPARE.split(paging.FORWARD_SUFFIX)[0])
    nested["now"]["at_parent_bar"] = {"recall": 0.5}
    assert dict(paging.numbers_of(json.dumps(nested)))["recall_at_parent_bar"] == "0.5"


def test_the_ledger_reads_the_real_v2_tools(tmp_path: Path, fx: FixtureBundle) -> None:
    """The real ``practice_metrics`` and ``compare_to_parent`` v2, run in a session: their
    ``at_parent_bar`` numbers reach the row."""
    v2 = [
        {"name": "practice_metrics", "version": "2", "cost": 1},
        {"name": "compare_to_parent", "version": "2", "cost": 2},
    ]
    steps = [
        ("forward_practice_metrics", {"split": "train"}),
        ("forward_compare_to_parent", {"split": "train"}),
        ("submit", {"answer": "a"}),
    ]
    s, _ = _session(tmp_path, fx, steps, observation_tools=v2)
    lines = _run(s)
    rows = [m["row"] for m in lines if m.get("event") == "ledger"]
    assert len(rows) == 2
    for row in rows:
        assert {"parent_bar", "recall_at_parent_bar", "far_at_parent_bar"} <= set(row["numbers"])


def test_the_ledger_reads_the_real_v3_tools(tmp_path: Path, fx: FixtureBundle) -> None:
    """2026-11-22: version 3's ``at_own_bar`` gives the row its recall and far, and a bias-only
    head is marked ``same_verdict_as_parent``, in both tools (``compare_to_parent`` from ``now``
    and its top level)."""
    v3 = [
        {"name": "practice_metrics", "version": "3", "cost": 1},
        {"name": "compare_to_parent", "version": "3", "cost": 2},
    ]
    steps = [
        ("write_tensor", {"name": "bias", "values": 0.75}),
        ("forward_practice_metrics", {"split": "train"}),
        ("forward_compare_to_parent", {"split": "train"}),
        ("submit", {"answer": "a"}),
        ("submit", {"answer": "b"}),
    ]
    s, _ = _session(tmp_path, fx, steps, observation_tools=v3)
    lines = _run(s)
    rows = [m["row"] for m in lines if m.get("event") == "ledger" and m["row"]["numbers"]]
    assert len(rows) == 2
    for row in rows:
        assert {"auroc", "recall", "far", "same_verdict_as_parent"} <= set(row["numbers"])
        assert "parent_bar" not in row["numbers"]


def _record(ledger: paging.Ledger, turn: int, name: str, args: dict[str, Any], output: str,
            before: str = "a" * 64, after: str = "a" * 64, measured: bool = False,
            change: str = "none") -> paging.Row | None:  # fmt: skip
    return ledger.record(
        turn,
        name,
        args,
        output,
        before=before,
        after=after,
        change=change,
        changed=[] if change == "none" else ["bias"],
        measured=measured,
    )


def test_rows_are_measurements_edits_and_events_only() -> None:
    ledger = paging.Ledger()
    assert _record(ledger, 1, "list_tensors", {}, "bias [1] float64 1") is None
    assert _record(ledger, 2, "read_tensor", {"name": "weight"}, "weight[:] ...") is None
    assert _record(ledger, 3, "python", {"code": "print(1)"}, "1") is None
    assert _record(ledger, 4, "recall", {"page": "p2"}, "weight[:] ...") is None
    m = _record(ledger, 5, "forward_practice_metrics", {"rule": "max"}, PRACTICE_METRICS,
                measured=True)  # fmt: skip
    e = _record(ledger, 6, "write_tensor", {"name": "bias", "values": -4.535}, WRITE,
                after="b" * 64, change="bias 1/1 |d|=0.3002")  # fmt: skip
    r = _record(ledger, 7, "write_tensor", {"name": "weight", "values": [0.0] * 4},
                "refused: tensor 'weight' is frozen. Nothing was changed.")  # fmt: skip
    c = _record(ledger, 8, "checkpoint", {"label": "best"}, "saved checkpoint 'best'")
    p = _record(ledger, 9, "python", {"code": "W['weight'][0] = 1"}, "(no output)",
                after="c" * 64, change="weight 1/4096 |d|=1")  # fmt: skip
    assert m is not None and e is not None and r is not None and c is not None and p is not None
    assert [x.kind for x in ledger.rows] == ["measurement", "edit", "refused", "checkpoint", "edit"]
    assert m.text() == (
        "5 | forward_practice_metrics rule=max | aaaaaaaa | - | "
        "auroc=0.9903 recall=0.1692 far=0.001403 bar=2.307"
    )
    assert e.text() == (
        "6 | write_tensor name=bias values=-4.535 | aaaaaaaa>bbbbbbbb | bias 1/1 |d|=0.3002 | -"
    )
    assert "values=<list of 4>" in r.text() and "| refused |" in r.text()
    assert p.action.startswith("python code#") and "(18 chars)" in p.action
    assert m.document()["numbers"] == {
        "auroc": "0.9903",
        "recall": "0.1692",
        "far": "0.001403",
        "bar": "2.307",
    }


def test_the_rendered_ledger_is_bounded() -> None:
    ledger = paging.Ledger()
    for t in range(1, 201):
        digest = f"{t // 2:064x}"
        if t % 2:
            before = f"{t // 2 - 1:064x}"
            _record(ledger, t, "write_tensor", {"name": f"t{t % 9}", "values": t}, WRITE,
                    before=before, after=digest, change=f"t{t % 9} 1/1 |d|=1")  # fmt: skip
        else:
            _record(ledger, t, "forward_compare_to_parent", {"split": "train"}, COMPARE,
                    before=digest, after=digest, measured=True)  # fmt: skip
    text = ledger.render()
    lines = text.splitlines()
    full = [x for x in lines if x[:1].isdigit()]
    earlier = [x for x in lines if x.startswith("earlier:")]
    assert len(full) == paging.LEDGER_ROWS and full[-1].startswith("200 |")
    assert len(earlier) <= 2 * (paging.LEDGER_SUMMARY_LINES + 1)
    assert len(text.encode()) < 3_000, len(text.encode())


# ----------------------------------------------------------------------------- the pager (unit)


def test_the_stub_names_its_page_and_recall_returns_the_text() -> None:
    assert paging.stub("p5", "read_tensor", "name=weight", 12052) == (
        '[page p5: read_tensor name=weight, 12052 bytes; recall("p5") to see it again]'
    )
    pager = paging.Pager(paging.Config(True, 8_192))
    message = {"role": "tool", "content": "x" * 2_000}
    assert pager.offer(5, "read_tensor", {"name": "weight"}, "x" * 2_000, message, ledgered=False)
    assert pager.recall("p5") == "x" * 2_000 and pager.recalls == 1
    refused = pager.recall("p9")
    assert refused.startswith("refused: no page 'p9'") and "p5" in refused
    assert pager.recalls == 1  # a refusal is not a recall


def test_paging_waits_keep_turns_then_batches_every_m_turns() -> None:
    config = paging.Config(True, 100_000)
    pager = paging.Pager(config)
    history: list[dict[str, Any]] = []
    for turn in (1, 2, 5):
        message = {"role": "tool", "content": "y" * 1_600}
        history.append(message)
        pager.offer(turn, "read_tensor", {"index": str(turn)}, "y" * 1_600, message, ledgered=False)
    # Turn 4: p1 is still inside the keep window (4 - 1 = 3 turns, not more).
    assert pager.due(4, history, 0) == ([], None)
    # Turn 5: p1 is past it, and no page event has happened yet.
    chosen, reason = pager.due(5, history, 0)
    assert [c.page for c in chosen] == ["p1"] and reason == "every"
    pager.page_out(5, chosen, reason)
    assert history[0]["content"].startswith("[page p1: read_tensor index=1")
    # p2 is eligible at turn 6, but the next batch is not before turn 9 (5 + 4) ...
    assert pager.eligible(6, history) and pager.due(6, history, 0) == ([], None)
    # ... unless the history passes PAGE_SHARE of the window.
    assert pager.due(6, history, 60_000)[1] == "pressure"
    assert [c.page for c in pager.due(9, history, 0)[0]] == ["p2", "p5"]


def test_a_small_result_pages_only_when_the_ledger_holds_its_numbers() -> None:
    pager = paging.Pager(paging.Config(True, 8_192))
    text = "z" * 300
    assert pager.offer(1, "python", {}, text, {"content": text}, ledgered=False) is None
    assert pager.offer(2, "forward_x", {}, text, {"content": text}, ledgered=True) is not None


def test_the_configuration_round_trips_through_the_transcript_header() -> None:
    config = paging.Config(True, 32_768)
    header = config.header()
    assert header is not None and paging.Config.from_header(header, 8_192) == config
    assert paging.Config(False, 8_192).header() is None
    assert paging.Config.from_header(None, 8_192).on is False


# ----------------------------------------------------------------------------- in-process sessions


class Recording(ScriptedClient):
    """A scripted client that keeps a deep copy of every request's messages and fields."""

    def __init__(self, replies: Sequence[Mapping[str, Any]], recorded: Sequence[list[str]] = ()):
        super().__init__(replies, recorded)
        self.sent: list[list[dict[str, Any]]] = []
        self.fields: list[dict[str, Any]] = []

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> Reply:
        self.sent.append(copy.deepcopy([dict(m) for m in messages]))
        self.fields.append(copy.deepcopy(dict(fields)))
        return super().complete(messages, fields=fields, max_tokens=max_tokens)


def _replies(steps: Sequence[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"c{i}", "name": n, "arguments": a}],
        }
        for i, (n, a) in enumerate(steps)
    ]


def _session(
    tmp: Path,
    fx: FixtureBundle,
    steps: Sequence[tuple[str, dict[str, Any]]],
    *,
    paging_on: bool = True,
    window: int = 32_768,
    task: str | None = None,
    **over: Any,
) -> tuple[session_mod.Session, Recording]:
    line = line_for(
        tmp,
        fx,
        over.pop("kind", "repair"),
        None,
        python=False,
        forward_passes=over.pop("forward_passes", 200),
        observation_tools=over.pop("observation_tools", TOOLS),
        max_steps=over.pop("max_steps", 80),
        tokens=10_000_000,
        editor={"context_window_tokens": window, **over.pop("editor", {})},
        **({"paging": "on"} if paging_on else {}),
        **over,
    )
    if task is not None:
        line["spec"]["request"]["task"] = task
    client = Recording(_replies(steps))
    s = session_mod.Session(line, client=client)
    s.writer = TranscriptWriter(s.out / "transcript.jsonl", s.header())
    return s, client


def _run(s: session_mod.Session) -> list[dict[str, Any]]:
    with contextlib.redirect_stdout(io.StringIO()):
        s._loop()
    assert s.writer is not None
    s.writer.close()
    return [json.loads(x) for x in (s.out / "transcript.jsonl").read_text().splitlines()]


def _measurements(n: int) -> list[tuple[str, dict[str, Any]]]:
    """``n`` distinct measurements on unchanged weights (the repeat guard runs each once)."""
    splits = ("train", "validation", "all")
    rules = ("mean", "max", "last", "softmax", "rolling_mean_max")
    combos = [("forward_practice_metrics", {"split": s, "rule": r}) for s in splits for r in rules]
    return combos[:n]


READ = ("read_tensor", {"name": "weight"})


def _carrying(sent: list[dict[str, Any]]) -> list[int]:
    """The indexes of the messages of one request that carry a ledger."""
    return [
        i
        for i, m in enumerate(sent)
        if paging.LEDGER_HEADING in str(m.get("content"))
        or paging.LEDGER_SINCE_HEADING in str(m.get("content"))
    ]


def test_the_ledger_is_written_into_the_history_at_page_events_only(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """At each page event the ledger moves to the end of the history (the message then last);
    between events it stays where it is, so the history only grows at its end (the prefix cache
    and the gateway's meter: ``paging``'s module doc). One ledger at a time, never in the
    transcript's messages, one ``ledger`` event per row."""
    steps = [*_measurements(3), ("write_tensor", {"name": "bias", "values": 0.5}),
             *_measurements(8)[3:], ("submit", {"answer": "a"})]  # fmt: skip
    s, client = _session(tmp_path, fx, steps)
    lines = _run(s)
    events = [m for m in lines if m.get("event") == "page_out"]
    paged_at = [e["turn"] for e in events]
    # The first result (turn 1) leaves the keep window at turn 5: the first event, and the first
    # ledger in the history.
    assert paged_at and paged_at[0] == 2 + paging.KEEP_TURNS
    for n, sent in enumerate(client.sent):
        turn = n + 1
        carrying = _carrying(sent)
        assert len(carrying) <= 1, (turn, carrying)
        if turn < paged_at[0]:
            assert carrying == [], turn
        elif turn in paged_at:
            assert carrying == [len(sent) - 1], turn  # moved to the end at this event
        else:
            assert len(carrying) == 1, turn  # still in the history, where the event put it
    assert all(e["ledger_rows"] >= 1 for e in events)
    assert all(paging.LEDGER_HEADING not in str(m.get("content")) for m in lines if m.get("role"))
    rows = [m["row"] for m in lines if m.get("event") == "ledger"]
    assert [r["kind"] for r in rows] == ["measurement"] * 3 + ["edit"] + ["measurement"] * 5
    assert rows[3]["before"] != rows[3]["after"] and rows[4]["before"] == rows[3]["after"]
    assert s.result()["engine"]["ledger_rows"] == 9
    assert prompts.PAGING in client.sent[0][0]["content"]


def test_the_ledger_in_the_history_holds_every_paged_result(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Nothing is paged before its row is in a ledger: at every request, each stubbed
    measurement's turn has a row in the ledger the history carries."""
    steps = [*_measurements(12), ("submit", {"answer": "a"})]
    s, client = _session(tmp_path, fx, steps)
    _run(s)
    for sent in client.sent:
        stubbed = [
            int(str(m["content"]).split()[1].rstrip(":")[1:])
            for m in sent
            if str(m.get("content")).startswith("[page p")
        ]
        if not stubbed:
            continue
        (where,) = _carrying(sent)
        table = str(sent[where]["content"]).split(paging.LEDGER_SEPARATOR)[-1]
        held = {int(line.split(" | ")[0]) for line in table.splitlines() if line[:1].isdigit()}
        for line in table.splitlines():
            if line.startswith("earlier: weights") and "turns " in line:
                held |= {int(t) for t in line.split("turns ")[1].split(";")[0].split(",")}
        assert set(stubbed) <= held, (stubbed, sorted(held))


def test_with_paging_off_nothing_changes(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [READ, *_measurements(6), ("submit", {"answer": "a"})]
    s, client = _session(tmp_path, fx, steps, paging_on=False)
    lines = _run(s)
    assert "recall" not in s.toolbox.names()
    assert all(paging.LEDGER_HEADING not in str(m.get("content")) for r in client.sent for m in r)
    assert not [m for m in lines if m.get("event") in ("page_out", "ledger", "recall")]
    assert lines[0]["paging"] is None and prompts.PAGING not in client.sent[0][0]["content"]
    engine = s.result()["engine"]
    assert (engine["pages_out"], engine["recalls"], engine["ledger_rows"]) == (0, 0, 0)


def test_a_bulky_read_is_stubbed_after_keep_turns_and_recall_brings_it_back(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [READ, *_measurements(5), ("recall", {"page": "p1"}), ("recall", {"page": "p1"}),
             ("recall", {"page": "p99"}), ("submit", {"answer": "a"})]  # fmt: skip
    s, client = _session(tmp_path, fx, steps)
    lines = _run(s)
    full = next(m["content"] for m in lines if m.get("role") == "tool")
    assert len(full.encode()) > paging.PAGE_THRESHOLD_BYTES
    stub = paging.stub("p1", "read_tensor", "name=weight", len(full.encode()))
    # Verbatim in the next KEEP_TURNS requests (2, 3, 4), stubbed from the batch at turn 5 on.
    for n in range(1, 1 + paging.KEEP_TURNS):
        assert client.sent[n][3]["content"] == full, n
    for n in range(1 + paging.KEEP_TURNS, len(client.sent)):
        assert client.sent[n][3]["content"] == stub, n
    events = [m for m in lines if m.get("event") == "page_out"]
    assert events[0]["turn"] == 2 + paging.KEEP_TURNS and events[0]["reason"] == "every"
    assert events[0]["pages"][0] == {
        "page": "p1",
        "turn": 1,
        "tool": "read_tensor",
        "bytes": len(full.encode()),
    }
    # recall: the full text, twice (the repeat guard does not treat it as a measurement), no
    # forward pass, no ledger row; an unknown page is refused by name.
    tools = [m["content"] for m in lines if m.get("role") == "tool"]
    assert tools[6] == full and tools[7] == full
    assert tools[8].startswith("refused: no page 'p99'; pages you can recall: p1")
    assert s.repeated_turns == 0 and s.pager.recalls == 2 and s.forward.used == 5
    recalls = [m for m in lines if m.get("event") == "recall"]
    assert [(r["page"], r["found"]) for r in recalls] == [
        ("p1", True),
        ("p1", True),
        ("p99", False),
    ]
    assert all(
        r["row"]["action"].split()[0] != "recall" for r in lines if r.get("event") == "ledger"
    )
    engine = s.result()["engine"]
    assert engine["recalls"] == 2 and engine["pages_out"] == sum(len(e["pages"]) for e in events)


def test_a_recalled_page_is_pageable_again_under_its_own_id(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [READ, *_measurements(4), ("recall", {"page": "p1"}), *_measurements(15)[4:],
             ("submit", {"answer": "a"})]  # fmt: skip
    s, client = _session(tmp_path, fx, steps)
    _run(s)
    recalled_turn = 6
    final = client.sent[-1]
    held = [
        m
        for m in final
        if m.get("role") == "tool" and m.get("tool_call_id") == f"c{recalled_turn - 1}"
    ]
    assert held and held[0]["content"].startswith("[page p1: read_tensor name=weight")
    assert "p6" not in s.pager.pages  # no second copy of the text


def test_page_events_are_batched_and_the_prefix_holds_between_them(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """Between two page events every request's history is a prefix of the next one's (llama.cpp's
    prefix cache), and page events are at least PAGE_EVERY_TURNS turns apart."""
    steps = [READ, *_measurements(14), ("submit", {"answer": "a"})]
    s, client = _session(tmp_path, fx, steps)
    lines = _run(s)
    paged_at = [m["turn"] for m in lines if m.get("event") == "page_out"]
    assert len(paged_at) >= 3
    assert all(b - a >= paging.PAGE_EVERY_TURNS for a, b in itertools.pairwise(paged_at))
    for n in range(len(client.sent) - 1):
        before, after = client.sent[n], client.sent[n + 1]
        turn_of_next = n + 2  # request n + 1 is sent at turn n + 2
        if turn_of_next in paged_at:
            assert after[: len(before)] != before
        else:
            assert after[: len(before)] == before, n


def test_the_same_transcript_gives_the_same_pages(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [READ, *_measurements(10), ("submit", {"answer": "a"})]
    a, ca = _session(tmp_path / "a", fx, steps)
    b, cb = _session(tmp_path / "b", fx, steps)
    la, lb = _run(a), _run(b)
    pick = ("page_out", "ledger", "recall")
    assert [m for m in la if m.get("event") in pick] == [m for m in lb if m.get("event") in pick]
    assert ca.sent == cb.sent and a.result()["engine"] == b.result()["engine"]


def _replay_in_process(tmp: Path, fx: FixtureBundle, original: Path) -> session_mod.Session:
    line = line_for(
        tmp,
        fx,
        "repair",
        None,
        mode="replay",
        replay_of_mode="session",
        python=False,
        forward_passes=200,
        observation_tools=TOOLS,
        max_steps=80,
        tokens=10_000_000,
        editor={"context_window_tokens": 32_768},
        extra=[("transcript.jsonl", "transcript", original)],
    )
    r = session_mod.Session(line)
    r.writer = TranscriptWriter(r.out / "transcript.jsonl", r.header())
    return r


def test_a_replay_pages_the_same_messages_at_the_same_turns(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The replay's request names no paging (a replay pushes the original request, but the
    engine pages by the transcript's header): the same decisions, from the transcript alone."""
    steps = [READ, *_measurements(8), ("recall", {"page": "p1"}), ("submit", {"answer": "a"})]
    s, _ = _session(tmp_path / "s", fx, steps)
    original = _run(s)
    r = _replay_in_process(tmp_path / "r", fx, s.out / "transcript.jsonl")
    assert r.paging == s.paging and "recall" in r.toolbox.names()
    again = _run(r)
    pick = ("page_out", "ledger", "recall")
    assert [m for m in again if m.get("event") in pick] == [
        m for m in original if m.get("event") in pick
    ]
    assert r.replay_divergence is None
    mine, theirs = r.result()["engine"], s.result()["engine"]
    for key in ("pages_out", "recalls", "ledger_rows"):
        assert mine[key] == theirs[key], key


def test_a_transcript_from_before_paging_replays_with_paging_off(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [READ, *_measurements(6), ("submit", {"answer": "a"})]
    s, _ = _session(tmp_path / "s", fx, steps, paging_on=False)
    _run(s)
    old = tmp_path / "old.jsonl"
    lines = (s.out / "transcript.jsonl").read_text().splitlines()
    header = json.loads(lines[0])
    del header["paging"]  # an engine from before paging wrote no such key
    old.write_text("\n".join([json.dumps(header), *lines[1:]]) + "\n")
    r = _replay_in_process(tmp_path / "r", fx, old)
    assert r.paging.on is False and "recall" not in r.toolbox.names()
    _run(r)
    assert r.pager.pages_out == 0 and r.replay_divergence is None


def test_the_handoff_carries_the_ledger_and_the_pages(tmp_path: Path, fx: FixtureBundle) -> None:
    after = [
        step
        for v in (0.1, 0.2, 0.3)
        for step in (
            ("write_tensor", {"name": "bias", "values": v}),
            ("forward_practice_metrics", {"split": "train"}),
            ("forward_practice_metrics", {"split": "validation"}),
        )
    ]
    steps = [READ, *_measurements(15), READ, *after, ("submit", {"answer": "a"})]
    # A long task, so a 16,384-token window (the least that pages) hands off mid-session.
    s, _ = _session(tmp_path, fx, steps, window=16_384, task="T" * 5_000)
    client = Counted(_replies(steps))
    s._client = client
    lines = _run(s)
    handoffs = [m for m in lines if m.get("event") == "handoff"]
    assert handoffs and s.handoffs == len(handoffs)
    last = handoffs[-1]["messages"][1]["content"]
    assert paging.LEDGER_HEADING in last and "Pages you can recall:" in last
    pages = last.split("Pages you can recall:")[1]
    assert re.search(r"\bp1\b", pages), pages  # the read, by id (the compact listing's "also")
    # Rows from before the handoff ride in its summary; a later ledger shows only the rows since.
    since = next(i for i, r in enumerate(client.sent) if r[1]["content"] == last)
    summary_rows = {int(x.split(" | ")[0]) for x in last.splitlines() if x[:1].isdigit()}
    later = [
        str(sent[i]["content"]).split(paging.LEDGER_SEPARATOR)[-1]
        for sent in client.sent[since + 1 :]
        for i in _carrying(sent)
        if i != 1
    ]
    assert later, "no ledger after the last handoff: the test needs a page event there"
    for table in later:
        assert table.startswith(paging.LEDGER_SINCE_HEADING), table[:80]
        shown = {int(x.split(" | ")[0]) for x in table.splitlines() if x[:1].isdigit()}
        assert not shown & summary_rows, (shown, summary_rows)


def test_a_handoff_that_would_crowd_the_window_keeps_less(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """A summary whose ledger would take more than a quarter of the room the bare restart leaves
    keeps the compact ledger (the newest rows), else the page ids alone
    (``session.HANDOFF_LEDGER_SHARE``), so a small window is not crowded by its own summary."""
    steps = [READ, *_measurements(15), ("submit", {"answer": "a"})]
    s, _ = _session(tmp_path, fx, steps)
    s._client = Counted(_replies(steps))
    _run(s)
    assert len(s.ledger.rows) == 15 and s.pager.pages
    compact = s.ledger.render(compact=True)
    assert len([x for x in compact.splitlines() if x[:1].isdigit()]) == paging.COMPACT_ROWS
    assert "earlier:" not in compact
    assert s.pager.listing("minimal").startswith("p1, ")

    s.writer = None

    def summary_at(window: int) -> str:
        s.guard.window = window
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}]
        s._handoff(messages, "intro")
        return str(messages[1]["content"])

    roomy, tight = summary_at(1_000_000), summary_at(3_500)
    assert paging.LEDGER_HEADING in roomy and "earlier:" in roomy  # the full ledger
    assert "Ledger: (left out to fit the context window)" in tight
    assert "Pages you can recall: p1, " in tight


def test_the_handoff_with_paging_off_is_unchanged(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [READ, *_measurements(15), ("submit", {"answer": "a"})]
    s, _ = _session(tmp_path, fx, steps, window=12_000, paging_on=False)
    s._client = Counted(_replies(steps))
    lines = _run(s)
    handoffs = [m for m in lines if m.get("event") == "handoff"]
    assert handoffs
    text = handoffs[0]["messages"][1]["content"]
    assert "Pages you can recall" not in text and paging.LEDGER_HEADING not in text
    assert "The conversation was restarted" in text


def test_the_never_page_set(tmp_path: Path, fx: FixtureBundle) -> None:
    """With both thresholds at zero, every result that RAN is a candidate. Still never paged:
    the system prompt, the intro, refusals, the repeat guard's notice, the nearly-up notice, the
    submit-only turn and the latest KEEP_TURNS turns."""
    steps = [
        *_measurements(2),
        _measurements(1)[0],  # a repeat: the guard's notice
        ("no_such_tool", {}),  # refused: not offered
        *_measurements(9)[2:9],
    ]
    s, client = _session(
        tmp_path,
        fx,
        [*steps, ("submit", {"answer": "a"})],
        max_steps=len(steps),
        forward_passes=11,
        task="T" * 4_000,
    )
    s.pager.config = paging.Config(True, 32_768, threshold=0, ledgered_threshold=0)
    lines = _run(s)
    final = client.sent[-1]
    assert final[0]["role"] == "system" and not final[0]["content"].startswith("[page")
    assert (
        final[1]["content"].startswith(prompts.MODES["fix"]) and "T" * 4_000 in final[1]["content"]
    )
    texts = [str(m.get("content")) for m in final]
    assert any(t.startswith(prompts.REPEATED.format(turn=1)) for t in texts)
    assert any(t == prompts.NOT_OFFERED.format(name="no_such_tool") for t in texts)
    assert any(t.startswith("Budget nearly used up") for t in texts)
    assert texts[-1].startswith(prompts.SUBMIT_NOW.format(why=prompts.STEP_LIMIT)) or any(
        t.startswith(prompts.STEP_LIMIT) for t in texts
    )
    # The latest KEEP_TURNS results are verbatim in the last request of the loop.
    tool_msgs = [m for m in final if m.get("role") == "tool"]
    assert all(not str(m["content"]).startswith("[page") for m in tool_msgs[-paging.KEEP_TURNS :])
    assert any(str(m["content"]).startswith("[page") for m in tool_msgs)
    # The submit-only turn's answer is never offered.
    paged = {p["page"] for e in lines if e.get("event") == "page_out" for p in e["pages"]}
    assert f"p{s.steps + 1}" not in s.pager.pages and f"p{s.steps + 1}" not in paged


def test_recall_is_offered_in_both_protocols_and_in_the_wire_schema(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    s, client = _session(tmp_path, fx, [("submit", {"answer": "a"})])
    _run(s)
    names = [t["function"]["name"] for t in client.fields[0]["tools"]]
    assert "recall" in names
    spec = s.toolbox.specs["recall"]
    assert spec.cost is None and spec.parameters["required"] == ["page"]
    wire = proto.one_action_schema(s.toolbox.specs, wire=True)
    recall = next(o for o in wire["anyOf"] if o["properties"]["tool"] == {"const": "recall"})
    assert recall["properties"]["arguments"]["additionalProperties"] is False
    assert "recall:" in proto.structured_tool_list(s.toolbox.specs)


def test_a_program_session_keeps_the_ledger_and_offers_recall(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    # A weight edit (2026-11-22): a bias-only program now draws the judging warning.
    code = "def edit(base, support):\n    return {'weight': base['weight'] + 0.25}\n"
    steps = [("test_program", {"code": code}), ("submit_program", {"code": code}),
             ("submit", {"answer": "a"})]  # fmt: skip
    s, _ = _session(tmp_path, fx, steps, mode="program", observation_tools=[])
    lines = _run(s)
    assert "recall" in s.toolbox.names()
    rows = [m["row"] for m in lines if m.get("event") == "ledger"]
    assert [r["kind"] for r in rows] == ["measurement", "program"]
    # Judged (2026-11-22): the fixture's 3 validation negatives at a 1% target cut no bar
    # (miWorkbench's calibrate: fire on nothing), so recall and far are there and bar is not.
    assert set(rows[0]["numbers"]) >= {"auroc", "recall", "far"} and rows[1]["change"] == "stored"
    assert rows[0]["action"].startswith("test_program code#")


# ----------------------------------------------------------------------------- end to end


class Counted(Recording):
    """Reports a usage the way a real tokenizer would, at ``ratio`` tokens per byte of what it
    was sent (messages and tool fields; 0.6 is the overall ratio measured on real WSP prompts,
    ``018_uncapped_tokens_2026-11-21.md``), so the context guard runs on counts, as it does
    live."""

    def __init__(
        self,
        replies: Sequence[Mapping[str, Any]],
        recorded: Sequence[list[str]] = (),
        ratio: float = 0.6,
    ):
        super().__init__(replies, recorded)
        self.ratio = ratio

    def complete(
        self,
        messages: Sequence[Mapping[str, Any]],
        *,
        fields: Mapping[str, Any],
        max_tokens: int | None,
    ) -> Reply:
        reply = super().complete(messages, fields=fields, max_tokens=max_tokens)
        sent = context.message_bytes(messages) + len(json.dumps(dict(fields)))
        reply.usage = proto.Usage(input_tokens=int(sent * self.ratio), output_tokens=10)
        return reply


def _long_session() -> list[tuple[str, dict[str, Any]]]:
    """An editor that reads as much of the weight as one read shows (a whole read of 4,096 values
    is a summary since 2026-11-22), looks at the top 50 rows, then edits the bias ten times,
    measuring three ways after each edit: 30 measurements (the D128 pattern)."""
    out: list[tuple[str, dict[str, Any]]] = [
        ("list_tensors", {}),
        ("describe_tensor", {"name": "weight"}),
        ("read_tensor", {"name": "weight", "index": "0:1600"}),
        ("forward_firing_positions", {"top_k": 50}),
    ]
    for i in range(10):
        out += [
            ("write_tensor", {"name": "bias", "values": -1.0 - 0.1 * i}),
            ("forward_practice_metrics", {"split": "train"}),
            ("forward_practice_metrics", {"split": "validation"}),
            ("forward_compare_to_parent", {"split": "validation"}),
        ]
    # Bias-only edits draw the judging warning once (2026-11-22): the editor submits again.
    return [*out, ("submit", {"answer": "done"}), ("submit", {"answer": "done"})]


#: The bounds of the end-to-end test, in the guard's own estimate on counts a tokenizer would give
#: (``Counted``, 0.6 tokens a byte; the gateway's rule: the last count plus new bytes at a token
#: each, times 1.1). Measured 2026-11-21: 25,613 without paging on an endless window, 16,504 with
#: it at 32,768 (where without paging the session hands off).
OFF_ABOVE = 20_000
ON_BELOW = 18_000


def test_a_long_session_stays_bounded_with_paging_and_grows_without(tmp_path: Path) -> None:
    """The D128 pattern at the Qwen editors' 32,768-token window. Without paging the prompt grows
    with every re-sent result: past ``OFF_ABOVE`` on an endless window, and at 32,768 it hands
    off (losing what the notes did not hold). With paging it stays under ``ON_BELOW`` and never
    hands off, and the session's prompts sum to under 60% of the unpaged ones."""
    big = make_bundle(tmp_path / "big", rule="mean", d_model=4096)
    runs: dict[str, tuple[session_mod.Session, Recording]] = {}
    for name, on, window in (
        ("off_endless", False, 1_000_000),
        ("off", False, 32_768),
        ("on", True, 32_768),
    ):
        s, _ = _session(tmp_path / name, big, _long_session(), paging_on=on, window=window)
        client = Counted(_replies(_long_session()))
        s._client = client
        _run(s)
        assert s.stop == "submitted" and s.forward.used == 41, name
        runs[name] = (s, client)
    endless, cut, paged = (runs[n][0].result()["engine"] for n in ("off_endless", "off", "on"))
    assert endless["handoffs"] == 0 and endless["peak_prompt_tokens"] > OFF_ABOVE, endless
    assert cut["handoffs"] >= 1, cut
    assert paged["handoffs"] == 0 and paged["peak_prompt_tokens"] < ON_BELOW, paged
    assert paged["pages_out"] >= 25 and paged["ledger_rows"] == 41
    summed = {
        n: sum(context.byte_estimate(sent) for sent in runs[n][1].sent)
        for n in ("off_endless", "on")
    }
    assert summed["on"] < 0.6 * summed["off_endless"], summed


# ----------------------------------------------------------------------------- the real child


def test_through_the_real_child_the_endpoint_receives_stubs_and_the_tail_and_a_replay_reproduces(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    steps = [{"calls": [READ]}]
    steps += [{"calls": [m]} for m in _measurements(5)]
    steps += [
        {"calls": [("write_tensor", {"name": "bias", "values": 0.25})]},
        {"calls": [("recall", {"page": "p1"})]},
        {"calls": [("submit", {"answer": "done"})]},
        # The bias-only edit draws the judging warning once (2026-11-22): submit again.
        {"calls": [("submit", {"answer": "done"})]},
    ]
    run = tmp_path / "run"
    with ScriptedEndpoint("openai_compatible", steps) as ep:
        line = line_for(
            run,
            fx,
            "repair",
            ep.gateway,
            python=False,
            forward_passes=20,
            observation_tools=TOOLS,
            max_steps=20,
            paging="on",
            editor={"context_window_tokens": 32_768},
        )
        seen = run_case(run, line)
    assert seen["exit"] == 0
    bodies = [r["body"] for r in ep.requests]
    assert "[page p1: read_tensor name=weight" in bodies[-1]["messages"][3]["content"]
    assert any(paging.LEDGER_HEADING in str(m["content"]) for m in bodies[-1]["messages"])
    assert "recall" in [t["function"]["name"] for t in bodies[0]["tools"]]
    out = result(run)
    assert out["stop_reason"] == "submitted" and out["engine"]["recalls"] == 1
    assert out["engine"]["pages_out"] >= 1 and out["engine"]["ledger_rows"] == 6
    lines = transcript(run)
    assert lines[0]["paging"] == paging.Config(True, 32_768).header()
    patch = json.loads((run / "out" / "patch" / "patch.json").read_text())
    # The real replay (the entry point, replay mode, the transcript pushed): the same patch, the
    # same page events, no divergence.
    again = tmp_path / "replay"
    rline = line_for(
        again,
        fx,
        "repair",
        None,
        mode="replay",
        replay_of_mode="session",
        python=False,
        forward_passes=20,
        observation_tools=TOOLS,
        max_steps=20,
        extra=[("transcript.jsonl", "transcript", run / "out" / "transcript.jsonl")],
    )
    done = run_main(again, rline, mode="replay")
    assert done["exit"] == 0, done["stderr"][-3000:]
    replayed = json.loads((again / "out" / "patch" / "patch.json").read_text())
    assert replayed["content_sha256"] == patch["content_sha256"]
    assert result(again)["replay_divergence"] is None
    pick = ("page_out", "ledger", "recall")
    assert [m for m in transcript(again) if m.get("event") in pick] == [
        m for m in lines if m.get("event") in pick
    ]


def test_after_a_page_event_the_guard_does_not_count_the_paged_text(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The endpoint's last count described the history BEFORE the stubs, so the request after a
    page event is estimated from the bytes (``guard.reset``), never from that stale count: with
    it, the request would be overestimated by the paged text and could hand off for nothing."""
    steps = [READ, *_measurements(12), ("submit", {"answer": "a"})]
    s, _ = _session(tmp_path, fx, steps)
    s._client = Counted(_replies(steps), ratio=0.5)
    bases: list[tuple[int, int | None]] = []
    ratios: list[tuple[int, list[float]]] = []

    class Spy(context.Guard):
        def prompt_estimate(
            self, messages: Sequence[Mapping[str, Any]], *, tools_bytes: int = 0
        ) -> int:
            bases.append((s.steps, self.last_prompt_tokens))
            ratios.append((s.steps, list(self.ratios)))
            return super().prompt_estimate(messages, tools_bytes=tools_bytes)

    s.guard = Spy(s.guard.window, s.guard.cap)
    lines = _run(s)
    paged_at = {m["turn"] for m in lines if m.get("event") == "page_out"}
    assert paged_at
    for turn, base in bases:
        if turn in paged_at:
            assert base is None, (turn, base)
    assert any(base is not None for turn, base in bases if turn not in paged_at and turn > 1)
    # The rewritten history is estimated at the ratio the endpoint's counts showed (0.5 here),
    # the gateway's rule, not at the first-call figure.
    for turn, seen in ratios:
        if turn in paged_at:
            assert seen and max(seen) == pytest.approx(0.5, abs=0.01), (turn, seen)


@pytest.mark.parametrize(
    ("window", "task"),
    [(32_768, None), (16_384, "T" * 9_000)],
    ids=["roomy", "under-pressure"],
)
def test_new_rows_refresh_the_ledger_on_the_schedule_with_nothing_to_page(
    tmp_path: Path, fx: FixtureBundle, window: int, task: str | None
) -> None:
    """Once something has been paged, rows added since (here edits, whose short answers never
    page) refresh the ledger in the history at the next batch, a page event with no pages; under
    pressure too, a refresh with nothing to page waits for the schedule (no rewrite a turn)."""
    steps = [*_measurements(1), *[("write_tensor", {"name": "bias", "values": v}) for v in
             (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8)], ("submit", {"answer": "a"})]  # fmt: skip
    s, client = _session(tmp_path, fx, steps, window=window, task=task)
    lines = _run(s)
    events = [m for m in lines if m.get("event") == "page_out"]
    if task is not None:  # the precondition: the history is past half the window
        assert context.byte_estimate(client.sent[-1]) > paging.PAGE_SHARE * window
    assert events[0]["pages"] and events[0]["turn"] == 2 + paging.KEEP_TURNS
    refresh = [e for e in events[1:] if not e["pages"]]
    assert refresh and refresh[0]["turn"] - events[0]["turn"] == paging.PAGE_EVERY_TURNS
    assert refresh[0]["ledger_rows"] > events[0]["ledger_rows"]
    sent = client.sent[refresh[0]["turn"] - 1]
    (where,) = _carrying(sent)
    assert where == len(sent) - 1 and "write_tensor name=bias values=0.4" in sent[where]["content"]


def test_pressure_brings_a_batch_forward(tmp_path: Path, fx: FixtureBundle) -> None:
    """The history's bytes, counted at a token each (the worst case), past half the window bring
    a page event forward of the schedule."""
    second = ("read_tensor", {"name": "weight", "precision": 5})
    steps = [READ, *_measurements(1), second, *_measurements(7)[1:], ("submit", {"answer": "a"})]
    s, _ = _session(tmp_path, fx, steps, window=20_000)
    lines = _run(s)
    reasons = [(m["turn"], m["reason"]) for m in lines if m.get("event") == "page_out"]
    assert (5, "every") in reasons and (7, "pressure") in reasons, reasons


@pytest.mark.parametrize(
    ("window", "on"),
    [(8_192, False), (paging.MIN_WINDOW_TOKENS - 1, False), (paging.MIN_WINDOW_TOKENS, True)],
    ids=["default-window", "just-under", "at-the-threshold"],
)
def test_paging_switches_itself_off_for_a_small_window(
    tmp_path: Path, fx: FixtureBundle, window: int, on: bool
) -> None:
    """The coordinator's decision (2026-11-21): under 16,384 tokens paging is off whatever the
    request asks, no recall tool and no paging rule are offered, and the transcript header and
    the result say why, so a replay reproduces it."""
    steps = [READ, *_measurements(6), ("submit", {"answer": "a"})]
    s, client = _session(tmp_path / "s", fx, steps, window=window)
    lines = _run(s)
    assert s.paging.on is on
    names = [t["function"]["name"] for t in client.fields[0]["tools"]]
    assert ("recall" in names) is on
    assert (prompts.PAGING in client.sent[0][0]["content"]) is on
    header, result = lines[0], s.result()
    if on:
        assert isinstance(header["paging"], dict) and header["paging_off_reason"] is None
        assert (result["paging"], result["paging_off_reason"]) == ("on", None)
        return
    assert (header["paging"], header["paging_off_reason"]) == ("off", paging.WINDOW_SMALL)
    assert (result["paging"], result["paging_off_reason"]) == ("off", "window_small")
    assert not [m for m in lines if m.get("event") in ("page_out", "ledger", "recall")]
    # A replay (whose own window could differ) reproduces it from the header.
    r = _replay_in_process(tmp_path / "r", fx, s.out / "transcript.jsonl")
    assert (r.paging.on, r.paging.off_reason) == (False, paging.WINDOW_SMALL)
    assert "recall" not in r.toolbox.names()
    _run(r)
    assert r.replay_divergence is None and r.result()["paging_off_reason"] == "window_small"


def test_a_request_that_asks_for_no_paging_gives_no_reason(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    s, _ = _session(tmp_path, fx, [("submit", {"answer": "a"})], paging_on=False, window=8_192)
    lines = _run(s)
    assert (lines[0]["paging"], lines[0]["paging_off_reason"]) == (None, None)
    assert (s.result()["paging"], s.result()["paging_off_reason"]) == ("off", None)
