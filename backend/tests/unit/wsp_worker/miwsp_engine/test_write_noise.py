"""Write noise (2026-11-26, FU-3's finding; session step 5b).

Run ``wsp_4f035a6d8a61`` (019 P4's repair pair, on the built-in protocol-test bundle): the editor
read ``weight`` at the default 4 significant digits, meant to scale ``weight[4]`` by 0.1, and
wrote all 8 entries back from the readings. The other 7 moved by their rounding error (FU-3's
record says each under 1e-4; on these weights it is up to 2.8e-4, under the 5e-4 a 4-digit
reading of a value near 1 rounds away). The write answered "8/8 changed", the candidate
recorded ``changed_entries`` 8, and the answer named only ``weight[4]``. No rule caught it.

Now a write's answer ends with ``prompts.WRITE_NOISE`` naming the entries it changed by no more
than a reading rounds away, and the transcript gets ``{event: write_noise}``. It is a flag: the
write is kept exactly as made, and the patch is the same. A replay follows the header.

In-process on a scripted client, as ``test_repeat_guard.py``; the bundle is the protocol test's
own generator and seed (``wsp.protocol_test``), so these are the run's weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.support.wsp_engine import line_for
from tests.support.wsp_fixtures import FixtureBundle, make_bundle
from wsp_worker.miwsp_engine import prompts
from wsp_worker.miwsp_engine import session as session_mod
from wsp_worker.miwsp_engine import tools as tools_mod
from wsp_worker.miwsp_engine.kernel import ExecResult
from wsp_worker.miwsp_engine.store import Accepted
from wsp_worker.miwsp_engine.transcript import ScriptedClient, TranscriptWriter, read_script

#: ``read_tensor weight`` as run ``wsp_4f035a6d8a61`` saw it (transcript entry 11), and what
#: it wrote (entry 39): the same 8 readings with index 4 scaled by 0.1.
RUN_READING = "-1.224 0.2973 -1.087 -0.4946 -0.2464 -0.6412 -0.447 0.6072"
RUN_WRITE = [-1.224, 0.2973, -1.087, -0.4946, -0.02464, -0.6412, -0.447, 0.6072]


@pytest.fixture()
def fx(tmp_path: Path) -> FixtureBundle:
    # wsp.protocol_test's built-in bundle: rule mean, d_model 8, 24 rows, seed 0.
    return make_bundle(tmp_path / "bundle", rule="mean", d_model=8, rows=24, seed=0)


def _replies(steps: list[tuple[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    return [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [{"id": f"c{i}", "name": n, "arguments": a}],
        }
        for i, (n, a) in enumerate(steps)
    ]


def _line(tmp: Path, fx: FixtureBundle, kind: str = "repair", **over: Any) -> dict[str, Any]:
    over.setdefault("python", False)
    return line_for(tmp, fx, kind, None, max_steps=20, **over)


def _run(s: session_mod.Session, header: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    s.writer = TranscriptWriter(
        s.out / "transcript.jsonl", s.header() if header is None else header
    )
    s._loop()
    s.writer.close()
    return [json.loads(x) for x in (s.out / "transcript.jsonl").read_text().splitlines()]


def _session(
    tmp: Path, fx: FixtureBundle, steps: list[tuple[str, dict[str, Any]]], **over: Any
) -> session_mod.Session:
    return session_mod.Session(
        _line(tmp / "run", fx, **over), client=ScriptedClient(_replies(steps))
    )


def _tool(lines: list[dict[str, Any]]) -> list[str]:
    return [str(m["content"]) for m in lines if m.get("role") == "tool"]


def _events(lines: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [m for m in lines if m.get("event") == "write_noise"]


SUBMIT = ("submit", {"answer": "Reduced weight[4] from -0.2464 to -0.02464."})


# ----------------------------------------------------------------------------- the run's shape


def test_the_runs_write_names_the_seven_rounded_entries(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [
        ("read_tensor", {"name": "weight"}),
        ("write_tensor", {"name": "weight", "values": RUN_WRITE}),
        SUBMIT,
    ]
    s = _session(tmp_path, fx, steps)
    lines = _run(s)
    read, wrote = _tool(lines)[:2]
    assert read.splitlines()[1] == RUN_READING  # the run's weights, at the run's precision
    # What it said before, unchanged: 8 of 9 changed, 8/8 in weight.
    assert wrote.startswith("wrote weight[:]: this action changed 8 of 9 weights\n")
    assert "  weight: 8/8 changed" in wrote
    # And now the flag, naming exactly the seven entries the editor did not mean to move.
    note = prompts.WRITE_NOISE.format(k=7, n=8, precision=4, where="weight at 0, 1, 2, 3, 5, 6, 7")
    assert wrote.endswith("\n" + note)
    assert _events(lines) == [
        {
            "event": "write_noise",
            "turn": 2,
            "tool": "write_tensor",
            "precision": 4,
            "changed": 8,
            "noise": 7,
            "tensors": [{"name": "weight", "entries": 7, "indices": [0, 1, 2, 3, 5, 6, 7]}],
        }
    ]
    # Each flagged entry moved by its rounding error only: at most half a unit in the 4th digit
    # (5e-5 for |w| < 1, 5e-4 for 1 <= |w| < 10). FU-3's "under 1e-4" holds for 4 of the 7.
    w0 = fx.tensors["weight"].values64().reshape(-1)
    moved = np.abs(s.store.values["weight"].astype(np.float64) - w0)
    half = 0.5 * 10.0 ** (np.floor(np.log10(np.abs(w0))) - 3)
    assert np.all(moved[[0, 1, 2, 3, 5, 6, 7]] <= half[[0, 1, 2, 3, 5, 6, 7]]) and moved[4] > 0.2
    assert 1e-4 < moved.max(where=np.arange(8) != 4, initial=0) < 5e-4


def test_the_flag_changes_nothing_the_run_computes_or_keeps(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    """The same script with the flag off: the same weights, the same patch bytes and counts, the
    same stop and answer. Only the write's answer and the event differ."""
    steps = [("write_tensor", {"name": "weight", "values": RUN_WRITE}), SUBMIT]
    on = _session(tmp_path / "on", fx, steps)
    off = _session(tmp_path / "off", fx, steps)
    off.write_noise = False
    on_lines, off_lines = _run(on), _run(off)
    assert on.store.sha256() == off.store.sha256()
    assert (on.stop, on.answer, on.steps) == (off.stop, off.answer, off.steps)
    on._write_outputs()
    off._write_outputs()
    patch_on = json.loads((on.out / "patch" / "patch.json").read_text())
    patch_off = json.loads((off.out / "patch" / "patch.json").read_text())
    assert patch_on["changed_entries_total"] == patch_off["changed_entries_total"] == 8
    # Everything but the moment each was written (created_at, and patch_sha256 over it): the
    # same content_sha256 (what a replay's "reproduced" compares), arrays and counts.
    stamped = {"created_at", "patch_sha256"}
    assert patch_on["content_sha256"] == patch_off["content_sha256"]
    assert {k: v for k, v in patch_on.items() if k not in stamped} == {
        k: v for k, v in patch_off.items() if k not in stamped
    }
    for name in sorted(p.name for p in (off.out / "patch").iterdir() if p.suffix != ".json"):
        assert (on.out / "patch" / name).read_bytes() == (off.out / "patch" / name).read_bytes()
    assert _events(on_lines) and not _events(off_lines)
    assert prompts.WRITE_NOISE.split("{")[0] not in "\n".join(_tool(off_lines))


# ----------------------------------------------------------------------------- not noise


def _deliberate(fx: FixtureBundle, how: str) -> list[float]:
    exact = fx.tensors["weight"].values64().reshape(-1)
    rounded = np.array([float(tools_mod.fmt_num(v)) for v in exact])
    if how == "scale":
        return (exact * 0.5).tolist()
    if how == "flip":
        return (-exact).tolist()
    if how == "shift_from_readings":
        return (rounded + 0.01).tolist()  # every entry moved on purpose, from rounded readings
    return [0.0] * exact.size


@pytest.mark.parametrize("how", ["scale", "flip", "shift_from_readings", "zero"])
def test_a_deliberate_full_tensor_write_is_not_noise(
    tmp_path: Path, fx: FixtureBundle, how: str
) -> None:
    steps = [("write_tensor", {"name": "weight", "values": _deliberate(fx, how)}), SUBMIT]
    lines = _run(_session(tmp_path, fx, steps))
    wrote = _tool(lines)[0]
    assert "this action changed 8 of 9 weights" in wrote
    assert "Note:" not in wrote and not _events(lines)


def test_a_write_of_the_named_entry_alone_is_not_noise(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [("write_tensor", {"name": "weight", "index": "4", "values": -0.02464}), SUBMIT]
    lines = _run(_session(tmp_path, fx, steps))
    assert "this action changed 1 of 9 weights" in _tool(lines)[0]
    assert "Note:" not in _tool(lines)[0] and not _events(lines)


def test_a_refused_write_says_nothing_of_noise(tmp_path: Path, fx: FixtureBundle) -> None:
    steps = [("write_tensor", {"name": "weight", "values": RUN_WRITE}), SUBMIT]
    lines = _run(_session(tmp_path, fx, steps, frozen_tensors=["weight"]))
    assert _tool(lines)[0].startswith("refused:") and not _events(lines)


# ----------------------------------------------------------------------------- python writes


class _Kernel:
    """The kernel's answer to one ``python`` run: its output and the deltas it made to ``W``."""

    def __init__(self, deltas: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
        self.deltas = deltas
        self.calls = 0
        self.synced: list[list[str]] = []

    def exec(self, code: str) -> ExecResult:
        self.calls += 1
        return ExecResult("ran", self.deltas)

    def sync(self, names: list[str]) -> None:
        self.synced.append(list(names))


def test_a_python_write_of_rounded_readings_is_flagged(tmp_path: Path, fx: FixtureBundle) -> None:
    """``W[name] = array`` is the other way to write, and the more usual one."""
    kernel = _Kernel({"weight": (np.arange(8), np.asarray(RUN_WRITE, dtype=np.float64))})
    s = _session(
        tmp_path, fx, [("python", {"code": "W['weight'][:] = [...]"}), SUBMIT], python=True
    )
    s.kernel = kernel  # type: ignore[assignment]
    lines = _run(s)
    out = _tool(lines)[0]
    assert kernel.calls == 1 and kernel.synced == [["weight"]]
    assert out == "ran\n" + prompts.WRITE_NOISE.format(
        k=7, n=8, precision=4, where="weight at 0, 1, 2, 3, 5, 6, 7"
    )
    (event,) = _events(lines)
    assert (event["tool"], event["changed"], event["noise"]) == ("python", 8, 7)


def test_a_python_write_of_one_entry_is_not_flagged(tmp_path: Path, fx: FixtureBundle) -> None:
    kernel = _Kernel({"weight": (np.array([4]), np.array([-0.02464]))})
    s = _session(tmp_path, fx, [("python", {"code": "W['weight'][4] *= 0.1"}), SUBMIT], python=True)
    s.kernel = kernel  # type: ignore[assignment]
    lines = _run(s)
    assert _tool(lines)[0] == "ran" and not _events(lines)


# ----------------------------------------------------------------------------- the measure


def _store(fx: FixtureBundle, tmp: Path) -> Any:
    return session_mod.Session(_line(tmp, fx), client=ScriptedClient([])).store


def test_the_bound_is_half_a_unit_in_the_last_digit_shown(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    store = _store(fx, tmp_path / "s")
    before = {"weight": store.values["weight"].copy()}
    w = store.values["weight"].astype(np.float64)
    half = 0.5 * 10.0 ** (np.floor(np.log10(np.abs(w))) - 3)  # 4 significant digits
    new = w.copy()
    new[0] += 0.9 * half[0]  # inside: noise
    new[1] -= 0.9 * half[1]  # inside, the other way
    new[2] += 1.6 * half[2]  # outside: an edit, if a small one
    new[3] += 3.0 * half[3]
    assert isinstance(store.try_apply({"weight": new}), Accepted)
    changed, noise = store.below_display(before, 4)
    assert changed == 4 and noise["weight"].tolist() == [0, 1]
    # More digits shown, a tighter bound: at 8 none of these is noise.
    assert store.below_display(before, 8) == (4, {})


def test_a_zero_or_an_unchanged_entry_is_never_noise(tmp_path: Path, fx: FixtureBundle) -> None:
    store = _store(fx, tmp_path / "s")
    store.values["weight"][0] = 0.0
    before = {"weight": store.values["weight"].copy()}
    new = store.values["weight"].copy()
    new[0] = 1e-30  # from an exact zero: shown as "0", so any change is meant
    assert isinstance(store.try_apply({"weight": new}), Accepted)
    assert store.below_display(before, 4) == (1, {})
    assert store.below_display({"weight": store.values["weight"].copy()}, 4) == (0, {})


def test_an_infinite_old_value_is_never_noise(tmp_path: Path, fx: FixtureBundle) -> None:
    """A bound taken from an infinite value is infinite: without the finite check any finite
    write over it would read as noise."""
    store = _store(fx, tmp_path / "s")
    store.values["weight"][2] = np.inf
    before = {"weight": store.values["weight"].copy()}
    new = store.values["weight"].copy()
    new[2] = 0.5
    assert isinstance(store.try_apply({"weight": new}), Accepted)
    assert store.below_display(before, 4) == (1, {})


def test_every_4_digit_round_trip_is_noise(tmp_path: Path, fx: FixtureBundle) -> None:
    """Writing float32 values back from their 4-digit readings is flagged for every entry it
    changes, at every scale. The written decimal's own float32 rounding can carry it just past
    half a unit (2.6465e-04 shows as 2.647e-04), which the bound's float32 term covers: these
    three were found by a search over 200,000 seeded values."""
    store = _store(fx, tmp_path / "s")
    rng = np.random.default_rng(0)
    edge = [2.6465e-04, -6.5915e-01, -6.2325e-01]
    for scale in (1e-3, 1e-1, 1.0, 10.0, 1e3):
        rest = (rng.standard_normal(5) * scale).tolist()
        store.values["weight"][:] = np.asarray(edge + rest, dtype=np.float32)
        before = {"weight": store.values["weight"].copy()}
        shown = [float(tools_mod.fmt_num(float(v))) for v in before["weight"]]
        assert isinstance(store.write("weight", shown), Accepted)
        changed, noise = store.below_display(before, tools_mod.DISPLAY_PRECISION)
        moved = np.flatnonzero(store.values["weight"] != before["weight"])
        assert changed == moved.size and changed >= 3
        assert noise["weight"].tolist() == moved.tolist(), scale


def test_a_float64_bias_written_back_from_its_reading_is_noise(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    store = _store(fx, tmp_path / "s")
    assert store.dtypes["bias"] == "float64"
    before = {"bias": store.values["bias"].copy()}
    shown = float(tools_mod.fmt_num(float(before["bias"][0])))
    assert shown != float(before["bias"][0])
    assert isinstance(store.write("bias", [shown]), Accepted)
    changed, noise = store.below_display(before, tools_mod.DISPLAY_PRECISION)
    assert changed == 1 and list(noise) == ["bias"] and noise["bias"].tolist() == [0]


def test_the_measure_uses_the_precision_a_reading_shows_by_default() -> None:
    import inspect

    assert inspect.signature(tools_mod.fmt_num).parameters["precision"].default == 4
    assert tools_mod.DISPLAY_PRECISION == 4
    read = inspect.signature(session_mod.Session._read_tensor).parameters["precision"].default
    assert read == tools_mod.DISPLAY_PRECISION


def test_a_matrix_names_positions_and_cuts_a_long_list() -> None:
    where = session_mod._noise_where("W", (2, 3), np.array([1, 5]))
    assert where == "W at (0, 1), (1, 2)"
    many = session_mod._noise_where("v", (40,), np.arange(20))
    assert many == "v at " + ", ".join(str(i) for i in range(16)) + " and 4 more"


# ----------------------------------------------------------------------------- header and replay


def test_the_header_records_it_in_an_editing_session_only(
    tmp_path: Path, fx: FixtureBundle
) -> None:
    edit = _session(tmp_path / "e", fx, [SUBMIT])
    assert edit.write_noise and edit.header()["write_noise"] == session_mod.WRITE_NOISE_VERSION
    read = session_mod.Session(_line(tmp_path / "i", fx, kind="inspect"), client=ScriptedClient([]))
    assert not read.write_noise and "write_noise" not in read.header()


@pytest.mark.parametrize("recorded_before", [True, False])
def test_a_replay_follows_the_header_it_was_recorded_with(
    tmp_path: Path, fx: FixtureBundle, recorded_before: bool
) -> None:
    """THE replay guarantee: a run recorded before this flag (no ``write_noise`` in its header)
    replays with no note and no divergence; one recorded with it replays the note bit for bit."""
    steps = [
        ("read_tensor", {"name": "weight"}),
        ("write_tensor", {"name": "weight", "values": RUN_WRITE}),
        SUBMIT,
    ]
    s = _session(tmp_path, fx, steps)
    if recorded_before:
        s.write_noise = False
    lines = _run(s)
    original = s.out / "transcript.jsonl"
    assert ("write_noise" in lines[0]) is not recorded_before
    again = _line(
        tmp_path / "replay",
        fx,
        mode="replay",
        replay_of_mode="session",
        extra=[("transcript.jsonl", "transcript", original)],
    )
    r = session_mod.Session(again, client=read_script(original))
    replayed = _run(r)
    assert r.write_noise is not recorded_before
    assert r.replay_divergence is None
    assert _tool(replayed) == _tool(lines)
    assert len(_events(replayed)) == len(_events(lines)) == (0 if recorded_before else 1)
    assert r.store.sha256() == s.store.sha256()
